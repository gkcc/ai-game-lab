"""Cumulative file-data budgets for this stage's dedicated Python processes.

Use ``with owned_writes(out, limit_bytes) as budget: ...`` around the fixed
writers.  Every accepted raw write is charged before entering the underlying
file write; failed/partial writes and overwritten bytes receive no refund.
TextIOWrapper performs encoding/newline conversion before this boundary.

This is a local Python IO adapter, not a filesystem sandbox or disk quota.
It covers open/Path writes, gzip, NumPy save/savez and shutil.copyfile used by
this stage.  Writable descriptors, FileIO, custom openers and sparse extension
fail closed.  Native code using its own system calls, aliases captured before
entry, pre-opened streams, mmap and subprocesses are outside this guarantee.
Standard output/error are pre-opened streams; the campaign bounds their pipes.
Only one context may be active per process.  Threads may write within it, but
must finish before exit.  Do not share its output tree with another writer.
"""
from __future__ import annotations

import _io
import builtins
import io
import os
from pathlib import Path
import shutil
import stat
import threading


class OutputBudgetExceeded(OSError):
    """A raw write would exceed the cumulative output allowance."""


class UnsafeWriteError(OSError):
    """A write cannot be bounded without changing or risking existing data."""


_CONTEXT_LOCK = threading.Lock()
_ACTIVE = None


def _absolute(value):
    return Path(os.path.abspath(os.fsdecode(os.fspath(value))))


def _no_links(path):
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise UnsafeWriteError(f'Links/junctions are not writable targets: {part}')


def _identity(info):
    return info.st_dev, info.st_ino


def _write_mode(mode):
    return isinstance(mode, str) and any(char in mode for char in 'wax+')


class _BudgetedRaw(io.RawIOBase):
    """No descriptor or unguarded .buffer/.raw escape is exposed."""

    def __init__(self, stream, budget, mode):
        self.__stream = stream
        self.__budget = budget
        self.mode = mode
        self.name = stream.name
        self.__append = 'a' in mode
        self.__readable = '+' in mode

    def writable(self):
        return True

    def readable(self):
        return self.__readable

    def seekable(self):
        return True

    def fileno(self):
        raise io.UnsupportedOperation('Budgeted files do not expose descriptors')

    def write(self, value):
        view = memoryview(value)
        if not view.c_contiguous:
            raise BufferError('write requires a contiguous buffer')
        with self.__budget._lock:
            self._checkClosed()
            if not self.__budget._active:
                raise UnsafeWriteError('The output budget context has ended')
            if self.__append:
                self.__stream.seek(0, os.SEEK_END)
            self.__budget._charge(view.nbytes)
            return self.__stream.write(view)

    def readinto(self, buffer):
        self._checkClosed()
        if not self.__readable:
            raise io.UnsupportedOperation('not readable')
        return self.__stream.readinto(buffer)

    def read(self, size=-1):
        self._checkClosed()
        if not self.__readable:
            raise io.UnsupportedOperation('not readable')
        return self.__stream.read(size)

    def tell(self):
        self._checkClosed()
        return self.__stream.tell()

    def seek(self, offset, whence=os.SEEK_SET):
        with self.__budget._lock:
            self._checkClosed()
            size = os.fstat(self.__stream.fileno()).st_size
            if whence == os.SEEK_SET:
                target = offset
            elif whence == os.SEEK_CUR:
                target = self.__stream.tell() + offset
            elif whence == os.SEEK_END:
                target = size + offset
            else:
                raise ValueError('Invalid seek origin')
            if target < 0 or target > size:
                raise UnsafeWriteError('Seeking beyond existing file data is unsupported')
            return self.__stream.seek(target)

    def truncate(self, size=None):
        with self.__budget._lock:
            self._checkClosed()
            size = self.tell() if size is None else size
            if size < 0 or size > os.fstat(self.__stream.fileno()).st_size:
                raise UnsafeWriteError('Extending a file with truncate is unsupported')
            return self.__stream.truncate(size)

    def flush(self):
        self._checkClosed()
        self.__stream.flush()

    def close(self):
        if not self.closed:
            try:
                super().close()
            finally:
                self.__stream.close()


class _WriteBudget:
    def __init__(self, out, limit_bytes, *, allowed_paths, allowed_prefixes):
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise ValueError('limit_bytes must be a nonnegative integer')
        self.root = _absolute(out)
        if self.root == Path(self.root.anchor):
            raise ValueError('out must be a dedicated output directory')
        self.limit_bytes = limit_bytes
        self._used_bytes = 0
        self._refused = 0
        self._created = {}
        self._streams = []
        self._patches = []
        self._lock = threading.RLock()
        self._active = False
        self._entered = False
        self._paths = (None if allowed_paths is None else
                       {self._allowed_path(path) for path in allowed_paths})
        self._prefixes = (None if allowed_prefixes is None else
                          tuple(self._allowed_path(path) for path in allowed_prefixes))

    def _allowed_path(self, path):
        path = Path(path)
        path = _absolute(path if path.is_absolute() else self.root / path)
        if not path.is_relative_to(self.root):
            raise ValueError('Allowlisted paths must be inside out')
        return path

    @property
    def used_bytes(self):
        with self._lock:
            return self._used_bytes

    @property
    def remaining_bytes(self):
        with self._lock:
            return self.limit_bytes - self._used_bytes

    @property
    def created_paths(self):
        with self._lock:
            return tuple(self._created)

    def snapshot(self):
        """A compact JSON-safe receipt, also available after failure/exit."""
        with self._lock:
            return dict(limit_bytes=self.limit_bytes, used_bytes=self._used_bytes,
                        remaining_bytes=self.limit_bytes - self._used_bytes,
                        refused=self._refused, created_count=len(self._created))

    def _charge(self, amount):
        if self._refused or amount > self.limit_bytes - self._used_bytes:
            self._refused += 1
            raise OutputBudgetExceeded(
                f'Output budget refused {amount} bytes; '
                f'{self._used_bytes}/{self.limit_bytes} bytes already charged')
        self._used_bytes += amount

    def _target(self, file):
        if isinstance(file, int):
            raise UnsafeWriteError('Writable file descriptors are unsupported; use a path')
        path = _absolute(file)
        if path == self.root or not path.is_relative_to(self.root):
            raise UnsafeWriteError(f'Writes must stay inside the owned output directory: {path}')
        if self._paths is not None or self._prefixes is not None:
            if (path not in (self._paths or ()) and not any(
                    path != prefix and path.is_relative_to(prefix)
                    for prefix in (self._prefixes or ()))):
                raise UnsafeWriteError(f'Output path is not allowlisted: {path}')
        if os.name == 'nt':
            for part in path.relative_to(self.root).parts:
                if ':' in part or part.rstrip(' .') != part or Path(part).is_reserved():
                    raise UnsafeWriteError('Windows device/alternate-stream/alias paths are unsupported')
        _no_links(path)
        return path

    def _open(self, file, mode='r', buffering=-1, encoding=None, errors=None,
              newline=None, closefd=True, opener=None):
        if not _write_mode(mode):
            return self._original_open(file, mode, buffering, encoding, errors,
                                       newline, closefd, opener)
        if (len(set(mode)) != len(mode) or any(c not in 'axrwbt+' for c in mode)
                or sum(c in mode for c in 'axrw') != 1 or ('b' in mode and 't' in mode)):
            raise ValueError(f'Invalid mode: {mode!r}')
        if not isinstance(buffering, int) or buffering < -1:
            raise ValueError('Invalid buffering value')
        binary = 'b' in mode
        if binary and any(value is not None for value in (encoding, errors, newline)):
            raise ValueError('Binary mode does not accept encoding/errors/newline')
        if not binary and buffering == 0:
            raise ValueError('Unbuffered text IO is unsupported')
        if not closefd or opener is not None:
            raise UnsafeWriteError('Custom openers and closefd=False are unsupported for writes')
        with self._lock:
            if not self._active:
                raise UnsafeWriteError('The output budget context has ended')
            path = self._target(file)
            try:
                info = path.lstat()
            except FileNotFoundError:
                if 'r' in mode:
                    raise
                stream = self._original_open(path, 'x+b' if '+' in mode else 'xb', buffering=0)
                self._created[path] = _identity(os.fstat(stream.fileno()))
            else:
                if (path not in self._created or self._created[path] != _identity(info)
                        or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
                    raise UnsafeWriteError(f'Existing input is immutable: {path}')
                if 'x' in mode:
                    raise FileExistsError(f'File already exists: {path}')
                stream = self._original_open(path, 'r+b', buffering=0)
                opened = os.fstat(stream.fileno())
                if _identity(opened) != self._created[path] or opened.st_nlink != 1:
                    stream.close()
                    raise UnsafeWriteError(f'Output identity changed: {path}')
                if 'w' in mode:
                    stream.truncate(0)
                elif 'a' in mode:
                    stream.seek(0, os.SEEK_END)
            raw = _BudgetedRaw(stream, self, mode)
            self._streams.append(raw)
            if binary:
                return raw
            try:
                text = io.TextIOWrapper(raw, encoding=encoding, errors=errors, newline=newline,
                                        line_buffering=buffering == 1, write_through=True)
                text.mode = mode
                return text
            except BaseException:
                raw.close()
                raise

    def _copyfile(self, src, dst, *, follow_symlinks=True):
        # Explicitly avoid shutil's platform-specific descriptor fast paths.
        try:
            same_file = os.path.samefile(src, dst)
        except FileNotFoundError:
            same_file = False
        if same_file:
            raise shutil.SameFileError(f'{src!r} and {dst!r} are the same file')
        if not follow_symlinks and Path(src).is_symlink():
            raise UnsafeWriteError('Copying a symbolic link as a link is unsupported')
        self._target(dst)
        with self._original_open(src, 'rb') as source, self._open(dst, 'wb') as target:
            while block := source.read(64 * 1024):
                target.write(block)
        return dst

    def _patch(self, obj, name, value):
        self._patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def __enter__(self):
        global _ACTIVE
        with _CONTEXT_LOCK:
            if self._entered or _ACTIVE is not None:
                raise RuntimeError('Output budget contexts cannot be nested or reused')
            _no_links(self.root)
            if self.root.exists() and not self.root.is_dir():
                raise ValueError('out must name a directory')
            self._original_open = builtins.open
            original_os_open = os.open
            original_fileio = io.FileIO

            def descriptor_open(path, flags, mode=0o777, *, dir_fd=None):
                writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
                if writing:
                    raise UnsafeWriteError('Writable os.open is unsupported; use open/Path.open')
                return original_os_open(path, flags, mode, dir_fd=dir_fd)

            class ReadOnlyFileIO(original_fileio):
                def __init__(self, file, mode='r', closefd=True, opener=None):
                    if _write_mode(mode):
                        raise UnsafeWriteError('Writable FileIO is unsupported; use open/Path.open')
                    super().__init__(file, mode, closefd, opener)

            self._active = True
            self._entered = True
            _ACTIVE = self
            try:
                for module, name, replacement in (
                    (builtins, 'open', self._open), (io, 'open', self._open),
                    (_io, 'open', self._open), (io, 'FileIO', ReadOnlyFileIO),
                    (_io, 'FileIO', ReadOnlyFileIO), (os, 'open', descriptor_open),
                    (shutil, 'copyfile', self._copyfile),
                ):
                    self._patch(module, name, replacement)
            except BaseException:
                self._restore()
                raise
        return self

    def _restore(self):
        global _ACTIVE
        for obj, name, original in reversed(self._patches):
            setattr(obj, name, original)
        self._patches.clear()
        self._active = False
        _ACTIVE = None

    def __exit__(self, exc_type, exc, traceback):
        close_error = None
        try:
            for stream in self._streams:
                try:
                    stream.close()
                except BaseException as error:
                    close_error = close_error or error
        finally:
            with _CONTEXT_LOCK:
                self._restore()
        if exc_type is None:
            if close_error is not None:
                raise close_error
            if self._refused:
                raise OutputBudgetExceeded('An output write exceeded the budget inside the context')
        return False


def owned_writes(out, limit_bytes, *, allowed_paths=None, allowed_prefixes=None):
    """Guard new outputs without creating ``out`` or touching existing files.

    Allowlist entries may be root-relative or absolute. ``allowed_prefixes``
    are directory prefixes, not filename/string prefixes; omitted allowlists
    allow all descendants. Reopening an output created by this same context
    supports append, ZIP header rewrites and shrinking, with no charge refund.
    Binary outputs are unbuffered; text outputs use write-through encoding.
    """
    return _WriteBudget(out, limit_bytes, allowed_paths=allowed_paths,
                        allowed_prefixes=allowed_prefixes)
