"""One bounded pass over a previously authenticated teacher training NPZ.

Only shared/candidates/offsets/teacher_choice_label payloads are decoded. The
other two members' headers are parsed but their numeric values are not decoded;
whole-file hashes and ZIP buffering can still read their payload bytes. All callbacks are
private and provisional until this function returns after its second hash.
The caller authenticates the historical registration and schema separately.
No complete feature matrix, model, engine, game or holdout is opened here.
"""
from contextlib import contextmanager
import hashlib
import hmac
import io
import math
import os
from pathlib import Path
import re
import stat
import struct
import threading
import zipfile

import numpy as np


MAX_ARCHIVE_BYTES = 256 * 1024**2
MAX_DECODED_BYTES = 256 * 1024**2
MAX_GROUPS = 8000
MAX_CANDIDATES = 400000
MAX_WIDTH = 256
HEADER_LIMIT = 4096
CHUNK = 1024**2
_ACTIVE = threading.Lock()
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_NAMES = frozenset(('shared', 'candidates', 'offsets', 'teacher_choice_label', 'ops', 'seed'))
_DECODED = ('shared', 'candidates', 'offsets', 'teacher_choice_label')


class ArchiveError(ValueError):
    """Archive identity or structure could not be established."""


class _CallbackRaised(BaseException):
    """Keep caller failures separate from malformed archive exceptions."""
    def __init__(self, original):
        self.original = original


def _require(value):
    if not value:
        raise ArchiveError('Invalid authenticated training archive')


@contextmanager
def _open_regular(path):
    """Open one regular file; reject symlinks, junctions and path swaps."""
    fd = None
    try:
        p = Path(path)
        before = p.lstat()
        _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
                 and not getattr(before, 'st_file_attributes', 0) & 0x400)
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(p, flags)
        current = os.fstat(fd)
        _require(stat.S_ISREG(current.st_mode) and 0 < current.st_size <= MAX_ARCHIVE_BYTES
                 and (before.st_dev, before.st_ino, before.st_size)
                 == (current.st_dev, current.st_ino, current.st_size))
        source = os.fdopen(fd, 'rb')
        fd = None
    except BaseException as exc:
        if fd is not None:
            os.close(fd)
        if isinstance(exc, OSError):
            raise ArchiveError('Training archive could not be opened') from None
        raise
    try:
        with source:
            yield source, current.st_size
    finally:
        if fd is not None:
            os.close(fd)


def _hash_stream(source, size, guard):
    guard()
    source.seek(0)
    digest, total = hashlib.sha256(), 0
    while True:
        guard()
        block = source.read(CHUNK)
        if not block:
            break
        total += len(block)
        _require(total <= size <= MAX_ARCHIVE_BYTES)
        digest.update(block)
    _require(total == size)
    guard()
    return digest.hexdigest()


def _shapes(groups, candidates):
    return dict(shared=((groups, 275), np.dtype('<f4')),
        candidates=((candidates, 98), np.dtype('<f4')),
        offsets=((groups + 1,), np.dtype('<i8')),
        teacher_choice_label=((groups,), np.dtype('<i8')),
        ops=((candidates,), np.dtype('u1')), seed=((groups,), np.dtype('<u4')))


def _directory_preflight(source, size, guard):
    """Bound ZIP central-directory allocation before ZipFile parses it."""
    guard()
    source.seek(max(0, size - (65535 + 22)))
    tail = source.read(65535 + 22)
    marker = tail.rfind(b'PK\x05\x06')
    _require(marker >= 0 and len(tail) - marker >= 22)
    signature, disk, directory_disk, here, total, directory_size, directory_offset, comment = (
        struct.unpack('<4s4H2IH', tail[marker:marker + 22]))
    end_offset = size - len(tail) + marker
    _require(signature == b'PK\x05\x06' and disk == directory_disk == 0
             and here == total == len(_NAMES) and directory_size <= 32768
             and directory_offset + directory_size == end_offset
             and marker + 22 + comment == len(tail))
    guard()


def _header(member, shape, dtype, member_size):
    try:
        version = np.lib.format.read_magic(member)
        _require(version in ((1, 0), (2, 0)))
        width = 2 if version == (1, 0) else 4
        length_bytes = member.read(width)
        _require(len(length_bytes) == width)
        length = int.from_bytes(length_bytes, 'little')
        _require(0 < length and member.tell() + length <= HEADER_LIMIT
                 and member.tell() + length <= member_size)
        header = member.read(length)
        _require(len(header) == length)
        # NumPy checks max_header_size only after its own read. Give it a
        # snapshot whose declared length was bounded before allocation.
        bounded = io.BytesIO(np.lib.format.magic(*version) + length_bytes + header)
        _require(np.lib.format.read_magic(bounded) == version)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        actual_shape, fortran, actual_dtype = reader(bounded, max_header_size=HEADER_LIMIT)
    except (ValueError, EOFError, TypeError, OverflowError, UnicodeError, SyntaxError):
        raise ArchiveError('Invalid training archive header') from None
    _require(type(actual_shape) is tuple and actual_shape == shape
             and all(type(x) is int and x >= 0 for x in actual_shape)
             and fortran is False and actual_dtype == dtype and not actual_dtype.hasobject
             and actual_dtype.fields is None and actual_dtype.subdtype is None
             and member.tell() <= HEADER_LIMIT
             and member.tell() + math.prod(shape) * dtype.itemsize == member_size)


def _read_payload(archive, info, shape, dtype, guard):
    """Decode one prechecked member in guarded chunks; never decode seed/ops."""
    _require(info.filename[:-4] in _DECODED)
    guard()
    data = np.empty(shape, dtype=dtype)
    raw = data.reshape(-1).view(np.uint8)
    with archive.open(info) as member:
        _header(member, shape, dtype, info.file_size)
        position = 0
        while position < raw.size:
            guard()
            chunk = member.read(min(CHUNK, raw.size - position))
            _require(bool(chunk) and position + len(chunk) <= raw.size)
            raw[position:position + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
            position += len(chunk)
        _require(not member.read(1))
    data.setflags(write=False)
    return data


def _validate_arrays(arrays, groups, candidates, guard):
    offsets, labels = arrays['offsets'], arrays['teacher_choice_label']
    _require(offsets.shape == (groups + 1,) and labels.shape == (groups,)
             and offsets[0] == 0 and offsets[-1] == candidates
             and np.all(offsets >= 0) and np.all(offsets <= candidates))
    widths = np.diff(offsets)
    _require(np.all(widths >= 1) and np.all(widths <= MAX_WIDTH)
             and np.all(labels >= 0) and np.all(labels < widths))
    for name in ('shared', 'candidates'):
        array = arrays[name]
        for start in range(0, len(array), 4096):
            guard()
            _require(bool(np.isfinite(array[start:start + 4096]).all()))


def _scan_stream(source, *, size, expected_sha256, expected_groups,
                 expected_candidates, guard, consume):
    first = _hash_stream(source, size, guard)
    _require(hmac.compare_digest(first, expected_sha256))
    _directory_preflight(source, size, guard)
    source.seek(0)
    shapes = _shapes(expected_groups, expected_candidates)
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        _require(len(infos) == len(_NAMES)
                 and {x.filename for x in infos} == {name + '.npy' for name in _NAMES}
                 and sum(x.file_size for x in infos) <= MAX_DECODED_BYTES
                 and all(not x.flag_bits & 1 and x.compress_type == zipfile.ZIP_STORED
                         and x.file_size == x.compress_size and x.file_size > 0 for x in infos))
        by_name = {x.filename[:-4]: x for x in infos}
        # Check every header before allocating even one array. seed and ops are
        # inspected only through their bounded headers and never decoded.
        for name in sorted(_NAMES):
            guard()
            info = by_name[name]
            with archive.open(info) as member:
                _header(member, *shapes[name], info.file_size)
        arrays = {name: _read_payload(archive, by_name[name], *shapes[name], guard)
                  for name in _DECODED}
    _validate_arrays(arrays, expected_groups, expected_candidates, guard)
    offsets, labels = arrays['offsets'], arrays['teacher_choice_label']
    for index in range(expected_groups):
        guard()
        start, end = int(offsets[index]), int(offsets[index + 1])
        # Private per-group immutable buffers isolate the four backing arrays
        # from callback mutation and avoid any full (rows,373) expansion.
        shared = np.frombuffer(arrays['shared'][index].tobytes(), dtype='<f4')
        candidates = np.frombuffer(arrays['candidates'][start:end].tobytes(), dtype='<f4').reshape(end-start, 98)
        consume(index, shared, candidates, int(labels[index]))
    second = _hash_stream(source, size, guard)
    _require(hmac.compare_digest(first, second) and hmac.compare_digest(second, expected_sha256))
    return dict(groups=expected_groups, candidates=expected_candidates, verified_after_scan=True)


def scan_archive(path, *, expected_sha256, expected_groups, expected_candidates, guard, consume):
    """Visit each complete group once, then rehash the same open descriptor.

    ``consume(index, shared, candidates, target_index)`` receives immutable,
    private float32 arrays for one group only. Caller-managed outputs remain
    provisional unless this function returns normally. It must avoid retaining
    all groups or starting a second process. The in-process gate prohibits
    simultaneous/reentrant scans; it is not a cross-process lock. Two hashes
    catch ordinary mutations but are not an immutable snapshot against a
    malicious actor mutating and restoring bytes during this call.
    """
    _require(type(expected_sha256) is str and _HEX64.fullmatch(expected_sha256) is not None
             and type(expected_groups) is int and 1 <= expected_groups <= MAX_GROUPS
             and type(expected_candidates) is int and expected_groups <= expected_candidates <= MAX_CANDIDATES
             and expected_candidates <= expected_groups * MAX_WIDTH
             and callable(guard) and callable(consume))
    if not _ACTIVE.acquire(blocking=False):
        raise ArchiveError('Another training archive scan is active')
    def caller(callback):
        def guarded(*args):
            try:
                return callback(*args)
            except BaseException as exc:
                raise _CallbackRaised(exc) from None
        return guarded
    try:
        with _open_regular(path) as (source, size):
            return _scan_stream(source, size=size, expected_sha256=expected_sha256,
                expected_groups=expected_groups, expected_candidates=expected_candidates,
                guard=caller(guard), consume=caller(consume))
    except _CallbackRaised as wrapped:
        raise wrapped.original.with_traceback(wrapped.original.__traceback__) from None
    except (zipfile.BadZipFile, EOFError, UnicodeError, OverflowError) as exc:
        raise ArchiveError('Invalid authenticated training archive') from None
    finally:
        _ACTIVE.release()
