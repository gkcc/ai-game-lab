"""Bounded Linux owned-session supervision with pidfd-targeted cleanup.

The direct leader is never reaped while any cleanup signal may be sent. The
caller must isolate child ownership, fix the command/budgets and control its
SIGCHLD behavior. This does not authenticate a child, provide a sandbox, or
guarantee cleanup after session escape, uninterruptible sleep or host failure.
"""

from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


MAX_SECONDS = 86_400
STAT_LIMIT = 4096
GATE_LIMIT = 16_384
GATE_SHA256 = '49ebd8c9de7c59db5f36e6c94c1fdf4eab770d85312ccf39f744c57c282cd2ea'
POLL_SECONDS = .05
DEAD_STATES = frozenset((b'Z', b'X', b'x'))
_ERROR_CODES = frozenset(('unsupported', 'permission', 'scan', 'anchor', 'pidfd',
                          'signal', 'cleanup_timeout', 'work_callback', 'spawn', 'reap', 'reservation', 'gate'))


class SessionGuardError(RuntimeError):
    """A sanitized failure; cleanup must be treated as unconfirmed."""
    def __init__(self, code):
        if code not in _ERROR_CODES:
            code = 'scan'
        self.code = code
        super().__init__('Owned session failed: ' + code)


@dataclass(frozen=True, repr=False)
class Identity:
    pid: int
    session: int
    process_group: int
    starttime: int
    state: bytes


def _parse_stat(raw, expected_pid):
    if type(raw) is not bytes or len(raw) > STAT_LIMIT or type(expected_pid) is not int or expected_pid <= 0:
        raise SessionGuardError('scan')
    before, separator, after = raw.rpartition(b')')
    try:
        prefix, opening, _ = before.partition(b' (')
        fields = after.split()
        if not separator or not opening or len(fields) < 20:
            raise ValueError
        pid = int(prefix)
        pgrp, session, started = int(fields[2]), int(fields[3]), int(fields[19])
        state = fields[0]
        if (pid != expected_pid or pgrp < 0 or session < 0 or started < 0
                or len(state) != 1):
            raise ValueError
        return Identity(pid, session, pgrp, started, state)
    except (ValueError, TypeError, OverflowError):
        raise SessionGuardError('scan') from None


def _stat(pid):
    """Read one bounded /proc stat; transient disappearance is explicit."""
    try:
        with (Path('/proc') / str(pid) / 'stat').open('rb') as stream:
            raw = stream.read(STAT_LIMIT + 1)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError:
        raise SessionGuardError('permission') from None
    except OSError:
        raise SessionGuardError('scan') from None
    return _parse_stat(raw, pid)


def _members(session):
    """Return one sampled set; a missing /proc entry is not a signal target."""
    result = []
    try:
        with os.scandir('/proc') as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    continue
                identity = _stat(int(entry.name))
                if identity is not None and identity.session == session and identity.state not in DEAD_STATES:
                    result.append(identity)
    except SessionGuardError:
        raise
    except PermissionError:
        raise SessionGuardError('permission') from None
    except OSError:
        raise SessionGuardError('scan') from None
    return result


def _anchor(pid):
    """Check ownership without reaping; a lost leader invalidates the session."""
    try:
        status = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except (ChildProcessError, ProcessLookupError):
        raise SessionGuardError('anchor') from None
    except PermissionError:
        raise SessionGuardError('permission') from None
    except OSError:
        raise SessionGuardError('anchor') from None
    if status is not None and status.si_pid != pid:
        raise SessionGuardError('anchor')
    return status


def _anchor_fd(fd, pid):
    """Confirm the stable handle is still a waitable direct child."""
    try:
        status = os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except (ChildProcessError, ProcessLookupError):
        raise SessionGuardError('anchor') from None
    except PermissionError:
        raise SessionGuardError('permission') from None
    except OSError:
        raise SessionGuardError('anchor') from None
    if status is not None and status.si_pid != pid:
        raise SessionGuardError('anchor')
    return status


def _verified_pidfd(identity, session):
    """Pin a PID then recheck SID and starttime before it can be signalled."""
    try:
        fd = os.pidfd_open(identity.pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return None
        raise SessionGuardError('permission' if exc.errno in (errno.EPERM, errno.EACCES) else 'pidfd') from None
    try:
        after = _stat(identity.pid)
    except BaseException:
        os.close(fd)
        raise
    if (after is None or after.session != session or after.starttime != identity.starttime
            or after.state in DEAD_STATES):
        os.close(fd)
        return None
    return fd


def _send(fd, sig):
    try:
        signal.pidfd_send_signal(fd, sig)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return
        raise SessionGuardError('permission' if exc.errno in (errno.EPERM, errno.EACCES) else 'signal') from None


def _cleanup(pid, leader_fd, *, term_seconds, cleanup_seconds):
    """Signal each still-matching session member through its own pidfd."""
    started = time.monotonic()
    deadline = started + cleanup_seconds
    term_until = started + term_seconds
    empty_scans = 0
    while True:
        status = _anchor_fd(leader_fd, pid)
        _anchor(pid)
        members = _members(pid)
        if not members and status is not None:
            empty_scans += 1
            if empty_scans >= 2:
                return
        else:
            empty_scans = 0
            sig = signal.SIGTERM if time.monotonic() < term_until else signal.SIGKILL
            for identity in members:
                _anchor_fd(leader_fd, pid)
                _anchor(pid)
                fd = _verified_pidfd(identity, pid)
                if fd is None:
                    continue
                try:
                    _anchor_fd(leader_fd, pid)
                    _anchor(pid)
                    _send(fd, sig)
                finally:
                    os.close(fd)
        now = time.monotonic()
        if now >= deadline:
            raise SessionGuardError('cleanup_timeout')
        time.sleep(min(POLL_SECONDS, max(0., deadline - now)))


def _capability():
    """Fail before spawning if this process cannot use the required APIs."""
    if (os.name != 'posix' or not all(hasattr(os, name) for name in
                                     ('WNOWAIT', 'WEXITED', 'WNOHANG', 'P_PID', 'P_PIDFD', 'waitid', 'pidfd_open'))
            or not hasattr(signal, 'pidfd_send_signal') or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL):
        raise SessionGuardError('unsupported')
    if _stat(os.getpid()) is None:
        raise SessionGuardError('unsupported')
    try:
        fd = os.pidfd_open(os.getpid(), 0)
    except OSError as exc:
        raise SessionGuardError('permission' if exc.errno in (errno.EPERM, errno.EACCES) else 'unsupported') from None
    try:
        _send(fd, 0)
        try:
            os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            # The current process is not its own child; ECHILD demonstrates
            # that this kernel recognized the pidfd wait mode.
            pass
        except PermissionError:
            raise SessionGuardError('permission') from None
        except OSError:
            raise SessionGuardError('unsupported') from None
        else:
            raise SessionGuardError('unsupported')
    finally:
        os.close(fd)


def _seconds(value):
    return type(value) in (int, float) and 0 < value <= MAX_SECONDS


def _reserve_slot():
    """Hold a close-on-exec descriptor before a child can be created."""
    if not hasattr(os, 'O_CLOEXEC'):
        raise SessionGuardError('unsupported')
    try:
        return os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        raise SessionGuardError('reservation') from None


def _release_slot(fd):
    try:
        os.close(fd)
    except OSError:
        raise SessionGuardError('reservation') from None


def _load_gate(expected_sha256):
    """Read one bounded, hash-pinned source snapshot before any child exists."""
    if type(expected_sha256) is not str or expected_sha256 != GATE_SHA256:
        raise SessionGuardError('gate')
    path = Path(__file__).resolve().with_name('gate.py')
    try:
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
                or getattr(before, 'st_file_attributes', 0) & 0x400 or before.st_size > GATE_LIMIT):
            raise SessionGuardError('gate')
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(path, flags)
        try:
            current = os.fstat(fd)
            if not stat.S_ISREG(current.st_mode) or current.st_size > GATE_LIMIT:
                raise SessionGuardError('gate')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(GATE_LIMIT + 1)
        finally:
            os.close(fd)
        if (len(raw) != current.st_size or len(raw) > GATE_LIMIT
                or hashlib.sha256(raw).hexdigest() != expected_sha256):
            raise SessionGuardError('gate')
        return raw.decode('utf-8')
    except SessionGuardError:
        raise
    except (OSError, UnicodeError):
        raise SessionGuardError('gate') from None


def supervise(command, work_seconds, cancelled, *, term_seconds, cleanup_seconds,
              gate_seconds, expected_gate_sha256):
    """Own one child session; no numeric PID/PGID is signalled after reaping.

    Work and cleanup times are sampled. The caller must install cancellation
    signal handlers, keep this process the sole child reaper, and independently
    enforce any overall deadline and source/device/receipt requirements. The
    registered gate bytes are executed from one checked snapshot with -I -S -B;
    no target command executes before the leader pidfd and session are bound.
    Return success only after two empty session scans and direct-child wait.
    On any error a sanitized ``SessionGuardError.code`` is raised; cleanup is
    not proven, even when direct-child fallback succeeds.
    """
    if (type(command) is not list or not command or any(type(x) is not str or not x for x in command)
            or not os.path.isabs(command[0]) or any('\x00' in x for x in command)
            or not callable(cancelled) or not _seconds(work_seconds) or not _seconds(gate_seconds)
            or not _seconds(term_seconds) or not _seconds(cleanup_seconds)
            or term_seconds >= cleanup_seconds or gate_seconds > work_seconds):
        raise ValueError('Invalid owned-session request')
    gate_source = _load_gate(expected_gate_sha256)
    _capability()
    try:
        before_spawn = cancelled()
        if type(before_spawn) is not bool:
            raise ValueError
    except BaseException:
        raise SessionGuardError('work_callback') from None
    if before_spawn:
        raise SessionGuardError('gate')
    started = time.monotonic()
    reserve_fd = _reserve_slot()
    gate_r = gate_w = None
    try:
        gate_r, gate_w = os.pipe()
        if os.get_inheritable(gate_w):
            raise SessionGuardError('gate')
        gate_command = [sys.executable, '-I', '-S', '-B', '-c', gate_source,
                        str(gate_r), str(gate_seconds), *command]
        process = subprocess.Popen(gate_command, start_new_session=True, pass_fds=(gate_r,))
    except BaseException as exc:
        for fd in (gate_r, gate_w):
            if fd is not None:
                os.close(fd)
        _release_slot(reserve_fd)
        if isinstance(exc, SessionGuardError):
            raise exc from None
        raise SessionGuardError('spawn') from None
    leader_fd = None
    leader_anchored = False
    released = False
    reaped = False
    try:
        read_end, gate_r = gate_r, None
        os.close(read_end)
        reserved, reserve_fd = reserve_fd, None
        _release_slot(reserved)
        _anchor(process.pid)
        try:
            leader_fd = os.pidfd_open(process.pid, 0)
        except OSError as exc:
            raise SessionGuardError('permission' if exc.errno in (errno.EPERM, errno.EACCES) else 'pidfd') from None
        # P_PIDFD verifies that this stable handle is still our unreaped child.
        _anchor_fd(leader_fd, process.pid)
        _anchor(process.pid)
        leader_anchored = True
        initial = _stat(process.pid)
        if initial is None or initial.session != process.pid or initial.process_group != process.pid:
            raise SessionGuardError('anchor')
        again = _stat(process.pid)
        if again is None or again.session != process.pid or again.starttime != initial.starttime:
            raise SessionGuardError('anchor')
        _anchor_fd(leader_fd, process.pid)
        # The gate has not forked or executed the command. Sample cancellation
        # before releasing it; the caller still owns the outer signal policy.
        try:
            should_cancel = cancelled()
            if type(should_cancel) is not bool:
                raise ValueError
        except BaseException:
            raise SessionGuardError('work_callback') from None
        if should_cancel or time.monotonic() - started >= work_seconds:
            raise SessionGuardError('gate')
        # Once a write is attempted, treat the target as possibly released even
        # if a signal interrupts Python before the return value is observed.
        released = True
        try:
            written = os.write(gate_w, b'G')
        except OSError:
            raise SessionGuardError('gate') from None
        if written != 1:
            raise SessionGuardError('gate')
        write_end, gate_w = gate_w, None
        os.close(write_end)
        reason = None
        callback_failed = False
        while True:
            if _anchor(process.pid) is not None:
                # We cannot determine the kernel's exact exit time. If the
                # first exit observation is at/after the deadline, reject it
                # conservatively even if actual exit might have been earlier.
                reason = ('deadline' if time.monotonic() - started >= work_seconds
                          else 'exited')
                break
            try:
                should_cancel = cancelled()
                if type(should_cancel) is not bool:
                    raise ValueError
            except BaseException:
                callback_failed = True
                break
            if should_cancel:
                reason = 'cancelled'
                break
            if time.monotonic() - started >= work_seconds:
                reason = 'deadline'
                break
            time.sleep(POLL_SECONDS)
        _cleanup(process.pid, leader_fd, term_seconds=term_seconds, cleanup_seconds=cleanup_seconds)
        if callback_failed:
            raise SessionGuardError('work_callback')
        try:
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError):
            raise SessionGuardError('reap') from None
        reaped = True
        return {'reason': reason, 'child_exit_code': process.returncode,
                'elapsed_seconds': time.monotonic() - started,
                'owned_session_empty_before_reap': True, 'direct_child_reaped': True,
                'used_pidfd': True, 'completed': reason == 'exited' and process.returncode == 0}
    except BaseException as exc:
        # Before release, EOF asks the verified no-fork gate to exit. Never
        # replace an unconfirmed pidfd with a numeric PID or PGID signal.
        if gate_w is not None:
            write_end, gate_w = gate_w, None
            try:
                os.close(write_end)
            except OSError:
                pass
        if not released and not reaped:
            try:
                process.wait(timeout=cleanup_seconds)
                reaped = True
            except (subprocess.TimeoutExpired, OSError):
                pass
        if leader_anchored and leader_fd is not None and not reaped:
            try:
                _anchor_fd(leader_fd, process.pid)
                _anchor(process.pid)
                _send(leader_fd, signal.SIGKILL)
            except SessionGuardError:
                pass
        if not reaped:
            try:
                process.wait(timeout=1)
                reaped = True
            except (subprocess.TimeoutExpired, OSError):
                pass
        if isinstance(exc, SessionGuardError):
            raise exc from None
        raise SessionGuardError('scan') from None
    finally:
        for fd in (gate_r, gate_w):
            if fd is not None:
                os.close(fd)
        if reserve_fd is not None:
            _release_slot(reserve_fd)
        if leader_fd is not None:
            os.close(leader_fd)
