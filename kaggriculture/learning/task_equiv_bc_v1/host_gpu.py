"""Windows transport supervision for one already-authenticated Linux GPU run.

The caller authenticates the stage, command, registration and downstream
training result. This module never kills WSL or any Linux process. It signals
only via the fresh stage-local cancellation marker and relies on the existing
Linux pidfd guard's receipt. Sampling is not a hard real-time memory limit.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


INITIAL_AVAILABLE = 5 * 1024**3
RUN_AVAILABLE = 4 * 1024**3
WALL_SECONDS = 900.0
CLEANUP_SECONDS = 30.0
POLL_SECONDS = 1.0
MAX_RECEIPT = 65536
STAGE_NAME = 'task-equivalent-bc-v1-20261031'
STDOUT = 'gpu-host.stdout.local.log'
STDERR = 'gpu-host.stderr.local.log'
CANCEL = 'gpu-cancel.local.json'
RECEIPT = 'gpu-supervision.local.json'


@dataclass(repr=False)
class HostResult:
    status: str
    reason: str
    exit_code: object
    cleanup_confirmed: bool
    child: object = None
    stdout_stream: object = None
    stderr_stream: object = None

    def __repr__(self):
        return '<PrivateHostGpuResult>'


def _stage(value):
    path = Path(value)
    if (not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir()
            or path.name != STAGE_NAME or path.parent.name != 'artifacts'):
        raise ValueError('Invalid authenticated stage')
    return path


def _command(argv, registration_commit):
    if (type(registration_commit) is not str or re.fullmatch(r'[0-9a-f]{40}', registration_commit) is None
            or type(argv) is not list or not argv
            or any(type(item) is not str or not item or '\x00' in item for item in argv)
            or Path(argv[0]).name.lower() != 'wsl.exe'
            or argv.count('-I') != 1 or argv.count('-B') != 1
            or argv.count('XLA_PYTHON_CLIENT_PREALLOCATE=false') != 1
            or argv.count('--registration-commit') != 1 or argv.count('--stage') != 1):
        raise ValueError('Invalid authenticated command')
    index = argv.index('--registration-commit')
    stage_index = argv.index('--stage')
    if (index + 1 >= len(argv) or argv[index + 1] != registration_commit
            or stage_index + 1 >= len(argv)
            or not any(item.endswith('/gpu_supervisor.py') for item in argv)):
        raise ValueError('Invalid authenticated command')
    return list(argv)


def _available(callback):
    value = callback()
    if type(value) is not int or value < 0:
        raise ValueError('Memory sample failed')
    return value


def _fresh(stage):
    if any(os.path.lexists(stage / name) for name in (STDOUT, STDERR, CANCEL, RECEIPT)):
        raise FileExistsError('GPU stage has already been attempted')


def _logs(stage):
    out = (stage / STDOUT).open('xb', buffering=0)
    try:
        err = (stage / STDERR).open('xb', buffering=0)
    except BaseException:
        out.close()
        raise
    return out, err


def _cancel(stage, registration_commit, reason):
    if os.path.lexists(stage / CANCEL):
        return
    body = json.dumps({'schema': 'task-equivalent-bc-gpu-cancel-v1',
                       'registration_commit': registration_commit, 'reason': reason},
                      separators=(',', ':'), allow_nan=False).encode() + b'\n'
    try:
        with (stage / CANCEL).open('xb') as stream:
            if stream.write(body) != len(body):
                raise OSError('Cancellation write incomplete')
    except FileExistsError:
        # A concurrent owner has already requested cancellation; never replace it.
        return


def _report(stage):
    path = stage / RECEIPT
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('Invalid Linux supervision receipt')
    with path.open('rb') as stream:
        raw = stream.read(MAX_RECEIPT + 1)
    if not 0 < len(raw) <= MAX_RECEIPT:
        raise ValueError('Invalid Linux supervision receipt')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate receipt field')
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError('Invalid receipt value')
    report = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    if (type(report) is not dict or type(report.get('completed')) is not bool
            or type(report.get('child_exit_code')) is not int
            or type(report.get('reason')) is not str):
        raise ValueError('Invalid Linux supervision receipt')
    return report


def _clean(report):
    return (report.get('used_pidfd') is True and report.get('direct_child_reaped') is True
            and report.get('owned_session_empty_before_reap') is True)


def _close_streams(out, err):
    out.close()
    err.close()


def supervise(stage, argv, registration_commit, *, popen=None, available=None, clock=time.monotonic):
    """Run once and return an explicit result, retaining handles if unconfirmed.

    The caller supplies the authenticated stage and argv; this entry validates
    their basic contract, does not invoke a shell and never retries. A return of
    ``cleanup_unconfirmed`` transfers the Popen object and both still-open log
    streams to the caller, which must keep them and continue waiting/observing.
    ``passed`` requires a zero transport exit plus the Linux pidfd/reap/empty
    receipt. A cancellation always fails even when Linux cleanup succeeds.
    """
    stage = _stage(stage)
    argv = _command(argv, registration_commit)
    if available is None:
        import psutil
        available = lambda: psutil.virtual_memory().available
    if popen is None:
        popen = subprocess.Popen
    if not callable(popen) or not callable(available) or not callable(clock):
        raise ValueError('Invalid host callbacks')
    started = clock()
    if not isinstance(started, (int, float)) or not math.isfinite(started):
        raise ValueError('Invalid host clock')
    if _available(available) < INITIAL_AVAILABLE:
        raise MemoryError('Initial memory gate')
    _fresh(stage)
    out, err = _logs(stage)
    try:
        # Resample immediately before starting; opening the logs never grants
        # permission to start below the initial floor.
        if _available(available) < INITIAL_AVAILABLE:
            _close_streams(out, err)
            return HostResult('failed', 'initial_memory_gate', None, True)
    except BaseException:
        _close_streams(out, err)
        return HostResult('failed', 'initial_memory_sample_error', None, True)
    try:
        child = popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000),
                      close_fds=True)
    except BaseException:
        # A constructor interruption may occur after process creation but
        # before the handle is returned. Never claim that Linux was cleaned.
        try:
            _cancel(stage, registration_commit, 'spawn_uncertain')
        except BaseException:
            pass
        return HostResult('cleanup_unconfirmed', 'spawn_uncertain', None, False, None, out, err)

    reason, cancel_started, exit_code = None, None, None
    try:
        while True:
            now = clock()
            if not isinstance(now, (int, float)) or not math.isfinite(now) or now < started:
                raise ValueError('Invalid host clock')
            # A wait can return after the deadline. Without an earlier exit
            # observation we cannot claim it finished inside the budget.
            if now - started >= WALL_SECONDS:
                reason = 'time_gate'
                break
            exit_code = child.poll()
            if exit_code is not None:
                if type(exit_code) is not int:
                    raise ValueError('Invalid transport exit')
                if _available(available) < RUN_AVAILABLE:
                    reason = 'memory_gate'
                break
            if os.path.lexists(stage / CANCEL):
                reason = 'canceled'
                break
            if _available(available) < RUN_AVAILABLE:
                reason = 'memory_gate'
                break
            try:
                child.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                pass
    except BaseException:
        reason = 'host_error'

    if reason is not None:
        try:
            _cancel(stage, registration_commit, reason)
        except BaseException:
            # Lack of a marker cannot establish Linux cleanup; retain handles.
            return HostResult('cleanup_unconfirmed', 'cancel_write_failed', None, False, child, out, err)
        try:
            cancel_started = clock()
            if not isinstance(cancel_started, (int, float)) or not math.isfinite(cancel_started):
                raise ValueError('Invalid cleanup clock')
            while True:
                now = clock()
                if (not isinstance(now, (int, float)) or not math.isfinite(now)
                        or now < cancel_started or now - cancel_started >= CLEANUP_SECONDS):
                    raise TimeoutError('Cleanup not confirmed')
                exit_code = child.poll()
                if exit_code is not None:
                    if type(exit_code) is not int:
                        raise ValueError('Invalid transport exit')
                    break
                try:
                    child.wait(timeout=min(POLL_SECONDS, CLEANUP_SECONDS - (now - cancel_started)))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            return HostResult('cleanup_unconfirmed', reason, None, False, child, out, err)

    try:
        # WSL may exit while Linux descendants survive; only the pidfd receipt
        # can establish the owned session was reaped and empty.
        report = _report(stage)
        clean = _clean(report)
        if not clean:
            return HostResult('cleanup_unconfirmed', 'linux_cleanup_unconfirmed', exit_code,
                              False, child, out, err)
        externally_canceled = os.path.lexists(stage / CANCEL)
        passed = (reason is None and not externally_canceled and exit_code == 0
                  and report['reason'] == 'exited' and report['completed'] is True
                  and report['child_exit_code'] == 0)
        _close_streams(out, err)
        if passed:
            return HostResult('passed', 'completed', exit_code, True)
        return HostResult('failed', reason or ('canceled' if externally_canceled else 'child_failed'),
                          exit_code, True)
    except BaseException:
        return HostResult('cleanup_unconfirmed', 'receipt_unavailable', exit_code, False, child, out, err)
