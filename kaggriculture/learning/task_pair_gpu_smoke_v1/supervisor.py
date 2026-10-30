"""Linux owned-session wrapper for the single synthetic GPU smoke attempt.

The child stdout is captured in a bounded private pipe and never printed;
stderr and native diagnostics are discarded. A zero exit alone is insufficient:
the owned child session must also provide one canonical success receipt with
this run's random token. The pipe does not authenticate the writer's PID. The
existing Linux guard owns and reaps the child session.
"""

import argparse
from contextlib import contextmanager
import hmac
import json
import math
import numbers
import os
from pathlib import Path
import re
import secrets
import signal
import sys


WORK_SECONDS = 120
TOTAL_SECONDS = 135
MAX_RECEIPT_BYTES = 4096
MAX_READS_PER_POLL = 16
TOKEN_PATTERN = re.compile(r'[0-9a-f]{32}\Z')
RECEIPT_SCHEMA = 'task-pair-gpu-smoke-receipt-v1'
IMPORT_GUARD_SHA = 'c5ec904b6c7af23ffb4a6ddd31dd8aa47a0e760cc892087f07e038b303c65f3d'
ERROR_TYPES = frozenset(('ValueError', 'RuntimeError', 'TimeoutError', 'ImportError',
                         'ModuleNotFoundError', 'OSError', 'MemoryError', 'KeyboardInterrupt',
                         'SystemExit'))


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid supervisor arguments')


def bind_project_imports():
    """Load a bounded, SHA-fixed guard before any project package import."""
    import hashlib
    import stat
    import types
    path = Path(__file__).resolve().with_name('import_guard.py')
    info = path.lstat()
    limit = 1024 * 1024
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400 or info.st_size > limit):
        raise ValueError('Untrusted import guard')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or current.st_size > limit:
            raise ValueError('Untrusted import guard')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
    finally:
        os.close(descriptor)
    if (len(raw) != current.st_size or len(raw) > limit
            or hashlib.sha256(raw).hexdigest() != IMPORT_GUARD_SHA):
        raise ValueError('Import guard identity differs')
    guard = types.ModuleType('smoke_import_guard')
    guard.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), guard.__dict__)
    root = guard.bind(Path(__file__).resolve().parents[2])
    return guard, root


def device_name(value):
    """Validate the local name without importing the child or echoing it."""
    if (type(value) is not str or not 1 <= len(value) <= 128 or value != value.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('Invalid requested device identity')
    return value


@contextmanager
def quiet_native_output():
    """Bootstrap without exposing dependency or child/native diagnostics."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = saved_err = sink = None
    try:
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        sink = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if saved_out is not None:
            os.dup2(saved_out, 1)
        if saved_err is not None:
            os.dup2(saved_err, 2)
        for descriptor in (saved_out, saved_err, sink):
            if descriptor is not None:
                os.close(descriptor)


def receipt_token(value):
    if type(value) is not str or TOKEN_PATTERN.fullmatch(value) is None:
        raise ValueError('Invalid receipt token')
    return value


def child_command(executable, expected_name, token):
    """Construct an isolated child command without serializing it to logs."""
    expected_name = device_name(expected_name)
    token = receipt_token(token)
    if type(executable) is not str or not executable:
        raise ValueError('Invalid child interpreter')
    return [executable, '-B', '-I', str(Path(__file__).with_name('run.py')),
            '--expected-device-name', expected_name, '--receipt-token', token]


class ReceiptPipe:
    """Bounded, nonblocking stdout collector; no wait for an inherited writer."""
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.data = bytearray()
        self.eof = False
        self.overflow = False

    def drain(self):
        for _ in range(MAX_READS_PER_POLL):
            try:
                part = os.read(self.descriptor, MAX_RECEIPT_BYTES)
            except BlockingIOError:
                return
            if not part:
                self.eof = True
                return
            room = max(0, MAX_RECEIPT_BYTES - len(self.data))
            if len(part) > room:
                self.overflow = True
            self.data.extend(part[:room])


@contextmanager
def capture_receipt_pipe():
    """Attach child stdout to a pipe, then nonblockingly drain and close it."""
    reader = writer = saved_out = None
    collector = None
    try:
        reader, writer = os.pipe()
        os.set_blocking(reader, False)
        collector = ReceiptPipe(reader)
        sys.stdout.flush()
        saved_out = os.dup(1)
        os.dup2(writer, 1)
        os.close(writer)
        writer = None
        yield collector
    finally:
        sys.stdout.flush()
        if saved_out is not None:
            os.dup2(saved_out, 1)
            os.close(saved_out)
        if writer is not None:
            os.close(writer)
        try:
            if collector is not None:
                collector.drain()
        finally:
            if reader is not None:
                os.close(reader)


def valid_receipt(raw, token):
    """Require exactly one canonical JSON receipt with strictly typed values."""
    token = receipt_token(token)
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_RECEIPT_BYTES:
        return False

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate receipt field')
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError('Invalid receipt constant')

    try:
        result = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                            parse_constant=reject_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError):
        return False
    expected = {'schema': RECEIPT_SCHEMA, 'token': token, 'passed': True,
                'unique_requested_gpu_matched': True, 'paired_updates_each': 2,
                'aux_nonzero_updates': 2, 'both_arms_changed': True, 'arms_differ': True,
                'fit_device_and_finite_checks_passed': True}
    if (type(result) is not dict or set(result) != set(expected)
            or any(type(result[key]) is not type(value) or result[key] != value
                   for key, value in expected.items())
            or not hmac.compare_digest(result['token'], token)):
        return False
    canonical = (json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    return raw == canonical


def project_guard(report, receipt_valid):
    """Publish only booleans/counts after both guard and receipt validation."""
    required = {'reason', 'child_exit_code', 'elapsed_seconds',
                'owned_process_group_signalled', 'direct_child_reaped',
                'no_live_owned_session_members_before_reap', 'completed'}
    if (type(receipt_valid) is not bool or type(report) is not dict or set(report) != required
            or report['reason'] not in ('exited', 'deadline', 'cancelled')
            or type(report['child_exit_code']) is not int
            or isinstance(report['elapsed_seconds'], bool)
            or not isinstance(report['elapsed_seconds'], numbers.Real)
            or not math.isfinite(report['elapsed_seconds']) or report['elapsed_seconds'] < 0
            or any(type(report[key]) is not bool for key in (
                'owned_process_group_signalled', 'direct_child_reaped',
                'no_live_owned_session_members_before_reap', 'completed'))):
        raise ValueError('Invalid owned-session receipt')
    clean = (report['owned_process_group_signalled'] and report['direct_child_reaped']
             and report['no_live_owned_session_members_before_reap'])
    within_budget = report['elapsed_seconds'] <= TOTAL_SECONDS
    passed = (report['reason'] == 'exited' and report['child_exit_code'] == 0
              and report['completed'] and clean and within_budget and receipt_valid)
    if not passed:
        kind = ('TimeoutError' if report['reason'] == 'deadline' or not within_budget else
                'Cancelled' if report['reason'] == 'cancelled' else
                'InvalidReceipt' if report['reason'] == 'exited' and report['child_exit_code'] == 0
                and report['completed'] and clean and not receipt_valid else 'ChildFailed')
        return {'passed': False, 'stage': 'receipt' if kind == 'InvalidReceipt' else 'supervisor', 'error_type': kind,
                'owned_session_clean': clean}
    return {'passed': True, 'child_exit_zero': True, 'owned_session_clean': True,
            'matching_success_receipt': True,
            'guard_with_cleanup_within_budget': True,
            'unique_requested_gpu_matched': True, 'paired_updates_each': 2,
            'aux_nonzero_updates': 2, 'both_arms_changed': True,
            'arms_differ': True, 'fit_device_and_finite_checks_passed': True}


def supervise_command(command, token, owned_session, cancelled):
    """Collect one child receipt while reusing the existing owned-session guard."""
    receipt_token(token)
    if type(command) is not list or not command or not callable(owned_session) or not callable(cancelled):
        raise ValueError('Invalid owned child request')
    with quiet_native_output():
        with capture_receipt_pipe() as collector:
            def poll():
                collector.drain()
                return cancelled()
            report = owned_session(command, WORK_SECONDS, poll)
        accepted = (collector.eof and not collector.overflow
                    and valid_receipt(bytes(collector.data), token))
        return project_guard(report, accepted)


def supervise(expected_name, import_guard):
    # Import dependencies only after low-level stdout and stderr are suppressed;
    # import errors, native diagnostics and child output remain private.
    with quiet_native_output():
        if os.name != 'posix':
            raise RuntimeError('Linux session supervision is required')
        token = secrets.token_hex(16)
        command = child_command(sys.executable, expected_name, token)
        import importlib
        with import_guard[0].protected_search_path(import_guard[1]):
            guard_module = importlib.import_module('learning.task_dagger_once_v1.gpu_guard')
        import_guard[0].verify_origins(import_guard[1])
        owned_session = guard_module.supervise

        cancelled = {'value': False}

        def cancel(signum, frame):
            del signum, frame
            cancelled['value'] = True

        prior = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for sig in prior:
                signal.signal(sig, cancel)
            return supervise_command(command, token, owned_session, lambda: cancelled['value'])
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)


def cli(argv=None):
    try:
        parser = QuietParser(add_help=False)
        parser.add_argument('--expected-device-name', required=True)
        args = parser.parse_args(argv)
        expected = device_name(args.expected_device_name)
        with quiet_native_output():
            binding = bind_project_imports()
            result = supervise(expected, binding)
            binding[0].verify_origins(binding[1])
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        kind = type(exc).__name__
        result = {'passed': False, 'stage': 'supervisor',
                  'error_type': kind if kind in ERROR_TYPES else 'OtherError'}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(cli())
