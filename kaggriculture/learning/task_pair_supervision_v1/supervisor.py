"""Parameterizable supervision using the existing owned-session and pipe guards.

The caller must authenticate registration, command, dependencies, source,
device and artifacts. Receipt data are child self-report, not independent
proof of training. This module does not start training by itself or write files.
"""

from dataclasses import dataclass
import hmac
import json
import math
import os
import signal
import threading

from learning.task_pair_pidfd_guard_v1 import guard as pidfd_guard
from learning.task_pair_gpu_smoke_v1 import supervisor as smoke


RECEIPT_SCHEMA = 'task-pair-training-supervisor-receipt-v1'
MAX_SECONDS = 86_400
MAX_COMMAND_ARGS = 64
MAX_COMMAND_CHARS = 65_536
_RECEIPT_KEYS = frozenset(('schema', 'token', 'passed', 'completed_updates',
                            'auxiliary_nonzero_updates', 'both_arms_changed', 'arms_differ'))
_RECEIPT_FLAGS = frozenset(('canonical_token_receipt', 'child_reported_success',
                             'registered_update_count_met', 'registered_auxiliary_threshold_met',
                             'both_arms_changed_and_differ'))
_GUARD_KEYS = frozenset(('reason', 'child_exit_code', 'elapsed_seconds',
                          'owned_session_empty_before_reap', 'direct_child_reaped',
                          'used_pidfd', 'completed'))


@dataclass(frozen=True)
class Contract:
    """Values independently fixed by the future registration, not defaults."""
    expected_updates: int
    minimum_auxiliary_nonzero: int
    work_seconds: float
    guard_with_cleanup_seconds: float
    term_seconds: float
    cleanup_seconds: float
    gate_seconds: float
    expected_gate_sha256: str


def _contract(value):
    if (type(value) is not Contract
            or type(value.expected_updates) is not int or not 1 <= value.expected_updates <= 1024
            or type(value.minimum_auxiliary_nonzero) is not int
            or not 1 <= value.minimum_auxiliary_nonzero <= value.expected_updates):
        raise ValueError('Invalid registered update contract')
    for seconds in (value.work_seconds, value.guard_with_cleanup_seconds,
                    value.term_seconds, value.cleanup_seconds, value.gate_seconds):
        if type(seconds) not in (int, float) or not 0 < seconds <= MAX_SECONDS:
            raise ValueError('Invalid registered supervision budget')
    if (value.term_seconds >= value.cleanup_seconds
            or value.gate_seconds > value.work_seconds
            or value.work_seconds + value.cleanup_seconds > value.guard_with_cleanup_seconds):
        raise ValueError('Missing cleanup time reservation')
    if type(value.expected_gate_sha256) is not str or value.expected_gate_sha256 != pidfd_guard.GATE_SHA256:
        raise ValueError('Invalid registered gate identity')
    return value


def _empty_receipt():
    return {key: False for key in _RECEIPT_FLAGS}


def inspect_receipt(raw, token, contract):
    """Check one canonical token receipt and report its separate gates.

    Counts and flags are child self-report; no private receipt content is
    returned. The expected token and contract must come from the supervisor.
    """
    contract = _contract(contract)
    token = smoke.receipt_token(token)
    empty = _empty_receipt()
    if type(raw) is not bytes or not 1 <= len(raw) <= smoke.MAX_RECEIPT_BYTES:
        return empty

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate receipt key')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('Invalid receipt constant')

    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError):
        return empty
    if (type(value) is not dict or set(value) != _RECEIPT_KEYS
            or value['schema'] != RECEIPT_SCHEMA or type(value['token']) is not str
            or smoke.TOKEN_PATTERN.fullmatch(value['token']) is None
            or not hmac.compare_digest(value['token'], token)
            or any(type(value[name]) is not bool for name in ('passed', 'both_arms_changed', 'arms_differ'))
            or type(value['completed_updates']) is not int
            or not 0 <= value['completed_updates'] <= 1024
            or type(value['auxiliary_nonzero_updates']) is not int
            or not 0 <= value['auxiliary_nonzero_updates'] <= value['completed_updates']):
        return empty
    canonical = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    if raw != canonical:
        return empty
    return {'canonical_token_receipt': True,
            'child_reported_success': value['passed'],
            'registered_update_count_met': value['completed_updates'] == contract.expected_updates,
            'registered_auxiliary_threshold_met':
                value['auxiliary_nonzero_updates'] >= contract.minimum_auxiliary_nonzero,
            'both_arms_changed_and_differ': value['both_arms_changed'] and value['arms_differ']}


def project_guard(report, receipt, contract):
    """Preliminarily combine gates; sink acceptance is checked separately."""
    contract = _contract(contract)
    if (type(report) is not dict or set(report) != _GUARD_KEYS
            or report['reason'] not in ('exited', 'deadline', 'cancelled')
            or type(report['child_exit_code']) is not int
            or type(report['elapsed_seconds']) not in (int, float)
            or (type(report['elapsed_seconds']) is float and not math.isfinite(report['elapsed_seconds']))
            or report['elapsed_seconds'] < 0
            or any(type(report[key]) is not bool for key in (
                'owned_session_empty_before_reap', 'direct_child_reaped',
                'used_pidfd', 'completed'))
            or report['completed'] != (report['reason'] == 'exited' and report['child_exit_code'] == 0)
            or type(receipt) is not dict or set(receipt) != _RECEIPT_FLAGS
            or any(type(value) is not bool for value in receipt.values())):
        raise ValueError('Invalid owned-session report')
    clean = (report['owned_session_empty_before_reap'] and report['direct_child_reaped']
             and report['used_pidfd'])
    result = {'child_started': True,
              'child_exit_observed': report['reason'] == 'exited',
              'child_exit_zero': report['child_exit_code'] == 0,
              'work_deadline_triggered': report['reason'] == 'deadline',
              'cancelled': report['reason'] == 'cancelled',
              'owned_session_empty_before_reap': report['owned_session_empty_before_reap'],
              'direct_child_reaped': report['direct_child_reaped'],
              'used_pidfd': report['used_pidfd'],
              'owned_session_clean': clean,
              'guard_with_cleanup_within_budget': report['elapsed_seconds'] <= contract.guard_with_cleanup_seconds,
              **receipt}
    result['passed'] = (result['child_exit_observed'] and result['child_exit_zero']
                        and result['owned_session_clean'] and result['guard_with_cleanup_within_budget']
                        and all(receipt.values()))
    return result


def _command(command, token):
    token = smoke.receipt_token(token)
    if (type(command) is not list or not 1 <= len(command) <= MAX_COMMAND_ARGS
            or any(type(item) is not str or not item or '\x00' in item for item in command)
            or sum(len(item) for item in command) > MAX_COMMAND_CHARS
            or command.count('--receipt-token') != 1):
        raise ValueError('Invalid owned child command')
    index = command.index('--receipt-token')
    if index + 1 >= len(command) or command[index + 1] != token or command.count(token) != 1:
        raise ValueError('Child token argument differs')
    return command.copy()


def supervise_command(command, token, contract, *, cancelled, receipt_sink):
    """Run one owned Linux session and return only sanitized boolean gates.

    ``command`` must pass the given token once via ``--receipt-token``. The
    caller fixes and authenticates that command, token, child and contract;
    each attempt needs a fresh cryptographically random 32-hex token, fixed
    privately before startup. Syntax validation cannot prove its provenance.
    Only after all gates and a post-cleanup cancellation sample pass is the
    bounded canonical raw receipt sent to an authenticated ``receipt_sink``.
    That callback must privately write and read back the exact bytes; its True
    return is self-report, not independent proof. This function does not
    validate data, source identity or GPU. A process
    exception yields a generic error, never a successful cleanup claim.
    The 24-hour bound is a validation ceiling, not a registered run budget.
    """
    contract = _contract(contract)
    command = _command(command, token)
    if os.name != 'posix' or not callable(cancelled) or not callable(receipt_sink):
        raise ValueError('Linux owned-session supervision is required')
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Owned-session supervision requires the main thread')
    state = {'signal': False}

    def on_signal(signum, frame):
        del signum, frame
        state['signal'] = True

    def requested():
        if state['signal']:
            return True
        value = cancelled()
        if type(value) is not bool:
            raise ValueError('Invalid cancellation callback')
        return state['signal'] or value

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, on_signal)
        with smoke.quiet_native_output():
            # Sample in the native-output suppression scope before creating a
            # child: callbacks may themselves print sensitive diagnostics.
            if requested():
                return {'passed': False, 'child_started': False, 'child_exit_observed': False,
                        'child_exit_zero': False, 'work_deadline_triggered': False,
                        'owned_session_empty_before_reap': False, 'direct_child_reaped': False,
                        'used_pidfd': False, 'receipt_sink_attempted': False, 'receipt_sink_accepted': False,
                        'cancelled': True, 'owned_session_clean': False,
                        'guard_with_cleanup_within_budget': False, **_empty_receipt()}
            with smoke.capture_receipt_pipe() as collector:
                def poll():
                    collector.drain()
                    return requested()
                report = pidfd_guard.supervise(command, contract.work_seconds, poll,
                                                term_seconds=contract.term_seconds,
                                                cleanup_seconds=contract.cleanup_seconds,
                                                gate_seconds=contract.gate_seconds,
                                                expected_gate_sha256=contract.expected_gate_sha256)
            raw = bytes(collector.data) if collector.eof and not collector.overflow else b''
            receipt = inspect_receipt(raw, token, contract)
            result = project_guard(report, receipt, contract)
            # This is a sampled decision after cleanup, not an exact timestamp.
            # A cancellation already visible here must override child success.
            if requested():
                result['cancelled'] = True
                result['passed'] = False
            result['receipt_sink_attempted'] = False
            result['receipt_sink_accepted'] = False
            if result['passed']:
                # A trusted sink alone handles the exact private bytes; no
                # raw receipt or token enters the sanitized result.
                result['receipt_sink_attempted'] = True
                try:
                    accepted = receipt_sink(raw)
                except BaseException:
                    accepted = False
                if accepted is True:
                    result['receipt_sink_accepted'] = True
                else:
                    result['passed'] = False
                # A later sampled cancellation wins, even though the sink
                # may already have saved a valid local receipt.
                if requested():
                    result['cancelled'] = True
                    result['passed'] = False
            return result
    except BaseException:
        raise RuntimeError('Owned session report unavailable; cleanup not confirmed') from None
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
