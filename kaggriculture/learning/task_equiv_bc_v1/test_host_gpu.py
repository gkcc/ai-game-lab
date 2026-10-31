"""Host-supervision regression using synthetic handles; never starts WSL."""
import io
import subprocess
import unittest
from unittest.mock import Mock, patch

from learning.task_equiv_bc_v1 import host_gpu as host


COMMIT = 'a' * 40
ARGV = ['wsl.exe', '-d', 'synthetic', '--exec', 'env',
        'XLA_PYTHON_CLIENT_PREALLOCATE=false', '/synthetic/python', '-I', '-B',
        '/synthetic/gpu_supervisor.py', '--stage', '/synthetic/stage', '--registration-commit', COMMIT]


class FakeStage:
    def __truediv__(self, item):
        return item


class FakeClock:
    def __init__(self):
        self.time = 0.
    def __call__(self):
        return self.time


class FakeProcess:
    def __init__(self, clock, *, exits_after=None, code=0, increment=1.):
        self.clock, self.exits_after, self.code, self.increment = clock, exits_after, code, increment
        self.waits = 0
        self.kill = Mock(side_effect=AssertionError('Must never kill transport'))
        self.terminate = Mock(side_effect=AssertionError('Must never terminate transport'))
    def poll(self):
        return self.code if self.exits_after is not None and self.waits >= self.exits_after else None
    def wait(self, timeout):
        self.waits += 1
        increment = (self.increment[min(self.waits - 1, len(self.increment) - 1)]
                     if isinstance(self.increment, tuple) else self.increment)
        self.clock.time += increment
        result = self.poll()
        if result is None:
            raise subprocess.TimeoutExpired('synthetic', timeout)
        return result


def report(completed=True):
    return {'reason': 'exited' if completed else 'canceled', 'completed': completed,
            'child_exit_code': 0 if completed else -15, 'used_pidfd': True,
            'direct_child_reaped': True, 'owned_session_empty_before_reap': True}


class HostTests(unittest.TestCase):
    def run_case(self, child, clock, samples, receipt, *, cancel_side_effect=None):
        out, err = io.BytesIO(), io.BytesIO()
        popen, cancel = Mock(return_value=child), Mock(side_effect=cancel_side_effect)
        available = Mock(side_effect=samples)
        with patch.object(host, '_stage', return_value=FakeStage()), patch.object(host, '_fresh'), \
             patch.object(host, '_logs', return_value=(out, err)), patch.object(host, '_report', return_value=receipt), \
             patch.object(host, '_cancel', cancel), patch.object(host.os.path, 'lexists', return_value=False):
            result = host.supervise('synthetic', ARGV, COMMIT, popen=popen,
                                    available=available, clock=clock)
        return result, popen, cancel, out, err

    def test_normal_exit_needs_linux_pidfd_and_reaped_empty_receipt(self):
        clock = FakeClock()
        child = FakeProcess(clock, exits_after=2)
        result, popen, cancel, out, err = self.run_case(
            child, clock, [host.INITIAL_AVAILABLE] * 5, report())
        self.assertEqual((result.status, result.exit_code, result.cleanup_confirmed), ('passed', 0, True))
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(popen.call_args.kwargs['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
        self.assertTrue(out.closed and err.closed)
        cancel.assert_not_called()
        child.kill.assert_not_called()

    def test_low_memory_cancels_and_never_kills_other_processes(self):
        clock = FakeClock()
        child = FakeProcess(clock, exits_after=2, code=1)
        result, popen, cancel, out, err = self.run_case(
            child, clock, [host.INITIAL_AVAILABLE, host.INITIAL_AVAILABLE, host.RUN_AVAILABLE - 1], report(False))
        self.assertEqual((result.status, result.reason, result.exit_code), ('failed', 'memory_gate', 1))
        self.assertTrue(result.cleanup_confirmed and out.closed and err.closed)
        self.assertEqual(cancel.call_args.args[-1], 'memory_gate')
        child.kill.assert_not_called()
        child.terminate.assert_not_called()

    def test_wall_timeout_cancels_then_reports_failure_even_after_clean_exit(self):
        clock = FakeClock()
        child = FakeProcess(clock, exits_after=2, code=0, increment=(901., 1.))
        result, _, cancel, _, _ = self.run_case(
            child, clock, [host.INITIAL_AVAILABLE] * 4, report(False))
        self.assertEqual((result.status, result.reason), ('failed', 'time_gate'))
        self.assertEqual(cancel.call_args.args[-1], 'time_gate')
        self.assertTrue(result.cleanup_confirmed)

    def test_cleanup_timeout_keeps_process_and_both_log_handles(self):
        clock = FakeClock()
        child = FakeProcess(clock, increment=31.)
        result, _, cancel, out, err = self.run_case(
            child, clock, [host.INITIAL_AVAILABLE, host.INITIAL_AVAILABLE, host.RUN_AVAILABLE - 1], report(False))
        self.assertEqual(result.status, 'cleanup_unconfirmed')
        self.assertIs(result.child, child)
        self.assertIs(result.stdout_stream, out)
        self.assertIs(result.stderr_stream, err)
        self.assertFalse(out.closed or err.closed)
        child.kill.assert_not_called()
        child.terminate.assert_not_called()
        out.close()
        err.close()

    def test_zero_transport_exit_without_linux_cleanup_is_unconfirmed(self):
        clock = FakeClock()
        child = FakeProcess(clock, exits_after=0)
        bad = dict(report(), owned_session_empty_before_reap=False)
        result, _, _, out, err = self.run_case(child, clock, [host.INITIAL_AVAILABLE] * 3, bad)
        self.assertEqual((result.status, result.exit_code), ('cleanup_unconfirmed', 0))
        self.assertIs(result.child, child)
        self.assertFalse(out.closed or err.closed)
        out.close()
        err.close()

    def test_final_memory_sample_can_fail_after_zero_exit(self):
        clock = FakeClock()
        child = FakeProcess(clock, exits_after=0)
        result, _, cancel, out, err = self.run_case(child, clock,
            [host.INITIAL_AVAILABLE, host.INITIAL_AVAILABLE, host.RUN_AVAILABLE - 1], report())
        self.assertEqual((result.status, result.reason, result.exit_code), ('failed', 'memory_gate', 0))
        self.assertTrue(result.cleanup_confirmed and out.closed and err.closed)
        self.assertEqual(cancel.call_args.args[-1], 'memory_gate')

    def test_transport_exit_observed_after_deadline_cannot_pass(self):
        clock = FakeClock()
        child = FakeProcess(clock, exits_after=1, increment=901.)
        result, _, cancel, _, _ = self.run_case(child, clock, [host.INITIAL_AVAILABLE] * 3, report())
        self.assertEqual((result.status, result.reason), ('failed', 'time_gate'))
        self.assertEqual(cancel.call_args.args[-1], 'time_gate')

    def test_cleanup_exit_observed_after_limit_is_unconfirmed(self):
        clock = FakeClock()
        child = FakeProcess(clock, exits_after=1, increment=31.)
        result, _, _, out, err = self.run_case(child, clock,
            [host.INITIAL_AVAILABLE, host.INITIAL_AVAILABLE, host.RUN_AVAILABLE - 1], report(False))
        self.assertEqual(result.status, 'cleanup_unconfirmed')
        self.assertIs(result.child, child)
        self.assertFalse(out.closed or err.closed)
        out.close()
        err.close()

    def test_cancel_marker_write_failure_keeps_handle(self):
        clock = FakeClock()
        child = FakeProcess(clock)
        result, _, _, out, err = self.run_case(child, clock,
            [host.INITIAL_AVAILABLE, host.INITIAL_AVAILABLE, host.RUN_AVAILABLE - 1], report(False),
            cancel_side_effect=OSError('synthetic'))
        self.assertEqual((result.status, result.reason), ('cleanup_unconfirmed', 'cancel_write_failed'))
        self.assertIs(result.child, child)
        child.kill.assert_not_called()
        out.close()
        err.close()

    def test_initial_low_memory_never_opens_logs_or_starts_child(self):
        spawn, logs = Mock(), Mock()
        with patch.object(host, '_stage', return_value=FakeStage()), patch.object(host, '_logs', logs):
            with self.assertRaises(MemoryError):
                host.supervise('synthetic', ARGV, COMMIT, popen=spawn,
                               available=lambda: host.INITIAL_AVAILABLE - 1, clock=FakeClock())
        spawn.assert_not_called()
        logs.assert_not_called()

    def test_spawn_exception_cannot_claim_linux_cleanup(self):
        out, err = io.BytesIO(), io.BytesIO()
        cancel = Mock()
        with patch.object(host, '_stage', return_value=FakeStage()), patch.object(host, '_fresh'), \
             patch.object(host, '_logs', return_value=(out, err)), patch.object(host, '_cancel', cancel):
            result = host.supervise('synthetic', ARGV, COMMIT, popen=Mock(side_effect=OSError('synthetic')),
                                    available=lambda: host.INITIAL_AVAILABLE, clock=FakeClock())
        self.assertEqual((result.status, result.reason), ('cleanup_unconfirmed', 'spawn_uncertain'))
        self.assertIsNone(result.child)
        self.assertIs(result.stdout_stream, out)
        self.assertFalse(out.closed or err.closed)
        self.assertEqual(cancel.call_args.args[-1], 'spawn_uncertain')
        out.close()
        err.close()

    def test_command_requires_matching_registration_and_isolation_flags(self):
        self.assertEqual(host._command(ARGV, COMMIT), ARGV)
        for invalid in ([*ARGV[:-1], 'b' * 40], [x for x in ARGV if x != '-I'],
                        [x for x in ARGV if x != 'XLA_PYTHON_CLIENT_PREALLOCATE=false']):
            with self.assertRaises(ValueError):
                host._command(invalid, COMMIT)

    def test_second_log_creation_failure_closes_first_without_spawning(self):
        out = io.BytesIO()
        class FakePath:
            def __init__(self, name):
                self.name = name
            def open(self, mode, buffering):
                if mode != 'xb' or buffering != 0:
                    raise AssertionError('Logs must be opened exclusively')
                if self.name == host.STDERR:
                    raise FileExistsError('synthetic')
                return out
        class Stage:
            def __truediv__(self, name):
                return FakePath(name)
        with self.assertRaises(FileExistsError):
            host._logs(Stage())
        self.assertTrue(out.closed)


if __name__ == '__main__':
    unittest.main()
