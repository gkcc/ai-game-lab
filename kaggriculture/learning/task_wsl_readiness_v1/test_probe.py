"""Pure fake-process and fake-clock regressions; never starts WSL."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from . import probe as p


class FakeClock:
    def __init__(self, factor=1.0):
        self.now, self.factor = 0.0, factor

    def read(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds * self.factor


class FakeProcess:
    def __init__(self, clock, *, exit_at=12.0, code=0):
        self.clock, self.exit_at, self.code = clock, exit_at, code
        self.wait_calls = []

    def poll(self):
        return self.code if self.clock.now >= self.exit_at else None

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.poll() is not None:
            return self.code
        if timeout is not None and self.clock.now + timeout >= self.exit_at:
            self.clock.now = self.exit_at
            return self.code
        if timeout is not None:
            self.clock.now += timeout
        raise subprocess.TimeoutExpired('private', timeout)


class ReadinessTests(unittest.TestCase):
    def run_probe(self, *, memory=None, process=None, clock=None):
        clock = clock or FakeClock()
        process = process or FakeProcess(clock)
        calls = []
        def spawn(argv, **kwargs):
            calls.append((argv, kwargs))
            return process
        result = p.probe('PRIVATE_EXECUTABLE', 'PRIVATE_DISTRO', available=memory or (lambda: 6*p.GIB),
                         clock=clock.read, sleep=clock.sleep, popen=spawn)
        return result, process, calls, clock

    def test_initial_low_memory_never_launches(self):
        def forbidden(*_args, **_kwargs):
            self.fail('must not start below memory floor')
        result = p.probe('private', 'private', available=lambda: 5*p.GIB-1,
                         clock=lambda: 0, sleep=lambda _value: None, popen=forbidden)
        self.assertIs(result.readiness, False)
        self.assertEqual(result.reason, 'initial_memory_low')
        self.assertFalse(result.started)
        self.assertFalse(result.transport_reaped)
        self.assertIsNone(result.exit_code)

    def test_normal_exit_and_sanitized_no_shell_launch(self):
        result, process, calls, _clock = self.run_probe(memory=lambda: 5*p.GIB)
        self.assertIs(result.readiness, True)
        self.assertTrue(result.started and result.transport_reaped)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[1:], ['--distribution', 'PRIVATE_DISTRO', '--exec', '/bin/sleep', '12'])
        self.assertIs(kwargs['shell'], False)
        self.assertEqual((kwargs['stdin'], kwargs['stdout'], kwargs['stderr']),
                         (subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL))
        self.assertEqual(process.wait_calls[-1], 0)
        text = json.dumps(result.public()) + repr(result)
        self.assertNotIn('PRIVATE_', text)
        self.assertFalse(result.public()['sufficient_for_gpu_training'])

    def test_running_memory_dip_is_remembered_until_transport_exit(self):
        clock = FakeClock()
        def memory():
            return 4*p.GIB-1 if 3 <= clock.now < 3.5 else 6*p.GIB
        result, process, _calls, clock = self.run_probe(clock=clock, memory=memory)
        self.assertEqual((result.readiness, result.reason), (False, 'running_memory_low'))
        self.assertTrue(result.transport_reaped)
        self.assertGreaterEqual(clock.now, 12)
        self.assertEqual(process.wait_calls[-1], 0)

    def test_last_two_seconds_must_all_meet_five_gib(self):
        clock = FakeClock()
        def memory():
            return 4*p.GIB + p.GIB//2 if 10.25 <= clock.now < 10.75 else 6*p.GIB
        result, _process, _calls, _clock = self.run_probe(clock=clock, memory=memory)
        self.assertEqual((result.readiness, result.reason), (False, 'tail_memory_low'))
        self.assertTrue(result.transport_reaped)

    def test_sampling_gap_or_exception_is_unknown_and_waits(self):
        result, process, _calls, _clock = self.run_probe(clock=FakeClock(factor=4))
        self.assertEqual((result.readiness, result.reason), (None, 'incomplete_sampling'))
        self.assertTrue(result.transport_reaped)
        self.assertEqual(process.wait_calls[-1], 0)
        clock = FakeClock()
        def broken_memory():
            if clock.now >= 1:
                raise OSError('PRIVATE_DETAIL')
            return 6*p.GIB
        result, process, _calls, _clock = self.run_probe(clock=clock, memory=broken_memory)
        self.assertEqual((result.readiness, result.reason), (None, 'sampling_or_exit_unknown'))
        self.assertTrue(result.transport_reaped)
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn('PRIVATE_DETAIL', repr(result) + json.dumps(result.public()))
        self.assertTrue(process.wait_calls)

    def test_slow_final_sample_cannot_claim_complete_coverage(self):
        clock = FakeClock()
        def slow_memory():
            if clock.now >= 11.75:
                clock.now += 1.0
            return 6*p.GIB
        result, _process, _calls, _clock = self.run_probe(clock=clock, memory=slow_memory)
        self.assertEqual((result.readiness, result.reason), (None, 'incomplete_sampling'))
        self.assertTrue(result.transport_reaped)

    def test_nonzero_or_too_short_transport_cannot_pass(self):
        for exit_at, code, reason in ((12, 3, 'transport_nonzero'), (2, 0, 'transport_too_short')):
            with self.subTest(reason=reason):
                clock = FakeClock()
                process = FakeProcess(clock, exit_at=exit_at, code=code)
                result, _process, _calls, _clock = self.run_probe(clock=clock, process=process)
                self.assertEqual((result.readiness, result.reason, result.exit_code), (False, reason, code))
                self.assertTrue(result.transport_reaped)

    def test_spawn_delay_does_not_count_as_warmup(self):
        clock = FakeClock()
        process = FakeProcess(clock, exit_at=12.0)
        def slow_spawn(*_args, **_kwargs):
            clock.now += 0.5
            return process
        result = p.probe('private', 'private', available=lambda: 6*p.GIB,
                         clock=clock.read, sleep=clock.sleep, popen=slow_spawn)
        self.assertEqual((result.readiness, result.reason), (False, 'transport_too_short'))
        self.assertTrue(result.transport_reaped)

    def test_timeout_retains_own_handle_then_waits_without_upgrading_unknown(self):
        clock = FakeClock()
        process = FakeProcess(clock, exit_at=100)
        result, _process, _calls, _clock = self.run_probe(clock=clock, process=process)
        self.assertEqual((result.readiness, result.reason), (None, 'transport_unconfirmed'))
        self.assertFalse(result.transport_reaped)
        self.assertIs(result._process, process)
        result = result.wait_retained(timeout=10)
        self.assertIs(result._process, process)
        process.exit_at = clock.now + 0.5
        result = result.wait_retained(timeout=10)
        self.assertIsNone(result.readiness)
        self.assertTrue(result.transport_reaped)
        self.assertEqual(result.exit_code, 0)
        self.assertIsNone(result._process)

    def test_exit_first_observed_at_deadline_is_unknown(self):
        clock = FakeClock()
        process = FakeProcess(clock, exit_at=p.TRANSPORT_DEADLINE)
        result, _process, _calls, _clock = self.run_probe(clock=clock, process=process)
        self.assertEqual((result.readiness, result.reason), (None, 'exit_observed_after_deadline'))
        self.assertTrue(result.transport_reaped)
        self.assertEqual(result.exit_code, 0)

    def test_spawn_exception_does_not_claim_cleanup(self):
        def broken(*_args, **_kwargs):
            raise OSError('PRIVATE_PATH')
        result = p.probe('private', 'private', available=lambda: 6*p.GIB,
                         clock=lambda: 0, sleep=lambda _value: None, popen=broken)
        self.assertEqual((result.readiness, result.reason, result.transport_reaped),
                         (None, 'spawn_unconfirmed', False))
        self.assertIsNone(result.started)
        self.assertNotIn('PRIVATE_PATH', repr(result) + json.dumps(result.public()))

    def test_contract_parser_uses_only_synthetic_memory_bytes(self):
        raw = json.dumps({'distribution': 'synthetic', 'schema': 'older-compatible'}).encode()
        with (patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_file', return_value=True),
              patch.object(Path, 'open', return_value=io.BytesIO(raw))):
            self.assertEqual(p._load_distribution('synthetic'), 'synthetic')
        duplicate = b'{"distribution":"a","distribution":"b"}'
        with (patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_file', return_value=True),
              patch.object(Path, 'open', return_value=io.BytesIO(duplicate)), self.assertRaises(ValueError)):
            p._load_distribution('synthetic')

    def test_cli_retains_unknown_handle_and_never_prints_private_inputs(self):
        clock = FakeClock()
        process = FakeProcess(clock, exit_at=0)
        result = p.ProbeResult(None, 'transport_unconfirmed', True, False, None, process)
        stdout = io.StringIO()
        with (patch.object(p, '_load_distribution', return_value='PRIVATE_DISTRO'),
              patch('learning.task_pair_wsl_launch_v1.host._system_wsl', return_value='PRIVATE_PATH'),
              patch.object(p, 'probe', return_value=result), redirect_stdout(stdout)):
            self.assertEqual(p.main(), 1)
        text = stdout.getvalue()
        self.assertNotIn('PRIVATE_', text)
        self.assertEqual(len(text.splitlines()), 2)
        self.assertTrue(process.wait_calls)
        self.assertIsNone(json.loads(text.splitlines()[-1])['readiness'])

    def test_broken_stdout_does_not_discard_unknown_transport(self):
        clock = FakeClock()
        process = FakeProcess(clock, exit_at=0)
        result = p.ProbeResult(None, 'transport_unconfirmed', True, False, None, process)
        with (patch.object(p, '_load_distribution', return_value='PRIVATE_DISTRO'),
              patch('learning.task_pair_wsl_launch_v1.host._system_wsl', return_value='PRIVATE_PATH'),
              patch.object(p, 'probe', return_value=result), patch('builtins.print', side_effect=BrokenPipeError)):
            self.assertEqual(p.main(), 1)
        self.assertTrue(process.wait_calls)

    def test_repeated_early_wait_errors_do_not_spin(self):
        class FlakyProcess:
            def __init__(self): self.calls = 0
            def wait(self, timeout):
                self.calls += 1
                if self.calls == 1: raise OSError('synthetic')
                return 0
        process = FlakyProcess()
        result = p.ProbeResult(None, 'transport_unconfirmed', True, False, None, process)
        with (patch.object(p, '_load_distribution', return_value='PRIVATE_DISTRO'),
              patch('learning.task_pair_wsl_launch_v1.host._system_wsl', return_value='PRIVATE_PATH'),
              patch.object(p, 'probe', return_value=result), patch.object(p.time, 'sleep') as pause,
              redirect_stdout(io.StringIO())):
            self.assertEqual(p.main(), 1)
        self.assertEqual(process.calls, 2)
        pause.assert_called_once()


if __name__ == '__main__':
    unittest.main()
