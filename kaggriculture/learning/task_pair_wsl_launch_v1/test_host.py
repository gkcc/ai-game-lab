"""Windows-host controller checks; all contracts and private I/O are synthetic."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_wsl_launch_v1 import host, launch


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def synthetic_contract():
    base = PureWindowsPath('C:' + chr(92)) / 'synthetic'
    return {'schema': host.HOST_SCHEMA, 'distribution': 'synthetic-distro',
        'windows_game_root': str(base / 'game'),
        'windows_training_contract': str(base / 'game' / 'artifacts' / 'training.json'),
        'windows_wsl_executable': str(base / 'wsl.exe'),
        'wsl_game_root': '/synthetic/game',
        'wsl_training_contract': '/synthetic/game/artifacts/training.json',
        'wsl_repo_root': '/synthetic/repo', 'wsl_venv_executable': '/synthetic/venv/bin/python',
        'training_contract_sha256': 'a' * 64}


def guard_result(passed=True):
    result = {name: True for name in launch._GUARD_RESULT_KEYS}
    result.update(passed=passed, work_deadline_triggered=False, cancelled=False)
    if not passed: result['receipt_sink_accepted'] = False
    return result


BOOTSTRAP_FIXTURE = r'''
import hashlib, io, json, os, pathlib, stat, sys, types
bootstrap, mode = sys.argv[1:3]
source = (b"import json,sys\nprint(json.dumps({'executed':True,'file':__file__,"
          b"'args':sys.argv[1:],'opens':sys.synthetic_open_count}))\n")
expected = hashlib.sha256(source).hexdigest()
current = {'bytes': source}
sys.synthetic_open_count = 0
class FakePath:
    def __init__(self, value): self.value = value
    def __str__(self): return self.value
    def lstat(self):
        size = 4 * 1024 * 1024 + 1 if mode == 'oversize' else len(current['bytes'])
        return types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_size=size)
class FakeStream(io.BytesIO):
    def read(self, size=-1):
        raw = super().read(size)
        current['bytes'] = b"raise RuntimeError('replacement must not execute')"
        return raw
def fake_open(path, flags):
    sys.synthetic_open_count += 1
    return 42
pathlib.Path = FakePath
os.open = fake_open
os.fstat = lambda fd: types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_size=len(source))
os.fdopen = lambda fd, mode, closefd: FakeStream(current['bytes'])
os.close = lambda fd: None
if mode == 'wrong_sha': expected = '0' * 64
sys.argv = ['-c', '/synthetic/launcher.py', expected, '--example', 'value']
try:
    exec(compile(bootstrap, '<bootstrap>', 'exec'), {'__name__':'__main__'})
except SystemExit as exc:
    print(json.dumps({'executed':False, 'exit':exc.code, 'opens':sys.synthetic_open_count}))
'''


class FakeClock:
    def __init__(self): self.value = 0.0
    def now(self): return self.value
    def sleep(self, seconds): self.value += seconds


class HostTests(unittest.TestCase):
    def test_bounded_bootstrap_executes_verified_single_snapshot_and_rejects_wrong_inputs(self):
        for mode in ('valid', 'wrong_sha', 'oversize'):
            result = subprocess.run([sys.executable, '-B', '-I', '-c', BOOTSTRAP_FIXTURE,
                                     host.BOUND_BOOTSTRAP, mode],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=15, check=False)
            self.assertEqual(result.returncode, 0, 'Synthetic bootstrap probe failed')
            observed = json.loads(result.stdout)
            if mode == 'valid':
                self.assertEqual(observed, {'executed': True, 'file': '/synthetic/launcher.py',
                                            'args': ['--example', 'value'], 'opens': 1})
            else:
                self.assertEqual(observed, {'executed': False, 'exit': 1,
                                            'opens': 0 if mode == 'oversize' else 1})

    def test_real_public_launcher_snapshot_rejects_wrong_digest_before_main(self):
        path = launch.REPO / launch.LAUNCH_REL
        digest = hashlib.sha256(path.read_bytes()).hexdigest()  # Public source only.
        def invoke(expected):
            return subprocess.run([sys.executable, '-B', '-I', '-c', host.BOUND_BOOTSTRAP,
                                   str(path), expected], stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=False)
        wrong = invoke('0' * 64)
        self.assertEqual((wrong.returncode, wrong.stdout), (1, b''))
        correct = invoke(digest)
        self.assertEqual(correct.returncode, 1)  # Missing required args, no registration or training.
        self.assertEqual(json.loads(correct.stdout)['stage'], 'arguments')

    def test_unregistered_or_different_launcher_source_stops_before_private_access(self):
        commit = 'f' * 40
        host_raw, launch_raw = b'synthetic-host', b'synthetic-launcher'
        registered = encode({'source_sha256': {
            host.HOST_REL: hashlib.sha256(host_raw).hexdigest(),
            host.LAUNCH_REL: hashlib.sha256(launch_raw).hexdigest(), host.HOST_DOC_REL: 'a' * 64}})
        def blob(requested, relative):
            if relative == host.REG_REL: return registered
            if relative == host.HOST_REL: return b'different-committed-host'
            return launch_raw
        def bounded(path, limit):
            if path == host.REPO / host.REG_REL: return registered
            if path == host.REPO / host.HOST_REL: return host_raw
            return launch_raw
        with patch.object(host, '_head', return_value=commit), patch.object(host, '_blob', side_effect=blob), \
             patch.object(host, '_bounded', side_effect=bounded), \
             patch.object(host.types, 'ModuleType') as module, self.assertRaises(ValueError):
            host.authenticate_sources(commit)
        module.assert_not_called()

    def test_host_contract_convention_must_be_registered_before_import(self):
        commit = 'f' * 40
        registered = encode({'source_sha256': {host.HOST_REL: '1' * 64, host.LAUNCH_REL: '2' * 64}})
        with patch.object(host, '_head', return_value=commit), patch.object(host, '_blob', return_value=registered), \
             patch.object(host, '_bounded', return_value=registered), \
             patch.object(host.types, 'ModuleType') as module, self.assertRaises(ValueError):
            host.authenticate_sources(commit)
        module.assert_not_called()

    def test_contract_digest_and_absolute_path_constraints(self):
        contract = synthetic_contract()
        raw = encode(contract)
        sha = hashlib.sha256(raw).hexdigest()
        self.assertEqual(host.parse_host_contract(raw, sha), contract)
        with self.assertRaises(ValueError):
            host.parse_host_contract(raw + b' ', sha)
        for patch_value in ({'windows_game_root': r'..\other'},
                            {'wsl_training_contract': '/synthetic/elsewhere/file'},
                            {'wsl_repo_root': '/synthetic/../other'},
                            {'wsl_venv_executable': 'relative/python'}):
            bad = encode(contract | patch_value)
            with self.subTest(field=tuple(patch_value)), self.assertRaises(ValueError):
                host.parse_host_contract(bad, hashlib.sha256(bad).hexdigest())

    def test_private_contract_is_read_only_after_registered_source_gate(self):
        identity = synthetic_contract()
        training = {'schema': 'synthetic-training-schema',
                    'runtime_identity': {'venv_executable': identity['wsl_venv_executable'],
                                         'venv_prefix': '/synthetic/venv'}}
        training_raw = encode(training)
        identity['training_contract_sha256'] = hashlib.sha256(training_raw).hexdigest()
        host_raw = encode(identity)
        expected = hashlib.sha256(host_raw).hexdigest()
        root = Path(identity['windows_game_root'])
        artifacts = root / 'artifacts'
        events = []
        fake_entry = SimpleNamespace(PRIVATE_SCHEMA='synthetic-training-schema')
        def root_fn(path):
            events.append('root')
            return root, artifacts
        def read(path, art, limit):
            events.append('private_read')
            return host_raw if len(events) == 3 else training_raw
        fake_entry._private_root, fake_entry._private_file = root_fn, read
        registration = {'limits': {'outer_total_seconds': 100}}
        def sources(commit):
            events.append('sources')
            return object(), fake_entry, registration
        with patch.object(host, 'authenticate_sources', side_effect=sources), \
             patch.object(host, '_system_wsl', return_value=Path(identity['windows_wsl_executable'])), \
             patch.object(Path, 'resolve', lambda path, strict=False: path):
            prepared = host.prepare('a' * 40, root, artifacts / 'host.json', expected,
                                    started=0, clock=lambda: 1)
        self.assertEqual(repr(prepared), '<PrivateRegisteredWindowsHost>')
        self.assertEqual(events, ['sources', 'root', 'private_read', 'private_read'])
        events.clear()
        with patch.object(host, 'authenticate_sources', side_effect=sources), self.assertRaises(ValueError):
            host.prepare('a' * 40, root, artifacts / 'host.json', 'f' * 64, started=0, clock=lambda: 1)
        self.assertEqual(events, ['sources', 'root', 'private_read'])

    def test_short_lived_real_process_uses_argv_without_gpu_or_files(self):
        evidence = {'state': 'passed', 'linux_cleanup_confirmed': True,
                    'development_screen_status': 'unknown'}
        result = host.wait_one([sys.executable, '-B', '-I', '-c', 'pass'],
            deadline=time.monotonic() + 10, cancel=lambda: False, observe=lambda: evidence)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['wsl_exit_observed'])
        self.assertEqual(result['development_screen_status'], 'unknown')
        # The evidence is synthetic; this checks process waiting, not WSL or cleanup.

    def test_timeout_requests_cancel_without_terminating_process(self):
        clock = FakeClock()
        proc = SimpleNamespace(poll=Mock(return_value=None), kill=Mock(), terminate=Mock())
        spawn = Mock(return_value=proc)
        cancel = Mock(return_value=True)
        observe = Mock()
        result = host.wait_one(['/synthetic/wsl', '--exec', '/synthetic/command'],
            deadline=0.3, grace=0.2, cancel=cancel, observe=observe,
            spawn=spawn, clock=clock.now, sleep=clock.sleep)
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['host_deadline_reached'])
        self.assertTrue(result['cancel_marker_written'])
        self.assertFalse(result['linux_cleanup_confirmed'])
        self.assertFalse(result['wsl_exit_observed'])
        self.assertIs(spawn.call_args.kwargs['shell'], False)
        proc.kill.assert_not_called()
        proc.terminate.assert_not_called()
        observe.assert_not_called()

    def test_spawn_exception_keeps_process_creation_unknown(self):
        clock = FakeClock()
        cancel = Mock(side_effect=[False, False, True])
        result = host.wait_one(['/synthetic/wsl'], deadline=1, cancel=cancel, observe=Mock(),
            grace=0.5, spawn=Mock(side_effect=OSError('synthetic-private')), clock=clock.now, sleep=clock.sleep)
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['wsl_started'])
        self.assertFalse(result['linux_cleanup_confirmed'])
        self.assertTrue(result['cancel_marker_written'])
        self.assertEqual(cancel.call_count, 3)  # Bound start appears after the constructor error.

        cancel = Mock(return_value=False)
        result = host.wait_one(['/synthetic/wsl'], deadline=10, cancel=cancel, observe=Mock(),
            spawn=Mock(side_effect=OSError('synthetic-private')),
            clock=Mock(side_effect=[0, RuntimeError('synthetic-clock')]), sleep=Mock())
        self.assertEqual(result['status'], 'unknown')
        cancel.assert_called_once()

    def test_clock_and_sleep_failures_request_cancel_and_remain_unknown(self):
        for failure in (RuntimeError('synthetic-private'), KeyboardInterrupt()):
            proc = SimpleNamespace(poll=Mock(return_value=None), kill=Mock(), terminate=Mock())
            cancel = Mock(return_value=True)
            clock = Mock(side_effect=[0, failure])
            result = host.wait_one(['/synthetic/wsl'], deadline=10, cancel=cancel, observe=Mock(),
                spawn=lambda *a, **k: proc, clock=clock, sleep=Mock())
            self.assertEqual(result['status'], 'unknown')
            self.assertTrue(result['host_wait_error'])
            self.assertTrue(result['cancel_marker_written'])
            cancel.assert_called_once()
            proc.kill.assert_not_called()
            proc.terminate.assert_not_called()
        proc = SimpleNamespace(poll=Mock(return_value=None), kill=Mock(), terminate=Mock())
        cancel = Mock(return_value=True)
        result = host.wait_one(['/synthetic/wsl'], deadline=10, cancel=cancel, observe=Mock(),
            spawn=lambda *a, **k: proc, clock=lambda: 0, sleep=Mock(side_effect=RuntimeError('synthetic-private')))
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['host_wait_error'])
        cancel.assert_called_once()
        proc.kill.assert_not_called()
        proc.terminate.assert_not_called()

    def test_transient_poll_error_cancels_then_observes_clean_exit_but_fails(self):
        clock = FakeClock()
        proc = SimpleNamespace(poll=Mock(side_effect=[RuntimeError('synthetic-private'), 0]),
                               kill=Mock(), terminate=Mock())
        cancel = Mock(return_value=True)
        evidence = {'state': 'passed', 'linux_cleanup_confirmed': True,
                    'development_screen_status': 'passed'}
        result = host.wait_one(['/synthetic/wsl'], deadline=10, grace=1, cancel=cancel,
            observe=lambda: evidence, spawn=lambda *a, **k: proc, clock=clock.now, sleep=clock.sleep)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['host_wait_error'])
        self.assertTrue(result['linux_cleanup_confirmed'])
        cancel.assert_called_once()
        proc.kill.assert_not_called()
        proc.terminate.assert_not_called()

    def test_late_clean_exit_is_failure_and_missing_evidence_is_unknown(self):
        clock = FakeClock()
        proc = SimpleNamespace(poll=Mock(side_effect=[None, None, None, None, 0]))
        good = {'state': 'passed', 'linux_cleanup_confirmed': True,
                'development_screen_status': 'passed'}
        result = host.wait_one(['/synthetic/wsl'], deadline=0.2, grace=1,
            cancel=lambda: True, observe=lambda: good, spawn=lambda *a, **k: proc,
            clock=clock.now, sleep=clock.sleep)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['linux_cleanup_confirmed'])
        clock = FakeClock()
        proc = SimpleNamespace(poll=Mock(return_value=0))
        result = host.wait_one(['/synthetic/wsl'], deadline=2,
            cancel=lambda: False, observe=lambda: {'state': 'unknown', 'linux_cleanup_confirmed': False},
            spawn=lambda *a, **k: proc, clock=clock.now, sleep=clock.sleep)
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['wsl_exit_zero'])

    def test_keyboard_interrupt_requests_cancel_and_never_reports_pass(self):
        clock = FakeClock()
        proc = SimpleNamespace(poll=Mock(side_effect=[KeyboardInterrupt(), None, 0]))
        cancel = Mock(return_value=True)
        good = {'state': 'passed', 'linux_cleanup_confirmed': True,
                'development_screen_status': 'passed'}
        result = host.wait_one(['/synthetic/wsl'], deadline=10, grace=1, cancel=cancel,
            observe=lambda: good, spawn=lambda *a, **k: proc, clock=clock.now, sleep=clock.sleep)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['host_cancelled'])
        self.assertTrue(result['linux_cleanup_confirmed'])
        cancel.assert_called_once()

    def test_cancel_only_writes_after_nonce_and_contract_match(self):
        root = Path.cwd().resolve() / 'synthetic-game'
        artifacts = root / 'artifacts'
        nonce, commit, token = 'b' * 32, 'c' * 40, 'd' * 32
        prepared = SimpleNamespace(artifacts=artifacts, launch=SimpleNamespace(OUTPUT_NAME='linux-output'),
            host={'training_contract_sha256': 'e' * 64}, entry=SimpleNamespace(_write_new=Mock()))
        start = {'schema': 'task-pair-wsl-supervisor-start-v1', 'attempt': 1,
                 'registration_commit': commit, 'token': token, 'host_nonce': 'f' * 32,
                 'private_contract_sha256': 'e' * 64}
        marker = []
        def read(path, maximum):
            if path.name == 'started.local.json': return encode(start)
            return marker[0]
        def write(path, raw): marker.append(raw)
        prepared.entry._write_new.side_effect = write
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_dir', return_value=True), \
             patch.object(host, '_bounded', side_effect=read), patch.object(os.path, 'lexists', return_value=False):
            self.assertFalse(host.request_cancel(prepared, commit, nonce))
            prepared.entry._write_new.assert_not_called()
            start['host_nonce'] = nonce
            self.assertTrue(host.request_cancel(prepared, commit, nonce))
        written_path = prepared.entry._write_new.call_args.args[0]
        self.assertEqual(written_path, artifacts / 'linux-output' / 'cancel.request')

    def test_linux_evidence_needs_nonce_receipt_cleanup_and_matching_result(self):
        output = Path.cwd().resolve() / 'synthetic-linux-output'
        token, nonce, commit = 'd' * 32, 'b' * 32, 'c' * 40
        start = {'token': token}
        receipt = {'schema': 'task-pair-training-supervisor-receipt-v1', 'token': token, 'passed': True,
                   'completed_updates': 64, 'auxiliary_nonzero_updates': 32,
                   'both_arms_changed': True, 'arms_differ': True}
        result = {'schema': 'task-pair-wsl-supervisor-private-result-v1',
                  'technical_execution_passed': True, 'owned_session_clean': True,
                  'model_files_reloaded': True, 'weights_distinct': True,
                  'development_screen_status': 'unknown',
                  'private_model_sha256': {'baseline': '1' * 64, 'control': '2' * 64, 'auxiliary': '3' * 64},
                  'outer_completion_confirmation': False, 'new_games': False,
                  'independent_competitive_improvement': False}
        files = {'guard.local.json': encode({'schema': 'task-pair-wsl-guard-result-v1', 'gates': guard_result()}),
                 'child-receipt.local.json': encode(receipt), 'supervisor-result.local.json': encode(result)}
        prepared = SimpleNamespace(launch=launch)
        with patch.object(host, 'bound_linux_start', return_value=(output, start)), \
             patch.object(host, '_bounded', side_effect=lambda path, limit: files[path.name]), \
             patch.object(os.path, 'lexists', return_value=False):
            observed = host.observe_linux(prepared, commit, nonce)
            self.assertEqual(observed['state'], 'passed')
            self.assertEqual(observed['development_screen_status'], 'unknown')
            files['child-receipt.local.json'] = encode(receipt | {'token': 'f' * 32})
            self.assertEqual(host.observe_linux(prepared, commit, nonce)['state'], 'unknown')
            files['child-receipt.local.json'] = encode(receipt) * 2
            self.assertEqual(host.observe_linux(prepared, commit, nonce)['state'], 'unknown')
            files['guard.local.json'] = encode({'schema': 'task-pair-wsl-guard-result-v1',
                                                'gates': guard_result(False)})
            failed = host.observe_linux(prepared, commit, nonce)
            self.assertEqual(failed, {'state': 'failed', 'linux_cleanup_confirmed': True})

    def test_one_shot_command_preserves_nonce_and_unknown_result(self):
        root = Path.cwd().resolve() / 'synthetic-game'
        artifacts = root / 'artifacts'
        files = {}
        def write_json(path, value):
            key = str(path)
            if key in files: raise FileExistsError
            files[key] = encode(value)
        fake_entry = SimpleNamespace(OUTPUT_NAME='child-output', _write_json=write_json)
        identity = synthetic_contract()
        registration = {'limits': {'outer_total_seconds': 100}, 'source_sha256': {host.LAUNCH_REL: 'a' * 64}}
        prepared = host.Prepared(SimpleNamespace(OUTPUT_NAME='linux-output'), fake_entry, registration,
            root, artifacts, artifacts / 'host.json', 'f' * 64, identity, artifacts / 'training.json', {}, 0)
        captured = {}
        def wait(command, **kwargs):
            captured.update(command=command, cancel=kwargs['cancel'], observe=kwargs['observe'])
            self.assertTrue(kwargs['cancel']())
            return {'status': 'unknown', 'wsl_started': True, 'wsl_exit_observed': False,
                    'wsl_exit_zero': False, 'host_deadline_reached': True, 'host_cancelled': False,
                    'cancel_marker_written': True, 'linux_cleanup_confirmed': False}
        with patch.object(Path, 'resolve', lambda path, strict=False: path), patch.object(Path, 'mkdir'), \
             patch.object(os.path, 'lexists', return_value=False), \
             patch.object(host, '_system_wsl', return_value=Path.cwd() / 'synthetic-wsl'), \
             patch.object(host, '_bounded', side_effect=lambda path, maximum: files[str(path)]), \
             patch.object(host.secrets, 'token_hex', return_value='b' * 32), \
             patch.object(host, 'request_cancel', return_value=True) as cancel, \
             patch.object(host, 'wait_one', side_effect=wait):
            result = host.run_prepared(prepared, 'c' * 40, clock=lambda: 1, sleep=lambda _: None)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(captured['command'][-1], 'b' * 32)
        self.assertIn(identity['wsl_venv_executable'], captured['command'])
        index = captured['command'].index('-c')
        self.assertEqual(captured['command'][index + 1], host.BOUND_BOOTSTRAP)
        self.assertEqual(captured['command'][index + 3], 'a' * 64)
        self.assertEqual(cancel.call_args.args[-1], 'b' * 32)
        output = artifacts / host.HOST_OUTPUT_NAME
        start = json.loads(files[str(output / 'started.local.json')])
        record = json.loads(files[str(output / 'host-result.local.json')])
        self.assertEqual(start['host_nonce'], 'b' * 32)
        self.assertFalse(record['host_exit_confirmation'])
        self.assertFalse(record['gates']['linux_cleanup_confirmed'])

    def test_base_exception_is_redacted(self):
        args = ['--registration-commit', 'a' * 40, '--game-root', 'synthetic-unopened',
                '--host-contract', 'synthetic-unopened', '--expected-host-contract-sha256', 'b' * 64]
        for kind in (BaseException, GeneratorExit):
            stream = io.StringIO()
            with self.subTest(kind=kind.__name__), patch.object(host.os, 'name', 'nt'), \
                 patch.object(host, 'prepare', side_effect=kind('synthetic-private-marker')), redirect_stdout(stream):
                status = host.main(args)
            self.assertEqual(status, 2)
            self.assertNotIn('synthetic-private-marker', stream.getvalue())
            self.assertEqual(json.loads(stream.getvalue())['error_type'], 'OtherError')


if __name__ == '__main__':
    unittest.main()
