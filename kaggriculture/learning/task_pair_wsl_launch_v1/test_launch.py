"""CPU synthetic launcher regressions; no real data, GPU or temp files."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from learning.task_pair_wsl_training_v1 import entry
from learning.task_pair_wsl_training_v1.test_entry import synthetic_metrics
from learning.task_pair_wsl_launch_v1 import launch


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def guard_result(passed=True):
    result = {name: True for name in launch._GUARD_RESULT_KEYS}
    result.update(passed=passed, work_deadline_triggered=False, cancelled=False)
    if not passed:
        result['receipt_sink_accepted'] = False
    return result


BOOTSTRAP_PROBE = r'''
import hashlib, importlib.metadata, json, pathlib, sys, types
repo = pathlib.Path.cwd().resolve()
entry_path = repo / 'kaggriculture/learning/task_pair_wsl_training_v1/entry.py'
entry = types.ModuleType('synthetic_entry_snapshot')
entry.__file__ = str(entry_path)
exec(compile(entry_path.read_bytes(), '<synthetic-entry>', 'exec'), entry.__dict__)
prefix = pathlib.Path(sys.prefix).resolve()
if sys.platform == 'win32':
    # This CPU-only child process removes a venv-root search entry which the
    # WSL production contract intentionally rejects; no environment changes.
    sys.path[:] = [p for p in sys.path if pathlib.Path(p).resolve() != prefix]
sources = {}
for base in ('kaggriculture/learning', 'kaggriculture/decision'):
    for path in (repo / base).rglob('*.py'):
        if path.is_file() and path.stat().st_size <= entry.MAX_SOURCE:
            sources[path.relative_to(repo).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
for rel in (entry.CERT_REL, entry.EARLY_REL, entry.ORIGINAL_REL, entry.CONDITIONAL_REL, entry.CACHE_REL,
            'review/evidence/task-pair-data-preflight-v1-20261030/registration.json',
            'review/evidence/task-pair-data-preflight-v1-result-20261030/result.json',
            'review/evidence/task-pair-mask-recheck-v1-20261030/registration.json',
            'review/evidence/task-pair-mask-recheck-v1-result-20261030/result.json'):
    sources[rel] = hashlib.sha256(entry._bounded_file(repo / rel, entry.MAX_SOURCE)).hexdigest()
assert 'numpy' not in sys.modules
binding = entry.prebind(sources)
assert 'numpy' not in sys.modules
search = [str(pathlib.Path(p).resolve()) for p in sys.path]
roots = [p for p in search if pathlib.Path(p).is_relative_to(prefix)
         and pathlib.Path(p) != prefix and pathlib.Path(p).is_dir()]
private = {'runtime_identity': {'venv_executable':sys.executable,
    'interpreter':str(pathlib.Path(sys.executable).resolve()), 'venv_prefix':str(prefix),
    'base_prefix':str(pathlib.Path(sys.base_prefix).resolve()), 'search_path':search, 'allowed_origins':roots}}
# Current-process versions only exercise import semantics; they are not a
# precommitted production registration, especially on native Windows.
entry.RUNTIME = {'platform':sys.platform, 'python':'.'.join(map(str,sys.version_info[:3])),
                 **{n:importlib.metadata.version(n) for n in ('numpy','jax','jaxlib','optax')}}
entry._runtime_pre(private)
assert 'numpy' not in sys.modules
entry.finish_bootstrap(binding, sources)
entry._runtime_after(private)
assert sys.modules['learning.task_pair_wsl_training_v1.entry'].CURRENT_STAGE == 'arguments'
assert sys.modules['learning.task_pair_wsl_launch_v1.launch'].STAGE == 'arguments'
print(json.dumps({'passed':True, 'scope':'synthetic_sources_cpu'}))
'''


class LauncherTests(unittest.TestCase):
    def test_actual_two_phase_bootstrap_without_executing_canonical_mains(self):
        env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='-1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONWARNINGS='ignore::DeprecationWarning')
        result = subprocess.run([sys.executable, '-B', '-I', '-c', BOOTSTRAP_PROBE],
            cwd=launch.REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, 'Source-only bootstrap probe failed')
        self.assertEqual(json.loads(result.stdout), {'passed': True, 'scope': 'synthetic_sources_cpu'})

    def test_missing_registration_stops_before_private_or_project_binding(self):
        args = ['--registration-commit', 'a' * 40, '--game-root', 'synthetic-unopened',
                '--private-contract', 'synthetic-unopened',
                '--expected-private-contract-sha256', 'b' * 64, '--host-nonce', 'd' * 32]
        output = io.StringIO()
        with patch.object(launch, 'authenticated_entry', side_effect=ValueError('private-marker')), \
             patch.object(launch, 'run_prepared') as run, redirect_stdout(output):
            code = launch.main(args)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue()),
                         {'passed': False, 'stage': 'registration', 'error_type': 'ValueError'})
        self.assertNotIn('private-marker', output.getvalue())
        run.assert_not_called()

    def test_base_exception_and_generator_exit_never_expose_diagnostics(self):
        args = ['--registration-commit', 'a' * 40, '--game-root', 'synthetic-unopened',
                '--private-contract', 'synthetic-unopened',
                '--expected-private-contract-sha256', 'b' * 64, '--host-nonce', 'd' * 32]
        for kind in (BaseException, GeneratorExit):
            output = io.StringIO()
            with self.subTest(kind=kind.__name__), \
                 patch.object(launch, 'authenticated_entry', side_effect=kind('synthetic-private-marker')), \
                 redirect_stdout(output):
                code = launch.main(args)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue()),
                {'passed': False, 'stage': 'registration', 'error_type': 'OtherError'})
            self.assertNotIn('synthetic-private-marker', output.getvalue())

    def test_source_blob_mismatch_is_rejected_before_compilation(self):
        commit = 'a' * 40
        source = b'raise RuntimeError("must not execute")\n'
        registration = encode({'source_sha256': {launch.ENTRY_REL: hashlib.sha256(source).hexdigest()}})
        def blob(requested, relative):
            return registration if relative == launch.REG_REL else b'different committed source'
        def read(path, maximum):
            return registration if path == launch.REPO / launch.REG_REL else source
        with patch.object(launch, '_head', return_value=commit), \
             patch.object(launch, '_blob', side_effect=blob), \
             patch.object(launch, '_bounded_file', side_effect=read), \
             patch.object(launch.types, 'ModuleType') as module, self.assertRaises(ValueError):
            launch.authenticated_entry(commit)
        module.assert_not_called()

    def test_prebinding_and_registration_precede_any_private_access(self):
        events = []
        def phase(name, value=None):
            def call(*args, **kwargs):
                events.append(name)
                return value
            return call
        root = Path.cwd().resolve() / 'synthetic-game'
        artifacts = root / 'artifacts'
        private_path = artifacts / 'contract.local.json'
        registration = {'limits': {'outer_total_seconds': 20},
                        'source_sha256': {launch.GATE_REL: 'a' * 64}}
        private = {'synthetic': True}
        modules = object()
        fake_entry = SimpleNamespace(MAX_PRIVATE=65536,
            prebind=phase('prebind', object()), _private_root=phase('private_root', (root, artifacts)),
            _private_file=phase('read_contract', b'contract'),
            parse_private_contract=phase('verify_contract', private), _runtime_pre=phase('runtime_pre'),
            finish_bootstrap=phase('finish', modules), _runtime_after=phase('runtime_after'))
        guard = SimpleNamespace(GATE_SHA256='a' * 64)
        with patch.object(launch, 'authenticated_entry', side_effect=phase('registration', (fake_entry, registration))), \
             patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.dict(sys.modules, {launch.SUPERVISOR_NAME: object(), launch.GUARD_NAME: guard}):
            prepared = launch.prepare('b' * 40, root, private_path, 'c' * 64, started=0, clock=lambda: 1)
        self.assertEqual(repr(prepared), '<PrivateRegisteredTrainingRun>')
        self.assertEqual(events, ['registration', 'prebind', 'private_root', 'read_contract',
                                  'verify_contract', 'runtime_pre', 'finish', 'runtime_after'])

        fake_entry.parse_private_contract = Mock(side_effect=ValueError('incorrect digest'))
        fake_entry.finish_bootstrap = Mock()
        with patch.object(launch, 'authenticated_entry', return_value=(fake_entry, registration)), \
             self.assertRaises(ValueError):
            launch.prepare('b' * 40, root, private_path, 'c' * 64, started=0, clock=lambda: 1)
        fake_entry.finish_bootstrap.assert_not_called()

    def test_supervision_result_never_treats_missing_or_false_gates_as_pass(self):
        self.assertTrue(launch.validate_guard_result(guard_result()))
        self.assertFalse(launch.validate_guard_result(guard_result(False)))
        false_success = guard_result()
        false_success['used_pidfd'] = False
        with self.assertRaises(ValueError):
            launch.validate_guard_result(false_success)
        inconsistent_cleanup = guard_result(False)
        inconsistent_cleanup['used_pidfd'] = False
        with self.assertRaises(ValueError):
            launch.validate_guard_result(inconsistent_cleanup)
        missing = guard_result()
        del missing['owned_session_empty_before_reap']
        with self.assertRaises(ValueError):
            launch.validate_guard_result(missing)

    @unittest.skipUnless(os.name == 'posix', 'Linux-only receipt pipe')
    def test_real_supervisor_api_delivers_exact_receipt_from_injected_guard(self):
        from learning.task_pair_supervision_v1 import supervisor
        from learning.task_pair_pidfd_guard_v1 import guard
        token = 'a' * 32
        contract = supervisor.Contract(expected_updates=64, minimum_auxiliary_nonzero=32,
            work_seconds=100, guard_with_cleanup_seconds=120, term_seconds=2, cleanup_seconds=5,
            gate_seconds=5, expected_gate_sha256=guard.GATE_SHA256)
        raw = encode({'schema': supervisor.RECEIPT_SCHEMA, 'token': token, 'passed': True,
            'completed_updates': 64, 'auxiliary_nonzero_updates': 32,
            'both_arms_changed': True, 'arms_differ': True})
        def fake_guard(command, work, cancel, **kwargs):
            os.write(1, raw)
            return {'reason': 'exited', 'child_exit_code': 0, 'elapsed_seconds': 1.0,
                'owned_session_empty_before_reap': True, 'direct_child_reaped': True,
                'used_pidfd': True, 'completed': True}
        stored = []
        with patch.object(guard, 'supervise', side_effect=fake_guard):
            result = supervisor.supervise_command(['/synthetic/python', '--receipt-token', token],
                token, contract, cancelled=lambda: False, receipt_sink=lambda data: stored.append(data) or True)
        self.assertTrue(launch.validate_guard_result(result))
        self.assertEqual(stored, [raw])
        # Guard process execution and cleanup are injected, not independently proven here.

    def test_existing_output_never_gets_modified(self):
        root = Path.cwd().resolve() / 'synthetic-game'
        writer = Mock()
        fake_entry = SimpleNamespace(OUTPUT_NAME='child-output', _write_json=writer)
        prepared = launch.Prepared(fake_entry, {'limits': {}, 'source_sha256': {}}, {}, root,
            root / 'artifacts', root / 'contract', 'a' * 64, object(), object(), object(), 0)
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(os.path, 'lexists', return_value=False), \
             patch.object(Path, 'mkdir', side_effect=FileExistsError) as mkdir, self.assertRaises(FileExistsError):
            launch.run_prepared(prepared, 'b' * 40, host_nonce='d' * 32, clock=lambda: 0)
        mkdir.assert_called_once()
        writer.assert_not_called()
        self.assertFalse(hasattr(prepared, 'owned_output'))

    def test_one_shot_receipt_and_unknown_screen_remain_separate(self):
        root = Path.cwd().resolve() / 'synthetic-game'
        artifacts = root / 'artifacts'
        files, calls, now, expire_on_result = {}, [], [1], [False]
        registration = {'limits': {'outer_total_seconds': 1000, 'work_seconds': 100,
            'guard_with_cleanup_seconds': 120, 'term_seconds': 2, 'cleanup_seconds': 5, 'gate_seconds': 5},
            'source_sha256': {launch.GATE_REL: 'a' * 64}}
        private = {'runtime_identity': 'synthetic'}
        def write(path, raw):
            key = str(path)
            if key in files: raise FileExistsError
            calls.append('write-' + path.name)
            files[key] = raw
            if path.name == 'supervisor-result.local.json' and expire_on_result[0]:
                now[0] = 1000
        def write_json(path, value):
            write(path, encode(value))
        fake_entry = SimpleNamespace(OUTPUT_NAME='child-output', MAX_PRIVATE=65536, _json=entry._json,
            _write_new=write, _write_json=write_json,
            child_interpreter=lambda _: 'synthetic-venv-python',
            registration_gate=lambda _: (b'', registration), verify_registered_sources=Mock(),
            _private_file=lambda *args: b'private-contract', parse_private_contract=lambda *args: private,
            _runtime_after=Mock())
        def inspect(raw, token, contract):
            data = json.loads(raw)
            return {'valid': data['token'] == token and data['completed_updates'] == 64
                    and data['auxiliary_nonzero_updates'] >= 32}
        def supervise(command, token, contract, *, cancelled, receipt_sink):
            calls.append('supervise')
            self.assertEqual(command[0], 'synthetic-venv-python')
            self.assertIn('write-started.local.json', calls)
            self.assertFalse(cancelled())
            raw = encode({'token': token, 'completed_updates': 64, 'auxiliary_nonzero_updates': 32})
            self.assertIs(receipt_sink(raw), True)
            return guard_result()
        fake_supervisor = SimpleNamespace(Contract=lambda **kwargs: SimpleNamespace(**kwargs),
            inspect_receipt=inspect, supervise_command=Mock(side_effect=supervise))
        extra = SimpleNamespace(TRAIN_MODULES=frozenset(), verify_loaded_origins=Mock())
        modules = SimpleNamespace(extra=extra, smoke=object(), root=object())
        prepared = launch.Prepared(fake_entry, registration, private, root, artifacts,
            artifacts / 'contract.local.json', 'b' * 64, modules, fake_supervisor, object(), 0)
        outputs = {'model_files_reloaded': True, 'weights_distinct': True,
                   'development_screen_status': 'unknown', 'model_sha256': {'synthetic': 'x'},
                   'child_reported_updates': 64, 'child_reported_auxiliary_nonzero': 32}
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(Path, 'mkdir'), patch.object(os.path, 'lexists', return_value=False), \
             patch.object(launch, '_bounded_file', side_effect=lambda path, maximum: files[str(path)]), \
             patch.object(launch, 'verify_private_outputs', return_value=outputs) as verify:
            result = launch.run_prepared(prepared, 'c' * 40, host_nonce='d' * 32, clock=lambda: now[0])
        self.assertEqual(result, {'passed': True})
        output = artifacts / launch.OUTPUT_NAME
        saved = json.loads(files[str(output / 'supervisor-result.local.json')])
        self.assertTrue(saved['technical_execution_passed'])
        self.assertEqual(saved['development_screen_status'], 'unknown')
        self.assertFalse(saved['independent_competitive_improvement'])
        self.assertIn(str(output / 'child-receipt.local.json'), files)
        self.assertEqual(json.loads(files[str(output / 'started.local.json')])['host_nonce'], 'd' * 32)
        self.assertLess(calls.index('write-started.local.json'), calls.index('supervise'))
        verify.assert_called_once()

        # A failed owned-session gate must not load checkpoints or retry.
        files.clear()
        calls.clear()
        fake_supervisor.supervise_command.reset_mock()
        fake_supervisor.supervise_command.side_effect = lambda *args, **kwargs: guard_result(False)
        retry = launch.Prepared(fake_entry, registration, private, root, artifacts,
            artifacts / 'contract.local.json', 'b' * 64, modules, fake_supervisor, object(), 0)
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(Path, 'mkdir'), patch.object(os.path, 'lexists', return_value=False), \
             patch.object(launch, '_bounded_file', side_effect=lambda path, maximum: files[str(path)]), \
             patch.object(launch, 'verify_private_outputs') as rejected, self.assertRaises(ValueError):
            launch.run_prepared(retry, 'c' * 40, host_nonce='d' * 32, clock=lambda: now[0])
        rejected.assert_not_called()
        fake_supervisor.supervise_command.assert_called_once()
        self.assertFalse(json.loads(files[str(output / 'guard.local.json')])['gates']['passed'])

        # A deadline exceeded during the final private write cannot be reported
        # as an accepted outer completion, even if an intermediate file exists.
        files.clear()
        calls.clear()
        now[0], expire_on_result[0] = 1, True
        fake_supervisor.supervise_command.side_effect = supervise
        overtime = launch.Prepared(fake_entry, registration, private, root, artifacts,
            artifacts / 'contract.local.json', 'b' * 64, modules, fake_supervisor, object(), 0)
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(Path, 'mkdir'), patch.object(os.path, 'lexists', return_value=False), \
             patch.object(launch, '_bounded_file', side_effect=lambda path, maximum: files[str(path)]), \
             patch.object(launch, 'verify_private_outputs', return_value=outputs), self.assertRaises(TimeoutError):
            launch.run_prepared(overtime, 'c' * 40, host_nonce='d' * 32, clock=lambda: now[0])
        incomplete = json.loads(files[str(output / 'supervisor-result.local.json')])
        self.assertIs(incomplete['outer_completion_confirmation'], False)

        # A cancellation observed before supervision prevents any child call.
        files.clear()
        calls.clear()
        now[0], expire_on_result[0] = 1, False
        fake_supervisor.supervise_command.reset_mock()
        cancelled = launch.Prepared(fake_entry, registration, private, root, artifacts,
            artifacts / 'contract.local.json', 'b' * 64, modules, fake_supervisor, object(), 0)
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(Path, 'mkdir'), \
             patch.object(os.path, 'lexists', side_effect=lambda path: Path(path).name == 'cancel.request'), \
             patch.object(launch, '_bounded_file', side_effect=lambda path, maximum: files[str(path)]), \
             patch.object(launch, 'verify_private_outputs') as unvisited, self.assertRaises(ValueError):
            launch.run_prepared(cancelled, 'c' * 40, host_nonce='d' * 32, clock=lambda: now[0])
        fake_supervisor.supervise_command.assert_not_called()
        unvisited.assert_not_called()

    def test_output_checks_use_bounded_snapshots_and_enforce_screen_consistency(self):
        root = Path.cwd().resolve() / 'synthetic-game'
        commit = 'c' * 40
        schema = 'd' * 64
        raw_models = {name: name.encode() for name in ('baseline', 'control', 'auxiliary')}
        metrics = synthetic_metrics()
        screen = entry.assess_development(metrics)
        child = {'schema': 'task-pair-wsl-child-result-v1', 'completed_updates': 64,
                 'auxiliary_nonzero_updates': 32, 'development_screen_status': screen['status'],
                 'supervisor_confirmation': False}
        files = {'manifest.json': encode({'numeric_feature_schema_sha256': schema}),
                 'selected-export.json': encode({'checkpoint_sha256': hashlib.sha256(raw_models['baseline']).hexdigest()}),
                 'evaluation.local.json': encode({'schema': 'task-pair-wsl-private-evaluation-v1',
                     'metrics': metrics, 'development_screen': screen, 'new_games': False,
                     'independent_game_evaluation': False}),
                 'child-result.local.json': encode(child)}
        files.update({name + '.npz': raw for name, raw in raw_models.items()})
        values = {'baseline': 0.0, 'control': 1.0, 'auxiliary': 2.0}
        def snapshot(raw, **kwargs):
            name = raw.decode()
            return SimpleNamespace(params={'p': np.asarray([values[name]], np.float32)},
                                   mean=np.zeros(1, np.float32), scale=np.ones(1, np.float32),
                                   schema_json='{}')
        bad_provenance = [False]
        def model(path, digest):
            name = path.stem
            item = snapshot(raw_models[name])
            metadata = {}
            if name != 'baseline':
                metadata['pair_training'] = {'schema': 'task-pair-training-provenance-v1', 'arm': name,
                    'training_source': 'original_teacher_train', 'initialization': 'frozen_early_task_head',
                    'registration_commit': commit, 'completed_updates': 63 if bad_provenance[0] else 64,
                    'worker_coefficient': 0.0 if name == 'control' else 0.25}
            return SimpleNamespace(params={'p': item.params['p'].astype(np.float64)},
                mean=item.mean.astype(np.float64), scale=item.scale.astype(np.float64), metadata=metadata)
        constructor = Mock(side_effect=model)
        modules = SimpleNamespace(np=np,
            snapshot=SimpleNamespace(MAX_ARCHIVE_BYTES=128, PARAMS=('p',), load_checkpoint=snapshot),
            model=SimpleNamespace(TaskNetwork=constructor))
        def read(path, maximum):
            value = files[path.name]
            self.assertLessEqual(len(value), maximum)
            return value
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_dir', return_value=True), \
             patch.object(launch, '_bounded_file', side_effect=read):
            checked = launch.verify_private_outputs(entry, modules, root, commit, check=lambda: None)
            self.assertTrue(checked['model_files_reloaded'])
            self.assertTrue(checked['weights_distinct'])
            self.assertEqual(checked['development_screen_status'], 'passed')
            self.assertEqual(constructor.call_count, 3)  # Simulated constructor, not disk acceptance.
            bad_provenance[0] = True
            with self.assertRaises(ValueError):
                launch.verify_private_outputs(entry, modules, root, commit, check=lambda: None)
            bad_provenance[0] = False
            child['development_screen_status'] = 'failed'
            files['child-result.local.json'] = encode(child)
            with self.assertRaises(ValueError):
                launch.verify_private_outputs(entry, modules, root, commit, check=lambda: None)


if __name__ == '__main__':
    unittest.main()
