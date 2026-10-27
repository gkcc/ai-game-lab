"""Focused in-process checks; no process launch, model, engine or raw game IO."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from . import analysis, analysis_entry as entry


def manifest():
    games, pairs = [], []
    for seed in (101, 202):
        for opponent in ('market', 'round2'):
            for side in (0, 1):
                ids = {}
                for arm in analysis.ARMS:
                    key = f'{seed}-{opponent}-{side}-{arm}'
                    ids[arm] = key
                    games.append(dict(key=key, id=key, seed=seed, opponent=opponent,
                        side=side, arm=arm, split='development', expected={},
                        replay=dict(path=f'games/{key}.json.gz', size=7, sha256='a' * 64),
                        trace=dict(path=f'traces/{key}.jsonl.gz', size=8, sha256='b' * 64)))
                pairs.append(dict(seed=seed, opponent=opponent, side=side, game_ids=ids,
                                  first_candidate_baseline_request_difference=None))
    return dict(schema='task-action-observation-input-manifest-v1', games=games, pairs=pairs,
                ledger_sha256='f' * 64, audit_sha256='f' * 64)


def config():
    return dict(validation_seeds=[101, 202], opponents=['market', 'round2'],
        analysis_limits=dict(work_seconds=420, process_seconds=450, target_seconds=360,
        cpu_seconds=300, address_space_bytes=2**31, output_bytes=128 * 2**20,
        parent_receipt_bytes=65536, compressed_bytes=2**30, decoded_bytes=12 * 2**30))


@contextmanager
def fake_experiment(inputs):
    writes = {}
    module = ModuleType('learning.task_action_observation_v1.experiment')
    module.read = lambda path: deepcopy(writes.get(Path(path).name, inputs.get(Path(path).name)))
    def write(path, value):
        name = Path(path).name
        if name in writes:
            raise FileExistsError('Exclusive receipt already exists')
        writes[name] = deepcopy(value)
    module.write_new = write
    module.sha = lambda path: 'f' * 64
    module.verify = module.ready = lambda out: config()
    module.deadline_check = lambda: None
    module.remaining_seconds = lambda: 3600
    with patch.dict(sys.modules, {module.__name__: module}):
        yield writes


@contextmanager
def fake_budget(*args, **kwargs):
    yield SimpleNamespace(used_bytes=0)


class ObservationAnalysisTests(unittest.TestCase):
    def setUp(self):
        # Deliberately nonexistent: all raw readers and receipt writes are fakes.
        self.out = Path(__file__).resolve().parent / 'not-created-synthetic-test'

    def test_new_shape_and_cross_game_identity_are_fixed(self):
        value = manifest()
        indexed, raw = analysis.validate_manifest(self.out, value)
        self.assertEqual((len(indexed), len(raw)), (24, 48))
        for mutate in (
                lambda v: v['games'].extend(deepcopy(v['games']) * 3),
                lambda v: v['pairs'].__setitem__(1, deepcopy(v['pairs'][0])),
                lambda v: v['pairs'][0]['game_ids'].__setitem__('baseline', 'different-game'),
                lambda v: v['games'][0].__setitem__('split', 'holdout'),
                lambda v: v['games'][1].__setitem__('trace', deepcopy(v['games'][0]['trace']))):
            changed = deepcopy(value)
            mutate(changed)
            with self.subTest(changed=changed['games'][0]['split']), self.assertRaises(ValueError):
                analysis.validate_manifest(self.out, changed)

    def test_raw_paths_cannot_read_historical_or_other_output(self):
        for relative in ('../previous/games/a.gz', '/games/a.gz', 'games/../a.gz',
                         'games//a.gz', 'traces/a.gz', chr(68) + ':/synthetic/games/a.gz',
                         'games\\a.gz', 'games/../../previous/a.gz'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                analysis.raw_path(self.out, dict(path=relative, size=1, sha256='a' * 64), 'replay')

    def test_run_uses_only_48_registered_files_and_reports_16_comparisons(self):
        value, seen = manifest(), []
        compact = dict(full=['same'] * 720, physical=['same'] * 720,
            requests=[['same', 'same']] * 719, banks=[[10, 10]] * 720,
            days=[0] * 719, positions=[[[0, 0]]] * 719)
        def completed(path, identity, meter):
            self.assertNotIn(identity['path'], seen)
            seen.append(identity['path'])
            meter.opened.add(str(path))
            meter.compressed += identity['size']
            meter.decoded += 1
            meter.receipts.append(dict(path=identity['path'], compressed_bytes=identity['size'],
                                       decoded_bytes=1, sha256=identity['sha256'], decode_passes=1))
        def replay(path, identity, meter):
            completed(path, identity, meter)
            return compact
        def trace(path, identity, meter):
            for step in range(719):
                yield dict(step=step, records=[], load_errors=0, inference_errors=0, fallbacks=0)
            completed(path, identity, meter)
        metadata = {'input-manifest.json': value, 'registration.json': config(),
                    'campaign-result.json': dict(passed=True, input_manifest_sha256='f' * 64)}
        with fake_experiment(metadata) as writes, \
                patch.object(analysis, 'read_json', replay), \
                patch.object(analysis, 'compact_replay', lambda replay, game: replay), \
                patch.object(analysis, 'trace_lines', trace):
            result = analysis.run(self.out)
        self.assertTrue(result['complete'])
        self.assertEqual((result['input_games'], result['triads'], result['pairs']), (24, 8, 16))
        self.assertEqual(len(seen), 48)
        self.assertEqual(result['raw_reads']['files_completed'], 48)
        self.assertEqual(len(writes['game-descriptors.json']), 24)
        self.assertEqual(result['observation_denominators']['completion_unknown_pairs'], 16)
        self.assertEqual(result['primary_pairs'], 8)
        self.assertNotIn('screen', result)
        self.assertFalse(result['screen_applied'])
        self.assertFalse(result['promotion_authorized'])
        self.assertEqual(result['neural_forwards'], 0)
        changed = deepcopy(metadata)
        for collection in ('games', 'pairs'):
            for row in changed['input-manifest.json'][collection]:
                if row['seed'] == 101:
                    row['seed'] = 303
        with fake_experiment(changed) as writes, patch.object(analysis, 'read_json') as raw_reader:
            with self.assertRaisesRegex(ValueError, 'frozen registration'):
                analysis.run(self.out)
            raw_reader.assert_not_called()
            self.assertEqual(writes['analysis-failure.local.json']['files_opened'], 0)

    def test_guard_reuses_session_supervisor_and_never_claims_unknown_cleanup(self):
        from learning.task_dagger_once_v1 import gpu_guard
        from learning.task_anchor_once_v1 import storage
        clean = dict(reason='exited', child_exit_code=0, elapsed_seconds=.1,
                     completed=True, **dict.fromkeys(entry.CLEANUP_FIELDS, True))
        for error in (None, RuntimeError('private failure detail')):
            with self.subTest(failure=error is not None), \
                    fake_experiment({'analysis-worker-result.json': {'passed': True}}) as writes, \
                    patch.object(entry, 'location', return_value=(self.out, self.out)), \
                    patch.object(entry.sys, 'platform', 'linux'), \
                    patch.object(Path, 'exists', return_value=False), \
                    patch.object(storage, 'owned_writes', fake_budget), \
                    patch.object(gpu_guard, 'supervise', return_value=clean, side_effect=error) as supervise:
                if error:
                    with self.assertRaises(RuntimeError):
                        entry.guard('relative-output')
                    result = writes['guard-result.json']
                    self.assertFalse(result['completed'])
                    self.assertEqual(result['cleanup_state'], 'unknown')
                    self.assertTrue(all(result[name] is None for name in entry.CLEANUP_FIELDS))
                    self.assertNotIn('private failure detail', str(result))
                else:
                    result = entry.guard('relative-output')
                    self.assertTrue(result['completed'])
                    self.assertEqual(supervise.call_args.args[1], 420)
                    self.assertIn(entry.MODULE, supervise.call_args.args[0])
                self.assertFalse(writes['analysis-started.json']['retry_allowed'])

    def test_worker_keeps_spawn_ipc_outside_artifact_guard(self):
        from learning.task_anchor_spawn_repair_v1 import launcher
        from learning.task_anchor_once_v1 import storage
        active, calls = [], []
        @contextmanager
        def budget(*args, **kwargs):
            active.append(True)
            try:
                yield SimpleNamespace(used_bytes=0)
            finally:
                active.pop()
        def launch(*args, **kwargs):
            self.assertFalse(active)
            calls.append((args, kwargs))
            return dict(passed=True, target_handle_closed=True, child_exit_code=0,
                        parent_receipt_refused=False, parent_receipt_bytes_total=100)
        data = {'analysis-started.json': dict(freeze_sha256='f' * 64, fixture=False),
                'analysis-result.json': dict(complete=True), 'analysis-target-result.json': dict(passed=True)}
        with fake_experiment(data) as writes, \
                patch.object(entry, 'location', return_value=(self.out, self.out)), \
                patch.object(storage, 'owned_writes', budget), patch.object(launcher, 'launch', launch):
            entry.worker('relative-output')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {'seconds': 360, 'receipt_limit': 65536})
        self.assertTrue(writes['analysis-worker-result.json']['passed'])

    def test_audit_and_fixture_have_no_model_or_process_escape(self):
        for event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.posix_spawn', 'os.exec'):
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                entry.worker_audit(event, ())
        for module in ('torch', 'numpy.linalg', 'jax', 'tensorflow', 'kaggle_environments',
                       'bots.foo', 'models', 'learning.task_dagger_once_v1.training'):
            with self.subTest(module=module), self.assertRaises(RuntimeError):
                entry.worker_audit('import', (module,))
        for module in ('decision.api', 'learning.task_action_impact_diagnostic_v1.analysis',
                       'learning.task_bc_error_diagnostics_v1.shadow'):
            entry.worker_audit('import', (module,))
        self.assertEqual(entry.limits(config(), True)['target_seconds'], 30)
        self.assertEqual(entry.names(True)['started'], 'fixture-started.json')
        self.assertNotEqual(entry.names(True)['guard'], entry.names(False)['guard'])


if __name__ == '__main__':
    unittest.main()
