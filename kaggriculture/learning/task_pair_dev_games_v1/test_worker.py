"""Synthetic worker binding and output regressions; no real game or model."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

GAME_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GAME_ROOT))
from learning.task_pair_dev_games_v1 import worker
from learning.task_pair_dev_games_v1.test_panel import synthetic_inputs
from learning.task_pair_dev_games_v1.panel import Panel


FAKE_SOURCE = b'''\
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
observed = []
def verify(stage):
    raise AssertionError('old verification was not rebound')
def task_identity(stage, plan, cfg):
    raise AssertionError('old identity was not rebound')
def read(path):
    raise AssertionError('old job reader was not rebound')
def _make_stable():
    # The real Worker imports this function from a separate audit module.
    def stable(record):
        if isinstance(record, dict):
            return {k: stable(v) for k, v in record.items() if k != 'inference_seconds'}
        if isinstance(record, list):
            return [stable(v) for v in record]
        return record
    return stable
stable = _make_stable()
def audit(*args, **kwargs):
    raise AssertionError('not a real audit')
def reexecute(*args, **kwargs):
    raise AssertionError('not a real replay')
def observation(*args, **kwargs):
    raise AssertionError('not a real observation')
def owned_writes(*args, **kwargs):
    raise AssertionError('not a real output budget')
def write_new(path, value):
    observed.append((path, value))
def wrapper(stage, plan, cfg):
    path, digest = task_identity(stage, plan, cfg)
    observed.append((path, digest, cfg['crop_checkpoint'], cfg['crop_checkpoint_sha256']))
    return stage / 'agents' / (plan['game'] + '.py')
def job(stage, key):
    cfg = verify(stage)
    plan = read(stage / 'job-inputs' / (key + '.json'))
    path, digest = task_identity(stage, plan, cfg)
    row = dict(plan, states=720, action_turns=719, passed=True)
    if task_identity(stage, row, cfg) != (path, digest):
        raise AssertionError('enriched audit row changed identity')
    observed.append((path, digest, cfg['crop_checkpoint'], cfg['crop_checkpoint_sha256']))
'''

FULL_FAKE_SOURCE = b'''\
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def verify(stage):
    raise AssertionError('old verification was not rebound')
def task_identity(stage, plan, cfg):
    raise AssertionError('old identity was not rebound')
def read(path):
    raise AssertionError('old job reader was not rebound')
def _make_stable():
    def stable(record):
        if isinstance(record, dict):
            return {k: stable(v) for k, v in record.items() if k != 'inference_seconds'}
        if isinstance(record, list):
            return [stable(v) for v in record]
        return record
    return stable
stable = _make_stable()
def wrapper(stage, plan, cfg):
    return stage / 'agents' / (plan['game'] + '.py')
def observation(frame, side):
    return frame[side]['observation']
def owned_writes(*args, **kwargs):
    raise AssertionError('fake run must not write')
def write_new(path, value):
    raise AssertionError('fake run must not write')
def reexecute(replay, *, trace, label_sink):
    assert trace is True
    for player in (0, 1):
        label_sink(dict(step=0, player=player, phase='unit', unit=0, status='pass',
                        requested=['PASS'], effective=['PASS'], position=[0, 0],
                        before={'money': 100}, after={'money': 100}))
    return dict(passed=True, official_transitions_reexecuted=719,
                both_public_and_private_states_match_each_step=True,
                inventory_order_checked_each_step=True,
                execution=dict(label_version=2, counts={'unit:pass': 2}), final_rewards=[100, 100])
def audit(stage, row, cfg):
    farms = [{'money': 100}, {'money': 100}]
    def frame(step, status):
        return [dict(observation=dict(step=step, farms=farms,
                                      private=dict(seeds={}, shed={})), status=status)
                for _ in (0, 1)]
    replay = {'steps': [frame(0, 'ACTIVE')] * 719 + [frame(719, 'DONE')]}
    seen = []
    official = reexecute(replay, trace=True, label_sink=seen.append)
    assert len(seen) == 2
    return dict(passed=True, official=official)
def job(stage, key):
    cfg = verify(stage)
    plan = read(stage / 'job-inputs' / (key + '.json'))
    task_identity(stage, plan, cfg)
    row = dict(plan, trace_sha256='7'*64, replay_gzip_sha256='8'*64)
    assert audit(stage, row, cfg)['passed']
'''


class WorkerTests(unittest.TestCase):
    def setUp(self):
        data = synthetic_inputs()
        data['worker_sha'] = hashlib.sha256(FAKE_SOURCE).hexdigest()
        self.panel = Panel.from_private_contract(**data)
        self.stage = GAME_ROOT / 'artifacts' / 'synthetic-not-created'
        self.plan = self.panel.plans()[0]
        self.job = dict(self.plan, bot_sha256='9' * 64, rival_sha256='e' * 64,
                        task_checkpoint_sha256='a' * 64)

    def test_snapshot_verification_happens_before_execution(self):
        executable = b"raise AssertionError('must not execute')"
        with self.assertRaises(ValueError):
            worker._load_snapshot(executable, '0' * 64, GAME_ROOT, lambda: True)
        with self.assertRaises(ValueError):
            worker._load_snapshot(FAKE_SOURCE, self.panel.worker_sha, GAME_ROOT, lambda: False)

    def test_only_new_module_is_bound_and_restored_on_base_exception(self):
        canonical = types.SimpleNamespace(verify=object(), task_identity=object(), read=object(), stable=object(), write_new=object())
        with mock.patch.dict(sys.modules, {'learning.task_anchor_once_v1.games': canonical}):
            with self.assertRaises(KeyboardInterrupt):
                with worker._bound_worker(FAKE_SOURCE, self.panel, GAME_ROOT, self.stage,
                                          self.plan, lambda: True) as isolated:
                    self.assertEqual(isolated.task_identity(self.stage, self.plan,
                                                           self.panel.worker_config())[1], 'a' * 64)
                    self.assertIsNot(isolated, canonical)
                    raise KeyboardInterrupt
            self.assertEqual(isolated.verify.__name__, 'verify')
            self.assertEqual(isolated.task_identity.__name__, 'task_identity')
            self.assertEqual(isolated.read.__name__, 'read')
            self.assertEqual(isolated.stable.__name__, 'stable')
            self.assertEqual(isolated.write_new.__name__, 'write_new')
            self.assertIs(sys.modules['learning.task_anchor_once_v1.games'], canonical)

    def test_reconstructed_trace_comparison_retains_order_and_numeric_type(self):
        with worker._bound_worker(FAKE_SOURCE, self.panel, GAME_ROOT, self.stage,
                                  self.plan, lambda: True) as isolated:
            self.assertNotEqual(isolated.stable({'a': 1, 'b': 2}),
                                isolated.stable({'b': 2, 'a': 1}))
            self.assertNotEqual(isolated.stable({'a': 1}), isolated.stable({'a': 1.0}))
            self.assertNotEqual(isolated.stable({'a': -0.0}), isolated.stable({'a': 0.0}))
            first = {'crop_records': [{'inference_seconds': .1, 'x': 1.0}]}
            second = {'crop_records': [{'inference_seconds': .9, 'x': 1.0}]}
            self.assertEqual(isolated.stable(first), isolated.stable(second))
            record = {'ordered': {'x': 1, 'y': 1.0}, 'coordinates': (1, 2),
                      'crop_records': [{'inference_seconds': .1, 'x': 1.0}]}
            self.assertEqual(isolated.stable(record),
                             isolated.stable(json.loads(json.dumps(record, allow_nan=False))))

    def test_only_current_storage_receipt_gets_nul_scope(self):
        storage = self.stage / 'job-results' / (self.plan['game'] + '.storage.json')
        original = {'game': self.plan['game'], 'passed': True,
                    'native_output_accounted_by': 'bounded parent process log'}
        with worker._bound_worker(FAKE_SOURCE, self.panel, GAME_ROOT, self.stage,
                                  self.plan, lambda: True) as isolated:
            isolated.write_new(storage, original)
            self.assertEqual(isolated.observed[-1][1]['native_output_accounted_by'],
                             worker.NATIVE_OUTPUT_SCOPE)
            self.assertEqual(original['native_output_accounted_by'], 'bounded parent process log')
            other = self.stage / 'job-results' / (self.plan['game'] + '.audit.json')
            isolated.write_new(other, original)
            self.assertIs(isolated.observed[-1][1], original)
            with self.assertRaises(ValueError):
                isolated.write_new(self.stage / 'job-results' / 'other.storage.json', original)
            with self.assertRaises(ValueError):
                isolated.write_new(storage, dict(original, native_output_accounted_by='unexpected'))

    def test_frozen_role_applies_to_wrapper_and_enriched_audit_row(self):
        with worker._bound_worker(FAKE_SOURCE, self.panel, GAME_ROOT, self.stage,
                                  self.job, lambda: True) as isolated:
            with mock.patch.object(worker, '_read_job', return_value=self.job):
                isolated.job(self.stage, self.job['game'])
            self.assertEqual(isolated.observed[0][1], 'a' * 64)
            self.assertEqual(isolated.observed[0][2:],
                             ('artifacts/synthetic/crop.npz', 'd' * 64))
            self.assertEqual(isolated.observed[0][0],
                             GAME_ROOT / 'artifacts/synthetic/baseline.npz')

    def test_wrong_private_job_is_rejected_before_worker_continues(self):
        altered = dict(self.job, task_checkpoint_sha256='b' * 64)
        with worker._bound_worker(FAKE_SOURCE, self.panel, GAME_ROOT, self.stage,
                                  self.job, lambda: True) as isolated:
            with mock.patch.object(worker, '_read_job', return_value=altered):
                with self.assertRaises(ValueError):
                    isolated.job(self.stage, self.job['game'])
            self.assertEqual(isolated.observed, [])

    def test_wrapper_uses_same_frozen_crop_and_registered_model(self):
        result = worker.make_wrapper(panel=self.panel, source=FAKE_SOURCE,
                                     game_root=GAME_ROOT, stage=self.stage, plan=self.plan,
                                     check=lambda: True)
        self.assertEqual(result, self.stage / 'agents' / (self.plan['game'] + '.py'))
        self.assertFalse(self.stage.exists())

    def test_job_runs_fake_worker_without_creating_files(self):
        from contextlib import contextmanager
        @contextmanager
        def fake_capture(*_args, **_kwargs):
            yield [{'game': self.job['game']}]
        with mock.patch.object(worker, '_read_job', return_value=self.job), \
             mock.patch.object(worker.capture, 'collect_one', fake_capture), \
             mock.patch.object(worker.capture, 'store_metrics') as saved:
            self.assertIsNone(worker.run_job(panel=self.panel, source=FAKE_SOURCE,
                                            game_root=GAME_ROOT, stage=self.stage,
                                            plan=self.job, check=lambda: True))
            saved.assert_called_once()
        self.assertFalse(self.stage.exists())

    def test_synthetic_full_worker_capture_and_consumer_are_connected(self):
        data = synthetic_inputs()
        data['worker_sha'] = hashlib.sha256(FULL_FAKE_SOURCE).hexdigest()
        panel = Panel.from_private_contract(**data)
        def trace(_path, _expected, consumer):
            for step in range(719):
                consumer.trace_turn(dict(step=step, records=[], load_errors=0,
                                         inference_errors=0, fallbacks=0))
        with mock.patch.object(worker, '_read_job', return_value=self.job), \
             mock.patch.object(worker.capture, 'feed_trace', side_effect=trace), \
             mock.patch.object(worker.capture, 'store_metrics') as saved:
            worker.run_job(panel=panel, source=FULL_FAKE_SOURCE, game_root=GAME_ROOT,
                           stage=self.stage, plan=self.job, check=lambda: True)
        saved.assert_called_once()
        payload = saved.call_args.args[2]
        self.assertEqual(payload['arm'], 'baseline')
        self.assertEqual(payload['result']['official_turns'], 719)
        self.assertEqual(payload['result']['economy']['own']['net_profit'], 0)
        self.assertFalse(self.stage.exists())

    def test_fake_worker_without_audit_or_failed_metric_write_cannot_succeed(self):
        with mock.patch.object(worker, '_read_job', return_value=self.job), \
             mock.patch.object(worker.capture, 'store_metrics') as saved:
            with self.assertRaisesRegex(RuntimeError, '^Development worker failed$'):
                worker.run_job(panel=self.panel, source=FAKE_SOURCE, game_root=GAME_ROOT,
                               stage=self.stage, plan=self.job, check=lambda: True)
            saved.assert_not_called()
        data = synthetic_inputs()
        data['worker_sha'] = hashlib.sha256(FULL_FAKE_SOURCE).hexdigest()
        panel = Panel.from_private_contract(**data)
        def trace(_path, _expected, consumer):
            for step in range(719):
                consumer.trace_turn(dict(step=step, records=[], load_errors=0,
                                         inference_errors=0, fallbacks=0))
        with mock.patch.object(worker, '_read_job', return_value=self.job), \
             mock.patch.object(worker.capture, 'feed_trace', side_effect=trace), \
             mock.patch.object(worker.capture, 'store_metrics', side_effect=OSError('private detail')):
            with self.assertRaisesRegex(RuntimeError, '^Development worker failed$'):
                worker.run_job(panel=panel, source=FULL_FAKE_SOURCE, game_root=GAME_ROOT,
                               stage=self.stage, plan=self.job, check=lambda: True)

    def test_bad_worker_or_plan_fails_without_raw_error(self):
        with self.assertRaisesRegex(RuntimeError, '^Development worker failed$'):
            worker.run_job(panel=self.panel, source=b'wrong', game_root=GAME_ROOT,
                           stage=self.stage, plan=self.job, check=lambda: True)
        with self.assertRaisesRegex(RuntimeError, '^Development worker failed$'):
            worker.run_job(panel=self.panel, source=FAKE_SOURCE, game_root=GAME_ROOT,
                           stage=self.stage, plan=dict(self.job, split='holdout'),
                           check=lambda: True)

    def test_non_main_or_extra_python_thread_rejected_before_output_redirection(self):
        with mock.patch.object(worker.threading, 'enumerate', return_value=[1, 2]):
            with mock.patch.object(worker, '_quiet_output') as quiet:
                with self.assertRaisesRegex(RuntimeError, '^Development wrapper failed$'):
                    worker.make_wrapper(panel=self.panel, source=FAKE_SOURCE, game_root=GAME_ROOT,
                                        stage=self.stage, plan=self.plan, check=lambda: True)
                quiet.assert_not_called()

    def test_missing_native_output_handle_stops_before_worker(self):
        with mock.patch.object(worker.os, 'dup', side_effect=OSError('synthetic sensitive detail')):
            with mock.patch.object(worker, '_load_snapshot') as load:
                with self.assertRaisesRegex(RuntimeError, '^Development worker failed$'):
                    worker.run_job(panel=self.panel, source=FAKE_SOURCE, game_root=GAME_ROOT,
                                   stage=self.stage, plan=self.job, check=lambda: True)
                load.assert_not_called()

    def test_job_reader_rejects_duplicate_nonfinite_and_oversized_data(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b' ' * (worker.MAX_JOB_BYTES + 1)):
            with self.subTest(size=len(raw)):
                with mock.patch.object(Path, 'open', return_value=io.BytesIO(raw)):
                    with self.assertRaises(ValueError):
                        worker._read_job(GAME_ROOT / 'synthetic')

    def test_python_and_native_output_are_suppressed_in_real_child(self):
        code = '''\
import os, sys
from learning.task_pair_dev_games_v1.worker import _quiet_output
with _quiet_output():
    print('SYNTHETIC_PRIVATE_STDOUT', flush=True)
    print('SYNTHETIC_PRIVATE_STDERR', file=sys.stderr, flush=True)
    os.write(1, b'SYNTHETIC_NATIVE_STDOUT')
    os.write(2, b'SYNTHETIC_NATIVE_STDERR')
print('outside-ok', flush=True)
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=GAME_ROOT,
                                capture_output=True, timeout=20, check=True)
        self.assertEqual(result.stdout.replace(b'\r\n', b'\n'), b'outside-ok\n')
        self.assertEqual(result.stderr, b'')


if __name__ == '__main__':
    unittest.main()
