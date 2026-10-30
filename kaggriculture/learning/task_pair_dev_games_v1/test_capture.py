"""In-memory synthetic replay/sink integration and private output checks."""
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

GAME_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GAME_ROOT))
from learning.task_pair_dev_games_v1 import capture
from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_dev_games_v1.test_panel import synthetic_inputs
from learning.task_pair_dev_metrics_v1.consumer import GameConsumer


def turns(count=719):
    return [dict(step=i, records=[], load_errors=0, inference_errors=0, fallbacks=0)
            for i in range(count)]


def compressed(rows):
    return gzip.compress(b''.join((json.dumps(row) + '\n').encode() for row in rows))


def unit_event(player):
    return dict(step=0, player=player, phase='unit', unit=0, status='pass',
                requested=['PASS'], effective=['PASS'], position=[0, 0],
                before={'money': 100}, after={'money': 100})


def observations(step):
    farms = [dict(money=100), dict(money=100)]
    return [dict(observation=dict(step=step, farms=farms,
                                  private=dict(seeds={}, shed={})),
                 status='DONE' if step == 719 else 'ACTIVE') for _ in (0, 1)]


class Recorder:
    def __init__(self):
        self.turns = []
    def trace_turn(self, value):
        self.turns.append(value['step'])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.panel = Panel.from_private_contract(**synthetic_inputs())
        self.stage = GAME_ROOT / 'artifacts' / 'synthetic-not-created'
        self.plan = self.panel.plans()[0]
        self.job = dict(self.plan, bot_sha256='9' * 64, rival_sha256='e' * 64,
                        task_checkpoint_sha256='a' * 64)
        self.row = dict(self.job, trace_sha256='7' * 64, replay_gzip_sha256='8' * 64)
        self.replay = {'steps': [observations(0)] * 719 + [observations(719)]}
        self.original_sink_events = []
        self.engine_calls = 0
        self.module = SimpleNamespace()

        def replay(replay, *, trace, label_sink):
            self.engine_calls += 1
            for player in (0, 1):
                label_sink(unit_event(player))
            return dict(passed=True, official_transitions_reexecuted=719,
                        both_public_and_private_states_match_each_step=True,
                        inventory_order_checked_each_step=True,
                        execution=dict(label_version=2, counts={'unit:pass': 2}),
                        final_rewards=[100, 100])

        def audit(out, row, cfg):
            official = self.module.reexecute(self.replay, trace=True,
                                             label_sink=self.original_sink_events.append)
            return dict(passed=True, official=official)

        self.module.audit = audit
        self.module.reexecute = replay
        self.module.observation = lambda frame, player: frame[player]['observation']

    def _feed_synthetic(self, _path, _sha, consumer):
        for turn in turns():
            consumer.trace_turn(turn)

    def test_bounded_trace_reads_719_in_order_without_real_files(self):
        raw = compressed(turns())
        seen = Recorder()
        with mock.patch.object(Path, 'open', return_value=io.BytesIO(raw)):
            capture.feed_trace(GAME_ROOT / 'synthetic', hashlib.sha256(raw).hexdigest(), seen)
        self.assertEqual(seen.turns, list(range(719)))

    def test_incomplete_extra_reordered_and_invalid_trace_fail(self):
        bad_rows = [turns(718), turns(720), turns(), turns()]
        bad_rows[2][5]['step'] = 4
        bad_rows[3][5]['invalid'] = float('nan')
        for rows in bad_rows:
            data = compressed(rows)
            with self.subTest(length=len(rows)):
                with mock.patch.object(Path, 'open', return_value=io.BytesIO(data)):
                    with self.assertRaises(ValueError):
                        capture.feed_trace(GAME_ROOT / 'synthetic', hashlib.sha256(data).hexdigest(), Recorder())

    def test_compressed_and_line_budgets_and_bad_hash_stop(self):
        data = compressed(turns())
        with mock.patch.object(Path, 'open', return_value=io.BytesIO(data)):
            with self.assertRaises(ValueError):
                capture.feed_trace(GAME_ROOT / 'synthetic', '0' * 64, Recorder())
        with mock.patch.object(Path, 'open', return_value=io.BytesIO(data)), \
             mock.patch.object(capture, 'MAX_COMPRESSED_BYTES', 10):
            with self.assertRaises(ValueError):
                capture.feed_trace(GAME_ROOT / 'synthetic', hashlib.sha256(data).hexdigest(), Recorder())
        with mock.patch.object(Path, 'open', return_value=io.BytesIO(data)), \
             mock.patch.object(capture, 'MAX_LINE_BYTES', 5):
            with self.assertRaises(ValueError):
                capture.feed_trace(GAME_ROOT / 'synthetic', hashlib.sha256(data).hexdigest(), Recorder())
        with mock.patch.object(Path, 'open', return_value=io.BytesIO(data)), \
             mock.patch.object(capture, 'MAX_DECODED_BYTES', 20):
            with self.assertRaises(ValueError):
                capture.feed_trace(GAME_ROOT / 'synthetic', hashlib.sha256(data).hexdigest(), Recorder())

    def test_original_replay_runs_once_with_both_sinks_and_cash_reconciliation(self):
        original_audit, original_replay = self.module.audit, self.module.reexecute
        with mock.patch.object(capture, 'feed_trace', side_effect=self._feed_synthetic):
            with capture.collect_one(self.module, stage=self.stage, panel=self.panel,
                                     expected=self.job, check=lambda: True,
                                     consumer_factory=GameConsumer) as results:
                certificate = self.module.audit(self.stage, self.row, self.panel.worker_config())
                self.assertTrue(certificate['passed'])
                self.assertEqual(len(results), 1)
                result = results[0]['result']
                self.assertEqual(result['official_turns'], 719)
                self.assertEqual(result['economy']['own']['net_profit'], 0)
                self.assertEqual(result['economy']['opponent']['net_profit'], 0)
        self.assertEqual(self.engine_calls, 1)
        self.assertEqual(self.original_sink_events, [unit_event(0), unit_event(1)])
        self.assertIs(self.module.audit, original_audit)
        self.assertIs(self.module.reexecute, original_replay)

    def test_original_audit_failure_and_consumer_sink_failure_leave_no_metrics(self):
        original = self.module.audit
        def rejected(out, row, cfg):
            original(out, row, cfg)
            raise ValueError('synthetic audit failure')
        self.module.audit = rejected
        with mock.patch.object(capture, 'feed_trace', side_effect=self._feed_synthetic):
            with capture.collect_one(self.module, stage=self.stage, panel=self.panel,
                                     expected=self.job, check=lambda: True,
                                     consumer_factory=GameConsumer) as results:
                with self.assertRaises(ValueError):
                    self.module.audit(self.stage, self.row, self.panel.worker_config())
                self.assertEqual(results, [])
        self.assertEqual(self.engine_calls, 1)

        self.module.audit = original
        class RejectingConsumer(GameConsumer):
            def sink(self, _event):
                raise ValueError('synthetic consumer failure')
        with mock.patch.object(capture, 'feed_trace', side_effect=self._feed_synthetic):
            with capture.collect_one(self.module, stage=self.stage, panel=self.panel,
                                     expected=self.job, check=lambda: True,
                                     consumer_factory=RejectingConsumer) as results:
                with self.assertRaises(ValueError):
                    self.module.audit(self.stage, self.row, self.panel.worker_config())
                self.assertEqual(results, [])

    def test_original_sink_failure_is_not_hidden_and_consumer_does_not_finish(self):
        seen = []
        class CountingConsumer(GameConsumer):
            def sink(self, event):
                seen.append(event)
                return super().sink(event)
        def bad_audit(_out, _row, _cfg):
            def bad_sink(_event):
                raise RuntimeError('synthetic original sink failure')
            self.module.reexecute(self.replay, trace=True, label_sink=bad_sink)
        self.module.audit = bad_audit
        with mock.patch.object(capture, 'feed_trace', side_effect=self._feed_synthetic):
            with capture.collect_one(self.module, stage=self.stage, panel=self.panel,
                                     expected=self.job, check=lambda: True,
                                     consumer_factory=CountingConsumer) as results:
                with self.assertRaises(RuntimeError):
                    self.module.audit(self.stage, self.row, self.panel.worker_config())
                self.assertEqual(results, [])
        self.assertEqual(seen, [])

    def test_two_replays_or_wrong_job_are_rejected(self):
        def twice(_out, _row, _cfg):
            first = self.module.reexecute(self.replay, trace=True, label_sink=lambda _: None)
            self.module.reexecute(self.replay, trace=True, label_sink=lambda _: None)
            return {'passed': True, 'official': first}
        self.module.audit = twice
        with mock.patch.object(capture, 'feed_trace', side_effect=self._feed_synthetic):
            with capture.collect_one(self.module, stage=self.stage, panel=self.panel,
                                     expected=self.job, check=lambda: True,
                                     consumer_factory=GameConsumer) as results:
                with self.assertRaises(ValueError):
                    self.module.audit(self.stage, self.row, self.panel.worker_config())
                self.assertEqual(results, [])
        self.assertEqual(self.engine_calls, 1)
        with capture.collect_one(self.module, stage=self.stage, panel=self.panel,
                                 expected=self.job, check=lambda: True,
                                 consumer_factory=GameConsumer) as results:
            with self.assertRaises(ValueError):
                self.module.audit(self.stage, dict(self.row, side=1), self.panel.worker_config())
            self.assertEqual(results, [])

    def test_failed_replay_cannot_be_silently_retried(self):
        def failing_replay(*_args, **_kwargs):
            self.engine_calls += 1
            raise ValueError('synthetic replay failure')
        self.module.reexecute = failing_replay
        def retrying_audit(_out, _row, _cfg):
            try:
                self.module.reexecute(self.replay, trace=True, label_sink=lambda _: None)
            except ValueError:
                pass
            self.module.reexecute(self.replay, trace=True, label_sink=lambda _: None)
        self.module.audit = retrying_audit
        with mock.patch.object(capture, 'feed_trace', side_effect=self._feed_synthetic):
            with capture.collect_one(self.module, stage=self.stage, panel=self.panel,
                                     expected=self.job, check=lambda: True,
                                     consumer_factory=GameConsumer) as results:
                with self.assertRaises(ValueError):
                    self.module.audit(self.stage, self.row, self.panel.worker_config())
                self.assertEqual(results, [])
        self.assertEqual(self.engine_calls, 1)

    def test_output_budget_fails_before_writing(self):
        fake_module = SimpleNamespace(owned_writes=mock.Mock())
        with mock.patch.object(capture, 'MAX_METRICS_BYTES', 100):
            with self.assertRaises(ValueError):
                capture.store_metrics(fake_module, self.stage,
                                      {'game': self.job['game'], 'large': 'x' * 100})
        fake_module.owned_writes.assert_not_called()

    def test_private_metric_and_receipt_exclusive_write_and_readback_in_memory(self):
        files = {}
        @contextmanager
        def budget(_root, limit, *, allowed_paths):
            self.assertEqual(limit, capture.MAX_METRICS_BYTES)
            self.assertEqual(len(allowed_paths), 2)
            yield
        class Writer(io.BytesIO):
            def __init__(self, key):
                super().__init__()
                self.key = key
            def close(self):
                if not self.closed:
                    files[self.key] = self.getvalue()
                super().close()
        def open_fake(path, mode):
            key = str(path)
            if mode == 'xb':
                if key in files:
                    raise FileExistsError('synthetic already exists')
                return Writer(key)
            if mode == 'rb':
                return io.BytesIO(files[key])
            raise AssertionError('Unexpected open mode')
        fake_module = SimpleNamespace(owned_writes=budget)
        payload = {'game': self.job['game'], 'result': {'events': [{'step': 0, 'status': 'pass'}]}}
        with mock.patch.object(Path, 'open', autospec=True, side_effect=open_fake):
            capture.store_metrics(fake_module, self.stage, payload)
            with self.assertRaises(FileExistsError):
                capture.store_metrics(fake_module, self.stage, payload)
        self.assertEqual(len(files), 2)
        metric = next(value for key, value in files.items() if key.endswith('.metrics.local.json'))
        receipt = json.loads(next(value for key, value in files.items()
                                  if key.endswith('.metrics-receipt.local.json')))
        self.assertEqual(receipt['metrics_sha256'], hashlib.sha256(metric).hexdigest())
        self.assertEqual(receipt['metrics_bytes'], len(metric))
        self.assertEqual(receipt['official_replay_passes'], 1)
        self.assertEqual(json.loads(metric)['result']['events'], [{'step': 0, 'status': 'pass'}])


if __name__ == '__main__':
    unittest.main()
