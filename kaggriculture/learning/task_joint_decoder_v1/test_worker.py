"""Synthetic identity and hook integration; no real models or game outputs."""
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_pair_dev_games_v1 import capture, worker as original
from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_dev_games_v1.test_panel import synthetic_inputs
from learning.task_pair_dev_games_v1.test_worker import FULL_FAKE_SOURCE
from learning.task_pair_dev_campaign_v1 import acceptance
from . import policy, worker
from .contract import JointPanel


ROOT = Path(__file__).resolve().parents[2]
SOURCE = FULL_FAKE_SOURCE + b'''
def actor(*_args):
    raise AssertionError('old actor must not load')
original_fake_audit = audit
def audit(stage, row, cfg):
    selected = actor(stage, row, cfg)
    assert selected.options['trace'] is None
    return original_fake_audit(stage, row, cfg)
'''


class FakeAgent:
    def __init__(self, *args, **options):
        self.args, self.options = args, options

    def __call__(self, observation, configuration):
        return dict(farmer=['PASS'], hands=[], market=[])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        values = synthetic_inputs()
        values['worker_sha'] = hashlib.sha256(SOURCE).hexdigest()
        self.base = Panel.from_private_contract(**values)
        self.panel = JointPanel(self.base)
        self.stage = ROOT / 'artifacts' / 'synthetic-joint-worker-not-created'
        self.manifest = SimpleNamespace(panel=self.base,
            opponent=lambda role: ('bots/' + role + '.py', self.base.rivals[('market', 'round2').index(role)]))
        self.plan = self.panel.plans()[0]
        self.job = self.make_job(self.plan)
        self.addCleanup(patch.stopall)
        patch.object(worker, 'GAME', ROOT).start()
        patch.object(worker, 'JointAgent', FakeAgent).start()

    def make_job(self, plan):
        return dict(plan, bot_sha256='9'*64,
            rival_sha256=self.base.rivals[('market', 'round2').index(plan['opponent'])],
            task_checkpoint_sha256=self.base.models[0][1])

    def test_real_panel_keeps_three_models_and_four_plans_use_only_baseline(self):
        self.assertEqual(len(self.base.models), 3)
        self.assertEqual(len(self.panel.plans()), 4)
        for plan in self.panel.plans():
            self.assertEqual(plan['policy'], 'joint_decoder')
            self.assertEqual(self.panel.task_model(plan), self.base.models[0])
            with self.assertRaises(ValueError):
                self.panel.validate_plan(dict(self.make_job(plan),
                    task_checkpoint_sha256=self.base.models[1][1]), job=True)

    def test_live_wrapper_and_audit_actor_share_weights_but_only_live_writes_trace(self):
        with original._bound_worker(SOURCE, self.panel, ROOT, self.stage, self.job, lambda: True) as module:
            unchanged = {name: getattr(module, name) for name in ('job', 'audit', 'reexecute', 'owned_writes', 'stable')}
            actor_before, wrapper_before = module.actor, module.wrapper
            with worker.joint_hooks(module, self.panel, ROOT, self.stage, lambda: True):
                agent = module.actor(self.stage, dict(self.job, states=720), self.panel.worker_config())
                text = io.StringIO()
                with patch.object(Path, 'mkdir'), patch.object(Path, 'open', return_value=nullcontext(text)):
                    path = module.wrapper(self.stage, self.plan, self.panel.worker_config())
                code = text.getvalue();namespace = {'__file__': str(path)}
                previous_path = list(sys.path)
                with patch.object(policy, 'JointAgent', FakeAgent):
                    exec(compile(code, str(path), 'exec'), namespace)
                live = namespace['_runner']
                self.assertEqual(sys.path, previous_path)
                self.assertEqual(live.args, agent.args)
                self.assertEqual(agent.args[2:], (ROOT / self.base.models[0][0], self.base.models[0][1]))
                self.assertIsNone(agent.options['trace'])
                self.assertEqual(live.options['trace'], self.stage / 'traces' / (self.plan['game'] + '.jsonl.gz'))
                self.assertEqual(live.options['limits'], self.base.worker_config()['game_limits'])
                self.assertEqual(namespace['agent']({}, {}), namespace['submission_agent']({}, {}))
                self.assertEqual({name: getattr(module, name) for name in unchanged}, unchanged)
                with self.assertRaises(ValueError):
                    module.actor(self.stage, dict(self.job, task_checkpoint_sha256=self.base.models[1][1]),
                                 self.panel.worker_config())
            self.assertIs(module.actor, actor_before)
            self.assertIs(module.wrapper, wrapper_before)
        self.assertFalse(self.stage.exists())

    def test_hooks_restore_on_base_exception_and_wrapper_is_seed_independent(self):
        with original._bound_worker(SOURCE, self.panel, ROOT, self.stage, self.plan, lambda: True) as module:
            before = module.actor, module.wrapper
            with self.assertRaises(KeyboardInterrupt):
                with worker.joint_hooks(module, self.panel, ROOT, self.stage, lambda: True):
                    raise KeyboardInterrupt
            self.assertEqual((module.actor, module.wrapper), before)
        args = (ROOT, self.stage, self.plan, self.panel.worker_config(), ROOT / self.base.models[0][0], self.base.models[0][1])
        alternative = (ROOT, self.stage, dict(self.plan, seed=0), *args[3:])
        self.assertEqual(worker._wrapper_code(*args), worker._wrapper_code(*alternative))

    def test_execute_reuses_capture_and_consumer_for_both_opponents_and_seats(self):
        # This is the existing synthetic audit fixture, not a real official
        # replay. Actual collect_one + GameConsumer consume its 719 trace rows.
        fake_reexecute = original._load_snapshot(SOURCE, self.panel.worker_sha, ROOT, lambda: True).reexecute
        def ordered_fixture(replay, *, guard, sink):
            self.assertIs(guard(), True)
            return dict(official=fake_reexecute(replay, trace=True, label_sink=sink), shed_order_frames_checked=720)
        for plan in self.panel.plans():
            with self.subTest(opponent=plan['opponent'], side=plan['side']):
                job = self.make_job(plan);written = {}
                def read(_root, _path, _maximum): return SOURCE
                def hashed(_root, path, _maximum):
                    return '9'*64 if path.parent.name == 'agents' else job['rival_sha256']
                def save(_root, path, value, **_kwargs): written[str(path)] = value
                dm = SimpleNamespace(_read=read, _hash=hashed, _write_new=save, MAX_SOURCE=4*1024**2)
                def trace(_path, _digest, consumer):
                    for step in range(719):
                        consumer.trace_turn(dict(step=step, records=[], load_errors=0, inference_errors=0, fallbacks=0))
                def stored_job(path): return written[str(path)]
                with patch.object(Path, 'mkdir'), \
                        patch.object(Path, 'open', side_effect=lambda *_a, **_k: nullcontext(io.StringIO())), \
                        patch.object(original, '_read_job', side_effect=stored_job), \
                        patch.object(capture, 'feed_trace', side_effect=trace), \
                        patch.object(capture, 'store_metrics') as metrics, \
                        patch.object(worker.replay_audit, 'reexecute', side_effect=ordered_fixture) as official, \
                        patch.object(worker, 'hidden_official_entry', side_effect=lambda *_a: nullcontext()):
                    worker.execute({'driver': dm, 'worker': original}, self.manifest, self.stage, plan, lambda: True)
                self.assertEqual(official.call_count, 1)
                self.assertEqual(written[str(self.stage / 'job-inputs' / (plan['game'] + '.json'))], job)
                self.assertEqual(metrics.call_count, 1)
                payload = metrics.call_args.args[2]
                self.assertEqual((payload['arm'], payload['side'], payload['opponent']),
                                 ('joint_decoder', plan['side'], plan['opponent']))
                self.assertEqual(payload['task_checkpoint_sha256'], self.base.models[0][1])
                self.assertEqual(payload['result']['official_turns'], 719)
                self.assertTrue(payload['result']['economy']['own']['cash_reconciled'])
        self.assertFalse(self.stage.exists())

    def test_same_replay_order_hook_rejects_incomplete_coverage_and_restores(self):
        prior = object();module = SimpleNamespace(reexecute=prior)
        with patch.object(worker.replay_audit, 'reexecute', return_value={
                'official': {'passed': True}, 'shed_order_frames_checked': 719}):
            with self.assertRaises(ValueError):
                with worker.ordered_main_replay(module, lambda: True):
                    module.reexecute({}, trace=True, label_sink=lambda _: None)
        self.assertIs(module.reexecute, prior)

    def test_verify_files_checks_every_identity_and_exact_job_types(self):
        row = dict(self.job, trace_sha256='6'*64, replay_gzip_sha256='7'*64)
        game = self.plan['game'];model, digest = self.base.models[0]
        rival, rival_sha = self.manifest.opponent(self.plan['opponent'])
        expected = {ROOT / model: digest, ROOT / self.base.crop[0]: self.base.crop[1],
            ROOT / rival: rival_sha, self.stage / 'sources' / (self.plan['opponent'] + '.py'): rival_sha,
            ROOT / original.WORKER_RELATIVE: self.panel.worker_sha,
            self.stage / 'agents' / (game + '.py'): self.job['bot_sha256'],
            self.stage / 'traces' / (game + '.jsonl.gz'): row['trace_sha256'],
            self.stage / 'games' / game / 'replay.json.gz': row['replay_gzip_sha256']}
        raw = json.dumps(self.job).encode()
        dm = SimpleNamespace(_read=lambda *_a: raw,
            _hash=lambda _root,path,_limit: expected[path], MAX_MODEL=16*1024**2, MAX_SOURCE=4*1024**2)
        modules = dict(driver=dm, worker=original, acceptance=acceptance)
        self.assertTrue(worker.verify_files(modules, self.manifest, self.stage, self.plan, row))
        for path, value in tuple(expected.items()):
            expected[path] = '0'*64
            with self.subTest(file_kind=path.suffix), self.assertRaises(ValueError):
                worker.verify_files(modules, self.manifest, self.stage, self.plan, row)
            expected[path] = value
        with self.assertRaises(ValueError):
            worker.verify_files(modules, self.manifest, self.stage, self.plan, dict(row, side=False))


if __name__ == '__main__':
    unittest.main()
