"""Synthetic bounded IO only, in the launcher's externally owned scratch run."""
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_dev_games_v1.test_panel import synthetic_inputs
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest
from learning.task_harvest_window_v1.test_worker import MemorySourceDriver
from . import contract, core, worker, selection
from . import test_core as fixtures


class SharedClockFactory(fixtures.FakeFactory):
    """The official raw layout has a shared clock only on player zero."""
    def __call__(self, *args, **kwargs):
        environment = super().__call__(*args, **kwargs)
        original = environment.step
        def step(joint):
            original(joint)
            environment.state[1]['observation'].pop('step', None)
        environment.step = step
        return environment


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get('AI_GAME_TEST_SCRATCH')
        if not value:
            raise unittest.SkipTest('Run inside the approved owned scratch context')
        cls.scratch = Path(value)
        if not cls.scratch.is_absolute() or not cls.scratch.is_dir():
            raise ValueError('Invalid owned test directory')
        cls.replay, cls.traces = fixtures.fixture(side=1, units=2)
        for frame in cls.replay['steps']:
            frame[1]['observation'].pop('step', None)

    def setUp(self):
        self.out = self.scratch/('unexpressed-'+self._testMethodName)
        self.out.mkdir(exist_ok=False)
        self.root = self.scratch/'synthetic-game-not-created'
        self.source = contract.sources()[1]
        manifests, rows, hashes = {}, {}, {}
        for cohort in ('C01', 'C02'):
            values = synthetic_inputs()
            if cohort == 'C02':
                values['seeds'] = {k: v+100000 for k, v in values['seeds'].items()}
            panel = Panel.from_private_contract(**values)
            manifest = PrivateManifest(panel, tuple((f'bots/{op}.py', digest)
                for op, digest in zip(('market', 'round2'), panel.rivals)))
            manifests[cohort] = manifest
            stage = self.root/'artifacts'/cohort
            for plan in panel.plans():
                if plan['policy'] == 'auxiliary':
                    continue
                rows[(cohort, plan['game'])] = dict(plan,
                    initial_banks=[f['money'] for f in self.replay['steps'][0][0]['observation']['farms']],
                    replay_gzip_sha256='8'*64, trace_sha256='7'*64,
                    bot_sha256='9'*64, rival_sha256=manifest.opponent(plan['opponent'])[1],
                    task_checkpoint_sha256=panel.models[0 if plan['policy']=='baseline' else 1][1],
                    crop_checkpoint_sha256=panel.crop[1], engine_sha256=contract.ENGINE_SHA256)
                hashes[stage/'agents'/(plan['game']+'.py')] = '9'*64
                hashes[stage/'games'/plan['game']/'replay.json.gz'] = '8'*64
                hashes[stage/'traces'/(plan['game']+'.jsonl.gz')] = '7'*64
            for path, digest in (*panel.models, panel.crop):
                hashes[self.root/path] = digest
            for op in ('market', 'round2'):
                path, digest = manifest.opponent(op)
                hashes[self.root/path] = hashes[stage/'sources'/(op+'.py')] = digest
            hashes[self.root/'learning/task_anchor_once_v1/games.py'] = panel.worker_sha
        replay = deepcopy(self.replay)
        replay['info']['seed'] = rows[(self.source['cohort'], self.source['game'])]['seed']
        self.context = SimpleNamespace(manifests=manifests, rows=rows)
        self.dm = MemorySourceDriver(replay, self.traces, hashes)
        self.modules = dict(driver=self.dm, acceptance=SimpleNamespace(_json=lambda raw, limit: json.loads(raw)),
            fresh=SimpleNamespace(GAME=self.root, STAGES=dict(C01='C01', C02='C02')),
            worker=SimpleNamespace(WORKER_RELATIVE='learning/task_anchor_once_v1/games.py'),
            first_yield_days=fixtures.DAYS)
        # Use the second unit to exercise hands without fabricating a market in trace.
        target = fixtures.CoreTests().target(self.traces, side=1, unit=1)
        self.selection = dict(schema_version=selection.SCHEMA_VERSION, target=target,
                              counts=dict(trace_lines=719), reason=None, eligibility={})
        self.actor_arguments = []
        self.engine, self.capture = SharedClockFactory(), fixtures.Capture()
        runtime = core.Runtime(self.engine, self.engine.identity, fixtures.observed, fixtures.stable, lambda actor: None)
        def actor(*args, **kwargs):
            self.actor_arguments.append((args, kwargs))
            return fixtures.Actor()
        self.addCleanup(patch.stopall)
        patch.object(worker, '_bounded_actor', side_effect=actor).start()
        patch.object(worker, '_load_rival', side_effect=lambda path: fixtures.FakeRival()).start()
        patch.object(core, 'official_runtime', return_value=runtime).start()
        patch.object(core, '_official_target_trace', self.capture).start()

    def execute(self, value=None):
        return worker.execute_case(self.modules, self.context, self.source,
            self.selection if value is None else value, self.out, lambda: None)

    def verify(self, value=None):
        return worker.verify_case(self.modules, self.context, self.source,
            self.selection if value is None else value, self.out, lambda: None)

    def rewrite_events(self, mutate):
        """Update byte receipts so a content failure reaches the consumer itself."""
        path = self.out/worker.EVENT_FILE
        with gzip.open(path, 'rb') as stream:
            events = [json.loads(line) for line in stream]
        mutate(events)
        lines = [(json.dumps(event, separators=(',', ':'), ensure_ascii=False)+'\n').encode() for event in events]
        raw = gzip.compress(b''.join(lines), mtime=0)
        path.write_bytes(raw)
        receipt = self.out/worker.CASE_FILE
        document = json.loads(receipt.read_bytes())
        document['events'].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), decoded_bytes=sum(map(len, lines)))
        document['output_budget']['used_bytes'] = len(raw)
        receipt.write_text(json.dumps(document), encoding='utf-8')

    def test_baseline_weights_hands_patch_and_raw_shared_clock_roundtrip(self):
        executed = self.execute()
        self.assertEqual((self.dm.replay_reads, self.dm.trace_starts, self.dm.trace_ends), (1, 2, 2))
        for args, kwargs in self.actor_arguments:
            panel = self.context.manifests['C01'].panel
            self.assertEqual(args, (self.root/panel.crop[0], panel.crop[1], self.root/panel.models[0][0], panel.models[0][1]))
            self.assertEqual(kwargs, dict(limits=core.GAME_LIMITS, trace=None))
        result = executed['document']['core']
        self.assertEqual(result['intervention']['target_execution']['requested'], ['HARVEST'])
        self.assertEqual(result['intervention']['target_proof']['original_unit_request'], ['WATER'])
        self.assertEqual((self.capture.entered, self.capture.exited), (2, 2))
        verified = self.verify()
        self.assertEqual(executed['receipt'], verified['receipt'])
        initial = [f['money'] for f in self.replay['steps'][0][0]['observation']['farms']]
        self.assertEqual(verified['document']['row']['intervention']['end'],
                         dict(own=initial[1]+1, opponent=initial[0]))
        self.assertEqual((self.dm.replay_reads, self.dm.trace_starts, self.dm.trace_ends), (1, 3, 3))
        self.assertEqual(len(self.actor_arguments), 2)

    def test_missing_does_not_construct_or_count_a_zero_reward_pair(self):
        missing = dict(schema_version=selection.SCHEMA_VERSION, target=None, counts=dict(trace_lines=719),
                       reason='synthetic missing', eligibility=None)
        result = self.execute(missing)
        self.assertEqual(self.verify(missing)['receipt'], result['receipt'])
        self.assertNotIn('baseline', result['document']['row'])
        self.assertEqual(self.actor_arguments, [])
        self.assertEqual((self.dm.replay_reads, self.dm.trace_starts), (0, 0))

    def test_tail_authentication_failure_never_starts_intervention(self):
        self.dm.tail_failure = True
        with self.assertRaisesRegex(ValueError, 'tail mismatch'):
            self.execute()
        value = json.loads((self.out/worker.CASE_FILE).read_bytes())
        self.assertFalse(value['passed'])
        self.assertEqual(value['meter']['calls']['official_step_returns'], 48)
        self.assertEqual(value['meter']['calls'].get('request_patch_returns', 0), 0)
        self.assertEqual(len(self.actor_arguments), 1)

    def test_consumer_rejects_extra_unit_or_market_edit_even_with_new_byte_receipts(self):
        original_replace = worker.replace_harvest
        for kind in ('other_unit', 'market'):
            with self.subTest(kind=kind):
                subdir = self.out/kind
                subdir.mkdir()
                previous, self.out = self.out, subdir
                try:
                    self.execute()
                    def change(events):
                        event = next(e for e in events if e.get('patch') is not None)
                        if kind == 'other_unit':
                            event['request']['farmer'] = ['PASS']
                        else:
                            event['request']['market'].append(['SYNTHETIC', 'extra'])
                    self.rewrite_events(change)
                    with patch.object(worker, 'replace_harvest', wraps=original_replace) as consume:
                        with self.assertRaises(ValueError):
                            self.verify()
                        consume.assert_called_once()
                finally:
                    self.out = previous

    def test_original_trace_and_endpoint_order_are_verified_not_just_target_summary(self):
        self.execute()
        def change(events):
            event = next(e for e in events if e.get('scope') == 'suffix'
                         and e['arm'] == 'intervene' and e['step'] == 241)
            event['trace']['records'][0]['unit_request'] = ['HARVEST']
            event['stable_trace_sha256'] = worker.ordered_hash(worker._stable(event['trace']))
        self.rewrite_events(change)
        with self.assertRaises(ValueError):
            self.verify()

    def test_endpoint_cash_and_order_cannot_be_replaced_in_receipt(self):
        self.execute()
        path = self.out/worker.CASE_FILE
        original = json.loads(path.read_bytes())
        for change in ('cash', 'shed_order'):
            with self.subTest(change=change):
                document = deepcopy(original)
                if change == 'cash':
                    document['row']['intervention']['end']['own'] += 100
                else:
                    document['core']['intervention']['endpoints']['end']['players'][0]['shed'].reverse()
                path.write_text(json.dumps(document), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.verify()

    def test_output_budget_keeps_reserved_failure_receipt(self):
        with patch.object(contract, 'LIMITS', dict(contract.LIMITS, per_case_output_bytes=worker.CASE_RESERVE+64)):
            with self.assertRaises(OSError):
                self.execute()
        value = json.loads((self.out/worker.CASE_FILE).read_bytes())
        self.assertFalse(value['passed'])
        self.assertEqual(value['error_kind'], 'output_budget')
        self.assertLessEqual(value['output_budget']['used_bytes'], 64)


if __name__ == '__main__':
    unittest.main()
