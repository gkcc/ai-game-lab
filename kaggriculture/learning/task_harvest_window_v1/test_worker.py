"""Synthetic Worker IO in an externally owned scratch run; no engine/models.

The test launcher supplies AI_GAME_TEST_SCRATCH from the allowed standalone
artifacts.scratch_directory context and deletes that run after tests finish.
"""
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_dev_games_v1.test_panel import synthetic_inputs
from learning.task_pair_dev_campaign_v1.manifest import PrivateManifest
from . import core, worker, contract
from .test_core import source_fixture, FakeActor, FakeRival, FakeFactory, observed, stable


class MemorySourceDriver:
    """Source bytes are synthetic; output files use real bounded Python IO."""
    MAX_MODEL, MAX_SOURCE = 16*1024**2, 4*1024**2

    def __init__(self, replay, traces, sources):
        self.replay, self.traces, self.sources = replay, traces, sources
        self.replay_reads = self.trace_starts = self.trace_ends = 0
        self.tail_failure = False

    def _no_links(self, root, path):
        if not path.is_relative_to(root):
            raise ValueError('fixture path escaped')

    def _hash(self, root, path, limit):
        if path in self.sources:
            return self.sources[path]
        raw = path.read_bytes()
        if len(raw) > limit:
            raise ValueError('fixture output exceeded limit')
        return hashlib.sha256(raw).hexdigest()

    def _read(self, root, path, limit):
        raw = path.read_bytes()
        if len(raw) > limit:
            raise ValueError('fixture output exceeded limit')
        return raw

    def _write_new(self, root, path, value, limit):
        raw = (json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode()
        if len(raw) > limit:
            raise ValueError('fixture receipt too large')
        with path.open('xb') as stream:
            stream.write(raw)

    def _replay(self, root, path, expected):
        if self.sources[path] != expected:
            raise ValueError('fixture source identity differs')
        self.replay_reads += 1
        return deepcopy(self.replay)

    def _trace(self, root, path, expected):
        if self.sources[path] != expected:
            raise ValueError('fixture trace identity differs')
        self.trace_starts += 1
        for trace in self.traces:
            yield deepcopy(trace)
        if self.tail_failure:
            raise ValueError('synthetic same-handle tail mismatch')
        self.trace_ends += 1


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get('AI_GAME_TEST_SCRATCH')
        if not value:
            raise unittest.SkipTest('Run inside the approved owned scratch context')
        cls.scratch = Path(value)
        if not cls.scratch.is_absolute() or not cls.scratch.is_dir():
            raise ValueError('Invalid owned synthetic test directory')

    def setUp(self):
        self.out = self.scratch/self._testMethodName
        self.out.mkdir(exist_ok=False)
        self.root = self.scratch/'synthetic-game-root-not-created'
        self.source = contract.sources()[1]  # Correct accounting for player 1.
        self.replay, self.traces = source_fixture(side=1)
        for trace in self.traces:
            trace['inference_seconds'] = 9.5  # Historical timing differs from reloaded actor timing.
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
                rows[(cohort, plan['game'])] = dict(plan, initial_banks=[10, 10],
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
        self.context = SimpleNamespace(manifests=manifests, rows=rows)
        self.replay['info']['seed'] = rows[(self.source['cohort'], self.source['game'])]['seed']
        self.dm = MemorySourceDriver(self.replay, self.traces, hashes)
        self.modules = dict(driver=self.dm, acceptance=SimpleNamespace(_json=lambda raw, limit: json.loads(raw)),
            fresh=SimpleNamespace(GAME=self.root, STAGES=dict(C01='C01', C02='C02')),
            worker=SimpleNamespace(WORKER_RELATIVE='learning/task_anchor_once_v1/games.py'))
        record = self.traces[240]['records'][0]
        self.selection = dict(target=dict(side=1, step=240, ordinal=0,
            input_sha256=ordered_hash(record['inputs']), original_choice='stay-water',
            harvest_choice='local-harvest'), counts=dict(trace_lines=719), reason=None, eligibility={})
        self.actor_arguments = []
        self.engine = FakeFactory()
        self.runtime = core.Runtime(self.engine, self.engine.identity, observed, stable, lambda actor: None)
        self.addCleanup(patch.stopall)

        def actor(*args, **kwargs):
            self.actor_arguments.append((args, kwargs))
            return FakeActor()

        patch.object(worker, '_bounded_actor', side_effect=actor).start()
        patch.object(worker, '_load_rival', side_effect=lambda _path: FakeRival()).start()
        patch.object(core, 'official_runtime', return_value=self.runtime).start()

    def execute(self, selection=None):
        return worker.execute_case(self.modules, self.context, self.source,
            self.selection if selection is None else selection, self.out, lambda: None)

    def verify(self, selection=None):
        return worker.verify_case(self.modules, self.context, self.source,
            self.selection if selection is None else selection, self.out, lambda: None)

    def test_success_uses_baseline_weights_streams_to_eof_and_readonly_verifies_side_one_cash(self):
        executed = self.execute()
        document = executed['document']
        self.assertEqual(self.dm.replay_reads, 1)
        self.assertEqual((self.dm.trace_starts, self.dm.trace_ends), (2, 2))
        panel = self.context.manifests['C01'].panel
        for args, kwargs in self.actor_arguments:
            self.assertEqual(args, (self.root/panel.crop[0], panel.crop[1],
                                    self.root/panel.models[0][0], panel.models[0][1]))
            self.assertIsNone(kwargs['trace'])
            self.assertEqual(kwargs['limits'], core.GAME_LIMITS)
        self.assertEqual(document['row']['baseline']['end'], dict(own=10, opponent=10))
        self.assertEqual(document['row']['intervention']['end'], dict(own=11, opponent=10))
        self.assertEqual(document['meter']['calls']['official_step_returns'], 96)
        with gzip.open(self.out/worker.EVENT_FILE, 'rt', encoding='utf-8') as stream:
            next(stream)  # Private arm_start.
            prefix = json.loads(next(stream))
        self.assertEqual(prefix['stable_trace_sha256'], ordered_hash(stable(self.traces[0])))
        self.assertNotEqual(prefix['trace_raw_ordered_sha256'], ordered_hash(self.traces[0]))
        # Parent receipts are allowed beside the two Worker-owned files.
        (self.out/'accepted.local.json').write_text('{}', encoding='utf-8')
        verified = self.verify()
        self.assertEqual(verified['receipt'], executed['receipt'])
        self.assertEqual(verified['document']['row'], document['row'])
        self.assertEqual(len(self.actor_arguments), 2)
        self.assertEqual(self.dm.replay_reads, 1)
        self.assertEqual((self.dm.trace_starts, self.dm.trace_ends), (3, 3))

    def test_missing_has_no_game_or_model_work_and_no_zero_cash_label(self):
        missing = dict(target=None, counts=dict(trace_lines=719), eligibility=None, reason='synthetic missing')
        result = self.execute(missing)
        self.assertEqual(result['document']['row']['status'], 'no_eligible_target')
        self.assertNotIn('baseline', result['document']['row'])
        self.assertEqual(self.actor_arguments, [])
        self.assertEqual(self.engine.calls, [])
        self.assertEqual((self.dm.replay_reads, self.dm.trace_starts), (0, 0))
        self.assertEqual(self.verify(missing)['receipt'], result['receipt'])

    def test_tail_hash_failure_prevents_intervention_and_preserves_failure_meter(self):
        self.dm.tail_failure = True
        with self.assertRaisesRegex(ValueError, 'tail mismatch'):
            self.execute()
        document = json.loads((self.out/worker.CASE_FILE).read_text(encoding='utf-8'))
        self.assertFalse(document['passed'])
        self.assertEqual(document['error_kind'], 'invalid')
        self.assertNotIn('row', document)
        self.assertEqual(document['meter']['calls']['official_step_returns'], 48)
        self.assertEqual(len(self.actor_arguments), 1)
        self.assertEqual(document['trace_scans'], [dict(lines=719, complete=False, closed=True)])

    def test_cash_tamper_is_rejected_against_owned_endpoint_frames(self):
        self.execute()
        path = self.out/worker.CASE_FILE
        document = json.loads(path.read_text(encoding='utf-8'))
        document['row']['intervention']['end']['own'] += 100
        path.write_text(json.dumps(document), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.verify()
        self.assertEqual(len(self.actor_arguments), 2)

    def test_compressed_budget_failure_keeps_reserved_failure_receipt(self):
        with patch.object(contract, 'LIMITS', dict(contract.LIMITS,
                          per_case_output_bytes=worker.CASE_RESERVE+64)):
            with self.assertRaises(OSError):
                self.execute()
        document = json.loads((self.out/worker.CASE_FILE).read_text(encoding='utf-8'))
        self.assertFalse(document['passed'])
        self.assertEqual(document['error_kind'], 'output_budget')
        self.assertLessEqual(document['output_budget']['used_bytes'], 64)
        self.assertNotIn('row', document)

    def test_emit_error_and_receipt_error_do_not_replace_primary(self):
        original = worker.EventSink.__call__

        def failed(sink, event):
            original(sink, event)
            if event.get('scope') == 'suffix':
                raise MemoryError('synthetic emit failure')

        with patch.object(worker.EventSink, '__call__', failed), \
                patch.object(worker, '_write_case', side_effect=OSError('synthetic receipt failure')):
            with self.assertRaises(MemoryError):
                self.execute()
        self.assertEqual(len(self.actor_arguments), 1)

    def test_event_limit_is_encoded_bytes_before_the_write(self):
        raw = io.BytesIO()
        sink = worker.EventSink(raw, lambda: None)
        with patch.object(worker, 'MAX_LINE_BYTES', 128):
            with self.assertRaises(ValueError):
                sink(dict(kind='arm_start', arm='control', frame={'text': '\u4e2d'*128}))
        self.assertEqual(raw.getvalue(), b'')
        self.assertEqual(sink.events, 0)


if __name__ == '__main__':
    unittest.main()
