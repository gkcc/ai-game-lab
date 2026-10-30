"""In-memory driver tests; no temporary directories or real game artifacts."""
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from . import driver
from .manifest import build_manifest
from .test_manifest import inputs
from .test_campaign import limits
from .test_acceptance import supervision
from learning.task_pair_dev_games_v1.test_capture import compressed, turns


ROOT = Path(__file__).resolve().parents[2]


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.manifest = build_manifest(**inputs())
        self.stage = ROOT / 'artifacts' / 'synthetic-not-created'
        self.plan = self.manifest.panel.plans()[0]
        self.instance = driver.LocalDriver(manifest=self.manifest, game_root=ROOT, stage=self.stage,
            commit='1'*40, command_for=lambda alias: ['synthetic-executable','--game',alias],
            check=lambda: True, available=lambda: 8*1024**3,
            disk_usage=lambda _: SimpleNamespace(free=20*1024**3),
            supervisor=lambda *_: supervision())

    def test_preflight_uses_owned_size_and_registered_reserve(self):
        with mock.patch.object(Path, 'exists', return_value=False), \
             mock.patch.object(Path, 'is_symlink', return_value=False):
            self.assertTrue(self.instance.preflight(None, limits(), 100))
        self.instance.claimed = True
        with mock.patch.object(Path, 'is_dir', return_value=True), \
             mock.patch.object(driver, '_owned_size', return_value=1):
            self.assertTrue(self.instance.preflight(self.plan, limits(), 100))
            self.assertTrue(self.instance.preflight_comparison(limits(), 100))
        self.instance.available = lambda: 0
        with mock.patch.object(Path, 'exists', return_value=False), \
             mock.patch.object(Path, 'is_symlink', return_value=False):
            self.assertFalse(self.instance.preflight(None, limits(), 100))

    def test_existing_stage_is_never_scanned_or_adopted(self):
        with mock.patch.object(Path, 'exists', return_value=True), \
             mock.patch.object(driver, '_owned_size') as scan:
            self.assertFalse(self.instance.preflight(None, limits(), 100))
            scan.assert_not_called()
        self.assertFalse(self.instance.claimed)

    def test_launch_passes_alias_not_seed_to_registered_command(self):
        observed = []
        self.instance.supervisor = lambda command, cwd, job: observed.append((command, cwd, job)) or supervision()
        result = self.instance.launch(self.plan, limits()['job'])
        self.assertTrue(result['passed'])
        self.assertEqual(observed[0][0][-2:], ['--game', self.plan['game']])
        self.assertNotIn(str(self.plan['seed']), observed[0][0])
        self.instance.command_for = lambda _: ['synthetic-executable','--game','wrong']
        with self.assertRaises(ValueError): self.instance.launch(self.plan, limits()['job'])

    def test_stream_trace_consumes_exactly_719_and_checks_sha(self):
        data = compressed(turns())
        with mock.patch.object(driver, '_no_links'), \
             mock.patch.object(Path, 'open', return_value=io.BytesIO(data)):
            values = list(driver._trace(self.stage, self.stage / 'synthetic', hashlib.sha256(data).hexdigest()))
        self.assertEqual([row['step'] for row in values], list(range(719)))
        with mock.patch.object(driver, '_no_links'), \
             mock.patch.object(Path, 'open', return_value=io.BytesIO(data)):
            with self.assertRaises(ValueError):
                list(driver._trace(self.stage, self.stage / 'synthetic', '0'*64))

    def test_replay_loader_checks_compressed_identity_before_and_after(self):
        data = gzip.compress(json.dumps({'steps': [], 'info': {}}).encode())
        with mock.patch.object(driver, '_hash', return_value='a'*64) as digest, \
             mock.patch.object(Path, 'open', return_value=io.BytesIO(data)):
            value = driver._replay(self.stage, self.stage / 'synthetic', 'a'*64)
        self.assertEqual(value['steps'], [])
        self.assertEqual(digest.call_count, 2)

    def test_supervision_and_acceptance_are_separate_private_receipts(self):
        bundle = {key: b'synthetic' for key in driver.acceptance.FILES}
        written = []
        with mock.patch.object(driver, '_write_new', side_effect=lambda _r, path, value, **_: written.append((path, value))), \
             mock.patch.object(driver, '_hash', return_value='a'*64):
            self.assertTrue(self.instance.record_supervision(self.plan, supervision()))
            self.assertTrue(self.instance.record_acceptance(self.plan, bundle))
        self.assertTrue(written[0][0].name.endswith('.supervision.local.json'))
        self.assertTrue(written[1][0].name.endswith('.acceptance.local.json'))
        self.assertEqual(set(written[1][1]['receipt_sha256']), set(bundle))

    def test_final_result_references_each_owned_supervision_and_acceptance(self):
        rows = [dict(game=p['game']) for p in self.manifest.panel.plans()]
        with mock.patch.object(driver, '_hash', return_value='a'*64), \
             mock.patch.object(driver, '_write_new') as write:
            self.assertTrue(self.instance.complete(rows, {'baseline_to_control': {}, 'control_to_auxiliary': {}}))
        result = write.call_args.args[2]
        self.assertEqual(len(result['receipts']), 24)
        self.assertEqual(len({v['game'] for v in result['receipts']}), 24)
        self.assertEqual(len(result['comparison_receipts']), 16)
        self.assertEqual(len({(v['reference_game'], v['candidate_game'])
                              for v in result['comparison_receipts']}), 16)
        self.assertFalse(result['independent_improvement'])

    def test_pair_comparison_persists_bound_sources_and_rejects_wrong_roles(self):
        plans = self.manifest.panel.plans()
        def row(policy):
            plan = next(p for p in plans if p['seed_alias'] == 'S02' and p['opponent'] == 'market'
                        and p['side'] == 0 and p['policy'] == policy)
            return {**{k: plan[k] for k in ('game', 'seed', 'seed_alias', 'opponent', 'side')},
                    'arm': policy, 'replay_sha256': 'a'*64, 'trace_sha256': 'b'*64}
        left, right = row('baseline'), row('control')
        summary = {'reference': 'baseline', 'candidate': 'control', 'result': {'comparison_complete': True}}
        with mock.patch.object(driver, '_replay', return_value={}), \
             mock.patch.object(driver, '_trace', side_effect=lambda *_: (v for v in ())), \
             mock.patch.object(driver, 'compare_replays', return_value=summary), \
             mock.patch.object(driver, '_write_new') as write:
            self.assertEqual(self.instance.compare(left, right), summary)
            receipt = write.call_args.args[2]
            self.assertEqual((receipt['reference_game'], receipt['candidate_game']),
                             (left['game'], right['game']))
            self.assertEqual(receipt['reference_replay_sha256'], left['replay_sha256'])
            self.assertEqual(receipt['candidate_trace_sha256'], right['trace_sha256'])
            self.assertEqual(receipt['comparison'], summary)
            bad = dict(right, game='../unexpected')
            with self.assertRaises(ValueError):
                self.instance.compare(left, bad)
            with self.assertRaises(ValueError):
                self.instance.compare(right, left)


if __name__ == '__main__':
    unittest.main()
