"""In-memory regressions for consumed-role and frozen-asset bindings."""
from contextlib import ExitStack
import hashlib
import unittest
from unittest.mock import patch

from learning.task_control_once_v1 import bindings as control_binding
from learning.task_control_once_v1.test_bindings import manifest as synthetic_manifest
from learning.task_control_once_v1.test_bindings_integration import Fixture
from learning.task_pair_dev_campaign_v1 import history
from . import bindings


def build_fixture(fixture, check=lambda: True):
    with ExitStack() as stack:
        for module in (control_binding, history, bindings):
            stack.enter_context(patch.object(module, 'ModuleType', fixture.checker_module))
        stack.enter_context(patch.object(history, '_safe_relative_file', fixture.asset_path))
        stack.enter_context(patch.object(history, '_file_digest', fixture.asset_digest))
        stack.enter_context(patch.object(control_binding, 'TEACHER_SHA', fixture.teacher_sha))
        return bindings.build(modules=fixture.modules, bootstrap=fixture.bootstrap,
            identity=fixture.identity, reg=fixture.reg, old_reg=fixture.cfg, freeze=fixture.freeze,
            game_root=fixture.root, repo=fixture.repo, status_path=fixture.root / 'virtual-status.json',
            prior_binding=fixture.prior, check=check)


class HistoricalBindingsTests(unittest.TestCase):
    def test_real_receipt_checker_binds_both_roles_and_market_assets(self):
        fixture = Fixture()
        manifest, previous, roles = build_fixture(fixture)
        self.assertEqual(roles, {'S01': 1000, 'S04': 1003})
        self.assertEqual(manifest.panel.seeds, (roles['S01'], roles['S04']))
        self.assertEqual(previous.panel.seeds, (1001, 1002))
        self.assertEqual(manifest.opponent('market'), previous.opponent('market'))
        self.assertNotEqual(manifest.opponent('market')[1], fixture.teacher_sha)
        self.assertEqual(manifest.panel.models, previous.panel.models)
        self.assertEqual(manifest.panel.crop, previous.panel.crop)
        self.assertTrue(bindings.verify_mapping(manifest, previous, roles))
        selected = [p for p in manifest.panel.plans() if p['opponent'] == 'market'
                    and p['policy'] in ('baseline', 'control')]
        self.assertEqual(len(selected), 8)

    def test_missing_s04_market_acceptance_is_rejected(self):
        fixture = Fixture()
        del fixture.files['panel/job-acceptance/holdout-S04-market-0-merged.json']
        # The existing S01/teacher binding alone remains valid; the new S04
        # market evidence is the missing condition.
        fixture.build()
        with self.assertRaises(ValueError):
            build_fixture(fixture)

    def test_wrong_market_source_or_external_check_is_rejected(self):
        fixture = Fixture()
        fixture.model_files['artifacts/historical/sources/market.py'] = b'altered synthetic market'
        with self.assertRaises(ValueError):
            build_fixture(fixture)
        with self.assertRaises(ValueError):
            build_fixture(Fixture(), check=lambda: False)

    def test_panel_consistency_does_not_authorize_wrong_roles_or_assets(self):
        previous = synthetic_manifest(seeds={'S02': 11, 'S03': 12})
        correct = synthetic_manifest(seeds={'S02': 10, 'S03': 13})
        roles = {'S01': 10, 'S04': 13}
        self.assertTrue(bindings.verify_mapping(correct, previous, roles))
        cases = (
            (synthetic_manifest(seeds={'S02': 13, 'S03': 10}), roles),
            (synthetic_manifest(seeds={'S02': 10, 'S03': 13}, teacher=True), roles),
            (synthetic_manifest(seeds={'S02': 10, 'S03': 13}, control='9'), roles),
            (synthetic_manifest(seeds={'S02': 11, 'S03': 13}), {'S01': 11, 'S04': 13}),
            (synthetic_manifest(seeds={'S02': 10, 'S03': 12}), {'S01': 10, 'S04': 12}),
            (correct, {'S01': True, 'S04': 13}),
        )
        for manifest, bad_roles in cases:
            with self.subTest(), self.assertRaises(ValueError):
                bindings.verify_mapping(manifest, previous, bad_roles)

    def test_one_inconsistent_historical_market_row_is_rejected(self):
        source = b'''SEED_ALIASES=(('S02',1),('S03',2))
OPPONENTS=('round2',)
ARMS=('merged','unweighted','teacher_task')
def checked_rows(panel,status):
 result={}
 for alias,index in SEED_ALIASES:
  for opponent in OPPONENTS:
   for side in (0,1):
    for arm in ARMS:
     seed=1000 if alias=='S01' else 1003
     if alias=='S04' and side==1 and arm=='unweighted': seed=1002
     result[(alias,opponent,side,arm)]={'row':dict(seed_alias=alias,opponent=opponent,side=side,policy=arm,seed=seed)}
 return result
'''
        with self.assertRaises(ValueError):
            bindings._market_roles(source=source, expected_sha=hashlib.sha256(source).hexdigest(),
                source_path='synthetic-checker.py', historical='synthetic-panel',
                status_path='synthetic-status', check=lambda: True)


if __name__ == '__main__':
    unittest.main()
