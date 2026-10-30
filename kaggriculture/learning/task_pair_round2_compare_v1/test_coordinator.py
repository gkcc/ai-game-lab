"""Synthetic coordinator tests; no real files, games, models or engine."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest import mock

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_campaign_v1.test_manifest import inputs
from learning.task_pair_historical_compare_v1.test_runner import FakeDriver, limits
from . import coordinator


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'artifacts' / 'task-pair-round2-compare-v1-synthetic'


def manifest_and_plans():
    manifest = build_manifest(**inputs())
    plans = [p for alias in ('S02', 'S03') for side in (0, 1)
             for arm in ('baseline', 'control') for p in manifest.panel.plans()
             if (p['seed_alias'], p['opponent'], p['side'], p['policy'])
             == (alias, 'round2', side, arm)]
    return manifest, plans


class Round2Driver(FakeDriver):
    def __init__(self, manifest, check, failure=None):
        super().__init__(manifest, check, failure)
        self.stage = STAGE

    def compare(self, left, right):
        result = super().compare(left, right)
        result['opponent'] = 'market' if self.fail == ('wrong_opponent', len(self.compared)) else 'round2'
        return result


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.plans = manifest_and_plans()
        self.check = lambda: True
        self.driver = Round2Driver(self.manifest, self.check)
        self.coordinator = coordinator.Coordinator()
        self.authorized, self.saved, self.failures = [], [], []

    def authorize(self, plan):
        self.authorized.append(dict(plan))
        return True

    def save(self, value):
        self.saved.append(value)
        self.driver.size += 100
        return True

    def run_case(self, **changes):
        values = dict(manifest=self.manifest, plans=self.plans, limits=limits(), driver=self.driver,
                      check=self.check, authorize_game=self.authorize, save_result=self.save,
                      on_failure=self.failures.append, clock=lambda: 0.)
        values.update(changes)
        assets = {ROOT / relative: digest for relative, digest in (
            *self.manifest.panel.models[:2], self.manifest.panel.crop, self.manifest.opponent('round2'),
            (coordinator.worker.WORKER_RELATIVE, self.manifest.panel.worker_sha))}
        with mock.patch.object(coordinator.driver_module, '_owned_size', side_effect=lambda _p: self.driver.size), \
             mock.patch.object(coordinator.driver_module, '_hash', side_effect=lambda _r, p, _m: assets.get(p)):
            return self.coordinator.run(**values)

    def test_eight_round2_games_four_pairs_and_safe_summary(self):
        result = self.run_case()
        self.assertEqual((len(self.driver.launched), len(self.driver.accepted), len(self.driver.compared)), (8, 8, 4))
        self.assertEqual([p['game'] for p in self.authorized], self.driver.launched)
        self.assertTrue(all(p['opponent'] == 'round2' for p in self.authorized))
        self.assertFalse(any('-market-' in game for game in self.driver.launched))
        self.assertEqual([(p['seed_alias'], p['side'], p['policy']) for p in self.authorized],
                         [(s, side, arm) for s in ('S02', 'S03') for side in (0, 1)
                          for arm in ('baseline', 'control')])
        self.assertEqual(len(self.saved), 1)
        payload = self.saved[0]
        self.assertEqual((len(payload['rows']), len(payload['comparisons'])), (8, 4))
        self.assertEqual((payload['paired']['expected_games'], payload['paired']['observed_games']), (8, 8))
        self.assertEqual(payload['paired']['seed_clusters'], 2)
        self.assertEqual(payload['paired']['statistical_significance'], 'unknown')
        self.assertFalse(result['promotion'])
        self.assertFalse(result['independent_improvement'])
        self.assertFalse(result['observed_outer_exit'])
        for secret in (str(self.manifest.panel.seeds[0]), str(ROOT), self.manifest.panel.crop[1]):
            self.assertNotIn(secret, json.dumps(result))
        self.assertEqual(self.failures, [])

    def test_invalid_plans_cannot_run_market_or_swap_roles(self):
        variants = []
        for key, value in (('opponent', 'market'), ('side', True), ('policy', 'auxiliary'), ('seed', 3)):
            bad = deepcopy(self.plans); bad[0][key] = value; variants.append(bad)
        variants.extend((self.plans[:7], list(reversed(self.plans))))
        for bad in variants:
            with self.subTest(kind=len(bad)):
                self.coordinator = coordinator.Coordinator()
                self.driver = Round2Driver(self.manifest, self.check)
                with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
                    self.run_case(plans=bad)
                self.assertFalse(self.driver.claimed)
                self.assertEqual(self.driver.launched, [])
        self.coordinator = coordinator.Coordinator()
        bad_limits = limits(); bad_limits['max_games'] = 9
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_case(limits=bad_limits)
        self.assertFalse(self.driver.claimed)
        self.assertEqual(self.driver.launched, [])

    def test_worker_and_acceptance_failure_stop_without_retry(self):
        for failure in (('supervision', 3), ('receipt', 3), ('files', 3),
                        ('task_forwards', 3), ('crop_forwards', 3)):
            with self.subTest(kind=failure[0]):
                self.coordinator = coordinator.Coordinator()
                self.driver = Round2Driver(self.manifest, self.check, failure)
                self.failures = []
                with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
                    self.run_case()
                self.assertEqual((len(self.driver.launched), len(self.driver.accepted)), (3, 2))
                self.assertEqual(self.driver.failures[0][1], 2)
                self.assertEqual(self.driver.compared, [])
                self.assertEqual(self.saved, [])
                self.assertEqual((self.failures[0]['attempted'], self.failures[0]['accepted']), (3, 2))
                self.assertIn(self.failures[0]['phase'], ('supervision', 'acceptance'))
                self.assertEqual(set(self.failures[0]), {'phase', 'category', 'attempted', 'accepted'})
                self.assertIn(self.failures[0]['phase'], coordinator.FAILURE_PHASES)
                self.assertIn(self.failures[0]['category'], coordinator.FAILURE_CATEGORIES)
                self.assertNotIn('PRIVATE_SYNTHETIC_VALUE', json.dumps(self.failures))
                with self.assertRaisesRegex(RuntimeError, '^Round2 comparison attempt already used$'):
                    self.run_case()
                self.assertEqual(len(self.driver.launched), 3)

    def test_bad_comparison_or_permission_never_creates_result(self):
        self.driver = Round2Driver(self.manifest, self.check, ('wrong_opponent', 2))
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_case()
        self.assertEqual((len(self.driver.launched), len(self.driver.compared)), (8, 2))
        self.assertEqual(self.saved, [])
        self.assertEqual(self.failures[0]['phase'], 'comparison')
        self.coordinator = coordinator.Coordinator()
        self.driver = Round2Driver(self.manifest, self.check)
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_case(authorize_game=lambda _plan: False)
        self.assertEqual(self.driver.launched, [])
        self.assertEqual(self.failures[-1]['phase'], 'authorization')

    def test_cancel_is_not_suppressed_by_failure_receipt(self):
        signal = KeyboardInterrupt()
        with mock.patch.object(self.driver, 'launch', side_effect=signal), \
             mock.patch.object(self.driver, 'failure', side_effect=ValueError('PRIVATE_SYNTHETIC_VALUE')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)
        self.assertEqual(self.failures[-1]['category'], 'BaseException')
        self.assertEqual(self.saved, [])


if __name__ == '__main__':
    unittest.main()
