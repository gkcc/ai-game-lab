"""Synthetic single-cohort checks: no matches, model files, GPU, or writes."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest import mock

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_campaign_v1.test_acceptance import supervision
from learning.task_pair_dev_campaign_v1.test_campaign import limits as base_limits
from learning.task_pair_dev_campaign_v1.test_manifest import inputs
from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_historical_compare_v1.test_runner import bundle_for

from . import coordinator


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'artifacts' / 'task-pair-fresh-multi-v1-synthetic'


def limits():
    value = base_limits()
    value.update(max_games=16, max_replays=16, max_comparisons=8, max_attempts=1)
    value['total_output_bytes'] = (16 * value['per_game_reserve_bytes']
                                   + 8 * coordinator.MAX_RECEIPT_BYTES
                                   + coordinator.driver_module.MAX_SUMMARY + 1024)
    return value


class FakeDriver:
    def __init__(self, manifest, check, failure=None):
        self.manifest, self.check = manifest, check
        self.root, self.stage = ROOT, STAGE
        self.claimed, self.failure_kind, self.size = False, failure, 100
        self.launched, self.accepted, self.compared, self.failures = [], [], [], []
        self.preflights, self.claim_calls = [], 0

    def preflight(self, plan, fixed, remaining):
        self.preflights.append((plan, remaining))
        return self.failure_kind != ('initial' if plan is None else 'preflight')

    def claim(self):
        self.claim_calls += 1
        if self.claimed or self.failure_kind == 'claim':
            return False
        self.claimed = True
        return True

    def launch(self, plan, _limits):
        self.launched.append(dict(plan))
        if self.failure_kind == ('raise', len(self.launched)):
            raise ValueError('PRIVATE_SYNTHETIC_VALUE')
        if self.failure_kind == ('supervision', len(self.launched)):
            return dict(supervision(), job_members_zero=False)
        return supervision()

    def record_supervision(self, _plan, _result):
        return self.failure_kind != ('record_supervision', len(self.launched))

    def read_bundle(self, plan):
        head = (self.failure_kind[0] if isinstance(self.failure_kind, tuple)
                and self.failure_kind[1] == len(self.launched)
                and self.failure_kind[0] in ('task_forwards', 'crop_forwards') else None)
        bundle = bundle_for(self.manifest, plan, head)
        if self.failure_kind == ('missing', len(self.launched)):
            del bundle['metrics']
        if self.failure_kind == ('corrupt', len(self.launched)):
            bundle['metrics'] += b' '
        return bundle

    def verify_files(self, _plan, _row):
        return self.failure_kind != ('files', len(self.launched))

    def record_acceptance(self, plan, _bundle):
        if self.failure_kind == ('record_acceptance', len(self.launched)):
            return False
        self.accepted.append(plan['game'])
        if self.failure_kind == ('reserve', len(self.launched)):
            fixed = limits()
            self.size = (fixed['total_output_bytes']
                         - (16 - len(self.launched)) * fixed['per_game_reserve_bytes']
                         - 8 * coordinator.MAX_RECEIPT_BYTES - coordinator.driver_module.MAX_SUMMARY + 1)
        return True

    def preflight_comparison(self, _fixed, _remaining):
        return self.failure_kind != 'comparison_preflight'

    def compare(self, left, right):
        self.compared.append((dict(left), dict(right)))
        if self.failure_kind == ('compare', len(self.compared)):
            raise ValueError('PRIVATE_SYNTHETIC_VALUE')
        self.size += (coordinator.MAX_RECEIPT_BYTES + 1
                      if self.failure_kind == ('large_compare', len(self.compared)) else 1)
        return dict(reference=left['arm'], candidate=right['arm'], seed=left['seed'],
                    seed_alias=left['seed_alias'], opponent=left['opponent'],
                    side=(1 - left['side'] if self.failure_kind == ('bad_compare', len(self.compared))
                          else left['side']),
                    result=dict(comparison_complete=True, matched_calls=3,
                                matched_multi_unit_opportunities=1))

    def failure(self, game, completed):
        self.failures.append((game, completed))


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.manifest = build_manifest(**inputs())
        self.check = lambda: True
        self.driver = FakeDriver(self.manifest, self.check)
        self.coordinator = coordinator.Coordinator()
        self.authorized = []
        self.budget = [limits()['campaign_wall_seconds']]

    def authorize(self, plan):
        self.authorized.append(dict(plan))
        return True

    def run_case(self, **overrides):
        values = dict(manifest=self.manifest, limits=limits(), driver=self.driver,
                      check=self.check, authorize_game=self.authorize, claim=self.driver.claim,
                      remaining=lambda: self.budget[0])
        values.update(overrides)
        with mock.patch.object(coordinator, '_verify_assets', return_value=None), \
             mock.patch.object(coordinator, '_stage_size', side_effect=lambda _driver: self.driver.size):
            return self.coordinator.run(**values)

    def test_full_cohort_uses_real_acceptance_and_excludes_auxiliary(self):
        fixed = limits()
        fixed['per_game_reserve_bytes'] = 160 * 1024**2
        fixed['total_output_bytes'] = 3 * 1024**3
        result = self.run_case(limits=fixed)
        expected = {(alias, opponent, side, arm) for alias in ('S02', 'S03')
                    for opponent in ('market', 'round2') for side in (0, 1)
                    for arm in ('baseline', 'control')}
        def coords(plan):
            return plan['seed_alias'], plan['opponent'], plan['side'], plan['policy']
        self.assertEqual({coords(p) for p in self.driver.launched}, expected)
        self.assertEqual(self.driver.launched, self.authorized)
        self.assertEqual((len(self.driver.launched), len(self.driver.accepted),
                          len(self.driver.compared)), (16, 16, 8))
        self.assertEqual((len(result['rows']), len(result['comparisons'])), (16, 8))
        self.assertEqual(set(result), {'rows', 'comparisons'})
        self.assertTrue(all(left['arm'] == 'baseline' and right['arm'] == 'control'
                            and all(left[k] == right[k] for k in
                                    ('seed_alias', 'seed', 'opponent', 'side'))
                            for left, right in self.driver.compared))
        self.assertEqual(len(self.driver.preflights), 17)
        self.assertEqual(self.driver.claim_calls, 1)

    def test_missing_or_duplicate_panel_coordinates_stop_before_claim(self):
        original = self.manifest.panel.plans()
        tampered = list(deepcopy(original)); tampered[0]['policy'] = 'auxiliary'
        for replacement in (original[:-1], (*original[:-1], original[0]), tuple(tampered)):
            self.coordinator = coordinator.Coordinator()
            self.driver = FakeDriver(self.manifest, self.check)
            with self.subTest(length=len(replacement)), mock.patch.object(Panel, 'plans', return_value=replacement):
                with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
                    self.run_case()
                self.assertEqual(self.driver.launched, [])
                self.assertEqual(self.driver.claim_calls, 0)

    def test_bad_game_stops_without_retry_or_comparison(self):
        for failure in (('supervision', 3), ('record_supervision', 3), ('raise', 3),
                        ('missing', 3), ('corrupt', 3), ('files', 3), ('record_acceptance', 3),
                        ('task_forwards', 3), ('crop_forwards', 3)):
            with self.subTest(kind=failure[0]):
                self.coordinator = coordinator.Coordinator()
                self.driver = FakeDriver(self.manifest, self.check, failure)
                with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$') as caught:
                    self.run_case()
                self.assertNotIn('PRIVATE_SYNTHETIC_VALUE', str(caught.exception))
                self.assertEqual((len(self.driver.launched), len(self.driver.accepted),
                                  len(self.driver.compared)), (3, 2, 0))
                self.assertEqual(self.driver.failures[0][1], 2)
                with self.assertRaisesRegex(RuntimeError, '^Cohort attempt already used$'):
                    self.run_case()
                self.assertEqual(len(self.driver.launched), 3)

    def test_bad_comparison_stops_after_first_bad_one(self):
        for failure in (('compare', 2), ('bad_compare', 2)):
            with self.subTest(kind=failure[0]):
                self.coordinator = coordinator.Coordinator()
                self.driver = FakeDriver(self.manifest, self.check, failure)
                with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
                    self.run_case()
                self.assertEqual((len(self.driver.launched), len(self.driver.accepted),
                                  len(self.driver.compared)), (16, 16, 2))
                self.assertEqual(self.driver.failures[0][1], 16)

    def test_invalid_scope_deadline_and_claim_fail_closed(self):
        bad_values = []
        for key, value in (('max_games', 17), ('max_replays', 15), ('max_comparisons', 9),
                           ('max_attempts', 2), ('total_output_bytes', 1)):
            bad = deepcopy(limits()); bad[key] = value; bad_values.append(bad)
        for bad in bad_values:
            self.coordinator = coordinator.Coordinator()
            self.driver = FakeDriver(self.manifest, self.check)
            with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
                self.run_case(limits=bad)
            self.assertEqual(self.driver.claim_calls, 0)
        for left in (0, True, float('nan'), limits()['campaign_wall_seconds'] + 1):
            self.coordinator = coordinator.Coordinator()
            self.driver = FakeDriver(self.manifest, self.check)
            with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
                self.run_case(remaining=lambda: left)
            self.assertEqual(self.driver.claim_calls, 0)
        self.coordinator = coordinator.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case(claim=lambda: True)
        self.assertEqual(self.driver.launched, [])

    def test_same_external_budget_cannot_reset_between_cohorts(self):
        self.run_case()
        self.budget[0] = limits()['job']['wall_seconds'] + limits()['job']['cleanup_seconds']
        self.coordinator = coordinator.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case()
        self.assertEqual(self.driver.launched, [])
        self.assertEqual(self.driver.claim_calls, 0)

    def test_storage_reserves_16_games_and_stops_when_remaining_space_is_insufficient(self):
        fixed = limits()
        fixed['per_game_reserve_bytes'] = 160 * 1024**2
        fixed['total_output_bytes'] = 3 * 1024**3
        self.assertEqual(coordinator.limits_snapshot(fixed)['max_games'], 16)
        self.driver.failure_kind = ('reserve', 2)
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case()
        self.assertEqual((len(self.driver.launched), len(self.driver.accepted),
                          len(self.driver.compared)), (2, 2, 0))
        self.assertEqual(self.driver.failures[0][1], 2)

        self.coordinator = coordinator.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check, ('large_compare', 1))
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case()
        self.assertEqual((len(self.driver.launched), len(self.driver.compared)), (16, 1))

    def test_claim_failure_never_adopts_an_already_owned_stage(self):
        self.driver.claimed = True
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case()
        self.assertEqual(self.driver.failures, [])
        self.assertEqual(self.driver.claim_calls, 0)

        self.coordinator = coordinator.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        def failed_after_claim():
            self.driver.claim()
            raise ValueError('PRIVATE_SYNTHETIC_VALUE')
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case(claim=failed_after_claim)
        self.assertEqual(self.driver.failures[0][1], 0)
        self.assertEqual(self.driver.launched, [])

    def test_nonmonotonic_external_deadline_stops_before_launch(self):
        values = iter((100.0, 101.0))
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case(remaining=lambda: next(values))
        self.assertEqual(self.driver.claim_calls, 1)
        self.assertEqual(self.driver.launched, [])
        self.assertEqual(self.driver.failures[0][1], 0)

    def test_expiry_after_authorization_prevents_launch(self):
        def authorize(plan):
            self.authorized.append(dict(plan))
            self.budget[0] = limits()['job']['wall_seconds'] + limits()['job']['cleanup_seconds']
            return True
        with self.assertRaisesRegex(RuntimeError, '^Cohort stopped$'):
            self.run_case(authorize_game=authorize)
        self.assertEqual(len(self.authorized), 1)
        self.assertEqual(self.driver.launched, [])
        self.assertEqual(self.driver.failures[0][1], 0)

    def test_cancellation_survives_failure_receipt_error(self):
        signal = KeyboardInterrupt()
        with mock.patch.object(self.driver, 'launch', side_effect=signal), \
             mock.patch.object(self.driver, 'failure', side_effect=ValueError('PRIVATE_SYNTHETIC_VALUE')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)

        self.coordinator = coordinator.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        with mock.patch.object(self.driver, 'launch', side_effect=ValueError('PRIVATE_SYNTHETIC_VALUE')), \
             mock.patch.object(self.driver, 'failure', side_effect=signal):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)


if __name__ == '__main__':
    unittest.main()
