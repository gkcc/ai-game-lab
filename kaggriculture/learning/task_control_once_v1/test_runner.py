"""Synthetic one-game coordination checks; no games, models or files."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from contextlib import contextmanager, redirect_stdout
from unittest import mock

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_campaign_v1.test_acceptance import supervision
from learning.task_pair_dev_campaign_v1.test_campaign import limits as campaign_limits
from learning.task_pair_dev_campaign_v1.test_manifest import inputs
from learning.task_pair_toy_smoke_v1.test_runner import bundle_for
from . import runner


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'artifacts' / 'task-control-once-v1-synthetic'


def manifest_and_plan():
    manifest = build_manifest(**inputs())
    plan = next(p for p in manifest.panel.plans()
                if (p['seed_alias'], p['opponent'], p['side'], p['policy'])
                == ('S02', 'market', 0, 'control'))
    return manifest, plan


def limits():
    value = campaign_limits()
    value.update(max_games=1, max_replays=1, max_comparisons=0, max_attempts=1)
    value['total_output_bytes'] = value['per_game_reserve_bytes'] + 16 * 1024**2
    return value


class FakeDriver:
    def __init__(self, manifest, check, fail=None):
        self.manifest, self.check, self.root, self.stage = manifest, check, ROOT, STAGE
        self.fail, self.claimed, self.size = fail, False, 100
        self.launched, self.accepted, self.failures, self.compared = 0, 0, [], 0

    def preflight(self, plan, _limits, _remaining):
        return not (self.fail == 'initial' or (plan is not None and self.fail == 'preflight'))

    def claim(self):
        if self.fail == 'claim':
            return False
        if self.claimed:
            raise ValueError('Already claimed')
        self.claimed = True
        return True

    def launch(self, plan, _job):
        self.launched += 1
        if self.fail == 'launch':
            raise ValueError('Synthetic private failure')
        return dict(supervision(), job_members_zero=False) if self.fail == 'job' else supervision()

    def record_supervision(self, _plan, _result):
        return self.fail != 'supervision'

    def read_bundle(self, plan):
        bundle = bundle_for(self.manifest, plan)
        if self.fail == 'receipt':
            bundle['metrics'] += b' '
        if self.fail in ('no_crop', 'no_task'):
            audit = json.loads(bundle['audit'])
            key = 'crop_forwards' if self.fail == 'no_crop' else 'task_forwards'
            audit['primary_model_counts'][key] = audit['live_model_counts'][key] = 0
            bundle['audit'] = json.dumps(audit, separators=(',', ':')).encode()
            complete = json.loads(bundle['complete'])
            complete['certificate_sha256'] = hashlib.sha256(bundle['audit']).hexdigest()
            bundle['complete'] = json.dumps(complete, separators=(',', ':')).encode()
        return bundle

    def verify_files(self, _plan, _row):
        return self.fail != 'identity'

    def record_acceptance(self, _plan, _bundle):
        if self.fail == 'acceptance':
            return False
        self.accepted += 1
        return True

    def compare(self, *_args):
        self.compared += 1
        raise AssertionError('A control game must never compare')

    def failure(self, game, completed):
        self.failures.append((game, completed))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.plan = manifest_and_plan()
        self.check = lambda: True
        self.driver = FakeDriver(self.manifest, self.check)
        self.coordinator = runner.Coordinator()
        self.saved = []

    def save(self, value):
        self.saved.append(value)
        self.driver.size += 100
        return True

    def run_case(self, **changes):
        values = dict(manifest=self.manifest, plan=self.plan, limits=limits(), driver=self.driver,
                      check=self.check, save_result=self.save, clock=lambda: 0.)
        values.update(changes)
        assets = {ROOT / relative: digest for relative, digest in (
            self.manifest.panel.task_model(self.plan), self.manifest.panel.crop,
            self.manifest.opponent('market'),
            (runner.worker.WORKER_RELATIVE, self.manifest.panel.worker_sha))}
        with mock.patch.object(runner.driver_module, '_owned_size', side_effect=lambda _p: self.driver.size), \
             mock.patch.object(runner.driver_module, '_hash', side_effect=lambda _r, p, _m: assets.get(p)):
            return self.coordinator.run(**values)

    def test_one_complete_job_uses_real_receipt_acceptance_and_no_comparison(self):
        result = self.run_case()
        self.assertEqual((self.driver.launched, self.driver.accepted, self.driver.compared), (1, 1, 0))
        self.assertEqual(result, self.saved[0])
        self.assertEqual((result['attempted'], result['accepted'], result['official_replay_passes']), (1, 1, 1))
        self.assertEqual((result['registered_game_limit'], result['generic_driver_claim_capacity']), (1, 24))
        self.assertFalse(result['observed_outer_exit'])
        self.assertFalse(result['diagnostic_verified'])
        self.assertFalse(result['independent_improvement'])
        self.assertFalse(result['training'])
        self.assertFalse(result['promotion'])
        for value in (str(self.plan['seed']), str(ROOT), self.manifest.panel.crop[1]):
            self.assertNotIn(value, json.dumps(result))

    def test_other_plans_or_limits_fail_before_claim_or_game(self):
        alternatives = []
        for key, value in (('seed_alias', 'S03'), ('opponent', 'round2'), ('side', 1),
                           ('policy', 'baseline'), ('seed', self.plan['seed'] + 1)):
            altered = deepcopy(self.plan); altered[key] = value; alternatives.append(('plan', altered))
        for key, value in (('max_games', 2), ('max_replays', 2), ('max_comparisons', 1),
                           ('max_attempts', 2), ('max_games', True), ('total_output_bytes', 1)):
            altered = limits(); altered[key] = value; alternatives.append(('limits', altered))
        for key, value in alternatives:
            with self.subTest(kind=key):
                self.coordinator = runner.Coordinator()
                self.driver = FakeDriver(self.manifest, self.check)
                with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
                    self.run_case(**{key: value})
                self.assertFalse(self.driver.claimed)
                self.assertEqual(self.driver.launched, 0)
                self.assertEqual(self.saved, [])

    def test_failure_after_claim_stops_without_retry_and_records_failure(self):
        for failure in ('preflight', 'launch', 'supervision', 'job', 'receipt',
                        'identity', 'no_crop', 'no_task', 'acceptance'):
            with self.subTest(kind=failure):
                self.coordinator = runner.Coordinator()
                self.driver = FakeDriver(self.manifest, self.check, fail=failure)
                with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
                    self.run_case()
                self.assertTrue(self.driver.claimed)
                self.assertLessEqual(self.driver.launched, 1)
                self.assertEqual(self.driver.accepted, 0)
                self.assertEqual(len(self.driver.failures), 1)
                self.assertEqual(self.driver.failures[0][1], 0)
                self.assertEqual(self.driver.compared, 0)
                self.assertEqual(self.saved, [])

    def test_save_failure_keeps_failure_receipt_and_never_relaunches(self):
        with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
            self.run_case(save_result=lambda _value: False)
        self.assertEqual((self.driver.launched, self.driver.accepted), (1, 1))
        self.assertEqual(self.driver.failures[0][1], 1)
        with self.assertRaisesRegex(RuntimeError, '^Single control attempt already used$'):
            self.run_case()
        self.assertEqual(self.driver.launched, 1)

    def test_single_instance_cannot_retry_success_or_initial_failure(self):
        self.run_case()
        with self.assertRaisesRegex(RuntimeError, '^Single control attempt already used$'):
            self.run_case()
        self.assertEqual(self.driver.launched, 1)
        self.coordinator = runner.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check, fail='initial')
        with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
            self.run_case()
        self.assertFalse(self.driver.claimed)
        self.assertEqual(self.driver.failures, [])  # Outer attempt marker owns pre-stage failures.
        with self.assertRaisesRegex(RuntimeError, '^Single control attempt already used$'):
            self.run_case()
        self.assertEqual(self.driver.launched, 0)

    def test_identity_or_resource_failure_stops_without_leaking_values(self):
        self.driver.stage = ROOT / 'artifacts' / 'task-pair-dev-games-old'
        with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
            self.run_case()
        self.assertFalse(self.driver.claimed)
        self.coordinator = runner.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        with mock.patch.object(runner, '_verify_assets', side_effect=ValueError('PRIVATE_SENTINEL')):
            with self.assertRaisesRegex(RuntimeError, '^Single control stopped$') as error:
                self.run_case()
        self.assertNotIn('PRIVATE_SENTINEL', str(error.exception))
        self.assertEqual(self.driver.launched, 0)

    def test_claim_resource_and_clock_failures_stop_before_game(self):
        self.driver = FakeDriver(self.manifest, self.check, fail='claim')
        with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
            self.run_case()
        self.assertEqual((self.driver.launched, self.driver.accepted), (0, 0))
        self.coordinator = runner.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        self.driver.size = limits()['total_output_bytes']
        with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
            self.run_case()
        self.assertEqual(self.driver.launched, 0)
        self.assertEqual(self.driver.failures[0][1], 0)
        self.coordinator = runner.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        ticks = iter((10., 9.))
        with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
            self.run_case(clock=lambda: next(ticks))
        self.assertEqual(self.driver.launched, 0)

    def test_identity_rechecked_after_acceptance_before_saving(self):
        checks = mock.Mock(side_effect=[None, None, ValueError('synthetic identity changed')])
        with mock.patch.object(runner, '_verify_assets', checks):
            with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
                self.run_case()
        self.assertEqual((self.driver.launched, self.driver.accepted), (1, 1))
        self.assertEqual(self.driver.failures[0][1], 1)
        self.assertEqual(self.saved, [])

    def test_external_check_failure_after_acceptance_prevents_result(self):
        outcomes = iter((True, True, True, True, False))
        self.check = lambda: next(outcomes)
        self.driver = FakeDriver(self.manifest, self.check)
        with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
            self.run_case()
        self.assertEqual((self.driver.launched, self.driver.accepted), (1, 1))
        self.assertEqual(self.driver.failures[0][1], 1)
        self.assertEqual(self.saved, [])

    def test_identity_changed_during_save_prevents_success(self):
        checks = mock.Mock(side_effect=[None, None, None, ValueError('synthetic identity changed')])
        with mock.patch.object(runner, '_verify_assets', checks):
            with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
                self.run_case()
        self.assertEqual((self.driver.launched, self.driver.accepted), (1, 1))
        self.assertEqual(self.driver.failures[0][1], 1)
        self.assertEqual(len(self.saved), 1)  # An inner summary alone is not final success.

    def test_cancellation_after_claim_preserves_original_even_if_receipt_fails(self):
        for signal in (KeyboardInterrupt(), SystemExit(17)):
            with self.subTest(kind=type(signal).__name__):
                self.coordinator = runner.Coordinator()
                self.driver = FakeDriver(self.manifest, self.check)
                def interrupted(*_args):
                    self.driver.launched += 1
                    raise signal
                with mock.patch.object(self.driver, 'launch', side_effect=interrupted), \
                     mock.patch.object(self.driver, 'failure', side_effect=OSError('synthetic receipt error')) as receipt:
                    with self.assertRaises(type(signal)) as caught:
                        self.run_case()
                self.assertIs(caught.exception, signal)
                receipt.assert_called_once()
                self.assertEqual(self.driver.launched, 1)
                with self.assertRaisesRegex(RuntimeError, '^Single control attempt already used$'):
                    self.run_case()

    def test_cancellation_before_claim_is_preserved_without_stage_receipt(self):
        signal = KeyboardInterrupt()
        with mock.patch.object(self.driver, 'preflight', side_effect=signal):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)
        self.assertFalse(self.driver.claimed)
        self.assertEqual(self.driver.failures, [])

    def test_cancellation_during_receipt_does_not_mask_an_existing_cancellation(self):
        original, later = KeyboardInterrupt(), SystemExit(23)
        with mock.patch.object(self.driver, 'launch', side_effect=original), \
             mock.patch.object(self.driver, 'failure', side_effect=later):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, original)

    def test_cancellation_during_receipt_for_ordinary_error_is_not_swallowed(self):
        self.driver.fail = 'receipt'
        signal = KeyboardInterrupt()
        with mock.patch.object(self.driver, 'failure', side_effect=signal):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)

    def test_output_restoration_failure_does_not_mask_cancellation(self):
        @contextmanager
        def failed_restoration():
            try:
                yield
            finally:
                raise OSError('synthetic output restoration failure')

        signal = KeyboardInterrupt()
        with mock.patch.object(runner.worker, '_quiet_output', side_effect=failed_restoration), \
             mock.patch.object(self.driver, 'launch', side_effect=signal):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)
        self.assertEqual(len(self.driver.failures), 1)

        self.coordinator = runner.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check, fail='receipt')
        signal = SystemExit(31)
        with mock.patch.object(runner.worker, '_quiet_output', side_effect=failed_restoration), \
             mock.patch.object(self.driver, 'failure', side_effect=signal):
            with self.assertRaises(SystemExit) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)

    def test_no_private_callback_output(self):
        visible = io.StringIO()
        def noisy(value):
            print('PRIVATE_SENTINEL')
            return self.save(value)
        with redirect_stdout(visible):
            result = self.run_case(save_result=noisy)
        self.assertEqual(visible.getvalue(), '')
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
