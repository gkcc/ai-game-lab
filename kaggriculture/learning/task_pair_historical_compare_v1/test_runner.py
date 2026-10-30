"""Synthetic coordinator regressions; no real games, models or files."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_campaign_v1.test_acceptance import fixture, supervision
from learning.task_pair_dev_campaign_v1.test_campaign import limits as campaign_limits
from learning.task_pair_dev_campaign_v1.test_manifest import inputs
from . import runner


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'artifacts' / 'task-pair-historical-compare-v1-synthetic'


def _raw(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def manifest_and_plans():
    manifest = build_manifest(**inputs())
    plans = [p for alias in ('S02', 'S03') for side in (0, 1)
             for arm in ('baseline', 'control') for p in manifest.panel.plans()
             if (p['seed_alias'], p['opponent'], p['side'], p['policy'])
             == (alias, 'market', side, arm)]
    return manifest, plans


def limits():
    result = campaign_limits()
    result.update(max_games=8, max_replays=8, max_comparisons=4, max_attempts=1)
    result['total_output_bytes'] = (8 * result['per_game_reserve_bytes']
                                     + 4 * runner.MAX_RECEIPT_BYTES
                                     + 2 * runner.driver_module.MAX_SUMMARY)
    return result


def bundle_for(manifest, plan, missing_head=None):
    _, _, original = fixture()
    values = {key: json.loads(raw) for key, raw in original.items()}
    row, audit, complete, storage = (values[key] for key in ('row', 'audit', 'complete', 'storage'))
    metrics, receipt = values['metrics'], values['metrics_receipt']
    side = plan['side']
    row.update(plan, task_checkpoint_sha256=manifest.panel.task_model(plan)[1],
               rival_sha256=manifest.opponent(plan['opponent'])[1])
    row.update(bank=row['banks'][side], opponent_bank=row['banks'][1-side],
               initial_bank=row['initial_banks'][side], opponent_initial_bank=row['initial_banks'][1-side])
    row.update(net_profit=row['bank']-row['initial_bank'],
               opponent_net_profit=row['opponent_bank']-row['opponent_initial_bank'],
               margin=row['bank']-row['opponent_bank'])
    row['score'] = 1. if row['margin'] > 0 else .5 if row['margin'] == 0 else 0.
    for item in (audit, complete, storage, receipt):
        item['game'] = plan['game']
    audit['primary_model_counts']['crop_forwards'] = 1
    audit['live_model_counts']['crop_forwards'] = 1
    if missing_head:
        audit['primary_model_counts'][missing_head] = 0
        audit['live_model_counts'][missing_head] = 0
    def economy(final):
        return dict(cash_reconciled=True, initial_cash=5, final_cash=final, net_profit=final-5,
                    successful_plants=1, seed_units_bought=2, sales_units=3, sales_revenue=4,
                    ending_seed_units=0, ending_warehouse_units=0)
    metrics.update(game=plan['game'], seed_alias=plan['seed_alias'], seed=plan['seed'],
                   opponent=plan['opponent'], side=side, arm=plan['policy'],
                   task_checkpoint_sha256=row['task_checkpoint_sha256'])
    metrics['result'] = dict(actor=side, official_turns=719,
        economy=dict(own=economy(row['bank']), opponent=economy(row['opponent_bank'])),
        margin=row['margin'], score=row['score'],
        selection_counts=dict(model_callbacks=3, model_callback_multi_unit_opportunities=1))
    raw = {key: _raw(value) for key, value in values.items()}
    complete.update(row_sha256=hashlib.sha256(raw['row']).hexdigest(),
                    certificate_sha256=hashlib.sha256(raw['audit']).hexdigest())
    receipt.update(metrics_bytes=len(raw['metrics']), metrics_sha256=hashlib.sha256(raw['metrics']).hexdigest())
    raw['complete'], raw['metrics_receipt'] = _raw(complete), _raw(receipt)
    return raw


class FakeDriver:
    def __init__(self, manifest, check, failure=None):
        self.manifest, self.check, self.root, self.stage = manifest, check, ROOT, STAGE
        self.claimed, self.fail, self.size = False, failure, 100
        self.launched, self.accepted, self.compared, self.failures = [], [], [], []

    def preflight(self, plan, _limits, _remaining):
        return self.fail != ('initial' if plan is None else 'preflight')

    def claim(self):
        if self.claimed or self.fail == 'claim':
            return False
        self.claimed = True
        return True

    def launch(self, plan, _job):
        self.launched.append(plan['game'])
        if self.fail == ('raise', len(self.launched)):
            raise ValueError('PRIVATE_SYNTHETIC_VALUE')
        if self.fail == ('supervision', len(self.launched)):
            return dict(supervision(), job_members_zero=False)
        return supervision()

    def record_supervision(self, _plan, _result):
        return True

    def read_bundle(self, plan):
        head = self.fail[0] if (isinstance(self.fail, tuple)
                               and self.fail[1] == len(self.launched)
                               and self.fail[0] in ('task_forwards', 'crop_forwards')) else None
        raw = bundle_for(self.manifest, plan, head)
        if self.fail == ('receipt', len(self.launched)):
            raw['metrics'] += b' '
        return raw

    def verify_files(self, _plan, _row):
        return self.fail != ('files', len(self.launched))

    def record_acceptance(self, plan, _bundle):
        self.accepted.append(plan['game'])
        return True

    def preflight_comparison(self, _limits, _remaining):
        return self.fail != 'comparison_preflight'

    def compare(self, left, right):
        self.compared.append((left['game'], right['game']))
        if self.fail == ('compare', len(self.compared)):
            raise ValueError('PRIVATE_SYNTHETIC_VALUE')
        value = dict(reference='baseline', candidate='control', seed=left['seed'],
                     seed_alias=left['seed_alias'], opponent='market', side=left['side'],
                     result=dict(comparison_complete=True, matched_calls=3,
                                 matched_multi_unit_opportunities=1))
        if self.fail == ('wrong_comparison', len(self.compared)):
            value['side'] = 1 - value['side']
        return value

    def failure(self, game, completed):
        self.failures.append((game, completed))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.plans = manifest_and_plans()
        self.check = lambda: True
        self.driver = FakeDriver(self.manifest, self.check)
        self.coordinator = runner.Coordinator()
        self.saved, self.authorized = [], []

    def authorize(self, plan):
        self.authorized.append(plan['game'])
        return True

    def save(self, value):
        self.saved.append(value)
        self.driver.size += 100
        return True

    def run_case(self, **changes):
        values = dict(manifest=self.manifest, plans=self.plans, limits=limits(), driver=self.driver,
                      check=self.check, authorize_game=self.authorize, save_result=self.save, clock=lambda: 0.)
        values.update(changes)
        assets = {ROOT / relative: digest for relative, digest in (
            *self.manifest.panel.models[:2], self.manifest.panel.crop, self.manifest.opponent('market'),
            (runner.worker.WORKER_RELATIVE, self.manifest.panel.worker_sha))}
        with mock.patch.object(runner.driver_module, '_owned_size', side_effect=lambda _p: self.driver.size), \
             mock.patch.object(runner.driver_module, '_hash', side_effect=lambda _r, p, _m: assets.get(p)):
            return self.coordinator.run(**values)

    def test_eight_games_use_real_acceptance_and_four_real_paired_coordinates(self):
        result = self.run_case()
        self.assertEqual((len(self.driver.launched), len(self.driver.accepted), len(self.driver.compared)), (8, 8, 4))
        self.assertEqual(self.authorized, self.driver.launched)
        self.assertEqual(len(self.saved), 1)
        payload = self.saved[0]
        self.assertEqual((len(payload['rows']), len(payload['comparisons'])), (8, 4))
        self.assertEqual((payload['paired']['expected_games'], payload['paired']['observed_games']), (8, 8))
        self.assertEqual(payload['paired']['seed_clusters'], 2)
        self.assertEqual(payload['paired']['statistical_significance'], 'unknown')
        self.assertFalse(result['promotion'])
        self.assertFalse(result['independent_improvement'])
        self.assertFalse(result['observed_outer_exit'])
        self.assertEqual(result['generic_driver_claim_capacity'], 24)
        for secret in (str(self.manifest.panel.seeds[0]), str(ROOT), self.manifest.panel.crop[1]):
            self.assertNotIn(secret, json.dumps(result))

    def test_invalid_coordinates_and_limits_stop_before_claim(self):
        changes = []
        for edit, value in (('opponent', 'round2'), ('side', True), ('policy', 'auxiliary'), ('seed', 3)):
            plans = deepcopy(self.plans); plans[0][edit] = value
            changes.append(dict(plans=plans))
        changes += [dict(plans=list(reversed(self.plans))), dict(plans=self.plans[:7])]
        for key, value in (('max_games', 9), ('max_games', True), ('max_comparisons', 3),
                           ('max_attempts', 2), ('total_output_bytes', 1)):
            altered = limits(); altered[key] = value; changes.append(dict(limits=altered))
        for change in changes:
            with self.subTest(change=next(iter(change))):
                self.coordinator = runner.Coordinator()
                self.driver = FakeDriver(self.manifest, self.check)
                with self.assertRaisesRegex(RuntimeError, '^Historical comparison stopped$'):
                    self.run_case(**change)
                self.assertFalse(self.driver.claimed)
                self.assertEqual(self.driver.launched, [])
                self.assertEqual(self.saved, [])

    def test_failure_stops_and_records_only_accepted_games(self):
        for failure in (('supervision', 3), ('receipt', 3), ('files', 3),
                        ('task_forwards', 3), ('crop_forwards', 3)):
            with self.subTest(kind=failure[0]):
                self.coordinator = runner.Coordinator()
                self.driver = FakeDriver(self.manifest, self.check, failure)
                with self.assertRaisesRegex(RuntimeError, '^Historical comparison stopped$'):
                    self.run_case()
                self.assertEqual(len(self.driver.launched), 3)
                self.assertEqual(len(self.driver.accepted), 2)
                self.assertEqual(self.driver.failures[0][1], 2)
                self.assertEqual(self.driver.compared, [])
                self.assertEqual(self.saved, [])

    def test_bad_comparison_or_save_fails_without_retry(self):
        for failure in (('compare', 2), ('wrong_comparison', 2)):
            self.coordinator = runner.Coordinator()
            self.driver = FakeDriver(self.manifest, self.check, failure)
            with self.assertRaisesRegex(RuntimeError, '^Historical comparison stopped$'):
                self.run_case()
            self.assertEqual((len(self.driver.launched), self.driver.failures[0][1]), (8, 8))
            self.assertEqual(len(self.driver.compared), 2)
            self.assertEqual(self.saved, [])
        self.coordinator = runner.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        with self.assertRaisesRegex(RuntimeError, '^Historical comparison stopped$'):
            self.run_case(save_result=lambda _value: False)
        self.assertEqual((len(self.driver.launched), self.driver.failures[0][1]), (8, 8))
        with self.assertRaisesRegex(RuntimeError, '^Historical comparison attempt already used$'):
            self.run_case()
        self.assertEqual(len(self.driver.launched), 8)

    def test_authorization_or_asset_failure_never_launches(self):
        with self.assertRaisesRegex(RuntimeError, '^Historical comparison stopped$'):
            self.run_case(authorize_game=lambda _plan: False)
        self.assertEqual(self.driver.launched, [])
        self.assertEqual(self.driver.failures[0][1], 0)
        self.coordinator = runner.Coordinator()
        self.driver = FakeDriver(self.manifest, self.check)
        with mock.patch.object(runner, '_verify_assets', side_effect=ValueError('PRIVATE_SYNTHETIC_VALUE')):
            with self.assertRaisesRegex(RuntimeError, '^Historical comparison stopped$') as error:
                self.run_case()
        self.assertNotIn('PRIVATE_SYNTHETIC_VALUE', str(error.exception))
        self.assertEqual(self.driver.launched, [])

    def test_cancel_preserves_original_if_failure_receipt_raises(self):
        signal = KeyboardInterrupt()
        with mock.patch.object(self.driver, 'launch', side_effect=signal), \
             mock.patch.object(self.driver, 'failure', side_effect=ValueError('PRIVATE_SYNTHETIC_VALUE')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_case()
        self.assertIs(caught.exception, signal)
        self.assertEqual(self.saved, [])


if __name__ == '__main__':
    unittest.main()
