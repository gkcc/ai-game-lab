"""In-memory toy coordination checks; no real games, models or files."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest import mock

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_campaign_v1.test_acceptance import fixture as receipt_fixture
from learning.task_pair_dev_campaign_v1.test_acceptance import supervision
from learning.task_pair_dev_campaign_v1.test_campaign import limits as old_limits
from learning.task_pair_dev_campaign_v1.test_manifest import inputs
from . import runner


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'artifacts' / 'task-pair-toy-smoke-v1-synthetic'


def manifest_and_plans():
    data = inputs()
    for role in ('baseline', 'control', 'auxiliary'):
        data['models'][role]['path'] = f'artifacts/{STAGE.name}/models/{role}.npz'
    data['crop']['path'] = f'artifacts/{STAGE.name}/models/crop.npz'
    manifest = build_manifest(**data)
    plans = [next(p for p in manifest.panel.plans()
                  if (p['seed_alias'], p['opponent'], p['side'], p['policy']) == ('S02', 'market', 0, arm))
             for arm in ('baseline', 'control')]
    return manifest, plans


def limits():
    value = old_limits()
    value.update(model_reserve_bytes=1024**2, max_games=2, max_replays=2,
                 max_comparisons=1, max_attempts=1)
    value['total_output_bytes'] = (2 * value['per_game_reserve_bytes']
                                    + value['model_reserve_bytes'] + 16 * 1024**2)
    return value


def _encode(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def bundle_for(manifest, plan):
    """Adapt the existing synthetic receipt fixture to either smoke plan."""
    _, _, bundle = receipt_fixture()
    parsed = {key: json.loads(value) for key, value in bundle.items()}
    row, audit, complete, storage = (parsed[key] for key in ('row', 'audit', 'complete', 'storage'))
    metrics, receipt = parsed['metrics'], parsed['metrics_receipt']
    row.update(plan)
    row['task_checkpoint_sha256'] = manifest.panel.task_model(plan)[1]
    row['rival_sha256'] = manifest.opponent(plan['opponent'])[1]
    audit['game'] = complete['game'] = storage['game'] = plan['game']
    audit['primary_model_counts']['crop_forwards'] = 4
    audit['live_model_counts']['crop_forwards'] = 4
    metrics.update(game=plan['game'], seed=plan['seed'], seed_alias=plan['seed_alias'],
                   opponent=plan['opponent'], side=plan['side'], arm=plan['policy'],
                   task_checkpoint_sha256=row['task_checkpoint_sha256'])
    receipt['game'] = plan['game']
    out = {key: _encode(value) for key, value in parsed.items()}
    complete.update(row_sha256=hashlib.sha256(out['row']).hexdigest(),
                    certificate_sha256=hashlib.sha256(out['audit']).hexdigest())
    receipt.update(metrics_bytes=len(out['metrics']), metrics_sha256=hashlib.sha256(out['metrics']).hexdigest())
    out['complete'], out['metrics_receipt'] = _encode(complete), _encode(receipt)
    return out


class FakeDriver:
    def __init__(self, manifest, fail=None):
        self.manifest, self.root, self.stage = manifest, ROOT, STAGE
        self.fail, self.claimed, self.calls, self.failures, self.size = fail, False, [], [], 100

    def preflight(self, plan, _limits, _remaining):
        self.calls.append(('preflight', None if plan is None else plan['policy']))
        return not (self.fail == 'preflight' or
                    (self.fail == 'input_preflight' and plan is not None
                     and ('prepare', None) not in self.calls))

    def claim(self):
        self.calls.append(('claim', None))
        if self.claimed:
            raise ValueError('Synthetic attempt already claimed')
        self.claimed = True
        return True

    def launch(self, plan, _job):
        self.calls.append(('launch', plan['policy']))
        result = supervision()
        if self.fail == 'job_' + plan['policy']:
            result['job_members_zero'] = False
        return result

    def record_supervision(self, plan, _result):
        self.calls.append(('supervision', plan['policy']))
        return True

    def read_bundle(self, plan):
        self.calls.append(('read', plan['policy']))
        result = bundle_for(self.manifest, plan)
        if self.fail == 'receipt_' + plan['policy']:
            result['metrics'] += b' '
        if self.fail == 'no_crop_' + plan['policy']:
            audit = json.loads(result['audit'])
            audit['primary_model_counts']['crop_forwards'] = 0
            audit['live_model_counts']['crop_forwards'] = 0
            result['audit'] = _encode(audit)
            complete = json.loads(result['complete'])
            complete['certificate_sha256'] = hashlib.sha256(result['audit']).hexdigest()
            result['complete'] = _encode(complete)
        return result

    def verify_files(self, plan, _row):
        self.calls.append(('verify_files', plan['policy']))
        return self.fail != 'files_' + plan['policy']

    def record_acceptance(self, plan, _bundle):
        self.calls.append(('acceptance', plan['policy']))
        return True

    def preflight_comparison(self, _limits, _remaining):
        self.calls.append(('preflight_comparison', None))
        return True

    def compare(self, left, right):
        self.calls.append(('compare', None))
        if self.fail == 'compare':
            raise ValueError('synthetic comparison failure')
        return dict(reference='baseline', candidate='control', seed=left['seed'], seed_alias='S02',
                    opponent='market', side=0, result=dict(comparison_complete=True,
                    matched_calls=0, matched_multi_unit_opportunities=0))

    def failure(self, game, completed):
        self.failures.append((game, completed))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.plans = manifest_and_plans()
        self.driver = FakeDriver(self.manifest)
        self.saved = []

    def prepare(self, manifest, driver, allowance):
        self.assertIs(manifest, self.manifest)
        self.assertTrue(driver.claimed)
        self.assertEqual(allowance, limits()['model_reserve_bytes'])
        driver.calls.append(('prepare', None))
        driver.size += 123
        return True

    def save(self, payload):
        self.driver.calls.append(('save', None))
        self.saved.append(payload)
        return True

    def run_case(self, **overrides):
        fixed = dict(manifest=self.manifest, plans=self.plans, limits=limits(), driver=self.driver,
                     check=lambda: True, prepare_inputs=self.prepare, save_result=self.save, clock=lambda: 0.)
        fixed.update(overrides)
        digests = {Path(relative).name: digest
                   for relative, digest in (*self.manifest.panel.models, self.manifest.panel.crop)}
        with mock.patch.object(runner.driver_module, '_owned_size', side_effect=lambda _stage: self.driver.size), \
             mock.patch.object(runner.driver_module, '_hash', side_effect=lambda _root, path, _max: digests[path.name]):
            return runner.run(**fixed)

    def test_two_games_use_real_receipt_acceptance_before_comparison_and_save(self):
        result = self.run_case()
        self.assertEqual(result['games'], 2)
        self.assertEqual(result['registered_game_limit'], 2)
        self.assertEqual(result['generic_driver_claim_capacity'], 24)
        self.assertFalse(result['promotion'])
        self.assertEqual(self.driver.calls, [
            ('preflight', None), ('claim', None), ('preflight', 'baseline'), ('prepare', None),
            ('preflight', 'baseline'), ('launch', 'baseline'), ('supervision', 'baseline'),
            ('read', 'baseline'), ('verify_files', 'baseline'), ('acceptance', 'baseline'),
            ('preflight', 'control'), ('launch', 'control'), ('supervision', 'control'),
            ('read', 'control'), ('verify_files', 'control'), ('acceptance', 'control'),
            ('preflight_comparison', None), ('compare', None), ('save', None)])
        self.assertEqual(len(self.saved), 1)
        self.assertEqual([row['arm'] for row in self.saved[0]['games']], ['baseline', 'control'])
        self.assertEqual(self.saved[0]['attempted'], 2)
        self.assertFalse(self.saved[0]['promotion'])
        self.assertNotIn(str(self.plans[0]['seed']), json.dumps(result))

    def test_out_of_scope_plans_stop_before_stage_claim(self):
        variants = [list(reversed(self.plans)), [self.plans[0], self.plans[0]], self.plans[:1]]
        for edit in ('side', 'seed_alias', 'opponent', 'seed'):
            value = deepcopy(self.plans)
            value[0][edit] = {'side': 1, 'seed_alias': 'S03', 'opponent': 'round2', 'seed': 3}[edit]
            variants.append(value)
        for candidate in variants:
            with self.subTest(kind=len(candidate)), self.assertRaisesRegex(RuntimeError, '^Toy smoke stopped$'):
                self.run_case(plans=candidate)
            self.assertEqual(self.driver.calls, [])
            self.assertEqual(self.driver.failures, [])

    def test_failed_first_or_second_job_stops_and_records_completed_count(self):
        for arm, completed in (('baseline', 0), ('control', 1)):
            with self.subTest(arm=arm):
                self.driver = FakeDriver(self.manifest, fail='job_' + arm)
                with self.assertRaisesRegex(RuntimeError, '^Toy smoke stopped$'):
                    self.run_case()
                self.assertEqual(self.driver.failures[0][1], completed)
                self.assertEqual(sum(name == 'launch' for name, _ in self.driver.calls), completed + 1)
                self.assertNotIn(('compare', None), self.driver.calls)
                self.assertEqual(self.saved, [])

    def test_bad_receipt_or_file_identity_does_not_enter_later_game(self):
        for failure, completed in (('receipt_baseline', 0), ('files_control', 1)):
            with self.subTest(failure=failure):
                self.driver = FakeDriver(self.manifest, fail=failure)
                with self.assertRaisesRegex(RuntimeError, '^Toy smoke stopped$'):
                    self.run_case()
                self.assertEqual(self.driver.failures[0][1], completed)
                self.assertEqual(sum(name == 'launch' for name, _ in self.driver.calls), completed + 1)
                self.assertNotIn(('compare', None), self.driver.calls)
                self.assertEqual(self.saved, [])

    def test_crop_head_must_act_in_each_accepted_game(self):
        for arm, completed in (('baseline', 0), ('control', 1)):
            with self.subTest(arm=arm):
                self.driver = FakeDriver(self.manifest, fail='no_crop_' + arm)
                with self.assertRaisesRegex(RuntimeError, '^Toy smoke stopped$'):
                    self.run_case()
                self.assertEqual(self.driver.failures[0][1], completed)
                self.assertNotIn(('acceptance', arm), self.driver.calls)
                self.assertNotIn(('compare', None), self.driver.calls)
                self.assertEqual(self.saved, [])

    def test_failure_comparing_or_saving_never_retries(self):
        self.driver = FakeDriver(self.manifest, fail='compare')
        with self.assertRaisesRegex(RuntimeError, '^Toy smoke stopped$'):
            self.run_case()
        self.assertEqual(self.driver.failures[0][1], 2)
        self.assertEqual(sum(name == 'launch' for name, _ in self.driver.calls), 2)
        self.assertEqual(self.saved, [])
        self.driver = FakeDriver(self.manifest)
        with self.assertRaisesRegex(RuntimeError, '^Toy smoke stopped$'):
            self.run_case(save_result=lambda _payload: False)
        self.assertEqual(self.driver.failures[0][1], 2)
        self.assertEqual(sum(name == 'compare' for name, _ in self.driver.calls), 1)

    def test_bad_limits_or_old_stage_fail_before_claim(self):
        for key, bad in (('max_games', 3), ('max_replays', 3), ('max_comparisons', 2),
                         ('max_attempts', 2), ('total_output_bytes', 1),
                         ('model_reserve_bytes', limits()['per_game_reserve_bytes'] + 1)):
            value = limits(); value[key] = bad
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.run_case(limits=value)
            self.assertEqual(self.driver.calls, [])
        self.driver.stage = ROOT / 'artifacts' / 'task-pair-dev-games-v1-20261030'
        with self.assertRaises(RuntimeError): self.run_case()
        self.assertEqual(self.driver.calls, [])

    def test_preparation_resource_or_model_identity_failure_stops_before_game(self):
        self.driver = FakeDriver(self.manifest, fail='input_preflight')
        with self.assertRaises(RuntimeError): self.run_case()
        self.assertNotIn(('prepare', None), self.driver.calls)
        self.assertEqual(self.driver.failures[0][1], 0)
        self.driver = FakeDriver(self.manifest)
        def oversized(manifest, driver, allowance):
            driver.size += allowance + 1
            return True
        with self.assertRaises(RuntimeError): self.run_case(prepare_inputs=oversized)
        self.assertEqual(sum(name == 'launch' for name, _ in self.driver.calls), 0)
        self.assertEqual(self.driver.failures[0][1], 0)
        self.driver = FakeDriver(self.manifest)
        with mock.patch.object(runner, '_models', side_effect=ValueError('synthetic hash mismatch')):
            with self.assertRaises(RuntimeError): self.run_case()
        self.assertEqual(sum(name == 'launch' for name, _ in self.driver.calls), 0)

    def test_backward_clock_stops_before_any_game(self):
        samples = iter((10., 11., 12., 11.))
        with self.assertRaisesRegex(RuntimeError, '^Toy smoke stopped$'):
            self.run_case(clock=lambda: next(samples))
        self.assertEqual(sum(name == 'launch' for name, _ in self.driver.calls), 0)
        self.assertEqual(self.driver.failures[0][1], 0)

    def test_callback_output_and_private_inputs_are_not_printed(self):
        visible = io.StringIO()
        def noisy_prepare(*args):
            print('SYNTHETIC_PRIVATE_INPUT')
            return self.prepare(*args)
        with redirect_stdout(visible):
            result = self.run_case(prepare_inputs=noisy_prepare)
        self.assertEqual(visible.getvalue(), '')
        self.assertNotIn('SYNTHETIC_PRIVATE_INPUT', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
