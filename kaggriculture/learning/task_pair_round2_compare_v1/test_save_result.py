"""Synthetic coordinator-to-projection wiring; no real games or disk I/O."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import psutil

from learning.task_pair_dev_campaign_v1 import acceptance
from learning.task_pair_historical_compare_v1.test_runner import bundle_for, limits
from . import coordinator, projection
from .test_coordinator import ROOT, STAGE, Round2Driver, manifest_and_plans


def encoded(value):
    return projection._json(value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def marker(game, kind):
    return (f'job-results/{game}.{kind}.local.json',
            dict(schema='synthetic-marker', game=game, kind=kind, retry_allowed=False))


class MemoryDriver(Round2Driver):
    """One in-memory synthetic receipt store with real linked receipt hashes."""
    def __init__(self, manifest, check):
        super().__init__(manifest, check)
        self.files, self.bundles, self.outputs = {}, {}, {}

    def launch(self, plan, job):
        game = plan['game']
        self.files[STAGE / 'job-inputs' / (game + '.json')] = encoded(dict(synthetic_job=game))
        self.bundles[game] = bundle_for(self.manifest, plan)
        return super().launch(plan, job)

    def record_supervision(self, plan, value):
        game = plan['game']
        self.files[STAGE / 'job-results' / (game + '.supervision.local.json')] = encoded(dict(
            schema='task-pair-dev-job-supervision-v1', game=game, supervision=value, retry_allowed=False))
        return True

    def read_bundle(self, plan):
        return self.bundles[plan['game']]

    def record_acceptance(self, plan, bundle):
        game = plan['game']
        supervision = self.files[STAGE / 'job-results' / (game + '.supervision.local.json')]
        job = self.files[STAGE / 'job-inputs' / (game + '.json')]
        self.files[STAGE / 'job-results' / (game + '.acceptance.local.json')] = encoded(dict(
            schema='task-pair-dev-job-acceptance-v1', game=game, passed=True, retry_allowed=False,
            supervision_sha256=digest(supervision), job_input_sha256=digest(job),
            receipt_sha256={name: digest(raw) for name, raw in bundle.items()}))
        self.accepted.append(game)
        return True

    def compare(self, left, right):
        result = super().compare(left, right)
        result['result']['stop_reason'] = None
        path = coordinator.driver_module._comparison_file(STAGE, left['game'], 'control')
        self.files[path] = encoded(dict(schema='task-pair-dev-comparison-v1',
            reference_game=left['game'], candidate_game=right['game'],
            reference_replay_sha256=left['replay_sha256'], candidate_replay_sha256=right['replay_sha256'],
            reference_trace_sha256=left['trace_sha256'], candidate_trace_sha256=right['trace_sha256'],
            comparison=result))
        return result


class SaveResultWiringTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.plans = manifest_and_plans()
        self.check = lambda: True
        self.driver = MemoryDriver(self.manifest, self.check)
        self.summary_checks, self.trace_pairs, self.failures, self.replay_calls = [], [], [], []
        self.available = lambda: 10**12
        self.ctx = dict(modules=dict(driver=coordinator.driver_module, acceptance=acceptance,
                                     child=SimpleNamespace(worker=coordinator.worker)),
                        stage=STAGE, check=self.check, plans=self.plans, marker=marker,
                        manifest=self.manifest, time_check=lambda: True, limits=limits())
        self.ctx['limits']['initial_available_bytes'] = 5 * 1024**3
        self.ctx['limits']['between_available_bytes'] = 4 * 1024**3
        self.ctx['limits']['job']['min_available_bytes'] = 3 * 1024**3

    def authorize(self, plan):
        for kind in ('permit', 'started', 'consumed'):
            relative, value = marker(plan['game'], kind)
            self.driver.files[STAGE / relative] = encoded(value)
        return True

    def replay(self, _root, path, _expected):
        self.replay_calls.append(path.parent.name)
        return dict(synthetic_game=path.parent.name)

    def trace(self, _root, path, _expected):
        def stream():
            yield dict(synthetic_game=path.name.removesuffix('.jsonl.gz'))
        return stream()

    def compare_saved(self, reference, candidate, left, right, *, seed, side, check):
        self.assertTrue(check())
        self.assertTrue(reference['synthetic_game'].endswith('-baseline'))
        self.assertTrue(candidate['synthetic_game'].endswith('-control'))
        ref_plan = next(plan for plan in self.plans if plan['game'] == reference['synthetic_game'])
        new_plan = next(plan for plan in self.plans if plan['game'] == candidate['synthetic_game'])
        self.assertEqual((seed, side), (ref_plan['seed'], ref_plan['side']))
        self.assertEqual((seed, side), (new_plan['seed'], new_plan['side']))
        self.assertEqual(list(left), [dict(synthetic_game=reference['synthetic_game'])])
        self.assertEqual(list(right), [dict(synthetic_game=candidate['synthetic_game'])])
        self.trace_pairs.append((reference['synthetic_game'], candidate['synthetic_game']))
        return dict(scope='strict_common_recorded_history', trace_turns_per_game=719,
                    matched_selection_records=3, stop_reason=None, first_stop_step=None,
                    first_different_recorded_choice=None,
                    complete_request_prefix=dict(scope='same_complete_history', status='no_difference',
                                                 reason='all_requests_equal', step=None))

    def run_with(self, tamper=None):
        errors = []
        assets = {ROOT / relative: expected for relative, expected in (
            *self.manifest.panel.models[:2], self.manifest.panel.crop, self.manifest.opponent('round2'),
            (coordinator.worker.WORKER_RELATIVE, self.manifest.panel.worker_sha))}

        def read(_root, path, _maximum):
            return self.driver.files[path]

        def file_hash(_root, path, _maximum):
            return digest(self.driver.files[path]) if path in self.driver.files else assets.get(path)

        def write(_root, path, value, *args, **kwargs):
            if path in self.driver.outputs:
                raise FileExistsError('synthetic duplicate')
            self.driver.outputs[path] = deepcopy(value)
            self.driver.size += len(encoded(value))

        def save(payload):
            if tamper is not None:
                tamper(payload, self.driver)
            self.summary_checks.append((len(self.driver.bundles), len(self.driver.compared)))
            try:
                return projection.save_result(self.ctx, self.driver, payload)
            except Exception as error:
                errors.append(error)
                raise

        dummy_counts = dict(by_source=dict(teacher=0), total=3)
        with mock.patch.object(coordinator.driver_module, '_read', side_effect=read), \
             mock.patch.object(coordinator.driver_module, '_hash', side_effect=file_hash), \
             mock.patch.object(coordinator.driver_module, '_owned_size', side_effect=lambda _p: self.driver.size), \
             mock.patch.object(coordinator.driver_module, '_write_new', side_effect=write), \
             mock.patch.object(coordinator.driver_module, '_replay', side_effect=self.replay), \
             mock.patch.object(coordinator.driver_module, '_trace', side_effect=self.trace), \
             mock.patch.object(projection, 'compare_saved', side_effect=self.compare_saved), \
             mock.patch.object(projection, '_decision_counts', return_value=dummy_counts), \
             mock.patch.object(projection.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)), \
             mock.patch.object(psutil, 'virtual_memory',
                               side_effect=lambda: SimpleNamespace(available=self.available())):
            try:
                return coordinator.Coordinator().run(
                    manifest=self.manifest, plans=self.plans, limits=self.ctx['limits'], driver=self.driver,
                    check=self.check, authorize_game=self.authorize, save_result=save,
                    on_failure=self.failures.append, clock=lambda: 0.)
            except RuntimeError:
                if tamper is None and errors:
                    raise errors[-1]
                raise

    def test_eight_accepted_bundles_and_four_pair_receipts_produce_two_outputs(self):
        result = self.run_with()
        self.assertTrue(result['passed'])
        self.assertEqual(self.summary_checks, [(8, 4)])
        self.assertEqual(len(self.driver.accepted), 8)
        self.assertEqual(len(self.trace_pairs), 4)
        self.assertEqual(len(self.replay_calls), 8)
        self.assertEqual(set(path.name for path in self.driver.outputs),
                         {'comparison-result.local.json', 'projected-summary.local.json'})
        private = self.driver.outputs[STAGE / 'comparison-result.local.json']
        public = self.driver.outputs[STAGE / 'projected-summary.local.json']
        self.assertEqual((len(private['ledger']), private['games'], private['comparisons']), (8, 8, 4))
        self.assertEqual((public['games'], len(public['pairs']), public['seed_clusters']), (8, 4, 2))
        self.assertEqual(public['statistical_significance'], 'unknown')
        self.assertFalse(public['promotion'])

    def test_coherent_wrong_marker_rejected_before_outputs(self):
        def tamper(_payload, driver):
            relative, value = marker(self.plans[0]['game'], 'permit')
            value['kind'] = 'consumed'
            driver.files[STAGE / relative] = encoded(value)
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_with(tamper)
        self.assertEqual(self.summary_checks, [(8, 4)])
        self.assertEqual(self.driver.outputs, {})
        self.assertEqual(self.failures[-1]['phase'], 'save_result')
        self.assertEqual(self.failures[-1]['category'], 'ValueError')
        self.assertEqual(self.trace_pairs, [])

    def test_memory_dip_after_first_replay_stops_before_second(self):
        threshold = self.ctx['limits']['between_available_bytes']
        self.available = lambda: threshold + 1 if not self.replay_calls else threshold - 1
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_with(lambda _payload, _driver: None)
        self.assertEqual(len(self.replay_calls), 1)
        self.assertEqual(self.ctx['last_available_bytes'], threshold - 1)
        self.assertEqual(self.failures[-1]['phase'], 'save_result')
        self.assertEqual(self.failures[-1]['category'], 'MemoryError')
        self.assertEqual(self.driver.outputs, {})
        self.assertEqual(self.trace_pairs, [])

    def test_coherent_tampered_comparison_and_payload_rejected_before_outputs(self):
        def tamper(payload, driver):
            item = payload['comparisons'][0]
            item['opponent'] = 'market'
            path = coordinator.driver_module._comparison_file(STAGE, self.plans[0]['game'], 'control')
            receipt = json.loads(driver.files[path])
            receipt['comparison'] = deepcopy(item)
            driver.files[path] = encoded(receipt)
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_with(tamper)
        self.assertEqual(self.summary_checks, [(8, 4)])
        self.assertEqual(self.driver.outputs, {})
        self.assertEqual(self.failures[-1]['phase'], 'save_result')
        self.assertEqual(self.failures[-1]['category'], 'ValueError')
        self.assertEqual(len(self.trace_pairs), 4)

    def test_self_consistent_failed_supervision_still_rejected_before_outputs(self):
        def tamper(_payload, driver):
            game = self.plans[0]['game']
            supervision_path = STAGE / 'job-results' / (game + '.supervision.local.json')
            acceptance_path = STAGE / 'job-results' / (game + '.acceptance.local.json')
            supervised = json.loads(driver.files[supervision_path])
            supervised['supervision']['exit_code'] = 1
            driver.files[supervision_path] = encoded(supervised)
            accepted = json.loads(driver.files[acceptance_path])
            accepted['supervision_sha256'] = digest(driver.files[supervision_path])
            driver.files[acceptance_path] = encoded(accepted)
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_with(tamper)
        self.assertEqual(self.driver.outputs, {})
        self.assertEqual(self.failures[-1]['phase'], 'save_result')
        self.assertEqual(self.failures[-1]['category'], 'ValueError')
        self.assertEqual(self.trace_pairs, [])

    def test_wrong_bundle_hash_in_acceptance_rejected_before_outputs(self):
        def tamper(_payload, driver):
            game = self.plans[0]['game']
            acceptance_path = STAGE / 'job-results' / (game + '.acceptance.local.json')
            accepted = json.loads(driver.files[acceptance_path])
            accepted['receipt_sha256']['metrics'] = '0' * 64
            driver.files[acceptance_path] = encoded(accepted)
        with self.assertRaisesRegex(RuntimeError, '^Round2 comparison stopped$'):
            self.run_with(tamper)
        self.assertEqual(self.driver.outputs, {})
        self.assertEqual(self.failures[-1]['phase'], 'save_result')
        self.assertEqual(self.failures[-1]['category'], 'ValueError')
        self.assertEqual(self.trace_pairs, [])


if __name__ == '__main__':
    unittest.main()
