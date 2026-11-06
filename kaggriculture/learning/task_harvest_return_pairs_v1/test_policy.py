"""Synthetic selector control and accounting, never real checkpoint evidence."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_harvest_return_pairs_v1.policy import BoundedPairAgent
from learning.task_harvest_return_pairs_v1.selection import ordered_hash, select_target
from learning.task_harvest_return_pairs_v1.test_selection import (
    CROP_SHA, TASK_SHA, patch_synthetic, scenario, source_rows)


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key != 'inference_seconds'}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


class PolicyTests(unittest.TestCase):
    def setUp(self):
        patch_synthetic(self)

    def target(self, *, downstream=True):
        rows, requests = source_rows(downstream=downstream)
        chosen = select_target(rows, side=0, task_sha=TASK_SHA, crop_sha=CROP_SHA)['target']
        return chosen, rows, requests

    def actor(self, target, arm):
        return BoundedPairAgent('crop-wheat', CROP_SHA, 'task-harvest', TASK_SHA, target=target, arm=arm)

    def test_h_arm_requests_and_all_trace_fields_equal_frozen_source(self):
        target, rows, requests = self.target()
        actor = self.actor(target, 'H')
        for step in range(3):
            obs, cfg = scenario(step, active=step == 1)
            self.assertEqual(actor(obs, cfg), requests[step])
            self.assertEqual(ordered_hash(stable(actor.last)), ordered_hash(stable(rows[step])))
        proof = actor.target_proof
        self.assertEqual(actor.override_count, 0)
        self.assertFalse(proof['override_applied'])
        self.assertEqual(proof['actual_choice'], target['harvest_choice'])
        self.assertEqual(proof['unit_request'], ['HARVEST'])
        self.assertEqual(proof['original_proposal'], rows[1]['records'][1]['task_model'])
        self.assertEqual(actor.network.score_calls, rows[2]['model_counts']['task_forwards'])
        self.assertEqual(actor.controller.crop.network.calls, rows[2]['model_counts']['crop_forwards'])

    def test_w_selector_choice_propagates_into_later_crop_reservations_and_market(self):
        target, _rows, _requests = self.target()
        h, w = self.actor(target, 'H'), self.actor(target, 'W')
        for actor in (h, w):
            actor(*scenario(0))
        h_request, w_request = h(*scenario(1, active=True)), w(*scenario(1, active=True))
        self.assertEqual(h_request['farmer'], w_request['farmer'])
        self.assertEqual(h_request['hands'][0], ['HARVEST'])
        self.assertEqual(w_request['hands'][0], ['WATER'])
        self.assertIn(['BUY_SEED', 'WHEAT', 1], h_request['market'])
        self.assertIn(['BUY_SEED', 'CARROT', 1], w_request['market'])
        self.assertEqual(h.last['final_reservations']['needed_seeds'], {'WHEAT': 1})
        self.assertEqual(w.last['final_reservations']['needed_seeds'], {'CARROT': 1})
        self.assertEqual(h.last['records'][0], w.last['records'][0])
        record = w.last['records'][target['ordinal']]
        self.assertEqual(record['chosen'], target['water_choice'])
        self.assertEqual(record['task_model']['chosen'], target['harvest_choice'])
        self.assertEqual(record['pair_choice']['proposed_choice'], target['harvest_choice'])
        self.assertEqual(record['pair_choice']['actual_choice'], target['water_choice'])
        self.assertEqual(record['unit_request'], ['WATER'])
        self.assertFalse(record['harvest_choice_extension']['selected_virtual_harvest'])
        later = w.last['records'][2]
        self.assertEqual(later['inputs']['planned']['unit_actions'][1], ['WATER'])
        self.assertNotIn('pair_choice', later)
        self.assertEqual(w.last['crop_records'][0]['prior']['unit_actions'][1], ['WATER'])
        self.assertEqual(w.override_count, 1)
        for key in ('target', 'inputs', 'input_sha256', 'original_proposal', 'prefix_records_sha256', 'budget_before'):
            self.assertEqual(ordered_hash(h.target_proof[key]), ordered_hash(w.target_proof[key]), key)
        for actor in (h, w):
            proof = actor.target_proof
            for scope in ('total', 'turn'):
                before, after = proof['budget_before'][scope], proof['budget_after'][scope]
                self.assertEqual(after['task_forwards'] - before['task_forwards'], 1)
                self.assertEqual(after['task_rows'] - before['task_rows'], len(proof['inputs']['candidates']))
                self.assertEqual(after['crop_forwards'], before['crop_forwards'])
            self.assertEqual(actor.network.score_calls, 3)
            self.assertEqual(actor.controller.crop.network.calls, 1)
            self.assertEqual(actor.last['fallbacks'], 0)

    def test_original_network_last_is_not_rewritten_as_intervened_water(self):
        target, _rows, _requests = self.target(downstream=False)
        actor = self.actor(target, 'W')
        actor(*scenario(0, downstream=False))
        result = actor(*scenario(1, active=True, downstream=False))
        self.assertEqual(result['hands'][0], ['WATER'])
        self.assertEqual(actor.network.last['chosen'], target['harvest_choice'])
        self.assertEqual(actor.last['records'][-1]['chosen'], target['water_choice'])
        self.assertEqual(actor.network.last['logits'], actor.target_proof['original_proposal']['logits'])
        self.assertEqual(actor.network.score_calls, 2)

    def test_override_happens_once_and_later_turn_uses_original_model(self):
        target, _rows, _requests = self.target()
        actor = self.actor(target, 'W')
        actor(*scenario(0))
        actor(*scenario(1, active=True))
        proof = ordered_hash(actor.target_proof)
        request = actor(*scenario(2, active=True))
        self.assertEqual(request['hands'][0], ['HARVEST'])
        self.assertTrue(all('pair_choice' not in record for record in actor.last['records']))
        self.assertEqual(actor.override_count, 1)
        self.assertEqual(ordered_hash(actor.target_proof), proof)
        calls = actor.network.score_calls
        with self.assertRaisesRegex(ValueError, 'out-of-order'):
            actor(*scenario(2, active=True))
        self.assertEqual(actor.network.score_calls, calls)

    def test_absent_target_wrong_identity_and_input_hash_fail_closed(self):
        target, _rows, _requests = self.target()
        with self.assertRaises(ValueError):
            self.actor(None, 'W')
        with self.assertRaisesRegex(ValueError, 'model identity'):
            BoundedPairAgent('crop-wheat', CROP_SHA, 'task-harvest', 'e' * 64, target=target, arm='W')
        for defect in ('ordinal', 'hash'):
            for arm in ('H', 'W'):
                with self.subTest(defect=defect, arm=arm):
                    altered = deepcopy(target)
                    altered['ordinal' if defect == 'ordinal' else 'input_sha256'] = 31 if defect == 'ordinal' else '0' * 64
                    actor = self.actor(altered, arm)
                    actor(*scenario(0))
                    with self.assertRaises(ValueError):
                        actor(*scenario(1, active=True))
                    self.assertIsNone(actor.last)
                    self.assertIsNone(actor.target_proof)
                    self.assertEqual(actor.override_count, 0)

    def test_original_proposal_must_be_h_and_temporary_selector_is_restored(self):
        target, _rows, _requests = self.target()
        actor = self.actor(target, 'W')
        actor(*scenario(0))
        original_score, original_selector = actor.network.score, actor.controller.task_selector
        def changed_at_target(shared, candidates):
            # Only the second callback's unit set changes its synthetic logits;
            # the prior callback and the target encoded input stay unchanged.
            if min(candidate[0] for candidate in candidates) == 1:
                return np.zeros(len(candidates))
            return original_score(shared, candidates)
        with patch.object(actor.network, 'score', changed_at_target):
            with self.assertRaisesRegex(ValueError, 'original model proposal'):
                actor(*scenario(1, active=True))
        self.assertIs(actor.controller.task_selector, original_selector)
        self.assertNotEqual(actor.network.last['chosen'], target['harvest_choice'])
        self.assertEqual(actor.override_count, 0)
        self.assertIsNone(actor.target_proof)
        self.assertIsNone(actor.last)
        self.assertEqual(actor.budget.counts['task_forwards'], 2)


if __name__ == '__main__':
    unittest.main()
