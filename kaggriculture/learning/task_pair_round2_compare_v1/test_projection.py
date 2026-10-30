"""Synthetic paired accounting, gate and privacy checks; no real games."""
from copy import deepcopy
import json
import unittest

from learning.task_pair_historical_compare_v1.test_projection import SEEDS, make_row, make_receipt
from .projection import project


def fixture():
    rows = [make_row(alias, side, arm) for alias in SEEDS for side in (0, 1)
            for arm in ('baseline', 'control')]
    for row in rows:
        row['opponent'] = 'round2'
        row['game'] = row['game'].replace('-market-', '-round2-')
    comparisons, details = [], {}
    for alias in SEEDS:
        for side in (0, 1):
            receipt = make_receipt(alias, side, rows, 'choice_difference')
            item = receipt['comparison']
            item['opponent'] = 'round2'
            comparisons.append(item)
            details[alias, side] = dict(scope='strict_common_recorded_history', trace_turns_per_game=719,
                matched_selection_records=item['result']['matched_calls'], stop_reason='choice_difference',
                first_stop_step=7, first_different_recorded_choice=dict(
                    baseline_op='WATER', control_op='WATER', same_unit=False, same_target=True,
                    selected_unit_request_ordered_equal=False, complete_actor_request_equal=False,
                    complete_opponent_request_equal=True, both_model_callbacks=True),
                complete_request_prefix=dict(scope='same_complete_history', status='observed',
                                             reason='actor_request_difference', step=7))
    return rows, comparisons, details


class ProjectionTests(unittest.TestCase):
    def test_eight_games_four_pairs_and_privacy(self):
        rows, comparisons, details = fixture()
        rows[0]['private_path'] = 'PRIVATE-PATH'
        rows[0]['result']['private_observation'] = 'PRIVATE-OBSERVATION'
        result = project(rows=rows, seeds=SEEDS, comparisons=comparisons, details=details)
        self.assertEqual((result['games'], result['seed_clusters'], len(result['pairs'])), (8, 2, 4))
        self.assertEqual(result['mean_delta']['own_net_profit'], 6)
        self.assertTrue(result['exploration_gate']['passed'])
        self.assertFalse(result['promotion'])
        self.assertEqual(result['statistical_significance'], 'unknown')
        raw = json.dumps(result)
        for secret in (str(SEEDS['S02']), str(SEEDS['S03']), 'PRIVATE-PATH',
                       'PRIVATE-OBSERVATION', 'PRIVATE-REQUEST', rows[0]['task_checkpoint_sha256']):
            self.assertNotIn(secret, raw)

    def test_missing_observed_request_on_a_seed_closes_gate(self):
        rows, comparisons, details = fixture()
        for side in (0, 1):
            details['S02', side]['complete_request_prefix'] = dict(
                scope='same_complete_history', status='unknown', reason='frame_difference', step=None)
        result = project(rows=rows, seeds=SEEDS, comparisons=comparisons, details=details)
        self.assertFalse(result['exploration_gate']['checks']['each_seed_actor_request_difference_observed'])
        self.assertFalse(result['exploration_gate']['passed'])

    def test_nonpositive_baseline_keeps_result_but_cannot_pass_fractional_gate(self):
        rows, comparisons, details = fixture()
        for row in rows:
            if row['seed_alias'] == 'S02' and row['arm'] == 'baseline':
                own, other = row['result']['economy']['own'], row['result']['economy']['opponent']
                own['final_cash'] = own['initial_cash']
                own['net_profit'] = 0
                row['result']['margin'] = own['final_cash'] - other['final_cash']
                row['result']['score'] = 0
        result = project(rows=rows, seeds=SEEDS, comparisons=comparisons, details=details)
        self.assertIsNone(result['seed_checks']['S01']['own_loss_within_one_percent'])
        self.assertFalse(result['exploration_gate']['passed'])
        self.assertEqual(len(result['pairs']), 4)

    def test_missing_duplicate_or_cross_opponent_data_is_rejected(self):
        rows, comparisons, details = fixture()
        for invalid in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                project(rows=invalid, seeds=SEEDS, comparisons=comparisons, details=details)
        altered = deepcopy(comparisons)
        altered[0]['opponent'] = 'market'
        with self.assertRaises(ValueError):
            project(rows=rows, seeds=SEEDS, comparisons=altered, details=details)
        altered_detail = deepcopy(details)
        altered_detail['S02', 0]['matched_selection_records'] += 1
        with self.assertRaises(ValueError):
            project(rows=rows, seeds=SEEDS, comparisons=comparisons, details=altered_detail)


if __name__ == '__main__':
    unittest.main()
