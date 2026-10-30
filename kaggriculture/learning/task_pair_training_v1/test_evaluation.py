"""Synthetic regression for private equal-cluster developmental metrics."""
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_choice_v1.features import OPS
from learning.task_pair_training_v1.evaluation import evaluate


PLANT = OPS.index('PLANT')
WATER = OPS.index('WATER')
STANDARD = ((0, 1, 1, PLANT), (1, 1, 1, PLANT),
            (0, 1, 1, WATER), (2, 2, 2, PLANT))


def group(cluster, scores, *, label=0, candidates=STANDARD, qualifying=True):
    return cluster, candidates, label, scores, qualifying


def fixture(groups):
    offsets = [0]
    for _, candidates, _, _, _ in groups:
        offsets.append(offsets[-1] + len(candidates))
    candidates = np.asarray([row[:3] for item in groups for row in item[1]], np.float32)
    ops = np.asarray([row[3] for item in groups for row in item[1]], np.uint8)
    view = types.SimpleNamespace(
        shared=np.zeros((len(groups), 1), np.float32),
        candidates=candidates,
        offsets=np.asarray(offsets, np.int64),
        labels=np.asarray([item[2] for item in groups], np.int64),
        ops=ops,
        private_clusters=np.asarray([item[0] for item in groups], np.uint32),
        feature_names=('unit.index', 'target.x', 'target.y'),
    )
    logits = np.asarray([score for item in groups for score in item[3]], np.float64)
    qualifying = np.asarray([item[4] for item in groups], dtype=bool)
    return view, logits, qualifying


class EvaluationTests(unittest.TestCase):
    def calculate(self, groups, *, selected=None, expected=(11, 22)):
        view, logits, qualifying = fixture(groups)
        if selected is None:
            selected = np.ones(len(groups), dtype=bool)
        return evaluate(view, logits, selected, qualifying, expected)

    def test_equal_weight_per_cluster_and_distinct_metric_meanings(self):
        groups = [group(11, [5, 1, 0, -1]) for _ in range(30)]
        groups += [group(22, [1, 5, 0, -1]) for _ in range(10)]
        result = self.calculate(groups)
        self.assertEqual(result['full_top1'], {'support': 'supported', 'value': 0.5})
        self.assertNotEqual(result['full_top1']['value'], 0.75)  # Group-weighted value.
        self.assertEqual(result['full_target_operation']['value'], 1.0)
        self.assertEqual(result['plant_same_target_unit']['value'], 0.5)
        self.assertEqual(result['cross_operation_recall']['PLANT']['value'], 1.0)
        self.assertEqual(result['cross_operation_recall']['WATER'],
                         {'support': 'unknown', 'value': None})

    def test_equal_logits_choose_first_in_original_order(self):
        groups = [group(cluster, [2, 2, 0, -1]) for cluster in (11, 22) for _ in range(10)]
        result = self.calculate(groups)
        self.assertEqual(result['full_top1']['value'], 1.0)
        self.assertEqual(result['plant_same_target_unit']['value'], 1.0)

    def test_global_operation_and_target_errors_do_not_change_conditional_ranking(self):
        groups = [group(11, [5, 1, 9, 0]) for _ in range(10)]
        groups += [group(22, [5, 1, 0, 9]) for _ in range(10)]
        result = self.calculate(groups)
        self.assertEqual(result['full_top1']['value'], 0.0)
        self.assertEqual(result['full_target_operation']['value'], 0.0)
        self.assertEqual(result['plant_same_target_unit']['value'], 1.0)
        self.assertEqual(result['cross_operation_recall']['PLANT']['value'], 0.5)

    def test_recall_requires_real_cross_operation_competition(self):
        plant_only = ((0, 1, 1, PLANT), (1, 1, 1, PLANT))
        groups = [group(cluster, [5, 1], candidates=plant_only)
                  for cluster in (11, 22) for _ in range(10)]
        groups += [group(11, [0, 1, 3, 0], label=2, qualifying=False) for _ in range(8)]
        groups += [group(22, [0, 1, 3, 0], label=2, qualifying=False) for _ in range(6)]
        groups += [group(22, [0, 3, 1, 0], label=2, qualifying=False) for _ in range(6)]
        result = self.calculate(groups)
        self.assertEqual(result['cross_operation_recall']['PLANT'],
                         {'support': 'unknown', 'value': None})
        self.assertEqual(result['cross_operation_recall']['WATER'],
                         {'support': 'supported', 'value': 0.75})
        self.assertEqual(result['plant_same_target_unit']['support'], 'supported')

    def test_missing_or_undersized_expected_cluster_is_unknown(self):
        groups = [group(11, [5, 1, 0, -1]) for _ in range(16)]
        groups += [group(22, [5, 1, 0, -1]) for _ in range(10)]
        selected = np.asarray([True] * 20 + [False] * 6, dtype=bool)
        result = self.calculate(groups, selected=selected)
        for name in ('full_top1', 'full_target_operation', 'plant_same_target_unit'):
            self.assertEqual(result[name], {'support': 'unknown', 'value': None})
        selected[-6:] = True
        result = self.calculate(groups, selected=selected, expected=(11, 22, 33))
        self.assertEqual(result['full_top1'], {'support': 'unknown', 'value': None})

    def test_selected_unmatched_subset_is_not_all_group_metric(self):
        groups = [group(11, [5, 1, 0, -1]) for _ in range(10)]
        groups += [group(11, [1, 5, 0, -1]) for _ in range(20)]
        groups += [group(22, [5, 1, 0, -1]) for _ in range(10)]
        all_groups = self.calculate(groups)
        selected = np.asarray([True] * 10 + [False] * 20 + [True] * 10, dtype=bool)
        unseen_groups = self.calculate(groups, selected=selected)
        self.assertAlmostEqual(all_groups['full_top1']['value'], 2 / 3)
        self.assertEqual(unseen_groups['full_top1']['value'], 1.0)

    def test_invalid_inputs_and_unbound_qualifying_mask_fail(self):
        groups = [group(cluster, [5, 1, 0, -1]) for cluster in (11, 22) for _ in range(10)]
        view, logits, qualifying = fixture(groups)
        selected = np.ones(len(groups), dtype=bool)
        wrong = qualifying.copy()
        wrong[0] = False
        with self.assertRaises(ValueError):
            evaluate(view, logits, selected, wrong, (11, 22))
        with self.assertRaises(ValueError):
            evaluate(view, logits, selected.astype(np.uint8), qualifying, (11, 22))
        nonfinite = logits.copy()
        nonfinite[0] = np.nan
        with self.assertRaises(ValueError):
            evaluate(view, nonfinite, selected, qualifying, (11, 22))
        with self.assertRaises(ValueError):
            evaluate(view, logits, selected, qualifying, (11, 33))
        with self.assertRaises(ValueError):
            evaluate(view, logits, selected, qualifying, (11, 11))
        original_label = view.labels[0]
        view.labels[0] = -1
        with self.assertRaises(ValueError):
            evaluate(view, logits, selected, qualifying, (11, 22))
        view.labels[0] = original_label
        view.feature_names = ('unit.index', 'target.x', 'missing')
        with self.assertRaises(ValueError):
            evaluate(view, logits, selected, qualifying, (11, 22))

    def test_duplicate_semantic_candidate_is_rejected(self):
        duplicate = ((0, 1, 1, PLANT), (0, 1, 1, PLANT))
        groups = [group(cluster, [5, 1], candidates=duplicate, qualifying=False)
                  for cluster in (11, 22) for _ in range(10)]
        with self.assertRaises(ValueError):
            self.calculate(groups)

    def test_empty_selection_and_single_cluster_are_unknown(self):
        groups = [group(11, [5, 1, 0, -1]) for _ in range(20)]
        empty = np.zeros(len(groups), dtype=bool)
        self.assertEqual(self.calculate(groups, selected=empty, expected=(11, 22))['full_top1'],
                         {'support': 'unknown', 'value': None})
        self.assertEqual(self.calculate(groups, expected=(11,))['full_top1'],
                         {'support': 'unknown', 'value': None})


if __name__ == '__main__':
    unittest.main()
