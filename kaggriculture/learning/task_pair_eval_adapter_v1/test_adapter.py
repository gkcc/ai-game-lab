"""In-memory evaluation adapter regression; no checkpoint, GPU or games."""

from dataclasses import fields
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_data_preflight_v1 import core
from learning.task_pair_data_preflight_v1.test_core import SCHEMA, make_arrays, record
from learning.task_pair_eval_adapter_v1 import adapter
from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork


def sample_view(*, conditional=False, clusters=(11, 22), sizes=(30, 10)):
    records = [record(i, 0, cluster)
               for cluster, size in zip(clusters, sizes) for i in range(size)]
    arrays = make_arrays(records, conditional=conditional)
    make = core.RaggedView.from_conditional if conditional else core.RaggedView.from_cache
    return make(arrays, SCHEMA)


def source_masks(count, seen=()):
    seen_mask = np.zeros(count, dtype=bool)
    seen_mask[list(seen)] = True
    return {'seen_in_train': seen_mask, 'label_conflict': np.zeros(count, dtype=bool),
            'qualifying_plant': np.ones(count, dtype=bool)}


def score_blocks(view, *, incorrect=(), water_wins=False, tie=False, choices=False,
                 max_groups=7, max_candidates=21):
    """Synthetic scores independent of teacher labels and cluster metadata."""
    incorrect = set(incorrect)
    for part in adapter.feature_blocks(view, expected_schema_sha256=SCHEMA['schema_sha256'],
                                       max_groups=max_groups, max_candidates=max_candidates):
        scores, predicted = [], []
        for i in range(part.stop - part.start):
            index = part.start + i
            group = [1., 2., 0.] if index in incorrect else [2., 1., 0.]
            if tie:
                group[:2] = [2., 2.]
            if water_wins:
                group[2] = 9.
            assert len(group) == part.offsets[i + 1] - part.offsets[i]
            scores.extend(group)
            predicted.append(int(np.argmax(group)))
        yield adapter.LogitBlock(part.start, part.stop, np.asarray(scores, np.float64),
                                 np.asarray(predicted, np.int64) if choices else None)


def run(view, blocks, masks, *, source='teacher_dev', clusters=(11, 22)):
    return adapter.evaluate_source(view, blocks, masks, source=source,
                                   expected_clusters=clusters,
                                   expected_schema_sha256=SCHEMA['schema_sha256'])


class AdapterTests(unittest.TestCase):
    def test_known_cluster_weighted_all_and_unseen_values(self):
        view = sample_view()
        seen = range(10, 30)
        result = run(view, score_blocks(view, incorrect=seen, choices=True), source_masks(40, seen))
        self.assertEqual(result['source'], 'teacher_dev')
        self.assertAlmostEqual(result['all']['full_top1']['value'], 2 / 3)
        self.assertEqual(result['unseen_train']['full_top1']['value'], 1.0)
        self.assertEqual(result['all']['cross_operation_recall']['PLANT']['value'], 1.0)

    def test_feature_blocks_preserve_whole_groups_and_hide_supervision(self):
        view = sample_view()
        parts = list(adapter.feature_blocks(view, expected_schema_sha256=SCHEMA['schema_sha256'],
                                            max_groups=4, max_candidates=5))
        self.assertEqual([field.name for field in fields(adapter.FeatureBlock)],
                         ['start', 'stop', 'shared', 'candidates', 'offsets'])
        self.assertEqual(parts[0].start, 0)
        self.assertEqual(parts[-1].stop, len(view.shared))
        self.assertTrue(all(part.stop - part.start == 1 for part in parts))
        for part in parts:
            self.assertEqual(part.shared.shape, (1, 275))
            self.assertEqual(part.candidates.shape, (3, 98))
            self.assertEqual(part.offsets.tolist(), [0, 3])
            self.assertTrue(all(not x.flags.writeable for x in (part.shared, part.candidates, part.offsets)))
            self.assertFalse(hasattr(part, 'labels'))
            self.assertFalse(hasattr(part, 'private_clusters'))
            self.assertFalse(hasattr(part, 'qualifying_plant'))
            np.testing.assert_array_equal(part.shared[0], view.shared[part.start])
        with self.assertRaises(ValueError):
            list(adapter.feature_blocks(view, expected_schema_sha256=SCHEMA['schema_sha256'],
                                        max_candidates=2))

    def test_actual_numpy_scorer_accepts_only_synthetic_feature_rows(self):
        # Construct an in-memory toy network; no model/checkpoint is opened.
        model = TaskNetwork.__new__(TaskNetwork)
        model.mean = np.zeros(373, dtype=np.float64)
        model.scale = np.ones(373, dtype=np.float64)
        model.params = {}
        for layer in range(3):
            model.params['w' + str(layer)] = np.zeros((DIMENSIONS[layer], DIMENSIONS[layer + 1]))
            model.params['b' + str(layer)] = np.zeros(DIMENSIONS[layer + 1])
        view = sample_view(sizes=(10, 10))
        blocks = []
        for part in adapter.feature_blocks(view, expected_schema_sha256=SCHEMA['schema_sha256']):
            per_group = [model.score(part.shared[i], part.candidates[part.offsets[i]:part.offsets[i + 1]])
                         for i in range(part.stop - part.start)]
            blocks.append(adapter.LogitBlock(part.start, part.stop, np.concatenate(per_group)))
        result = run(view, blocks, source_masks(20))
        self.assertEqual(result['all']['full_top1']['value'], 1.0)  # First candidate wins ties.

    def test_old_and_new_are_evaluated_separately(self):
        old = sample_view(conditional=True)
        new = sample_view(conditional=True, clusters=(33, 44))
        old_result = run(old, score_blocks(old), source_masks(40),
                         source='conditional_old', clusters=(11, 22))
        new_result = run(new, score_blocks(new, incorrect=range(40)), source_masks(40),
                         source='conditional_new', clusters=(33, 44))
        self.assertEqual(old_result['all']['full_top1']['value'], 1.0)
        self.assertEqual(new_result['all']['full_top1']['value'], 0.0)
        self.assertEqual(old_result['source'], 'conditional_old')
        self.assertEqual(new_result['source'], 'conditional_new')
        with self.assertRaises(ValueError):
            run(old, score_blocks(old), source_masks(40), source='conditional_dev')
        with self.assertRaises(ValueError):
            run(old, score_blocks(old), source_masks(40), source='train')

    def test_missing_expected_cluster_and_unseen_subset_remain_unknown(self):
        view = sample_view()
        only_first = source_masks(40, range(30, 40))
        result = run(view, score_blocks(view), only_first)
        self.assertEqual(result['all']['full_top1']['value'], 1.0)
        self.assertEqual(result['unseen_train']['full_top1'], {'support': 'unknown', 'value': None})
        result = run(view, score_blocks(view), source_masks(40), clusters=(11, 22, 33))
        self.assertEqual(result['all']['full_top1'], {'support': 'unknown', 'value': None})

    def test_global_choice_cannot_replace_conditional_ranking_logits(self):
        view = sample_view()
        first = run(view, score_blocks(view, water_wins=True, choices=True), source_masks(40))
        second = run(view, score_blocks(view, water_wins=True, incorrect=range(40), choices=True),
                     source_masks(40))
        self.assertEqual(first['all']['full_top1']['value'], 0.0)
        self.assertEqual(second['all']['full_top1']['value'], 0.0)
        self.assertEqual(first['all']['plant_same_target_unit']['value'], 1.0)
        self.assertEqual(second['all']['plant_same_target_unit']['value'], 0.0)

    def test_first_maximum_tie_choices_and_choice_mismatch(self):
        view = sample_view()
        blocks = list(score_blocks(view, tie=True, choices=True))
        result = run(view, blocks, source_masks(40))
        self.assertEqual(result['all']['full_top1']['value'], 1.0)
        first = blocks[0]
        wrong = first.choices.copy()
        wrong[0] = 1
        blocks[0] = adapter.LogitBlock(first.start, first.stop, first.logits, wrong)
        with self.assertRaises(ValueError):
            run(view, blocks, source_masks(40))

    def test_missing_duplicate_out_of_order_and_invalid_scores_fail(self):
        view = sample_view()
        blocks = list(score_blocks(view))
        first = blocks[0]
        bad = first.logits.copy()
        bad[0] = np.nan
        for invalid in (blocks[:-1], blocks[1:], [blocks[0], blocks[0]] + blocks[1:],
                        [adapter.LogitBlock(first.start, first.stop, bad)] + blocks[1:],
                        [adapter.LogitBlock(first.start, first.stop, first.logits[:-1])] + blocks[1:],
                        [adapter.LogitBlock(first.start, first.stop, first.logits.astype(np.int64))] + blocks[1:]):
            with self.subTest(case=len(invalid)), self.assertRaises(ValueError):
                run(view, invalid, source_masks(40))
        choices = list(score_blocks(view, choices=True))
        choices[0] = adapter.LogitBlock(choices[0].start, choices[0].stop, choices[0].logits)
        with self.assertRaises(ValueError):
            run(view, choices, source_masks(40))

    def test_unbound_masks_schema_or_population_are_rejected(self):
        view = sample_view()
        incorrect = source_masks(40)
        incorrect['qualifying_plant'][0] = False
        with self.assertRaises(ValueError):
            run(view, score_blocks(view), incorrect)
        conflict = source_masks(40)
        conflict['label_conflict'][0] = True
        for mask in (conflict, {'qualifying_plant': np.ones(40, dtype=bool)}):
            with self.assertRaises(ValueError):
                run(view, score_blocks(view), mask)
        for clusters in ((11, 11), (11, True), (), (11, 33)):
            with self.assertRaises(ValueError):
                run(view, score_blocks(view), source_masks(40), clusters=clusters)
        with self.assertRaises(ValueError):
            adapter.evaluate_source(view, score_blocks(view), source_masks(40), source='teacher_dev',
                                    expected_clusters=(11, 22), expected_schema_sha256='f' * 64)

    def test_oversized_view_is_rejected_before_score_allocation(self):
        rows = adapter.MAX_CANDIDATES + 1
        shared = np.zeros((1, 275), np.float32)
        candidates = np.broadcast_to(np.zeros((1, 98), np.float32), (rows, 98))
        ops = np.broadcast_to(np.zeros(1, np.int8), (rows,))
        view = core.RaggedView(shared, candidates, np.asarray([0, rows], np.int64),
                               np.asarray([0], np.int64), ops, np.asarray([11], np.int64),
                               tuple(SCHEMA['shared_feature_names']), tuple(SCHEMA['feature_names']),
                               SCHEMA['schema_sha256'])
        with patch.object(adapter.np, 'empty', side_effect=AssertionError('Must not allocate')):
            with self.assertRaises(ValueError):
                run(view, (), source_masks(1), clusters=(11,))


if __name__ == '__main__':
    unittest.main()
