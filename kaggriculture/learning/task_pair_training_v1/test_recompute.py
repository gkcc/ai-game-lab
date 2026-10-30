import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_training_v1.recompute import verify_masks


def shard(first):
    return {'seen_in_train': np.asarray([first, False], dtype=bool),
            'label_conflict': np.asarray([False, False], dtype=bool),
            'qualifying_plant': np.asarray([True, first], dtype=bool)}


class RecomputeTests(unittest.TestCase):
    def setUp(self):
        self.fresh = {'teacher_dev': [shard(True)], 'conditional_dev': [shard(True), shard(False)]}
        self.saved = copy.deepcopy(self.fresh)

    def test_equal_source_bound_masks_pass(self):
        self.assertTrue(verify_masks(self.fresh, self.saved))

    def test_same_length_changed_or_swapped_shards_are_rejected(self):
        self.saved['teacher_dev'][0]['seen_in_train'][1] = True
        with self.assertRaises(ValueError):
            verify_masks(self.fresh, self.saved)
        self.saved = copy.deepcopy(self.fresh)
        self.saved['conditional_dev'].reverse()
        with self.assertRaises(ValueError):
            verify_masks(self.fresh, self.saved)

    def test_conflict_missing_field_and_wrong_dtype_are_rejected(self):
        self.fresh['teacher_dev'][0]['label_conflict'][0] = True
        self.saved = copy.deepcopy(self.fresh)
        with self.assertRaises(ValueError):
            verify_masks(self.fresh, self.saved)
        self.fresh['teacher_dev'][0]['label_conflict'][0] = False
        self.saved = copy.deepcopy(self.fresh)
        del self.saved['teacher_dev'][0]['seen_in_train']
        with self.assertRaises(ValueError):
            verify_masks(self.fresh, self.saved)
        self.saved = copy.deepcopy(self.fresh)
        self.saved['teacher_dev'][0]['seen_in_train'] = self.saved['teacher_dev'][0]['seen_in_train'].astype(np.int8)
        with self.assertRaises(ValueError):
            verify_masks(self.fresh, self.saved)


if __name__ == '__main__':
    unittest.main()
