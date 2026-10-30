from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_choice_v1.features import OPS
from learning.task_pair_training_v1.targets import plant_targets, qualifying_groups


class TargetMaskTests(unittest.TestCase):
    def setUp(self):
        plant, water = OPS.index('PLANT'), OPS.index('WATER')
        self.data = {
            'offsets': np.asarray([0, 4, 7, 8], dtype=np.int64),
            'labels': np.asarray([0, 0, 0], dtype=np.int64),
            'candidate_ops': np.asarray([plant, plant, water, plant, water, plant, plant, plant], dtype=np.int8),
            'candidate_identity': np.asarray([
                [0, 1, 1], [1, 1, 1], [0, 1, 1], [2, 2, 2],
                [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=np.int16),
        }

    def test_only_same_target_plant_is_available_in_eligible_group(self):
        eligible, mask = plant_targets(self.data, [0, 1, 2], 4, 4,
                                       np.asarray([True, False, False]))
        np.testing.assert_array_equal(eligible, [True, False, False, False])
        np.testing.assert_array_equal(mask, [[True, True, False, False],
                                             [False, False, False, False],
                                             [False, False, False, False],
                                             [False, False, False, False]])

    def test_group_order_is_respected(self):
        eligible, mask = plant_targets(self.data, [2, 0], 2, 4)
        np.testing.assert_array_equal(eligible, [False, True])
        np.testing.assert_array_equal(mask[1], [True, True, False, False])

    def test_rejects_truncation_duplicate_ids_and_changed_plan(self):
        with self.assertRaises(ValueError):
            plant_targets(self.data, [0], 1, 3)
        with self.assertRaises(ValueError):
            plant_targets(self.data, [0, 0], 2, 4)
        with self.assertRaises(ValueError):
            plant_targets(self.data, [0], 1, 4, np.asarray([False, False, False]))
        with self.assertRaises(ValueError):
            plant_targets(self.data, [0], 4097, 4)
        with self.assertRaises(ValueError):
            qualifying_groups(self.data, width=257)
        with self.assertRaises(ValueError):
            qualifying_groups(self.data, chunk_size=4097)

    def test_rejects_duplicate_units_on_the_same_plant_target(self):
        self.data['candidate_identity'][1, 0] = 0
        with self.assertRaises(ValueError):
            plant_targets(self.data, [0], 1, 4)

    def test_anchor_eligibility_matches_preflight_on_validated_groups(self):
        from learning.task_pair_data_preflight_v1 import core
        from learning.task_pair_data_preflight_v1.test_core import fixture
        arrays, views, _ = fixture()
        expected = np.asarray([core._group(views[0], i)[4] for i in range(len(views[0].labels))])
        np.testing.assert_array_equal(qualifying_groups(arrays[0], width=4, chunk_size=3), expected)
        np.testing.assert_array_equal(qualifying_groups(arrays[0], width=4, chunk_size=32), expected)


if __name__ == '__main__':
    unittest.main()
