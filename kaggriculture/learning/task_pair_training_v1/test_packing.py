from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_data_preflight_v1.test_core import fixture
from learning.task_pair_training_v1 import packing
from learning.task_pair_training_v1.targets import qualifying_groups


class PackingTests(unittest.TestCase):
    def setUp(self):
        arrays, _, _ = fixture()
        self.data = arrays[0]
        self.mean = np.arange(373, dtype=np.float64) / 1000
        self.scale = np.ones(373, np.float64) * 2
        self.weights = np.asarray([1, 2, 3, 4, 5], np.float32)
        self.eligible = qualifying_groups(self.data, width=4)

    def test_rows_reconstruct_from_original_ids(self):
        parts = packing.pack_training_batch(self.data, np.asarray([3, 0, 1]), self.mean,
                                           self.scale, self.weights, self.eligible)
        self.assertEqual(sum(len(part['ids']) for part in parts), 3)
        for part in parts:
            for row, index in enumerate(part['ids']):
                lo, hi = self.data['offsets'][index:index + 2]
                length = int(hi - lo)
                raw = np.concatenate((np.broadcast_to(self.data['shared'][index], (length, 275)),
                                      self.data['candidates'][lo:hi]), axis=1)
                np.testing.assert_array_equal(part['features'][row, :length],
                                              ((raw - self.mean) / self.scale).astype(np.float32))
                self.assertEqual(int(part['labels'][row]), int(self.data['labels'][index]))
                self.assertEqual(bool(part['eligible'][row]), bool(self.eligible[index]))
                self.assertEqual(float(part['group_weights'][row]),
                                 float(self.weights[self.data['selected_ops'][index]]))

    def test_mismatched_feature_row_with_correct_ids_is_rejected(self):
        original = packing.train.padded

        def corrupt(*args, **kwargs):
            features, mask, labels, weights = original(*args, **kwargs)
            features[0, 0, 0] += 1
            return features, mask, labels, weights

        with patch.object(packing.train, 'padded', side_effect=corrupt):
            with self.assertRaises(ValueError):
                packing.pack_training_batch(self.data, np.asarray([3, 0, 1]), self.mean,
                                            self.scale, self.weights, self.eligible)

    def test_invalid_ids_or_changed_eligibility_fail(self):
        with self.assertRaises(ValueError):
            packing.pack_training_batch(self.data, np.asarray([0, 0]), self.mean,
                                        self.scale, self.weights, self.eligible)
        altered = self.eligible.copy()
        altered[0] = not altered[0]
        with self.assertRaises(ValueError):
            packing.pack_training_batch(self.data, np.asarray([0]), self.mean,
                                        self.scale, self.weights, altered)

    def test_wrong_target_operation_is_rejected_even_with_teacher_in_mask(self):
        original = packing.plant_targets

        def corrupt(*args, **kwargs):
            active, targets = original(*args, **kwargs)
            targets[0, 2] = True  # This candidate is WATER, not PLANT.
            return active, targets

        with patch.object(packing, 'plant_targets', side_effect=corrupt):
            with self.assertRaises(ValueError):
                packing.pack_training_batch(self.data, np.asarray([0]), self.mean,
                                            self.scale, self.weights, self.eligible)

    def test_missing_group_is_rejected_even_without_fit_core(self):
        original = packing.train.microbatches

        def omit(data, ids):
            for group_ids, capacity, width in original(data, ids):
                yield group_ids[:-1], capacity, width

        with patch.object(packing.train, 'microbatches', side_effect=omit):
            with self.assertRaises(ValueError):
                packing.pack_training_batch(self.data, np.asarray([0, 1, 2]), self.mean,
                                            self.scale, self.weights, self.eligible)


if __name__ == '__main__':
    unittest.main()
