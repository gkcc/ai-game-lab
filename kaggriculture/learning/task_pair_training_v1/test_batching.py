"""Synthetic regression tests for the shared anchored batch plan."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_training_v1.batching import make_batches


class BatchingTests(unittest.TestCase):
    @staticmethod
    def fixture():
        eligible = np.zeros(48, dtype=bool)
        eligible[:30] = True
        clusters = np.repeat(np.arange(4, dtype=np.uint32), 12)
        return eligible, clusters

    def test_anchors_cover_eligible_once_before_repeating(self):
        eligible, clusters = self.fixture()
        batches, anchors = make_batches(eligible, clusters, rng_seed=17,
                                        updates=90, batch_size=8)
        first_cycle = anchors[:30]
        self.assertEqual(set(first_cycle), set(np.flatnonzero(eligible)))
        self.assertEqual(len(np.unique(first_cycle)), 30)
        self.assertNotEqual(clusters[anchors[0]], clusters[anchors[1]])
        shuffled = np.random.default_rng(17).permutation(np.flatnonzero(eligible))
        first = shuffled[0]
        second = next(index for index in shuffled if clusters[index] != clusters[first])
        remaining = [index for index in shuffled if index != first and index != second]
        np.testing.assert_array_equal(first_cycle, [first, second, *remaining])
        np.testing.assert_array_equal(anchors[30:60], first_cycle)
        np.testing.assert_array_equal(anchors[60:90], first_cycle)
        np.testing.assert_array_equal(batches[:, 0], anchors)

    def test_each_batch_has_no_repeats_and_can_cover_entire_split(self):
        eligible, clusters = self.fixture()
        for batch_size in (2, 8, len(eligible)):
            batches, anchors = make_batches(eligible, clusters, rng_seed=29,
                                            updates=12, batch_size=batch_size)
            for row, anchor in zip(batches, anchors, strict=True):
                self.assertTrue(eligible[anchor])
                self.assertEqual(len(np.unique(row)), batch_size)
                self.assertTrue(np.all((0 <= row) & (row < len(eligible))))
            if batch_size == len(eligible):
                for row in batches:
                    self.assertEqual(set(row), set(range(len(eligible))))

    def test_reproducibility_and_inputs_are_not_mutated(self):
        eligible, clusters = self.fixture()
        e0, c0 = eligible.copy(), clusters.copy()
        one = make_batches(eligible, clusters, rng_seed=72, updates=10, batch_size=8)
        two = make_batches(eligible, clusters, rng_seed=72, updates=10, batch_size=8)
        other = make_batches(eligible, clusters, rng_seed=73, updates=10, batch_size=8)
        for a, b in zip(one, two, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(one[0], other[0]))
        np.testing.assert_array_equal(eligible, e0)
        np.testing.assert_array_equal(clusters, c0)

    def test_insufficient_eligible_groups_or_clusters_rejected(self):
        eligible, clusters = self.fixture()
        few = eligible.copy()
        few[19:] = False
        with self.assertRaises(ValueError):
            make_batches(few, clusters, rng_seed=1, batch_size=8)
        single_cluster = np.zeros_like(clusters)
        with self.assertRaises(ValueError):
            make_batches(eligible, single_cluster, rng_seed=1, batch_size=8)
        # Multiple clusters in the split cannot substitute for qualifying
        # groups in at least two clusters.
        qualified_only_first = np.zeros_like(eligible)
        qualified_only_first[:24] = True
        clusters[0:24] = 0
        with self.assertRaises(ValueError):
            make_batches(qualified_only_first, clusters, rng_seed=1, batch_size=8)

    def test_invalid_metadata_is_rejected(self):
        eligible, clusters = self.fixture()
        cases = [
            (eligible.astype(np.int8), clusters),
            (eligible, clusters.astype(float)),
            (eligible, clusters.astype(bool)),
            (eligible[:-1], clusters),
            (eligible.reshape(6, 8), clusters),
            (np.array([], dtype=bool), np.array([], dtype=np.uint32)),
            (eligible, np.full(48, -1, dtype=np.int64)),
            (eligible, np.full(48, 2**32, dtype=np.uint64)),
        ]
        for e, c in cases:
            with self.subTest(dtype=(e.dtype, c.dtype), shape=(e.shape, c.shape)):
                with self.assertRaises(ValueError):
                    make_batches(e, c, rng_seed=1, batch_size=8)

    def test_invalid_numeric_settings_rejected(self):
        eligible, clusters = self.fixture()
        bad_settings = [
            {'rng_seed': -1}, {'rng_seed': 2**64}, {'rng_seed': True},
            {'rng_seed': 1, 'updates': 0}, {'rng_seed': 1, 'updates': True},
            {'rng_seed': 1, 'updates': 1025},
            {'rng_seed': 1, 'batch_size': 1}, {'rng_seed': 1, 'batch_size': True},
            {'rng_seed': 1, 'batch_size': 49},
        ]
        for settings in bad_settings:
            settings.setdefault('batch_size', 8)
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    make_batches(eligible, clusters, **settings)


if __name__ == '__main__':
    unittest.main()
