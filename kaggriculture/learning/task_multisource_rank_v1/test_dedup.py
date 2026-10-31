"""Pure synthetic regressions; no real arrays, checkpoints or games."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_multisource_rank_v1 import dedup as d


def item(value, *, pool='old_train', role='R1', target=0, width=2):
    shared = np.zeros(d.SHARED_DIM, dtype=np.float32)
    candidates = np.zeros((width, d.LOCAL_DIM), dtype=np.float32)
    candidates[:, 0] = value + np.arange(width, dtype=np.float32)
    return d.Item(shared, candidates, target, pool, role)


class DedupTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.zeros(d.DIM, dtype=np.float64)
        self.scale = np.ones(d.DIM, dtype=np.float64)

    def dedup(self, values, guard=lambda: None):
        return d.deduplicate(values, mean=self.mean, scale=self.scale, guard=guard)

    def test_train_priority_preserves_original_order_and_no_array_copy(self):
        old_diag = item(1, pool='old_diag')
        new_dev = item(1, pool='new_dev')
        new_train = item(1, pool='new_train')
        old_train = item(1, pool='old_train')
        other = item(20, pool='old_diag')
        retained, summary = self.dedup([old_diag, new_dev, new_train, old_train, other])
        self.assertEqual(len(retained), 2)
        self.assertIs(retained[0], new_train)
        self.assertIs(retained[1], other)
        self.assertIs(retained[0].candidates, new_train.candidates)
        self.assertEqual(summary['input_by_pool'],
                         {'old_train': 1, 'new_train': 1, 'old_diag': 2, 'new_dev': 1})
        self.assertEqual(summary['duplicate_removed_by_pool'],
                         {'old_train': 1, 'new_train': 0, 'old_diag': 1, 'new_dev': 1})
        self.assertEqual(summary['retained_count'], 2)
        self.assertNotIn('R1', repr(summary))

    def test_new_dev_beats_old_diag_without_training_group(self):
        first = item(7, pool='old_diag')
        second = item(7, pool='new_dev')
        retained, summary = self.dedup([first, second])
        self.assertEqual(retained, [second])
        self.assertEqual(summary['duplicate_removed_by_pool']['old_diag'], 1)

    def test_transitive_raw_and_normalized_link_removes_entire_conflict(self):
        # B exactly matches A. C differs raw but collides after normalization;
        # C's incompatible target must remove A, B and C as one component.
        self.mean[d.SHARED_DIM] = 2**30
        a = item(1, pool='new_dev', target=0)
        b = item(1, pool='old_train', target=0)
        c = item(2, pool='old_diag', target=1)
        unrelated = item(5, pool='new_train')
        unrelated.candidates[:, 1] = 10
        retained, summary = self.dedup([a, b, c, unrelated])
        self.assertEqual(retained, [unrelated])
        self.assertEqual(summary['conflict_components'], 1)
        self.assertEqual(sum(summary['conflict_removed_by_pool'].values()), 3)
        self.assertEqual(sum(summary['duplicate_removed_by_pool'].values()), 0)

    def test_positive_and_negative_zero_are_the_same_key(self):
        first, second = item(3, pool='new_dev'), item(3, pool='old_train')
        second.shared[0] = -0.0
        second.candidates[0, 1] = -0.0
        self.assertTrue(np.signbit(second.shared[0]))
        retained, summary = self.dedup([first, second])
        self.assertEqual(retained, [second])
        self.assertEqual(summary['duplicate_removed_by_pool']['new_dev'], 1)

    def test_forced_digest_collisions_do_not_merge_distinct_groups(self):
        values = [item(1), item(10, pool='old_diag'), item(20, pool='new_dev'),
                  item(10, pool='new_train')]
        # A digest is only an index, including after deliberately forcing all
        # raw and normalized keys into the same bucket. The last item must
        # skip the earlier unequal key and still match the later equal key.
        with patch.object(d, '_digest', return_value=b'x' * 32):
            retained, summary = self.dedup(values)
        self.assertEqual(retained, [values[0], values[2], values[3]])
        self.assertEqual(summary['component_count'], 3)
        self.assertEqual(summary['duplicate_removed_by_pool']['old_diag'], 1)

    def test_order_and_conflicting_target_both_matter(self):
        first = item(1, pool='old_train')
        reversed_group = item(1, pool='new_dev')
        reversed_group.candidates[:] = reversed_group.candidates[::-1].copy()
        retained, _summary = self.dedup([first, reversed_group])
        self.assertEqual(retained, [first, reversed_group])
        conflict = item(1, pool='new_dev', target=1)
        retained, summary = self.dedup([first, conflict])
        self.assertEqual(retained, [])
        self.assertEqual(summary['conflict_components'], 1)
        self.assertEqual(summary['conflict_removed_by_pool']['old_train'], 1)
        self.assertEqual(summary['conflict_removed_by_pool']['new_dev'], 1)

    def test_invalid_inputs_fail_without_partial_result(self):
        good = item(1)
        invalid = [replace(good, shared=np.zeros(274, dtype=np.float32)),
            replace(good, shared=good.shared.astype(np.float64)),
            replace(good, candidates=good.candidates.astype(np.float64)),
            replace(good, candidates=np.zeros((0, 98), dtype=np.float32)),
            replace(good, candidates=np.zeros((257, 98), dtype=np.float32)),
            replace(good, candidates=np.zeros((2, 99), dtype=np.float32)),
            replace(good, target=True), replace(good, target=-1), replace(good, target=2),
            replace(good, pool='unknown'), replace(good, role='../private')]
        bad = item(1); bad.candidates[0, 0] = np.nan; invalid.append(bad)
        for value in invalid:
            with self.subTest(kind=repr(value)), self.assertRaises(ValueError): self.dedup([value])
        with self.assertRaises(ValueError): self.dedup([good] * (d.MAX_GROUPS + 1))
        for mean, scale in ((self.mean.astype(np.float32), self.scale),
                            (self.mean, np.zeros(d.DIM, dtype=np.float64)),
                            (np.full(d.DIM, np.nan), self.scale)):
            with self.assertRaises(ValueError):
                d.deduplicate([good], mean=mean, scale=scale, guard=lambda: None)

    def test_guard_failure_propagates_and_input_mutation_is_detected(self):
        class Stop(Exception): pass
        with self.assertRaises(Stop):
            self.dedup([item(1)], guard=lambda: (_ for _ in ()).throw(Stop()))
        values = [item(1), item(20)]
        calls = []
        def mutate():
            calls.append(None)
            if len(calls) == 3:
                values[0].candidates[0, 0] += 100
        with self.assertRaises(ValueError): self.dedup(values, guard=mutate)


if __name__ == '__main__':
    unittest.main()
