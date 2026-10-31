"""Synthetic-only split deduplication checks, with no private source reads."""
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_multisource_rank_v1 import dedup
from . import split_dedup


def item(pool, role, value, target=0):
    shared = np.zeros(275, dtype=np.float32)
    shared[0] = value
    candidates = np.zeros((2, 98), dtype=np.float32)
    candidates[1, 0] = 1
    return dedup.Item(shared, candidates, target, pool, role)


def run(items, *, mean=None, scale=None, guard=lambda: None):
    return split_dedup.deduplicate(items,
        mean=np.zeros(373, np.float64) if mean is None else mean,
        scale=np.ones(373, np.float64) if scale is None else scale, guard=guard)


class PoisonTarget:
    def __eq__(self, _other): raise AssertionError('mixed diagnostic target was read')
    def __hash__(self): raise AssertionError('mixed diagnostic target was read')
    def __repr__(self): raise AssertionError('mixed diagnostic target was read')


class SplitDedupTests(unittest.TestCase):
    def test_train_priority_pure_diagnostic_priority_and_original_order(self):
        items = [item('old_train', 'T01', 1), item('old_diag', 'T02', 3),
                 item('new_train', 'A03', 1), item('new_train', 'A04', 2),
                 item('new_dev', 'A09', 3), item('new_dev', 'A10', 4)]
        kept, summary = run(items)
        self.assertEqual([value.role for value in kept], ['A03', 'A04', 'A09', 'A10'])
        self.assertEqual(summary['duplicate_removed_by_pool']['old_train'], 1)
        self.assertEqual(summary['duplicate_removed_by_pool']['old_diag'], 1)
        self.assertEqual(summary['retained_count'], 4)

    def test_train_and_pure_diagnostic_conflicts_remove_only_their_groups(self):
        train_a = item('new_train', 'A03', 1, 0)
        train_b = item('old_train', 'T01', 1, 1)
        mixed = item('new_dev', 'A09', 1, PoisonTarget())
        diag_a = item('new_dev', 'A10', 2, 0)
        diag_b = item('old_diag', 'T02', 2, 1)
        kept, summary = run([train_a, train_b, mixed, diag_a, diag_b])
        self.assertEqual(kept, [])
        self.assertEqual(summary['train_conflict_components'], 1)
        self.assertEqual(summary['diagnostic_conflict_components'], 1)
        self.assertEqual(summary['diagnostic_excluded_train_connected_by_pool']['new_dev'], 1)
        self.assertEqual(summary['conflict_removed_by_pool']['new_train'], 1)
        self.assertEqual(summary['conflict_removed_by_pool']['old_train'], 1)
        self.assertEqual(summary['conflict_removed_by_pool']['new_dev'], 1)
        self.assertEqual(summary['conflict_removed_by_pool']['old_diag'], 1)

    def test_connected_diagnostic_target_is_never_read(self):
        train = item('new_train', 'A03', 1, 0)
        for value in (0, 1, -1, PoisonTarget()):
            diagnostic = item('new_dev', 'A09', 1, value)
            kept, summary = run([diagnostic, train])
            self.assertEqual(len(kept), 1)
            self.assertIs(kept[0], train)
            self.assertEqual(summary['diagnostic_excluded_train_connected_by_pool']['new_dev'], 1)
        # An invalid label on a pure diagnostic component is still corruption.
        with self.assertRaises(split_dedup.SplitDedupError):
            run([train, item('new_dev', 'A09', 2, -1)])

    def test_diagnostic_feature_bridge_never_connects_training_graph(self):
        first = item('old_train', 'T01', 1, 0)
        bridge = item('new_dev', 'A09', 2, PoisonTarget())
        second = item('new_train', 'A03', 3, 1)
        isolated = item('new_dev', 'A09', 4, 0)
        # Synthetic keys make an explicit two-representation bridge. The
        # independent training graph may never use the diagnostic vertex.
        keys = {1: (b'raw-a', b'norm-a'), 2: (b'raw-a', b'norm-b'),
                3: (b'raw-b', b'norm-b'), 4: (b'raw-c', b'norm-c')}
        def representations(value, _mean, _scale): return keys[int(value.shared[0])]
        with patch.object(dedup, '_keys', side_effect=representations):
            kept, summary = run([first, bridge, second])
            self.assertEqual([value.role for value in kept], ['T01', 'A03'])
            self.assertEqual(summary['train_component_count'], 2)
            self.assertEqual(summary['diagnostic_excluded_train_connected_by_pool']['new_dev'], 1)
            changed, _ = run([first, isolated, second])
        self.assertEqual([value.role for value in changed if value.pool in split_dedup.TRAIN], ['T01', 'A03'])

    def test_raw_and_normalized_matches_form_components(self):
        mean = np.zeros(373, np.float64); mean[0] = 2**30
        first = item('old_train', 'T01', 0)
        same_raw = item('new_train', 'A03', 0)
        same_normalized = item('new_train', 'A04', 1)
        kept, summary = run([first, same_raw, same_normalized], mean=mean)
        self.assertEqual([value.role for value in kept], ['A03'])
        self.assertEqual(summary['train_component_count'], 1)
        self.assertEqual(summary['duplicate_removed_by_pool']['old_train'], 1)
        self.assertEqual(summary['duplicate_removed_by_pool']['new_train'], 1)

    def test_hash_collisions_still_require_equal_bytes(self):
        values = [item('old_train', 'T01', 1), item('new_train', 'A03', 2),
                  item('new_dev', 'A09', 3), item('old_diag', 'T02', 4)]
        with patch.object(dedup, '_digest', return_value=b'x' * 32):
            kept, summary = run(values)
        self.assertEqual([value.role for value in kept], [value.role for value in values])
        self.assertEqual(summary['retained_count'], 4)

    def test_guards_shape_labels_and_budget_fail_closed(self):
        signal = MemoryError('synthetic guard')
        def stop(): raise signal
        with self.assertRaises(MemoryError) as caught: run([item('new_train', 'A03', 1)], guard=stop)
        self.assertIs(caught.exception, signal)
        with self.assertRaises(split_dedup.SplitDedupError):
            run([item('new_train', 'A03', 1)] * (dedup.MAX_GROUPS + 1))
        with self.assertRaises(split_dedup.SplitDedupError):
            run([item('new_train', 'A03', 1, True)])
        with self.assertRaises(split_dedup.SplitDedupError):
            run([item('new_train', 'A03', 1)], scale=np.zeros(373, np.float64))
        bad = item('new_train', 'A03', 1)
        bad.shared[0] = np.nan
        with self.assertRaises(split_dedup.SplitDedupError): run([bad])


if __name__ == '__main__':
    unittest.main()
