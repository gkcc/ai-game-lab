"""Synthetic boundaries for coarse developmental coverage projections."""
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_choice_v1.features import OPS
from learning.task_pair_data_preflight_v1 import projection


class ProjectionTests(unittest.TestCase):
    def test_operation_and_primary_coverage_boundaries(self):
        self.assertEqual(projection.support(20, 2), 'supported')
        self.assertEqual(projection.support(19, 2), 'unknown')
        self.assertEqual(projection.support(20, 1), 'unknown')
        stats = {'qualifying_per_cluster_sorted': [5, 15], 'qualifying_plant_groups': 20}
        self.assertEqual(projection.primary_support(stats, 'teacher_dev'), 'supported')
        stats['qualifying_per_cluster_sorted'] = [4, 16]
        self.assertEqual(projection.primary_support(stats, 'teacher_dev'), 'unknown')
        self.assertEqual(projection.primary_support(stats, 'train'), 'supported')

    def test_unmatched_support_uses_only_unseen_groups(self):
        count = 40
        view = types.SimpleNamespace(
            ops=np.full(count, OPS.index('PLANT'), dtype=np.uint8),
            offsets=np.arange(count + 1, dtype=np.int64),
            labels=np.zeros(count, dtype=np.int64),
            private_clusters=np.array([10] * 20 + [11] * 20, dtype=np.uint32),
        )
        masks = {'seen_in_train': np.zeros(count, dtype=bool),
                 'qualifying_plant': np.ones(count, dtype=bool)}
        observed = projection.unmatched_coverage([view], [masks], [10, 11])
        self.assertEqual(observed['qualifying_plant'], 'supported')
        self.assertEqual(observed['operations']['PLANT'], 'supported')
        self.assertEqual(observed['operations']['DIG'], 'unknown')
        masks['seen_in_train'][:17] = True
        observed = projection.unmatched_coverage([view], [masks], [10, 11])
        self.assertEqual(observed['qualifying_plant'], 'unknown')
        self.assertEqual(observed['operations']['PLANT'], 'supported')

    def test_public_projection_contains_no_counts_or_group_masks(self):
        def coverage(cluster_counts):
            return {'qualifying_plant_groups': sum(cluster_counts),
                    'qualifying_per_cluster_sorted': cluster_counts,
                    'operations': {op: {'groups': sum(cluster_counts), 'clusters_with_groups': 2} for op in OPS}}

        def source(first):
            count = 20
            view = types.SimpleNamespace(
                ops=np.full(count, OPS.index('PLANT'), dtype=np.uint8),
                offsets=np.arange(count + 1, dtype=np.int64),
                labels=np.zeros(count, dtype=np.int64),
                private_clusters=np.array([first] * 10 + [first + 1] * 10, dtype=np.uint32),
            )
            mask = {'seen_in_train': np.zeros(count, dtype=bool),
                    'label_conflict': np.zeros(count, dtype=bool),
                    'qualifying_plant': np.ones(count, dtype=bool)}
            return view, mask

        teacher, teacher_mask = source(10)
        old, old_mask = source(20)
        new, new_mask = source(30)
        private = types.SimpleNamespace(
            coverage={'train': {'aggregate': coverage([1, 19])},
                      'teacher_dev': {'aggregate': coverage([10, 10])},
                      'conditional_dev': {'aggregate': coverage([10, 10, 10, 10]),
                                          'sources': [coverage([10, 10]), coverage([10, 10])]}},
            masks={'teacher_dev': [teacher_mask], 'conditional_dev': [old_mask, new_mask]},
        )
        roles = {'teacher_dev': [teacher], 'conditional_dev': [old, new]}
        clusters = {'teacher_dev': [10, 11], 'conditional_dev': [20, 21, 30, 31]}
        out = projection.project(private, roles, clusters)
        self.assertTrue(out['all_groups_primary_ready'])
        self.assertTrue(out['unmatched_primary_ready'])
        self.assertEqual(out['overlap']['teacher_dev'], 'below_20')
        self.assertNotIn('10', str(out))
        self.assertNotIn('20', str(out).replace('below_20', ''))
        old_mask['label_conflict'][0] = True
        out = projection.project(private, roles, clusters)
        self.assertEqual(out['label_conflict'], 'detected')
        self.assertFalse(out['all_groups_primary_ready'])
        old_mask['label_conflict'][0] = False
        teacher_mask['seen_in_train'][:] = True
        out = projection.project(private, roles, clusters)
        self.assertTrue(out['all_groups_primary_ready'])
        self.assertFalse(out['unmatched_primary_ready'])

        private.coverage['conditional_dev']['aggregate'] = coverage([5, 5, 5, 5])
        private.coverage['conditional_dev']['sources'] = [coverage([5, 5]), coverage([5, 5])]
        out = projection.project(private, roles, clusters)
        self.assertEqual(out['coverage']['conditional_dev']['qualifying_plant'], 'supported')
        self.assertEqual(out['conditional_sources']['old']['coverage']['qualifying_plant'], 'unknown')
        self.assertFalse(out['all_groups_primary_ready'])


if __name__ == '__main__':
    unittest.main()
