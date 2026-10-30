"""Synthetic read-only adapter checks; no actual dataset, GPU or checkpoint."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_data_adapter_v1 import adapter
from learning.task_pair_data_preflight_v1 import core
from learning.task_pair_data_preflight_v1.test_core import SCHEMA, make_arrays, record
from learning.task_pair_training_v1.batching import make_batches
from learning.task_pair_training_v1.packing import pack_training_batch
from learning.task_pair_training_v1.targets import qualifying_groups


def freeze(arrays):
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def small_train():
    arrays = freeze(make_arrays([record(i, i % 2, 1 + i % 8) for i in range(24)]))
    return arrays, core.RaggedView.from_cache(arrays, SCHEMA)


def small_adapt(view, **overrides):
    options = {'role': 'train', 'expected_clusters': tuple(range(1, 9)),
               'expected_schema_sha256': SCHEMA['schema_sha256'],
               'expected_groups': 24, 'expected_candidates': 72}
    options.update(overrides)
    return adapter._adapt(view, **options)


def virtual_r8():
    """Public-size logical shapes backed by tiny read-only broadcast buffers.

    The adapter only reads scalar metadata and compact integer supervision;
    it must never materialize either large virtual feature array. This fixture
    is not semantically valid game data or evidence of source authentication.
    """
    n, m = adapter.R8_GROUPS, adapter.R8_CANDIDATES
    base, remainder = divmod(m, n)
    index = np.arange(n + 1, dtype=np.int64)
    arrays = {
        'shared': np.broadcast_to(np.zeros((1, 275), np.float32), (n, 275)),
        'candidates': np.broadcast_to(np.zeros((1, 98), np.float32), (m, 98)),
        'offsets': index * base + np.minimum(index, remainder),
        'labels': np.zeros(n, np.int64),
        'selected_ops': np.zeros(n, np.int8),
        'candidate_ops': np.broadcast_to(np.zeros(1, np.int8), (m,)),
        'candidate_identity': np.broadcast_to(np.zeros((1, 3), np.int16), (m, 3)),
        'seeds': np.arange(n, dtype=np.int64) % 8 + 1,
    }
    freeze(arrays)
    return arrays, core.RaggedView.from_cache(arrays, SCHEMA)


class AdapterTests(unittest.TestCase):
    def test_public_r8_contract_preserves_exact_objects_and_order(self):
        arrays, view = virtual_r8()
        mapped = adapter.adapt_r8(view, role='train', expected_clusters=tuple(range(1, 9)),
                                  expected_schema_sha256=SCHEMA['schema_sha256'])
        self.assertEqual(tuple(mapped), tuple(arrays))
        for name in arrays:
            self.assertIs(mapped[name], arrays[name])
            self.assertFalse(mapped[name].flags.writeable)
        with self.assertRaises(TypeError):
            mapped['shared'] = None
        with self.assertRaises(ValueError):
            mapped['labels'][0] = 5

    def test_real_packer_and_batcher_consume_same_small_read_only_view(self):
        arrays, view = small_train()
        data = small_adapt(view)
        for name in arrays:
            self.assertIs(data[name], arrays[name])
        eligible = qualifying_groups(data, width=4, chunk_size=16)
        plan, anchors = make_batches(eligible, data['seeds'], rng_seed=23, updates=2, batch_size=8)
        self.assertNotEqual(data['seeds'][anchors[0]], data['seeds'][anchors[1]])
        mean = np.zeros(373, np.float64)
        scale = np.ones(373, np.float64)
        weights = np.ones(len(adapter.OPS), np.float32)
        for batch, anchor in zip(plan, anchors):
            parts = pack_training_batch(data, batch, mean, scale, weights, eligible)
            ids = np.concatenate([part['ids'] for part in parts])
            np.testing.assert_array_equal(np.sort(ids), np.sort(batch))
            self.assertEqual(batch[0], anchor)
            self.assertTrue(any(np.any(part['eligible']) for part in parts))
        self.assertTrue(all(not value.flags.writeable for value in arrays.values()))

    def test_public_entry_rejects_unrelated_sizes_and_role(self):
        _, view = small_train()
        for kwargs in ({'role': 'train'}, {'role': 'teacher_dev'}, {'role': 'conditional_dev'}):
            with self.subTest(role=kwargs['role']), self.assertRaises(ValueError):
                adapter.adapt_r8(view, expected_clusters=tuple(range(1, 9)),
                                 expected_schema_sha256=SCHEMA['schema_sha256'], **kwargs)
        _, proper_shape = virtual_r8()
        for bad in ('teacher_dev', 'conditional_dev', 'Train', None):
            with self.subTest(role=bad), self.assertRaises(ValueError):
                adapter.adapt_r8(proper_shape, role=bad, expected_clusters=tuple(range(1, 9)),
                                 expected_schema_sha256=SCHEMA['schema_sha256'])

    def test_missing_extra_or_noncache_view_is_rejected(self):
        arrays, view = small_train()
        with self.assertRaises(ValueError):
            small_adapt(replace(view, selected_ops=None))
        with self.assertRaises(ValueError):
            small_adapt(replace(view, candidate_identity=None))
        with self.assertRaises(ValueError):
            small_adapt(object())
        extra = replace(view)
        object.__setattr__(extra, 'unregistered_source', arrays['shared'])
        with self.assertRaises(ValueError):
            small_adapt(extra)

    def test_role_schema_and_independent_cluster_contracts_are_checked(self):
        _, view = small_train()
        bad_options = ({'role': 'teacher_dev'}, {'expected_schema_sha256': 'f' * 64},
                       {'expected_schema_sha256': 'invalid'},
                       {'expected_clusters': tuple(range(2, 10))},
                       {'expected_clusters': (1, 1, 3, 4, 5, 6, 7, 8)},
                       {'expected_clusters': (True, 2, 3, 4, 5, 6, 7, 8)},
                       {'expected_groups': 25}, {'expected_candidates': 71})
        for options in bad_options:
            with self.subTest(keys=tuple(options)), self.assertRaises(ValueError):
                small_adapt(view, **options)

    def test_writable_wrong_dtype_and_reordered_group_metadata_are_rejected(self):
        arrays, view = small_train()
        bad_arrays = []
        writable = arrays['labels'].copy()
        bad_arrays.append(replace(view, labels=writable))
        wrong_dtype = arrays['seeds'].astype(np.uint32)
        wrong_dtype.setflags(write=False)
        bad_arrays.append(replace(view, private_clusters=wrong_dtype))
        wrong_offsets = arrays['offsets'].copy()
        wrong_offsets[1] = wrong_offsets[0]
        wrong_offsets.setflags(write=False)
        bad_arrays.append(replace(view, offsets=wrong_offsets))
        wrong_selected = arrays['selected_ops'].copy()
        wrong_selected[0] = (wrong_selected[0] + 1) % len(adapter.OPS)
        wrong_selected.setflags(write=False)
        bad_arrays.append(replace(view, selected_ops=wrong_selected))
        bad_ops = arrays['candidate_ops'].copy()
        bad_ops[-1] = -1
        bad_ops.setflags(write=False)
        bad_arrays.append(replace(view, ops=bad_ops))
        for candidate in bad_arrays:
            with self.subTest(case=len(bad_arrays)), self.assertRaises(ValueError):
                small_adapt(candidate)


if __name__ == '__main__':
    unittest.main()
