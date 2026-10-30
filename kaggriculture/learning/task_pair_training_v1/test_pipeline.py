"""Synthetic source-to-update checks; no real data, checkpoints, GPU or games."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_data_preflight_v1.test_core import fixture
from learning.task_pair_training_v1 import fit_core, packing
from learning.task_pair_training_v1.targets import qualifying_groups


@unittest.skipUnless(importlib.util.find_spec('jax') is not None
                     and importlib.util.find_spec('optax') is not None, 'JAX and Optax are required')
class PipelineTests(unittest.TestCase):
    def setUp(self):
        arrays, _, _ = fixture()
        self.data = arrays[0]
        self.eligible = qualifying_groups(self.data, width=4)
        self.mean = np.zeros(373, np.float64)
        self.scale = np.ones(373, np.float64)
        self.weights = np.ones(5, np.float32)
        rng = np.random.default_rng(21)  # Synthetic initialization only.
        self.params = {
            'w0': rng.normal(0, .1, (373, 4)).astype(np.float32),
            'b0': np.ones(4, np.float32),
            'w1': rng.normal(0, .1, (4, 2)).astype(np.float32),
            'b1': np.ones(2, np.float32),
            'w2': rng.normal(0, .1, (2, 1)).astype(np.float32),
            'b2': np.zeros(1, np.float32),
        }
        self.batches = np.asarray([[3, 0, 1, 2], [4, 5, 6, 7]], np.int64)

    def pack(self, ids):
        return packing.pack_training_batch(self.data, ids, self.mean, self.scale,
                                           self.weights, self.eligible)

    def test_source_packer_runs_once_per_planned_batch(self):
        import jax
        cpu = jax.devices('cpu')[0]
        calls = []

        def observe(ids):
            calls.append(ids.copy())
            return self.pack(ids)

        with jax.default_device(cpu):
            result = fit_core.fit_pair(self.params, self.batches, observe, device=cpu, check=lambda: None)
        self.assertEqual(len(calls), len(self.batches))
        for call, expected in zip(calls, self.batches, strict=True):
            np.testing.assert_array_equal(call, expected)
        self.assertEqual(result['completed_updates'], 2)
        self.assertEqual(result['device_platform'], 'cpu')
        self.assertTrue(result['control_changed_from_initial'])
        self.assertTrue(result['experiment_changed_from_initial'])
        self.assertTrue(result['arms_differ'])
        self.assertGreater(result['aux_nonzero_updates'], 0)

    def test_missing_packed_group_fails_before_the_first_update(self):
        import jax
        import optax
        cpu = jax.devices('cpu')[0]
        original = packing.train.microbatches
        original_chain = optax.chain
        updates = []

        def omit(data, ids):
            for group_ids, capacity, width in original(data, ids):
                yield group_ids[:-1], capacity, width

        def instrumented_chain(*transforms):
            chain = original_chain(*transforms)

            def update(*args, **kwargs):
                updates.append(True)
                return chain.update(*args, **kwargs)

            return optax.GradientTransformation(chain.init, update)

        with (jax.default_device(cpu), patch.object(packing.train, 'microbatches', side_effect=omit),
              patch.object(optax, 'chain', side_effect=instrumented_chain)):
            with self.assertRaises(ValueError):
                fit_core.fit_pair(self.params, self.batches, self.pack, device=cpu, check=lambda: None)
        self.assertEqual(updates, [])


if __name__ == '__main__':
    unittest.main()
