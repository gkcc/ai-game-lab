"""Synthetic tests for the paired task-head worker-ranking objective."""

import importlib.util
from functools import wraps
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_training_v1.loss import conditional_worker_loss, validate_inputs


def cpu_jax(test):
    @wraps(test)
    def run(*args, **kwargs):
        import jax
        with jax.default_device(jax.devices('cpu')[0]):
            return test(*args, **kwargs)
    return run


class WorkerLossTests(unittest.TestCase):
    def fixture(self):
        logits = np.array([[np.log(2.0), 0.0, 100.0, -3.0],
                           [-2.0, 0.0, 0.0, 30.0],
                           [9.0, -2.0, 3.0, 2.0]], dtype=np.float64)
        mask = np.array([[True, True, False, False],
                         [False, True, True, False],
                         [False, False, False, False]])
        labels = np.array([0, 2, 3], dtype=np.int64)
        eligible = np.array([True, True, False])
        return logits, mask, labels, eligible

    def test_known_values_ignore_non_target_and_ineligible_rows(self):
        z, mask, labels, eligible = self.fixture()
        validate_inputs(z, mask, labels, eligible, 2)
        got = conditional_worker_loss(z, mask, labels, eligible, 2)
        self.assertAlmostEqual(got, (np.log(1.5) + np.log(2.0)) / 2)
        z[0, 2] = -1e10
        z[1, 0] = 1e10
        z[2] = np.array([-1e10, 2e10, 3e10, -4e10])
        self.assertAlmostEqual(conditional_worker_loss(z, mask, labels, eligible, 2), got)

    def test_all_ineligible_and_zero_denominator_are_exact_zero(self):
        z, mask, labels, eligible = self.fixture()
        eligible[:] = False
        validate_inputs(z, mask, labels, eligible, 0)
        self.assertEqual(conditional_worker_loss(z, mask, labels, eligible, 0), 0.0)
        self.assertEqual(conditional_worker_loss(z, mask, labels, eligible, 2), 0.0)

    def test_whole_batch_denominator_across_microbatches(self):
        z, mask, labels, eligible = self.fixture()
        total = int(eligible.sum())
        whole = conditional_worker_loss(z, mask, labels, eligible, total)
        pieces = 0.0
        for index in (slice(0, 1), slice(1, 2), slice(2, 3)):
            args = (z[index], mask[index], labels[index], eligible[index], total)
            validate_inputs(*args)
            pieces += conditional_worker_loss(*args)
        self.assertAlmostEqual(whole, pieces)

    def test_reordering_candidates_with_labels_preserves_loss(self):
        z, mask, labels, eligible = self.fixture()
        permutation = np.array([3, 1, 0, 2])
        inverse = np.argsort(permutation)
        z2, mask2, labels2 = z[:, permutation], mask[:, permutation], inverse[labels]
        validate_inputs(z2, mask2, labels2, eligible, 2)
        self.assertAlmostEqual(conditional_worker_loss(z, mask, labels, eligible, 2),
                               conditional_worker_loss(z2, mask2, labels2, eligible, 2))

    def test_validator_rejects_bad_inputs(self):
        z, mask, labels, eligible = self.fixture()
        cases = [
            (z[:, :0], mask[:, :0], labels, eligible, 2),
            (z.astype(np.int64), mask, labels, eligible, 2),
            (np.full_like(z, np.nan), mask, labels, eligible, 2),
            (z, mask.astype(np.int8), labels, eligible, 2),
            (z, mask, labels.astype(float), eligible, 2),
            (z, mask, labels, eligible.astype(np.int8), 2),
            (z, mask, np.array([4, 2, 0]), eligible, 2),
            (z, mask, np.array([0, 0, 0]), eligible, 2),
            (z, np.array([[True, False, False, False], mask[1], mask[2]]), labels, eligible, 2),
            (z, mask, labels, eligible, 1),
            (z, mask, labels, eligible, True),
            (z, mask, labels, eligible, -1),
        ]
        for index, args in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    validate_inputs(*args)

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'JAX is not installed')
    @cpu_jax
    def test_cpu_jax_gradients_match_microbatch_accumulation(self):
        import jax
        import jax.numpy as jnp

        cpu = jax.devices('cpu')[0]
        z, mask, labels, eligible = self.fixture()
        mask = jax.device_put(mask, cpu)
        labels = jax.device_put(labels, cpu)
        eligible = jax.device_put(eligible, cpu)
        # Two parameter layers ensure all parameter leaves have finite gradients.
        features = jax.device_put(np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 20, cpu)
        params = jax.device_put({'w1': np.array([[.2, -.1], [.3, .4], [-.2, .1]], np.float32),
                                 'b1': np.array([.1, -.1], np.float32),
                                 'w2': np.array([.3, -.25], np.float32),
                                 'b2': np.array(.05, np.float32)}, cpu)

        def piece(p, lo, hi):
            hidden = jnp.tanh(features[lo:hi] @ p['w1'] + p['b1'])
            logits = hidden @ p['w2'] + p['b2']
            return conditional_worker_loss(logits, mask[lo:hi], labels[lo:hi],
                                           eligible[lo:hi], 2, jnp)

        whole, whole_grad = jax.value_and_grad(lambda p: piece(p, 0, 3))(params)
        parts, parts_grad = jax.value_and_grad(lambda p: piece(p, 0, 1) + piece(p, 1, 2)
                                                 + piece(p, 2, 3))(params)
        self.assertTrue(np.isfinite(np.asarray(whole)))
        self.assertAlmostEqual(float(whole), float(parts), places=6)
        for name in params:
            self.assertTrue(np.isfinite(np.asarray(whole_grad[name])).all(), name)
            np.testing.assert_allclose(np.asarray(whole_grad[name]), np.asarray(parts_grad[name]),
                                       rtol=2e-5, atol=2e-6)

        @jax.jit
        def compiled(p, row_eligible, denominator):
            def objective(parameters):
                predicted = jnp.tanh(features @ parameters['w1'] + parameters['b1']) @ parameters['w2'] + parameters['b2']
                return conditional_worker_loss(predicted, mask, labels, row_eligible, denominator, jnp)
            return jax.value_and_grad(objective)(p)

        compiled_value, compiled_grad = compiled(params, eligible, jnp.asarray(2, dtype=jnp.int32))
        self.assertAlmostEqual(float(whole), float(compiled_value), places=6)
        for name in params:
            np.testing.assert_allclose(np.asarray(whole_grad[name]), np.asarray(compiled_grad[name]),
                                       rtol=2e-5, atol=2e-6)

        empty_mask = jnp.zeros_like(mask)
        empty_eligible = jnp.zeros_like(eligible)
        empty_value, empty_grad = jax.value_and_grad(
            lambda p: conditional_worker_loss(
                jnp.tanh(features @ p['w1'] + p['b1']) @ p['w2'] + p['b2'],
                empty_mask, labels, empty_eligible, 0, jnp))(params)
        self.assertEqual(float(empty_value), 0.0)
        for name in params:
            self.assertTrue(np.isfinite(np.asarray(empty_grad[name])).all(), name)
            np.testing.assert_array_equal(np.asarray(empty_grad[name]), np.zeros_like(params[name]))
        compiled_empty, compiled_empty_grad = compiled(params, empty_eligible, jnp.asarray(0, dtype=jnp.int32))
        self.assertEqual(float(compiled_empty), 0.0)
        for name in params:
            self.assertTrue(np.isfinite(np.asarray(compiled_empty_grad[name])).all(), name)
            np.testing.assert_array_equal(np.asarray(compiled_empty_grad[name]), np.zeros_like(params[name]))


if __name__ == '__main__':
    unittest.main()
