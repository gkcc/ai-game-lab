"""Synthetic loss and CPU autodiff checks; no dataset or checkpoint access."""

import importlib.util
from functools import wraps
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_bc_v1.model import numpy_logits
from learning.task_bc_v1.train import weighted_group_loss
from learning.task_pair_training_v1.objective import (
    ce_only_objective, combined_objective, forward, objective_components, validate_inputs, validate_parameters,
)


def cpu_jax(test):
    @wraps(test)
    def run(*args, **kwargs):
        import jax
        with jax.default_device(jax.devices('cpu')[0]):
            return test(*args, **kwargs)
    return run


def fixture():
    features = np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        [[.2, .6, .1], [.3, .1, .8], [.5, .2, .4], [0, 0, 0]],
        [[1, 1, 0], [.1, .2, .3], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ], dtype=np.float32)
    params = {
        'w0': np.array([[.4, -.3, .1, .2], [.1, .6, -.2, .3], [-.2, .3, .5, .1]], np.float32),
        'b0': np.array([.2, .1, .2, .4], np.float32),
        'w1': np.array([[.3, .4], [-.2, .3], [.5, -.2], [.1, .3]], np.float32),
        'b1': np.array([.2, .4], np.float32),
        'w2': np.array([[.6], [-.5]], np.float32),
        'b2': np.array([.1], np.float32),
    }
    mask = np.array([[True, True, True, False], [True, True, True, False],
                     [True, True, False, False], [True, False, False, False]])
    labels = np.array([0, 2, 1, 0], dtype=np.int32)
    weights = np.array([1.5, .5, 2.0, 0.0], dtype=np.float32)
    targets = np.array([[True, True, False, False], [False, False, False, False],
                        [True, True, False, False], [False, False, False, False]])
    eligible = np.array([True, False, True, False])
    return params, features, mask, labels, weights, targets, eligible


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        (self.params, self.features, self.mask, self.labels,
         self.weights, self.targets, self.eligible) = fixture()
        self.ce_denominator = float(np.sum(self.weights))
        self.eligible_count = int(np.count_nonzero(self.eligible))
        validate_parameters(self.params, self.features.shape[-1])
        validate_inputs(self.features, self.mask, self.labels, self.weights, self.ce_denominator,
                        self.targets, self.eligible, self.eligible_count)

    def test_forward_matches_existing_numpy_scorer(self):
        np.testing.assert_allclose(forward(self.params, self.features),
                                   numpy_logits(self.params, self.features),
                                   rtol=1e-6, atol=1e-6)

    def test_zero_coefficient_equals_existing_ce_and_positive_adds_auxiliary(self):
        logits = forward(self.params, self.features)
        reference = weighted_group_loss(logits, self.mask, self.labels, self.weights, self.ce_denominator)
        ce, auxiliary = objective_components(self.params, self.features, self.mask, self.labels,
                                             self.weights, self.ce_denominator, self.targets,
                                             self.eligible, self.eligible_count)
        zero = combined_objective(self.params, self.features, self.mask, self.labels, self.weights,
                                  self.ce_denominator, self.targets, self.eligible,
                                  self.eligible_count, worker_coefficient=0.0)
        control = ce_only_objective(self.params, self.features, self.mask, self.labels,
                                    self.weights, self.ce_denominator)
        with_auxiliary = combined_objective(self.params, self.features, self.mask, self.labels,
                                            self.weights, self.ce_denominator, self.targets,
                                            self.eligible, self.eligible_count)
        self.assertAlmostEqual(float(reference), float(ce), places=7)
        self.assertAlmostEqual(float(reference), float(zero), places=7)
        self.assertAlmostEqual(float(reference), float(control), places=7)
        self.assertGreater(float(auxiliary), 0)
        self.assertAlmostEqual(float(with_auxiliary), float(reference + .25 * auxiliary), places=7)

    def test_control_does_not_evaluate_the_auxiliary_path(self):
        with patch('learning.task_pair_training_v1.objective.conditional_worker_loss',
                   side_effect=AssertionError('control reached auxiliary loss')):
            value = ce_only_objective(self.params, self.features, self.mask, self.labels,
                                      self.weights, self.ce_denominator)
        self.assertTrue(np.isfinite(value))

    def test_empty_eligibility_and_padded_microbatch_are_safe(self):
        ineligible = slice(1, 2)
        no_worker = objective_components(self.params, self.features[ineligible], self.mask[ineligible],
                                        self.labels[ineligible], self.weights[ineligible], self.ce_denominator,
                                        self.targets[ineligible], self.eligible[ineligible], self.eligible_count)
        self.assertEqual(float(no_worker[1]), 0.0)
        padding = slice(3, 4)
        result = objective_components(self.params, self.features[padding], self.mask[padding],
                                      self.labels[padding], self.weights[padding], self.ce_denominator,
                                      self.targets[padding], self.eligible[padding], self.eligible_count)
        self.assertEqual(tuple(map(float, result)), (0.0, 0.0))
        empty_eligible = np.zeros_like(self.eligible)
        empty_targets = np.zeros_like(self.targets)
        validate_inputs(self.features, self.mask, self.labels, self.weights, self.ce_denominator,
                        empty_targets, empty_eligible, 0)
        self.assertEqual(float(objective_components(self.params, self.features, self.mask, self.labels,
                                                   self.weights, self.ce_denominator, empty_targets,
                                                   empty_eligible, 0)[1]), 0.0)

    def test_rejects_invalid_model_and_microbatch_contracts(self):
        broken = dict(self.params)
        broken['b1'] = np.zeros(3, np.float32)
        with self.assertRaises(ValueError):
            validate_parameters(broken, 3)
        invalid_denominators = [0, -1, np.nan, True, 1.0]
        for denominator in invalid_denominators:
            with self.subTest(ce_denominator=denominator), self.assertRaises(ValueError):
                validate_inputs(self.features, self.mask, self.labels, self.weights, denominator,
                                self.targets, self.eligible, self.eligible_count)
        bad_labels = self.labels.copy()
        bad_labels[0] = 3  # Candidate exists in padding width, but is masked out.
        with self.assertRaises(ValueError):
            validate_inputs(self.features, self.mask, bad_labels, self.weights, self.ce_denominator,
                            self.targets, self.eligible, self.eligible_count)
        bad_targets = self.targets.copy()
        bad_targets[0, 3] = True
        with self.assertRaises(ValueError):
            validate_inputs(self.features, self.mask, self.labels, self.weights, self.ce_denominator,
                            bad_targets, self.eligible, self.eligible_count)
        bad_mask = self.mask.copy()
        bad_mask[3, 1] = True
        with self.assertRaises(ValueError):
            validate_inputs(self.features, bad_mask, self.labels, self.weights, self.ce_denominator,
                            self.targets, self.eligible, self.eligible_count)
        with self.assertRaises(ValueError):
            validate_inputs(self.features, self.mask, self.labels, self.weights, self.ce_denominator,
                            self.targets, self.eligible, 1)
        with self.assertRaises(ValueError):
            validate_inputs(self.features, self.mask, self.labels, self.weights, self.ce_denominator,
                            self.targets, self.eligible, self.eligible_count, worker_coefficient=-.1)
        with self.assertRaises(ValueError):
            validate_inputs(self.features, self.mask, self.labels, self.weights, self.ce_denominator,
                            self.targets.tolist(), self.eligible, self.eligible_count)

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'JAX is not installed')
    @cpu_jax
    def test_cpu_jax_gradient_is_nonzero_and_whole_equals_microbatches(self):
        import jax
        import jax.numpy as jnp

        cpu = jax.devices('cpu')[0]
        params = jax.device_put(self.params, cpu)
        arrays = [jax.device_put(item, cpu) for item in
                  (self.features, self.mask, self.labels, self.weights, self.targets, self.eligible)]
        features, mask, labels, weights, targets, eligible = arrays

        def components(p, lo, hi, width=4):
            return objective_components(p, features[lo:hi, :width], mask[lo:hi, :width],
                                        labels[lo:hi], weights[lo:hi],
                                        self.ce_denominator, targets[lo:hi, :width], eligible[lo:hi],
                                        self.eligible_count, jnp)

        def combined(p, lo, hi, width=4):
            return combined_objective(p, features[lo:hi, :width], mask[lo:hi, :width],
                                      labels[lo:hi], weights[lo:hi],
                                      self.ce_denominator, targets[lo:hi, :width], eligible[lo:hi],
                                      self.eligible_count, xp=jnp)

        auxiliary, aux_gradient = jax.value_and_grad(lambda p: components(p, 0, 4)[1])(params)
        self.assertGreater(float(auxiliary), 0)
        squared_norm = sum(float(jnp.sum(jnp.square(leaf))) for leaf in jax.tree_util.tree_leaves(aux_gradient))
        self.assertGreater(squared_norm, 1e-16)
        for leaf in jax.tree_util.tree_leaves(aux_gradient):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())

        whole, whole_gradient = jax.jit(jax.value_and_grad(lambda p: combined(p, 0, 4)))(params)
        # Distinct padded widths and separate JIT calls match the real accumulation
        # pattern: differentiate each microbatch, then sum every parameter leaf.
        pieces = 0.0
        pieces_gradient = {name: jnp.zeros_like(value) for name, value in params.items()}
        for lo, width in enumerate((4, 4, 2, 1)):
            value, gradient = jax.jit(jax.value_and_grad(
                lambda p: combined(p, lo, lo + 1, width)))(params)
            pieces = pieces + value
            pieces_gradient = jax.tree_util.tree_map(lambda total, part: total + part,
                                                      pieces_gradient, gradient)
        self.assertAlmostEqual(float(whole), float(pieces), places=6)
        for name in params:
            self.assertTrue(np.isfinite(np.asarray(whole_gradient[name])).all(), name)
            np.testing.assert_allclose(np.asarray(whole_gradient[name]), np.asarray(pieces_gradient[name]),
                                       rtol=2e-5, atol=2e-6, err_msg=name)

        _, control_gradient = jax.value_and_grad(lambda p: ce_only_objective(
            p, features, mask, labels, weights, self.ce_denominator, xp=jnp))(params)
        _, zero_gradient = jax.value_and_grad(lambda p: combined_objective(
            p, features, mask, labels, weights, self.ce_denominator, targets, eligible,
            self.eligible_count, worker_coefficient=0.0, xp=jnp))(params)
        for name in params:
            np.testing.assert_allclose(np.asarray(control_gradient[name]), np.asarray(zero_gradient[name]),
                                       rtol=2e-5, atol=2e-6, err_msg=name)

        empty_aux, empty_aux_gradient = jax.value_and_grad(lambda p: components(p, 1, 2)[1])(params)
        padded_loss, padded_gradient = jax.value_and_grad(lambda p: combined(p, 3, 4))(params)
        self.assertEqual(float(empty_aux), 0.0)
        self.assertEqual(float(padded_loss), 0.0)
        for name in params:
            for gradient in (empty_aux_gradient, padded_gradient):
                self.assertTrue(np.isfinite(np.asarray(gradient[name])).all(), name)
                np.testing.assert_array_equal(np.asarray(gradient[name]), np.zeros_like(self.params[name]))

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'JAX is not installed')
    @cpu_jax
    def test_relu_zero_gradient_matches_original_jax_relu(self):
        import jax
        import jax.numpy as jnp

        params = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in {
            'w0': [[1.0]], 'b0': [0.0], 'w1': [[1.0]], 'b1': [1.0],
            'w2': [[1.0]], 'b2': [0.0],
        }.items()}
        ours = jax.grad(lambda x: forward(params, x, jnp).sum())(jnp.zeros((1, 1), jnp.float32))
        reference = jax.grad(lambda x: jax.nn.relu(jax.nn.relu(x) + 1.0).sum())(
            jnp.zeros((1, 1), jnp.float32))
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(reference))


if __name__ == '__main__':
    unittest.main()
