"""Synthetic numerical and CPU autodiff checks; no model or game data."""
import os
import unittest

import numpy as np

from . import loss


def batch():
    return dict(logits=np.array([[1.2, 0.4, -0.7, 2.0, 1e20]], dtype=np.float64),
                frozen_logits=np.array([[0.8, 0.0, -1.0, 1.3, -1e20]], dtype=np.float64),
                candidate_mask=np.array([[1, 1, 1, 1, 0]], dtype=bool),
                positive_mask=np.array([[1, 1, 0, 0, 0]], dtype=bool),
                verified_mask=np.array([[1, 1, 1, 0, 0]], dtype=bool),
                teacher_index=np.array([0], dtype=np.int32),
                student_index=np.array([1], dtype=np.int32))


class LossTests(unittest.TestCase):
    def test_exact_objectives_and_kl_direction(self):
        values = batch()
        control, control_parts = loss.objective(**values, arm='control')
        grouped, set_parts = loss.objective(**values, arm='set')
        z, old = values['logits'][0, :4], values['frozen_logits'][0, :4]
        teacher = np.logaddexp.reduce(z[:3]) - z[0]
        group = np.logaddexp.reduce(z[:3]) - np.logaddexp.reduce(z[:2])
        old_logp = old - np.logaddexp.reduce(old)
        new_logp = z - np.logaddexp.reduce(z)
        expected_kl = np.sum(np.exp(old_logp) * (old_logp - new_logp))
        self.assertAlmostEqual(control_parts[0], teacher)
        self.assertAlmostEqual(set_parts[0], group)
        self.assertAlmostEqual(control_parts[1], expected_kl)
        self.assertAlmostEqual(set_parts[1], expected_kl)
        self.assertAlmostEqual(control, teacher + .02 * expected_kl)
        self.assertAlmostEqual(grouped, group + .02 * expected_kl)
        self.assertEqual(loss.KL_WEIGHT, .02)

    def test_single_positive_degenerates_to_teacher_objective(self):
        values = batch()
        values['student_index'][0] = values['teacher_index'][0]
        values['positive_mask'][0, 1] = False
        values['verified_mask'][0, 1] = False
        teacher, grouped, _kl = loss.per_group(**values)
        np.testing.assert_allclose(teacher, grouped, rtol=0, atol=1e-14)
        for arm in ('control', 'set'):
            loss.validate_finite_outputs(*loss.objective(**values, arm=arm))

    def test_unknown_is_excluded_from_supervision_but_participates_in_kl(self):
        values = batch()
        before = loss.per_group(**values)
        values['logits'][0, 3] += 4.0  # Candidate 3 is real but absent from V.
        after = loss.per_group(**values)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertNotEqual(float(before[2][0]), float(after[2][0]))

    def test_padding_does_not_affect_loss(self):
        values = batch()
        before = [loss.objective(**values, arm=arm)[0] for arm in ('control', 'set')]
        values['logits'][0, 4] = -1e300
        values['frozen_logits'][0, 4] = 1e300
        after = [loss.objective(**values, arm=arm)[0] for arm in ('control', 'set')]
        np.testing.assert_array_equal(before, after)

    def test_batch_mean_weights_each_group_once(self):
        first, second = batch(), batch()
        second['logits'][0, 0] -= 2
        second['frozen_logits'][0, 1] += .5
        combined = {key: np.concatenate((first[key], second[key]), axis=0) for key in first}
        for arm in ('control', 'set'):
            expected = (loss.objective(**first, arm=arm)[0] + loss.objective(**second, arm=arm)[0]) / 2
            self.assertAlmostEqual(loss.objective(**combined, arm=arm)[0], expected)

    def test_large_finite_scores_are_stable_without_positive_underflow(self):
        values = batch()
        values['logits'][0] = [10000., -10000., 9999., 9000., 1e300]
        values['teacher_index'][0], values['student_index'][0] = 1, 2
        values['positive_mask'][0] = [0, 1, 1, 0, 0]
        control, grouped, kl = loss.per_group(**values)
        loss.validate_finite_outputs(control, grouped, kl)
        self.assertAlmostEqual(control[0], np.logaddexp.reduce([10000., -10000., 9999.]) + 10000.)
        self.assertAlmostEqual(grouped[0], np.logaddexp.reduce([10000., -10000., 9999.]) - 9999.)

    def test_bad_masks_indices_shapes_and_nonfinite_values_fail(self):
        variants = []
        def mutated(key, operation):
            v = batch(); operation(v[key]); variants.append(v)
        mutated('positive_mask', lambda x: x.__setitem__((0, 1), False))
        mutated('positive_mask', lambda x: x.__setitem__((0, 3), True))
        mutated('verified_mask', lambda x: x.__setitem__((0, 1), False))
        mutated('verified_mask', lambda x: x.__setitem__((0, 2), False))
        mutated('verified_mask', lambda x: x.__setitem__((0, 3), True))
        mutated('verified_mask', lambda x: x.__setitem__((0, 4), True))
        mutated('teacher_index', lambda x: x.__setitem__(0, -1))
        mutated('student_index', lambda x: x.__setitem__(0, 5))
        mutated('logits', lambda x: x.__setitem__((0, 4), np.nan))
        mutated('frozen_logits', lambda x: x.__setitem__((0, 2), np.inf))
        bad = batch(); bad['student_index'] = np.array([True]); variants.append(bad)
        bad = batch(); bad['candidate_mask'] = np.zeros((1, 4), bool); variants.append(bad)
        bad = batch(); bad['frozen_logits'] = bad['frozen_logits'].astype(np.float32); variants.append(bad)
        for index, value in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                loss.validate_batch(**value)
        with self.assertRaises(ValueError):
            loss.objective(**batch(), arm='unknown')
        for invalid in (np.array([np.nan]), np.array([np.inf]), np.array(['private'])):
            with self.assertRaises(ValueError):
                loss.validate_finite_outputs(invalid)
        extreme = batch()
        extreme['logits'][0, 0], extreme['logits'][0, 1] = 1e308, -1e308
        with np.errstate(over='ignore', invalid='ignore'), self.assertRaises(ValueError):
            loss.objective(**extreme, arm='control')


class JaxCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Per-process test configuration only: never initialize a GPU backend.
        os.environ['JAX_PLATFORMS'] = 'cpu'
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            raise unittest.SkipTest('JAX unavailable')
        cls.jax, cls.jnp = jax, jnp
        if jax.default_backend() != 'cpu':
            raise unittest.SkipTest('CPU-only JAX backend unavailable')

    def test_jit_gradients_respect_unknown_padding_and_frozen_baseline(self):
        values = batch()
        loss.validate_batch(**values)
        jax, jnp = self.jax, self.jnp
        device = {key: jnp.asarray(value) for key, value in values.items()}
        for arm in ('control', 'set'):
            def target(current, reference):
                v = dict(device, logits=current, frozen_logits=reference)
                return loss.objective(**v, arm=arm, xp=jnp)[0]
            value, gradient = jax.jit(jax.value_and_grad(target, argnums=0))(
                device['logits'], device['frozen_logits'])
            frozen_gradient = jax.grad(target, argnums=1)(device['logits'], device['frozen_logits'])
            loss.validate_finite_outputs(np.asarray(value), np.asarray(gradient), np.asarray(frozen_gradient))
            np.testing.assert_array_equal(np.asarray(gradient)[:, 4], np.array([0.]))
            np.testing.assert_array_equal(np.asarray(frozen_gradient), np.zeros((1, 5)))
            self.assertNotEqual(float(np.asarray(gradient)[0, 3]), 0.)

            # Independent analytic gradient: supervised V distribution minus
            # teacher one-hot or P distribution, plus .02 * (new - old) on C.
            z, old = values['logits'][0, :4], values['frozen_logits'][0, :4]
            def softmax(x):
                q = np.exp(x - np.max(x))
                return q / q.sum()
            expected = np.zeros(5)
            expected[:3] = softmax(z[:3])
            if arm == 'control':
                expected[0] -= 1
            else:
                expected[:2] -= softmax(z[:2])
            expected[:4] += .02 * (softmax(z) - softmax(old))
            np.testing.assert_allclose(np.asarray(gradient)[0], expected, rtol=1e-5, atol=1e-6)

            def primary(current):
                v = dict(device, logits=current)
                t, s, _kl = loss.per_group(**v, xp=jnp)
                return (t if arm == 'control' else s).sum()
            primary_gradient = np.asarray(jax.grad(primary)(device['logits']))
            self.assertEqual(primary_gradient[0, 3], 0.)
            self.assertEqual(primary_gradient[0, 4], 0.)

    def test_single_positive_has_equal_losses_and_gradients(self):
        values = batch()
        values['student_index'][0] = values['teacher_index'][0]
        values['positive_mask'][0, 1] = False
        values['verified_mask'][0, 1] = False
        loss.validate_batch(**values)
        jax, jnp = self.jax, self.jnp
        device = {key: jnp.asarray(value) for key, value in values.items()}
        def target(current, arm):
            return loss.objective(**dict(device, logits=current), arm=arm, xp=jnp)[0]
        first = jax.value_and_grad(lambda x: target(x, 'control'))(device['logits'])
        second = jax.value_and_grad(lambda x: target(x, 'set'))(device['logits'])
        np.testing.assert_allclose(np.asarray(first[0]), np.asarray(second[0]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(first[1]), np.asarray(second[1]), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
