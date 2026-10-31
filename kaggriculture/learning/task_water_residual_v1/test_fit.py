"""Small synthetic WATER residual tests; GPU and real data are never used."""
import os
import unittest

import numpy as np

from . import fit as f


def group(role, *, width=2, label=1):
    x = np.zeros((width, f.DIM), dtype=np.float32)
    x[:, f.OP_START + f.OPS.index('WATER')] = 1
    x[:, f.SHARED_DIM] = np.arange(width, dtype=np.float32)
    return dict(role=role, features=x, baseline_logits=np.linspace(-0.3, 0.2, width, dtype=np.float32),
                teacher_index=label)


def pools(*, widths=(2,)):
    return [group(role, width=widths[i % len(widths)]) for role in (0, 1) for i in range(8)]


def prepare(source):
    return f.prepare_groups(source, mean=np.zeros(f.DIM, np.float64),
                            scale=np.ones(f.DIM, np.float64), guard=lambda: None)


class WaterFitTests(unittest.TestCase):
    def test_equal_role_sampling_is_fixed_and_without_replacement(self):
        prepared = prepare(pools())
        first, second = np.random.default_rng(817), np.random.default_rng(817)
        for _ in range(10):
            a, b = f.sample_indices(prepared, first), f.sample_indices(prepared, second)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(set(map(int, a))), 16)
            self.assertEqual([prepared[int(i)].role for i in a].count(0), 8)
            self.assertEqual([prepared[int(i)].role for i in a].count(1), 8)

    def test_padding_mask_targets_and_role_are_not_feature_columns(self):
        prepared = prepare(pools(widths=(2, 3)))
        batch = f.make_batch(prepared, f.sample_indices(prepared, np.random.default_rng(2)), guard=lambda: None)
        self.assertEqual(batch.features.shape, (16, 3, f.DIM))
        self.assertEqual(batch.mask.dtype, np.dtype(bool))
        self.assertEqual(sorted(batch.mask.sum(axis=1).tolist()), [2] * 8 + [3] * 8)
        self.assertTrue(batch.mask[np.arange(16), batch.teacher_index].all())
        self.assertEqual(set(batch.__slots__), {'features', 'baseline_logits', 'mask', 'teacher_index'})
        weights = np.ones(f.DIM, dtype=np.float32)
        initial = f._numpy_components(weights, batch)
        batch.features[~batch.mask] = 1e30
        batch.baseline_logits[~batch.mask] = 1e30
        self.assertEqual(initial, f._numpy_components(weights, batch))

    def test_zero_residual_preserves_baseline_and_kl(self):
        batch = f.make_batch(prepare(pools()), np.arange(16, dtype=np.int64), guard=lambda: None)
        weights = np.zeros(f.DIM, dtype=np.float32)
        score = batch.baseline_logits + np.sum(batch.features * weights, axis=2)
        np.testing.assert_array_equal(score[batch.mask], batch.baseline_logits[batch.mask])
        value, ce, kl = f._numpy_components(weights, batch)
        self.assertAlmostEqual(kl, 0.0, places=7)
        self.assertAlmostEqual(value, ce, places=7)
        self.assertEqual(repr(batch), '<PrivateWaterFitBatch>')

    def test_teacher_ce_and_baseline_to_new_kl_are_correct(self):
        batch = f.make_batch(prepare(pools()), np.arange(16, dtype=np.int64), guard=lambda: None)
        weights = np.zeros(f.DIM, dtype=np.float32)
        weights[f.SHARED_DIM] = 0.4
        value, ce, kl = f._numpy_components(weights, batch)
        old = np.array([-0.3, 0.2], dtype=np.float64)
        new = old + [0.0, 0.4]
        oldp = np.exp(old - old.max()); oldp /= oldp.sum()
        newp = np.exp(new - new.max()); newp /= newp.sum()
        expected_ce = -np.log(newp[1])
        expected_kl = np.sum(oldp * np.log(oldp / newp))
        self.assertAlmostEqual(ce, expected_ce, places=6)
        self.assertAlmostEqual(kl, expected_kl, places=6)
        self.assertAlmostEqual(value, expected_ce + 0.1 * expected_kl, places=6)

    def test_numpy_finite_difference_shows_a_descent_update(self):
        batch = f.make_batch(prepare(pools()), np.arange(16, dtype=np.int64), guard=lambda: None)
        weights = np.zeros(f.DIM, dtype=np.float32)
        delta = np.float32(0.001)
        plus, minus = weights.copy(), weights.copy()
        plus[f.SHARED_DIM], minus[f.SHARED_DIM] = delta, -delta
        gradient = (f._numpy_components(plus, batch)[0] - f._numpy_components(minus, batch)[0]) / (2 * delta)
        self.assertLess(gradient, 0)
        weights[f.SHARED_DIM] -= np.float32(0.01 * gradient)
        self.assertLess(f._numpy_components(weights, batch)[0], f._numpy_components(np.zeros_like(weights), batch)[0])

    def test_prepared_features_are_private_immutable_copies(self):
        source = pools()
        prepared = prepare(source)
        snapshot = prepared[0].features.copy()
        source[0]['features'][0, 0] = 99
        source[0]['baseline_logits'][0] = 99
        np.testing.assert_array_equal(prepared[0].features, snapshot)
        self.assertFalse(prepared[0].features.flags.writeable)
        self.assertFalse(prepared[0].baseline_logits.flags.writeable)
        with self.assertRaises(ValueError): prepared[0].features.setflags(write=True)
        with self.assertRaises(ValueError): prepared[0].baseline_logits.setflags(write=True)
        self.assertEqual(repr(prepared[0]), '<PrivateWaterFitGroup>')

    def test_invalid_data_are_rejected_before_training(self):
        variants = []
        source = pools(); source[0]['features'][0, f.OP_START + f.OPS.index('WATER')] = 0
        source[0]['features'][0, f.OP_START + f.OPS.index('DIG')] = 1; variants.append(source)
        source = pools(); source[0]['features'][1, 0] = 1; variants.append(source)
        source = pools(); source[0]['features'][0, 0] = np.nan; variants.append(source)
        source = pools(); source[0]['baseline_logits'][0] = np.inf; variants.append(source)
        source = pools(); source[0]['teacher_index'] = 2; variants.append(source)
        source = pools(); source[0]['teacher_index'] = True; variants.append(source)
        source = pools(); source[0]['role'] = True; variants.append(source)
        source = pools(); source[0]['seed'] = 123; variants.append(source)
        source = pools(); source[0]['features'] = source[0]['features'].astype(np.float64); variants.append(source)
        source = pools(); source[0]['baseline_logits'] = source[0]['baseline_logits'].astype(np.float64); variants.append(source)
        source = pools(); source[0] = group(0, width=1, label=0); variants.append(source)
        variants.append([group(0) for _ in range(16)])
        for index, source in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(f.WaterFitError):
                prepare(source)
        with self.assertRaises(f.WaterFitError):
            f.fit(pools(), mean=np.zeros(f.DIM, np.float64), scale=np.ones(f.DIM, np.float64),
                  device=None, guard=lambda: None, rng_seed=True)

    def test_guard_failure_and_bad_batch_fail_closed(self):
        signal = MemoryError('synthetic stop')
        def guard(): raise signal
        with self.assertRaises(MemoryError) as caught:
            f.prepare_groups(pools(), mean=np.zeros(f.DIM, np.float64), scale=np.ones(f.DIM, np.float64), guard=guard)
        self.assertIs(caught.exception, signal)
        prepared = prepare(pools())
        with self.assertRaises(f.WaterFitError):
            f.make_batch(prepared, np.zeros(16, dtype=np.int64), guard=lambda: None)


@unittest.skipUnless(os.environ.get('WATER_FIT_JAX_CPU') == '1',
                     'JAX CPU test requires explicit opt-in after the memory check')
class WaterFitJaxCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psutil
        if psutil.virtual_memory().available < 5 * 1024**3:
            raise unittest.SkipTest('Initial physical-memory floor not met')
        cls.psutil = psutil
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import jax
        cls.jax = jax
        if jax.default_backend() != 'cpu':
            raise unittest.SkipTest('Explicit CPU backend unavailable')

    def test_64_synchronized_cpu_updates_change_weights(self):
        def guard():
            if self.psutil.virtual_memory().available < 4 * 1024**3:
                raise MemoryError('Synthetic CPU test memory floor')
        result = f.fit(pools(), mean=np.zeros(f.DIM, np.float64), scale=np.ones(f.DIM, np.float64),
                       device=self.jax.devices('cpu')[0], guard=guard, rng_seed=817)
        self.assertEqual(result.updates_completed, 64)
        self.assertTrue(result.nonzero_gradient_seen and result.weights_changed)
        self.assertEqual(result.weights.dtype, np.dtype('float32'))
        self.assertFalse(result.weights.flags.writeable)
        with self.assertRaises(ValueError): result.weights.setflags(write=True)
        self.assertEqual(repr(result), '<PrivateWaterFitResult>')

    def test_masked_jax_gradient_matches_independent_formula(self):
        import jax.numpy as jnp
        if self.psutil.virtual_memory().available < 4 * 1024**3:
            raise MemoryError('Synthetic CPU test memory floor')
        batch = f.make_batch(prepare(pools(widths=(2, 3))),
                             np.arange(16, dtype=np.int64), guard=lambda: None)
        weights = np.zeros(f.DIM, np.float32)
        weights[f.SHARED_DIM] = 0.4

        def score(w, features, base, mask, labels):
            return f._components(w, features, base, mask, labels, jnp)[0]

        gradient_fn = self.jax.value_and_grad(score)
        arguments = tuple(jnp.asarray(value) for value in
                          (weights, batch.features, batch.baseline_logits, batch.mask, batch.teacher_index))
        value, gradient = gradient_fn(*arguments)
        gradient.block_until_ready()
        expected = np.zeros(f.DIM, dtype=np.float64)
        for row in range(16):
            mask = batch.mask[row]
            x = batch.features[row, mask].astype(np.float64)
            old = batch.baseline_logits[row, mask].astype(np.float64)
            new = old + x @ weights.astype(np.float64)
            oldp = np.exp(old - old.max()); oldp /= oldp.sum()
            newp = np.exp(new - new.max()); newp /= newp.sum()
            target = int(batch.teacher_index[row])
            expected += (newp @ x - x[target] + 0.1 * ((newp - oldp) @ x)) / 16
        np.testing.assert_allclose(np.asarray(gradient), expected, atol=1e-6, rtol=1e-6)
        batch.features[~batch.mask] = 1e30
        batch.baseline_logits[~batch.mask] = 1e30
        padded = tuple(jnp.asarray(v) for v in
                       (weights, batch.features, batch.baseline_logits, batch.mask, batch.teacher_index))
        changed_value, changed_gradient = gradient_fn(*padded)
        changed_gradient.block_until_ready()
        self.assertAlmostEqual(float(value), float(changed_value), places=6)
        np.testing.assert_allclose(np.asarray(gradient), np.asarray(changed_gradient), atol=1e-6, rtol=0)

    def test_identical_candidates_have_exactly_zero_jax_gradient(self):
        import jax.numpy as jnp
        if self.psutil.virtual_memory().available < 4 * 1024**3:
            raise MemoryError('Synthetic CPU test memory floor')
        rng = np.random.default_rng(271)
        first = rng.normal(size=(16, 1, f.DIM)).astype(np.float32)
        features = np.repeat(first, 7, axis=1)
        base = rng.normal(size=(16, 7)).astype(np.float32)
        mask = np.ones((16, 7), dtype=bool)
        mask[::2, 4:] = False
        # Even huge finite padding is ignored before reference subtraction.
        features[~mask] = 1e30
        base[~mask] = 1e30
        labels = np.ones(16, dtype=np.int32)
        weights = np.zeros(f.DIM, dtype=np.float32)
        fn = self.jax.jit(self.jax.value_and_grad(
            lambda w, x, b, m, y: f._components(w, x, b, m, y, jnp)[0]))
        value, gradient = fn(*(jnp.asarray(v) for v in (weights, features, base, mask, labels)))
        gradient.block_until_ready()
        self.assertTrue(np.isfinite(float(value)))
        np.testing.assert_array_equal(np.asarray(gradient), np.zeros(f.DIM, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
