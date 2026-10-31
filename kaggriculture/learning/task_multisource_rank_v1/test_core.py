"""Synthetic tests; no real dataset, model, game, or accelerator is used."""
import os
import unittest

import numpy as np

from . import core


def group(source, *, width=2, target=None):
    shared = np.arange(core.SHARED_DIM, dtype=np.float32) / 100
    candidates = np.zeros((width, core.LOCAL_DIM), dtype=np.float32)
    for index in range(width):
        candidates[index, core.OP_OFFSET + index % len(core.OPS)] = 1
        candidates[index, 0] = index
    baseline = np.linspace(0.2, -0.3, width, dtype=np.float64)
    return core.Group(shared, candidates, baseline, width - 1 if target is None else target, source)


def pools():
    return [group(source, width=1 + (i % 3)) for source in core.SOURCES for i in range(10)]


def prepared(groups):
    return core.prepare_groups(groups, mean=np.zeros(core.DIM, dtype=np.float64),
                               scale=np.ones(core.DIM, dtype=np.float64), guard=lambda: None)


class RankCoreTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.zeros(core.DIM, dtype=np.float64)
        self.scale = np.ones(core.DIM, dtype=np.float64)

    def test_balanced_fixed_sampling_without_replacement(self):
        groups = prepared(pools())
        a, b = np.random.default_rng(817), np.random.default_rng(817)
        for _ in range(5):
            left, right = core.sample_indices(groups, a), core.sample_indices(groups, b)
            np.testing.assert_array_equal(left, right)
            self.assertEqual(len(set(map(int, left))), 16)
            self.assertEqual([groups[int(i)].source for i in left].count('old'), 8)
            self.assertEqual([groups[int(i)].source for i in left].count('new'), 8)

    def test_preparation_preserves_split_storage_and_private_copies(self):
        source = pools()
        snapshot = prepared(source)
        self.assertEqual(snapshot[1].shared.shape, (core.SHARED_DIM,))
        self.assertEqual(snapshot[1].candidates.shape, (2, core.LOCAL_DIM))
        saved = snapshot[1].candidates.copy()
        source[1].candidates[0, 0] = 999
        source[1].baseline[0] = 999
        np.testing.assert_array_equal(snapshot[1].candidates, saved)
        for array in (snapshot[1].shared, snapshot[1].candidates, snapshot[1].baseline):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError): array.setflags(write=True)
        self.assertEqual(repr(snapshot[1]), '<PrivatePreparedRankGroup>')

    def test_float64_to_float32_argmax_change_rejected_before_fit(self):
        source = pools()
        chosen = source[1]
        changed = core.Group(chosen.shared, chosen.candidates,
                             np.array([1.0, 1.0 + 1e-8], dtype=np.float64),
                             chosen.target, chosen.source)
        self.assertEqual(int(np.argmax(changed.baseline)), 1)
        self.assertEqual(int(np.argmax(changed.baseline.astype(np.float32))), 0)
        source[1] = changed
        with self.assertRaises(core.RankError): prepared(source)

    def test_mask_and_loss_match_independent_probabilities(self):
        pool = prepared(pools())
        batch = core.make_batch(pool, core.sample_indices(pool, np.random.default_rng(3)), guard=lambda: None)
        self.assertEqual(batch.features.shape, (16, 3, core.DIM))
        self.assertEqual(set(batch.__slots__), {'features', 'baseline', 'mask', 'target'})
        self.assertTrue(batch.mask[np.arange(16), batch.target].all())
        weights = np.zeros(core.DIM, dtype=np.float32)
        weights[core.SHARED_DIM] = np.float32(0.4)
        total, (ce, kl) = core._components(weights, batch.features, batch.baseline,
                                          batch.mask, batch.target, np)
        ces, kls = [], []
        for row in range(16):
            m = batch.mask[row]
            old = batch.baseline[row, m].astype(np.float64)
            x = batch.features[row, m].astype(np.float64)
            new = old + x @ weights.astype(np.float64)
            oldp = np.exp(old - old.max()); oldp /= oldp.sum()
            newp = np.exp(new - new.max()); newp /= newp.sum()
            ces.append(-np.log(newp[int(batch.target[row])]))
            kls.append(np.sum(oldp * np.log(oldp / newp)))
        self.assertAlmostEqual(float(ce), float(np.mean(ces)), places=6)
        self.assertAlmostEqual(float(kl), float(np.mean(kls)), places=6)
        self.assertAlmostEqual(float(total), float(np.mean(ces) + 0.1 * np.mean(kls)), places=6)
        batch.features[~batch.mask] = 1e30
        batch.baseline[~batch.mask] = 1e30
        again, _ = core._components(weights, batch.features, batch.baseline, batch.mask, batch.target, np)
        self.assertEqual(float(total), float(again))

    def test_zero_residual_preserves_float64_argmax_and_cross_operation_switch(self):
        source = group('old')
        zero = np.zeros(core.DIM, dtype=np.float32)
        baseline = np.array([1.0, 1.0 + 1e-8], dtype=np.float64)
        source = core.Group(source.shared, source.candidates, baseline, 1, 'old')
        self.assertEqual(core.select(source, self.mean, self.scale, zero), 1)
        tied = core.Group(source.shared, source.candidates, np.ones(2, np.float64), 1, 'old')
        self.assertEqual(core.select(tied, self.mean, self.scale, zero), 0)
        weights = zero.copy()
        weights[core.SHARED_DIM + core.OP_OFFSET + 1] = np.float32(0.1)
        self.assertNotEqual(int(np.argmax(tied.candidates[0, core.OP_OFFSET:core.OP_OFFSET + len(core.OPS)])),
                            int(np.argmax(tied.candidates[1, core.OP_OFFSET:core.OP_OFFSET + len(core.OPS)])))
        self.assertEqual(core.select(tied, self.mean, self.scale, weights), 1)
        self.assertEqual(core.select_values(tied.shared, tied.candidates, tied.baseline,
                                            self.mean, self.scale, weights), 1)

    def test_single_candidate_is_supported(self):
        only = group('new', width=1)
        self.assertEqual(core.select(only, self.mean, self.scale, np.ones(core.DIM, np.float32)), 0)

    def test_invalid_inputs_and_missing_sources_are_rejected(self):
        variants = []
        source = pools(); g = source[0]; source[0] = core.Group(g.shared, g.candidates, g.baseline, True, g.source); variants.append(source)
        source = pools(); g = source[0]; source[0] = core.Group(g.shared, g.candidates, g.baseline, 9, g.source); variants.append(source)
        source = pools(); g = source[0]; source[0] = core.Group(g.shared, g.candidates, g.baseline, 0, 'other'); variants.append(source)
        source = pools(); source[0].shared[0] = np.nan; variants.append(source)
        source = pools(); source[0].candidates[0, 0] = np.inf; variants.append(source)
        source = pools(); source[0].baseline[0] = np.nan; variants.append(source)
        source = pools(); source[0].candidates[0, core.OP_OFFSET] = 0; variants.append(source)
        source = pools(); source[0].candidates[0, core.OP_OFFSET + 1] = 1; variants.append(source)
        source = pools(); g = source[0]; source[0] = core.Group(g.shared.astype(np.float64), g.candidates, g.baseline, 0, g.source); variants.append(source)
        source = pools(); g = source[0]; source[0] = core.Group(g.shared, g.candidates, g.baseline.astype(np.float32), 0, g.source); variants.append(source)
        source = pools(); source[0] = group('old', width=257); variants.append(source)
        variants.append([group('old') for _ in range(20)])
        for index, source in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(core.RankError): prepared(source)
        bad_scale = self.scale.copy(); bad_scale[0] = 0
        with self.assertRaises(core.RankError):
            core.prepare_groups(pools(), mean=self.mean, scale=bad_scale, guard=lambda: None)
        bad_weights = np.zeros(core.DIM, dtype=np.float32); bad_weights[0] = np.nan
        with self.assertRaises(core.RankError): core.select(group('old'), self.mean, self.scale, bad_weights)

    def test_guard_and_invalid_batch_fail_closed(self):
        signal = MemoryError('synthetic stop')
        def guard(): raise signal
        with self.assertRaises(MemoryError) as caught:
            core.prepare_groups(pools(), mean=self.mean, scale=self.scale, guard=guard)
        self.assertIs(caught.exception, signal)
        pool = prepared(pools())
        with self.assertRaises(core.RankError):
            core.make_batch(pool, np.zeros(16, dtype=np.int64), guard=lambda: None)


@unittest.skipUnless(os.environ.get('MULTISOURCE_JAX_CPU') == '1',
                     'Explicit bounded synthetic JAX CPU test was not requested')
class RankCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psutil
        if psutil.virtual_memory().available < 5 * 1024**3:
            raise unittest.SkipTest('Initial physical-memory floor not met')
        import jax
        cls.psutil, cls.jax = psutil, jax
        if jax.default_backend() != 'cpu':
            raise unittest.SkipTest('Explicit CPU backend unavailable')

    def test_64_synchronized_adam_updates_on_cpu_change_weights(self):
        def guard():
            if self.psutil.virtual_memory().available < 4 * 1024**3:
                raise MemoryError('Synthetic CPU test memory floor')
        result = core.fit(pools(), mean=np.zeros(core.DIM, np.float64),
                          scale=np.ones(core.DIM, np.float64),
                          device=self.jax.devices('cpu')[0], guard=guard, rng_seed=817)
        self.assertEqual(result.updates_completed, 64)
        self.assertTrue(result.nonzero_gradient_seen and result.weights_changed)
        self.assertEqual(result.weights.shape, (core.DIM,))
        self.assertEqual(result.weights.dtype, np.dtype('float32'))
        self.assertFalse(result.weights.flags.writeable)
        np.testing.assert_array_equal(result.weights[:core.SHARED_DIM], np.zeros(core.SHARED_DIM, np.float32))
        with self.assertRaises(ValueError): result.weights.setflags(write=True)

    def test_masked_jax_gradient_matches_probability_formula(self):
        import jax.numpy as jnp
        if self.psutil.virtual_memory().available < 4 * 1024**3:
            raise MemoryError('Synthetic CPU test memory floor')
        pool = prepared(pools())
        batch = core.make_batch(pool, core.sample_indices(pool, np.random.default_rng(4)), guard=lambda: None)
        weights = np.zeros(core.DIM, np.float32)
        weights[core.SHARED_DIM] = np.float32(0.4)
        weights[core.SHARED_DIM + core.OP_OFFSET + 1] = np.float32(-0.2)
        fn = self.jax.jit(self.jax.value_and_grad(
            lambda w, x, b, m, y: core._components(w, x, b, m, y, jnp)[0]))
        arguments = tuple(jnp.asarray(value) for value in
                          (weights, batch.features, batch.baseline, batch.mask, batch.target))
        value, gradient = fn(*arguments)
        gradient.block_until_ready()
        expected = np.zeros(core.DIM, np.float64)
        for row in range(16):
            mask = batch.mask[row]
            x = batch.features[row, mask].astype(np.float64)
            old = batch.baseline[row, mask].astype(np.float64)
            new = old + x @ weights.astype(np.float64)
            oldp = np.exp(old - old.max()); oldp /= oldp.sum()
            newp = np.exp(new - new.max()); newp /= newp.sum()
            expected += (newp @ x - x[int(batch.target[row])] + 0.1 * ((newp - oldp) @ x)) / 16
        np.testing.assert_allclose(np.asarray(gradient), expected, atol=1e-6, rtol=1e-6)
        batch.features[~batch.mask] = 1e30
        batch.baseline[~batch.mask] = 1e30
        padded = tuple(jnp.asarray(v) for v in
                       (weights, batch.features, batch.baseline, batch.mask, batch.target))
        changed_value, changed_gradient = fn(*padded)
        changed_gradient.block_until_ready()
        self.assertAlmostEqual(float(value), float(changed_value), places=6)
        np.testing.assert_allclose(np.asarray(gradient), np.asarray(changed_gradient), atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
