"""Pure synthetic checks; no real data, model, games or GPU are used."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_multisource_rank_v1 import core as rank
from learning.task_multisource_rank_v1.test_core import pools
from . import core


def interaction_params():
    ws = np.zeros((core.SHARED_DIM, core.HIDDEN), np.float32)
    wc = np.zeros((core.LOCAL_DIM, core.HIDDEN), np.float32)
    b, v = np.zeros(core.HIDDEN, np.float32), np.zeros(core.HIDDEN, np.float32)
    ws[0, 0], ws[0, 1] = 1, -1
    wc[0, 0], wc[0, 1] = 1, 1
    v[0], v[1] = 1, -1
    return core.Params(ws, wc, b, v)


def candidates():
    rows = np.zeros((2, core.LOCAL_DIM), np.float32)
    rows[:, 0] = (0, 1)
    rows[0, rank.OP_OFFSET + rank.OPS.index('DIG')] = 1
    rows[1, rank.OP_OFFSET + rank.OPS.index('WATER')] = 1
    return rows


def no_signal_pools():
    groups = []
    for source in rank.SOURCES:
        for _ in range(8):
            row = candidates()[:1]
            groups.append(rank.Group(np.zeros(core.SHARED_DIM, np.float32),
                np.repeat(row, 2, axis=0), np.array([0.2, -0.3], np.float64), 1, source))
    return groups


class StateResidualTests(unittest.TestCase):
    def setUp(self):
        self.mean, self.scale = np.zeros(core.DIM, np.float64), np.ones(core.DIM, np.float64)

    def test_bounded_reproducible_zero_output_initialization(self):
        first, again, other = (core.initial_params(seed) for seed in (817, 817, 818))
        for name in ('ws', 'wc', 'b', 'v'):
            a, b = getattr(first, name), getattr(again, name)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.dtype, np.dtype('float32'))
            self.assertFalse(a.flags.writeable)
            with self.assertRaises(ValueError): a.setflags(write=True)
        self.assertTrue(np.any(first.ws != 0) and np.any(first.wc != 0))
        self.assertLessEqual(float(np.max(np.abs(first.ws))), float(np.float32(core.INIT_BOUND)))
        self.assertLessEqual(float(np.max(np.abs(first.wc))), float(np.float32(core.INIT_BOUND)))
        self.assertFalse(np.any(first.b) or np.any(first.v))
        self.assertFalse(np.array_equal(first.ws, other.ws))
        self.assertEqual(repr(first), '<PrivateStateResidualParams>')

    def test_zero_output_exactly_preserves_float64_argmax_and_ties(self):
        shared, local, params = np.zeros(core.SHARED_DIM, np.float32), candidates(), core.initial_params(1)
        almost_tied = np.array([1., np.nextafter(np.float64(1.), np.float64(2.))], np.float64)
        self.assertEqual(core.select_values(shared, local, almost_tied, self.mean, self.scale, params), 1)
        self.assertEqual(core.select_values(shared, local, np.ones(2, np.float64),
                                            self.mean, self.scale, params), 0)
        self.assertEqual(core.select_values(shared, local, np.array([1e308, -1e308]),
                                            self.mean, self.scale, params), 0)

    def test_shared_state_can_reverse_cross_operation_preference(self):
        local, baseline, params = candidates(), np.zeros(2, np.float64), interaction_params()
        shared = np.zeros(core.SHARED_DIM, np.float32)
        shared[0] = -2
        first = core.select_values(shared, local, baseline, self.mean, self.scale, params)
        shared[0] = 2
        second = core.select_values(shared, local, baseline, self.mean, self.scale, params)
        self.assertEqual((first, second), (1, 0))

    def test_common_offsets_do_not_erase_baseline_or_residual_differences(self):
        shared, local = np.zeros(core.SHARED_DIM, np.float32), candidates()
        ws = np.zeros((core.SHARED_DIM, core.HIDDEN), np.float32)
        wc = np.zeros((core.LOCAL_DIM, core.HIDDEN), np.float32)
        bias, output = np.zeros(core.HIDDEN, np.float32), np.zeros(core.HIDDEN, np.float32)
        bias[0], output[0] = 1, 1
        params = core.Params(ws, wc, bias, output)
        baseline = np.array([0., 1e-20], np.float64)
        common = np.float64(np.tanh(np.float32(1)))
        self.assertEqual(int(np.argmax(baseline + common)), 0)  # naive addition loses the gap
        self.assertEqual(core.select_values(shared, local, baseline, self.mean, self.scale, params), 1)
        shared[0] = -2
        baseline = np.array([1e20, 1e20], np.float64)
        params = interaction_params()
        hidden = np.tanh(shared @ params.ws + local @ params.wc + params.b)
        residual = hidden @ params.v
        self.assertEqual(int(np.argmax(baseline + residual.astype(np.float64))), 0)
        self.assertEqual(core.select_values(shared, local, baseline, self.mean, self.scale, params), 1)

    def test_nonfinite_shape_dtype_and_categories_are_rejected(self):
        base = np.zeros(2, np.float64)
        shared, local, good = np.zeros(core.SHARED_DIM, np.float32), candidates(), interaction_params()
        variants = [core.Params(good.ws[:-1], good.wc, good.b, good.v),
                    core.Params(good.ws.astype(np.float64), good.wc, good.b, good.v),
                    core.Params(good.ws, good.wc, good.b, np.full(core.HIDDEN, np.nan, np.float32)),
                    core.Params(good.ws, good.wc, np.zeros(core.HIDDEN + 1, np.float32), good.v)]
        for params in variants:
            with self.subTest(case=repr(params)), self.assertRaises(ValueError):
                core.select_values(shared, local, base, self.mean, self.scale, params)
        bad = local.copy(); bad[0, rank.OP_OFFSET] = np.nan
        with self.assertRaises(ValueError): core.select_values(shared, bad, base, self.mean, self.scale, good)
        bad = local.copy(); bad[0, rank.OP_OFFSET:rank.OP_OFFSET + len(rank.OPS)] = 0
        with self.assertRaises(ValueError): core.select_values(shared, bad, base, self.mean, self.scale, good)
        with self.assertRaises(ValueError): core.select_values(shared, local, np.array([0., np.inf]), self.mean, self.scale, good)
        invalid_scale = self.scale.copy(); invalid_scale[0] = 0
        with self.assertRaises(ValueError): core.select_values(shared, local, base, self.mean, invalid_scale, good)
        extreme = interaction_params()
        extreme.ws[0, 0] = np.finfo(np.float32).max
        shared[0] = np.finfo(np.float32).max
        with self.assertRaises(ValueError): core.select_values(shared, local, base, self.mean, self.scale, extreme)

    def test_guard_and_float32_baseline_stability_fail_before_training(self):
        stop = MemoryError('synthetic guard')
        def guard(): raise stop
        with self.assertRaises(MemoryError) as caught:
            core.fit(pools(), mean=self.mean, scale=self.scale, device=None, guard=guard, rng_seed=1)
        self.assertIs(caught.exception, stop)
        source = pools()
        group = source[1]
        source[1] = rank.Group(group.shared, group.candidates,
                               np.array([-1e-50, 0.], np.float64), group.target, group.source)
        with self.assertRaises(ValueError):
            core.fit(source, mean=self.mean, scale=self.scale, device=None, guard=lambda: None, rng_seed=1)
        with self.assertRaises(ValueError): core.initial_params(True)

    def test_baseline_centering_preserves_probabilities_and_rejects_overflow(self):
        baseline = np.array([1e8, 1e8 - 4], np.float64)
        direct = baseline.astype(np.float32)
        self.assertEqual(float(direct[0]), float(direct[1]))
        centered64, centered32 = core._training_baseline(baseline)
        true_p = np.exp(baseline - baseline.max()); true_p /= true_p.sum()
        train_p = np.exp(centered32.astype(np.float64)); train_p /= train_p.sum()
        np.testing.assert_allclose(train_p, true_p, atol=1e-8, rtol=1e-8)
        np.testing.assert_array_equal(centered64, np.array([0., -4.], np.float64))
        source = pools()
        group = source[1]
        source[1] = rank.Group(group.shared, group.candidates, baseline, group.target, group.source)
        prepared = core.prepare_groups(source, mean=self.mean, scale=self.scale, guard=lambda: None)
        np.testing.assert_array_equal(prepared[1].baseline, np.array([0., -4.], np.float32))
        with self.assertRaises(core.StateResidualError):
            core._training_baseline(np.array([1e308, -1e308], np.float64))


@unittest.skipUnless(os.environ.get('STATE_RESIDUAL_JAX_CPU') == '1',
                     'Explicit bounded synthetic JAX CPU test was not requested')
class StateResidualCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psutil
        if psutil.virtual_memory().available < 5 * 1024**3:
            raise unittest.SkipTest('Initial physical-memory floor not met')
        import jax
        if jax.default_backend() != 'cpu':
            raise unittest.SkipTest('Explicit CPU backend unavailable')
        cls.psutil, cls.jax = psutil, jax

    def guard(self):
        if self.psutil.virtual_memory().available < 4 * 1024**3:
            raise MemoryError('Synthetic CPU test memory floor')

    def test_64_synchronized_cpu_updates_change_output_layer(self):
        result = core.fit(pools(), mean=np.zeros(core.DIM, np.float64),
                          scale=np.ones(core.DIM, np.float64),
                          device=self.jax.devices('cpu')[0], guard=self.guard, rng_seed=817)
        self.assertEqual(result.updates_completed, 64)
        self.assertTrue(result.nonzero_gradient_seen and result.output_layer_changed)
        self.assertTrue(np.any(result.params.v != 0))
        self.assertEqual(repr(result), '<PrivateStateResidualFitResult>')
        for name in ('ws', 'wc', 'b', 'v'):
            value = getattr(result.params, name)
            self.assertEqual(value.dtype, np.dtype('float32'))
            self.assertTrue(np.isfinite(value).all())
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError): value.setflags(write=True)

    def test_masked_gradient_matches_independent_probability_formula(self):
        import jax.numpy as jnp
        self.guard()
        pool = core.prepare_groups(pools(), mean=np.zeros(core.DIM, np.float64),
                                   scale=np.ones(core.DIM, np.float64), guard=self.guard)
        batch = rank.make_batch(pool, rank.sample_indices(pool, np.random.default_rng(4)), guard=self.guard)
        init = core.initial_params(81)
        raw = {'ws': init.ws.copy(), 'wc': init.wc.copy(), 'b': init.b.copy(),
               'v': np.linspace(-0.2, 0.2, core.HIDDEN, dtype=np.float32)}
        fn = self.jax.jit(self.jax.value_and_grad(
            lambda params, x, b, mask, y: core._components(params, x, b, mask, y, jnp)[0]))
        params = {key: jnp.asarray(value) for key, value in raw.items()}
        inputs = tuple(jnp.asarray(value) for value in
                       (batch.features, batch.baseline, batch.mask, batch.target))
        value, gradient = fn(params, *inputs)
        gradient['v'].block_until_ready()
        expected = np.zeros(core.HIDDEN, np.float64)
        for row in range(16):
            mask = batch.mask[row]
            x = batch.features[row, mask]
            old = batch.baseline[row, mask].astype(np.float64)
            h = np.tanh(x[:1, :core.SHARED_DIM] @ raw['ws'] + x[:, core.SHARED_DIM:] @ raw['wc'] + raw['b'])
            relative = h.astype(np.float64) - h[:1].astype(np.float64)
            new = old + relative @ raw['v'].astype(np.float64)
            oldp = np.exp(old - old.max()); oldp /= oldp.sum()
            newp = np.exp(new - new.max()); newp /= newp.sum()
            expected += (newp @ relative - relative[int(batch.target[row])]
                         + 0.1 * ((newp - oldp) @ relative)) / 16
        np.testing.assert_allclose(np.asarray(gradient['v']), expected, atol=1e-6, rtol=1e-6)
        batch.features[~batch.mask] = 1e30
        batch.baseline[~batch.mask] = 1e30
        padded = tuple(jnp.asarray(value) for value in
                       (batch.features, batch.baseline, batch.mask, batch.target))
        changed_value, changed_gradient = fn(params, *padded)
        changed_gradient['v'].block_until_ready()
        self.assertAlmostEqual(float(value), float(changed_value), places=6)
        for name in raw:
            np.testing.assert_allclose(np.asarray(gradient[name]), np.asarray(changed_gradient[name]),
                                       atol=1e-6, rtol=0)

    def test_wrong_device_and_zero_signal_do_not_return_a_model(self):
        with self.assertRaises(core.StateResidualError):
            core.fit(pools(), mean=np.zeros(core.DIM, np.float64), scale=np.ones(core.DIM, np.float64),
                     device=None, guard=self.guard, rng_seed=1)
        # A single synthetic step suffices to exercise the failure contract;
        # successful production fits are fixed at 64 updates.
        with patch.object(core, 'UPDATES', 1), self.assertRaises(core.StateResidualError):
            core.fit(no_signal_pools(), mean=np.zeros(core.DIM, np.float64),
                     scale=np.ones(core.DIM, np.float64), device=self.jax.devices('cpu')[0],
                     guard=self.guard, rng_seed=1)


if __name__ == '__main__':
    unittest.main()
