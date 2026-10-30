"""Synthetic CPU-only checks of the in-memory paired update core."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_training_v1.fit_core import fit_pair
from learning.task_pair_training_v1.test_objective import fixture


AVAILABLE = importlib.util.find_spec('jax') is not None and importlib.util.find_spec('optax') is not None


def compiled_components():
    """Construct independent CE and worker gradients for synthetic checks."""
    import jax
    import jax.numpy as jnp
    from learning.task_pair_training_v1.objective import ce_only_objective, objective_components

    def ce(p, x, mask, labels, weights, denominator):
        return ce_only_objective(p, x, mask, labels, weights, denominator, xp=jnp)

    def worker(p, x, mask, labels, weights, denominator, targets, eligible, count):
        return objective_components(p, x, mask, labels, weights, denominator,
                                    targets, eligible, count, xp=jnp)[1]

    return jax.jit(jax.value_and_grad(ce)), jax.jit(jax.value_and_grad(worker))


def component_totals(params, parts, device, functions):
    """Sum components at one fixed parameter point with complete-batch denominators."""
    import jax
    import jax.numpy as jnp

    denominator = np.float32(sum(float(np.sum(p['group_weights'], dtype=np.float64)) for p in parts))
    count = sum(int(np.count_nonzero(p['eligible'])) for p in parts)
    denominator, count = jax.device_put((denominator, np.int32(count)), device)
    ce_value = jax.device_put(np.float32(0), device)
    worker_value = jax.device_put(np.float32(0), device)
    ce_total = jax.tree_util.tree_map(jnp.zeros_like, params)
    worker_total = jax.tree_util.tree_map(jnp.zeros_like, params)
    for part in parts:
        x, mask, labels, weights, targets, eligible = jax.device_put(tuple(
            part[name] for name in ('features', 'candidate_mask', 'labels', 'group_weights',
                                    'target_mask', 'eligible')), device)
        ce, cg = functions[0](params, x, mask, labels, weights, denominator)
        worker, wg = functions[1](params, x, mask, labels, weights, denominator,
                                  targets, eligible, count)
        ce_value += ce
        worker_value += worker
        ce_total = jax.tree_util.tree_map(lambda a, b: a + b, ce_total, cg)
        worker_total = jax.tree_util.tree_map(lambda a, b: a + b, worker_total, wg)
    return ce_value, worker_value, ce_total, worker_total


def reference_adam(initial, plan, packer, device):
    """Independently apply fresh Optax states to one fixed partition schedule."""
    import jax
    import optax

    functions = compiled_components()
    control = jax.device_put({name: value.copy() for name, value in initial.items()}, device)
    experiment = jax.device_put({name: value.copy() for name, value in initial.items()}, device)
    opt_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    opt_b = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    state_a, state_b = opt_a.init(control), opt_b.init(experiment)

    @jax.jit
    def update_a(params, state, gradient):
        changes, state = opt_a.update(gradient, state, params)
        return optax.apply_updates(params, changes), state

    @jax.jit
    def update_b(params, state, gradient):
        changes, state = opt_b.update(gradient, state, params)
        return optax.apply_updates(params, changes), state

    for row in plan:
        parts = packer(row)
        _, _, ce_a, _ = component_totals(control, parts, device, functions)
        _, _, ce_b, worker_b = component_totals(experiment, parts, device, functions)
        total_b = jax.tree_util.tree_map(lambda ce, aux: ce + 0.25 * aux, ce_b, worker_b)
        control, state_a = update_a(control, state_a, ce_a)
        experiment, state_b = update_b(experiment, state_b, total_b)
    return control, experiment


@unittest.skipUnless(AVAILABLE, 'JAX and Optax are required')
class FitCoreTests(unittest.TestCase):
    def setUp(self):
        import jax
        self.device = jax.devices('cpu')[0]
        cpu_context = jax.default_device(self.device)
        cpu_context.__enter__()
        self.addCleanup(cpu_context.__exit__, None, None, None)
        (self.params, self.features, self.mask, self.labels,
         self.weights, self.targets, self.eligible) = fixture()
        self.plan = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)

    def part(self, identifiers, *, width=4, padding=0):
        ids = np.asarray(identifiers, dtype=np.int64)
        row_count = len(ids)
        capacity = row_count + padding
        features = np.zeros((capacity, width, self.features.shape[-1]), np.float32)
        mask = np.zeros((capacity, width), bool)
        mask[:, 0] = True
        labels = np.zeros(capacity, np.int32)
        weights = np.zeros(capacity, np.float32)
        targets = np.zeros((capacity, width), bool)
        eligible = np.zeros(capacity, bool)
        features[:row_count] = self.features[ids, :width]
        mask[:row_count] = self.mask[ids, :width]
        labels[:row_count] = self.labels[ids]
        weights[:row_count] = self.weights[ids]
        targets[:row_count] = self.targets[ids, :width]
        eligible[:row_count] = self.eligible[ids]
        return dict(ids=ids, features=features, candidate_mask=mask, labels=labels,
                    group_weights=weights, target_mask=targets, eligible=eligible)

    def pack_whole(self, ids):
        return [self.part(ids, padding=1)]

    def pack_split(self, ids):
        # Reordering and different widths are legal when each group occurs once.
        return [self.part([int(group)], width=2 if group == 2 else 4, padding=1)
                for group in ids[::-1]]

    @staticmethod
    def assert_same_params(test, first, second, *, rtol=2e-5, atol=2e-6):
        for name in first:
            np.testing.assert_allclose(np.asarray(first[name]), np.asarray(second[name]),
                                       rtol=rtol, atol=atol, err_msg=name)

    def test_both_arms_update_from_same_initialization_and_aux_changes_result(self):
        before = {name: value.copy() for name, value in self.params.items()}
        calls = []
        def pack(ids):
            calls.append(len(ids))
            return self.pack_whole(ids)
        result = fit_pair(self.params, self.plan, pack, device=self.device, check=lambda: None)
        self.assertEqual(calls, [3, 3])
        self.assertEqual(result['completed_updates'], 2)
        self.assertEqual(result['aux_nonzero_updates'], 2)
        self.assertTrue(result['control_changed_from_initial'])
        self.assertTrue(result['experiment_changed_from_initial'])
        self.assertTrue(result['arms_differ'])
        self.assertTrue(all(step['aux_gradient_nonzero'] for step in result['steps']))
        self.assertTrue(all(step['control_changed'] and step['experiment_changed'] for step in result['steps']))
        self.assertEqual(result['device_platform'], 'cpu')
        self.assertFalse(any('seed' in key or 'ids' in key for key in result))
        for name in before:
            np.testing.assert_array_equal(self.params[name], before[name])
            self.assertEqual(result['control_params'][name].devices(), {self.device})
            self.assertEqual(result['experiment_params'][name].devices(), {self.device})

    def test_bucketed_microbatch_accumulation_matches_whole_batch(self):
        # Original positive biases make a near-zero bias gradient, which Adam's
        # epsilon can amplify differently for two floating-point partitions.
        # This separate example has observable candidate-dependent ReLU gates.
        params = {name: value.copy() for name, value in self.params.items()}
        params['b1'][:] = -0.2
        hidden = np.maximum(self.features @ params['w0'] + params['b0'], 0) @ params['w1'] + params['b1']
        gate_differs = [np.any((hidden[i, self.mask[i]] > 0) != (hidden[i, 0] > 0)) for i in range(3)]
        self.assertTrue(any(gate_differs))
        import jax
        with jax.default_matmul_precision('highest'):
            current = jax.device_put(params, self.device)
            _, _, gradient, _ = component_totals(current, self.pack_whole(self.plan[0]),
                                                  self.device, compiled_components())
            self.assertTrue(np.all(np.abs(np.asarray(gradient['b1'])) > 1e-3))
            whole = fit_pair(params, self.plan, self.pack_whole, device=self.device, check=lambda: None)
            split = fit_pair(params, self.plan, self.pack_split, device=self.device, check=lambda: None)
        self.assert_same_params(self, whole['control_params'], split['control_params'])
        self.assert_same_params(self, whole['experiment_params'], split['experiment_params'])
        self.assertEqual(whole['aux_nonzero_updates'], split['aux_nonzero_updates'])

    def test_near_zero_bias_gradients_match_at_same_parameter_point(self):
        import jax
        # The original example has equal candidate activation gates within each
        # group, so its bias gradient is mathematically zero. Tiny FP32 residuals
        # can be amplified by Adam; compare gradients at the same point instead
        # of demanding tight multi-step parameter equality across partitions.
        hidden = np.maximum(self.features @ self.params['w0'] + self.params['b0'], 0)
        hidden = hidden @ self.params['w1'] + self.params['b1']
        for i in range(3):
            gates = hidden[i, self.mask[i]] > 0
            self.assertTrue(np.all(gates == gates[0]))
        with jax.default_matmul_precision('highest'):
            params = jax.device_put(self.params, self.device)
            functions = compiled_components()
            whole = component_totals(params, self.pack_whole(self.plan[0]), self.device, functions)
            split = component_totals(params, self.pack_split(self.plan[0]), self.device, functions)
        for first, second in zip(whole[:2], split[:2], strict=True):
            np.testing.assert_allclose(np.asarray(first), np.asarray(second), rtol=0, atol=5e-7)
        for whole_grad, split_grad in zip(whole[2:], split[2:], strict=True):
            for name in self.params:
                np.testing.assert_allclose(np.asarray(whole_grad[name]), np.asarray(split_grad[name]),
                                           rtol=0, atol=5e-7, err_msg=name)
        self.assertLessEqual(float(np.max(np.abs(np.asarray(whole[2]['b1'])))), 5e-7)
        self.assertLessEqual(float(np.max(np.abs(np.asarray(split[2]['b1'])))), 5e-7)

    def test_each_fixed_partition_matches_independent_fresh_adam(self):
        import jax
        # Plan order, microbatch partition and software version are part of the
        # reproducibility contract; do not infer bitwise equality across them.
        with jax.default_matmul_precision('highest'):
            for packer in (self.pack_whole, self.pack_split):
                with self.subTest(partition=packer.__name__):
                    expected_a, expected_b = reference_adam(self.params, self.plan, packer, self.device)
                    actual = fit_pair(self.params, self.plan, packer, device=self.device, check=lambda: None)
                    self.assert_same_params(self, actual['control_params'], expected_a)
                    self.assert_same_params(self, actual['experiment_params'], expected_b)

    def test_control_matches_independent_fresh_adam_ce_update(self):
        import jax
        import jax.numpy as jnp
        import optax
        from learning.task_pair_training_v1.objective import ce_only_objective
        plan = self.plan[:1]
        part = self.pack_whole(plan[0])[0]
        params = jax.device_put(self.params, self.device)
        arrays = jax.device_put(tuple(part[key] for key in ('features', 'candidate_mask', 'labels', 'group_weights')),
                                self.device)
        denominator = np.float32(np.sum(part['group_weights'], dtype=np.float64))
        gradients = jax.grad(lambda p: ce_only_objective(p, *arrays, denominator, xp=jnp))(params)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
        changes, _ = optimizer.update(gradients, optimizer.init(params), params)
        expected = optax.apply_updates(params, changes)
        result = fit_pair(self.params, plan, self.pack_whole, device=self.device, check=lambda: None)
        self.assert_same_params(self, result['control_params'], expected)

    def test_bad_group_membership_and_empty_eligibility_fail(self):
        plan = self.plan[:1]
        def missing(ids):
            return [self.part(ids[:-1])]
        def duplicate(ids):
            return [self.part([int(ids[0]), int(ids[0]), int(ids[1])])]
        def duplicate_across_parts(ids):
            return [self.part([int(ids[0]), int(ids[1])]),
                    self.part([int(ids[1])])]
        def no_eligible(ids):
            part = self.part(ids)
            part['eligible'][:] = False
            part['target_mask'][:] = False
            return [part]
        for pack in (missing, duplicate, duplicate_across_parts, no_eligible):
            with self.subTest(pack=pack.__name__), self.assertRaises(ValueError):
                fit_pair(self.params, plan, pack, device=self.device, check=lambda: None)
        with self.assertRaises(ValueError):
            fit_pair(self.params, np.array([[0, 0]], np.int64), self.pack_whole,
                     device=self.device, check=lambda: None)

    def test_oversized_or_wrong_dtype_microbatch_rejected(self):
        def oversized(ids):
            part = self.part(ids)
            # Broadcast view is tiny; rejecting it must happen before copying.
            part['features'] = np.broadcast_to(np.zeros(1, np.float32), (4096, 256, 3))
            return [part]
        def wrong_dtype(ids):
            part = self.part(ids)
            part['features'] = part['features'].astype(np.float64)
            return [part]
        for pack in (oversized, wrong_dtype):
            with self.subTest(pack=pack.__name__), self.assertRaises(ValueError):
                fit_pair(self.params, self.plan[:1], pack, device=self.device, check=lambda: None)

    def test_bad_or_oversized_initial_model_rejected_before_any_parameter_copy(self):
        scalar = np.asarray(1.0, dtype=np.float32)
        too_wide = dict(self.params)
        too_wide['w0'] = np.broadcast_to(scalar, (3, 1_000_000_000))
        too_wide['b0'] = np.broadcast_to(scalar, (1_000_000_000,))
        too_many_bytes = {
            'w0': np.broadcast_to(scalar, (1024, 1024)), 'b0': np.broadcast_to(scalar, (1024,)),
            'w1': np.broadcast_to(scalar, (1024, 1024)), 'b1': np.broadcast_to(scalar, (1024,)),
            'w2': np.broadcast_to(scalar, (1024, 1)), 'b2': np.broadcast_to(scalar, (1,)),
        }
        wrong_dtype = dict(self.params)
        wrong_dtype['w0'] = self.params['w0'].astype(np.float64)
        wrong_keys = dict(self.params)
        wrong_keys['extra'] = np.ones(1, dtype=np.float32)
        nonfinite = dict(self.params)
        nonfinite['w0'] = np.full_like(self.params['w0'], np.nan)
        original_array = np.array
        for model in (too_wide, too_many_bytes, wrong_dtype, wrong_keys, nonfinite):
            protected = {id(value) for value in model.values()}
            def tracked_array(value, *args, **kwargs):
                if id(value) in protected:
                    raise AssertionError('Invalid parameter was copied before validation')
                return original_array(value, *args, **kwargs)
            with self.subTest(case=len(protected)), patch.object(np, 'array', side_effect=tracked_array):
                with self.assertRaises(ValueError):
                    fit_pair(model, self.plan[:1], self.pack_whole, device=self.device, check=lambda: None)

    def test_nonfinite_loss_stops_before_either_optimizer_update(self):
        import optax
        initial = {name: np.full_like(value, 4.0) for name, value in self.params.items()}
        def extreme(ids):
            part = self.part(ids)
            part['features'][:] = np.finfo(np.float32).max / 2
            return [part]
        original_chain = optax.chain
        updates = []
        def instrumented_chain(*transforms):
            chain = original_chain(*transforms)
            def instrumented_update(*args, **kwargs):
                updates.append(True)
                return chain.update(*args, **kwargs)
            return optax.GradientTransformation(chain.init, instrumented_update)
        with patch.object(optax, 'chain', side_effect=instrumented_chain):
            with self.assertRaises(FloatingPointError):
                fit_pair(initial, self.plan[:1], extreme, device=self.device, check=lambda: None)
        self.assertEqual(updates, [])

    def test_device_and_cancellation_are_explicit(self):
        for invalid in (None, 'cpu'):
            with self.subTest(device=invalid), self.assertRaises(ValueError):
                fit_pair(self.params, self.plan[:1], self.pack_whole, device=invalid, check=lambda: None)
        class Cancelled(Exception):
            pass
        def cancel():
            raise Cancelled()
        with self.assertRaises(Cancelled):
            fit_pair(self.params, self.plan[:1], self.pack_whole, device=self.device, check=cancel)

    def test_explicit_non_default_cpu_device_is_used_throughout(self):
        import jax
        devices = jax.devices('cpu')
        if len(devices) < 2:
            self.skipTest('Run with two CPU devices for this device-placement regression')
        selected = devices[1]
        result = fit_pair(self.params, self.plan[:1], self.pack_whole, device=selected, check=lambda: None)
        self.assertEqual(result['completed_updates'], 1)
        for name in self.params:
            self.assertEqual(result['control_params'][name].devices(), {selected})
            self.assertEqual(result['experiment_params'][name].devices(), {selected})


if __name__ == '__main__':
    unittest.main()
