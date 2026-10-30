"""Fixed 2x8 synthetic CPU composition; never reads real data or checkpoints."""

import importlib.util
import os
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest.mock import patch


os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from learning.task_pair_data_preflight_v1.test_core import make_arrays, record
from learning.task_pair_orchestration_v1 import core


def synthetic_data(label_fn=lambda i: i % 2, cluster_fn=lambda i: i % 8 + 1):
    arrays = make_arrays([record(i, label_fn(i), cluster_fn(i)) for i in range(24)])
    for value in arrays.values():
        value.setflags(write=False)
    return MappingProxyType(arrays)


def synthetic_parameters():
    return {
        'w0': (np.sin(np.arange(373 * 4, dtype=np.float32) * np.float32(0.03))
               .reshape(373, 4) * np.float32(0.1)),
        'b0': np.asarray([0.4, 0.3, 0.2, 0.5], np.float32),
        'w1': np.asarray([[0.2, -0.1], [0.3, 0.2], [-0.1, 0.4], [0.1, 0.3]], np.float32),
        'b1': np.asarray([0.2, 0.1], np.float32),
        'w2': np.asarray([[0.5], [-0.3]], np.float32),
        'b2': np.zeros(1, np.float32),
    }


@unittest.skipUnless(importlib.util.find_spec('jax') is not None
                     and importlib.util.find_spec('optax') is not None, 'JAX and Optax are required')
class PairedCompositionTests(unittest.TestCase):
    def setUp(self):
        import jax
        self.jax = jax
        self.cpu = jax.devices('cpu')[0]
        self.context = jax.default_device(self.cpu)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.data = synthetic_data()
        self.params = synthetic_parameters()
        self.mean = np.zeros(373, np.float64)
        self.scale = np.ones(373, np.float64)
        self.weights = np.ones(5, np.float32)

    def train(self, *, data=None, params=None, device=None, check=lambda: None, **overrides):
        kwargs = {'device': self.cpu if device is None else device, 'rng_seed': 23,
                  'updates': 2, 'batch_size': 8, 'check': check}
        kwargs.update(overrides)
        return core.train_pair(self.data if data is None else data,
                               self.params if params is None else params,
                               self.mean, self.scale, self.weights, **kwargs)

    def test_real_synthetic_pair_is_repeatable_without_exposing_plan_or_seed(self):
        prior = {key: value.copy() for key, value in self.params.items()}
        with patch.object(core.packing, 'pack_training_batch',
                          wraps=core.packing.pack_training_batch) as pack:
            first = self.train()
        self.assertEqual(pack.call_count, 2)
        second = self.train()
        self.assertEqual(repr(first), '<PrivatePairedUpdateResult>')
        self.assertEqual(first.completed_updates, 2)
        self.assertGreaterEqual(first.auxiliary_nonzero_updates, 1)
        self.assertTrue(first.both_arms_changed and first.arms_differ)
        self.assertEqual(set(vars(first)), {'control_params', 'experiment_params', 'completed_updates',
                                            'auxiliary_nonzero_updates', 'both_arms_changed', 'arms_differ'})
        for key in prior:
            np.testing.assert_array_equal(self.params[key], prior[key])
            np.testing.assert_array_equal(np.asarray(first.control_params[key]),
                                          np.asarray(second.control_params[key]))
            np.testing.assert_array_equal(np.asarray(first.experiment_params[key]),
                                          np.asarray(second.experiment_params[key]))
        with self.assertRaises(TypeError):
            first.control_params['w0'] = None
        self.assertTrue(all(not value.flags.writeable for value in self.data.values()))

    def test_empty_eligibility_or_single_cluster_fails_before_fit(self):
        examples = (synthetic_data(label_fn=lambda _: 2), synthetic_data(cluster_fn=lambda _: 1))
        for data in examples:
            with self.subTest(kind=len(np.unique(data['seeds']))), \
                 patch.object(core.fit_core, 'fit_pair') as fit, self.assertRaises(ValueError):
                self.train(data=data)
            fit.assert_not_called()

    def test_bad_budget_writable_mapping_and_missing_device_are_rejected(self):
        for overrides in ({'updates': 0}, {'updates': True}, {'batch_size': 257},
                          {'rng_seed': -1}, {'rng_seed': True}, {'device': object()}):
            with self.subTest(keys=tuple(overrides)), self.assertRaises(ValueError):
                self.train(**overrides)
        mutable = dict(self.data)
        with self.assertRaises(ValueError):
            self.train(data=mutable)
        mutable['labels'] = mutable['labels'].copy()
        with self.assertRaises(ValueError):
            self.train(data=MappingProxyType(mutable))

    def test_cancel_before_source_processing_or_update(self):
        def cancel():
            raise TimeoutError('synthetic cancellation')
        with patch.object(core.targets, 'qualifying_groups') as qualify, \
             patch.object(core.fit_core, 'fit_pair') as fit, self.assertRaises(TimeoutError):
            self.train(check=cancel)
        qualify.assert_not_called()
        fit.assert_not_called()

    def test_first_callback_cannot_change_frozen_initial_state(self):
        baseline = self.train()
        calls = []
        def mutate():
            if not calls:
                self.mean[0] = 9
                self.scale[0] = 0
                self.weights[0] = -1
                self.params['w0'][0, 0] += 10
            calls.append(True)
        after = self.train(check=mutate)
        self.assertTrue(calls)
        self.assertEqual(self.mean[0], 9)
        self.assertEqual(self.scale[0], 0)
        self.assertEqual(self.weights[0], -1)
        for name in self.params:
            np.testing.assert_array_equal(np.asarray(after.control_params[name]),
                                          np.asarray(baseline.control_params[name]))
            np.testing.assert_array_equal(np.asarray(after.experiment_params[name]),
                                          np.asarray(baseline.experiment_params[name]))

    def test_mutation_between_batches_cannot_change_frozen_state(self):
        baseline = self.train()
        pack = core.packing.pack_training_batch
        observed = []
        def mutate_after_first(source, ids, mean, scale, weights, eligible):
            observed.append((mean.copy(), scale.copy(), weights.copy()))
            self.assertFalse(mean.flags.writeable or scale.flags.writeable or weights.flags.writeable)
            parts = pack(source, ids, mean, scale, weights, eligible)
            if len(observed) == 1:
                self.mean[0], self.scale[0], self.weights[0] = 10, 2, 3
                self.params['w0'][0, 0] += 10
            return parts
        with patch.object(core.packing, 'pack_training_batch', side_effect=mutate_after_first):
            result = self.train()
        self.assertEqual(len(observed), 2)
        for before, after in zip(observed[0], observed[1]):
            np.testing.assert_array_equal(before, after)
        for name in self.params:
            np.testing.assert_array_equal(np.asarray(result.control_params[name]),
                                          np.asarray(baseline.control_params[name]))
            np.testing.assert_array_equal(np.asarray(result.experiment_params[name]),
                                          np.asarray(baseline.experiment_params[name]))

    def test_oversized_initial_model_is_rejected_before_any_array_copy(self):
        too_large = np.broadcast_to(np.zeros(1, np.float32), (100000, 100000))
        bad = dict(self.params, w0=too_large)
        with patch.object(core.np, 'array', side_effect=AssertionError('unexpected copy')) as copy, \
             self.assertRaises(ValueError):
            self.train(params=bad)
        copy.assert_not_called()

    def test_invalid_normalization_is_rejected_before_batch_generation(self):
        for name, value in (('mean', np.full(373, np.nan)),
                            ('scale', np.zeros(373)), ('weights', np.zeros(5))):
            original = getattr(self, name)
            setattr(self, name, value)
            try:
                with self.subTest(name=name), patch.object(core.batching, 'make_batches') as batches, \
                     self.assertRaises(ValueError):
                    self.train()
                batches.assert_not_called()
            finally:
                setattr(self, name, original)

    def test_bounded_float_sequences_keep_existing_vector_semantics(self):
        for name, value in (('mean', self.mean), ('scale', self.scale), ('weight', self.weights)):
            snapshot = core._vector_snapshot(value.tolist(), len(value), name)
            np.testing.assert_array_equal(snapshot, value)
            self.assertFalse(snapshot.flags.writeable)

    def test_zero_auxiliary_evidence_and_partial_budget_are_not_accepted(self):
        def fake(params, batches, pack, *, device, check):
            del params, check
            for ids in batches:
                pack(ids)
            step = {'aux_gradient_nonzero': False, 'control_changed': True,
                    'experiment_changed': True, 'arms_differ': True}
            return {'control_params': {}, 'experiment_params': {}, 'steps': [dict(step), dict(step)],
                    'completed_updates': 2, 'aux_nonzero_updates': 0,
                    'control_changed_from_initial': True, 'experiment_changed_from_initial': True,
                    'arms_differ': True, 'device_platform': device.platform}
        with patch.object(core.fit_core, 'fit_pair', side_effect=fake), self.assertRaises(ValueError):
            self.train()

        def partial(params, batches, pack, *, device, check):
            value = fake(params, batches, pack, device=device, check=check)
            value['steps'][0]['aux_gradient_nonzero'] = True
            value['aux_nonzero_updates'] = 1
            value['completed_updates'] = 1
            return value
        with patch.object(core.fit_core, 'fit_pair', side_effect=partial), self.assertRaises(ValueError):
            self.train()


if __name__ == '__main__':
    unittest.main()
