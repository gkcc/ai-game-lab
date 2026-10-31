"""Synthetic loss and CPU integration tests. No real data, model or GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from dataclasses import replace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_equiv_bc_v1 import loss, train
from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork, numpy_logits
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash
from learning.task_pair_checkpoint_v1 import snapshot


class LossTests(unittest.TestCase):
    def test_unknown_remains_in_teacher_ce_but_not_auxiliary(self):
        logits = np.array([[2., 1., 0., 5.]], dtype=np.float64)
        candidate = np.ones((1, 4), dtype=bool)
        positive = np.array([[1, 1, 0, 0]], dtype=bool)
        verified = np.array([[1, 1, 1, 0]], dtype=bool)
        labels, weights = np.array([0]), np.array([1.])
        loss.validate_batch(logits, candidate, positive, verified, labels, weights)
        ce, ca, ea, eligible = loss.components(logits, candidate, positive, verified, labels, weights)
        expected = np.log(np.exp(logits[0] - logits[0].max()).sum()) + logits[0].max()
        self.assertAlmostEqual(ce, expected - logits[0, 0])
        self.assertAlmostEqual(ca, np.log(np.exp(logits[0, :3]).sum()) - logits[0, 0])
        self.assertAlmostEqual(ea, np.log(np.exp(logits[0, :3]).sum())
                               - np.log(np.exp(logits[0, :2]).sum()))
        self.assertTrue(eligible[0])
        logits[0, 3] = 50.
        ce2, ca2, ea2, _ = loss.components(logits, candidate, positive, verified, labels, weights)
        self.assertGreater(ce2, ce)
        self.assertAlmostEqual(ca2, ca)
        self.assertAlmostEqual(ea2, ea)

    def test_degenerate_groups_have_zero_auxiliary_and_weighted_denominator(self):
        logits = np.array([[1., 0., 0.], [1., 2., 0.], [1., 2., 0.]], np.float32)
        candidate = np.ones((3, 3), bool)
        positive = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]], bool)
        verified = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]], bool)
        labels, weights = np.zeros(3, np.int32), np.array([2., 3., 4.], np.float32)
        loss.validate_batch(logits, candidate, positive, verified, labels, weights)
        ce, control, experiment, eligible = loss.components(logits, candidate, positive, verified, labels, weights)
        self.assertEqual(eligible.tolist(), [False, False, True])
        third = loss.components(logits[2:], candidate[2:], positive[2:], verified[2:], labels[2:], weights[2:])
        self.assertAlmostEqual(control, third[1] * 4 / 9, places=6)
        self.assertAlmostEqual(experiment, third[2] * 4 / 9, places=6)
        self.assertGreater(ce, 0)

    def test_extreme_logits_stay_stable(self):
        logits = np.array([[10000., -10000., 9999., 9000.]], dtype=np.float32)
        candidates = np.ones((1, 4), bool)
        positive = np.array([[0, 1, 1, 0]], bool)
        verified = np.array([[1, 1, 1, 0]], bool)
        labels, weights = np.array([1], np.int32), np.array([1.], np.float32)
        loss.validate_batch(logits, candidates, positive, verified, labels, weights)
        for arm in ('control', 'experiment'):
            value, _ = loss.objective(logits, candidates, positive, verified, labels, weights, arm=arm)
            self.assertTrue(np.isfinite(value))
            self.assertGreater(value, 19999)

    def test_teacher_must_be_verified_positive_and_p_must_be_subset(self):
        logits = np.zeros((1, 3), np.float32)
        all_candidates = np.ones((1, 3), bool)
        labels, weights = np.array([0], np.int32), np.array([1.], np.float32)
        with self.assertRaises(ValueError):
            loss.validate_batch(logits, all_candidates, np.array([[0, 1, 0]], bool),
                                all_candidates, labels, weights)
        with self.assertRaises(ValueError):
            loss.validate_batch(logits, all_candidates, np.array([[1, 1, 0]], bool),
                                np.array([[1, 0, 0]], bool), labels, weights)

    def test_real_export_gate_rejects_partial_or_inactive_evidence(self):
        kind = 'Synthetic accelerator'
        records = tuple(dict(device_platform='gpu', device_kind=kind,
                             gradient_gap_at_current_arm_parameters=2e-8) for _ in range(32))
        valid = train.PairedResult({}, {}, None, records, 32, True, True, True, True, 'synthetic')
        train._real_export_gate(valid, kind)  # Synthetic evidence only, no actual device is queried.
        cases = (replace(valid, completed_updates=31), replace(valid, control_changed=False),
                 replace(valid, experiment_changed=False), replace(valid, arms_differ=False),
                 replace(valid, records=records[:-1]),
                 replace(valid, records=(dict(records[0], gradient_gap_at_current_arm_parameters=1e-8),
                                         *records[1:])),
                 replace(valid, records=(dict(records[0], gradient_gap_at_current_arm_parameters=float('nan')),
                                         *records[1:])),
                 replace(valid, records=(dict(records[0], device_platform='cpu'), *records[1:])),
                 replace(valid, records=(dict(records[0], device_kind='Different synthetic accelerator'),
                                         *records[1:])))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                train._real_export_gate(case, kind)

    def test_private_device_kind_must_be_valid_and_exact(self):
        train._verify_expected_device('Synthetic accelerator', 'Synthetic accelerator')
        for expected in ('', ' ', '\n', 'accelerator\x7f', 'x' * 129,
                         'accelerator', 'Synthetic', 'different', None):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                train._verify_expected_device('Synthetic accelerator', expected)
        with self.assertRaises(ValueError):
            train._verify_expected_device(None, 'accelerator')


HAS_JAX = importlib.util.find_spec('jax') is not None and importlib.util.find_spec('optax') is not None


@unittest.skipUnless(HAS_JAX, 'JAX and Optax required for the synthetic CPU integration')
class CpuIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import jax
        import jax.numpy as jnp
        cls.jax, cls.jnp = jax, jnp
        cls.cpu = jax.devices('cpu')[0]

    def test_auxiliary_gradient_excludes_unknown_and_differs_between_arms(self):
        jax, jnp = self.jax, self.jnp
        candidates = jnp.array([[True, True, True, True]])
        positive = jnp.array([[True, True, False, False]])
        verified = jnp.array([[True, True, True, False]])
        labels, weights = jnp.array([0]), jnp.array([1.])
        values = jnp.array([[2., 1., 0., 3.]])
        def auxiliary(z, index):
            return loss.components(z, candidates, positive, verified, labels, weights, xp=jnp)[index]
        control = np.asarray(jax.grad(lambda z: auxiliary(z, 1))(values))[0]
        experiment = np.asarray(jax.grad(lambda z: auxiliary(z, 2))(values))[0]
        self.assertEqual(control[3], 0.)
        self.assertEqual(experiment[3], 0.)
        self.assertGreater(control[1], 0.)
        self.assertLess(experiment[1], 0.)
        base = np.asarray(jax.grad(lambda z: loss.components(
            z, candidates, positive, verified, labels, weights, xp=jnp)[0])(values))[0]
        self.assertGreater(base[3], 0.)

    def test_padding_logits_never_affect_either_arm_loss_or_gradient(self):
        jax, jnp = self.jax, self.jnp
        labels, weights = jnp.array([0]), jnp.array([1.])
        valid3 = jnp.array([[True, True, True]])
        valid5 = jnp.array([[True, True, True, False, False]])
        p3, u3 = jnp.array([[True, True, False]]), valid3
        p5 = jnp.array([[True, True, False, False, False]])
        scores3 = jnp.array([[2., 1., 0.]])
        scores5 = jnp.array([[2., 1., 0., 9000., -9000.]])
        for arm in ('control', 'experiment'):
            def obj3(x):
                return loss.objective(x, valid3, p3, u3, labels, weights, arm=arm, xp=jnp)[0]
            def obj5(x):
                return loss.objective(x, valid5, p5, valid5, labels, weights, arm=arm, xp=jnp)[0]
            self.assertAlmostEqual(float(obj3(scores3)), float(obj5(scores5)), places=6)
            g3, g5 = np.asarray(jax.grad(obj3)(scores3)), np.asarray(jax.grad(obj5)(scores5))
            np.testing.assert_allclose(g3, g5[:, :3], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(g5[:, 3:], np.zeros((1, 2)))

    def _synthetic_checkpoint(self):
        shared = [f'shared_{i}' for i in range(275)]
        features = [f'candidate_{i}' for i in range(98)]
        schema = dict(schema_version=SCHEMA_VERSION, schema_sha256=schema_hash(features, shared),
                      shared_feature_names=shared, feature_names=features, ops=list(OPS),
                      dimensions=list(DIMENSIONS))
        rng = np.random.default_rng(3)
        params = {k: v for i, (a, b) in enumerate(zip(DIMENSIONS[:-1], DIMENSIONS[1:]))
                  for k, v in ((f'w{i}', rng.normal(0, .05, (a, b)).astype(np.float32)),
                               (f'b{i}', np.full(b, .01, dtype=np.float32)))}
        return snapshot.Checkpoint(MappingProxyType(params), np.zeros(373), np.ones(373),
                                   json.dumps(schema), schema['schema_sha256'])

    def test_same_batches_cpu_updates_and_existing_tasknetwork_reload(self):
        rng = np.random.default_rng(23)
        def features(size):
            value = rng.normal(size=(size, 373)).astype(np.float32)
            value[:, :275] = value[0, :275]
            return value
        groups = [dict(features=features(4), teacher_index=0,
                       positive_mask=np.array([1, 1, 0, 0], bool),
                       verified_mask=np.array([1, 1, 1, 0], bool), weight=1.),
                  dict(features=features(3), teacher_index=1,
                       positive_mask=np.array([0, 1, 0], bool),
                       verified_mask=np.array([1, 1, 0], bool), weight=1.)]
        batches = np.array([[0, 1], [1, 0]], np.int32)
        checkpoint = self._synthetic_checkpoint()
        count = {'guards': 0}
        def guard():
            count['guards'] += 1
        result = train.fit_paired(groups, checkpoint=checkpoint, batches=batches, device=self.cpu, guard=guard)
        self.assertEqual(result.completed_updates, 2)
        self.assertTrue(result.initial_equal)
        self.assertTrue(result.control_changed and result.experiment_changed and result.arms_differ)
        self.assertGreater(count['guards'], 2)
        self.assertEqual(result.records[0]['eligible_groups'], 1)
        self.assertAlmostEqual(result.records[0]['control_base'], result.records[0]['experiment_base'], places=6)
        self.assertEqual(result.records[0]['device_platform'], 'cpu')
        self.assertGreater(result.records[0]['control_gradient_norm'], 0.)
        self.assertGreater(result.records[0]['experiment_gradient_norm'], 0.)
        self.assertGreater(result.records[0]['control_unscaled_aux_gradient_norm'], 0.)
        self.assertGreater(result.records[0]['experiment_unscaled_aux_gradient_norm'], 0.)
        self.assertGreater(result.records[0]['gradient_gap_at_current_arm_parameters'], 0.)
        archives = train.export_pair(result, checkpoint=checkpoint, registration_commit='a' * 40,
                                     source_kind='synthetic_only')
        self.assertEqual(len(archives), 2)
        with self.assertRaises(ValueError):
            train.export_pair(result, checkpoint=checkpoint, registration_commit='a' * 40,
                              source_kind='validated_counterfactual_requests',
                              expected_device_kind=str(self.cpu.device_kind))
        changed_mean = checkpoint.mean.copy()
        changed_mean[0] = 1.
        wrong_checkpoint = snapshot.Checkpoint(checkpoint.params, changed_mean, checkpoint.scale,
                                               checkpoint.schema_json, checkpoint.bound_schema_sha256)
        with self.assertRaises(ValueError):
            train.export_pair(result, checkpoint=wrong_checkpoint, registration_commit='a' * 40,
                              source_kind='synthetic_only')
        if not hasattr(os, 'memfd_create'):
            self.skipTest('In-memory file descriptor needed to exercise the real path-based loader')
        for raw, parameters in zip(archives, (result.control_params, result.experiment_params)):
            fd = os.memfd_create('synthetic-task-checkpoint')
            try:
                with os.fdopen(os.dup(fd), 'wb') as stream:
                    stream.write(raw)
                model = TaskNetwork('/proc/self/fd/' + str(fd), hashlib.sha256(raw).hexdigest())
                scores = model.score(groups[0]['features'][0, :275], groups[0]['features'][:, 275:])
                self.assertTrue(np.isfinite(scores).all())
                expected = numpy_logits({key: np.asarray(value) for key, value in parameters.items()},
                                        groups[0]['features'])
                np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)
                self.assertEqual(int(np.argmax(scores)), int(np.argmax(expected)))
            finally:
                os.close(fd)

    def test_no_eligible_multi_positive_groups_gives_identical_updates(self):
        rng = np.random.default_rng(8)
        features = rng.normal(size=(3, 373)).astype(np.float32)
        features[:, :275] = features[0, :275]
        group = dict(features=features, teacher_index=0, positive_mask=np.array([1, 0, 0], bool),
                     verified_mask=np.array([1, 1, 0], bool), weight=1.)
        result = train.fit_paired([group], checkpoint=self._synthetic_checkpoint(),
                                  batches=np.array([[0], [0]], np.int32), device=self.cpu, guard=lambda: None)
        self.assertFalse(result.arms_differ)
        for step in result.records:
            self.assertEqual(step['eligible_groups'], 0)
            self.assertEqual(step['control_aux'], 0.)
            self.assertEqual(step['experiment_aux'], 0.)
            self.assertEqual(step['gradient_gap_at_current_arm_parameters'], 0.)
        for key in train.PARAMS:
            np.testing.assert_array_equal(np.asarray(result.control_params[key]),
                                          np.asarray(result.experiment_params[key]))


if __name__ == '__main__':
    unittest.main()
