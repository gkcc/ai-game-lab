"""Synthetic aggregate and JAX CPU parity regressions only."""
import math
import os
import unittest

import numpy as np

from learning.task_multisource_rank_v1 import core as rank
from learning.task_water_residual_v1.test_policy import metadata_and_inputs
from . import core, evaluate
from .test_core import candidates, interaction_params


def group(target=1, *, baseline=None, shared_value=-2, source='new'):
    shared = np.zeros(core.SHARED_DIM, np.float32)
    shared[0] = shared_value
    return rank.Group(shared, candidates(),
                      np.zeros(2, np.float64) if baseline is None else np.asarray(baseline, np.float64),
                      target, source)


def close_tie_params():
    ws = np.zeros((core.SHARED_DIM, core.HIDDEN), np.float32)
    wc = np.zeros((core.LOCAL_DIM, core.HIDDEN), np.float32)
    b, v = np.zeros(core.HIDDEN, np.float32), np.zeros(core.HIDDEN, np.float32)
    wc[0, 0] = 100
    v[0] = np.float32(1e-4)
    return core.Params(ws, wc, b, v)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.metadata, _unused = metadata_and_inputs()
        self.mean, self.scale = np.zeros(core.DIM, np.float64), np.ones(core.DIM, np.float64)

    def test_aggregate_target_changes_and_nll_by_role_and_operation(self):
        params = interaction_params()
        first, second = group(1, source='old'), group(0, source='new')
        result = evaluate.assess([first, second], self.mean, self.scale, params, self.metadata,
                                 roles=['T01', 'A07'], role_labels=('T01', 'A07', 'A08'))
        overall = result['overall']
        self.assertEqual({key: overall[key] for key in ('groups','baseline_correct','residual_correct',
                         'changed','op_flips','improved','regressed')},
                         dict(groups=2, baseline_correct=1, residual_correct=1,
                              changed=2, op_flips=2, improved=1, regressed=1))
        self.assertEqual(result['by_target_operation']['WATER']['improved'], 1)
        self.assertEqual(result['by_target_operation']['DIG']['regressed'], 1)
        self.assertEqual(result['by_target_operation']['PLANT']['coverage'], 'unknown')
        self.assertIsNone(result['by_target_operation']['PLANT']['baseline_nll'])
        self.assertEqual(result['by_role']['A08']['overall']['coverage'], 'unknown')
        self.assertFalse(result['independent_evaluation'])
        x = candidates()
        h = np.tanh(first.shared @ params.ws + x @ params.wc + params.b)
        scores = (h @ params.v).astype(np.float64)
        maximum = float(scores.max())
        normalizer = sum(math.exp(float(z) - maximum) for z in scores)
        expected = (maximum - float(scores[1])) + math.log(normalizer)
        self.assertAlmostEqual(result['by_target_operation']['WATER']['residual_nll'], expected, places=7)

    def test_zero_residual_and_extreme_nll(self):
        params = core.initial_params(2)
        example = group(1, baseline=(1000., -1000.))
        result = evaluate.assess([example], self.mean, self.scale, params, self.metadata)
        self.assertEqual(result['overall']['baseline_nll'], 2000.)
        self.assertEqual(result['overall']['residual_nll'], 2000.)
        self.assertEqual(result['overall']['changed'], 0)
        self.assertEqual(result['by_role'], {})
        good = group(0, baseline=(1e308, -1e308))
        self.assertEqual(evaluate.assess([good], self.mean, self.scale, params, self.metadata)
                         ['overall']['baseline_nll'], 0.)
        with self.assertRaises(ValueError):
            evaluate.assess([group(1, baseline=(1e308, -1e308))], self.mean, self.scale, params, self.metadata)

    def test_invalid_inputs_and_guard_do_not_return_aggregates(self):
        params = interaction_params()
        value = group()
        bad = rank.Group(value.shared, value.candidates, np.array([np.nan, 0.], np.float64), 1, 'new')
        with self.assertRaises(ValueError):
            evaluate.assess([bad], self.mean, self.scale, params, self.metadata)
        with self.assertRaises(ValueError):
            evaluate.assess([value], self.mean, self.scale, params, self.metadata,
                            roles=['BAD'], role_labels=['A07'])
        signal = MemoryError('synthetic guard')
        def stop(): raise signal
        with self.assertRaises(MemoryError) as caught:
            evaluate.assess([value], self.mean, self.scale, params, self.metadata, guard=stop)
        self.assertIs(caught.exception, signal)

    def test_invalid_training_width_is_rejected_before_device_use(self):
        for width in (0, 257, True, 2.0):
            with self.subTest(width=width), self.assertRaises(evaluate.EvaluationError):
                evaluate.parity([group()], self.mean, self.scale, core.initial_params(1),
                                device=None, guard=lambda: None, train_width=width)


@unittest.skipUnless(os.environ.get('STATE_RESIDUAL_JAX_CPU') == '1',
                     'Explicit bounded synthetic JAX CPU test was not requested')
class CpuParityTests(unittest.TestCase):
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

    def check(self, groups, params, device=None, train_width=2):
        return evaluate.parity(groups, np.zeros(core.DIM, np.float64), np.ones(core.DIM, np.float64),
                               params, device=self.jax.devices('cpu')[0] if device is None else device,
                               guard=self.guard, train_width=train_width)

    def test_zero_residual_and_nonzero_scores_match_on_cpu(self):
        centered = group(0, baseline=(1e8, 1e8 - 4))
        value = self.check([centered], core.initial_params(7))
        self.assertEqual(value['groups'], 1)
        self.assertTrue(value['allclose'])
        self.assertEqual(value['baseline_argmax_mismatches'], 0)
        self.assertEqual(value['training_choice_mismatches'], 0)
        self.assertEqual((value['fixed_batch'], value['fixed_width'], value['device']), (16, 256, 'cpu'))
        self.assertEqual(value['groups_by_width'], {'2': 1, '256': 1})
        value = self.check([group(0), group(1, shared_value=2)], interaction_params())
        self.assertEqual(value['groups'], 2)

    def test_train_width_and_full_width_coverage_with_mixed_candidates(self):
        first = group(0, baseline=(0.5, 0.1))
        one = rank.Group(first.shared, first.candidates[:1], np.array([0.5], np.float64), 0, 'old')
        three = rank.Group(first.shared, np.concatenate((first.candidates, first.candidates[:1])),
                           np.array([0.5, 0.1, -0.2], np.float64), 0, 'new')
        value = self.check([one, first, three], core.initial_params(3), train_width=2)
        self.assertEqual(value['groups'], 3)
        self.assertEqual(value['train_width'], 2)
        self.assertEqual(value['train_width_groups'], 2)
        self.assertEqual(value['full_width_groups'], 3)
        self.assertEqual(value['groups_by_width'], {'2': 2, '256': 3})
        empty = self.check([first], core.initial_params(3), train_width=1)
        self.assertEqual(empty['groups_by_width'], {'1': 0, '256': 1})
        same = self.check([one, first, three], core.initial_params(3), train_width=256)
        self.assertEqual(same['groups_by_width'], {'256': 3})
        self.assertEqual((same['train_width_groups'], same['full_width_groups']), (3, 3))

    def test_centered_float32_baseline_choice_change_is_independently_rejected(self):
        # Centering still cannot preserve an f64 gap below the f32 subnormal range.
        value = group(1, baseline=(-1e-50, 0.))
        self.assertEqual(int(np.argmax(value.baseline)), 1)
        with self.assertRaisesRegex(evaluate.EvaluationError, 'Centered baseline'):
            self.check([value], core.initial_params(1))
        with self.assertRaisesRegex(evaluate.EvaluationError, 'Centered baseline'):
            self.check([group(0, baseline=(1e308, -1e308))], core.initial_params(1))

    def test_training_float32_addition_mismatch_is_independently_rejected(self):
        params = close_tie_params()
        # The centered baseline argmax remains candidate zero; the float32
        # sum rounds the two training scores to a tie while float64 deployment
        # retains a positive candidate-one gap after adding the same residual.
        offset = float(params.v[0])
        value = group(1, baseline=(0., -offset + 1e-12), shared_value=0)
        _f64, f32 = core._training_baseline(value.baseline)
        self.assertEqual(int(np.argmax(value.baseline)), int(np.argmax(f32)))
        self.assertEqual(core.select_values(value.shared, value.candidates, value.baseline,
                                            np.zeros(core.DIM, np.float64), np.ones(core.DIM, np.float64), params), 1)
        for width in (2, 256):
            with self.subTest(width=width), self.assertRaisesRegex(
                    evaluate.EvaluationError, 'training and float64 deployment'):
                self.check([value], params, train_width=width)

    def test_wrong_device_and_empty_groups_are_rejected(self):
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.parity([group()], np.zeros(core.DIM, np.float64), np.ones(core.DIM, np.float64),
                            core.initial_params(1), device='not-a-device', guard=self.guard, train_width=2)
        with self.assertRaises(evaluate.EvaluationError): self.check([], core.initial_params(1))


if __name__ == '__main__':
    unittest.main()
