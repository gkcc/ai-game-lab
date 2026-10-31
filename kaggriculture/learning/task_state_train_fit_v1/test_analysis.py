"""Synthetic-only training diagnostic regressions."""
from copy import deepcopy
import json
import unittest

import numpy as np

from learning.task_multisource_rank_v1 import core as rank
from learning.task_state_residual_v1 import core as state
from learning.task_state_residual_v1.test_core import candidates, interaction_params
from learning.task_water_residual_v1.test_policy import metadata_and_inputs
from . import analysis


def group(*, target=1, baseline=(0., 0.), shared_value=-2, local=None, source='new'):
    shared = np.zeros(rank.SHARED_DIM, np.float32)
    shared[0] = shared_value
    return rank.Group(shared, candidates() if local is None else local,
                      np.asarray(baseline, np.float64), target, source)


def non_teacher_change():
    local = np.zeros((3, rank.LOCAL_DIM), np.float32)
    for i, op in enumerate(('PLANT', 'DIG', 'WATER')):
        local[i, rank.OP_OFFSET + rank.OPS.index(op)] = 1
    local[2, 0] = 1
    ws = np.zeros((rank.SHARED_DIM, state.HIDDEN), np.float32)
    wc = np.zeros((rank.LOCAL_DIM, state.HIDDEN), np.float32)
    bias, output = np.zeros(state.HIDDEN, np.float32), np.zeros(state.HIDDEN, np.float32)
    wc[0, 0], output[0] = 1, 1
    return group(target=0, baseline=(0., 1., 0.9), shared_value=0, local=local), state.Params(ws, wc, bias, output)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.mean, self.scale = np.zeros(rank.DIM, np.float64), np.ones(rank.DIM, np.float64)
        self.metadata, _unused = metadata_and_inputs()
        self.roles = ['A03'] * 21 + ['A04'] * 21

    def assess(self, item, student, params):
        return analysis.assess([item] * 42, [student] * 42, self.roles, mean=self.mean,
                               scale=self.scale, params=params, metadata=self.metadata, guard=lambda: None)

    def test_zero_residual_preserves_scores_and_empty_buckets_are_unknown(self):
        item = group(target=0, baseline=(0., 1.))
        result = self.assess(item, 1, state.initial_params(7))
        whole = result['overall']
        self.assertEqual((whole['count'], whole['baseline_correct'], whole['candidate_correct'],
                          whole['changed'], whole['op_flips']), (42, 0, 0, 0, 0))
        self.assertEqual((whole['margin_increased'], whole['margin_decreased'], whole['margin_within_tolerance']),
                         (0, 0, 42))
        self.assertEqual((whole['mean_base_margin'], whole['mean_candidate_margin']), (-1., -1.))
        self.assertEqual(whole['mean_base_nll'], whole['mean_candidate_nll'])
        self.assertEqual(result['by_role']['A03']['count'], 21)
        empty = result['by_target_operation']['WATER']
        self.assertEqual((empty['count'], empty['coverage']), (0, 'unknown'))
        self.assertTrue(all(empty[key] is None for key in analysis.MEAN_FIELDS))
        self.assertTrue(analysis.validate(result))
        encoded = json.dumps(result, allow_nan=False)
        for forbidden in ('student_index', 'target_index', 'features', 'logits', 'params', 'seed'):
            self.assertNotIn(forbidden, encoded)

    def test_actual_corrected_target_and_strict_positive_margin(self):
        result = self.assess(group(), 0, interaction_params())
        whole = result['overall']
        self.assertEqual((whole['baseline_correct'], whole['candidate_correct'], whole['changed'], whole['op_flips']),
                         (0, 42, 42, 42))
        self.assertEqual((whole['margin_increased'], whole['margin_decreased'], whole['margin_within_tolerance']),
                         (42, 0, 0))
        self.assertEqual(whole['mean_base_margin'], 0.)  # first-argmax loses an exact tie
        self.assertGreater(whole['mean_candidate_margin'], 0.)
        self.assertEqual(result['by_target_operation']['WATER']['candidate_correct'], 42)

    def test_improved_margin_without_changing_first_choice(self):
        result = self.assess(group(baseline=(1., 0.)), 0, interaction_params())
        whole = result['overall']
        self.assertEqual((whole['candidate_correct'], whole['changed'], whole['op_flips']), (0, 0, 0))
        self.assertEqual((whole['margin_increased'], whole['margin_decreased']), (42, 0))
        self.assertGreater(whole['mean_candidate_margin'], whole['mean_base_margin'])

    def test_wrong_to_wrong_choice_and_operation_flip_are_not_corrections(self):
        item, params = non_teacher_change()
        result = self.assess(item, 1, params)
        whole = result['overall']
        self.assertEqual((whole['candidate_correct'], whole['changed'], whole['op_flips']), (0, 42, 42))
        self.assertEqual((whole['margin_increased'], whole['margin_decreased']), (0, 42))
        self.assertLess(whole['mean_candidate_margin'], whole['mean_base_margin'])

    def test_identical_candidates_follow_actual_first_argmax(self):
        local = np.repeat(candidates()[:1], 2, axis=0)
        result = self.assess(group(local=local), 0, interaction_params())
        whole = result['overall']
        self.assertEqual((whole['candidate_correct'], whole['changed'], whole['op_flips']), (0, 0, 0))
        self.assertEqual((whole['mean_base_margin'], whole['mean_candidate_margin']), (0., 0.))
        self.assertEqual(whole['margin_within_tolerance'], 42)

    def test_subtolerance_real_residual_is_not_reported_as_margin_growth(self):
        ws = np.zeros((rank.SHARED_DIM, state.HIDDEN), np.float32)
        wc = np.zeros((rank.LOCAL_DIM, state.HIDDEN), np.float32)
        bias, output = np.zeros(state.HIDDEN, np.float32), np.zeros(state.HIDDEN, np.float32)
        wc[0, 0], output[0] = 1, np.float32(1e-7)
        result = self.assess(group(baseline=(1., 0.), shared_value=0), 0,
                             state.Params(ws, wc, bias, output))['overall']
        delta = result['mean_candidate_margin'] - result['mean_base_margin']
        self.assertGreater(delta, 0.)
        self.assertLess(delta, analysis.MARGIN_TOLERANCE)
        self.assertNotEqual(result['mean_base_nll'], result['mean_candidate_nll'])
        self.assertEqual((result['candidate_correct'], result['changed']), (0, 0))
        self.assertEqual((result['margin_increased'], result['margin_decreased'],
                          result['margin_within_tolerance']), (0, 0, 42))

    def test_margin_tolerance_boundary_is_inclusive(self):
        tol = analysis.MARGIN_TOLERANCE
        self.assertEqual(analysis._margin_category(0., tol), 'margin_within_tolerance')
        self.assertEqual(analysis._margin_category(0., -tol), 'margin_within_tolerance')
        self.assertEqual(analysis._margin_category(0., float(np.nextafter(tol, np.inf))), 'margin_increased')
        self.assertEqual(analysis._margin_category(0., float(np.nextafter(-tol, -np.inf))), 'margin_decreased')

    def test_large_scores_are_stable_or_rejected_if_unrepresentable(self):
        zero = self.assess(group(baseline=(1000., -1000.)), 0, state.initial_params(2))
        self.assertEqual(zero['overall']['mean_base_nll'], 2000.)
        self.assertEqual(zero['overall']['mean_base_margin'], -2000.)
        offset = self.assess(group(baseline=(1e20, 1e20)), 0, interaction_params())
        self.assertEqual(offset['overall']['candidate_correct'], 42)
        self.assertEqual(offset['overall']['mean_base_margin'], 0.)
        with self.assertRaises(ValueError):
            self.assess(group(baseline=(1e308, -1e308)), 0, state.initial_params(2))

    def test_mismatched_student_roles_shape_and_nonfinite_rejected(self):
        params = state.initial_params(2)
        value = group()
        for students, roles in (([0] * 41, self.roles), ([True] * 42, self.roles),
                                ([1] * 42, self.roles), ([0] * 42, ['A09'] * 42),
                                (np.array(0), self.roles)):
            with self.subTest(students=type(students).__name__), self.assertRaises(ValueError):
                analysis.assess([value] * 42, students, roles, mean=self.mean, scale=self.scale,
                                params=params, metadata=self.metadata, guard=lambda: None)
        with self.assertRaises(ValueError):
            self.assess(group(target=0), 0, params)
        with self.assertRaises(ValueError):
            self.assess(group(source='old'), 0, params)
        with self.assertRaises(ValueError):
            self.assess(group(baseline=(np.inf, 0.)), 0, params)
        bad = group(); bad.candidates[0, rank.OP_OFFSET:rank.OP_OFFSET + len(rank.OPS)] = 0
        with self.assertRaises(ValueError): self.assess(bad, 0, params)
        invalid = state.Params(params.ws, params.wc, params.b, np.full(state.HIDDEN, np.nan, np.float32))
        with self.assertRaises(ValueError): self.assess(value, 0, invalid)

    def test_guard_propagates_and_aggregate_tampering_is_rejected(self):
        signal = MemoryError('synthetic guard')
        def stop(): raise signal
        with self.assertRaises(MemoryError) as caught:
            analysis.assess([group()] * 42, [0] * 42, self.roles, mean=self.mean, scale=self.scale,
                            params=state.initial_params(2), metadata=self.metadata, guard=stop)
        self.assertIs(caught.exception, signal)
        result = self.assess(group(), 0, interaction_params())
        bad = deepcopy(result)
        bad['by_role']['A03']['candidate_correct'] = 0
        with self.assertRaises(analysis.AnalysisError): analysis.validate(bad)
        bad = deepcopy(result)
        bad['overall']['op_flips'] = 43
        with self.assertRaises(analysis.AnalysisError): analysis.validate(bad)


if __name__ == '__main__':
    unittest.main()
