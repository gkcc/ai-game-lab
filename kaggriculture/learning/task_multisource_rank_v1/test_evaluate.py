"""Synthetic-only regressions for aggregate offline ranker evaluation."""
from dataclasses import dataclass, replace
import importlib
import json
import math
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_choice_v1.features import OPS
from learning.task_water_residual_v1.test_policy import metadata_and_inputs
from . import evaluate as e


@dataclass(frozen=True)
class SyntheticGroup:
    shared: np.ndarray
    candidates: np.ndarray
    baseline: np.ndarray
    target: int
    source: str


def fake_core():
    core = ModuleType('learning.task_multisource_rank_v1.core')
    core.Group = SyntheticGroup
    def select(group, mean, scale, weights):
        raw = np.concatenate((np.broadcast_to(group.shared, (len(group.candidates), 275)),
                              group.candidates), axis=1)
        if np.any(weights):
            norm = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
            scores = group.baseline + (norm @ weights).astype(np.float64)
        else:
            scores = group.baseline
        return int(np.argmax(scores))
    core.select = select
    return core


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.metadata, _unused = metadata_and_inputs()
        self.mean, self.scale = np.zeros(373, np.float64), np.ones(373, np.float64)
        self.weights = np.zeros(373, np.float32)
        self.marker = self.metadata['feature_names'].index('unit.index')
        self.weights[275 + self.marker] = 2
        self.core = fake_core()
        self.core_patch = patch.dict(sys.modules, {self.core.__name__: self.core})
        self.core_patch.start()
        self.addCleanup(self.core_patch.stop)

    def group(self, *, ops=('DIG', 'WATER'), logits=(2., 1.), target=1, source='old'):
        shared = np.zeros(275, dtype=np.float32)
        candidates = np.zeros((len(ops), 98), dtype=np.float32)
        for i, op in enumerate(ops):
            candidates[i, self.marker] = i
            candidates[i, self.metadata['feature_names'].index('task.op.' + op)] = 1
        return SyntheticGroup(shared, candidates, np.asarray(logits, dtype=np.float64), target, source)

    def assess(self, groups, **kwargs):
        return e.assess(groups, self.mean, self.scale, self.weights, self.metadata, **kwargs)

    def test_cross_operation_improvement_regression_and_conditional_nll(self):
        first = self.group(target=1, source='old')
        second = self.group(target=0, source='new')
        result = self.assess([first, second], roles=['T01', 'A07'], role_labels=('T01', 'A07', 'A08'))
        overall = result['overall']
        self.assertEqual({key: overall[key] for key in ('groups', 'baseline_correct', 'residual_correct',
                         'changed', 'op_flips', 'improved', 'regressed')},
                         dict(groups=2, baseline_correct=1, residual_correct=1, changed=2,
                              op_flips=2, improved=1, regressed=1))
        water, dig = result['by_target_operation']['WATER'], result['by_target_operation']['DIG']
        self.assertEqual((water['groups'], water['improved'], water['op_flips']), (1, 1, 1))
        self.assertEqual((dig['groups'], dig['regressed'], dig['op_flips']), (1, 1, 1))
        self.assertAlmostEqual(water['baseline_nll'], math.log1p(math.exp(1)))
        self.assertAlmostEqual(water['residual_nll'], math.log1p(math.exp(-1)))
        self.assertAlmostEqual(dig['baseline_nll'], math.log1p(math.exp(-1)))
        self.assertAlmostEqual(dig['residual_nll'], math.log1p(math.exp(1)))
        self.assertEqual(result['by_source']['old']['groups'], 1)
        self.assertEqual(result['by_source']['new']['groups'], 1)
        self.assertEqual(result['by_role']['T01']['overall']['improved'], 1)
        self.assertEqual(result['by_role']['A07']['overall']['regressed'], 1)
        self.assertEqual(result['by_role']['A08']['overall']['coverage'], 'unknown')
        self.assertIsNone(result['by_role']['A08']['overall']['baseline_nll'])
        self.assertFalse(result['independent_evaluation'])

    def test_zero_residual_and_missing_categories_remain_unknown(self):
        self.weights[:] = 0
        original = self.group(target=0)
        before = (original.shared.copy(), original.candidates.copy(), original.baseline.copy())
        wrapped = SimpleNamespace(group=original, role='T01')
        result = self.assess([wrapped], role_labels=('T01', 'T02'))
        self.assertEqual((result['overall']['changed'], result['overall']['op_flips'],
                          result['overall']['improved'], result['overall']['regressed']), (0, 0, 0, 0))
        self.assertEqual(result['overall']['baseline_nll'], result['overall']['residual_nll'])
        self.assertEqual(result['by_target_operation']['PLANT']['coverage'], 'unknown')
        self.assertIsNone(result['by_target_operation']['PLANT']['residual_nll'])
        self.assertEqual(result['by_role']['T02']['overall']['coverage'], 'unknown')
        for actual, expected in zip((original.shared, original.candidates, original.baseline), before):
            np.testing.assert_array_equal(actual, expected)

    def test_extreme_logits_are_stable_or_rejected_if_loss_unrepresentable(self):
        self.weights[:] = 0
        result = self.assess([self.group(logits=(1000., -1000.), target=1)])
        self.assertEqual(result['overall']['baseline_nll'], 2000.)
        result = self.assess([self.group(logits=(1e308, -1e308), target=0)])
        self.assertEqual(result['overall']['baseline_nll'], 0.)
        with self.assertRaises(e.EvaluationError):
            self.assess([self.group(logits=(1e308, -1e308), target=1)])

    def test_aggregates_never_include_features_scores_indices_or_seed(self):
        result = self.assess([self.group()], roles=['T01'], role_labels=('T01',))
        encoded = json.dumps(result, allow_nan=False)
        for forbidden in ('features', 'scores', 'logits', 'target_index', 'seed', 'PRIVATE_MARKER'):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(set(result), {'overall', 'by_target_operation', 'by_source', 'by_role',
                                       'independent_evaluation'})

    def test_bad_shape_dtype_labels_onehot_and_nonfinite_fail(self):
        original = self.group()
        bad_onehot = original.candidates.copy()
        bad_onehot[0, self.metadata['feature_names'].index('task.op.WATER')] = 1
        variants = [replace(original, shared=np.zeros(274, np.float32)),
                    replace(original, candidates=original.candidates.astype(np.float64)),
                    replace(original, candidates=np.zeros((0, 98), np.float32), baseline=np.zeros(0, np.float64)),
                    replace(original, candidates=np.zeros((257, 98), np.float32), baseline=np.zeros(257, np.float64)),
                    replace(original, baseline=original.baseline.astype(np.float32)),
                    replace(original, baseline=np.array([np.nan, 1], np.float64)),
                    replace(original, target=True), replace(original, target=2),
                    replace(original, source='PRIVATE_MARKER'), replace(original, candidates=bad_onehot)]
        for index, bad in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.assess([bad])
        bad_weights = self.weights.copy(); bad_weights[0] = np.nan
        with self.assertRaises(ValueError):
            e.assess([original], self.mean, self.scale, bad_weights, self.metadata)
        bad_metadata = dict(self.metadata, schema_sha256='invalid')
        with self.assertRaises(ValueError):
            e.assess([original], self.mean, self.scale, self.weights, bad_metadata)

    def test_role_contract_and_core_disagreement_fail_closed(self):
        group = self.group()
        for roles, vocabulary in ((['T01'], ()), (['PRIVATE'], ('T01',)), ([], ('T01',)),
                                  (['T01'], ('T01', 'T01')), (['T01'], ('PRIVATE',))):
            with self.subTest(vocabulary=vocabulary), self.assertRaises(e.EvaluationError):
                self.assess([group], roles=roles, role_labels=vocabulary)
        self.core.select = lambda *_args: 0
        with self.assertRaises(e.EvaluationError):
            self.assess([group])

    def test_guard_exception_is_not_reclassified(self):
        signal = MemoryError('synthetic')
        def guard(): raise signal
        with self.assertRaises(MemoryError) as caught:
            self.assess([self.group()], guard=guard)
        self.assertIs(caught.exception, signal)

    def test_empty_input_does_not_claim_coverage(self):
        result = self.assess([], roles=[], role_labels=('T01', 'A07'))
        self.assertEqual(result['overall']['coverage'], 'unknown')
        self.assertTrue(all(value['coverage'] == 'unknown' for value in result['by_target_operation'].values()))
        self.assertTrue(all(value['overall']['coverage'] == 'unknown' for value in result['by_role'].values()))
        self.assertFalse(result['independent_evaluation'])


class ActualCoreTests(unittest.TestCase):
    def test_real_core_selector_matches_aggregate_on_synthetic_group(self):
        # Import here, rather than at module load, so the evaluator can still
        # be exercised with a fake while core is independently developed.
        from . import core
        metadata, _unused = metadata_and_inputs()
        candidates = np.zeros((2, 98), np.float32)
        for index, op in enumerate(('DIG', 'WATER')):
            candidates[index, metadata['feature_names'].index('task.op.' + op)] = 1
            candidates[index, metadata['feature_names'].index('unit.index')] = index
        group = core.Group(np.zeros(275, np.float32), candidates,
                           np.array([2, 1], np.float64), 1, 'old')
        weights = np.zeros(373, np.float32)
        weights[275 + metadata['feature_names'].index('unit.index')] = 2
        result = e.assess([group], np.zeros(373, np.float64), np.ones(373, np.float64),
                          weights, metadata, roles=['T01'], role_labels=['T01'])
        self.assertEqual((result['overall']['improved'], result['overall']['op_flips']), (1, 1))
        self.assertEqual(result['by_role']['T01']['by_target_operation']['WATER']['groups'], 1)


if __name__ == '__main__':
    unittest.main()
