"""Synthetic selection and reporting checks; no model file or GPU access."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import assessment, policy
from .test_policy import FakeNetwork, metadata_and_inputs, raw_candidates


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.metadata, _ = metadata_and_inputs()
        self.network = FakeNetwork(self.metadata, logits=[2, 1])
        self.unit = policy.SHARED_DIM + self.metadata['feature_names'].index('unit.index')
        self.weights = np.zeros(policy.DIM, dtype=np.float32)
        self.weights[self.unit] = 2

    def group(self, role, *, ops=('WATER', 'WATER'), label=1, opponent='market', side=0):
        raw = raw_candidates(self.metadata, ops)
        raw[:, self.unit] = np.arange(len(ops))
        return SimpleNamespace(role=role, opponent=opponent, side=side, features=raw,
                               target_index=label, op=ops[label])

    def test_training_requires_water_target_and_original_water_route(self):
        water = self.group('A01')
        mixed = self.group('A02', ops=('DIG', 'WATER'))
        selected, report = assessment.prepare_training([water, mixed], self.network,
                                                        self.metadata, guard=lambda: None)
        self.assertEqual(len(selected), 1)
        self.assertEqual(report['selected_by_role'], {'A01': 1, 'A02': 0})
        self.assertEqual(selected[0]['teacher_index'], 1)
        self.assertEqual(selected[0]['role'], 0)

    def test_screen_uses_whole_role_counts_and_conditional_nll(self):
        groups = [self.group('A07', side=0), self.group('A07', side=1),
                  self.group('A08', opponent='round2', side=0),
                  self.group('A08', opponent='round2', side=1)]
        result, rows = assessment.assess_development(groups, self.network, self.weights,
                                                      self.metadata, guard=lambda: None)
        self.assertEqual((result['eligible_groups'], result['improved'], result['regressed']), (4, 4, 0))
        self.assertEqual(result['screen'], 'passed')
        self.assertEqual(result['operation_flips'], 0)
        self.assertTrue(all(row['residual_nll'] < row['baseline_nll'] for row in rows))
        self.assertEqual(result['opponents']['market']['eligible_groups'], 2)
        self.assertEqual(result['sides']['1']['eligible_groups'], 2)

    def test_no_development_role_is_unknown_and_nonwater_route_is_unchanged(self):
        groups = [self.group('A07', ops=('DIG', 'WATER'), label=1)]
        result, rows = assessment.assess_development(groups, self.network, self.weights,
                                                      self.metadata, guard=lambda: None)
        self.assertEqual(result['screen'], 'unknown')
        self.assertEqual(result['eligible_groups'], 0)
        self.assertIsNone(result['checks']['each_role_no_accuracy_drop'])
        self.assertIsNone(result['checks']['equal_role_nll_no_increase'])
        self.assertIsNone(result['checks']['two_net_improvements'])
        self.assertFalse(rows[0]['changed'])
        self.assertFalse(rows[0]['operation_flip'])

    def test_bad_schema_label_and_roles_fail_closed(self):
        bad = self.group('A01'); bad.op = 'PLANT'
        with self.assertRaises(ValueError):
            assessment.prepare_training([bad], self.network, self.metadata, guard=lambda: None)
        with self.assertRaises(ValueError):
            assessment.assess_development([self.group('A01')], self.network, self.weights,
                                         self.metadata, guard=lambda: None)

    def test_guard_interrupts_before_scoring(self):
        def stop(): raise MemoryError('synthetic')
        with self.assertRaises(MemoryError):
            assessment.prepare_training([self.group('A01')], self.network, self.metadata, guard=stop)
        self.assertEqual(self.network.calls, 0)

    def test_nll_is_stable_for_large_common_offset(self):
        scores = np.array([1e16, 1e16], dtype=np.float64)
        self.assertAlmostEqual(assessment._nll(scores, 0), np.log(2.0))
        self.assertAlmostEqual(assessment._nll(scores - 1e16, 0), np.log(2.0))

    def test_known_hard_failure_precedes_missing_role_unknown(self):
        group = self.group('A07', ops=('DIG', 'WATER'), label=1)
        with patch.object(policy, 'select', return_value=1):
            result, _ = assessment.assess_development([group], self.network, self.weights,
                                                       self.metadata, guard=lambda: None)
        self.assertEqual(result['operation_flips'], 1)
        self.assertEqual(result['screen'], 'failed')
        # An observed accuracy drop is also already a known failure even if a
        # second role has no eligible data.
        group = self.group('A07', label=0)
        result, _ = assessment.assess_development([group], self.network, self.weights,
                                                  self.metadata, guard=lambda: None)
        self.assertEqual(result['regressed'], 1)
        self.assertEqual(result['screen'], 'failed')


if __name__ == '__main__':
    unittest.main()
