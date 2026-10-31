"""Synthetic WATER-residual inference and real selector-interface checks."""
from copy import deepcopy
import unittest

import numpy as np

from learning.task_bc_v1.model import DIMENSIONS
from learning.task_choice_v1.features import OPS, candidate_id
from learning.task_teacher_support_probe_v1.test_alignment import make_stream
from . import policy


def metadata_and_inputs():
    _obs, _cfg, records, schema = make_stream()
    metadata = dict(schema, ops=list(OPS), dimensions=list(DIMENSIONS))
    return metadata, deepcopy(records[0]['inputs'])


def raw_candidates(metadata, ops):
    raw = np.zeros((len(ops), policy.DIM), dtype=np.float32)
    names = metadata['feature_names']
    for index, op in enumerate(ops):
        raw[index, policy.SHARED_DIM + names.index('task.op.' + op)] = 1
    return raw


class FakeNetwork:
    def __init__(self, metadata, logits=None):
        self.metadata = deepcopy(metadata)
        self.mean = np.zeros(policy.DIM, dtype=np.float64)
        self.scale = np.ones(policy.DIM, dtype=np.float64)
        self.logits, self.calls = logits, 0

    def score(self, shared, candidates):
        self.calls += 1
        if self.logits is None:
            return np.arange(len(candidates), 0, -1, dtype=np.float64)
        return np.asarray(self.logits, dtype=np.float64).copy()

    def __call__(self, inputs):
        scores = self.score(inputs['shared_x'], [candidate['x'] for candidate in inputs['candidates']])
        return inputs['candidates'][int(np.argmax(scores))]['id']


class WaterResidualTests(unittest.TestCase):
    def setUp(self):
        self.metadata, self.inputs = metadata_and_inputs()
        self.mean = np.zeros(policy.DIM, dtype=np.float64)
        self.scale = np.ones(policy.DIM, dtype=np.float64)
        self.weights = np.zeros(policy.DIM, dtype=np.float32)
        self.marker = policy.SHARED_DIM + self.metadata['feature_names'].index('unit.index')

    def choose(self, raw, logits):
        return policy.select(raw, np.asarray(logits, dtype=np.float64), self.mean, self.scale,
                             self.weights, self.metadata)

    def test_non_water_baseline_choice_never_flips(self):
        raw = raw_candidates(self.metadata, ('WATER', 'DIG', 'WATER'))
        raw[:, self.marker] = (100, 0, 200)
        self.weights[self.marker] = 100
        result = self.choose(raw, (2, 3, 1))
        self.assertEqual(result, 1)
        self.assertIs(type(result), int)

    def test_water_reranks_only_water_even_if_non_water_would_win(self):
        raw = raw_candidates(self.metadata, ('WATER', 'PLANT', 'WATER'))
        raw[:, self.marker] = (0, 1000, 2)
        self.weights[self.marker] = 2
        result = self.choose(raw, (10, 9, 8))
        self.assertEqual(result, 2)
        # If WATER tied a later non-WATER globally and falls after correction,
        # the explicitly registered router still stays within WATER.
        raw = raw_candidates(self.metadata, ('WATER', 'DIG', 'WATER'))
        raw[:, self.marker] = (2, 0, 3)
        self.weights[self.marker] = -10
        self.assertEqual(self.choose(raw, (10, 10, 9)), 0)

    def test_negative_weight_and_first_tie_follow_original_order(self):
        raw = raw_candidates(self.metadata, ('WATER', 'WATER', 'WATER'))
        raw[:, self.marker] = (2, 1, 0)
        self.weights[self.marker] = -1
        self.assertEqual(self.choose(raw, (4, 3, 1)), 0)
        self.assertEqual(self.choose(raw, (4, 3, 3)), 2)
        # A non-WATER candidate preceding a tied WATER is the base argmax.
        raw = raw_candidates(self.metadata, ('DIG', 'WATER'))
        self.assertEqual(self.choose(raw, (2, 2)), 0)

    def test_zero_weights_preserve_float64_argmax_and_extreme_values(self):
        raw = raw_candidates(self.metadata, ('WATER', 'WATER'))
        larger = np.nextafter(np.float64(1), np.float64(2))
        result = self.choose(raw, (1, larger))
        self.assertEqual(result, 1)
        self.mean[self.marker] = 1e300
        self.assertEqual(self.choose(raw, (1, larger)), 1)
        self.weights[self.marker] = 1
        with self.assertRaises(policy.ResidualError):
            self.choose(raw, (1, larger))

    def test_float64_baseline_and_float32_residual_are_both_preserved(self):
        raw = raw_candidates(self.metadata, ('WATER', 'WATER'))
        larger = np.nextafter(np.float64(1), np.float64(2))
        # A nonzero weight but identical raw inputs leaves logits unchanged;
        # casting baseline scores to float32 would incorrectly choose index 0.
        self.weights[self.marker] = np.float32(1e-30)
        self.assertEqual(self.choose(raw, (1, larger)), 1)
        raw[:, self.marker] = np.array([0, 3], dtype=np.float32)
        self.mean[self.marker], self.scale[self.marker] = 1.0, 2.0
        self.weights[self.marker] = 1
        self.assertEqual(self.choose(raw, (1, 0)), 1)

    def test_invalid_schema_onehot_dtypes_shapes_and_nonfinite_rejected(self):
        raw = raw_candidates(self.metadata, ('WATER', 'DIG'))
        scores = np.array([1, 2], dtype=np.float64)
        variants = []
        bad = raw.copy(); bad[0, self._op_col('WATER')] = 0.5; variants.append((bad, scores, self.mean, self.scale, self.weights, self.metadata))
        bad = raw.copy(); bad[0, self._op_col('DIG')] = 1; variants.append((bad, scores, self.mean, self.scale, self.weights, self.metadata))
        bad = raw.copy(); bad[0, 0] = np.nan; variants.append((bad, scores, self.mean, self.scale, self.weights, self.metadata))
        bad = raw.copy(); bad[1, 0] = 1; variants.append((bad, scores, self.mean, self.scale, self.weights, self.metadata))
        variants.append((raw.astype(np.float64), scores, self.mean, self.scale, self.weights, self.metadata))
        variants.append((raw, scores.astype(np.float32), self.mean, self.scale, self.weights, self.metadata))
        variants.append((raw, scores, self.mean, self.scale, self.weights.astype(np.float64), self.metadata))
        bad = self.weights.copy(); bad[0] = np.inf; variants.append((raw, scores, self.mean, self.scale, bad, self.metadata))
        bad = self.mean.copy(); bad[0] = np.nan; variants.append((raw, scores, bad, self.scale, self.weights, self.metadata))
        bad = self.scale.copy(); bad[0] = 0.5; variants.append((raw, scores, self.mean, bad, self.weights, self.metadata))
        bad = deepcopy(self.metadata); bad['schema_sha256'] = 'not-authenticated'; variants.append((raw, scores, self.mean, self.scale, self.weights, bad))
        variants.append((raw[:0], scores[:0], self.mean, self.scale, self.weights, self.metadata))
        for index, args in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(policy.ResidualError):
                policy.select(*args)

    def _op_col(self, op):
        return policy.SHARED_DIM + self.metadata['feature_names'].index('task.op.' + op)

    def test_selector_scores_original_float64_values_and_preserves_inputs(self):
        network = FakeNetwork(self.metadata, logits=[1, np.nextafter(1., 2.)])
        self.inputs['shared_x'][0] = 1.0 + 2**-40
        received = {}
        def score(shared, local):
            received['shared'] = shared.copy()
            return np.array(network.logits, dtype=np.float64)
        network.score = score
        selector = policy.WaterResidualSelector(network, self.weights, self.metadata)
        original = deepcopy(self.inputs)
        chosen = selector(self.inputs)
        self.assertEqual(chosen, self.inputs['candidates'][1]['id'])
        self.assertEqual(received['shared'][0], self.inputs['shared_x'][0])
        self.assertNotEqual(received['shared'][0], float(np.float32(self.inputs['shared_x'][0])))
        self.assertEqual(self.inputs, original)
        self.assertEqual(selector.last, 1)

    def test_selector_can_reorder_water_without_masking_other_ops(self):
        for candidate in self.inputs['candidates']:
            candidate['op'] = 'WATER'
            candidate['id'] = candidate_id(candidate['unit'], candidate['target'], candidate['op'])
            for op in OPS:
                candidate['x'][self.metadata['feature_names'].index('task.op.' + op)] = float(op == 'WATER')
        network = FakeNetwork(self.metadata, logits=[3, 2])
        self.weights[self.marker] = 2
        selector = policy.WaterResidualSelector(network, self.weights, self.metadata)
        self.weights[self.marker] = 0  # Constructor retained an immutable copy.
        self.assertEqual(selector(self.inputs), self.inputs['candidates'][1]['id'])
        self.assertEqual(selector.last, 1)
        self.assertEqual(network.calls, 1)

    def test_selector_rejects_forged_schema_id_features_and_model_changes(self):
        network = FakeNetwork(self.metadata, logits=[1, 2])
        selector = policy.WaterResidualSelector(network, self.weights, self.metadata)
        variants = []
        bad = deepcopy(self.inputs); bad['schema_sha256'] = 'wrong'; variants.append(bad)
        bad = deepcopy(self.inputs); bad['candidates'][0]['id'] = 'forged'; variants.append(bad)
        bad = deepcopy(self.inputs); bad['candidates'][0]['x'][-1] = '1.0'; variants.append(bad)
        bad = deepcopy(self.inputs); bad['candidates'][0]['x'][self.metadata['feature_names'].index('task.op.WATER')] = 0.5; variants.append(bad)
        bad = deepcopy(self.inputs); bad['candidates'][0]['x'][0] += 1; variants.append(bad)
        for bad in variants:
            with self.subTest(variant='synthetic'), self.assertRaises(policy.ResidualError):
                selector(bad)
            self.assertIsNone(selector.last)
        selector(self.inputs)
        network.mean[self.marker] = 2
        with self.assertRaises(policy.ResidualError):
            selector(self.inputs)
        self.assertIsNone(selector.last)

    def test_zero_residual_runs_through_actual_taskagent_selector_interface(self):
        from learning.task_choice_v1.agent import TaskAgent
        from learning.task_teacher_support_probe_v1.test_terminal import sample
        obs, cfg = sample()
        baseline = TaskAgent(task_selector=FakeNetwork(self.metadata), crop_mode='teacher')
        model = FakeNetwork(self.metadata)
        student = TaskAgent(task_selector=policy.WaterResidualSelector(model, self.weights, self.metadata),
                            crop_mode='teacher')
        self.assertEqual(student(obs, cfg), baseline(obs, cfg))
        self.assertGreater(model.calls, 0)


if __name__ == '__main__':
    unittest.main()
