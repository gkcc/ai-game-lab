"""Focused regressions: bounded intervention and deterministic reward fit."""
from copy import deepcopy
import json
import unittest

import numpy as np

from learning.task_reward_gate_v1.policy import GateModel, GateSelector, gate_features


class FakeNetwork:
    """A deterministic scorer double; no checkpoint or game is opened."""
    def __init__(self):
        self.calls = 0
        self.last = None

    def __call__(self, inputs):
        self.calls += 1
        logits = [c['test_logit'] for c in inputs['candidates']]
        chosen = inputs['candidates'][int(np.argmax(logits))]['id']
        self.last = dict(logits=logits, chosen=chosen)
        return chosen


def sample(step, player=0, logits=(4., 3., 2.), ops=('WATER', 'HARVEST', 'PLANT')):
    return dict(observation=dict(step=step, player=player), candidates=[
        dict(id=f'candidate-{i}', op=ops[i], test_logit=score)
        for i, score in enumerate(logits)])


def constant_model(value):
    return GateModel([0., 0.], [1., 1.], [0.] * 12, value)


class GateSelectorTests(unittest.TestCase):
    def test_canonical_ties_single_forward_and_suffix_is_old_student(self):
        network = FakeNetwork()
        gate = GateSelector(network, 'intervention', 0)
        gate.begin_turn(80, 0)
        inputs = sample(80, logits=(4., 4., 4.))
        self.assertEqual(gate(inputs), 'candidate-1')
        self.assertEqual(network.last['chosen'], 'candidate-0')
        self.assertEqual(gate(inputs), 'candidate-0')
        self.assertEqual(network.calls, 2)
        self.assertEqual(gate.counts['task_forwards'], 2)
        self.assertEqual(gate.counts['task_rows'], 6)
        self.assertEqual(len(gate.events), 1)
        self.assertEqual(gate.events[0]['gap'], 0.)
        gate.begin_turn(81, 0)
        self.assertEqual(gate(sample(81)), 'candidate-0')
        self.assertEqual(gate.events, [])

    def test_only_first_callback_and_first_eligible_turn(self):
        gate = GateSelector(FakeNetwork(), 'intervention', 0)
        gate.begin_turn(80, 0)
        self.assertEqual(gate(sample(80, logits=(2.,), ops=('WATER',))), 'candidate-0')
        self.assertEqual(gate(sample(80)), 'candidate-0')
        self.assertEqual(gate.events, [])
        gate.begin_turn(81, 0)
        self.assertEqual(gate(sample(81)), 'candidate-1')
        self.assertEqual(gate.events[0]['step'], 81)

    def test_nonpositive_gate_consumes_window_and_zero_is_baseline(self):
        for value in (0., -1.):
            gate = GateSelector(FakeNetwork(), 'gated', model=constant_model(value))
            for step in (79, 80, 81, 119, 120, 260, 261, 440, 479, 480):
                gate.begin_turn(step, 0)
                self.assertEqual(gate(sample(step)), 'candidate-0')
                self.assertEqual(gate(sample(step)), 'candidate-0')
            self.assertEqual(gate.counts['gate_evaluations'], 3)

    def test_positive_gate_once_per_window_and_reset_for_new_game_seat(self):
        gate = GateSelector(FakeNetwork(), 'gated', model=constant_model(1.))
        for step in (80, 260, 440):
            gate.begin_turn(step, 0)
            self.assertEqual(gate(sample(step)), 'candidate-1')
        self.assertEqual(gate.counts['gate_evaluations'], 3)
        with self.assertRaises(ValueError):
            gate.begin_turn(441, 1)
        with self.assertRaises(ValueError):
            gate.begin_turn(80, 0)
        gate.begin_turn(0, 1)
        self.assertEqual(gate.counts['task_forwards'], 0)
        gate.begin_turn(80, 1)
        self.assertEqual(gate(sample(80, player=1)), 'candidate-1')
        self.assertEqual(gate.counts['gate_evaluations'], 1)

    def test_assigned_window_only_and_baseline_records_without_override(self):
        gate = GateSelector(FakeNetwork(), 'baseline', 1)
        for step in (80, 260, 440):
            gate.begin_turn(step, 0)
            self.assertEqual(gate(sample(step)), 'candidate-0')
            self.assertEqual(len(gate.events), int(step == 260))
        self.assertEqual(gate.counts['gate_evaluations'], 0)

    def test_invalid_context_and_nonfinite_fail_closed(self):
        gate = GateSelector(FakeNetwork(), 'baseline')
        with self.assertRaises(ValueError):
            gate(sample(80))
        gate.begin_turn(80, 0)
        with self.assertRaises(ValueError):
            gate(sample(81))
        with self.assertRaises(ValueError):
            gate(sample(80, logits=(float('nan'), 2.)))
        self.assertEqual(gate.events, [])

    def test_budget_prevents_forward_and_counts_actual_input_rows(self):
        network = FakeNetwork()
        gate = GateSelector(network, limits=dict(task_forwards=2, crop_forwards=2,
                                                 neural_input_rows=4, gate_evaluations=3))
        gate.begin_turn(0, 0)
        gate.count_crop_forward()
        gate(sample(0))
        self.assertEqual(gate.counts['task_rows'] + gate.counts['crop_rows'], 4)
        with self.assertRaises(RuntimeError):
            gate(sample(0))
        self.assertEqual(network.calls, 1)
        with self.assertRaises(RuntimeError):
            gate.count_crop_forward()

    def test_reward_inputs_ignore_teacher_labels_and_private_extras(self):
        left = GateSelector(FakeNetwork(), 'intervention', 0)
        right = GateSelector(FakeNetwork(), 'intervention', 0)
        left.begin_turn(80, 0)
        right.begin_turn(80, 0)
        original = sample(80)
        extra = deepcopy(original)
        extra['teacher_label'] = 'not-a-candidate'
        extra['teacher_score'] = 1e99
        extra['observation']['private_opponent'] = {'money': 1e99}
        self.assertEqual(left(original), right(extra))
        self.assertEqual(left.events, right.events)
        self.assertEqual(len(left.events[0]['x12']), 12)

    def test_turn_limits_reset_without_resetting_game_limits(self):
        network = FakeNetwork()
        gate = GateSelector(network, limits=dict(max_task_forwards_per_turn=1,
                                                 max_crop_forwards_per_turn=1))
        gate.begin_turn(0, 0)
        gate(sample(0))
        gate.count_crop_forward()
        with self.assertRaises(RuntimeError):
            gate(sample(0))
        with self.assertRaises(RuntimeError):
            gate.count_crop_forward()
        self.assertEqual(network.calls, 1)
        gate.begin_turn(1, 0)
        gate(sample(1))
        gate.count_crop_forward()
        self.assertEqual(gate.counts['task_forwards'], 2)
        self.assertEqual(gate.counts['crop_forwards'], 2)
        self.assertEqual(gate.turn_counts['task_forwards'], 1)


class GateModelTests(unittest.TestCase):
    def test_ridge_constant_target_leaves_unpenalized_intercept(self):
        x = np.asarray([gate_features(80+i, {'op': 'WATER'}, {'op': 'HARVEST'}, float(i))
                        for i in range(8)])
        model = GateModel.fit(x, np.full(8, 3.))
        np.testing.assert_allclose(model.predict(x), 3., atol=1e-12)
        np.testing.assert_allclose(model.coefficients, 0., atol=1e-12)
        self.assertAlmostEqual(model.intercept, 3.)

    def test_fit_and_json_reload_predictions_and_normalization(self):
        x = np.asarray([gate_features(80+i*10, {'op': 'WATER'}, {'op': 'HARVEST'}, float(i))
                        for i in range(8)])
        target = np.asarray([-3., -2., -1., 0., 1., 2., 3., 4.])
        model = GateModel.fit(x, target)
        restored = GateModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
        np.testing.assert_array_equal(model.predict(x), restored.predict(x))
        np.testing.assert_array_equal(model.mean, x[:, :2].mean(axis=0))
        np.testing.assert_array_equal(model.scale, np.maximum(x[:, :2].std(axis=0), 1.))
        # Feature repetition cannot change a mean-MSE ridge fit.
        repeated = GateModel.fit(np.repeat(x, 2, axis=0), np.repeat(target, 2))
        np.testing.assert_allclose(model.predict(x), repeated.predict(x), atol=1e-12)
        self.assertEqual(model.to_dict()['ops'], ['DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER'])

    def test_invalid_model_fit_prediction_and_unknown_schema_fail(self):
        with self.assertRaises(ValueError):
            GateModel.fit(np.zeros((0, 12)), [])
        with self.assertRaises(ValueError):
            GateModel.fit(np.zeros((2, 12)), [0., float('inf')])
        with self.assertRaises(ValueError):
            constant_model(0.).predict([float('nan')] * 12)
        model = constant_model(0.).to_dict()
        model['ops'] = list(reversed(model['ops']))
        with self.assertRaises(ValueError):
            GateModel.from_dict(model)
        with self.assertRaises(ValueError):
            GateModel([0., 0.], [1., .5], [0.] * 12, 0.)


if __name__ == '__main__':
    unittest.main()
