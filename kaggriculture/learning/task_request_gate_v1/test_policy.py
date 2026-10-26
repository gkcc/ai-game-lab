"""Synthetic regressions for whole-request selection and its shared budget."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_request_gate_v1 import policy


STUDENT = dict(farmer=['WATER'], hands=[['PASS']],
               market=[['BUY_SEED', 'WHEAT', 1], ['SELL', 'WHEAT', 2]])
TEACHER = dict(farmer=['HARVEST'], hands=[['WATER']],
               market=[['SELL', 'WHEAT', 3], ['BUY_SEED', 'CARROT', 2]])


def observation(step, player=0):
    farm = dict(money=100, tiles=[[None]], farmer=[0, 0], hands=[],
                unlocked_quadrants=[0], hires_today=0)
    return dict(step=step, player=player, day=1, hour=0, farms=[deepcopy(farm), deepcopy(farm)],
                private=dict(shed={}, seeds={}, inventories=[{}]),
                market=dict(inventory={}, prices={}, params={}), town=dict(unlocked_shops=[]))


def record(identifier, op, distance, alternatives=()):
    candidate = dict(id=identifier, op=op, x=[float(distance), 17.])
    candidates = [candidate] + [dict(id=name, op=kind, x=[float(value), 99.])
                                for name, kind, value in alternatives]
    return dict(chosen=identifier, op=op, teacher_label='unused-label',
                inputs=dict(feature_names=['task.distance', 'unused'], candidates=candidates))


def constant_model(value):
    return policy.RequestModel([0.] * 8, [1.] * 8, [0.] * 8, value)


class FakeTaskNetwork:
    constructed = 0
    def __init__(self, path, sha):
        type(self).constructed += 1
        self.calls, self.last = 0, None

    def __call__(self, inputs):
        self.calls += 1
        chosen = inputs['candidates'][0]['id']
        self.last = dict(chosen=chosen, logits=[0.] * len(inputs['candidates']))
        return chosen


class FakeCropNetwork:
    def __init__(self):
        self.calls = 0

    def logits(self, x, names):
        self.calls += 1
        return [0.] * 5


def controllers(teacher_action=None, student_action=None, teacher_records=None, student_records=None):
    teacher_action = teacher_action or (lambda _obs: TEACHER)
    student_action = student_action or (lambda _obs: STUDENT)
    teacher_records = teacher_records or (lambda _obs: [record('teacher', 'HARVEST', 2)])
    student_records = student_records or (lambda _obs: [record('student', 'WATER', 4)])

    class FakeController:
        def __init__(self, selector, crop_mode, crop_path, crop_sha):
            if crop_mode != 'model':
                raise AssertionError('Both controllers must retain the frozen crop model')
            self.selector = selector
            self.crop = SimpleNamespace(network=FakeCropNetwork(), records=[], last=None)
            self.records, self.last, self.calls = [], None, 0

        def __call__(self, obs, cfg):
            self.calls += 1
            student = self.selector is not None
            records = deepcopy((student_records if student else teacher_records)(obs))
            for row in records:
                if student:
                    row['chosen'] = self.selector(row['inputs'])
                self.crop.network.logits([1.], ['synthetic'])
            self.last = dict(step=obs['step'], records=records, crop_records=[],
                             fallbacks=0, load_errors=0, inference_errors=0,
                             checkpoint_sha256='crop-sha', visible_observation=deepcopy(obs))
            return deepcopy((student_action if student else teacher_action)(obs))
    return FakeController


class RequestAgentTests(unittest.TestCase):
    def make_agent(self, mode='baseline', window=None, model=None, limits=None, **fake_options):
        with patch.object(policy, 'TaskNetwork', FakeTaskNetwork), patch.object(policy, 'TaskAgent', controllers(**fake_options)):
            return policy.RequestAgent('unused-crop', 'crop-sha', 'unused-task', 'task-sha',
                                       mode, window, model, limits=limits)

    def test_replace_one_whole_request_then_restore_student(self):
        agent = self.make_agent('intervention', 0)
        result = agent(observation(80), {})
        self.assertEqual(result, TEACHER)
        self.assertEqual(agent.last['proposals'], dict(student=STUDENT, teacher=TEACHER))
        self.assertEqual(agent.last['selected_trace'], agent.last['teacher_trace'])
        self.assertTrue(agent.last['request_events'][0]['changed_request'])
        self.assertEqual(agent.last['request_events'][0]['x8'], [80/719, 1., 0., 0., 1., 0., -1., -2.])
        self.assertEqual(agent.last['request_counts']['task_forwards'], 1)
        self.assertEqual(agent.last['request_counts']['crop_forwards'], 2)
        self.assertEqual(agent.last['request_counts']['controller_calls'], 2)
        self.assertEqual(agent.last['request_counts']['teacher_proposals'], 1)
        result['market'].reverse()
        self.assertEqual(agent.last['proposals']['teacher'], TEACHER)
        self.assertEqual(agent(observation(81), {}), STUDENT)
        self.assertEqual(agent.last['request_events'], [])
        self.assertIsNone(agent.last['proposals'])
        self.assertEqual(agent.teacher.calls, 1)

    def test_first_different_turn_not_first_equal_turn(self):
        agent = self.make_agent('intervention', 0,
                                teacher_action=lambda obs: STUDENT if obs['step'] < 82 else TEACHER)
        for step in (80, 81):
            self.assertEqual(agent(observation(step), {}), STUDENT)
            self.assertEqual(agent.last['request_events'], [])
            self.assertIsNotNone(agent.last['teacher_trace'])
        self.assertEqual(agent(observation(82), {}), TEACHER)
        self.assertEqual(agent.last['request_events'][0]['step'], 82)
        self.assertEqual(agent.last['request_counts']['teacher_proposals'], 3)
        self.assertEqual(agent(observation(83), {}), STUDENT)

    def test_equal_requests_never_consume_and_window_ends_after_40_proposals(self):
        agent = self.make_agent('intervention', 0, teacher_action=lambda _obs: STUDENT)
        for step in range(80, 120):
            self.assertEqual(agent(observation(step), {}), STUDENT)
            self.assertEqual(agent.last['request_events'], [])
        self.assertEqual(agent(observation(120), {}), STUDENT)
        self.assertEqual(agent.last['request_counts']['teacher_proposals'], 40)
        self.assertIsNone(agent.last['teacher_trace'])

    def test_nonpositive_model_consumes_all_three_windows_without_switch(self):
        for value in (0., -1.):
            agent = self.make_agent('gated', model=constant_model(value))
            for step in (80, 81, 260, 261, 440, 441):
                self.assertEqual(agent(observation(step), {}), STUDENT)
                if step in (80, 260, 440):
                    self.assertFalse(agent.last['request_events'][0]['selected_teacher'])
                    self.assertFalse(agent.last['request_events'][0]['changed_request'])
                else:
                    self.assertEqual(agent.last['request_events'], [])
            self.assertEqual(agent.last['request_counts']['gate_evaluations'], 3)
            self.assertEqual(agent.teacher.calls, 3)

    def test_positive_model_and_cross_game_reset(self):
        agent = self.make_agent('gated', model=constant_model(1.))
        self.assertEqual(agent(observation(80), {}), TEACHER)
        with self.assertRaises(ValueError):
            agent(observation(81, 1), {})
        with self.assertRaises(ValueError):
            agent(observation(79), {})
        agent(observation(0, 1), {})
        self.assertEqual(agent.last['request_counts']['controller_calls'], 1)
        self.assertEqual(agent.last['request_counts']['teacher_proposals'], 0)
        self.assertEqual(agent(observation(80, 1), {}), TEACHER)
        self.assertEqual(agent.last['request_counts']['gate_evaluations'], 1)

    def test_pure_student_and_full_teacher_construct_only_needed_models(self):
        FakeTaskNetwork.constructed = 0
        baseline = self.make_agent()
        self.assertIsNone(baseline.teacher)
        self.assertEqual(baseline(observation(80), {}), STUDENT)
        self.assertEqual(baseline.last['request_counts']['controller_calls'], 1)
        self.assertEqual(baseline.last['request_events'], [])
        self.assertEqual(FakeTaskNetwork.constructed, 1)
        teacher = self.make_agent('teacher_task')
        self.assertIsNone(teacher.student)
        self.assertEqual(teacher(observation(80), {}), TEACHER)
        self.assertEqual(FakeTaskNetwork.constructed, 1)
        self.assertEqual(teacher.last['request_counts']['task_forwards'], 0)
        self.assertEqual(teacher.last['request_counts']['teacher_proposals'], 0)
        self.assertEqual(teacher.last['request_counts']['controller_calls'], 1)

    def test_assigned_baseline_records_opportunity_without_switching(self):
        agent = self.make_agent('baseline', 1)
        self.assertEqual(agent(observation(80), {}), STUDENT)
        self.assertIsNone(agent.last['teacher_trace'])
        self.assertEqual(agent(observation(260), {}), STUDENT)
        self.assertFalse(agent.last['request_events'][0]['changed_request'])
        self.assertEqual(agent.last['request_events'][0]['window'], 1)
        self.assertEqual(agent.last['request_counts']['gate_evaluations'], 0)

    def test_shared_budget_blocks_teacher_crop_before_forward(self):
        agent = self.make_agent('intervention', 0, limits=dict(crop_forwards=1))
        with self.assertRaises(RuntimeError):
            agent(observation(80), {})
        self.assertEqual(agent.student.crop.network.calls, 1)
        self.assertEqual(agent.teacher.crop.network.calls, 0)
        self.assertEqual(agent.budget.counts['crop_forwards'], 1)
        self.assertEqual(agent.budget.counts['task_forwards'], 1)

    def test_shared_rows_and_per_turn_caps_fail_before_excess_work(self):
        for limits in (dict(neural_input_rows=2), dict(max_crop_forwards_per_turn=1)):
            agent = self.make_agent('intervention', 0, limits=limits)
            with self.assertRaises(RuntimeError):
                agent(observation(80), {})
            self.assertEqual(agent.teacher.crop.network.calls, 0)
        agent = self.make_agent(limits=dict(max_task_forwards_per_turn=1))
        agent(observation(0), {})
        agent(observation(1), {})
        self.assertEqual(agent.last['request_counts']['task_forwards'], 2)
        self.assertEqual(agent.last['turn_request_counts']['task_forwards'], 1)

    def test_visibility_filter_excludes_opponent_private_extras(self):
        agent = self.make_agent('intervention', 0)
        obs = observation(80)
        obs['farms'][1]['private'] = {'seeds': {'MELON': 999999}}
        obs['future_market'] = [999999]
        agent(obs, {})
        for key in ('student_trace', 'teacher_trace'):
            visible = agent.last[key]['visible_observation']
            self.assertNotIn('private', visible['farms'][1])
            self.assertNotIn('future_market', visible)


class FeaturesAndEqualityTests(unittest.TestCase):
    def test_aggregate_own_chosen_ids_without_aligning_divergent_callbacks(self):
        student = dict(step=80, records=[record('s0', 'WATER', 5, [('unused', 'HARVEST', 100)]),
                                         record('s1', 'DIG', 1)])
        teacher = dict(step=80, records=[record('t0', 'HARVEST', 2)])
        expected = [80/719, 2., -1., 0., 1., 0., -1., -4.]
        self.assertEqual(policy.request_features(80, student, teacher), expected)
        student['records'].reverse()
        student['records'][0]['inputs']['candidates'].reverse()
        teacher['records'][0]['teacher_label'] = 'arbitrary'
        teacher['teacher_priority'] = 1e99
        self.assertEqual(policy.request_features(80, student, teacher), expected)
        teacher['records'][0]['chosen'] = 'missing'
        with self.assertRaises(ValueError):
            policy.request_features(80, student, teacher)

    def test_request_types_order_and_nonfinite(self):
        self.assertTrue(policy.same_request(STUDENT, dict(reversed(list(STUDENT.items())))))
        self.assertEqual(policy.request_digest(STUDENT), policy.request_digest(dict(reversed(list(STUDENT.items())))))
        altered = deepcopy(STUDENT)
        altered['market'].reverse()
        self.assertFalse(policy.same_request(STUDENT, altered))
        for value in (True, 1.):
            changed = deepcopy(STUDENT)
            changed['market'][0][2] = value
            self.assertFalse(policy.same_request(STUDENT, changed))
        self.assertTrue(policy.same_request([0.], [-0.]))
        self.assertEqual(policy.request_digest([0.]), policy.request_digest([-0.]))
        with self.assertRaises(ValueError):
            policy.same_request([float('nan')], [float('nan')])


class RequestModelTests(unittest.TestCase):
    def test_ridge_mean_loss_normalization_and_roundtrip(self):
        x = np.arange(48, dtype=float).reshape(6, 8)
        y = np.asarray([-2., 1., 4., -1., 3., 2.])
        model = policy.RequestModel.fit(x, y)
        np.testing.assert_array_equal(model.mean, x.mean(axis=0))
        np.testing.assert_array_equal(model.scale, np.maximum(x.std(axis=0), 1.))
        repeated = policy.RequestModel.fit(np.repeat(x, 2, axis=0), np.repeat(y, 2))
        np.testing.assert_allclose(model.predict(x), repeated.predict(x), atol=1e-12)
        restored = policy.RequestModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
        np.testing.assert_array_equal(model.predict(x), restored.predict(x))
        constant = policy.RequestModel.fit(x, np.full(6, 3.))
        np.testing.assert_allclose(constant.coefficients, 0., atol=1e-12)
        self.assertAlmostEqual(constant.intercept, 3.)

    def test_model_file_identity_and_schema_without_disk_artifacts(self):
        original = constant_model(1.)
        raw = json.dumps(original.to_dict(), allow_nan=False).encode()
        with patch.object(Path, 'read_bytes', return_value=raw):
            loaded = policy.RequestModel.load('mock-model', hashlib.sha256(raw).hexdigest())
            self.assertEqual(loaded.predict([0.] * 8), 1.)
            with self.assertRaises(ValueError):
                policy.RequestModel.load('mock-model', '0' * 64)
        wrong = original.to_dict()
        wrong['schema'] = 'task-reward-gate-v1'
        with self.assertRaises(ValueError):
            policy.RequestModel.from_dict(wrong)
        with patch.object(Path, 'read_bytes', return_value=b'{"value":NaN}'):
            with self.assertRaises(ValueError):
                policy.RequestModel.load('mock-model')

    def test_invalid_fit_scale_and_prediction_fail_closed(self):
        with self.assertRaises(ValueError):
            policy.RequestModel.fit([], [])
        with self.assertRaises(ValueError):
            policy.RequestModel.fit(np.zeros((2, 8)), [0., float('inf')])
        with self.assertRaises(ValueError):
            constant_model(0.).predict([float('nan')] * 8)
        with self.assertRaises(ValueError):
            policy.RequestModel([0.] * 8, [.5] * 8, [0.] * 8, 0.)


if __name__ == '__main__':
    unittest.main()
