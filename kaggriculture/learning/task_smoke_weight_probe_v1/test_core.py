"""Synthetic cases only: no real replay, checkpoint, agent inference or game."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from learning.task_dagger_once_v1.policy import LIMITS
from learning.task_request_gate_v1.policy import _Budget
from . import core


TASK_SHA = 'a' * 64
CROP_SHA = 'b' * 64


class FakeTask:
    def __init__(self, *, zero=False, bad_flip=False):
        self.sha, self.metadata, self.last = TASK_SHA, {'schema': 'synthetic'}, None
        self.mean, self.scale = np.zeros(1), np.ones(1)
        self.params = {'w0': np.ones((1, 1)), 'b0': np.zeros(1),
                       'w1': np.ones((1, 1)), 'b1': np.zeros(1),
                       'w2': np.zeros((1, 1)) if zero else np.ones((1, 1)),
                       'b2': np.zeros(1) if zero else np.asarray([0.25])}
        self.bad_flip = bad_flip

    def __call__(self, inputs):
        ids = [v['id'] for v in inputs['candidates']]
        logits = np.asarray([v['x'][0] * self.params['w2'][0, 0] + self.params['b2'][0]
                             for v in inputs['candidates']], dtype=np.float64)
        if self.bad_flip and self.sha != TASK_SHA:
            logits += 1
        choice = ids[int(np.argmax(logits))]
        self.last = {'chosen': choice, 'logits': logits.tolist(),
                     'checkpoint_sha256': self.sha, 'fallbacks': 0}
        return choice


class FakeCrop:
    def __init__(self):
        self.sha, self.meta = CROP_SHA, {'schema': 'synthetic'}
        self.mean, self.scale = np.zeros(1), np.ones(1)
        self.weights = [(np.ones((1, 1)), np.zeros(1))]


class FakeAgent:
    """A small deterministic synthetic controller with the same state contract."""
    def __init__(self, *, tie=False, no_task=False, zero=False,
                 contaminate=False, bad_flip=False, changed_control=False):
        self.budget = _Budget(LIMITS)
        self.network = FakeTask(zero=zero, bad_flip=bad_flip)
        self.controller = SimpleNamespace(crop=SimpleNamespace(network=FakeCrop()))
        self.task_sha, self.previous, self.last = TASK_SHA, None, None
        self.tie, self.no_task, self.contaminate = tie, no_task, contaminate
        self.changed_control = changed_control

    def __call__(self, obs, cfg):
        step, side = obs['step'], obs['player']
        if step == 0:
            self.budget.reset()
            self.previous = None
        if self.previous is None:
            assert step == 0
        else:
            assert self.previous == (side, step - 1)
        self.budget.begin_turn()
        self.budget.controller()
        records, choice = [], 'a'
        if not self.no_task:
            candidates = [{'id': 'a', 'x': [1.0]}, {'id': 'b', 'x': [1.0 if self.tie else -1.0]}]
            inputs = {'observation': deepcopy(obs), 'configuration': deepcopy(cfg),
                      'planned': {'synthetic': 1}, 'candidates': candidates}
            if self.contaminate and self.task_sha != TASK_SHA:
                inputs['planned']['synthetic'] = 2
            self.budget.forward('task', len(candidates))
            choice = self.network(deepcopy(inputs))
            records.append({'inputs': inputs, 'chosen': choice, 'teacher_label': 'synthetic-label'})
        action = {'farmer': ['EAST' if choice == 'a' else 'WEST'], 'hands': [], 'market': []}
        if self.changed_control:
            action['farmer'] = ['PASS']
        self.last = {'step': step, 'records': records, 'crop_records': [], 'checkpoint_sha256': CROP_SHA,
                     'task_checkpoint_sha256': self.task_sha, 'fallbacks': 0,
                     'load_errors': 0, 'inference_errors': 0,
                     'model_counts': dict(self.budget.counts), 'turn_model_counts': dict(self.budget.turn)}
        self.previous = (side, step)
        return action


def observation(step, side):
    return {'player': side, 'step': step, 'day': 1, 'hour': 0,
            'farms': [{'money': 1, 'tiles': [[None]], 'farmer': [0, 0], 'hands': []},
                      {'money': 2, 'tiles': [[None]], 'farmer': [0, 0], 'hands': []}],
            'private': {'shed': {}, 'seeds': {}, 'inventories': [{}]},
            'market': {'inventory': {}, 'prices': {}, 'params': {}},
            'town': {'unlocked_shops': []}}


def fixture(*, tie=False, no_task=False):
    configuration = {'episodeSteps': 720, 'boardSize': 1, 'marketParams': {}}
    frames = [[{'observation': observation(step, side)} for side in (0, 1)] for step in range(720)]
    traces, baseline = [], FakeAgent(tie=tie, no_task=no_task)
    for step in range(719):
        obs, cfg = core.legal_view(frames[step], 0, configuration)
        action = baseline(obs, cfg)
        frames[step + 1][0]['action'] = action
        traces.append(deepcopy(baseline.last))
    return {'steps': frames, 'configuration': configuration}, traces


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay, cls.traces = fixture()

    def compare(self, replay=None, traces=None, factory=FakeAgent):
        return core.compare(self.replay if replay is None else replay,
                            self.traces if traces is None else traces,
                            factory, LIMITS, TASK_SHA, CROP_SHA)

    def test_fixed_unique_32_turns_and_full_counterfactual(self):
        self.assertEqual(core.SAMPLE_TURNS, tuple(i * 718 // 31 for i in range(32)))
        self.assertEqual((len(set(core.SAMPLE_TURNS)), core.SAMPLE_TURNS[0], core.SAMPLE_TURNS[-1]), (32, 0, 718))
        report = self.compare()
        self.assertTrue(report['passed'])
        self.assertEqual(report['counts']['baseline_actions_reproduced'], 32)
        self.assertEqual(report['counts']['control_traces_equal'], 32)
        self.assertEqual(report['counts']['turns_with_first_choice_difference'], 32)
        self.assertEqual(report['counts']['turns_with_request_difference'], 32)
        self.assertEqual(report['counts']['comparable_selector_calls'], 32)
        self.assertEqual(report['counts']['offline_controller_calls'], 96)
        self.assertTrue(report['task_choice_control_observed'])
        self.assertTrue(report['action_control_observed'])
        self.assertNotEqual(report['baseline_task_parameter_sha256'], report['intervention_task_parameter_sha256'])
        self.assertNotIn('synthetic-label', str(report))
        self.assertNotIn('logits', report)

    def test_all_719_trace_rows_are_validated_before_actor_construction(self):
        changes = (('step', 8), ('task_checkpoint_sha256', 'c' * 64), ('fallbacks', 1))
        for name, value in changes:
            traces = deepcopy(self.traces)
            traces[1][name] = value
            calls = []
            with self.subTest(field=name), self.assertRaises(core.ProbeError):
                self.compare(traces=traces, factory=lambda: calls.append(True))
            self.assertEqual(calls, [])
        traces = deepcopy(self.traces)
        traces[1]['model_counts']['task_rows'] += 1
        with self.assertRaises(core.ProbeError): self.compare(traces=traces)
        traces = deepcopy(self.traces)
        traces[1]['turn_model_counts']['controller_calls'] = True
        with self.assertRaises(core.ProbeError): self.compare(traces=traces)
        with self.assertRaises(core.ProbeError): self.compare(traces=self.traces[:-1])

    def test_unsampled_trace_records_must_reconcile_with_counts(self):
        def missing_task(row): row['records'] = []
        def extra_task(row): row['records'].append(deepcopy(row['records'][0]))
        def bad_candidates(row): row['records'][0]['inputs']['candidates'] = 'invalid'
        def extra_candidate(row): row['records'][0]['inputs']['candidates'].append({'id': 'c'})
        def missing_crop_field(row): del row['crop_records']
        def extra_crop(row): row['crop_records'].append({'synthetic': True})
        def malformed_crop(row): row['crop_records'] = ['invalid']
        for mutate in (missing_task, extra_task, bad_candidates, extra_candidate,
                       missing_crop_field, extra_crop, malformed_crop):
            traces = deepcopy(self.traces)
            mutate(traces[1])  # step 1 is neither sampled nor a sampled predecessor
            calls = []
            with self.subTest(mutation=mutate.__name__), self.assertRaises(core.ProbeError):
                self.compare(traces=traces, factory=lambda: calls.append(True))
            self.assertEqual(calls, [])
        traces = deepcopy(self.traces)
        for row in traces[1:]:
            row['model_counts']['crop_forwards'] += 1
            row['model_counts']['crop_rows'] += 1
        traces[1]['turn_model_counts']['crop_forwards'] = 1
        traces[1]['turn_model_counts']['crop_rows'] = 1
        with self.assertRaises(core.ProbeError):
            core.validate_trace(traces, LIMITS, TASK_SHA, CROP_SHA)
        traces[1]['crop_records'] = [{}]
        core.validate_trace(traces, LIMITS, TASK_SHA, CROP_SHA)

    def test_same_weight_control_and_recorded_action_mismatch_fail(self):
        calls = []
        def factory():
            calls.append(1)
            return FakeAgent(changed_control=len(calls) == 2)
        with self.assertRaises(core.ProbeError): self.compare(factory=factory)
        replay = deepcopy(self.replay)
        replay['steps'][1][0]['action']['farmer'] = ['PASS']
        with self.assertRaises(core.ProbeError): self.compare(replay=replay)

    def test_changed_input_or_wrong_logit_intervention_fail(self):
        with self.assertRaises(core.ProbeError): self.compare(factory=lambda: FakeAgent(contaminate=True))
        with self.assertRaises(core.ProbeError): self.compare(factory=lambda: FakeAgent(bad_flip=True))
        with self.assertRaises(core.ProbeError): self.compare(factory=lambda: FakeAgent(zero=True))

    def test_exact_ties_and_no_callback_do_not_read_stale_logits(self):
        replay, traces = fixture(tie=True)
        report = self.compare(replay, traces, factory=lambda: FakeAgent(tie=True))
        self.assertEqual(report['counts']['baseline_tied_calls'], 32)
        self.assertEqual(report['counts']['intervention_tied_calls'], 32)
        self.assertEqual(report['counts']['turns_with_first_choice_difference'], 0)
        self.assertFalse(report['task_choice_control_observed'])
        self.assertFalse(report['action_control_observed'])
        replay, traces = fixture(no_task=True)
        report = self.compare(replay, traces, factory=lambda: FakeAgent(no_task=True))
        self.assertEqual(report['counts']['zero_task_turns'], 32)
        self.assertEqual(report['counts']['comparable_selector_calls'], 0)
        self.assertIsNone(report['max_absolute_logit_negation_error'])
        self.assertFalse(report['task_choice_control_observed'])
        self.assertFalse(report['action_control_observed'])

    def test_private_opponent_seed_and_future_fields_are_filtered(self):
        replay = deepcopy(self.replay)
        replay['configuration']['seed'] = 'synthetic-hidden-marker'
        for frame in replay['steps']:
            frame[0]['observation']['seed'] = 'synthetic-hidden-marker'
            frame[0]['observation']['future_market'] = 'synthetic-hidden-marker'
            frame[0]['observation']['farms'][1]['private'] = {'synthetic': 'synthetic-hidden-marker'}
            frame[1]['observation']['private'] = {'synthetic': 'synthetic-hidden-marker'}
        baseline = self.compare()
        changed = self.compare(replay=replay)
        self.assertEqual(baseline, changed)
        self.assertNotIn('synthetic-hidden-marker', str(changed))
        malformed = deepcopy(replay['steps'][0])
        malformed[0]['observation']['info'] = {'seed': 'synthetic-hidden-marker'}
        with self.assertRaises(core.ProbeError): core.legal_view(malformed, 0, replay['configuration'])

    def test_parameter_alias_between_control_and_intervention_is_rejected(self):
        agents = [FakeAgent(), FakeAgent(), FakeAgent()]
        agents[2].network.params['w0'] = agents[1].network.params['w0']
        with self.assertRaises(core.ProbeError): core._prepare_three(agents, TASK_SHA)


if __name__ == '__main__':
    unittest.main()
