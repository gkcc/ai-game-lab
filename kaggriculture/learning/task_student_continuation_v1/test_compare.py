"""In-memory synthetic tests of actual BoundedAgent/TaskAgent call paths."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_choice_v1.agent import TaskAgent
from learning.task_choice_v1.features import candidate_id
from learning.task_dagger_once_v1.policy import BoundedAgent, LIMITS
from learning.task_request_gate_v1.policy import _Budget
from .compare import ComparisonStopped, compare_turn


TASK_SHA, CROP_SHA = 'a' * 64, 'b' * 64


def observation(step=0):
    farm = dict(money=100, tiles=[[None]], farmer=[0, 0], hands=[[0, 0]],
                unlocked_quadrants=[], hires_today=0)
    return dict(player=0, step=step, day=0, hour=step, farms=[deepcopy(farm), deepcopy(farm)],
                private=dict(shed={}, seeds={}, inventories=[{}, {}]),
                market=dict(inventory={}, prices={}, params={}), town=dict(unlocked_shops=[]))


def configuration():
    return dict(boardSize=2, turnsPerDay=24, marketParams={})


class FakeNetwork:
    def __init__(self):
        self.sha, self.calls = TASK_SHA, []

    def __call__(self, inputs):
        self.calls.append(inputs['candidates'][0]['id'])
        return inputs['candidates'][0]['id']


class FakeCrop:
    def __init__(self):
        self.network = SimpleNamespace(sha=CROP_SHA)
        self.trace, self.records, self.last = None, [], None

    def finish(self, ctx, _result):
        for record in self.records:
            record['unit_request'] = deepcopy(ctx['actions'][record['unit']])
            record['buy_seed_orders'] = []
        self.last = dict(records=deepcopy(self.records),
                         final_reservations=dict(seeds=deepcopy(ctx['seeds']),
                             supply=deepcopy(ctx['supply']), needed_seeds=deepcopy(ctx['needed_seeds'])),
                         checkpoint_sha256=CROP_SHA)


def fake_encode(context):
    return deepcopy(context['inputs']), deepcopy(context['lookup'])


class SyntheticBounded(BoundedAgent):
    """Uses the real BoundedAgent.__call__ and TaskAgent.select/finish."""
    def __init__(self, scenario='same'):
        self.budget, self.network = _Budget(LIMITS), FakeNetwork()
        self.task_sha, self.trace, self.last, self.previous = TASK_SHA, None, None, None
        controller = self.controller = TaskAgent.__new__(TaskAgent)
        controller.records, controller.last, controller.crop = [], None, FakeCrop()

        def student(inputs):
            self.budget.forward('task', len(inputs['candidates']))
            return self.network(inputs)

        controller.task_selector = student

        def runner(obs, cfg, select, finish):
            ctx = dict(observation=obs, tasks=[], choices=[], units=[[0, 0], [0, 0]], actions=[['PASS'], ['PASS']],
                       seeds={}, needed_seeds={}, supply={}, projected_shed={}, stored_fertilizer=0,
                       carried_total=0, day=0, hour=obs['hour'], tpd=24, remaining=24, today=0,
                       final_day=False, last_plant_age=0, available={0, 1})

            def set_choices(descriptors):
                ctx['tasks'] = [(xy, [op], priority) for _unit, xy, op, priority in descriptors]
                ctx['choices'] = [(priority, unit, index)
                                  for index, (unit, _xy, _op, priority) in enumerate(descriptors)]
                candidates = [dict(id=candidate_id(unit, xy, op), unit=unit, target=xy, op=op)
                              for unit, xy, op, _priority in descriptors]
                ctx['inputs'] = dict(observation=deepcopy(obs), configuration=deepcopy(cfg), candidates=candidates)
                ctx['lookup'] = {c['id']: choice for c, choice in zip(candidates, ctx['choices'])}

            first_descriptors = [(0, [0, 0], 'WATER', 1), (0, [1, 0], 'WATER', 3),
                                 (0, [2, 0], 'WATER', 2)]
            if scenario == 'wide_prefix':
                first_descriptors = [(0, [i, 0], 'WATER', 1 + i % 2) for i in range(257)]
            set_choices(first_descriptors)
            first = select(ctx)
            forced = first[-1] != 0
            if scenario == 'crop_after' and forced:
                controller.crop.records.append(dict(step=obs['step'], decision=0, unit=0,
                    target=[0, 0], x=[1.0], schema_sha256='c' * 64, teacher='WHEAT',
                    chosen='WHEAT', outcome='plant_request', gate_reason=None,
                    inference_seconds=0.001, logits=[0.0] * 5, prior={},
                    selected_seed_before=1, source='model'))
            ctx['available'].remove(0)
            if not (scenario == 'short_suffix' and forced):
                # The second forward receives new candidates; it is not a replayed suffix.
                suffix = [(1, [3, 1] if forced else [0, 1], 'HARVEST', 1),
                          (1, [1, 1], 'HARVEST', 3), (1, [2, 1], 'HARVEST', 2)]
                if scenario == 'wide_suffix' and forced:
                    suffix = [(1, [i, 1], 'HARVEST', 1 + i % 2) for i in range(257)]
                set_choices(suffix)
                select(ctx)
                ctx['available'].remove(1)
            # Set the same final transient context where requested by the scenario.
            if scenario != 'short_suffix':
                set_choices([(1, [0, 1], 'HARVEST', 1), (1, [1, 1], 'HARVEST', 3),
                             (1, [2, 1], 'HARVEST', 2)])
            if scenario == 'different' and forced:
                ctx['actions'][0] = ['WATER']
            if scenario == 'reservation' and forced:
                ctx['supply']['WHEAT'] = 1
            result = dict(farmer=deepcopy(ctx['actions'][0]), hands=deepcopy(ctx['actions'][1:]),
                          market=[['SELL', 'WHEAT', 1]])
            return finish(ctx, result)

        namespace = dict(_task_select=controller.select, _task_finish=controller.finish, runner=runner)
        exec('def execute(obs, cfg):\n    return runner(obs, cfg, _task_select, _task_finish)\n', namespace)
        controller.controller = namespace['execute']


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.encoder = patch('learning.task_choice_v1.agent.encode_candidates', side_effect=fake_encode)
        self.encoder.start()
        self.addCleanup(self.encoder.stop)
        self.created = []
        self.scenario = 'same'
        self.limits = _Budget(LIMITS).limits

    def factory(self):
        agent = SyntheticBounded(self.scenario)
        self.created.append(agent)
        return agent

    def source(self, step=0):
        agent = self.factory()
        obs, cfg = observation(), configuration()
        action = agent(obs, cfg)
        if step:
            previous = deepcopy(agent.last)
            obs = observation(step)
            action = agent(obs, cfg)
        else:
            previous = None
        doc = deepcopy(agent.last)
        self.created.clear()
        return doc, obs, cfg, action, previous

    def compare(self, source, *, alternative='teacher', guard=lambda: None, **overrides):
        doc, obs, cfg, action, previous = source
        values = dict(task_sha256=TASK_SHA, crop_sha256=CROP_SHA, game_limits=self.limits,
                      agent_factory=self.factory, guard=guard, previous_document=previous,
                      alternative=alternative)
        values.update(overrides)
        return compare_turn(doc, obs, cfg, action, 0, **values)

    def test_same_request_and_state_consumes_original_forward_and_new_suffix(self):
        source = self.source()
        result = self.compare(source)
        self.assertEqual(result.status, 'same')
        self.assertTrue(all(result.checks.values()))
        self.assertEqual(len(self.created), 2)
        self.assertEqual([len(a.network.calls) for a in self.created], [2, 2])
        self.assertNotEqual(self.created[0].network.calls[1], self.created[1].network.calls[1])
        self.assertEqual(repr(result), '<PrivateComparison>')
        self.assertNotIn('reference_action', result.public_summary())

    def test_different_complete_request_is_not_equivalence(self):
        self.scenario = 'different'
        result = self.compare(self.source())
        self.assertEqual(result.status, 'different')
        self.assertFalse(result.checks['request_equal'])

    def test_equal_request_but_changed_reservation_is_unknown(self):
        self.scenario = 'reservation'
        result = self.compare(self.source())
        self.assertEqual(result.status, 'unknown')
        self.assertTrue(result.checks['request_equal'])
        self.assertFalse(result.checks['reservations_equal'])

    def test_equal_request_with_different_future_budget_is_unknown(self):
        self.scenario = 'short_suffix'
        result = self.compare(self.source())
        self.assertEqual(result.status, 'unknown')
        self.assertTrue(result.checks['request_equal'])
        self.assertFalse(result.checks['next_turn_control_equal'])

    def test_equal_request_with_different_crop_trace_is_unknown(self):
        self.scenario = 'crop_after'
        result = self.compare(self.source())
        self.assertEqual(result.status, 'unknown')
        self.assertTrue(result.checks['request_equal'])
        self.assertFalse(result.checks['stable_crop_equal'])

    def test_distinct_third_candidate_and_missing_candidate(self):
        source = self.source()
        third = source[0]['records'][0]['inputs']['candidates'][2]['id']
        self.assertEqual(self.compare(source, alternative=third).status, 'same')
        self.created.clear()
        self.assertEqual(self.compare(source, alternative=None).reason, 'no_third_candidate')
        self.assertEqual(self.compare(source, alternative='not-a-candidate').status, 'unknown')
        self.assertFalse(self.created)

    def test_previous_bounded_state_is_required_and_restored(self):
        source = self.source(step=1)
        self.assertEqual(self.compare(source).status, 'same')
        self.created.clear()
        missing = (*source[:4], None)
        self.assertEqual(self.compare(missing).reason, 'previous_control_state_unavailable')
        self.assertFalse(self.created)
        bad = deepcopy(source[4])
        bad['model_counts']['task_forwards'] += 1
        with self.assertRaises(ComparisonStopped):
            self.compare(source, previous_document=bad)

    def test_later_callback_preserves_all_prior_student_choices(self):
        source = self.source()
        doc, obs, cfg, action, previous = source
        result = compare_turn(doc, obs, cfg, action, 1, task_sha256=TASK_SHA, crop_sha256=CROP_SHA,
            game_limits=self.limits, agent_factory=self.factory, guard=lambda: None,
            previous_document=previous)
        self.assertEqual(result.status, 'same')
        self.assertEqual([len(a.network.calls) for a in self.created], [2, 2])

    def test_reused_controller_instance_cannot_be_equivalence_evidence(self):
        source = self.source()
        singleton = SyntheticBounded()
        with self.assertRaises(ComparisonStopped):
            self.compare(source, agent_factory=lambda: singleton)

    def test_overwide_callback_before_target_stops_before_student_forward(self):
        self.scenario = 'wide_prefix'
        doc, obs, cfg, action, previous = self.source()
        with self.assertRaisesRegex(ComparisonStopped, 'candidate limit'):
            compare_turn(doc, obs, cfg, action, 1, task_sha256=TASK_SHA, crop_sha256=CROP_SHA,
                game_limits=self.limits, agent_factory=self.factory, guard=lambda: None,
                previous_document=previous)
        self.assertEqual(len(self.created), 1)
        self.assertEqual(len(self.created[0].network.calls), 0)

    def test_overwide_intervention_suffix_stops_before_suffix_forward(self):
        self.scenario = 'wide_suffix'
        source = self.source()
        with self.assertRaisesRegex(ComparisonStopped, 'candidate limit'):
            self.compare(source)
        self.assertEqual(len(self.created), 2)
        self.assertEqual([len(a.network.calls) for a in self.created], [2, 1])

    def test_source_or_action_mismatch_stops_before_intervention(self):
        source = self.source()
        bad_action = deepcopy(source[3])
        bad_action['market'].reverse()
        bad_action['market'].append(['HIRE'])
        with self.assertRaises(ComparisonStopped):
            self.compare((*source[:3], bad_action, source[4]))
        self.assertEqual(len(self.created), 1)
        with self.assertRaises(ComparisonStopped):
            self.compare(source, task_sha256='c' * 64)

    def test_guard_failure_propagates_and_game_limit_identity_is_checked(self):
        source = self.source()
        with self.assertRaisesRegex(RuntimeError, 'resource stop'):
            self.compare(source, guard=lambda: (_ for _ in ()).throw(RuntimeError('resource stop')))
        limits = deepcopy(self.limits)
        limits['controller_calls'] -= 1
        with self.assertRaises(ComparisonStopped):
            self.compare(source, game_limits=limits)


if __name__ == '__main__':
    unittest.main()
