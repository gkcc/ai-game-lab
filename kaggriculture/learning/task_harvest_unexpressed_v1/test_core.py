"""Small in-memory orchestration fixtures, not an alternate game simulator."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import unittest

from decision.api import observe
from decision.execution import resources
from learning.task_harvest_window_v1.test_core import FakeFactory, FakeRival, observed, stable, fixture_tick
from learning.task_harvest_window_v1.test_selection import DAYS, candidate, record
from learning.task_request_gate_v1.policy import _Budget
from . import core
from .selection import TARGET_SCHEMA


class Actor:
    """Stateful counted stand-in; its only exposed task is WATER, never HARVEST."""
    def __init__(self, *, fail_step=None, extra_charge_step=None, tamper_step=None):
        self.budget = _Budget(core.GAME_LIMITS)
        self.network = object()
        self.controller = SimpleNamespace(crop=SimpleNamespace(network=object()), records=[])
        self.controller.task_selector = self.select
        self.original_selector = self.controller.task_selector
        self.last, self.previous, self.history = None, None, 0
        self.steps_seen = []
        self.fail_step, self.extra_charge_step, self.tamper_step = fail_step, extra_charge_step, tamper_step

    def select(self, inputs):
        self.budget.forward('task', len(inputs['candidates']))
        return inputs['candidates'][0]['id']

    def __call__(self, observation, configuration):
        view = observe(observation, configuration)
        observation, configuration = view.observation, view.configuration
        side, step = observation['player'], observation['step']
        if self.previous != (None if step == 0 else (side, step-1)):
            raise AssertionError('full sequential history required')
        self.steps_seen.append(step)
        self.budget.begin_turn()
        self.budget.controller()
        if step == self.extra_charge_step:
            self.budget.gate()
        farm = observation['farms'][side]
        positions = [farm['farmer'], *farm['hands']]
        actions, records = [['PASS'] for _ in positions], []
        for unit, position in enumerate(positions):
            inputs = record(step, side=side)['inputs']
            inputs['observation'], inputs['configuration'] = deepcopy(dict(observation)), deepcopy(configuration)
            inputs['candidates'] = [candidate(i, positions[i], 'WATER') for i in range(unit, len(positions))]
            inputs['planned']['available_units'] = list(range(unit, len(positions)))
            inputs['planned']['unit_actions'] = deepcopy(actions)
            inputs['inventory_item_order'] = [list(inv) for inv in observation['private']['inventories']]
            inputs['shed_item_order'] = list(observation['private']['shed'])
            chosen = self.controller.task_selector(inputs)
            actions[unit] = ['WATER']
            records.append(dict(inputs=inputs, chosen=chosen, teacher_label=chosen, unit=unit,
                target=list(position), op='WATER', unit_request=['WATER'], source='callback'))
        if step == self.fail_step:
            raise MemoryError('synthetic actor failure')
        self.history += step+1
        request = dict(farmer=actions[0], hands=actions[1:],
                       market=[['SYNTHETIC', self.history, farm['money']]])
        if step == self.tamper_step:
            records[-1]['teacher_label'] = 'synthetic tamper after original decision'
        self.controller.records = records
        self.last = dict(step=step, records=deepcopy(records), crop_records=[],
            final_reservations={'unit_actions': deepcopy(actions)},
            model_counts=dict(self.budget.counts), turn_model_counts=dict(self.budget.turn),
            fallbacks=0, load_errors=0, inference_errors=0, inference_seconds=0.123)
        self.previous = (side, step)
        return request


def fixture(side=0, units=1):
    template = record(0)['inputs']
    configuration = dict(template['configuration'], seed=None)
    farms = deepcopy(template['observation']['farms'])
    for farm in farms:
        farm['tiles'] = deepcopy(farms[0]['tiles'])
        farm['hands'] = [] if units == 1 else [[2, 1]]
    frame = []
    for player in (0, 1):
        obs = deepcopy(template['observation'])
        obs.update(player=player, farms=deepcopy(farms))
        obs['private'] = dict(inventories=[dict(WHEAT=player+1, MELON=2) for _ in range(units)],
                              shed=dict(WHEAT=3, MELON=4), seeds={'WHEAT': 0})
        frame.append(dict(observation=obs, action=None, reward=0, status='ACTIVE'))
    frames, traces = [frame], []
    actor, rival = Actor(), FakeRival()
    for step in range(719):
        joint = [None, None]
        joint[side] = actor(observed(frame, side), configuration)
        joint[1-side] = rival(observed(frame, 1-side), configuration)
        traces.append(deepcopy(actor.last))
        frame = fixture_tick(frame, joint, step)
        frames.append(frame)
    return dict(name='kaggriculture', configuration=configuration, info={'seed': 17}, steps=frames), traces


class Capture:
    """Capture test-clock records; these fabricated effects prove no game rule."""
    def __init__(self, *, blocked=False, bad_restore=False):
        self.entered = self.exited = 0
        self.blocked, self.bad_restore = blocked, bad_restore

    @contextmanager
    def __call__(self, environment):
        self.entered += 1
        original_step, interpreter = environment.step, environment.interpreter
        tracker = SimpleNamespace(records=[])
        environment.interpreter = object()

        def step(joint):
            before = deepcopy(environment.state)
            original_step(joint)  # Exactly one original fake clock advancement.
            after = environment.state
            for side in (0, 1):
                farm = before[0]['observation']['farms'][side]
                positions = [farm['farmer'], *farm['hands']]
                actions = [joint[side]['farmer'], *joint[side]['hands']]
                for unit, action in enumerate(actions):
                    x, y = positions[unit]
                    blocked = self.blocked and action == ['HARVEST']
                    tracker.records.append(dict(step=before[0]['observation']['step'],
                        phase='unit', player=side, unit=unit, requested=deepcopy(action),
                        effective=['PASS'] if blocked else deepcopy(action), position=[x, y],
                        status='blocked' if blocked else 'no_effect', reason='synthetic capture only',
                        before=resources(farm, before[side]['observation']['private']),
                        after=resources(after[0]['observation']['farms'][side], after[side]['observation']['private']),
                        tile_before=deepcopy(farm['tiles'][y][x]),
                        tile_after=deepcopy(after[0]['observation']['farms'][side]['tiles'][y][x])))

        environment.step = step
        try:
            yield tracker
        finally:
            environment.step = original_step
            if not self.bad_restore:
                environment.interpreter = interpreter
            self.exited += 1


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay, cls.traces = fixture()

    def setUp(self):
        self.meter, self.events, self.actors, self.rivals = core.Meter(), [], [], []
        self.closed = self.rows_read = 0

    def target(self, traces=None, *, side=0, unit=0):
        trace = (self.traces if traces is None else traces)[240]
        inputs = trace['records'][0]['inputs']
        own = inputs['observation']['farms'][side]
        positions = [own['farmer'], *own['hands']]
        actions = trace['final_reservations']['unit_actions']
        return dict(schema_version=TARGET_SCHEMA, side=side, step=240, ordinal=0, unit=unit,
                    target=positions[unit], input_sha256=core.ordered_hash(inputs),
                    original_unit_request=actions[unit], unit_requests_sha256=core.ordered_hash(actions))

    def run_probe(self, *, replay=None, traces=None, target=None, factory=None, capture=None,
                  fail_step=None, close_error=False, intervene_charge=None, intervene_tamper=None,
                  observer=observed, horizon=48):
        replay = self.replay if replay is None else replay
        traces = self.traces if traces is None else traces
        factory, capture = factory or FakeFactory(), capture or Capture()

        @contextmanager
        def trace_factory():
            def rows():
                for value in traces:
                    self.rows_read += 1
                    yield deepcopy(value)
            try:
                yield rows()
            finally:
                self.closed += 1
                if close_error:
                    raise OSError('synthetic close failure')

        def actor_factory():
            actor = Actor(fail_step=fail_step,
                extra_charge_step=intervene_charge if self.actors else None,
                tamper_step=intervene_tamper if self.actors else None)
            self.actors.append(actor)
            return actor

        def rival_factory():
            rival = FakeRival()
            self.rivals.append(rival)
            return rival

        runtime = core.Runtime(factory, factory.identity, observer, stable, lambda _actor: None)
        return core.run_pair(replay, trace_factory, target=target or self.target(), first_yield_days=DAYS,
            actor_factory=actor_factory, rival_factory=rival_factory, runtime=runtime,
            target_trace_factory=capture, meter=self.meter, guard=lambda: None,
            emit=self.events.append, horizon=horizon)

    def test_complete_factual_before_one_patch_keeps_original_intent_and_both_histories(self):
        capture = Capture()
        result = self.run_probe(capture=capture)
        self.assertEqual(self.meter.calls['official_step_attempts'], 96)
        self.assertEqual(self.meter.calls['official_step_returns'], 96)
        self.assertEqual(self.meter.calls['request_patch_attempts'], 1)
        self.assertEqual(self.meter.calls['request_patch_returns'], 1)
        self.assertEqual((capture.entered, capture.exited), (2, 2))
        self.assertEqual((self.rows_read, self.closed), (1438, 2))
        self.assertEqual(result['control']['factual_states_verified'], 49)
        self.assertEqual(result['intervention']['trace_rows_verified'], 719)
        for actor in self.actors:
            self.assertIs(actor.controller.task_selector, actor.original_selector)
            self.assertEqual(actor.steps_seen, list(range(288)))
        self.assertEqual([r.steps_seen for r in self.rivals], [list(range(288))]*2)
        turns = [e for e in self.events if e['kind'] == 'turn']
        patched = [e for e in turns if e['patch'] is not None]
        self.assertEqual(len(patched), 1)
        event = patched[0]
        self.assertEqual((event['arm'], event['step']), ('intervene', 240))
        self.assertEqual(event['request']['farmer'], ['HARVEST'])
        self.assertEqual(event['original_request']['farmer'], ['WATER'])
        self.assertEqual(event['trace']['records'][0]['unit_request'], ['WATER'])
        self.assertEqual(event['request']['market'], event['original_request']['market'])
        self.assertEqual(event['target_execution']['requested'], ['HARVEST'])
        for value in turns:
            if value is not event:
                self.assertTrue(core.same_request(value['request'], value['original_request']))
        self.assertEqual(result['physical_changed_by_offset'][:2], [False, True])
        self.assertNotEqual(result['control']['request_sha256'][1][1], result['intervention']['request_sha256'][1][1])
        self.assertEqual(list(result['control']['endpoints']), ['start', 'after_target', 'end'])
        self.assertEqual(result['control']['endpoints']['end']['step'], 288)

    def test_factual_order_mismatch_prevents_intervention_and_patch(self):
        with self.assertRaisesRegex(ValueError, 'state or order'):
            self.run_probe(factory=FakeFactory(reorder='shed'))
        self.assertEqual(len(self.actors), 1)
        self.assertEqual(self.meter.calls['official_step_returns'], 1)
        self.assertEqual(self.meter.calls['request_patch_attempts'], 0)

    def test_target_full_original_trace_and_actual_budget_must_match_before_patch(self):
        for changes in ({'intervene_tamper': 240}, {'intervene_charge': 240}):
            with self.subTest(changes=changes):
                self.setUp()
                with self.assertRaisesRegex(ValueError, 'suffix trace'):
                    self.run_probe(**changes)
                self.assertEqual(self.meter.calls['official_step_returns'], 48)
                self.assertEqual(self.meter.calls['request_patch_attempts'], 0)

    def test_original_complete_request_binds_market_not_just_units(self):
        replay = deepcopy(self.replay)
        replay['steps'][241][0]['action']['market'].reverse()
        replay['steps'][241][0]['action']['market'].append(['EXTRA', 1])
        with self.assertRaisesRegex(ValueError, 'suffix request'):
            self.run_probe(replay=replay)
        self.assertEqual(self.meter.calls['official_step_attempts'], 0)

    def test_tail_missing_or_broken_cannot_start_second_arm_after_48_valid_steps(self):
        for traces in (self.traces[:-1], self.traces + [dict(step=719)]):
            with self.subTest(length=len(traces)):
                self.setUp()
                with self.assertRaises(ValueError):
                    self.run_probe(traces=traces)
                self.assertEqual(self.meter.calls['official_step_returns'], 48)
                self.assertEqual(self.meter.calls['arm_returns'], 0)
                self.assertEqual(len(self.actors), 1)
                self.assertFalse(any(e['kind'] == 'arm_end' for e in self.events))

    def test_close_failure_preserves_primary_and_never_reports_successful_arm(self):
        with self.assertRaises(MemoryError):
            self.run_probe(fail_step=240, close_error=True)
        self.assertEqual(self.meter.costs['suffix']['controller_calls'], 1)
        self.assertEqual(self.meter.costs['suffix']['task_forwards'], 1)
        self.assertEqual(self.meter.calls['trace_close_failures'], 1)
        self.assertEqual(self.meter.calls['official_step_attempts'], 0)
        self.setUp()
        with self.assertRaises(OSError):
            self.run_probe(close_error=True)
        self.assertEqual(self.meter.calls['official_step_returns'], 48)
        self.assertEqual(self.meter.calls['arm_returns'], 0)

    def test_step_failure_counts_attempt_only_and_target_hook_is_closed(self):
        capture = Capture()
        with self.assertRaises(OSError):
            self.run_probe(factory=FakeFactory(fail_step=240), capture=capture)
        self.assertEqual(self.meter.calls['official_step_attempts'], 1)
        self.assertEqual(self.meter.calls['official_step_returns'], 0)
        self.assertEqual((capture.entered, capture.exited), (1, 1))
        self.assertEqual(self.closed, 1)

    def test_blocked_target_execution_is_recorded_without_inventing_harvest_success(self):
        result = self.run_probe(capture=Capture(blocked=True))
        record = result['intervention']['target_execution']
        self.assertEqual(record['status'], 'blocked')
        self.assertEqual(record['requested'], ['HARVEST'])
        self.assertEqual(record['effective'], ['PASS'])
        self.assertNotIn('harvest_profit', result)

    def test_unrestored_hook_rejected_before_intervention(self):
        with self.assertRaisesRegex(ValueError, 'hooks were not restored'):
            self.run_probe(capture=Capture(bad_restore=True))
        self.assertEqual(self.meter.calls['official_step_returns'], 1)
        self.assertEqual(len(self.actors), 1)

    def test_stage_limits_and_invalid_target_stop_without_extra_real_work(self):
        with self.assertRaises(ValueError):
            self.run_probe(horizon=47)
        self.assertEqual(len(self.actors), 0)
        self.meter.calls['official_step_attempts'] = 384
        with self.assertRaisesRegex(ValueError, 'transition budget'):
            self.run_probe()
        self.assertEqual(self.meter.calls['official_step_attempts'], 384)
        self.setUp()
        self.meter.calls['arm_attempts'] = 8
        with self.assertRaisesRegex(ValueError, 'Eight-arm'):
            self.run_probe()
        self.assertEqual(len(self.actors), 0)

    def test_other_private_or_seed_envelope_never_reaches_actor(self):
        def leak(frame, side):
            observation = observed(frame, side)
            observation['info'] = {'seed': 17}
            return observation
        with self.assertRaises(ValueError):
            self.run_probe(observer=leak)
        self.assertEqual(self.meter.calls['actor_attempts'], 0)

    def test_hand_patch_on_side_one_preserves_farmer_and_other_trace_choices(self):
        replay, traces = fixture(side=1, units=2)
        result = self.run_probe(replay=replay, traces=traces, target=self.target(traces, side=1, unit=1))
        event = next(e for e in self.events if e.get('patch') is not None)
        self.assertEqual(event['request']['farmer'], ['WATER'])
        self.assertEqual(event['request']['hands'], [['HARVEST']])
        self.assertEqual(event['original_request']['hands'], [['WATER']])
        self.assertEqual([r['unit_request'] for r in event['trace']['records']], [['WATER'], ['WATER']])
        self.assertEqual(result['intervention']['target_execution']['unit'], 1)


if __name__ == '__main__':
    unittest.main()
