"""Pure in-memory orchestration checks; the fake clock is not a game simulator."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import unittest

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from learning.task_request_gate_v1.policy import _Budget
from .core import Meter, Runtime, frame_digest, run_pair
from .gate import HarvestSelector


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k != 'inference_seconds'}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def observed(frame, side):
    value = deepcopy(frame[side]['observation'])
    for key in ('step', 'farms', 'market', 'town', 'day', 'hour'):
        value[key] = deepcopy(frame[0]['observation'][key])
    return value


def inputs(observation, unit=0):
    # The real BoundedAgent/TaskAgent observation adapter also emits plain dicts.
    prefix = '' if unit == 0 else f'u{unit}-'
    return dict(observation=deepcopy(dict(observation)), candidates=[
        dict(id=prefix+'stay-water', unit=unit, target=[unit, 0], op='WATER'),
        dict(id=prefix+'local-harvest', unit=unit, target=[unit, 0], op='HARVEST')])


class FakeActor:
    """A stateful stand-in: a missing/wrong prefix changes subsequent requests."""
    def __init__(self, fail_step=None, extra_gate_step=None, callbacks=1, tamper_step=None):
        self.budget = _Budget(GAME_LIMITS)
        self.network = object()
        self.controller = SimpleNamespace(records=[], crop=SimpleNamespace(network=object()))
        self.controller.task_selector = self.select
        self.previous = None
        self.history = 0
        self.last = None
        self.fail_step = fail_step
        self.extra_gate_step = extra_gate_step
        self.callbacks, self.tamper_step = callbacks, tamper_step
        self.steps_seen = []

    def select(self, value):
        self.budget.forward('task', len(value['candidates']))
        return value['candidates'][0]['id']

    def __call__(self, observation, configuration):
        step, player = observation['step'], observation['player']
        if observation['private']['inventories'][0]['WHEAT'] != player + 1:
            raise AssertionError('wrong private seat')
        if self.previous is None:
            if step != 0:
                raise AssertionError('prefix was skipped')
        elif self.previous != (player, step - 1):
            raise AssertionError('history was not rebuilt')
        self.steps_seen.append(step)
        self.budget.begin_turn()
        self.budget.controller()
        if step == self.extra_gate_step:
            self.budget.gate()
        self.controller.records = []
        actions = []
        for unit in range(self.callbacks):
            value = inputs(observation, unit)
            choice = self.controller.task_selector(value)
            action = ['HARVEST'] if choice.endswith('local-harvest') else ['WATER']
            actions.append(action)
            record = dict(inputs=value, teacher_label=value['candidates'][0]['id'], chosen=choice,
                unit=unit, target=[unit, 0], op=action[0], source='callback', unit_request=action)
            self.controller.records.append(record)
        if step == self.fail_step:
            raise MemoryError('synthetic actor failure')
        self.history += step + 1
        request = dict(farmer=actions[0], hands=actions[1:], market=[['SYNTHETIC', self.history]])
        if step == self.tamper_step:
            self.controller.records[0]['teacher_label'] = 'tampered-prior-record'
        self.last = dict(step=step, records=deepcopy(self.controller.records), crop_records=[],
            model_counts=dict(self.budget.counts), turn_model_counts=dict(self.budget.turn),
            fallbacks=0, load_errors=0, inference_errors=0, inference_seconds=0.123)
        self.previous = (player, step)
        return request


class FakeRival:
    def __init__(self):
        self.steps_seen = []
        self.history = 0

    def __call__(self, observation, configuration):
        step = observation['step']
        if observation['private']['inventories'][0]['WHEAT'] != observation['player'] + 1:
            raise AssertionError('wrong private seat')
        if step != len(self.steps_seen):
            raise AssertionError('rival prefix skipped')
        self.steps_seen.append(step)
        self.history += step + 1
        return dict(farmer=['PASS'], hands=[],
            market=[['SYNTHETIC', self.history, observation['farms'][0]['money']]])


def fixture_tick(frame, joint, step):
    """Only a test clock and observable flag, with no Kaggriculture rules."""
    after = deepcopy(frame)
    for side in (0, 1):
        if any(action == ['HARVEST'] for action in [joint[side]['farmer'], *joint[side]['hands']]):
            for player in after:
                player['observation']['farms'][side]['money'] += 1
        after[side]['action'] = deepcopy(joint[side])
    for player in after:
        player['observation']['step'] = step + 1
        player['observation']['day'], player['observation']['hour'] = divmod(step + 1, 24)
        if step + 1 == 719:
            player['status'] = 'DONE'
    return after


def source_fixture(side=0, callbacks=1):
    farms = [dict(money=10, farmer=[0, 0], hands=[[i, 0] for i in range(1, callbacks)], tiles=[[
        dict(kind='PLANT', crop='STRAWBERRY', yield_units=2) for _ in range(callbacks)]]) for _ in (0, 1)]
    frame = [dict(observation=dict(player=player, step=0, day=0, hour=0,
        farms=deepcopy(farms), market={}, town={}, private=dict(
            inventories=[dict(WHEAT=player+1, MELON=2) for _ in range(callbacks)],
            shed=dict(WHEAT=3, MELON=4))),
        action=None, reward=0, status='ACTIVE') for player in (0, 1)]
    frames, traces = [frame], []
    actor, rival = FakeActor(callbacks=callbacks), FakeRival()
    configuration = dict(episodeSteps=720, seed=None)
    for step in range(719):
        joint = [None, None]
        joint[side] = actor(observed(frame, side), configuration)
        joint[1-side] = rival(observed(frame, 1-side), configuration)
        traces.append(deepcopy(actor.last))
        frame = fixture_tick(frame, joint, step)
        frames.append(frame)
    return dict(name='kaggriculture', configuration=configuration,
                info=dict(seed=17, extra='synthetic'), steps=frames), traces


class FakeEngine:
    def __init__(self, factory, *, configuration, info, steps, debug):
        self.owner = factory
        self.configuration, self.info, self.steps = configuration, info, steps
        self.state = steps[-1]
        self.interpreter = factory.identity

    def step(self, joint):
        step = len(self.steps) - 1
        if self.owner.fail_step == step:
            raise OSError('synthetic step failure')
        self.state = fixture_tick(self.state, joint, step)
        if self.owner.reorder == 'shed':
            old = self.state[0]['observation']['private']['shed']
            self.state[0]['observation']['private']['shed'] = dict(reversed(list(old.items())))
        elif self.owner.reorder == 'inventory':
            old = self.state[0]['observation']['private']['inventories'][0]
            self.state[0]['observation']['private']['inventories'][0] = dict(reversed(list(old.items())))
        self.steps.append(self.state)


class FakeFactory:
    def __init__(self, *, reorder=None, fail_step=None):
        self.identity = object()
        self.calls = []
        self.reorder, self.fail_step = reorder, fail_step

    def __call__(self, name, **kwargs):
        if name != 'kaggriculture':
            raise AssertionError('wrong environment')
        self.calls.append(deepcopy(kwargs))
        return FakeEngine(self, **kwargs)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay, cls.traces = source_fixture()

    def setUp(self):
        self.meter, self.actors, self.rivals, self.events = Meter(), [], [], []
        self.trace_closed = 0

    def target(self, step=2):
        return dict(side=0, step=step, ordinal=0,
            input_sha256=ordered_hash(self.traces[step]['records'][0]['inputs']),
            original_choice='stay-water', harvest_choice='local-harvest')

    def run_probe(self, *, replay=None, traces=None, factory=None, target=None,
                  actor_failure=None, close_error=False, horizon=48, runtime_observer=observed,
                  intervene_extra_gate_step=None, callbacks=1, intervene_tamper_step=None):
        replay = self.replay if replay is None else replay
        traces = self.traces if traces is None else traces
        factory = factory or FakeFactory()

        @contextmanager
        def trace_factory():
            try:
                yield iter(deepcopy(traces))
            finally:
                self.trace_closed += 1
                if close_error:
                    raise OSError('synthetic close failure')

        def actor_factory():
            value = FakeActor(actor_failure, intervene_extra_gate_step if self.actors else None,
                callbacks, intervene_tamper_step if self.actors else None)
            self.actors.append(value)
            return value

        def rival_factory():
            value = FakeRival()
            self.rivals.append(value)
            return value

        runtime = Runtime(factory, factory.identity, runtime_observer, stable, lambda _actor: None)
        return run_pair(replay, trace_factory, target=target or self.target(), actor_factory=actor_factory,
            rival_factory=rival_factory, runtime=runtime, meter=self.meter,
            guard=lambda: None, emit=self.events.append, horizon=horizon)

    def test_both_actors_rebuild_history_and_factually_reproduce_before_intervention(self):
        factory = FakeFactory()
        result = self.run_probe(factory=factory)
        self.assertTrue(result['passed'])
        self.assertEqual([a.steps_seen for a in self.actors], [list(range(50))] * 2)
        self.assertEqual([a.steps_seen for a in self.rivals], [list(range(50))] * 2)
        self.assertEqual(result['control']['factual_states_verified'], 49)
        self.assertEqual(self.meter.calls['official_step_attempts'], 96)
        self.assertEqual(self.meter.calls['official_step_returns'], 96)
        self.assertEqual(self.meter.costs['prefix']['controller_calls'], 4)
        self.assertEqual(self.meter.costs['suffix']['controller_calls'], 96)
        self.assertEqual(self.meter.costs['suffix']['task_forwards'], 96)
        self.assertEqual(self.trace_closed, 2)
        for arguments in factory.calls:
            self.assertEqual(arguments['info'], self.replay['info'])
            self.assertEqual(arguments['configuration'], self.replay['configuration'])
            self.assertEqual(arguments['steps'][:2], [None, None])
            self.assertEqual(arguments['steps'][2], self.replay['steps'][2])
        self.assertTrue(result['request_changed_by_offset'][0])
        self.assertEqual(result['physical_changed_by_offset'][:2], [False, True])
        # Live rival decisions respond to the altered public state.
        self.assertNotEqual(result['control']['request_sha256'][1][1],
                            result['intervention']['request_sha256'][1][1])
        proof = result['intervention']['target_proof']
        self.assertEqual(proof['teacher_label'], 'stay-water')
        self.assertEqual(proof['unit_request'], ['HARVEST'])
        self.assertEqual(proof['budget_after']['total']['task_forwards'] -
                         proof['budget_before']['total']['task_forwards'], 1)

    def test_prefix_trace_mismatch_stops_before_any_official_construction(self):
        traces = deepcopy(self.traces)
        traces[1]['model_counts']['task_rows'] += 1
        factory = FakeFactory()
        with self.assertRaisesRegex(ValueError, 'Prefix trace'):
            self.run_probe(traces=traces, factory=factory)
        self.assertEqual(factory.calls, [])
        self.assertEqual(len(self.actors), 1)
        self.assertEqual(self.trace_closed, 1)

    def test_factual_order_changes_stop_before_intervention(self):
        for kind in ('shed', 'inventory'):
            with self.subTest(kind=kind):
                self.setUp()
                with self.assertRaisesRegex(ValueError, 'state or order'):
                    self.run_probe(factory=FakeFactory(reorder=kind))
                self.assertEqual(len(self.actors), 1)
                self.assertEqual(self.meter.calls['official_step_returns'], 1)

    def test_wrong_target_digest_or_harvest_candidate_fails_before_transition(self):
        for field, value in (('input_sha256', '0' * 64), ('harvest_choice', 'missing')):
            with self.subTest(field=field):
                self.setUp()
                target = self.target()
                target[field] = value
                with self.assertRaises(ValueError):
                    self.run_probe(target=target)
                self.assertEqual(self.meter.calls['official_step_attempts'], 0)

    def test_failed_actor_preserves_charges_and_primary_exception_when_reader_close_fails(self):
        with self.assertRaises(MemoryError):
            self.run_probe(actor_failure=2, close_error=True)
        self.assertEqual(self.meter.costs['suffix']['controller_calls'], 1)
        self.assertEqual(self.meter.costs['suffix']['task_forwards'], 1)
        self.assertEqual(self.meter.calls['trace_close_failures'], 1)
        self.assertEqual(self.meter.calls['official_step_attempts'], 0)

    def test_successful_computation_with_reader_close_error_is_not_arm_success(self):
        with self.assertRaises(OSError):
            self.run_probe(close_error=True)
        self.assertEqual(self.meter.calls['arm_returns'], 0)
        self.assertEqual(len(self.actors), 1)
        self.assertEqual(self.meter.calls['official_step_returns'], 48)

    def test_step_failure_counts_attempt_not_return_and_does_not_start_alternate(self):
        with self.assertRaises(OSError):
            self.run_probe(factory=FakeFactory(fail_step=2))
        self.assertEqual(self.meter.calls['official_step_attempts'], 1)
        self.assertEqual(self.meter.calls['official_step_returns'], 0)
        self.assertEqual(len(self.actors), 1)

    def test_fixed_window_stage_limits_and_seed_free_policy_observation(self):
        with self.assertRaises(ValueError):
            self.run_probe(horizon=47)
        self.assertEqual(len(self.actors), 0)
        self.meter.calls['official_step_attempts'] = 384
        with self.assertRaisesRegex(ValueError, 'transition budget'):
            self.run_probe()
        self.assertEqual(self.meter.calls['official_step_attempts'], 384)

    def test_info_or_another_private_observation_is_rejected_before_actor(self):
        for leak in ('info', 'private'):
            with self.subTest(leak=leak):
                self.setUp()

                def observer(frame, side):
                    value = observed(frame, side)
                    if leak == 'info':
                        value['info'] = dict(seed=17)
                    else:
                        value['private']['shed']['OTHER_PRIVATE_ONLY'] = 1
                    return value

                with self.assertRaises(ValueError):
                    self.run_probe(runtime_observer=observer)
                self.assertEqual(self.meter.calls['actor_attempts'], 0)
                self.assertEqual(self.meter.calls['official_step_attempts'], 0)

    def test_restore_without_original_info_is_rejected_before_steps(self):
        class MissingInfo(FakeFactory):
            def __call__(self, name, **kwargs):
                kwargs['info'] = {}
                return super().__call__(name, **kwargs)

        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.run_probe(factory=MissingInfo())
        self.assertEqual(self.meter.calls['restore_returns'], 1)
        self.assertEqual(self.meter.calls['official_step_attempts'], 0)

    def test_arm_target_budget_disagreement_stops_before_first_intervention_step(self):
        with self.assertRaisesRegex(ValueError, 'target proofs'):
            self.run_probe(intervene_extra_gate_step=2)
        self.assertEqual(self.meter.calls['official_step_returns'], 48)
        self.assertEqual(self.meter.calls['official_step_attempts'], 48)
        self.assertEqual(len(self.actors), 2)

    def test_side_one_second_callback_success_and_pre_target_tamper_stop(self):
        replay, traces = source_fixture(side=1, callbacks=2)
        record = traces[2]['records'][1]
        target = dict(side=1, step=2, ordinal=1, input_sha256=ordered_hash(record['inputs']),
            original_choice='u1-stay-water', harvest_choice='u1-local-harvest')
        result = self.run_probe(replay=replay, traces=traces, target=target, callbacks=2)
        self.assertTrue(result['passed'])
        self.assertEqual(self.meter.calls['official_step_returns'], 96)
        self.assertEqual(result['intervention']['target_proof']['harvest_unit'], 1)
        self.assertEqual(result['intervention']['target_proof']['unit_request'], ['HARVEST'])
        self.assertEqual(self.meter.costs['suffix']['task_forwards'], 192)
        self.assertEqual(result['physical_changed_by_offset'][:2], [False, True])
        self.assertTrue(all(actor.previous == (1, 49) for actor in self.actors))
        self.setUp()
        with self.assertRaisesRegex(ValueError, 'Pre-target callback'):
            self.run_probe(replay=replay, traces=traces, target=target, callbacks=2,
                           intervene_tamper_step=2)
        self.assertEqual(self.meter.calls['official_step_returns'], 48)
        self.assertEqual(self.meter.calls['official_step_attempts'], 48)
        self.assertEqual(self.trace_closed, 2)

    def test_ninth_arm_is_rejected_before_actor_construction(self):
        self.meter.calls['arm_attempts'] = 8
        with self.assertRaisesRegex(ValueError, 'Eight-arm'):
            self.run_probe()
        self.assertEqual(len(self.actors), 0)

    def test_harvest_override_requires_exactly_one_original_forward(self):
        budget = _Budget(GAME_LIMITS)
        value = self.traces[0]['records'][0]['inputs']

        def double_forward(_inputs):
            for _ in range(2):
                budget.forward('task', len(_inputs['candidates']))
            return 'stay-water'

        gate = HarvestSelector(double_forward, lambda: 0,
            lambda: dict(total=dict(budget.counts), turn=dict(budget.turn)), self.target(0), 'intervene')
        with self.assertRaisesRegex(ValueError, 'budget increment'):
            gate(value)
        self.assertEqual(budget.counts['task_forwards'], 2)
        with self.assertRaises(ValueError):
            gate(value)

    def test_canonical_order_tracks_inventory_shed_but_not_record_action(self):
        before = deepcopy(self.replay['steps'][2])
        changed = deepcopy(before)
        changed[0]['action'] = dict(farmer=['HARVEST'], hands=[], market=[])
        self.assertEqual(frame_digest(before, physical=True), frame_digest(changed, physical=True))
        self.assertNotEqual(frame_digest(before), frame_digest(changed))
        changed = deepcopy(before)
        inv = changed[0]['observation']['private']['inventories'][0]
        changed[0]['observation']['private']['inventories'][0] = dict(reversed(list(inv.items())))
        self.assertNotEqual(frame_digest(before, physical=True), frame_digest(changed, physical=True))

    def test_struct_dict_subclass_observation_is_supported(self):
        class Struct(dict):
            pass

        def observer(frame, side):
            return Struct(observed(frame, side))

        self.assertTrue(self.run_probe(runtime_observer=observer)['passed'])


if __name__ == '__main__':
    unittest.main()
