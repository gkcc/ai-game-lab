"""In-memory orchestration regressions; fabricated clocks prove no game rules.

No official environment, checkpoint, file or historical trace is opened. The
tiny clock emits prearranged target facts solely to exercise the consumer and
the terminal/prefix contract; official mechanism evidence comes from real runs.
"""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import unittest

from decision.execution import resources
from learning.task_harvest_window_v1.test_core import observed, stable, FakeRival
from learning.task_request_gate_v1.policy import _Budget
from . import core
from .selection import TARGET_SCHEMA


TASK_SHA, CROP_SHA = 'a'*64, 'b'*64


def candidate(unit, op):
    return dict(id=f'u{unit}:x{unit}:y0:{op}',unit=unit,target=[unit,0],op=op)


def snapshot(budget):
    return dict(total=dict(budget.counts),turn=dict(budget.turn))


class Actor:
    def __init__(self, arm, target, *, source=False, fail_step=None):
        self.arm,self.target,self.source=arm,deepcopy(target),source
        self.target_proof,self.override_count=None,0
        self.budget=_Budget(core.GAME_LIMITS)
        self.network=object()
        self.controller=SimpleNamespace(crop=SimpleNamespace(network=object()),records=[])
        self.controller.task_selector=self.select
        self.previous,self.history,self.last=None,0,None
        self.fail_step,self.steps_seen=fail_step,[]

    def select(self, inputs):
        self.budget.forward('task',len(inputs['candidates']))
        return inputs['candidates'][0]['id']

    def __call__(self, observation, configuration):
        side,step=observation['player'],observation['step']
        if self.previous != (None if step==0 else (side,step-1)):
            raise AssertionError('Full policy history is required')
        if observation['private']['inventories'][0]['WHEAT'] != side+1:
            raise AssertionError('Wrong private seat')
        self.steps_seen.append(step)
        self.budget.begin_turn();self.budget.controller()
        records=[];actions=[['PASS'],['PASS']]
        for unit in range(2):
            target_turn=step==self.target['step'] and unit==self.target['unit']
            ops=['HARVEST','WATER'] if target_turn else ['PASS']
            # W may alter later callbacks and their model-input row costs. Those
            # post-target differences must not be rejected as prefix drift.
            if self.arm=='W' and step==self.target['step'] and unit>self.target['unit']:
                ops=['DROP','PASS']
            inputs=dict(observation=deepcopy(observation),configuration=deepcopy(configuration),
                candidates=[candidate(unit,op) for op in ops],planned=deepcopy(actions))
            before=snapshot(self.budget)
            proposed=self.controller.task_selector(inputs)
            after=snapshot(self.budget)
            proposal=dict(checkpoint_sha256=TASK_SHA,chosen=proposed,
                logits=[1.]+[0.]*(len(ops)-1),argmax_index=0,fallbacks=0)
            choice=proposed
            if target_turn and self.arm=='W':
                choice=inputs['candidates'][1]['id'];self.override_count+=1
            selected=next(c for c in inputs['candidates'] if c['id']==choice)
            actions[unit]=[selected['op']]
            row=dict(inputs=inputs,chosen=choice,teacher_label=proposed,unit=unit,
                target=[unit,0],op=selected['op'],unit_request=deepcopy(actions[unit]),
                source='callback',task_model=proposal)
            if target_turn and not self.source:
                self.target_proof=dict(schema_version='task-harvest-return-selector-proof-v1',
                    arm=self.arm,target=deepcopy(self.target),inputs=deepcopy(inputs),
                    input_sha256=core.ordered_hash(inputs),original_proposal=deepcopy(proposal),
                    actual_choice=choice,override_applied=self.arm=='W',override_count=self.override_count,
                    prefix_records_sha256=core.ordered_hash(records),budget_before=before,budget_after=after,
                    unit_request=deepcopy(actions[unit]))
            if target_turn and self.arm=='W':
                row['pair_choice']=dict(schema_version='task-harvest-return-choice-v1',arm='W',
                    input_sha256=core.ordered_hash(inputs),proposed_choice=proposed,
                    actual_choice=choice,override_applied=True)
            records.append(row)
        if step==self.fail_step:
            raise MemoryError('synthetic actor failure')
        self.history+=step+1
        request=dict(farmer=actions[0],hands=actions[1:],market=[['SYNTHETIC',self.history,
            observation['farms'][side]['money'],int(self.arm=='W' and self.override_count>0)]])
        self.last=dict(step=step,records=records,crop_records=[],final_reservations=dict(unit_actions=actions),
            model_counts=dict(self.budget.counts),turn_model_counts=dict(self.budget.turn),
            fallbacks=0,load_errors=0,inference_errors=0,inference_seconds=0.1)
        self.previous=(side,step)
        return request


def tick(frame, joint, target):
    """Fabricated state flags and test clock, deliberately no game simulator."""
    after=deepcopy(frame);step=frame[0]['observation']['step']
    side,unit=target['side'],target['unit']
    if step==target['step']:
        action=[joint[side]['farmer'],*joint[side]['hands']][unit]
        tile=after[0]['observation']['farms'][side]['tiles'][0][unit]
        if action==['HARVEST']:
            after[side]['observation']['private']['inventories'][unit]['MELON']+=tile['yield_units']
            tile=None
            change=2
        elif action==['WATER']:
            tile['watered_today']=True
            change=1
        else:raise AssertionError('Unexpected synthetic target request')
        for state in after:
            state['observation']['farms'][side]['tiles'][0][unit]=deepcopy(tile)
            state['observation']['farms'][side]['money']+=change
    for player in (0,1):
        after[player]['action']=deepcopy(joint[player])
        after[player]['observation']['day'],after[player]['observation']['hour']=divmod(step+1,24)
        if step+1==719:
            after[player]['status']='DONE'
            after[player]['reward']=after[0]['observation']['farms'][player]['money']
    after[0]['observation']['step']=step+1  # Official shared layout: no side 1 step.
    return after


def fixture(*,side=0,unit=0,start=716):
    farms=[dict(money=10,farmer=[0,0],hands=[[1,0]],unlocked_quadrants=[0],hires_today=1,
        tiles=[[dict(kind='PLANT',crop='MELON',yield_units=5,watered_today=False) for _ in range(2)]])
        for _ in (0,1)]
    frame=[dict(observation=dict(player=p,day=0,hour=0,farms=deepcopy(farms),market={},town={},
        private=dict(inventories=[dict(WHEAT=p+1,MELON=2) for _ in range(2)],
                     shed=dict(WHEAT=3,MELON=4),seeds=dict(WHEAT=0))),
        action=None,reward=0,status='ACTIVE') for p in (0,1)]
    frame[0]['observation']['step']=0
    config=dict(episodeSteps=720,seed=None)
    provisional=dict(side=side,step=start,unit=unit)
    actor,rival=Actor('H',provisional,source=True),FakeRival()
    frames,traces=[frame],[]
    for step in range(719):
        joint=[None,None]
        joint[side]=actor(observed(frame,side),config)
        joint[1-side]=rival(observed(frame,1-side),config)
        traces.append(deepcopy(actor.last))
        frame=tick(frame,joint,provisional);frames.append(frame)
    inputs=traces[start]['records'][unit]['inputs']
    target=dict(schema_version=TARGET_SCHEMA,side=side,step=start,ordinal=unit,unit=unit,target=[unit,0],
        board_size=2,unit_count=2,input_sha256=core.ordered_hash(inputs),
        task_checkpoint_sha256=TASK_SHA,crop_checkpoint_sha256=CROP_SHA,
        harvest_choice=candidate(unit,'HARVEST')['id'],water_choice=candidate(unit,'WATER')['id'],
        unit_request=['HARVEST'])
    return dict(name='kaggriculture',configuration=config,info=dict(seed=17),steps=frames),traces,target


class Environment:
    def __init__(self, owner, *,configuration,info,steps,debug):
        self.owner,self.configuration,self.info=owner,configuration,info
        self.steps,self.state,self.interpreter=steps,steps[-1],owner.identity
        if owner.reset_requested:self.reset()

    def reset(self):
        raise AssertionError('Reset must be denied before this original method')

    def step(self, joint):
        step=self.state[0]['observation']['step']
        if step==self.owner.fail_step:raise self.owner.primary
        self.state=tick(self.state,joint,self.owner.target)
        if self.owner.reorder:
            private=self.state[0]['observation']['private']
            if self.owner.reorder=='shed':private['shed']=dict(reversed(list(private['shed'].items())))
            else:private['inventories'][0]=dict(reversed(list(private['inventories'][0].items())))
        self.steps.append(self.state)


class Factory:
    def __init__(self,target,*,fail_step=None,reorder=None,reset_requested=False):
        self.target,self.identity=target,object()
        self.fail_step,self.reorder,self.reset_requested=fail_step,reorder,reset_requested
        self.primary=OSError('synthetic official-call failure')
        self.calls=[]

    def __call__(self, name, **kwargs):
        if name!='kaggriculture':raise AssertionError('Wrong engine name')
        self.calls.append(kwargs)
        return Environment(self,**kwargs)


class Capture:
    def __init__(self,target,*,blocked=False,exit_error=False):
        self.target,self.blocked,self.exit_error=target,blocked,exit_error

    @contextmanager
    def __call__(self,environment):
        original,interpreter=environment.step,environment.interpreter
        tracer=SimpleNamespace(records=[])
        def step(joint):
            before=deepcopy(environment.state);original(joint);after=environment.state
            side,unit=self.target['side'],self.target['unit'];x,y=self.target['target']
            action=[joint[side]['farmer'],*joint[side]['hands']][unit]
            farm=before[0]['observation']['farms'][side]
            tracer.records.append(dict(step=before[0]['observation']['step'],phase='unit',player=side,
                unit=unit,requested=deepcopy(action),effective=deepcopy(action),position=[x,y],
                status='blocked' if self.blocked else 'executed',
                before=resources(farm,before[side]['observation']['private']),
                after=resources(after[0]['observation']['farms'][side],after[side]['observation']['private']),
                tile_before=deepcopy(farm['tiles'][y][x]),
                tile_after=deepcopy(after[0]['observation']['farms'][side]['tiles'][y][x])))
        environment.step,environment.interpreter=step,object()
        try:yield tracer
        finally:
            environment.step,environment.interpreter=original,interpreter
            if self.exit_error:raise RuntimeError('synthetic tracer restoration failure')


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay,cls.traces,cls.target=fixture()
        cls.side1=fixture(side=1,unit=1)

    def setUp(self):
        self.events,self.actors,self.rivals=[],[],[]
        self.closed,self.read=0,0

    def run_case(self,*,data=None,factory=None,capture=None,close_error=False,emit=None,actor_fail=None):
        replay,traces,target=data or (self.replay,self.traces,self.target)
        self.meter=core.Meter(target['step'])
        factory=factory or Factory(target);capture=capture or Capture(target)
        @contextmanager
        def trace_factory():
            def rows():
                for trace in traces:
                    self.read+=1;yield deepcopy(trace)
            try:yield rows()
            finally:
                self.closed+=1
                if close_error:raise RuntimeError('synthetic source close failure')
        def actor_factory(arm):
            actor=Actor(arm,target,fail_step=actor_fail)
            self.actors.append(actor);return actor
        def rival_factory():
            rival=FakeRival();self.rivals.append(rival);return rival
        runtime=core.Runtime(factory,factory.identity,observed,stable,lambda _:None,
            core.same_request,Environment,capture)
        return core.run_pair(replay,trace_factory,target=target,actor_factory=actor_factory,
            rival_factory=rival_factory,runtime=runtime,meter=self.meter,guard=lambda:None,
            emit=self.events.append if emit is None else emit)

    def test_complete_history_terminal_counts_and_after_callback_freedom(self):
        original=(Environment.__init__,Environment.reset)
        result=self.run_case()
        self.assertEqual((Environment.__init__,Environment.reset),original)
        self.assertEqual((self.read,self.closed),(1438,2))
        calls=result['meter']['calls']
        self.assertEqual((calls['actor_returns'],calls['rival_returns']),(1438,1438))
        self.assertEqual((calls['environment_construct_returns'],calls['restore_returns'],
                          calls['environment_reset_attempts'],calls['official_step_returns']),(2,2,0,6))
        self.assertTrue(all(a.steps_seen==list(range(719)) for a in self.actors+self.rivals))
        self.assertEqual(result['terminal_delta_H_minus_W'],
            dict(own_net_profit=1,opponent_net_profit=0,margin=1,score=0.))
        h,w=result['arms']['H'],result['arms']['W']
        self.assertEqual(h['complete_source_requests_and_traces_verified'],719)
        self.assertEqual(h['factual_states_verified'],4)
        self.assertEqual((h['override_count'],w['override_count']),(0,1))
        self.assertEqual(w['episode_budget']['task_rows']-h['episode_budget']['task_rows'],1)
        target_events=[e for e in self.events if e['kind']=='turn' and e['step']==self.target['step']]
        self.assertNotEqual(target_events[0]['request']['hands'],target_events[1]['request']['hands'])
        self.assertLess(next(i for i,e in enumerate(self.events) if e['kind']=='arm_end' and e['arm']=='H'),
                        next(i for i,e in enumerate(self.events) if e['kind']=='arm_start' and e['arm']=='W'))

    def test_side_one_shared_clock_and_nonzero_callback_ordinal(self):
        result=self.run_case(data=self.side1)
        self.assertEqual(result['arms']['W']['target_proof']['target']['ordinal'],1)
        self.assertNotIn('step',self.side1[0]['steps'][0][1]['observation'])
        self.assertEqual(result['meter']['calls']['official_step_returns'],6)

    def test_full_source_tail_and_close_before_second_arm(self):
        cases=[]
        changed=deepcopy(self.traces);changed[-1]['records'][0]['teacher_label']='tampered'
        cases.append((self.replay,changed,self.target))
        cases.append((self.replay,self.traces+[deepcopy(self.traces[-1])],self.target))
        for data in cases:
            with self.subTest(extra=len(data[1])>719):
                self.setUp()
                with self.assertRaises(ValueError):self.run_case(data=data)
                self.assertEqual(len(self.actors),1)
                self.assertEqual(self.closed,1)
        self.setUp()
        with self.assertRaisesRegex(RuntimeError,'source close'):self.run_case(close_error=True)
        self.assertEqual(len(self.actors),1)
        self.assertFalse(any(e['kind']=='arm_end' for e in self.events))

    def test_inventory_and_shed_order_drift_rejected_before_water(self):
        for order in ('inventory','shed'):
            with self.subTest(order=order):
                self.setUp()
                with self.assertRaisesRegex(ValueError,'state/order/reward'):
                    self.run_case(factory=Factory(self.target,reorder=order))
                self.assertEqual(len(self.actors),1)

    def test_request_is_not_execution_and_reset_is_denied(self):
        with self.assertRaisesRegex(ValueError,'did not execute'):
            self.run_case(capture=Capture(self.target,blocked=True))
        self.assertEqual(len(self.actors),1)
        self.setUp()
        with self.assertRaisesRegex(ValueError,'budget exhausted'):
            self.run_case(factory=Factory(self.target,reset_requested=True))
        self.assertEqual(self.meter.calls['environment_reset_denials'],1)
        self.assertEqual(self.meter.calls['official_step_attempts'],0)

    def test_primary_official_error_survives_tracer_emit_and_source_close_errors(self):
        factory=Factory(self.target,fail_step=self.target['step'])
        def emit(event):
            if event['kind']=='turn' and event['scope']=='suffix':
                raise BufferError('synthetic evidence write failure')
            self.events.append(event)
        with self.assertRaises(OSError) as caught:
            self.run_case(factory=factory,capture=Capture(self.target,exit_error=True),
                close_error=True,emit=emit)
        self.assertIs(caught.exception,factory.primary)
        self.assertIsInstance(caught.exception.__cause__,BufferError)
        self.assertEqual(self.meter.calls['official_step_attempts'],1)
        self.assertEqual(self.meter.calls['official_step_returns'],0)
        self.assertEqual(self.meter.calls['evidence_write_failures'],1)
        self.assertEqual(self.meter.calls['trace_close_failures'],1)
        self.assertEqual(len(self.actors),1)

    def test_actor_failure_preserves_charges_and_terminal_score_is_real_cash(self):
        with self.assertRaises(MemoryError):self.run_case(actor_fail=3)
        self.assertEqual(self.meter.calls['actor_attempts'],4)
        self.assertEqual(self.meter.calls['actor_returns'],3)
        self.assertEqual(self.meter.costs['prefix']['controller_calls'],4)
        changed=deepcopy(self.replay['steps'][-1]);changed[0]['reward']+=1
        with self.assertRaisesRegex(ValueError,'rewards and cash'):
            core.terminal(changed,self.replay,0)

    def test_all_event_write_failures_are_counted_and_block_water(self):
        for failure_at in ('arm_start','prefix','arm_end'):
            with self.subTest(failure_at=failure_at):
                self.setUp()
                original=OSError('synthetic event write failure')
                def emit(event):
                    key=event['scope'] if event['kind']=='turn' else event['kind']
                    if key==failure_at:raise original
                    self.events.append(event)
                with self.assertRaises(OSError) as caught:self.run_case(emit=emit)
                self.assertIs(caught.exception,original)
                self.assertEqual(self.meter.calls['evidence_write_failures'],1)
                self.assertEqual(len(self.actors),1)
                self.assertEqual(self.closed,1)
                self.assertEqual(self.meter.calls['arm_returns'],0)
                if failure_at=='arm_end':
                    self.assertEqual(self.meter.calls['actor_returns'],719)
                    self.assertEqual(self.meter.calls['trace_close_returns'],1)


if __name__=='__main__':unittest.main()
