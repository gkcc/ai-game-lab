"""One frozen HARVEST/WATER choice, then two official terminal continuations.

No server logic, real data, weights or official package are loaded at import.
The caller authenticates source replay/trace and identities, owns process and
output limits, and supplies a single-attempt job. Prefix calls reconstruct both
policies; only the restored suffix is executed by the official environment.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
import sys

from learning.task_harvest_window_v1.core import (
    BUDGET_KEYS, GAME_LIMITS, Runtime as OldRuntime, _counts, _observation,
    _restore, _rival_call, frame_digest, ordered_hash, request_digest, same_request)
from learning.task_harvest_unexpressed_v1.core import _actor_call, _execution
from .selection import validate_target


TURNS = 719
ARMS = ('H','W')
PROOF_KEYS = frozenset(('schema_version','arm','target','inputs','input_sha256',
    'original_proposal','actual_choice','override_applied','override_count',
    'prefix_records_sha256','budget_before','budget_after','unit_request'))
COMMON_PROOF = ('inputs','input_sha256','original_proposal','prefix_records_sha256','budget_before')


def require(value, message='Terminal return pair contract differs'):
    if not value:
        raise ValueError(message)


def _emit(emit, meter, event):
    try:
        emit(event)
    except BaseException:
        meter.calls['evidence_write_failures'] += 1
        raise


class Meter:
    """Per-pair hard caps; observed budget charges are not successful forwards."""
    def __init__(self, start):
        require(type(start) is int and 0 <= start < TURNS)
        self.target_step = start
        self.caps = dict(arm=2,actor_construct=2,rival_construct=2,actor=2*TURNS,rival=2*TURNS,
            restore=2,environment_construct=2,environment_reset=0,
            official_step=2*(TURNS-start),trace_open=2,trace_close=2)
        self.calls = Counter({name+'_'+suffix:0 for name in self.caps for suffix in ('attempts','returns')})
        self.costs = {scope:Counter({key:0 for key in BUDGET_KEYS}) for scope in ('prefix','suffix')}
        self.unknown_budget_observations = 0

    def start(self, operation, scope=None):
        require(operation in self.caps)
        if self.calls[operation+'_attempts'] >= self.caps[operation]:
            self.calls[operation+'_denials'] += 1
            raise ValueError('Terminal pair operation budget exhausted')
        require(scope is None or scope in self.costs)
        self.calls[operation+'_attempts'] += 1
        if scope is not None:self.calls[scope+'_'+operation+'_attempts'] += 1

    def returned(self, operation, scope=None):
        require(operation in self.caps and self.calls[operation+'_returns'] < self.calls[operation+'_attempts'])
        self.calls[operation+'_returns'] += 1
        if scope is not None:
            require(scope in self.costs)
            self.calls[scope+'_'+operation+'_returns'] += 1

    def observe_budget(self, scope, before, after):
        require(scope in self.costs and set(before) == set(after) == BUDGET_KEYS)
        delta = {key:after[key]-before[key] for key in BUDGET_KEYS}
        require(all(type(value) is int and value >= 0 for value in delta.values()), 'Actor budget went backwards')
        self.costs[scope].update(delta)
        total = self.costs['prefix']+self.costs['suffix']
        require(total['task_forwards'] <= 2*GAME_LIMITS['task_forwards']
            and total['crop_forwards'] <= 2*GAME_LIMITS['crop_forwards']
            and total['task_rows']+total['crop_rows'] <= 2*GAME_LIMITS['neural_input_rows']
            and total['controller_calls'] <= 2*TURNS
            and total['gate_evaluations'] <= 2*GAME_LIMITS['gate_evaluations']
            and total['teacher_proposals'] <= 2*GAME_LIMITS['teacher_proposals'],
            'Aggregate pair model charges exceeded their budget')

    def snapshot(self):
        return dict(target_step=self.target_step,caps=deepcopy(self.caps),calls=dict(self.calls),
            budget_by_scope={scope:dict(values) for scope,values in self.costs.items()},
            unknown_budget_observations=self.unknown_budget_observations,
            counter_scope='actual call attempts/returns; original forward budget charges include failed calls')

    def complete(self):
        require(all(self.calls[name+'_attempts'] == self.calls[name+'_returns'] == cap
                    for name,cap in self.caps.items()), 'Incomplete pair operation counts')
        require(self.unknown_budget_observations == 0
            and not any(value for key,value in self.calls.items() if key.endswith(('_denials','_failures'))))
        require(all(self.calls[scope+'_'+who+'_returns'] == 2*number
                    for scope,number in (('prefix',self.target_step),('suffix',TURNS-self.target_step))
                    for who in ('actor','rival')))


@dataclass(frozen=True)
class Runtime(OldRuntime):
    compare: object
    environment_class: object
    target_trace_factory: object
    identity: object = None


def official_runtime():
    """Prepare before protected output writes; construct zero actors or envs."""
    from learning.task_harvest_unexpressed_runtime_v2.runtime import prepare_runtime
    from learning.task_harvest_struct_compat_v1 import compat
    from learning.task_student_continuation_v1 import official_step
    from .policy import BoundedPairAgent
    prepared = prepare_runtime()
    original = prepared.runtime
    struct, binding = compat.authenticate(official_step.official_core,same_request)
    comparator = compat.Comparator(struct,same_request)
    def validate(actor):
        require(type(actor) is BoundedPairAgent and actor.trace is None
            and actor.controller.crop.trace is None and actor.network is not None
            and actor.budget.limits == GAME_LIMITS and actor.previous is None
            and actor.target_proof is None and actor.override_count == 0,
            'Expected a fresh frozen pair actor')
    return Runtime(original.make,original.interpreter,original.observation,original.stable,validate,
        comparator,official_step.official_core.Environment,prepared.target_trace_factory,
        dict(runtime_preparation=deepcopy(prepared.receipt),request_container_binding=binding))


@contextmanager
def _environment_counts(runtime, meter):
    """Scope real constructor/reset entries, including nested initialization."""
    cls = runtime.environment_class
    require(type(cls) is type)
    original_construct, original_reset = cls.__init__, cls.reset
    def construct(environment,*args,**kwargs):
        meter.start('environment_construct')
        result = original_construct(environment,*args,**kwargs)
        meter.returned('environment_construct')
        return result
    def reset(environment,*args,**kwargs):
        meter.start('environment_reset')  # Zero cap: deny before original reset.
        result = original_reset(environment,*args,**kwargs)
        meter.returned('environment_reset')
        return result
    try:
        cls.__init__,cls.reset = construct,reset
        yield
    finally:
        cls.__init__,cls.reset = original_construct,original_reset


def _same_target(left, right):
    require(type(left) is dict and type(right) is dict and set(left) == set(right)
        and all(ordered_hash(left[key]) == ordered_hash(right[key]) for key in left), 'Target binding differs')


def verify_proof(proof, target, arm):
    """Validate one callback receipt, not whole-turn equality after the choice."""
    require(type(proof) is dict and set(proof) == PROOF_KEYS
        and proof['schema_version'] == 'task-harvest-return-selector-proof-v1' and proof['arm'] == arm)
    _same_target(proof['target'],target)
    require(ordered_hash(proof['inputs']) == proof['input_sha256'] == target['input_sha256'])
    candidates = proof['inputs']['candidates']
    ids = [candidate['id'] for candidate in candidates]
    proposal = proof['original_proposal']
    require(type(proposal) is dict and set(proposal) == {
        'checkpoint_sha256','chosen','logits','argmax_index','fallbacks'})
    logits, index = proposal['logits'],proposal['argmax_index']
    require(type(logits) is list and len(logits) == len(ids) and bool(ids) and len(set(ids)) == len(ids)
        and all(type(value) is float and math.isfinite(value) for value in logits)
        and type(index) is int and index == max(range(len(logits)),key=logits.__getitem__)
        and ids[index] == proposal['chosen'] == target['harvest_choice']
        and proposal['checkpoint_sha256'] == target['task_checkpoint_sha256']
        and type(proposal['fallbacks']) is int and proposal['fallbacks'] == 0)
    require(type(proof['prefix_records_sha256']) is str and len(proof['prefix_records_sha256']) == 64
        and all(c in '0123456789abcdef' for c in proof['prefix_records_sha256']))
    chosen = target['harvest_choice' if arm == 'H' else 'water_choice']
    require(proof['actual_choice'] == chosen and type(proof['override_applied']) is bool
        and proof['override_applied'] == (arm == 'W') and type(proof['override_count']) is int
        and proof['override_count'] == int(arm == 'W')
        and proof['unit_request'] == ['HARVEST' if arm == 'H' else 'WATER'])
    for scope in ('total','turn'):
        before, after = proof['budget_before'][scope],proof['budget_after'][scope]
        require(set(before) == set(after) == BUDGET_KEYS
            and all(type(value) is int and value >= 0 for value in (*before.values(),*after.values())))
        require(all(after[key]-before[key] == (1 if key == 'task_forwards' else len(ids) if key == 'task_rows' else 0)
                    for key in BUDGET_KEYS), 'Target callback cost differs from one model forward')
    return deepcopy(proof)


def same_target_proofs(harvest, water):
    require(harvest['arm'] == 'H' and water['arm'] == 'W')
    _same_target(harvest['target'],water['target'])
    require(all(ordered_hash(harvest[key]) == ordered_hash(water[key]) for key in COMMON_PROOF),
            'Target input, original proposal, prior callbacks or entry budget differ')


def _execution_effect(records, target, request, before, after, arm):
    execution = _execution(records,target,request)
    op = 'HARVEST' if arm == 'H' else 'WATER'
    require(execution['status'] == 'executed' and execution['requested'] == execution['effective'] == [op],
            'Target action did not execute as registered')
    side,unit = target['side'],target['unit']
    x,y = target['target']
    tile = before[0]['observation']['farms'][side]['tiles'][y][x]
    require(ordered_hash(execution['tile_before']) == ordered_hash(tile)
        and isinstance(tile,dict) and tile.get('kind') == 'PLANT' and tile.get('crop') == 'MELON'
        and tile.get('watered_today') is False and type(tile.get('yield_units')) is int and tile['yield_units'] > 0)
    inventory_before = dict(execution['before']['inventories'][unit])
    inventory_after = dict(execution['after']['inventories'][unit])
    require(ordered_hash(inventory_before) == ordered_hash(before[side]['observation']['private']['inventories'][unit]))
    if arm == 'H':
        require(execution['tile_after'] is None
            and inventory_after.get('MELON',0)-inventory_before.get('MELON',0) == tile['yield_units'])
    else:
        atomic_after = execution['tile_after']
        require(isinstance(atomic_after,dict) and atomic_after.get('kind') == 'PLANT'
            and atomic_after.get('crop') == 'MELON' and atomic_after.get('watered_today') is True
            and ordered_hash(inventory_after) == ordered_hash(inventory_before))
    # Day-end logistics/production or subsequent units may change this whole
    # step again. Keep its endpoint separately; do not equate it with atomic after.
    execution['whole_step_target_tile'] = deepcopy(after[0]['observation']['farms'][side]['tiles'][y][x])
    inventories = after[side]['observation']['private']['inventories']
    execution['whole_step_unit_inventory'] = deepcopy(inventories[unit]) if unit < len(inventories) else None
    return execution


def terminal(frame, replay, side):
    require(frame[0]['observation']['step'] == TURNS
        and [state['status'] for state in frame] == ['DONE','DONE'], 'Continuation did not reach DONE')
    cash = [farm['money'] for farm in frame[0]['observation']['farms']]
    initial = [farm['money'] for farm in replay['steps'][0][0]['observation']['farms']]
    rewards = [state['reward'] for state in frame]
    require(len(cash) == len(initial) == 2
        and all(type(value) in (int,float) and math.isfinite(value) for value in (*cash,*initial,*rewards))
        and rewards == cash, 'Terminal rewards and cash differ')
    profits = [cash[i]-initial[i] for i in (0,1)]
    margin = cash[side]-cash[1-side]
    return dict(initial_cash=initial,final_cash=cash,rewards=rewards,net_profit=profits,
        own_net_profit=profits[side],opponent_net_profit=profits[1-side],margin=margin,
        score=1. if margin>0 else .5 if margin==0 else 0.)


def _run_arm(replay, trace_factory, target, actor_factory, rival_factory, runtime,
             target_trace_factory, meter, guard, emit, arm, prior=None, harvest_proof=None):
    meter.start('arm');guard()
    meter.start('actor_construct');actor=actor_factory(arm);meter.returned('actor_construct')
    runtime.validate_actor(actor)
    require(actor.arm == arm and actor.budget.limits == GAME_LIMITS and not any(_counts(actor).values()))
    _same_target(actor.target,target)
    meter.start('rival_construct');rival=rival_factory();meter.returned('rival_construct')
    require(callable(rival))
    if prior is not None:
        require(actor is not prior[0] and rival is not prior[1]
            and actor.network is not prior[0].network
            and actor.controller.crop.network is not prior[0].controller.crop.network,
            'Arms share a policy or model instance')
    selector = actor.controller.task_selector
    side,start = target['side'],target['step']
    context=trace_factory();entered=False;trace_lines=0
    try:
        meter.start('trace_open');traces=iter(context.__enter__());entered=True;meter.returned('trace_open')
        def read_trace():
            nonlocal trace_lines
            value=next(traces)
            require(trace_lines<TURNS and type(value) is dict and type(value.get('step')) is int
                and value['step']==trace_lines, 'Source trace is discontinuous or too long')
            trace_lines+=1
            return value
        _emit(emit,meter,dict(kind='arm_start',arm=arm,step=start,frame=deepcopy(replay['steps'][start])))
        for step in range(start):
            guard();before,after=replay['steps'][step:step+2]
            request,trace,budget_before,budget_after=_actor_call(actor,
                _observation(runtime,before,side,step),replay['configuration'],meter,'prefix')
            other=_rival_call(rival,_observation(runtime,before,1-side,step),replay['configuration'],meter,'prefix')
            stored=read_trace()
            require(actor.controller.task_selector is selector and actor.target_proof is None and actor.override_count==0)
            require(runtime.compare(request,after[side]['action']) and runtime.compare(other,after[1-side]['action'])
                and ordered_hash(runtime.stable(trace))==ordered_hash(runtime.stable(stored)), 'Prefix reproduction differs')
            _emit(emit,meter,dict(kind='turn',arm=arm,scope='prefix',step=step,request=request,rival_request=other,
                trace=trace,target_execution=None,budget_before=budget_before,budget_after=budget_after,
                stable_trace_sha256=ordered_hash(runtime.stable(trace))))
        prefix=_counts(actor);require(prefix['controller_calls']==start);guard()
        environment=_restore(replay,start,runtime,meter)
        requests=[];frames=[frame_digest(environment.state,physical=True)]
        proof=effect=None
        for step in range(start,TURNS):
            guard();before=deepcopy(environment.state)
            require(all(state['status']=='ACTIVE' for state in before), 'Continuation ended early')
            request,trace,budget_before,budget_after=_actor_call(actor,
                _observation(runtime,before,side,step),replay['configuration'],meter,'suffix')
            require(actor.controller.task_selector is selector)
            other=_rival_call(rival,_observation(runtime,before,1-side,step),replay['configuration'],meter,'suffix')
            stored=read_trace()  # W future source is authenticated, never fed into its actor.
            if arm=='H':
                require(runtime.compare(request,replay['steps'][step+1][side]['action'])
                    and runtime.compare(other,replay['steps'][step+1][1-side]['action'])
                    and ordered_hash(runtime.stable(trace))==ordered_hash(runtime.stable(stored)),
                    'Factual suffix request or complete trace differs')
            if step==start:
                proof=verify_proof(actor.target_proof,target,arm)
                record=trace['records'][target['ordinal']]
                require(ordered_hash(record['inputs'])==target['input_sha256']
                    and record['chosen']==proof['actual_choice'] and record['unit']==target['unit']
                    and record['unit_request']==proof['unit_request']
                    and ordered_hash(record['task_model'])==ordered_hash(proof['original_proposal']))
                if arm=='W':
                    same_target_proofs(harvest_proof,proof)
                    require(runtime.compare(other,replay['steps'][step+1][1-side]['action']),
                            'Rival changed before the target official transition')
            else:
                require(ordered_hash(actor.target_proof)==ordered_hash(proof), 'Target proof changed later')
            require(actor.override_count==int(arm=='W'))
            joint=[None,None];joint[side],joint[1-side]=request,other
            event=dict(kind='turn',arm=arm,scope='suffix',step=step,request=deepcopy(request),
                rival_request=other,trace=trace,frame=None,target_execution=None,
                budget_before=budget_before,budget_after=budget_after,
                stable_trace_sha256=ordered_hash(runtime.stable(trace)),transition_returned=False,error_kind=None)
            failure=None;tracer=None
            try:
                if step==start:
                    transition_error=None
                    try:
                        with target_trace_factory(environment) as tracer:
                            try:
                                meter.start('official_step','suffix');environment.step(deepcopy(joint))
                                meter.returned('official_step','suffix');event['transition_returned']=True
                            except BaseException as error:
                                transition_error=error
                                raise
                    except BaseException as error:
                        if transition_error is not None and error is not transition_error:
                            raise transition_error from error
                        raise
                    require(environment.interpreter is runtime.interpreter, 'Target trace hooks were not restored')
                else:
                    meter.start('official_step','suffix');environment.step(deepcopy(joint))
                    meter.returned('official_step','suffix');event['transition_returned']=True
                guard();after=environment.state
                require(after[0]['observation']['step']==step+1
                    and not any(state['status'] in ('ERROR','INVALID','TIMEOUT') for state in after))
                require(all(runtime.compare(after[i]['action'],joint[i]) for i in (0,1)), 'Official accepted request differs')
                if arm=='H':
                    require(frame_digest(after)==frame_digest(replay['steps'][step+1]), 'Factual state/order/reward differs')
                if step==start:
                    effect=_execution_effect(tracer.records,target,request,before,after,arm)
                    event['target_execution']=deepcopy(effect)
            except BaseException as error:
                failure=error
            try:
                event['frame']=deepcopy(environment.state)
                if tracer is not None:event['target_execution_records']=deepcopy(tracer.records)
                event['error_kind']=None if failure is None else type(failure).__name__
                _emit(emit,meter,event)
            except BaseException as error:
                if failure is None:raise
                raise failure from error
            if failure is not None:raise failure
            requests.append([request_digest(action) for action in joint])
            frames.append(frame_digest(environment.state,physical=True))
        try:read_trace()
        except StopIteration:pass
        else:raise ValueError('Source trace has extra rows')
        require(trace_lines==TURNS and _counts(actor)['controller_calls']==TURNS and proof is not None and effect is not None)
        require(len(environment.steps)==TURNS+1, 'Official continuation history length differs')
        ending=terminal(environment.state,replay,side)
        total=_counts(actor)
        result=dict(arm=arm,prefix_turns=start,suffix_turns=TURNS-start,policy_turns=TURNS,
            trace_rows_verified=trace_lines,prefix_budget=prefix,
            suffix_budget={key:total[key]-prefix[key] for key in total},episode_budget=total,
            target_proof=proof,target_execution=effect,terminal=ending,
            request_sha256=requests,physical_state_sha256=frames,
            factual_states_verified=TURNS-start+1 if arm=='H' else 0,
            complete_source_requests_and_traces_verified=TURNS if arm=='H' else start,
            override_count=actor.override_count)
    finally:
        active=sys.exc_info()
        if entered:
            meter.start('trace_close')
            try:
                context.__exit__(*active)
                meter.returned('trace_close')
            except BaseException as error:
                meter.calls['trace_close_failures']+=1
                if active[1] is None:raise
    # H's EOF, close and successful final evidence emission precede W creation.
    _emit(emit,meter,dict(kind='arm_end',arm=arm,result=deepcopy(result),frame=deepcopy(environment.state)))
    meter.returned('arm')
    return result,(actor,rival)


def run_pair(replay, trace_factory, *, target, actor_factory, rival_factory,
             runtime=None, meter, guard, emit, target_trace_factory=None):
    target=validate_target(target)
    require(type(meter) is Meter and meter.target_step==target['step'] and not any(meter.calls.values())
        and all(callable(value) for value in (trace_factory,actor_factory,rival_factory,guard,emit)))
    require(type(replay) is dict and replay.get('name')=='kaggriculture'
        and type(replay.get('steps')) is list and len(replay['steps'])==720
        and type(replay.get('configuration')) is dict and replay['configuration'].get('episodeSteps')==720
        and replay['configuration'].get('seed') is None and type(replay.get('info')) is dict
        and type(replay['info'].get('seed')) is int and 0 <= replay['info']['seed'] < 2**32
        and all(state.get('status')=='DONE' for state in replay['steps'][-1]))
    guard()
    runtime=official_runtime() if runtime is None else runtime
    require(type(runtime) is Runtime and callable(runtime.compare))
    target_trace_factory=runtime.target_trace_factory if target_trace_factory is None else target_trace_factory
    require(callable(target_trace_factory))
    with _environment_counts(runtime,meter):
        harvest,prior=_run_arm(replay,trace_factory,target,actor_factory,rival_factory,runtime,
            target_trace_factory,meter,guard,emit,'H')
        guard()
        water,_=_run_arm(replay,trace_factory,target,actor_factory,rival_factory,runtime,
            target_trace_factory,meter,guard,emit,'W',prior,harvest['target_proof'])
    same_target_proofs(harvest['target_proof'],water['target_proof']);meter.complete()
    fields=('own_net_profit','opponent_net_profit','margin','score')
    difference={key:harvest['terminal'][key]-water['terminal'][key] for key in fields}
    require(difference['margin']==difference['own_net_profit']-difference['opponent_net_profit'])
    return dict(passed=True,scope='one consumed-state selector choice to official terminal return; no training label admission',
        arms=dict(H=harvest,W=water),target_pair=dict(H=harvest['target_proof'],W=water['target_proof']),
        terminal_delta_H_minus_W=difference,
        target_conflict=difference['own_net_profit']*difference['margin']<0,
        meter=meter.snapshot(),runtime_identity=deepcopy(runtime.identity),
        complete_games=0,terminal_suffixes=2,extra_official_replays=0,training=False)
