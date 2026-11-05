"""Thin two-arm continuation of authenticated historical matches.

No data/model/engine is loaded on import. Runtime dependencies are injected for
synthetic checks; official_runtime() loads the existing pinned engine only when
explicitly reached by a caller. The caller owns source authentication, process
limits, private storage and the preregistered four-point selection.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import sys

from decision.semantics import canonical
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from learning.task_request_gate_v1.policy import request_digest, same_request
from learning.task_teacher_counterfactual_pair_v1.gate import BUDGET_KEYS
from .gate import HarvestSelector, require, validate_target


def _counts(actor):
    value = dict(actor.budget.counts)
    require(set(value) == BUDGET_KEYS and all(type(x) is int and x >= 0 for x in value.values()),
            'Actor counters differ')
    return value


class Meter:
    """Bounded aggregate accounting; attempted calls are distinct from returns."""
    def __init__(self):
        self.calls = Counter()
        self.costs = {scope: Counter({k: 0 for k in BUDGET_KEYS}) for scope in ('prefix', 'suffix')}
        self.unknown_budget_observations = 0

    def start(self, operation, scope=None):
        if operation == 'arm':
            require(self.calls['arm_attempts'] < 8, 'Eight-arm budget exhausted')
        if operation == 'official_step':
            require(self.calls['official_step_attempts'] < 384, 'Official transition budget exhausted')
        self.calls[operation + '_attempts'] += 1
        if scope is not None:
            require(scope in self.costs)
            self.calls[scope + '_' + operation + '_attempts'] += 1

    def returned(self, operation, scope=None):
        self.calls[operation + '_returns'] += 1
        if scope is not None:
            self.calls[scope + '_' + operation + '_returns'] += 1

    def observe_budget(self, scope, before, after):
        require(scope in self.costs and set(before) == set(after) == BUDGET_KEYS)
        delta = {key: after[key] - before[key] for key in BUDGET_KEYS}
        require(all(type(v) is int and v >= 0 for v in delta.values()), 'Actor budget moved backwards')
        self.costs[scope].update(delta)
        total = self.costs['prefix'] + self.costs['suffix']
        require(total['task_forwards'] <= 32000 and total['crop_forwards'] <= 64000
                and total['task_rows'] + total['crop_rows'] <= 1280000
                and total['controller_calls'] <= 5752, 'Aggregate actor budget exhausted')

    def snapshot(self):
        return dict(calls=dict(self.calls), budget_by_scope={k: dict(v) for k, v in self.costs.items()},
                    unknown_budget_observations=self.unknown_budget_observations,
                    counter_scope='original budget charges; call attempts and returns separately')


@dataclass(frozen=True)
class Runtime:
    make: object
    interpreter: object
    observation: object
    stable: object
    validate_actor: object


def official_runtime():
    """Lazy binding, deliberately not used by the synthetic regression suite."""
    from pathlib import Path
    from learning.task_student_continuation_v1 import official_step
    from learning.task_dagger_once_v1.policy import BoundedAgent
    from decision.audit_market_teacher import observation
    from learning.bc_crop.audit import stable

    require(hashlib.sha256(Path(official_step.rules.__file__).read_bytes()).hexdigest()
            == official_step.ENGINE_SHA, 'Official rules changed')
    require(hashlib.sha256(Path(official_step.official_core.__file__).read_bytes()).hexdigest()
            == official_step.ENGINE_CORE_SHA256, 'Official core changed')

    def validate(actor):
        require(type(actor) is BoundedAgent and actor.trace is None
                and actor.controller.crop.trace is None and actor.network is not None
                and actor.budget.limits == GAME_LIMITS and actor.previous is None,
                'Expected a fresh frozen BoundedAgent')

    return Runtime(official_step.make, official_step.rules.interpreter, observation, stable, validate)


def frame_digest(frame, *, physical=False):
    """v2 unit-inventory order plus explicit shed order, without timing noise."""
    require(isinstance(frame, list) and len(frame) == 2)
    value = ([{key: player[key] for key in ('observation', 'reward', 'status')}
              for player in frame] if physical else frame)
    sheds = []
    for player in frame:
        shed = player['observation']['private']['shed']
        require(isinstance(shed, dict))
        sheds.append([[key, item] for key, item in shed.items()])
    raw = json.dumps(dict(frame=canonical(value), shed_order=sheds), sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _request(value):
    require(type(value) is dict and set(value) == {'farmer', 'hands', 'market'}
            and all(type(value[key]) is list for key in value), 'Incomplete request')
    request_digest(value)
    return value


def _observation(runtime, frame, side, step):
    observation = runtime.observation(deepcopy(frame), side)
    require(isinstance(observation, dict) and type(observation.get('player')) is int
            and observation['player'] == side and type(observation.get('step')) is int
            and observation['step'] == step
            and not any(key in observation for key in ('info', 'steps', 'observation')),
            'Invalid single-player observation')
    require(ordered_hash(observation['private']) == ordered_hash(frame[side]['observation']['private']),
            'Own private observation differs')
    return observation


def _actor_call(actor, observation, configuration, runtime, meter, scope, gate):
    before = _counts(actor)
    meter.start('actor', scope)
    primary = None
    try:
        request = _request(actor(observation, deepcopy(configuration)))
        meter.returned('actor', scope)
        gate.verify_turn(actor.last)
        require(all(actor.last.get(k) == 0 for k in ('fallbacks', 'load_errors', 'inference_errors')))
        return deepcopy(request), deepcopy(actor.last)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            meter.observe_budget(scope, before, _counts(actor))
        except BaseException:
            meter.unknown_budget_observations += 1
            if primary is None:
                raise


def _rival_call(rival, observation, configuration, meter, scope):
    meter.start('rival', scope)
    request = _request(rival(observation, deepcopy(configuration)))
    meter.returned('rival', scope)
    return deepcopy(request)


def _restore(replay, step, runtime, meter):
    before = replay['steps'][step]
    meter.start('restore')
    environment = runtime.make('kaggriculture', configuration=deepcopy(replay['configuration']),
        info=deepcopy(replay['info']), steps=[None] * step + [deepcopy(before)], debug=False)
    meter.returned('restore')
    require(environment.interpreter is runtime.interpreter and len(environment.steps) == step + 1
            and frame_digest(environment.state) == frame_digest(before), 'Restored state differs')
    require(canonical(environment.configuration) == canonical(replay['configuration'])
            and environment.info == replay['info'], 'Restored metadata differs')
    return environment


def _same_target_proof(control, intervention):
    require(control['mode'] == 'control' and intervention['mode'] == 'intervene')
    for key in ('step', 'ordinal', 'input_sha256', 'original_choice', 'harvest_choice',
                'harvest_unit', 'budget_before', 'budget_after', 'teacher_label'):
        require(ordered_hash(control[key]) == ordered_hash(intervention[key]), 'Arm target proofs differ')


def _run_arm(replay, trace_factory, target, actor_factory, rival_factory, runtime,
             meter, guard, emit, mode, prior=None, factual_proof=None):
    meter.start('arm')
    meter.start('actor_construct')
    actor = actor_factory()
    meter.returned('actor_construct')
    runtime.validate_actor(actor)
    require(actor.budget.limits == GAME_LIMITS and not any(_counts(actor).values()))
    meter.start('rival_construct')
    rival = rival_factory()
    meter.returned('rival_construct')
    require(callable(rival))
    if prior is not None:
        require(actor is not prior[0] and rival is not prior[1]
                and actor.network is not prior[0].network
                and actor.controller.crop.network is not prior[0].controller.crop.network,
                'Arms share an actor or model instance')
    original = actor.controller.task_selector
    gate = HarvestSelector(original, lambda: len(actor.controller.records),
        lambda: dict(total=dict(actor.budget.counts), turn=dict(actor.budget.turn)), target, mode)
    actor.controller.task_selector = gate
    side, start = target['side'], target['step']
    trace_context = trace_factory()
    entered = False
    try:
        traces = iter(trace_context.__enter__())
        entered = True
        emit(dict(kind='arm_start', arm=mode, step=start, frame=deepcopy(replay['steps'][start])))
        for step in range(start):
            guard()
            before, after = replay['steps'][step:step + 2]
            request, trace = _actor_call(actor, _observation(runtime, before, side, step),
                replay['configuration'], runtime, meter, 'prefix', gate)
            other = _rival_call(rival, _observation(runtime, before, 1-side, step),
                replay['configuration'], meter, 'prefix')
            stored = next(traces)
            require(same_request(request, after[side]['action'])
                    and same_request(other, after[1-side]['action']), 'Prefix request differs')
            require(ordered_hash(runtime.stable(trace)) == ordered_hash(runtime.stable(stored)),
                    'Prefix trace differs')
            emit(dict(kind='turn', arm=mode, scope='prefix', step=step,
                      request=request, rival_request=other, trace=trace,
                      stable_trace_sha256=ordered_hash(runtime.stable(trace))))
        prefix_counts = _counts(actor)
        require(prefix_counts['controller_calls'] == start)
        guard()
        environment = _restore(replay, start, runtime, meter)
        frame_hashes = [frame_digest(environment.state, physical=True)]
        request_hashes = []
        for offset in range(48):
            guard()
            step = start + offset
            before = deepcopy(environment.state)
            require(all(player['status'] == 'ACTIVE' for player in before), 'Window ended early')
            request, trace = _actor_call(actor, _observation(runtime, before, side, step),
                replay['configuration'], runtime, meter, 'suffix', gate)
            other = _rival_call(rival, _observation(runtime, before, 1-side, step),
                replay['configuration'], meter, 'suffix')
            joint = [None, None]
            joint[side], joint[1-side] = request, other
            if mode == 'control':
                stored = next(traces)
                require(ordered_hash(runtime.stable(trace)) == ordered_hash(runtime.stable(stored)),
                        'Factual suffix trace differs')
                require(all(same_request(joint[i], replay['steps'][step+1][i]['action'])
                            for i in (0, 1)), 'Factual suffix request differs')
            elif offset == 0:
                stored = next(traces)
                require(ordered_hash(runtime.stable(trace['records'][:target['ordinal']]))
                        == ordered_hash(runtime.stable(stored['records'][:target['ordinal']])),
                        'Pre-target callback history differs')
                _same_target_proof(factual_proof, gate.proof())
            meter.start('official_step', 'suffix')
            environment.step(deepcopy(joint))
            meter.returned('official_step', 'suffix')
            guard()
            after = environment.state
            require(not any(p['status'] in ('ERROR', 'INVALID', 'TIMEOUT') for p in after),
                    'Official transition failed')
            require(all(same_request(after[i]['action'], joint[i]) for i in (0, 1)),
                    'Official accepted request differs')
            if mode == 'control':
                require(frame_digest(after) == frame_digest(replay['steps'][step+1]),
                        'Factual suffix state or order differs')
            request_hashes.append(tuple(request_digest(x) for x in joint))
            frame_hashes.append(frame_digest(after, physical=True))
            emit(dict(kind='turn', arm=mode, scope='suffix', step=step,
                request=request, rival_request=other, trace=trace, frame=deepcopy(after),
                stable_trace_sha256=ordered_hash(runtime.stable(trace))))
        proof = gate.finish()
        total = _counts(actor)
        require(total['controller_calls'] == start + 48)
        result = dict(arm=mode, prefix_turns=start, suffix_turns=48,
            prefix_budget=prefix_counts, suffix_budget={k: total[k]-prefix_counts[k] for k in total},
            episode_budget=total, target_proof=proof, request_sha256=request_hashes,
            physical_state_sha256=frame_hashes,
            factual_states_verified=49 if mode == 'control' else 0)
        emit(dict(kind='arm_end', arm=mode, result=deepcopy(result), frame=deepcopy(environment.state)))
    finally:
        actor.controller.task_selector = original
        active = sys.exc_info()
        if entered:
            try:
                trace_context.__exit__(*active)
            except BaseException:
                meter.calls['trace_close_failures'] += 1
                if active[1] is None:
                    raise
    meter.returned('arm')
    return result, (actor, rival)


def run_pair(replay, trace_factory, *, target, actor_factory, rival_factory,
             runtime=None, meter, guard, emit, horizon=48):
    """Run exactly two bounded windows; all inputs/emitted records stay private.

    trace_factory must return a fresh context manager yielding decoded historical
    trace dictionaries in step order. emit is a trusted private sink, never a
    policy input. It receives full suffix frames but need only hash prefix data.
    The caller retains meter on failures. No return/cash labels are computed.
    """
    target = validate_target(target)
    require(type(horizon) is int and horizon == 48 and isinstance(meter, Meter)
            and all(callable(x) for x in (trace_factory, actor_factory, rival_factory, guard, emit)))
    require(type(replay) is dict and replay.get('name') == 'kaggriculture'
            and type(replay.get('steps')) is list and len(replay['steps']) == 720
            and type(replay.get('configuration')) is dict
            and replay['configuration'].get('episodeSteps') == 720
            and replay['configuration'].get('seed') is None and type(replay.get('info')) is dict
            and type(replay['info'].get('seed')) is int and 0 <= replay['info']['seed'] < 2**32
            and all(p.get('status') == 'DONE' for p in replay['steps'][-1]), 'Incomplete authenticated source')
    guard()
    runtime = official_runtime() if runtime is None else runtime
    require(isinstance(runtime, Runtime))
    control, prior = _run_arm(replay, trace_factory, target, actor_factory, rival_factory,
        runtime, meter, guard, emit, 'control')
    guard()
    intervention, _ = _run_arm(replay, trace_factory, target, actor_factory, rival_factory,
        runtime, meter, guard, emit, 'intervene', prior, control['target_proof'])
    request_changed = [left != right for left, right in zip(control['request_sha256'],
                                                           intervention['request_sha256'])]
    physical_changed = [left != right for left, right in zip(control['physical_state_sha256'],
                                                            intervention['physical_state_sha256'])]
    return dict(passed=True, scope='48-step historical-development probe; no terminal-return claim',
        control=control, intervention=intervention, request_changed_by_offset=request_changed,
        physical_changed_by_offset=physical_changed, meter=meter.snapshot(),
        complete_games=0, training=False, extra_official_replays=0)
