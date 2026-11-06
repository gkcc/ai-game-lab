"""Bounded offline request patch over two freshly reconstructed frozen policies.

Original policy intent is never rewritten: trace and original_request are kept
apart from request (the actual submission). Only one submitted unit action is
patched, after a complete factual 48-step continuation and authenticated EOF.
Importing this module loads no checkpoint, private source, or official engine.
"""
from copy import deepcopy
import sys

from learning.task_harvest_window_v1.core import (
    BUDGET_KEYS, GAME_LIMITS, Meter, Runtime, _counts, _observation, _request,
    _restore, _rival_call, frame_digest, official_runtime, ordered_hash,
    request_digest, same_request,
)
from learning.task_harvest_window_v1.gate import require
from .adapter import replace_harvest
from .selection import validate_target, validate_target_trace


def _actor_call(actor, observation, configuration, meter, scope):
    """Run the unchanged actor once and account for failures as well as success."""
    before = _counts(actor)
    meter.start('actor', scope)
    primary = None
    try:
        request = _request(actor(observation, deepcopy(configuration)))
        meter.returned('actor', scope)
        after, turn = _counts(actor), dict(actor.budget.turn)
        trace = actor.last
        require(type(trace) is dict and trace.get('step') == observation['step']
                and type(trace.get('records')) is list and type(trace.get('crop_records')) is list,
                'Original actor trace is incomplete')
        require(all(type(trace.get(key)) is int and trace[key] == 0
                    for key in ('fallbacks', 'load_errors', 'inference_errors')),
                'Original actor reported an error')
        require(set(turn) == BUDGET_KEYS and all(type(n) is int and n >= 0 for n in turn.values())
                and turn == {key: after[key]-before[key] for key in BUDGET_KEYS}
                and turn['controller_calls'] == 1,
                'Original turn budget differs from actual cumulative charges')
        require(ordered_hash(trace.get('model_counts')) == ordered_hash(after)
                and ordered_hash(trace.get('turn_model_counts')) == ordered_hash(turn),
                'Original trace budget differs from actual actor budget')
        actions = [request['farmer'], *request['hands']]
        require(type(trace.get('final_reservations')) is dict
                and ordered_hash(trace['final_reservations'].get('unit_actions')) == ordered_hash(actions),
                'Original trace unit requests differ from original complete request')
        for record in trace['records']:
            unit = record.get('unit')
            require(type(unit) is int and 0 <= unit < len(actions)
                    and record.get('source') == 'callback'
                    and ordered_hash(record.get('unit_request')) == ordered_hash(actions[unit]),
                    'Original task record differs from original complete request')
        return (deepcopy(request), deepcopy(trace), dict(total=before),
                dict(total=after, turn=turn))
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


def _official_target_trace(environment):
    """Existing tracer around the same target step; no extra interpretation."""
    from decision.execution import ExecutionTrace
    from learning.task_student_continuation_v1.official_step import rules
    return ExecutionTrace(rules).installed(environment)


def _execution(records, target, submitted):
    """Retain an atomic official unit record, including a blocked/no-effect result."""
    require(type(records) is list, 'Target execution records are missing')
    selected = [record for record in records if record.get('phase') == 'unit'
                and record.get('player') == target['side'] and record.get('unit') == target['unit']]
    require(len(selected) == 1, 'Expected exactly one target unit execution record')
    record = selected[0]
    actions = [submitted['farmer'], *submitted['hands']]
    require(record.get('step') == target['step']
            and ordered_hash(record.get('requested')) == ordered_hash(actions[target['unit']])
            and record.get('position') == target['target']
            and type(record.get('effective')) is list
            and record.get('status') in ('executed', 'blocked', 'no_effect', 'partial', 'pass')
            and type(record.get('before')) is dict and type(record.get('after')) is dict
            and all(type(record[when].get('inventories')) is list
                    and len(record[when]['inventories']) > target['unit'] for when in ('before', 'after'))
            and 'tile_before' in record and 'tile_after' in record,
            'Target execution record differs from submitted action or coordinate')
    return deepcopy(record)


def _endpoint(frame, target):
    """Private factual snapshots only; inventory is not priced into cash."""
    farms = frame[0]['observation']['farms']
    players = []
    x, y = target['target']
    for side, farm in enumerate(farms):
        private = frame[side]['observation']['private']
        plants = [dict(target=[tx, ty], tile=deepcopy(tile))
                  for ty, row in enumerate(farm['tiles']) for tx, tile in enumerate(row)
                  if isinstance(tile, dict) and tile.get('kind') == 'PLANT']
        players.append(dict(side=side,
            inventories=[[[key, value] for key, value in inv.items()] for inv in private['inventories']],
            shed=[[key, value] for key, value in private['shed'].items()],
            seeds=[[key, value] for key, value in private['seeds'].items()],
            live_crops=plants, target_tile=deepcopy(farm['tiles'][y][x])))
    return dict(step=frame[0]['observation']['step'], cash=[farm['money'] for farm in farms], players=players)


def _proof(mode, target, original, trace, runtime, before, after):
    return dict(mode=mode,
        **{key: deepcopy(target[key]) for key in ('side', 'step', 'ordinal', 'unit', 'target',
             'input_sha256', 'unit_requests_sha256', 'original_unit_request')},
        original_request_sha256=ordered_hash(original),
        stable_trace_sha256=ordered_hash(runtime.stable(trace)),
        budget_before=deepcopy(before), budget_after=deepcopy(after), patch=None)


def _same_target_proof(control, intervention):
    require(control.get('mode') == 'control' and intervention.get('mode') == 'intervene'
            and set(control) == set(intervention) and control.get('patch') is None,
            'Arm target proof schema differs')
    for key in control:
        if key not in ('mode', 'patch'):
            require(ordered_hash(control[key]) == ordered_hash(intervention[key]),
                    'Arm target original request, trace or budget differs')


def _run_arm(replay, trace_factory, target, first_yield_days, actor_factory, rival_factory,
             runtime, target_trace_factory, meter, guard, emit, mode, prior=None, factual_proof=None):
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
    selector = actor.controller.task_selector
    side, start = target['side'], target['step']
    trace_context = trace_factory()
    entered, trace_lines = False, 0
    try:
        traces = iter(trace_context.__enter__())
        entered = True

        def read_trace():
            nonlocal trace_lines
            stored = next(traces)
            require(trace_lines < 719 and type(stored) is dict
                    and type(stored.get('step')) is int and stored['step'] == trace_lines,
                    'Historical trace is discontinuous or has extra rows')
            trace_lines += 1
            return stored

        emit(dict(kind='arm_start', arm=mode, step=start, frame=deepcopy(replay['steps'][start])))
        for step in range(start):
            guard()
            before, after = replay['steps'][step:step+2]
            request, trace, _budget_before, _budget_after = _actor_call(actor,
                _observation(runtime, before, side, step), replay['configuration'], meter, 'prefix')
            require(actor.controller.task_selector is selector, 'Original selector was replaced')
            other = _rival_call(rival, _observation(runtime, before, 1-side, step),
                                replay['configuration'], meter, 'prefix')
            stored = read_trace()
            require(same_request(request, after[side]['action'])
                    and same_request(other, after[1-side]['action']), 'Prefix request differs')
            require(ordered_hash(runtime.stable(trace)) == ordered_hash(runtime.stable(stored)),
                    'Prefix trace differs')
            emit(dict(kind='turn', arm=mode, scope='prefix', step=step, request=request,
                original_request=deepcopy(request), patch=None, target_execution=None,
                rival_request=other, trace=trace, stable_trace_sha256=ordered_hash(runtime.stable(trace))))
        prefix_counts = _counts(actor)
        require(prefix_counts['controller_calls'] == start)
        guard()
        environment = _restore(replay, start, runtime, meter)
        endpoints = dict(start=_endpoint(environment.state, target))
        frame_hashes = [frame_digest(environment.state, physical=True)]
        request_hashes, proof, target_execution = [], None, None
        for offset in range(48):
            guard()
            step = start + offset
            before = deepcopy(environment.state)
            require(all(player['status'] == 'ACTIVE' for player in before), 'Window ended early')
            original, trace, budget_before, budget_after = _actor_call(actor,
                _observation(runtime, before, side, step), replay['configuration'], meter, 'suffix')
            require(actor.controller.task_selector is selector, 'Original selector was replaced')
            other = _rival_call(rival, _observation(runtime, before, 1-side, step),
                                replay['configuration'], meter, 'suffix')
            request, patch = deepcopy(original), None
            if mode == 'control' or offset == 0:
                stored = read_trace()
                require(ordered_hash(runtime.stable(trace)) == ordered_hash(runtime.stable(stored)),
                        'Factual original suffix trace differs')
                require(same_request(original, replay['steps'][step+1][side]['action'])
                        and same_request(other, replay['steps'][step+1][1-side]['action']),
                        'Factual original suffix request differs')
            if offset == 0:
                validate_target_trace(target, trace, first_yield_days=first_yield_days)
                proof = _proof(mode, target, original, trace, runtime, budget_before, budget_after)
                if mode == 'intervene':
                    _same_target_proof(factual_proof, proof)
                    original_identity, trace_identity = ordered_hash(original), ordered_hash(trace)
                    meter.start('request_patch', 'suffix')
                    replacement = replace_harvest(original, target=target, trace=trace,
                                                   first_yield_days=first_yield_days)
                    meter.returned('request_patch', 'suffix')
                    require(ordered_hash(original) == original_identity
                            and ordered_hash(trace) == trace_identity
                            and same_request(replacement['original'], original),
                            'Request patch mutated original policy intent')
                    request, patch = _request(replacement['submitted']), deepcopy(replacement['change'])
                    proof['patch'] = deepcopy(patch)
            joint = [None, None]
            joint[side], joint[1-side] = request, other
            execution = None
            if offset == 0:
                with target_trace_factory(environment) as tracer:
                    meter.start('official_step', 'suffix')
                    environment.step(deepcopy(joint))
                    meter.returned('official_step', 'suffix')
                require(environment.interpreter is runtime.interpreter, 'Target execution hooks were not restored')
                execution = _execution(tracer.records, target, request)
                target_execution = deepcopy(execution)
            else:
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
            if offset == 0:
                endpoints['after_target'] = _endpoint(after, target)
            request_hashes.append(tuple(request_digest(x) for x in joint))
            frame_hashes.append(frame_digest(after, physical=True))
            emit(dict(kind='turn', arm=mode, scope='suffix', step=step, request=deepcopy(request),
                original_request=deepcopy(original), patch=patch, target_execution=execution,
                rival_request=other, trace=trace, frame=deepcopy(after),
                stable_trace_sha256=ordered_hash(runtime.stable(trace))))
        # Explicit EOF is needed even for the intervention, whose future old
        # trace is evidence to authenticate, never a source of future actions.
        while True:
            guard()
            try:
                read_trace()
            except StopIteration:
                break
        require(trace_lines == 719, 'Historical trace ended before all 719 rows')
        total = _counts(actor)
        require(total['controller_calls'] == start + 48 and proof is not None)
        endpoints['end'] = _endpoint(environment.state, target)
        result = dict(arm=mode, prefix_turns=start, suffix_turns=48,
            prefix_budget=prefix_counts, suffix_budget={k: total[k]-prefix_counts[k] for k in total},
            episode_budget=total, target_proof=proof, target_execution=target_execution,
            endpoints=endpoints, request_sha256=request_hashes, physical_state_sha256=frame_hashes,
            factual_states_verified=49 if mode == 'control' else 0, trace_rows_verified=trace_lines)
    finally:
        active = sys.exc_info()
        if entered:
            try:
                trace_context.__exit__(*active)
            except BaseException:
                meter.calls['trace_close_failures'] += 1
                if active[1] is None:
                    raise
    # A successful arm is not emitted until source EOF and its context close
    # both succeed. On failure the caller retains attempts and partial events.
    emit(dict(kind='arm_end', arm=mode, result=deepcopy(result), frame=deepcopy(environment.state)))
    meter.returned('arm')
    return result, (actor, rival)


def run_pair(replay, trace_factory, *, target, first_yield_days, actor_factory, rival_factory,
             runtime=None, meter, guard, emit, horizon=48, target_trace_factory=None):
    """Run factual then one request intervention; all outputs are private facts."""
    target = validate_target(target)
    require(type(horizon) is int and horizon == 48 and isinstance(meter, Meter)
            and all(callable(x) for x in (trace_factory, actor_factory, rival_factory, guard, emit)))
    require(type(first_yield_days) is dict and first_yield_days
            and all(type(k) is str and type(v) is int and v >= 0 for k, v in first_yield_days.items()),
            'Authenticated official maturity table is required')
    require(type(replay) is dict and replay.get('name') == 'kaggriculture'
            and type(replay.get('steps')) is list and len(replay['steps']) == 720
            and type(replay.get('configuration')) is dict
            and replay['configuration'].get('episodeSteps') == 720
            and replay['configuration'].get('seed') is None and type(replay.get('info')) is dict
            and type(replay['info'].get('seed')) is int and 0 <= replay['info']['seed'] < 2**32
            and all(p.get('status') == 'DONE' for p in replay['steps'][-1]), 'Incomplete authenticated source')
    guard()
    if runtime is None:
        runtime = official_runtime()
        target_trace_factory = _official_target_trace if target_trace_factory is None else target_trace_factory
    require(isinstance(runtime, Runtime) and callable(target_trace_factory),
            'Custom runtimes require an explicit target execution trace context')
    control, prior = _run_arm(replay, trace_factory, target, first_yield_days, actor_factory,
        rival_factory, runtime, target_trace_factory, meter, guard, emit, 'control')
    guard()
    intervention, _ = _run_arm(replay, trace_factory, target, first_yield_days, actor_factory,
        rival_factory, runtime, target_trace_factory, meter, guard, emit, 'intervene', prior, control['target_proof'])
    _same_target_proof(control['target_proof'], intervention['target_proof'])
    return dict(passed=True, scope='48-step consumed-source request probe; no terminal-return or training claim',
        control=control, intervention=intervention,
        request_changed_by_offset=[a != b for a, b in zip(control['request_sha256'], intervention['request_sha256'])],
        physical_changed_by_offset=[a != b for a, b in zip(control['physical_state_sha256'], intervention['physical_state_sha256'])],
        meter=meter.snapshot(), complete_games=0, training=False, extra_official_replays=0)
