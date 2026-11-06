"""Eight bounded official transitions over public constructed input states.

The caller owns identity, freeze, resource/output guards and a single-attempt
directory. This module never writes files or imports/constructs an engine at
import time. Placeholder counters are not replay history. Effects are checked
only after the injected official Environment.step returns.
"""
from copy import deepcopy
import json

from . import fixtures


PASS = {'farmer':['PASS'], 'hands':[], 'market':[]}
COUNT_KEYS = ('runs_attempted', 'runs_returned',
    'initializations_attempted', 'initializations_returned',
    'action_steps_attempted', 'action_steps_returned',
    'agent_constructions_attempted', 'agent_constructions_returned',
    'agent_calls_attempted', 'agent_calls_returned',
    'selector_calls_attempted', 'selector_calls_returned',
    'direct_boundary_requests', 'emissions_attempted', 'emissions_returned',
    'evidence_write_failures')
CAPS = {'runs':1, 'initializations':8, 'action_steps':8,
        'agent_constructions':6, 'agent_calls':6, 'selector_calls':6, 'emissions':8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    # JSON-compatible evidence copy, preserving array and insertion order.
    # No conversion is applied to the actual action submitted to the engine.
    return json.loads(json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False))


def _call(meter, name, guard, function):
    guard()
    require(meter[name+'_attempted'] < CAPS[name], 'Probe operation budget exhausted')
    meter[name+'_attempted'] += 1
    result = function()
    meter[name+'_returned'] += 1
    guard()
    return result


def _install(environment, rules, name):
    """Set public constructed inputs; the interpreter alone advances them."""
    obs, config = fixtures.observation(name)
    require(environment.interpreter is rules.interpreter and len(environment.state) == 2,
            'Unexpected official runtime')
    require(all(environment.configuration[k] == value for k,value in config.items()),
            'Official defaults do not match the frozen fixture')
    # Verify literal _new_plant defaults without creating another environment
    # or transition. The mature fixture deliberately overrides produced yield.
    for crop, planted_day in (('WHEAT', fixtures.DAY), ('MELON', 0), ('MELON', fixtures.DAY)):
        expected = fixtures._plant(crop, planted_day, watered=False, units=1)
        require(plain(rules._new_plant(crop, planted_day, config['turnsPerDay'])) == expected,
                'Official plant construction differs from the fixture schema')
    for side, state in enumerate(environment.state):
        require(state.status == 'ACTIVE', 'Initialized player is not active')
        state.observation.player = side
        for key in ('farms', 'market', 'town', 'day', 'hour'):
            state.observation[key] = obs[key]
        state.observation.private = obs['private'] if side == 0 else fixtures.empty_private()
    environment.state[0].observation.step = fixtures.START_STEP
    # Only framework len(steps) bookkeeping; never expose or publish as replay.
    environment.steps = [None]*fixtures.START_STEP + [deepcopy(environment.state)]
    return config


def _agent_request(runtime, environment, config, name, op, guard, meter):
    captured = []

    def choose(inputs):
        def select():
            require(not any(k in inputs for k in ('teacher_label', 'teacher_priority', 'chosen')),
                    'Teacher reference leaked into selector input')
            observed = inputs['observation']
            require(observed['player'] == 0 and 'seed' not in observed
                    and 'seed' not in inputs['configuration']
                    and not any(k in observed['farms'][1] for k in ('private','seeds','shed','inventories')),
                    'Non-observable information exposed to selector')
            candidates = inputs['candidates']
            require(len({c['id'] for c in candidates}) == len(candidates),
                    'Duplicate candidate identity')
            matches = [c for c in candidates if c['unit'] == 0
                       and c['target'] == list(fixtures.TARGET) and c['op'] == op]
            require(len(matches) == 1, 'Predeclared action is not uniquely selectable')
            captured.append(dict(inputs=deepcopy(inputs), selected_id=matches[0]['id']))
            return matches[0]['id']
        return _call(meter, 'selector_calls', guard, select)

    agent = _call(meter, 'agent_constructions', guard, lambda:runtime.agent_factory(choose))
    require(agent.crop.mode == 'teacher' and agent.crop.network is None,
            'Probe unexpectedly constructed a model')
    request = _call(meter, 'agent_calls', guard,
        lambda:agent(runtime.obs(environment, 0), deepcopy(config)))
    require(request == {'farmer':[op], 'hands':[], 'market':[]}
            and len(captured) == len(agent.last['records']) == 1,
            'Agent request or callback count differs from the fixture')
    record = agent.last['records'][0]
    require(record['chosen'] == captured[0]['selected_id'] and record['op'] == op
            and record['unit_request'] == [op] and not agent.last['crop_records']
            and agent.last['checkpoint_sha256'] is None,
            'Selected candidate was not consumed by the original controller')
    extension = record['harvest_choice_extension']
    count = 2 if name in ('mature5dry', 'mature5dry_full_low') else 1
    added = 1 if count == 2 else 0
    require(len(captured[0]['inputs']['candidates']) == count
            and len(extension['added_candidate_ids']) == added
            and extension['selected_virtual_harvest'] == (added == 1 and op == 'HARVEST'),
            'Bounded candidate expansion differs')
    return request, dict(captured=captured, trace=deepcopy(agent.last)), dict(
        candidate_count=count, added_harvest_count=added,
        selector_calls=1, selected_virtual_harvest=extension['selected_virtual_harvest'])


def _check_effect(name, op, before, after, records, joint):
    """Consume observed official output; never calculate a replacement state."""
    expected = fixtures.expectation(name, op)
    require(after[0]['observation']['step'] == fixtures.START_STEP+1
            and all(s['status'] == 'ACTIVE' for s in after)
            and all((s['observation']['day'],s['observation']['hour']) == (fixtures.DAY,fixtures.HOUR+1)
                    for s in after), 'Official counter, clock or status differs')
    require(all(s['action'] == joint[i] for i,s in enumerate(after)),
            'Official returned request differs')
    matches = [r for r in records if r.get('phase') == 'unit'
               and r.get('player') == 0 and r.get('unit') == 0]
    require(len(matches) == 1 and len(records) == 2
            and all(r.get('phase') == 'unit' for r in records),
            'Unexpected execution events')
    target = matches[0]
    require(target['step'] == fixtures.START_STEP and target['requested'] == target['effective'] == [op]
            and target['status'] == expected['status'], 'Actual atomic execution differs')
    x, y = fixtures.TARGET
    prior = before[0]['observation']
    current = after[0]['observation']
    old_tile, new_tile = prior['farms'][0]['tiles'][y][x], current['farms'][0]['tiles'][y][x]
    require(target['tile_before'] == old_tile and target['tile_after'] == new_tile,
            'Atomic and complete state tile effects differ')
    if expected['target_removed']:
        require(new_tile is None, 'Harvest did not remove the constructed annual crop')
    else:
        require(type(new_tile) is dict and new_tile['kind'] == 'PLANT'
                and new_tile['crop'] == 'MELON' and new_tile['yield_units'] == expected['yield_after']
                and new_tile['watered_today'] is expected['watered_after'],
                'Constructed crop output differs')
    before_units = prior['private']['inventories'][0].get('MELON', 0)
    after_units = current['private']['inventories'][0].get('MELON', 0)
    require(after_units-before_units == expected['harvested_units'],
            'Actual harvested inventory differs')
    require(current['private']['shed'] == prior['private']['shed']
            and current['private']['seeds'] == prior['private']['seeds']
            and all(current['farms'][i]['money'] == prior['farms'][i]['money'] == 0 for i in (0,1)),
            'Unexpected purchase, sale, stock transfer or cash change')
    require(current['market'] == prior['market'] and current['town'] == prior['town'],
            'Unexpected market or town change')
    return dict(execution_status=target['status'], target_removed=new_tile is None,
        yield_before=old_tile['yield_units'], yield_after=None if new_tile is None else new_tile['yield_units'],
        watered_before=old_tile['watered_today'], watered_after=None if new_tile is None else new_tile['watered_today'],
        harvested_inventory_units=after_units-before_units,
        shed_units_before=sum(prior['private']['shed'].values()),
        shed_units_after=sum(current['private']['shed'].values()),
        own_cash_change=0, opponent_cash_change=0)


def _transition(runtime, environment, config, name, op, joint, before,
                agent_evidence, direct, guard, emit, meter):
    """Preserve attempted-step output, without bypassing the caller's writer.

    Resource/storage failures can prevent evidence publication. Such failures
    remain counted, and never replace the original execution exception.
    """
    interpreter = environment.interpreter
    functions = {key:getattr(runtime.rules,key) for key in
        ('_apply_unit_action','_process_market','_parse_order','_commit_unit','_do_hire','_do_buy_land')}
    tracer = runtime.ExecutionTrace(runtime.rules)
    attempted, returned = meter['action_steps_attempted'], meter['action_steps_returned']
    error = secondary = None
    try:
        with tracer.installed(environment):
            try:
                _call(meter, 'action_steps', guard, lambda:environment.step(deepcopy(joint)))
            except BaseException as exc:
                error = exc
                raise
    except BaseException as exc:
        if error is None:
            error = exc
        elif exc is not error:
            secondary = exc
    try:
        require(environment.interpreter is interpreter
                and all(getattr(runtime.rules,key) is value for key,value in functions.items()),
                'Execution tracer did not restore the official runtime')
    except BaseException as exc:
        if error is None:
            error = exc
        else:
            secondary = exc
    if meter['action_steps_attempted'] == attempted:
        # A pre-step guard/trace installation failure did not attempt an action.
        if secondary is not None:
            raise error from secondary
        raise error
    evidence = dict(schema='task-harvest-choice-official-transition-v1', scene=name, op=op,
        selection_source='direct_boundary_nonagent' if direct else 'agent_callback',
        synthetic_not_natural=True, counter_placeholders_not_replay=True,
        configuration=deepcopy(config), before=before, request=plain(joint),
        after=None, execution=None, agent=plain(agent_evidence),
        transition_returned=meter['action_steps_returned'] == returned+1,
        capture_error_kind=None)
    try:
        evidence['after'] = plain(environment.state)
        evidence['execution'] = plain(tracer.records)
    except BaseException as exc:
        evidence['capture_error_kind'] = type(exc).__name__
        if error is None:
            error = exc
        else:
            secondary = exc
    evidence.update(error_kind=None if error is None else type(error).__name__, counts=deepcopy(meter))
    try:
        require(meter['emissions_attempted'] < CAPS['emissions'], 'Probe evidence budget exhausted')
        meter['emissions_attempted'] += 1
        # No outer guard that would skip this attempt after a failed post-step
        # guard. The root's emit still enforces its own guard and output limit.
        emit(name+'-'+op.lower()+'.local.json', evidence)
        meter['emissions_returned'] += 1
    except BaseException as exc:
        meter['evidence_write_failures'] += 1
        if error is None:
            raise
        raise error from exc
    if error is not None:
        if secondary is not None:
            raise error from secondary
        raise error
    return evidence


def run(runtime, guard, emit, meter):
    """Run once after freeze; ``emit(fixed_name, raw_record)`` persists evidence.

    ``meter`` must be a fresh caller-owned empty dict. Attempt/return counters
    survive any exception. This function cannot retry or return partial success.
    """
    require(__debug__ and type(meter) is dict and not meter,
            'Probe requires a fresh single-attempt counter dictionary')
    require(callable(guard) and callable(emit), 'Probe output/resource hooks are missing')
    meter.update({key:0 for key in COUNT_KEYS})
    meter['runs_attempted'] = 1
    rows = []
    for name in fixtures.NAMES:
        for op in fixtures.OPS:
            environment = _call(meter, 'initializations', guard, runtime.env)
            config = _install(environment, runtime.rules, name)
            guard()
            before = plain(environment.state)
            direct = fixtures.direct_boundary(name, op)
            if direct:
                require(meter['direct_boundary_requests'] < 2, 'Direct boundary budget exhausted')
                meter['direct_boundary_requests'] += 1
                request, agent_evidence = {'farmer':[op], 'hands':[], 'market':[]}, None
                selection = dict(candidate_count=None, added_harvest_count=None,
                                 selector_calls=0, selected_virtual_harvest=False)
            else:
                request, agent_evidence, selection = _agent_request(
                    runtime, environment, config, name, op, guard, meter)
            joint = [request, deepcopy(PASS)]
            # Preserve observed output before validating expected effects. A
            # failed assertion remains a failed attempt with its raw evidence.
            evidence = _transition(runtime, environment, config, name, op, joint, before,
                                   agent_evidence, direct, guard, emit, meter)
            after, records = evidence['after'], evidence['execution']
            effects = _check_effect(name,op,before,after,records,joint)
            rows.append(dict(scene=name, op=op, selection_source=evidence['selection_source'],
                             **selection, **effects))
            del environment, before, after, records, evidence, agent_evidence
    require(len(rows) == 8 and meter['direct_boundary_requests'] == 2
            and meter['evidence_write_failures'] == 0,
            'Probe did not complete the frozen eight branches')
    for key, expected in CAPS.items():
        if key != 'runs':
            require(meter[key+'_attempted'] == meter[key+'_returned'] == expected,
                    'Probe operation counts did not close')
    guard()
    meter['runs_returned'] = 1
    return dict(passed=True, rows=rows, counts=deepcopy(meter),
        model_forwards=0, complete_games=0, training=False, gpu=False,
        independent_evaluation=False, future_return_evaluated=False,
        counter_placeholders_not_replay=True)
