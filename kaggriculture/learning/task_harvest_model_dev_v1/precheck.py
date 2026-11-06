"""Eight counted controller calls and eight scores at four authenticated points.

The caller owns source authentication, one-shot execution and bounded local
storage. No source stream, checkpoint or environment is read on import. The
captured context contains only the encoder's finite set of required fields.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_dagger_once_v1.policy import BoundedAgent
from learning.task_harvest_choice_v1.adapter import _expanded
from learning.task_harvest_unexpressed_v1.selection import validate_target
from learning.task_choice_v1.features import candidate_id, encode_candidates
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS, _model
from learning.task_request_gate_v1.policy import same_request
from .contract import PRECHECK_LIMITS, ROLES, sources
from .policy import BoundedHarvestAgent


COST_KEYS = ('task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
             'gate_evaluations', 'controller_calls', 'teacher_proposals')
CONTEXT_KEYS = ('observation', 'configuration', 'remaining', 'today', 'last_plant_age',
                'final_day', 'units', 'available', 'tasks', 'actions', 'seeds',
                'needed_seeds', 'supply', 'projected_shed', 'choices')
WRAPPER_KEYS = frozenset(('model_counts', 'turn_model_counts', 'task_checkpoint_sha256'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _counts(value):
    _require(type(value) is dict and set(value) == set(COST_KEYS)
             and all(type(x) is int and x >= 0 for x in value.values()), 'Invalid model counters')
    return dict(value)


def _stable(value):
    # The existing bc_crop.audit.stable timing-only transform, kept here to
    # avoid importing that audit's official environment dependencies.
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k != 'inference_seconds'}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _same(left, right):
    return ordered_hash(_stable(left)) == ordered_hash(_stable(right))


def _initialize(meter):
    _require(type(meter) is dict and not meter, 'Precheck requires a fresh empty meter; no retry')
    meter.update(schema_version='task-harvest-model-precheck-meter-v1', scope='precheck', status='running',
        point_attempts=0, point_returns=0, actor_construction_attempts=0, actor_construction_returns=0,
        controller_attempts=0, controller_returns=0, score_attempts=0, score_returns=0,
        model_costs={key: 0 for key in COST_KEYS}, accounting_failures=0, emission_failures=0)


def _start(meter, kind):
    cap = {'point': 4, 'actor_construction': 8,
           'controller': PRECHECK_LIMITS['controller_calls'],
           'score': PRECHECK_LIMITS['standalone_scores']}[kind]
    _require(meter[kind + '_attempts'] < cap, 'Precheck operation budget exhausted')
    meter[kind + '_attempts'] += 1


def _emit(emit, meter, event):
    try:
        emit(event)
    except BaseException:
        meter['emission_failures'] += 1
        raise


def _account(meter, before, after):
    before, after = _counts(before), _counts(after)
    delta = {key: after[key] - before[key] for key in COST_KEYS}
    _require(all(value >= 0 for value in delta.values()), 'Model budget moved backwards')
    for key, value in delta.items():
        meter['model_costs'][key] += value
    c = meter['model_costs']
    _require(c['task_forwards'] <= PRECHECK_LIMITS['task_forwards']
             and c['crop_forwards'] <= PRECHECK_LIMITS['crop_forwards']
             and c['task_rows'] + c['crop_rows'] <= PRECHECK_LIMITS['neural_input_rows']
             and c['controller_calls'] <= PRECHECK_LIMITS['controller_calls']
             and c['gate_evaluations'] == c['teacher_proposals'] == 0,
             'Aggregate precheck model budget exhausted')


def _counted(actor, kind, meter, operation):
    _start(meter, kind)
    before, primary = _counts(actor.budget.counts), None
    try:
        result = operation()
        meter[kind + '_returns'] += 1
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _account(meter, before, actor.budget.counts)
        except BaseException:
            meter['accounting_failures'] += 1
            if primary is None:
                raise


@contextmanager
def _capture(controller, ordinal):
    namespace = controller.controller.__globals__
    original, captured, seen = namespace['_task_select'], [], 0

    def select(context):
        nonlocal seen
        if seen == ordinal:
            captured.append({key: deepcopy(context[key]) for key in CONTEXT_KEYS})
        seen += 1
        return original(context)

    namespace['_task_select'] = select
    try:
        yield captured
    finally:
        namespace['_task_select'] = original


def _core_trace(trace, task_sha, *, added_evidence=False):
    core = deepcopy(trace)
    if added_evidence:
        for record in core['records']:
            evidence = record.pop('task_model')
            candidates, logits = record['inputs']['candidates'], evidence['logits']
            _require(set(evidence) == {'checkpoint_sha256', 'chosen', 'logits', 'argmax_index', 'fallbacks'}
                     and evidence['checkpoint_sha256'] == task_sha and evidence['chosen'] == record['chosen']
                     and type(evidence['fallbacks']) is int and evidence['fallbacks'] == 0
                     and type(logits) is list and len(logits) == len(candidates)
                     and all(type(v) is float and np.isfinite(v) for v in logits)
                     and type(evidence['argmax_index']) is int, 'Task score evidence differs')
            index = int(np.argmax(logits))
            _require(evidence['argmax_index'] == index and candidates[index]['id'] == record['chosen'],
                     'Task score evidence is not the current candidate argmax')
    return core


def _validate_point(point, role, case):
    _require(type(point) is dict and set(point) == {'role', 'target', 'inputs', 'original_request',
             'original_trace', 'task', 'crop'} and point['role'] == role, 'Point fields or fixed role differ')
    target = validate_target(point['target'])
    inputs, history = point['inputs'], point['original_trace']
    _require(target['side'] == case['side'] and type(inputs) is dict
             and inputs['observation']['player'] == target['side']
             and inputs['observation']['step'] == target['step']
             and ordered_hash(inputs) == target['input_sha256'], 'Target input binding differs')
    _require(type(history) is dict and history['step'] == target['step']
             and type(history['records']) is list and len(history['records']) > target['ordinal']
             and ordered_hash(history['records'][target['ordinal']]['inputs']) == target['input_sha256'],
             'Target ordinal or original trace input differs')
    actions = history['final_reservations']['unit_actions']
    request = point['original_request']
    _require(ordered_hash(actions) == target['unit_requests_sha256']
             and ordered_hash(actions[target['unit']]) == ordered_hash(target['original_unit_request'])
             and type(request) is dict and set(request) == {'farmer', 'hands', 'market'}
             and ordered_hash([request['farmer'], *request['hands']]) == ordered_hash(actions),
             'Original whole request or final unit binding differs')
    models = {}
    for kind in ('task', 'crop'):
        pair = point[kind]
        _require(type(pair) in (tuple, list) and len(pair) == 2, 'Invalid model pair')
        models[kind] = _model({'path': pair[0], 'sha256': pair[1]})
    _require(history['task_checkpoint_sha256'] == models['task'][1]
             and history['checkpoint_sha256'] == models['crop'][1], 'Historical model identities differ')
    cumulative, turn = _counts(history['model_counts']), _counts(history['turn_model_counts'])
    _require(all(cumulative[key] >= turn[key] for key in COST_KEYS), 'Historical cumulative counts below turn')
    return target, models


def _controller(actor, point, meter):
    def execute():
        actor.budget.begin_turn()
        actor.budget.controller()
        inputs = point['inputs']
        return actor.controller(inputs['observation'], inputs['configuration'])
    request = _counted(actor, 'controller', meter, execute)
    _require(actor.previous is None, 'Middle-turn check advanced the episode ordering state')
    return dict(request=deepcopy(request), trace=deepcopy(actor.controller.last),
                counts=_counts(actor.budget.counts))


def _score(actor, inputs, meter):
    rows = len(inputs['candidates'])
    _require(1 <= rows <= PRECHECK_LIMITS['score_rows'], 'Standalone score row budget exceeded')
    def execute():
        actor.budget.begin_turn()
        actor.budget.forward('task', rows)
        return actor.network(inputs)
    chosen = _counted(actor, 'score', meter, execute)
    evidence = deepcopy(actor.network.last)
    _require(evidence['chosen'] == chosen and evidence['checkpoint_sha256'] == actor.task_sha,
             'Standalone score identity differs')
    _require(type(evidence['logits']) is list and len(evidence['logits']) == rows
             and all(type(v) is float and np.isfinite(v) for v in evidence['logits']),
             'Standalone score is nonfinite or incomplete')
    return dict(inputs=deepcopy(inputs), model=deepcopy(evidence), counts=_counts(actor.budget.counts))


def run(points, *, game_root, guard, emit, meter):
    """Run the fixed four-point precheck once; preserve attempts on any failure.

    ``emit`` receives private controller/score records before parity assertions.
    The returned report contains checks and counters, not duplicate full traces.
    ``meter`` must be an empty caller-owned dict and remains inspectable on error.
    """
    _initialize(meter)
    role, phase, reports = None, 'validate_points', []
    try:
        _require(type(points) in (list, tuple) and len(points) == 4
                 and callable(guard) and callable(emit), 'Expected four points and bounded callbacks')
        for point, role, case in zip(points, ROLES, sources()):
            _start(meter, 'point')
            phase = 'validate_point'
            guard()
            target, models = _validate_point(point, role, case)
            kwargs = dict(crop_path=Path(game_root) / models['crop'][0], crop_sha=models['crop'][1],
                          task_path=Path(game_root) / models['task'][0], task_sha=models['task'][1],
                          limits=dict(GAME_LIMITS))
            actors, controller_results = [], []
            captured = None
            for branch, constructor in (('original', BoundedAgent), ('disabled', BoundedHarvestAgent)):
                phase = branch + '_construct'
                guard()
                _start(meter, 'actor_construction')
                actor = constructor(**kwargs, **({'enabled': False} if branch == 'disabled' else {}))
                meter['actor_construction_returns'] += 1
                actors.append(actor)
                phase = branch + '_controller'
                guard()
                if branch == 'disabled':
                    _require(actor.controller.controller.__globals__ is not actors[0].controller.controller.__globals__,
                             'Controllers share a mutable AST namespace')
                    with _capture(actor.controller, target['ordinal']) as captured:
                        result = _controller(actor, point, meter)
                else:
                    result = _controller(actor, point, meter)
                controller_results.append(result)
                _emit(emit, meter, dict(event='precheck_controller', scope='precheck', role=role, branch=branch,
                                       target=deepcopy(target), **deepcopy(result), meter=deepcopy(meter)))
                core = _core_trace(result['trace'], models['task'][1], added_evidence=branch == 'disabled')
                history = point['original_trace']
                _require(set(history) == set(core) | WRAPPER_KEYS, 'Historical trace fields differ')
                history_core = {key: value for key, value in history.items() if key not in WRAPPER_KEYS}
                _require(same_request(result['request'], point['original_request']), 'Historical whole request differs')
                _require(_same(core, history_core), 'Historical core trace differs')
                _require(result['counts'] == history['turn_model_counts'], 'Historical turn budget differs')
            _require(len(captured) == 1, 'Target callback was not captured exactly once')
            _require(same_request(controller_results[0]['request'], controller_results[1]['request'])
                     and controller_results[0]['counts'] == controller_results[1]['counts'],
                     'Disabled full request or model cost differs')
            original_inputs, _ = encode_candidates(captured[0])
            expanded_inputs, _, virtual = _expanded(captured[0])
            harvest_id = candidate_id(target['unit'], target['target'], 'HARVEST')
            water_id = candidate_id(target['unit'], target['target'], 'WATER')
            _require(harvest_id in virtual and virtual[harvest_id]['water_id'] == water_id
                     and any(c['id'] == water_id for c in original_inputs['candidates'])
                     and any(c['id'] == harvest_id for c in expanded_inputs['candidates']),
                     'Frozen target has no corresponding virtual HARVEST from its own WATER')
            _require(ordered_hash(original_inputs) == target['input_sha256'], 'Captured input differs')
            n = len(original_inputs['candidates'])
            _require(ordered_hash(expanded_inputs['candidates'][:n]) == ordered_hash(original_inputs['candidates'])
                     and ordered_hash({k: v for k, v in expanded_inputs.items() if k not in ('candidates', 'candidate_scope')})
                     == ordered_hash({k: v for k, v in original_inputs.items() if k not in ('candidates', 'candidate_scope')}),
                     'Expanded encoding changed original features')
            scored = []
            for actor, branch, encoded in zip(actors, ('original_pool', 'expanded_pool'), (original_inputs, expanded_inputs)):
                phase = branch + '_score'
                guard()
                result = _score(actor, encoded, meter)
                scored.append(result)
                _emit(emit, meter, dict(event='precheck_score', scope='precheck', role=role, branch=branch,
                                       target=deepcopy(target), **deepcopy(result), meter=deepcopy(meter)))
            old, expanded = (np.asarray(row['model']['logits']) for row in scored)
            prefix = expanded[:n]
            _require(np.allclose(old, prefix, atol=PRECHECK_LIMITS['atol'], rtol=PRECHECK_LIMITS['rtol'])
                     and int(np.argmax(old)) == int(np.argmax(prefix)), 'Original candidate scores or argmax differ')
            index = int(np.argmax(expanded))
            tied_prefix = any(value == expanded[index] for value in prefix)
            _require(not tied_prefix or index < n, 'Exact tie did not preserve original candidate prefix')
            report = dict(role=role, passed=True, ordinal=target['ordinal'],
                original_candidates=n, added_candidates=len(virtual), original_argmax=int(np.argmax(old)),
                expanded_argmax=index, expanded_selected_new=index >= n,
                max_original_logit_error=float(np.max(np.abs(old - prefix))),
                historical_request_core_trace_turn_cost_match=True,
                historical_cumulative_counts_reconstructed=False,
                disabled_request_crop_reservations_match=True, exact_tie_prefix_preserved=True)
            meter['point_returns'] += 1
            reports.append(report)
            _emit(emit, meter, dict(event='precheck_point', scope='precheck', **deepcopy(report), meter=deepcopy(meter)))
        _require(meter['controller_returns'] == 8 and meter['score_returns'] == 8
                 and meter['point_returns'] == 4 and meter['accounting_failures'] == 0,
                 'Precheck did not complete the fixed operation counts')
        meter['status'] = 'passed'
        return dict(schema_version='task-harvest-model-precheck-v1', passed=True,
                    points=reports, meter=deepcopy(meter), tolerances=dict(atol=PRECHECK_LIMITS['atol'], rtol=PRECHECK_LIMITS['rtol']))
    except BaseException as primary:
        meter['status'], meter['failure_type'] = 'failed', type(primary).__name__
        try:
            _emit(emit, meter, dict(event='precheck_failure', scope='precheck', role=role, phase=phase,
                                   failure_type=type(primary).__name__, meter=deepcopy(meter)))
        except BaseException:
            pass  # _emit already counted this failed write; preserve primary.
        raise
