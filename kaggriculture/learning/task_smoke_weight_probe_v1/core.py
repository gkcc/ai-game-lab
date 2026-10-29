"""Pure, bounded same-observation task-head intervention.

Inputs are supplied by a separately verified offline reader. Nothing in this
module advances the official engine, persists private data, or computes game
rewards. An intervention here is not an executed move or a performance test.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from decision.api import observe
from learning.bc_crop.audit import stable
from learning.task_dagger_once_v1.policy import BoundedAgent, LIMITS
from learning.task_choice_v1.agent import TaskAgent
from learning.bc_crop.agent import CropAgent, Network as CropNetwork
from learning.task_request_gate_v1.policy import same_request


TURNS = 719
SAMPLE_TURNS = tuple(i * 718 // 31 for i in range(32))
MAX_CONTROLLER_CALLS = 96
COUNT_KEYS = ('task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
              'gate_evaluations', 'controller_calls', 'teacher_proposals')
PARAM_KEYS = ('w0', 'b0', 'w1', 'b1', 'w2', 'b2')
LOGIT_ATOL = 1e-12
LOGIT_RTOL = 1e-12


class ProbeError(ValueError):
    """A comparison invariant failed; exception messages contain no data."""


def _require(value, message):
    if not value:
        raise ProbeError(message)


def _ordered(value):
    """Preserve JSON object/list order and scalar types for input comparison."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                          sort_keys=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProbeError('Non-JSON comparison input') from exc


def _limits(limits):
    _require(isinstance(limits, dict) and set(limits) == set(LIMITS), 'Invalid budget keys')
    _require(all(type(v) is int and v > 0 for v in limits.values()), 'Invalid budget values')
    return dict(limits)


def _counts(value):
    _require(isinstance(value, dict) and set(value) == set(COUNT_KEYS), 'Invalid count keys')
    _require(all(type(value[k]) is int and value[k] >= 0 for k in COUNT_KEYS), 'Invalid counts')
    return {k: value[k] for k in COUNT_KEYS}


def validate_trace(lines, limits, task_sha, crop_sha):
    """Validate every trace row, retaining only sampled rows and predecessors."""
    limits = _limits(limits)
    retained = {}
    wanted = set(SAMPLE_TURNS) | {step - 1 for step in SAMPLE_TURNS if step}
    previous = {k: 0 for k in COUNT_KEYS}
    length = 0
    for step, trace in enumerate(lines):
        _require(step < TURNS and isinstance(trace, dict), 'Unexpected trace length')
        _require(type(trace.get('step')) is int and trace['step'] == step, 'Trace step mismatch')
        _require(all(type(trace.get(k)) is int and trace[k] == 0
                     for k in ('fallbacks', 'load_errors', 'inference_errors')), 'Trace errors')
        _require(trace.get('task_checkpoint_sha256') == task_sha
                 and trace.get('checkpoint_sha256') == crop_sha, 'Trace checkpoint mismatch')
        current, turn = _counts(trace.get('model_counts')), _counts(trace.get('turn_model_counts'))
        records, crops = trace.get('records'), trace.get('crop_records')
        _require(isinstance(records, list) and isinstance(crops, list)
                 and all(isinstance(r, dict) for r in crops), 'Invalid selector record lists')
        rows = 0
        for record in records:
            _require(isinstance(record, dict) and isinstance(record.get('inputs'), dict), 'Invalid task record')
            candidates = record['inputs'].get('candidates')
            _require(isinstance(candidates, list) and candidates
                     and all(isinstance(c, dict) and isinstance(c.get('id'), str) for c in candidates)
                     and len({c['id'] for c in candidates}) == len(candidates), 'Invalid candidate records')
            rows += len(candidates)
        _require(len(records) == turn['task_forwards'] and rows == turn['task_rows']
                 and len(crops) == turn['crop_forwards'] == turn['crop_rows'], 'Record and budget counts differ')
        _require(all(current[k] == previous[k] + turn[k] for k in COUNT_KEYS), 'Count sequence mismatch')
        _require(turn['controller_calls'] == 1 and current['controller_calls'] == step + 1,
                 'Invalid controller sequence')
        _require(current['gate_evaluations'] == current['teacher_proposals'] == 0,
                 'Unexpected policy branch')
        _require(current['task_forwards'] <= limits['task_forwards']
                 and current['crop_forwards'] <= limits['crop_forwards']
                 and current['controller_calls'] <= limits['controller_calls']
                 and current['task_rows'] + current['crop_rows'] <= limits['neural_input_rows']
                 and current['task_rows'] >= current['task_forwards']
                 and current['crop_rows'] == current['crop_forwards'], 'Total budget mismatch')
        _require(turn['task_forwards'] <= limits['max_task_forwards_per_turn']
                 and turn['crop_forwards'] <= limits['max_crop_forwards_per_turn']
                 and turn['controller_calls'] <= limits['max_controller_calls_per_turn']
                 and turn['task_rows'] + turn['crop_rows'] <= limits['max_neural_input_rows_per_turn']
                 and turn['task_rows'] >= turn['task_forwards']
                 and turn['crop_rows'] == turn['crop_forwards'], 'Turn budget mismatch')
        if step in wanted:
            retained[step] = deepcopy(trace)
        previous = current
        length += 1
    _require(length == TURNS and set(retained) == wanted, 'Incomplete trace')
    return retained


def legal_view(frame, side, configuration):
    """Use only this player's private data and the official public projection."""
    _require(type(side) is int and side in (0, 1), 'Invalid seat')
    _require(isinstance(frame, list) and len(frame) == 2, 'Invalid observation frame')
    try:
        own = deepcopy(frame[side]['observation'])
        public = frame[0]['observation']
        for name in ('step', 'farms', 'market', 'town', 'day', 'hour'):
            own[name] = deepcopy(public[name])
        view = observe(own, configuration)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProbeError('Invalid legal observation') from exc
    _require(view.observation['player'] == side, 'Observation seat mismatch')
    return view.observation, view.configuration


def _parameter_digest(network):
    """Identify the complete in-memory task model, not a checkpoint file."""
    _require(set(network.params) == set(PARAM_KEYS), 'Invalid task parameters')
    h = hashlib.sha256(b'task-smoke-weight-probe-v1:float64-parameters\x00')
    h.update(_ordered(network.metadata))
    for key, value in (('mean', network.mean), ('scale', network.scale),
                       *((key, network.params[key]) for key in PARAM_KEYS)):
        arr = np.asarray(value, dtype='<f8', order='C')
        _require(np.isfinite(arr).all(), 'Nonfinite model parameter')
        h.update(key.encode('ascii') + b'\x00')
        h.update(json.dumps(list(arr.shape), separators=(',', ':')).encode('ascii') + b'\x00')
        h.update(arr.tobytes(order='C'))
    return h.hexdigest()


def _same_array(a, b):
    return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            and not np.shares_memory(a, b))


def _closure(function):
    code = getattr(function, '__code__', None)
    cells = getattr(function, '__closure__', None)
    _require(code is not None and cells is not None and len(code.co_freevars) == len(cells),
             'Missing controller closure')
    try:
        return {name: cell.cell_contents for name, cell in zip(code.co_freevars, cells)}
    except ValueError as exc:
        raise ProbeError('Empty controller closure') from exc


def _bound_method(value, owner, function):
    return getattr(value, '__self__', None) is owner and getattr(value, '__func__', None) is function


def _verify_agent_bindings(agent):
    """Detect copied/misbound frozen controller closures before inference."""
    _require(type(agent) is BoundedAgent and type(agent.controller) is TaskAgent,
             'Unexpected real controller type')
    controller, crop = agent.controller, agent.controller.crop
    _require(type(crop) is CropAgent and type(crop.network) is CropNetwork, 'Unexpected real crop type')
    select = _closure(controller.task_selector)
    crop_call = _closure(crop.network.logits)
    _require(select.get('self') is agent and crop_call.get('self') is agent
             and _bound_method(crop_call.get('original_crop'), crop.network, CropNetwork.logits),
             'Controller closure bound to another actor')
    task_globals = controller.controller.__globals__
    crop_globals = crop.controller.__globals__
    _require(_bound_method(task_globals.get('_task_select'), controller, TaskAgent.select)
             and _bound_method(task_globals.get('_crop_select'), crop, CropAgent.select)
             and _bound_method(task_globals.get('_task_finish'), controller, TaskAgent.finish)
             and _bound_method(crop_globals.get('_bc_select'), crop, CropAgent.select)
             and _bound_method(crop_globals.get('_bc_finish'), crop, CropAgent.finish),
             'AST controller bound to another actor')


def _prepare_three(agents, source_sha):
    """Check independent models, then negate only the third task output layer."""
    _require(len(agents) == 3 and len({id(a) for a in agents}) == 3, 'Actors are not independent')
    for agent in agents:
        if isinstance(agent, BoundedAgent):
            _verify_agent_bindings(agent)
    networks = [a.network for a in agents]
    crops = [a.controller.crop.network for a in agents]
    _require(all(n is not None for n in networks) and len({id(n) for n in networks}) == 3,
             'Task networks are not independent')
    _require(all(c is not None for c in crops) and len({id(c) for c in crops}) == 3,
             'Crop networks are not independent')
    _require(all(a.task_sha == source_sha and n.sha == source_sha for a, n in zip(agents, networks)),
             'Task source identity mismatch')
    for left, right in ((0, 1), (0, 2), (1, 2)):
        base, other = networks[left], networks[right]
        _require(_ordered(base.metadata) == _ordered(other.metadata), 'Task metadata differs')
        _require(_same_array(base.mean, other.mean) and _same_array(base.scale, other.scale),
                 'Task normalization differs or aliases')
        _require(set(base.params) == set(other.params) == set(PARAM_KEYS)
                 and all(_same_array(base.params[k], other.params[k]) for k in PARAM_KEYS),
                 'Task parameters differ or alias')
        base, other = crops[left], crops[right]
        _require(base.sha == other.sha and _ordered(base.meta) == _ordered(other.meta)
                 and _same_array(base.mean, other.mean) and _same_array(base.scale, other.scale)
                 and len(base.weights) == len(other.weights)
                 and all(len(pair) == 2 for pair in (*base.weights, *other.weights)),
                 'Crop model differs or aliases')
        _require(all(_same_array(x, y) for pair_a, pair_b in zip(base.weights, other.weights)
                     for x, y in zip(pair_a, pair_b)), 'Crop parameters differ or alias')
    original = _parameter_digest(networks[0])
    _require(_parameter_digest(networks[1]) == original == _parameter_digest(networks[2]),
             'Task parameter identities differ')
    for key in ('w2', 'b2'):
        networks[2].params[key] = -networks[2].params[key].copy()
        _require(np.isfinite(networks[2].params[key]).all()
                 and np.array_equal(networks[0].params[key], -networks[2].params[key]),
                 'Invalid output-layer intervention')
    _require(any(not np.array_equal(networks[0].params[k], networks[2].params[k]) for k in ('w2', 'b2')),
             'Intervention did not change numeric parameters')
    _require(all(np.array_equal(networks[0].params[k], networks[2].params[k])
                 for k in ('w0', 'b0', 'w1', 'b1')), 'Intervention changed another layer')
    derived = _parameter_digest(networks[2])
    _require(derived != original and _parameter_digest(networks[0]) == _parameter_digest(networks[1]) == original,
             'Intervention identity mismatch')
    # An in-memory identity is deliberately distinct from the source .npz SHA.
    # No modified checkpoint is written or claimed to exist on disk.
    networks[2].sha = derived
    agents[2].task_sha = derived
    return original, derived


class _Capture:
    """Record only in memory the exact input and result of each model call."""
    def __init__(self, network):
        self.network, self.calls = network, []

    def __call__(self, inputs):
        _require(isinstance(inputs, dict) and not {'teacher_label', 'teacher_priority', 'chosen'} & set(inputs),
                 'Training-only value in task input')
        snapshot = deepcopy(inputs)
        # The network normally clears this at entry; doing so here also makes a
        # missing update fail rather than reusing a prior callback's logits.
        self.network.last = None
        choice = self.network(inputs)
        last = self.network.last
        _require(isinstance(last, dict) and last.get('chosen') == choice
                 and last.get('checkpoint_sha256') == self.network.sha and last.get('fallbacks') == 0,
                 'Task callback result mismatch')
        candidates = snapshot.get('candidates')
        _require(isinstance(candidates, list) and candidates, 'Missing candidates')
        ids = [entry['id'] for entry in candidates]
        logits = np.asarray(last.get('logits'), dtype=np.float64)
        _require(logits.shape == (len(ids),) and np.isfinite(logits).all()
                 and len(ids) == len(set(ids)), 'Invalid task logits')
        _require(choice == ids[int(np.argmax(logits))], 'Task argmax or tie behavior mismatch')
        self.calls.append((snapshot, choice, logits.copy()))
        return choice


def _invoke(agent, capture, obs, cfg, step, side, prior):
    _require(agent.budget.limits == _limits(agent.budget.limits), 'Invalid actor budget')
    agent.budget.counts = dict(prior)
    agent.previous = None if step == 0 else (side, step - 1)
    agent.network = capture
    request = agent(deepcopy(obs), deepcopy(cfg))
    _require(agent.budget.counts['controller_calls'] == prior['controller_calls'] + 1
             and agent.budget.turn['controller_calls'] == 1, 'Unexpected controller call count')
    _require(isinstance(agent.last, dict) and isinstance(agent.last.get('records'), list), 'Missing actor trace')
    records = agent.last['records']
    _require(len(records) == len(capture.calls), 'Callback count differs from trace')
    for record, (inputs, choice, _logits) in zip(records, capture.calls):
        _require(_ordered(record['inputs']) == _ordered(inputs) and record['chosen'] == choice,
                 'Callback and trace differ')
        _require(_ordered(inputs['observation']) == _ordered(obs)
                 and _ordered(inputs['configuration']) == _ordered(cfg), 'Model received a different view')
    return request, agent.last


def compare(replay, trace_lines, make_agent, limits, task_sha, crop_sha, *, side=0):
    """Compare 32 fixed source states with separate original/control/flipped actors."""
    limits = _limits(limits)
    _require(callable(make_agent) and type(side) is int and side in (0, 1), 'Invalid comparison setup')
    _require(isinstance(replay, dict) and isinstance(replay.get('steps'), list)
             and len(replay['steps']) == 720 and isinstance(replay.get('configuration'), dict), 'Incomplete replay')
    traces = validate_trace(trace_lines, limits, task_sha, crop_sha)
    totals = dict(sampled_turns=0, baseline_actions_reproduced=0, baseline_traces_reproduced=0,
                  control_actions_equal=0, control_traces_equal=0, comparable_selector_calls=0,
                  baseline_selector_calls=0, intervention_selector_calls=0,
                  zero_task_turns=0, single_candidate_calls=0, multi_candidate_calls=0,
                  baseline_tied_calls=0, intervention_tied_calls=0,
                  turns_with_first_choice_difference=0, turns_without_choice_difference=0,
                  turns_with_selection_sequence_difference=0, turns_with_request_difference=0,
                  offline_controller_calls=0)
    max_error = None
    identities = None
    for step in SAMPLE_TURNS:
        obs, cfg = legal_view(replay['steps'][step], side, replay['configuration'])
        _require(type(obs['step']) is int and obs['step'] == step, 'Replay step mismatch')
        expected = replay['steps'][step + 1][side]['action']
        recorded = traces[step]
        _require(recorded.get('step') == step, 'Sample trace step mismatch')
        prior = ({k: 0 for k in COUNT_KEYS} if step == 0 else _counts(traces[step - 1]['model_counts']))
        _require(prior['controller_calls'] == step, 'Invalid pre-turn controller count')
        agents = [make_agent() for _ in range(3)]
        _require(all(a.budget.limits == limits for a in agents), 'Actor budget changed')
        current_ids = _prepare_three(agents, task_sha)
        if identities is None:
            identities = current_ids
        _require(current_ids == identities, 'Model identities changed between samples')
        captures = [_Capture(a.network) for a in agents]
        results = []
        for agent, capture in zip(agents, captures):
            _require(totals['offline_controller_calls'] < MAX_CONTROLLER_CALLS, 'Offline controller limit exceeded')
            results.append(_invoke(agent, capture, obs, cfg, step, side, prior))
            totals['offline_controller_calls'] += 1
        (baseline_action, baseline_trace), (control_action, control_trace), (altered_action, _altered_trace) = results
        _require(same_request(baseline_action, expected), 'Baseline action mismatch')
        _require(same_request(stable(baseline_trace), stable(recorded)), 'Baseline trace mismatch')
        _require(same_request(control_action, baseline_action)
                 and same_request(stable(control_trace), stable(baseline_trace)), 'Same-weight control mismatch')
        b, c, a = (capture.calls for capture in captures)
        _require(len(b) == len(c) and all(_ordered(x[0]) == _ordered(y[0]) and x[1] == y[1]
                 and np.array_equal(x[2], y[2]) for x, y in zip(b, c)), 'Same-weight inputs differ')
        totals['baseline_selector_calls'] += len(b)
        totals['intervention_selector_calls'] += len(a)
        totals['turns_with_selection_sequence_difference'] += int([x[1] for x in b] != [x[1] for x in a])
        first_difference = False
        if not b:
            totals['zero_task_turns'] += 1
        for original, changed in zip(b, a):
            _require(_ordered(original[0]) == _ordered(changed[0]), 'Inputs differ before first choice difference')
            _require(original[2].shape == changed[2].shape, 'Intervention logit shape changed')
            err = np.abs(original[2] + changed[2])
            _require(np.isfinite(err).all()
                     and np.allclose(changed[2], -original[2], rtol=LOGIT_RTOL, atol=LOGIT_ATOL),
                     'Intervention logits do not negate within tolerance')
            maximum = float(np.max(err))
            max_error = maximum if max_error is None else max(max_error, maximum)
            totals['comparable_selector_calls'] += 1
            totals['single_candidate_calls' if len(original[2]) == 1 else 'multi_candidate_calls'] += 1
            totals['baseline_tied_calls'] += int(np.count_nonzero(original[2] == np.max(original[2])) > 1)
            totals['intervention_tied_calls'] += int(np.count_nonzero(changed[2] == np.max(changed[2])) > 1)
            if original[1] != changed[1]:
                first_difference = True
                break
        if not first_difference:
            _require(len(b) == len(a), 'Call sequence changed before a choice difference')
            _require(same_request(baseline_action, altered_action), 'Request changed without a choice difference')
        totals['turns_with_first_choice_difference' if first_difference else 'turns_without_choice_difference'] += 1
        totals['turns_with_request_difference'] += int(not same_request(baseline_action, altered_action))
        totals['sampled_turns'] += 1
        totals['baseline_actions_reproduced'] += 1
        totals['baseline_traces_reproduced'] += 1
        totals['control_actions_equal'] += 1
        totals['control_traces_equal'] += 1
    _require(totals['offline_controller_calls'] == MAX_CONTROLLER_CALLS, 'Offline controller calls incomplete')
    return dict(passed=True, passed_meaning='comparison_invariants_passed_only', counts=totals,
                task_choice_control_observed=totals['turns_with_first_choice_difference'] > 0,
                action_control_observed=totals['turns_with_request_difference'] > 0,
                max_absolute_logit_negation_error=max_error,
                logit_atol=LOGIT_ATOL, logit_rtol=LOGIT_RTOL,
                source_task_checkpoint_sha256=task_sha, source_crop_checkpoint_sha256=crop_sha,
                baseline_task_parameter_sha256=identities[0], intervention_task_parameter_sha256=identities[1],
                intervention='negate_task_w2_and_b2_in_memory',
                interpretation='Offline same-observation sensitivity only; no official counterfactual execution or competitive improvement')
