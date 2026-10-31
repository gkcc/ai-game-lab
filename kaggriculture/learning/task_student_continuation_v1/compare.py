"""Compare one authenticated student turn under a single task intervention.

Inputs, actions, snapshots and the returned object are private.  The caller must
authenticate the trace, previous trace, observations, action, model files and
limits against the same game and seat before calling.  This module reads no
files and neither executes the official engine nor establishes game outcomes.
The caller also owns hard resource/process limits; ``guard`` is a cooperative
stop check, not a replacement for those limits.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import re

from decision.api import observe
from learning.task_bc_error_diagnostics_v1.shadow import first_difference
from learning.task_bc_first_water_unit_requests_v1.probe import (
    _context_snapshot, _forced_choice, _student_choice,
)
from learning.task_choice_v1.features import candidate_id
from learning.task_dagger_once_v1.policy import BoundedAgent
from learning.task_single_intervention_v2.runner import (
    crop_selection_before_target, trace_without_crop_timing,
)

MAX_CANDIDATES_PER_CALLBACK = 256


class ComparisonStopped(ValueError):
    """Inconsistent authenticated input or controller state: do not retry."""


def _require(condition, message):
    if not condition:
        raise ComparisonStopped(message)


def _same(left, right):
    # This preserves object and list order, scalar types and float bits.
    return first_difference(left, right) is None


def _check_counts(value, keys):
    _require(type(value) is dict and list(value) == keys, 'counter schema mismatch')
    _require(all(type(v) is int and v >= 0 for v in value.values()), 'invalid counter')


@dataclass(frozen=True, repr=False)
class PrivateComparison:
    """Local only.  ``same`` concerns this turn, not official or economic value."""

    status: str
    reason: str
    checks: dict = field(default_factory=dict, repr=False)
    reference_action: object = field(default=None, repr=False)
    alternative_action: object = field(default=None, repr=False)

    def __repr__(self):
        return '<PrivateComparison>'

    def public_summary(self):
        return dict(status=self.status, reason=self.reason, checks=deepcopy(self.checks),
                    scope='bounded student single turn; official effect and future return unknown')


def _run(factory, guard, observation, configuration, document, previous_counts,
         ordinal, forced_id, expected_limits, prior_contexts=None, prior_agent=None):
    """Run one full BoundedAgent call, instrumenting only its private namespace."""
    guard()
    agent = factory()
    _require(isinstance(agent, BoundedAgent), 'expected bounded student')
    if prior_agent is not None:
        _require(agent is not prior_agent and agent.network is not prior_agent.network
                 and agent.controller.crop.network is not prior_agent.controller.crop.network,
                 'branches must have independent controller and network instances')
    _require(agent.trace is None and agent.controller.crop.trace is None,
             'trace-writing controller is forbidden')
    _require(agent.task_sha == document['task_checkpoint_sha256']
             and getattr(agent.network, 'sha', None) == document['task_checkpoint_sha256']
             and getattr(agent.controller.crop.network, 'sha', None) == document['checkpoint_sha256'],
             'loaded model identity mismatch')
    _require(agent.budget.limits == expected_limits, 'authenticated game limits mismatch')
    step, player = observation['step'], observation['player']
    keys = list(agent.budget.counts)
    _check_counts(previous_counts, keys)
    _check_counts(document.get('model_counts'), keys)
    _check_counts(document.get('turn_model_counts'), keys)
    _require(all(previous_counts[k] + document['turn_model_counts'][k] == document['model_counts'][k]
                 for k in keys), 'previous and target counters do not join')
    agent.budget.counts = deepcopy(previous_counts)
    agent.previous = None if step == 0 else (player, step - 1)

    controller = agent.controller
    namespace = controller.controller.__globals__
    original_select, original_finish = namespace.get('_task_select'), namespace.get('_task_finish')
    _require(original_select == controller.select and original_finish == controller.finish,
             'controller interception points differ')
    original_student = controller.task_selector
    _require(callable(original_student), 'student selector missing')
    contexts, events, ending = [], [], []
    before_crop_count = []

    def select(context):
        guard()
        index = len(controller.records)
        choices = context.get('choices')
        _require(type(choices) is list and 0 < len(choices) <= MAX_CANDIDATES_PER_CALLBACK,
                 'task callback candidate limit exceeded')
        snapshot = _context_snapshot(context)
        if index <= ordinal:
            if prior_contexts is not None:
                _require(index < len(prior_contexts) and _same(snapshot, prior_contexts[index]),
                         'intervention controller prefix mismatch')
            contexts.append(snapshot)
        best = max(context['choices'])
        unit, task_index = best[-2:]
        target, action, _ = context['tasks'][task_index]
        live_teacher = candidate_id(unit, target, action[0])
        if index == ordinal:
            before_crop_count.append(len(controller.crop.records))

        def choose(inputs):
            guard()
            _require(type(inputs.get('candidates')) is list and len(inputs['candidates']) == len(choices),
                     'encoded task candidate count differs')
            if index <= ordinal:
                _require(index < len(document['records']), 'historical callback missing')
                saved = document['records'][index]
                _require(_same(inputs, saved['inputs']) and live_teacher == saved['teacher_label'],
                         'historical callback input or teacher mismatch')
            # Even at the forced callback, do the ordinary student forward so
            # the shared budget and numerical path are not silently changed.
            chosen = _student_choice(inputs, original_student)
            if index <= ordinal:
                _require(chosen == document['records'][index]['chosen'], 'student prefix mismatch')
            if index == ordinal and forced_id is not None:
                chosen = _forced_choice(inputs, document['records'][index]['inputs'], forced_id)
            events.append((index, bool(index == ordinal and forced_id is not None)))
            guard()
            return chosen

        controller.task_selector = choose
        try:
            selected = original_select(context)
        finally:
            controller.task_selector = original_student
        _require(controller.records[-1]['teacher_label'] == live_teacher, 'live teacher mismatch')
        return selected

    def finish(context, action):
        ending.append(_context_snapshot(context))
        return original_finish(context, action)

    namespace['_task_select'], namespace['_task_finish'] = select, finish
    try:
        guard()
        action = agent(deepcopy(observation), deepcopy(configuration))
        guard()
    finally:
        namespace['_task_select'], namespace['_task_finish'] = original_select, original_finish
        controller.task_selector = original_student

    _require(type(action) is dict and set(action) == {'farmer', 'hands', 'market'}
             and type(action['farmer']) is list and type(action['hands']) is list
             and type(action['market']) is list, 'invalid complete request')
    trace = trace_without_crop_timing(agent.last)
    _require(all(trace.get(k) == 0 for k in ('load_errors', 'inference_errors', 'fallbacks')),
             'controller reported an error')
    _require(len(before_crop_count) == len(ending) == 1 and len(contexts) == ordinal + 1,
             'target or finish was not reached exactly once')
    _require(sum(x[1] for x in events) == int(forced_id is not None), 'incorrect override count')
    _require(_same(trace['final_reservations']['unit_actions'], [action['farmer'], *action['hands']]),
             'final reservations do not bind the complete request')
    _require(agent.previous == (player, step) and _same(agent.budget.counts, trace['model_counts'])
             and _same(agent.budget.turn, trace['turn_model_counts']), 'post-turn budget mismatch')
    crop_prefix = [crop_selection_before_target(r)
                   for r in trace['crop_records'][:before_crop_count[0]]]
    _require(len(crop_prefix) == before_crop_count[0], 'crop prefix incomplete')
    return dict(action=deepcopy(action), trace=trace, contexts=contexts, ending=ending[0],
                crop_prefix=crop_prefix, previous=agent.previous,
                limits=deepcopy(agent.budget.limits), instance=agent)


def compare_turn(document, observation, configuration, historical_action, ordinal, *,
                 task_sha256, crop_sha256, game_limits, agent_factory, guard, previous_document=None,
                 alternative='teacher'):
    """Return same/different/unknown for one *conditional* in-memory comparison.

    ``alternative`` is ``'teacher'`` or a distinct preselected third candidate
    ID; ``None`` means no third candidate.  No candidate search is performed.
    An authenticated immediately preceding student trace is mandatory after
    step zero.  The branch consumes the same original forward at the override,
    then recomputes the subsequent student choices from their new inputs.

    ``same`` additionally requires equal final reservations, stable crop trace,
    final controller context and bounded-agent next-turn control state.  It
    does not establish official effects, future trajectory or financial value.
    """
    _require(callable(agent_factory) and callable(guard) and type(game_limits) is dict,
             'missing factory, guard or authenticated game limits')
    guard()
    _require(type(document) is dict and type(ordinal) is int and ordinal >= 0,
             'invalid trace or ordinal')
    _require(document.get('task_checkpoint_sha256') == task_sha256
             and document.get('checkpoint_sha256') == crop_sha256
             and isinstance(task_sha256, str) and re.fullmatch('[0-9a-f]{64}', task_sha256)
             and isinstance(crop_sha256, str) and re.fullmatch('[0-9a-f]{64}', crop_sha256),
             'historical model identity mismatch')
    _require(all(type(document.get(k)) is int and document[k] == 0
                 for k in ('load_errors', 'inference_errors', 'fallbacks')),
             'historical controller failure')
    view = observe(deepcopy(observation), deepcopy(configuration))
    observation, configuration = view.observation, view.configuration
    step, player = observation['step'], observation['player']
    _require(type(step) is int and 0 <= step < 719 and player in (0, 1)
             and document.get('step') == step, 'observation identity mismatch')
    records = document.get('records')
    _require(type(records) is list and ordinal < len(records), 'target callback missing')
    target = records[ordinal]
    _require(type(target) is dict and type(target.get('inputs')) is dict
             and _same(target['inputs'].get('observation'), observation)
             and _same(target['inputs'].get('configuration'), configuration),
             'target observation or configuration mismatch')
    _require(target.get('source') == 'callback' and target.get('chosen') != target.get('teacher_label'),
             'target is not a student/teacher disagreement')
    candidates = target['inputs'].get('candidates')
    _require(type(candidates) is list and all(type(c) is dict and isinstance(c.get('id'), str) for c in candidates),
             'invalid candidate list')
    ids = [c['id'] for c in candidates]
    _require(len(ids) == len(set(ids)) and target.get('chosen') in ids
             and target.get('teacher_label') in ids, 'candidate identity mismatch')
    if alternative is None:
        return PrivateComparison('unknown', 'no_third_candidate')
    if alternative == 'teacher':
        forced_id = target['teacher_label']
    else:
        _require(isinstance(alternative, str), 'invalid alternative identity')
        if alternative not in ids or alternative in (target['chosen'], target['teacher_label']):
            return PrivateComparison('unknown', 'no_distinct_third_candidate')
        forced_id = alternative

    if step:
        if previous_document is None:
            return PrivateComparison('unknown', 'previous_control_state_unavailable')
        _require(type(previous_document) is dict and previous_document.get('step') == step - 1
                 and previous_document.get('task_checkpoint_sha256') == task_sha256
                 and previous_document.get('checkpoint_sha256') == crop_sha256
                 and all(type(previous_document.get(k)) is int and previous_document[k] == 0
                         for k in ('load_errors', 'inference_errors', 'fallbacks')),
                 'previous trace identity mismatch')
        previous_counts = previous_document.get('model_counts')
    else:
        _require(previous_document is None, 'unexpected previous trace at episode start')
        turn = document.get('turn_model_counts')
        _require(type(turn) is dict, 'missing initial counters')
        previous_counts = {k: 0 for k in turn}

    reference = _run(agent_factory, guard, observation, configuration, document, previous_counts,
                     ordinal, None, game_limits)
    _require(_same(reference['trace'], trace_without_crop_timing(document)), 'historical baseline trace mismatch')
    _require(_same(reference['action'], historical_action), 'historical complete request mismatch')
    branch = _run(agent_factory, guard, observation, configuration, document, previous_counts,
                  ordinal, forced_id, game_limits, reference['contexts'], reference['instance'])
    _require(_same(reference['crop_prefix'], branch['crop_prefix']), 'pre-target crop prefix mismatch')
    checks = dict(
        request_equal=_same(reference['action'], branch['action']),
        reservations_equal=_same(reference['trace']['final_reservations'], branch['trace']['final_reservations']),
        stable_crop_equal=_same(reference['trace']['crop_records'], branch['trace']['crop_records']),
        final_context_equal=_same(reference['ending'], branch['ending']),
        next_turn_control_equal=(_same(reference['trace']['model_counts'], branch['trace']['model_counts'])
            and _same(reference['trace']['turn_model_counts'], branch['trace']['turn_model_counts'])
            and reference['previous'] == branch['previous'] and _same(reference['limits'], branch['limits'])),
    )
    if not checks['request_equal']:
        status, reason = 'different', 'complete_request_differs'
    elif all(checks.values()):
        status, reason = 'same', 'same_in_turn_and_bounded_control_state'
    else:
        status, reason = 'unknown', 'request_same_but_other_state_differs'
    guard()
    return PrivateComparison(status, reason, checks, reference['action'], branch['action'])
