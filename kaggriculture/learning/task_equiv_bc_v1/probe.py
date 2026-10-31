"""Run one bounded, private teacher-continuation comparison in memory.

The caller authenticates the historical source, its teacher/schema, the own
observation, and its next-frame request. No replay, game, model or file is
opened here. Output arrays and candidate IDs contain sensitive local data and
are not safe to log or publish. The comparison does not assess future return.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from decision.api import observe
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_request_gate_v1.policy import same_request
from learning.task_single_intervention_v2.runner import (
    crop_selection_before_target, trace_without_crop_timing,
)
from learning.task_teacher_choice_bc_v1.extract import extract


def task_agent_factory(ordinal, forced_id):
    """Create a fresh pinned teacher, with at most one forced task choice.

    Crop mode and all later task choices remain the original teacher. The
    subclass only supplies a selector for the target call, and never edits the
    frozen teacher or the official rules. The two public probe attributes are
    captured before that task selection, not inferred from end-of-turn fields.
    """
    from learning.task_choice_v1.agent import TaskAgent

    class Branch(TaskAgent):
        def __init__(self):
            self.probe_target_hits = 0
            self.probe_forced_hits = 0
            self.probe_prior_crop_count = None
            super().__init__(task_selector=None, crop_mode='teacher')

        def select(self, context):
            if len(self.records) != ordinal:
                return super().select(context)
            self.probe_target_hits += 1
            self.probe_prior_crop_count = len(self.crop.records)
            if forced_id is None:
                return super().select(context)
            self.task_selector = lambda _inputs: forced_id
            try:
                selected = super().select(context)
                self.probe_forced_hits += 1
                return selected
            finally:
                self.task_selector = None

    return Branch()


def _same_ordered(left, right):
    return ordered_hash(left) == ordered_hash(right)


@dataclass(frozen=True, repr=False)
class _TeacherReference:
    """Private in-memory reference; the digest detects accidental mutation only."""
    ordinal: int
    teacher_label: str
    teacher_prefix: tuple
    action: object
    trace: object
    crop_prefix: object
    integrity: str

    def __repr__(self):
        return '<PrivateTeacherReference>'


def _reference_hash(ordinal, teacher_label, teacher_prefix, action, trace, crop_prefix):
    return ordered_hash([ordinal, teacher_label, list(teacher_prefix), action, trace, crop_prefix])


def _stable_trace(trace):
    # Only top-level crop-record inference_seconds is nondeterministic. The
    # existing helper also checks its type, finiteness, and nonnegativity.
    return trace_without_crop_timing(trace)


class _GuardStopped(Exception):
    pass


def _guard(guard):
    try:
        guard()
    except Exception as exc:
        raise _GuardStopped('Probe resource guard stopped') from exc


def _item_order(observation):
    private = observation['private']
    return ([list(inventory) for inventory in private['inventories']], list(private['shed']))


def _validate_run(agent, action, trace, observation, configuration, ordinal, forced_id):
    if (type(action) is not dict or set(action) != {'farmer', 'hands', 'market'}
            or type(action['farmer']) is not list or type(action['hands']) is not list
            or type(action['market']) is not list or type(trace) is not dict
            or type(trace.get('step')) is not int or trace['step'] != observation['step']
            or any(type(trace.get(k)) is not int or trace[k] != 0
                   for k in ('load_errors', 'inference_errors', 'fallbacks'))):
        raise ValueError('Invalid completed teacher run')
    records = trace.get('records')
    crops = trace.get('crop_records')
    reservations = trace.get('final_reservations')
    if (type(records) is not list or len(records) <= ordinal
            or type(crops) is not list or type(reservations) is not dict
            or set(reservations) != {'seeds', 'needed_seeds', 'supply', 'projected_shed', 'unit_actions'}
            or type(getattr(agent, 'probe_target_hits', None)) is not int
            or agent.probe_target_hits != 1
            or type(getattr(agent, 'probe_forced_hits', None)) is not int
            or agent.probe_forced_hits != int(forced_id is not None)):
        raise ValueError('Invalid controller completion')
    crop_count = getattr(agent, 'probe_prior_crop_count', None)
    if type(crop_count) is not int or not 0 <= crop_count <= len(crops):
        raise ValueError('Missing pre-target crop count')
    stable = _stable_trace(trace)
    # Only selection-time crop fields can belong to the prefix; unit_request
    # and buy_seed_orders are attached only when the whole turn finishes.
    crop_prefix = [crop_selection_before_target(r) for r in stable['crop_records'][:crop_count]]
    actions = [action['farmer'], *action['hands']]
    if not _same_ordered(reservations['unit_actions'], actions):
        raise ValueError('Final reservations do not bind the request')
    seen = set()
    for index, record in enumerate(records):
        if type(record) is not dict or type(record.get('inputs')) is not dict:
            raise ValueError('Invalid task record')
        inputs = record['inputs']
        if (not same_request(inputs.get('observation'), observation)
                or _item_order(inputs['observation']) != _item_order(observation)
                or not same_request(inputs.get('configuration'), configuration)
                or type(inputs.get('candidates')) is not list):
            raise ValueError('Task observation differs')
        candidates = inputs['candidates']
        chosen = [c for c in candidates if type(c) is dict and c.get('id') == record.get('chosen')]
        unit = record.get('unit')
        if (len(chosen) != 1 or type(unit) is not int or not 0 <= unit < len(actions)
                or unit in seen or any(record.get(k) != chosen[0].get(k) for k in ('unit', 'target', 'op'))
                or not same_request(record.get('unit_request'), actions[unit])):
            raise ValueError('Task request does not bind the emitted action')
        seen.add(unit)
        if forced_id is not None and index == ordinal:
            if record.get('source') != 'callback' or record.get('chosen') != forced_id:
                raise ValueError('Forced choice was not observed')
        elif record.get('source') != 'teacher' or record.get('chosen') != record.get('teacher_label'):
            raise ValueError('Continuation did not use the teacher')
    # Validate all JSON-domain content before any comparison.
    ordered_hash(stable)
    return stable, crop_prefix


def _execute(factory, observation, configuration, ordinal, forced_id, guard, on_called=None):
    _guard(guard)
    agent = factory(ordinal, forced_id)
    if not callable(agent):
        raise ValueError('Controller is not callable')
    own_observation, own_configuration = deepcopy(observation), deepcopy(configuration)
    if on_called is not None:
        on_called()
    action = agent(own_observation, own_configuration)
    trace = deepcopy(agent.last)
    stable, crop_prefix = _validate_run(agent, action, trace, observation, configuration, ordinal, forced_id)
    _guard(guard)
    return deepcopy(action), stable, crop_prefix


def _prefix(records, ordinal):
    """Exclude the request appended after the full turn, but nothing else."""
    return [{k: v for k, v in record.items() if k != 'unit_request'}
            for record in records[:ordinal]]


def _pick_alternatives(candidates, teacher):
    """First in canonical order per disjoint category; never backfill."""
    categories = (lambda c: c['op'] == teacher['op'] and c['target'] == teacher['target']
                             and c['unit'] != teacher['unit'],
                  lambda c: c['op'] == teacher['op'] and c['target'] != teacher['target'],
                  lambda c: c['op'] != teacher['op'])
    chosen = []
    for condition in categories:
        match = next((c['id'] for c in candidates if c['id'] != teacher['id']
                      and c['id'] not in chosen and condition(c)), None)
        if match is not None:
            chosen.append(match)
    return chosen


def probe_group(observation, configuration, ordinal, *, expected_inputs, expected_label,
                expected_teacher_prefix, expected_request, expected_schema, agent_factory=task_agent_factory,
                guard=lambda: None, max_candidates=256, retain_reference=False):
    """Return one private candidate group or an unknown reference status.

    The caller supplies an independently authenticated demonstration target,
    projection, full earlier teacher-label prefix, schema and recorded actor
    request from frame ``step+1``. The earlier prefix must contain exactly one
    original teacher label per prior task selection. The
    factory signature is ``(ordinal, forced_id)`` and each call must return a
    fresh callable agent exposing ``last``, ``probe_target_hits``,
    ``probe_forced_hits`` and ``probe_prior_crop_count``. Resource guard errors
    propagate; individual controller errors yield unknown without exception text.

    Verified equal candidates only reproduce this turn under a fixed teacher
    continuation; different requests are not evidence of inferior value.
    Unknown and untested candidates remain in the full feature group but are
    absent from the auxiliary verified mask. Returned values are local only.
    ``retain_reference=True`` additionally returns an opaque private object
    containing the full validated teacher turn for in-memory development use;
    by default it is omitted and this function does not write or log it.
    """
    if (type(ordinal) is not int or ordinal < 0 or not callable(agent_factory)
            or not callable(guard) or type(max_candidates) is not int
            or not 1 <= max_candidates <= 256 or type(expected_label) is not str
            or not expected_label or type(observation) is not dict or type(configuration) is not dict
            or type(expected_teacher_prefix) not in (list, tuple) or len(expected_teacher_prefix) != ordinal
            or any(type(label) is not str or not label for label in expected_teacher_prefix)
            or type(retain_reference) is not bool):
        raise ValueError('Invalid probe parameters')
    _guard(guard)
    view = observe(observation, configuration)
    if not same_request(view.observation, observation) or not same_request(view.configuration, configuration):
        raise ValueError('Input is not the sanitized actor observation')
    observation, configuration = view.observation, view.configuration
    try:
        action, trace, crop_prefix = _execute(agent_factory, observation, configuration, ordinal, None, guard)
        action2, trace2, crop_prefix2 = _execute(agent_factory, observation, configuration, ordinal, None, guard)
        if (not same_request(action, expected_request) or not same_request(action, action2)
                or not _same_ordered(trace, trace2) or not _same_ordered(crop_prefix, crop_prefix2)):
            return {'status': 'unknown', 'reason': 'reference_mismatch'}
        if any(prior.get('chosen') != label or prior.get('teacher_label') != label
               for prior, label in zip(trace['records'][:ordinal], expected_teacher_prefix)):
            return {'status': 'unknown', 'reason': 'historical_prefix_mismatch'}
        record = trace['records'][ordinal]
        if (record.get('chosen') != expected_label or record.get('teacher_label') != expected_label
                or not same_request(record.get('inputs'), expected_inputs)):
            return {'status': 'unknown', 'reason': 'historical_target_mismatch'}
        parsed = extract(record, expected_schema, side=observation['player'],
                         step=observation['step'], max_candidates=max_candidates)
    except _GuardStopped:
        raise
    except Exception:
        return {'status': 'unknown', 'reason': 'reference_invalid'}

    candidates = record['inputs']['candidates']
    ids = [c['id'] for c in candidates]
    teacher_index = parsed['teacher_choice_label']
    if ids[teacher_index] != expected_label:
        return {'status': 'unknown', 'reason': 'teacher_identity_mismatch'}
    teacher = candidates[teacher_index]
    positive = np.zeros(len(ids), dtype=bool)
    verified = np.zeros(len(ids), dtype=bool)
    positive[teacher_index] = verified[teacher_index] = True
    alternatives, different_ids, unknown_ids = [], [], []
    for candidate_id in _pick_alternatives(candidates, teacher):
        _guard(guard)
        entry = dict(candidate_id=candidate_id, request='unknown', reservations='unknown',
                     crop_records='unknown', outcome='unknown')
        try:
            next_action, next_trace, next_crop_prefix = _execute(
                agent_factory, observation, configuration, ordinal, candidate_id, guard)
            target = next_trace['records'][ordinal]
            if (not _same_ordered(_prefix(trace['records'], ordinal), _prefix(next_trace['records'], ordinal))
                    or not _same_ordered(crop_prefix, next_crop_prefix)
                    or not _same_ordered(record['inputs'], target['inputs'])
                    or target.get('teacher_label') != expected_label):
                raise ValueError('Target prefix differs')
            entry['request'] = 'same' if same_request(action, next_action) else 'different'
            entry['reservations'] = ('same' if _same_ordered(trace['final_reservations'],
                                        next_trace['final_reservations']) else 'different')
            entry['crop_records'] = ('same' if _same_ordered(trace['crop_records'],
                                         next_trace['crop_records']) else 'different')
            if entry['request'] == 'different':
                entry['outcome'] = 'different_request'
                different_ids.append(candidate_id)
                verified[ids.index(candidate_id)] = True
            elif entry['reservations'] == entry['crop_records'] == 'same':
                entry['outcome'] = 'same_under_teacher_continuation'
                positive[ids.index(candidate_id)] = verified[ids.index(candidate_id)] = True
            else:
                unknown_ids.append(candidate_id)
        except _GuardStopped:
            raise
        except Exception:
            unknown_ids.append(candidate_id)
        alternatives.append(entry)
        _guard(guard)
    shared = np.broadcast_to(parsed['shared'], (len(ids), 275))
    features = np.concatenate((shared, parsed['candidates']), axis=1).astype(np.float32, copy=False)
    result = dict(status='ok', features=features, teacher_index=int(teacher_index), weight=1.0,
                  positive_mask=positive, verified_mask=verified,
                  positive_ids=[ids[i] for i in np.flatnonzero(positive)],
                  different_request_ids=different_ids, unknown_ids=unknown_ids,
                  alternatives=alternatives,
                  interpretation='Conditional current-turn behavior only; different is not a value judgment.')
    if retain_reference:
        prefix = tuple(expected_teacher_prefix)
        reference_action, reference_trace, reference_crops = deepcopy(action), deepcopy(trace), deepcopy(crop_prefix)
        identity = _reference_hash(ordinal, expected_label, prefix, reference_action, reference_trace, reference_crops)
        result['reference'] = _TeacherReference(ordinal, expected_label, prefix, reference_action,
                                                reference_trace, reference_crops, identity)
    return result


def evaluate_choice(observation, configuration, ordinal, *, expected_inputs, expected_label,
                    expected_teacher_prefix, reference, chosen_id, agent_factory=task_agent_factory,
                    guard=lambda: None):
    """Compare one model choice with a previously validated private reference.

    At most one new controller invocation occurs. The teacher choice directly
    matches its validated reference; another candidate is forced once, followed
    by the frozen teacher. A changed complete request or final reservation is
    ``different``; no value or future-return judgment follows. Any non-resource
    exception, unavailable ID, mutated reference or mismatching target prefix is
    ``unknown``. Returned fields contain no candidate IDs or private state.
    """
    called = False

    def result(outcome):
        return {'outcome': outcome, 'controller_called': called}

    def mark_called():
        nonlocal called
        called = True

    try:
        if (not callable(guard) or not callable(agent_factory) or type(ordinal) is not int or ordinal < 0
                or type(expected_label) is not str or not expected_label
                or type(chosen_id) is not str or not chosen_id
                or type(expected_teacher_prefix) not in (list, tuple)
                or len(expected_teacher_prefix) != ordinal
                or any(type(label) is not str or not label for label in expected_teacher_prefix)
                or not isinstance(reference, _TeacherReference)
                or reference.ordinal != ordinal or reference.teacher_label != expected_label
                or reference.teacher_prefix != tuple(expected_teacher_prefix)):
            return result('unknown')
        _guard(guard)
        if (_reference_hash(reference.ordinal, reference.teacher_label, reference.teacher_prefix,
                            reference.action, reference.trace, reference.crop_prefix) != reference.integrity):
            return result('unknown')
        view = observe(observation, configuration)
        if (not same_request(view.observation, observation)
                or not same_request(view.configuration, configuration)):
            return result('unknown')
        observation, configuration = view.observation, view.configuration
        record = reference.trace['records'][ordinal]
        inputs = record['inputs']
        if (record['chosen'] != expected_label or record['teacher_label'] != expected_label
                or not same_request(inputs, expected_inputs)
                or not same_request(inputs['observation'], observation)
                or _item_order(inputs['observation']) != _item_order(observation)
                or not same_request(inputs['configuration'], configuration)
                or any(prior.get('chosen') != label or prior.get('teacher_label') != label
                       for prior, label in zip(reference.trace['records'][:ordinal], expected_teacher_prefix))):
            return result('unknown')
        ids = [candidate['id'] for candidate in inputs['candidates']]
        if len(ids) != len(set(ids)) or ids.count(chosen_id) != 1:
            return result('unknown')
        if chosen_id == expected_label:
            return result('match')
        next_action, next_trace, next_crop_prefix = _execute(
            agent_factory, observation, configuration, ordinal, chosen_id, guard, on_called=mark_called)
        if (_reference_hash(reference.ordinal, reference.teacher_label, reference.teacher_prefix,
                            reference.action, reference.trace, reference.crop_prefix) != reference.integrity):
            return result('unknown')
        target = next_trace['records'][ordinal]
        if (not _same_ordered(_prefix(reference.trace['records'], ordinal),
                              _prefix(next_trace['records'], ordinal))
                or not _same_ordered(reference.crop_prefix, next_crop_prefix)
                or not _same_ordered(inputs, target['inputs'])
                or target.get('teacher_label') != expected_label):
            return result('unknown')
        if (not same_request(reference.action, next_action)
                or not _same_ordered(reference.trace['final_reservations'],
                                     next_trace['final_reservations'])):
            return result('different')
        return result('match')
    except _GuardStopped:
        raise
    except Exception:
        return result('unknown')
