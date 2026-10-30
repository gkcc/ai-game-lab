"""Consume authenticated task traces and official execution labels in memory.

This code does not authenticate its arguments, replay the official engine, read
files or run a policy.  The caller must independently authenticate the trace,
model and complete official replay and pass their verified audit result here.
All returned rows, identities, event counts and economics are private.
"""
from collections import Counter
from copy import deepcopy
import json
import math

from decision.semantics import clean
from learning.bc_crop_economic_diagnostic.analysis import EconomicConsumer
from learning.task_bc_error_diagnostics_v1.shadow import _typed


OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
UNIT_STATUS = ('executed', 'partial', 'no_effect', 'blocked', 'pass')
OFFICIAL_TURNS = 719


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    """Exact bytes of the existing order- and type-preserving JSON encoding."""
    def valid(v):
        kind = type(v)
        if kind in (type(None), bool, int, str):
            return True
        if kind is float:
            return math.isfinite(v)
        if kind is list:
            return all(valid(item) for item in v)
        if kind is dict:
            return all(type(key) is str and valid(item) for key, item in v.items())
        return False
    _require(valid(value), 'Non-JSON comparison value')
    return json.dumps(_typed(value), ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _point(value):
    return (type(value) is list and len(value) == 2
            and all(type(item) is int and item >= 0 for item in value))


def _op(action):
    if type(action) is list and action and type(action[0]) is str and action[0]:
        return action[0]
    return 'invalid'


def _groups(candidates):
    groups, identities = {}, set()
    for item in candidates:
        unit, target, op = item.get('unit'), item.get('target'), item.get('op')
        _require(type(unit) is int and unit >= 0 and _point(target)
                 and type(op) is str and op in OPS, 'Invalid candidate descriptor')
        group = (op, target[0], target[1])
        identity = (unit, *group)
        _require(identity not in identities, 'Duplicate semantic candidate')
        identities.add(identity)
        groups.setdefault(group, set()).add(unit)
    return groups


def _turn(trace, step, player):
    _require(type(trace) is dict and trace.get('step') == step
             and type(trace.get('step')) is int, 'Trace turn differs')
    _require(all(type(trace.get(k)) is int and trace[k] == 0
                 for k in ('load_errors', 'inference_errors', 'fallbacks')),
             'Trace reports an error')
    records = trace.get('records')
    _require(type(records) is list, 'Missing task records')
    units = set()
    for record in records:
        _require(type(record) is dict and record.get('source') in ('callback', 'teacher'),
                 'Unknown selection source')
        inputs = record.get('inputs')
        _require(type(inputs) is dict and type(inputs.get('observation')) is dict
                 and type(inputs.get('configuration')) is dict, 'Incomplete model inputs')
        obs = inputs['observation']
        _require(type(obs.get('player')) is int and obs['player'] == player
                 and type(obs.get('step')) is int and obs['step'] == step,
                 'Actor observation differs')
        _bytes(inputs)  # Validate finite JSON and preserve all input ordering.
        candidates = inputs.get('candidates')
        _require(type(candidates) is list and bool(candidates)
                 and all(type(c) is dict and type(c.get('id')) is str for c in candidates),
                 'Invalid candidates')
        ids = [c['id'] for c in candidates]
        _require(len(ids) == len(set(ids)) and record.get('chosen') in ids
                 and record.get('teacher_label') in ids, 'Invalid choice identity')
        _groups(candidates)
        choice = candidates[ids.index(record['chosen'])]
        unit, target, op = choice.get('unit'), choice.get('target'), choice.get('op')
        _require(type(unit) is int and unit >= 0 and unit not in units
                 and _point(target) and op in OPS, 'Invalid selected task')
        _require(type(record.get('unit')) is int and record['unit'] == unit
                 and _bytes(record.get('target')) == _bytes(target)
                 and record.get('op') == op and type(record.get('unit_request')) is list,
                 'Selected descriptor or unit request differs')
        _bytes(record['unit_request'])
        units.add(unit)
    return records


class GameConsumer:
    """Join one actor's task selections to official labels by step/player/unit.

    Feed each authenticated actor trace once via ``trace_turn`` and each
    official reexecution label (for both players) via ``sink``. Their order of
    arrival is immaterial. A missing unit label remains unknown. A contradictory
    trace/label, duplicate event or incomplete 719-turn run raises instead.
    """

    def __init__(self, player):
        _require(type(player) is int and player in (0, 1), 'Invalid actor')
        self.player = player
        self.economy = EconomicConsumer()
        self.turns = set()
        self.selected = {}
        self.unit_labels = {}
        self.market_labels = set()
        self.market_counts = [Counter(), Counter()]
        self.label_counts = Counter()
        self.failed = False
        self.finished = False

    def _open(self):
        _require(not self.failed and not self.finished, 'Consumer is closed')

    def trace_turn(self, trace):
        self._open()
        try:
            _require(type(trace) is dict and type(trace.get('step')) is int,
                     'Invalid trace turn')
            step = trace['step']
            _require(0 <= step < OFFICIAL_TURNS and step not in self.turns,
                     'Duplicate or out-of-range trace turn')
            records = _turn(trace, step, self.player)
            for record in records:
                key = (step, self.player, record['unit'])
                groups = _groups(record['inputs']['candidates'])
                selected_group = (record['op'], *record['target'])
                self.selected[key] = dict(source=record['source'], op=record['op'],
                    target=list(record['target']), request=deepcopy(record['unit_request']),
                    any_multi_unit=any(len(units) >= 2 for units in groups.values()),
                    selected_multi_unit=len(groups[selected_group]) >= 2)
            self.turns.add(step)
        except BaseException:
            self.failed = True
            raise

    def sink(self, event):
        self._open()
        # ExecutionTrace.unit emits the failure label while handling the
        # original official exception, then re-raises it.  Do not replace that
        # exception with a consumer error; permanently invalidate this result.
        if type(event) is dict and event.get('status') == 'error':
            self.failed = True
            return
        try:
            _require(type(event) is dict and type(event.get('step')) is int
                     and 0 <= event['step'] < OFFICIAL_TURNS, 'Invalid label turn')
            player, phase = event.get('player'), event.get('phase')
            _require(type(player) is int and player in (0, 1)
                     and phase in ('unit', 'market'), 'Invalid label identity')
            if phase == 'unit':
                unit = event.get('unit')
                _require(type(unit) is int and unit >= 0
                         and event.get('status') in UNIT_STATUS, 'Invalid unit label')
                key = (event['step'], player, unit)
                _require(key not in self.unit_labels, 'Duplicate unit label')
                self.unit_labels[key] = deepcopy({field: event.get(field) for field in
                    ('requested', 'effective', 'status', 'position', 'successful_plant')})
            else:
                order = event.get('order')
                _require(type(order) is int and order >= 0, 'Invalid market order identity')
                key = (event['step'], player, order)
                _require(key not in self.market_labels, 'Duplicate market label')
                self.market_labels.add(key)
                opcode = _op(event.get('requested'))
                counts = self.market_counts[player]
                counts['orders:' + opcode] += 1
                wanted = event.get('requested_units')
                if event.get('status') == 'ignored' or event.get('invalid') is True or wanted is None:
                    counts['unknown_requested_units_orders:' + opcode] += 1
                else:
                    _require(type(wanted) is int and wanted >= 0, 'Invalid requested market units')
                    counts['known_requested_units:' + opcode] += wanted
            self.economy.consume(event)
            self.label_counts[phase + ':' + event['status']] += 1
        except BaseException:
            self.failed = True
            raise

    def finish(self, initial_observations, final_observations, statuses, audit):
        self._open()
        try:
            _require(self.turns == set(range(OFFICIAL_TURNS)), 'Incomplete trace')
            _require(type(audit) is dict and audit.get('passed') is True
                     and audit.get('official_transitions_reexecuted') == OFFICIAL_TURNS
                     and audit.get('both_public_and_private_states_match_each_step') is True
                     and audit.get('inventory_order_checked_each_step') is True,
                     'Complete official replay certificate required')
            execution = audit.get('execution')
            _require(type(execution) is dict and execution.get('label_version') == 2
                     and type(execution.get('counts')) is dict
                     and all(type(k) is str and type(v) is int and v >= 0
                             for k, v in execution['counts'].items())
                     and execution['counts'] == dict(self.label_counts),
                     'Official execution label counts differ')
            _require(statuses == ['DONE', 'DONE'], 'Unfinished game')
            players = self.economy.finish(initial_observations, final_observations)
            _require(audit.get('final_rewards') == [p['final_cash'] for p in players],
                     'Rewards and cash differ')
            joined, counts = [], Counter()
            counts.update(model_callbacks=0, model_callback_multi_unit_opportunities=0,
                          model_callback_selected_multi_unit_opportunities=0)
            for (step, player, unit), selected in sorted(self.selected.items()):
                event = self.unit_labels.get((step, player, unit))
                row = dict(step=step, player=player, unit=unit, source=selected['source'],
                           selected_op=selected['op'], request_op=_op(selected['request']))
                counts['selected:' + selected['op']] += 1
                counts['request:' + row['request_op']] += 1
                if selected['source'] == 'callback':
                    counts['model_callbacks'] += 1
                    counts['model_callback_multi_unit_opportunities'] += int(selected['any_multi_unit'])
                    counts['model_callback_selected_multi_unit_opportunities'] += int(selected['selected_multi_unit'])
                if event is None:
                    row.update(label_support='unknown', effective_op=None,
                               status=None, same_target_atomic_effect=None)
                    counts['unmatched_unit_label'] += 1
                else:
                    _require(_bytes(event.get('requested')) == _bytes(selected['request']),
                             'Trace request differs from official label')
                    effective, status = _op(event.get('effective')), event['status']
                    position = event.get('position')
                    target_effect = None
                    if _point(position) and effective != 'invalid':
                        target_effect = (position == selected['target'] and effective == selected['op']
                                         and status in ('executed', 'partial'))
                        if selected['op'] == 'PLANT' and target_effect:
                            target_effect = event.get('successful_plant') is True
                    row.update(label_support='observed', effective_op=effective,
                               status=status, same_target_atomic_effect=target_effect)
                    counts['unit_status:' + status] += 1
                    if target_effect is None:
                        counts['target_effect_unknown'] += 1
                    elif target_effect:
                        counts['same_target_atomic_effect'] += 1
                joined.append(row)
            side = self.player
            own, opponent = players[side], players[1-side]
            margin = own['final_cash'] - opponent['final_cash']
            result = dict(actor=side, events=joined, selection_counts=dict(counts),
                          execution_support=('unknown' if not joined or counts['unmatched_unit_label'] else 'observed'),
                          economy=dict(own=own, opponent=opponent),
                          market_requests=dict(own=dict(self.market_counts[side]),
                                               opponent=dict(self.market_counts[1-side])),
                          margin=margin, score=1. if margin > 0 else .5 if margin == 0 else 0.,
                          official_turns=OFFICIAL_TURNS,
                          interpretation='Atomic effects are not task-duration or profit causation.')
            self.finished = True
            return result
        except BaseException:
            self.failed = True
            raise


class CommonHistoryComparison:
    """Compare same-input callbacks only on a matching complete history prefix.

    ``before`` is the supplied complete two-player replay frame, and ``actions``
    the joint official request for the current turn. The caller must verify
    both against authenticated replays; this class cannot establish that origin.
    No later turns are compared after any input, choice, action or frame split.
    """

    def __init__(self, player):
        _require(type(player) is int and player in (0, 1), 'Invalid actor')
        self.player, self.next_step = player, 0
        self.stopped = False
        self.reason = None
        self.matched_calls = 0
        self.choice_differences = 0
        self.request_differences = 0
        self.matched_multi_unit_opportunities = 0
        self.matched_selected_multi_unit = 0
        self.matched_selected_different_unit = 0

    def observe(self, step, left_before, right_before, left_trace, right_trace,
                left_actions, right_actions):
        _require(not self.stopped, 'Comparison already diverged')
        try:
            _require(type(step) is int and step == self.next_step and step < OFFICIAL_TURNS,
                     'Non-contiguous comparison turn')
            for frame in (left_before, right_before):
                _require(type(frame) is list and len(frame) == 2
                         and all(type(state) is dict and type(state.get('observation')) is dict
                                 for state in frame)
                         and type(frame[0]['observation'].get('step')) is int
                         and frame[0]['observation']['step'] == step, 'Invalid replay frame')
            for actions in (left_actions, right_actions):
                _require(type(actions) is list and len(actions) == 2
                         and all(type(action) is dict
                                 and set(action) == {'farmer', 'hands', 'market'}
                                 and all(type(action[k]) is list for k in action)
                                 for action in actions), 'Invalid joint requests')
            self.next_step += 1
            # Framework timing is not game state.  Apply the existing replay
            # clean() only to complete frames, then preserve *all* remaining
            # object and inventory order in the comparison encoding.
            if _bytes(clean(left_before)) != _bytes(clean(right_before)):
                self.stopped, self.reason = True, 'frame_difference'
                return
            left = _turn(left_trace, step, self.player)
            right = _turn(right_trace, step, self.player)
            for a, b in zip(left, right):
                if _bytes(a['inputs']) != _bytes(b['inputs']):
                    self.stopped, self.reason = True, 'input_difference'
                    break
                self.matched_calls += 1
                self.request_differences += int(_bytes(a['unit_request']) != _bytes(b['unit_request']))
                if a['source'] == b['source'] == 'callback':
                    groups = _groups(a['inputs']['candidates'])
                    self.matched_multi_unit_opportunities += int(any(len(units) >= 2 for units in groups.values()))
                    if a['op'] == b['op'] and a['target'] == b['target']:
                        multiple = len(groups[(a['op'], *a['target'])]) >= 2
                        self.matched_selected_multi_unit += int(multiple)
                        self.matched_selected_different_unit += int(multiple and a['unit'] != b['unit'])
                if a['chosen'] != b['chosen']:
                    self.choice_differences += 1
                    self.stopped, self.reason = True, 'choice_difference'
                    break
            if not self.stopped and len(left) != len(right):
                self.stopped, self.reason = True, 'callback_count_difference'
            if not self.stopped and _bytes(left_actions) != _bytes(right_actions):
                self.stopped, self.reason = True, 'joint_action_difference'
        except BaseException:
            self.stopped, self.reason = True, 'invalid_input'
            raise

    def result(self):
        valid = self.reason != 'invalid_input'
        return dict(support='observed' if valid and self.matched_calls else 'unknown',
                    comparison_complete=valid and (self.stopped or self.next_step == OFFICIAL_TURNS),
                    matched_calls=self.matched_calls if valid else None,
                    first_choice_disagreement=self.choice_differences if valid else None,
                    matched_call_request_differences=self.request_differences if valid else None,
                    matched_multi_unit_opportunities=self.matched_multi_unit_opportunities if valid else None,
                    matched_selected_multi_unit=self.matched_selected_multi_unit if valid else None,
                    matched_selected_different_unit=self.matched_selected_different_unit if valid else None,
                    stopped=self.stopped, stop_reason=self.reason,
                    interpretation='Typed ordered input bytes on common history only; not causal return.')
