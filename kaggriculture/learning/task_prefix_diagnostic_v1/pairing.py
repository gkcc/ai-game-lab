"""Describe saved-policy differences only while their official histories agree.

Inputs must already have passed the replay and trace validation in the caller.
Both neural selections and rule-teacher selections are selection records;
paired record counts do not claim equal numbers of model invocations.
Only counters, action categories and booleans leave this module. A matching v2
frame is semantic equality, not raw byte equality; matching task inputs does not
establish equal hidden policy state, action execution, completion or causation.
"""
from collections import Counter

from learning.task_action_impact_diagnostic_v1.analysis import (
    bank_comparison, compare_callbacks, first_request_difference,
)
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash


TURNS = 719
OPPONENTS = ('round2', 'market', 'teacher')
REFERENCES = ('unweighted', 'teacher_task')
COMPONENTS = ('farmer', 'hands', 'market')


def _check_compact(value):
    if not isinstance(value, dict):
        raise ValueError('Missing compact replay')
    for key in ('full', 'physical', 'banks'):
        if not isinstance(value.get(key), list) or len(value[key]) != TURNS + 1:
            raise ValueError('Invalid state count')
    for key in ('requests', 'components', 'actor_observation', 'configuration'):
        if not isinstance(value.get(key), list) or len(value[key]) != TURNS:
            raise ValueError('Invalid action turn count')
    if any(not isinstance(pair, list) or len(pair) != 2 for pair in value['requests']):
        raise ValueError('Invalid request pairs')
    if any(not isinstance(bank, list) or len(bank) != 2 for bank in value['banks']):
        raise ValueError('Invalid bank pairs')
    if any(not isinstance(item, dict) or set(item) != set(COMPONENTS)
           for item in value['components']):
        raise ValueError('Invalid actor request components')


def _first_branch(left, right, side):
    """Classify the first unequal semantic state or submitted requests."""
    for step in range(TURNS):
        if left['full'][step] != right['full'][step]:
            return 'state', step
        actor = left['requests'][step][side] != right['requests'][step][side]
        rival = left['requests'][step][1-side] != right['requests'][step][1-side]
        components = any(left['components'][step][key] != right['components'][step][key]
                         for key in COMPONENTS)
        if actor != components:
            raise ValueError('Actor request and components disagree')
        if actor or rival:
            kind = 'simultaneous' if actor and rival else 'actor' if actor else 'opponent'
            if kind == 'actor' and first_request_difference(left, right, side) != step:
                raise ValueError('First actor request cross-check failed')
            return kind, step
    if left['full'][-1] != right['full'][-1]:
        return 'terminal_state', TURNS
    if first_request_difference(left, right, side) is not None:
        raise ValueError('Equal request cross-check failed')
    return 'none', None


def _selected_multiset(records):
    # Hashes are only used in memory; neither targets nor hashes are output.
    return Counter(ordered_hash([record['unit'], record['target'], record['op']])
                   for record in records)


class PairComparison:
    """Consume all 719 paired, prevalidated trace records in turn order."""

    def __init__(self, candidate, reference, side, opponent, reference_name):
        if (type(side) is not int or side not in (0, 1) or opponent not in OPPONENTS
            or reference_name not in REFERENCES):
            raise ValueError('Invalid public pair identity')
        _check_compact(candidate)
        _check_compact(reference)
        self.candidate, self.reference, self.side = candidate, reference, side
        self.kind, self.step = _first_branch(candidate, reference, side)
        self.next_step = 0
        self.result = dict(side=side, opponent=opponent, reference=reference_name,
            first_branch_kind=self.kind, first_branch_step=self.step,
            branch_attribution='unknown', first_actor_request_components_differ=None,
            eligible_turns=0, actor_observation_or_configuration_mismatch_turns=0,
            zero_both_selection_record_turns=0, input_difference_turns=0,
            selection_record_count_difference_turns=0, equal_recorded_choices_turns=0,
            choice_difference_turns=0, choice_difference_same_actor_request_turns=0,
            choice_difference_different_actor_request_turns=0,
            choice_difference_same_request_same_task_multiset_turns=0,
            matched_context_selection_records=0, teacher_label_consistent_selection_records=0,
            teacher_label_disagreement_selection_records=0, candidate_teacher_agreement_selection_records=0,
            reference_teacher_agreement_selection_records=0, first_choice_difference=None,
            choice_difference_at_branch=None, post_branch_windows=None)
        if self.kind in ('actor', 'simultaneous'):
            self.result['first_actor_request_components_differ'] = {
                key: candidate['components'][self.step][key] != reference['components'][self.step][key]
                for key in COMPONENTS}

    def _eligible(self, step):
        if self.kind == 'state':
            return step < self.step
        if self.kind in ('actor', 'opponent', 'simultaneous'):
            # Both decisions observe the same recorded pre-action semantic state;
            # even when the rival also changes, consequences remain unknown.
            return step <= self.step
        return True

    def observe_turn(self, step, candidate_records, reference_records):
        if type(step) is not int or step != self.next_step or not 0 <= step < TURNS:
            raise ValueError('Trace turns must be consumed once in order')
        if not isinstance(candidate_records, list) or not isinstance(reference_records, list):
            raise ValueError('Missing prevalidated trace records')
        self.next_step += 1
        if not self._eligible(step):
            return
        a, b, out = self.candidate, self.reference, self.result
        out['eligible_turns'] += 1
        if (a['actor_observation'][step] != b['actor_observation'][step]
            or a['configuration'][step] != b['configuration'][step]):
            out['actor_observation_or_configuration_mismatch_turns'] += 1
            return
        if not candidate_records and not reference_records:
            out['zero_both_selection_record_turns'] += 1
            if step == self.step and self.kind in ('actor', 'opponent', 'simultaneous'):
                out['choice_difference_at_branch'] = False
            return
        # The historical helper's internal keys use "callback", but it compares
        # records identically whether their source is callback or teacher.
        comparison = compare_callbacks(candidate_records, reference_records)
        matched = comparison['matched_context_calls']
        out['matched_context_selection_records'] += matched
        for left, right in zip(candidate_records[:matched], reference_records[:matched]):
            if left['teacher_label'] != right['teacher_label']:
                out['teacher_label_disagreement_selection_records'] += 1
            else:
                out['teacher_label_consistent_selection_records'] += 1
                out['candidate_teacher_agreement_selection_records'] += left['chosen'] == left['teacher_label']
                out['reference_teacher_agreement_selection_records'] += right['chosen'] == right['teacher_label']
        choice = comparison['choice']
        if choice is not None:
            out['choice_difference_turns'] += 1
            same_request = a['requests'][step][self.side] == b['requests'][step][self.side]
            count_key = ('choice_difference_same_actor_request_turns' if same_request else
                         'choice_difference_different_actor_request_turns')
            out[count_key] += 1
            if same_request and _selected_multiset(candidate_records) == _selected_multiset(reference_records):
                out['choice_difference_same_request_same_task_multiset_turns'] += 1
            if out['first_choice_difference'] is None:
                out['first_choice_difference'] = dict(step=step, selection_record_index=choice['callback'],
                    candidate_op=choice['candidate_op'], reference_op=choice['reference_op'],
                    same_unit=choice['same_unit'], same_target=choice['same_target'],
                    candidate_request_kind=choice['candidate_request_kind'],
                    reference_request_kind=choice['reference_request_kind'],
                    selected_unit_request_equal=choice['emitted_unit_request_equal'],
                    complete_actor_request_equal=same_request)
        elif comparison['context_stop'] == 'input_difference':
            out['input_difference_turns'] += 1
        elif comparison['context_stop'] == 'callback_count_difference':
            out['selection_record_count_difference_turns'] += 1
        elif comparison['context_stop'] is None:
            out['equal_recorded_choices_turns'] += 1
        else:
            raise ValueError('Unknown selection-record comparison result')
        if step == self.step and self.kind in ('actor', 'opponent', 'simultaneous'):
            out['choice_difference_at_branch'] = choice is not None if comparison['context_stop'] is None else None

    def finish(self):
        if self.next_step != TURNS:
            raise ValueError('Trace turns incomplete')
        out = self.result
        categories = ('actor_observation_or_configuration_mismatch_turns', 'zero_both_selection_record_turns',
                      'input_difference_turns', 'selection_record_count_difference_turns',
                      'equal_recorded_choices_turns', 'choice_difference_turns')
        if (sum(out[key] for key in categories) != out['eligible_turns']
            or out['teacher_label_consistent_selection_records'] + out['teacher_label_disagreement_selection_records'] != out['matched_context_selection_records']
            or out['choice_difference_same_actor_request_turns'] + out['choice_difference_different_actor_request_turns'] != out['choice_difference_turns']):
            raise ValueError('Pair counters do not reconcile')
        if self.kind == 'actor':
            base = bank_comparison(self.candidate, self.reference, self.side, self.step)
            windows = {}
            for horizon in (1, 24):
                state = min(self.step + horizon, TURNS)
                later = bank_comparison(self.candidate, self.reference, self.side, state)
                windows[str(horizon)] = dict(state_index=state,
                    semantic_physical_state_equal=later['observation_equal'],
                    own_bank_delta_changed=later['delta']['own_bank'] != base['delta']['own_bank'],
                    opponent_bank_delta_changed=later['delta']['opponent_bank'] != base['delta']['opponent_bank'],
                    margin_delta_changed=later['delta']['margin'] != base['delta']['margin'])
            out['post_branch_windows'] = windows
        return dict(out)
