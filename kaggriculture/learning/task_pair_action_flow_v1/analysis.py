"""Describe selection differences on a common, previously accepted history.

The caller authenticates replay/trace identities, the official acceptance and
both model identities before calling. Compact states use the existing v2
semantic digest; equality is not a new official replay, byte equality, or a
claim about hidden policy state. The caller's trace iterators must validate
their compressed identity at EOF. No game, policy, model or training executes.
"""
from contextlib import ExitStack

from learning.task_action_impact_diagnostic_v1.analysis import (
    bank_comparison, compare_callbacks, descriptor, validate_trace,
)
from learning.task_pair_dev_metrics_v1.consumer import _turn
from learning.task_pair_dev_metrics_v1.replay_pair import _trace_end, _trace_line
from learning.task_prefix_diagnostic_v1.pairing import (
    COMPONENTS, TURNS, _check_compact, _first_branch, _selected_multiset,
)


def _guard(check):
    if check() is not True:
        raise ValueError('Analysis resource guard stopped')


def _eligible(kind, branch_step, step):
    if kind == 'state':
        return step < branch_step
    if kind in ('actor', 'opponent', 'simultaneous'):
        return step <= branch_step
    return True


def _window(candidate, reference, side, index):
    value = bank_comparison(candidate, reference, side, index)
    return dict(state_index=index, semantic_physical_state_equal=value['observation_equal'],
                cash_delta=dict(value['delta']))


def compare_pair(reference_compact, candidate_compact, reference_trace, candidate_trace, *, side, check):
    """Return allowlisted per-turn counts through the first common-history split.

    Inputs are v2 compact replays produced by the existing ``compact_replay``;
    their index ``t`` represents state t and the complete request in frame
    ``t+1``. Trace iterators provide complete line dictionaries for each arm.
    The existing validator binds every line's selected tasks to its recorded
    observable state, configuration, inventory ordering and unit request.
    Both iterators are consumed through EOF, even after the prefix splits.
    Eligible selection records must be model callbacks for the same actor.

    Per turn, matching callback inputs are compared only until the first input
    mismatch, selected choice mismatch or count mismatch. A different choice
    with unchanged complete request does not stop comparison of later turns.
    The matched-context count includes the first differing choice when its
    recorded inputs match, and counts records rather than whole turns.
    The task multiset contains only recorded (unit, target, operation), not
    task completion or economic value. Equal task multisets and requests do not
    establish that sequential choices are interchangeable: later candidates and
    reservations may depend on earlier choices. Request-value equality compares
    the selected request values even when the selected units differ; it does not
    imply the same unit was selected. The actor-request change and whether the
    first request branch is actor-only are reported separately; simultaneous,
    opponent or state branches cannot establish actor-only attribution. The
    post-branch windows are full-policy observations, never causal estimates.
    """
    if type(side) is not int or side not in (0, 1) or not callable(check):
        raise ValueError('Invalid comparison parameters')
    _guard(check)
    _check_compact(reference_compact)
    _check_compact(candidate_compact)
    reference, candidate = reference_compact, candidate_compact
    kind, branch = _first_branch(candidate, reference, side)
    counts = dict(eligible_turns=0, actor_observation_or_configuration_mismatch_turns=0,
                  zero_both_selection_record_turns=0, input_difference_turns=0,
                  selection_record_count_difference_turns=0, equal_recorded_choices_turns=0,
                  choice_difference_turns=0, matched_context_selection_records=0,
                  choice_same_request_same_task_multiset_turns=0,
                  choice_same_request_different_task_multiset_turns=0,
                  choice_different_request_same_task_multiset_turns=0,
                  choice_different_request_different_task_multiset_turns=0)
    first_choice = None
    desc_ref, desc_new = descriptor(), descriptor()
    with ExitStack() as stack:
        refs, news = iter(reference_trace), iter(candidate_trace)
        for iterator in (refs, news):
            close = getattr(iterator, 'close', None)
            if callable(close):
                stack.callback(close)
        for step in range(TURNS):
            _guard(check)
            ref_line, new_line = _trace_line(refs, step), _trace_line(news, step)
            ref_records = validate_trace(ref_line, reference, step, desc_ref)
            new_records = validate_trace(new_line, candidate, step, desc_new)
            if not _eligible(kind, branch, step):
                continue
            _turn(ref_line, step, side)
            _turn(new_line, step, side)
            if any(record.get('source') != 'callback' for record in ref_records + new_records):
                raise ValueError('Unexpected non-model selection record')
            counts['eligible_turns'] += 1
            if (candidate['actor_observation'][step] != reference['actor_observation'][step]
                    or candidate['configuration'][step] != reference['configuration'][step]):
                counts['actor_observation_or_configuration_mismatch_turns'] += 1
                continue
            if not new_records and not ref_records:
                counts['zero_both_selection_record_turns'] += 1
                continue
            comparison = compare_callbacks(new_records, ref_records)
            counts['matched_context_selection_records'] += comparison['matched_context_calls']
            choice = comparison['choice']
            if choice is not None:
                counts['choice_difference_turns'] += 1
                same_request = candidate['requests'][step][side] == reference['requests'][step][side]
                same_tasks = _selected_multiset(new_records) == _selected_multiset(ref_records)
                key = ('choice_' + ('same' if same_request else 'different') + '_request_'
                       + ('same' if same_tasks else 'different') + '_task_multiset_turns')
                counts[key] += 1
                if first_choice is None:
                    first_choice = dict(step=step, selection_record_index=choice['callback'],
                        baseline_op=choice['reference_op'], control_op=choice['candidate_op'],
                        same_unit=choice['same_unit'], same_target=choice['same_target'],
                        selected_request_values_equal=choice['emitted_unit_request_equal'],
                        complete_actor_request_equal=same_request, task_multiset_equal=same_tasks,
                        label_interchangeability='not_inferred')
            elif comparison['context_stop'] == 'input_difference':
                counts['input_difference_turns'] += 1
            elif comparison['context_stop'] == 'callback_count_difference':
                counts['selection_record_count_difference_turns'] += 1
            elif comparison['context_stop'] is None:
                counts['equal_recorded_choices_turns'] += 1
            else:
                raise ValueError('Unknown selection-record comparison')
            _guard(check)
        _trace_end(refs)
        _trace_end(news)
        _guard(check)
    categories = ('actor_observation_or_configuration_mismatch_turns', 'zero_both_selection_record_turns',
                  'input_difference_turns', 'selection_record_count_difference_turns',
                  'equal_recorded_choices_turns', 'choice_difference_turns')
    matrix = ('choice_same_request_same_task_multiset_turns',
              'choice_same_request_different_task_multiset_turns',
              'choice_different_request_same_task_multiset_turns',
              'choice_different_request_different_task_multiset_turns')
    if (sum(counts[key] for key in categories) != counts['eligible_turns']
            or sum(counts[key] for key in matrix) != counts['choice_difference_turns']
            or desc_ref['trace_turns'] != TURNS or desc_new['trace_turns'] != TURNS):
        raise ValueError('Comparison counters do not reconcile')
    components = None
    if kind in ('actor', 'opponent', 'simultaneous'):
        components = {key: candidate['components'][branch][key] != reference['components'][branch][key]
                      for key in COMPONENTS}
    windows = None
    if kind != 'none':
        windows = {str(n): _window(candidate, reference, side, min(branch + n, TURNS)) for n in (1, 24)}
        windows['terminal'] = _window(candidate, reference, side, TURNS)
    return dict(scope='v2_semantic_common_history', trace_turns_per_game=TURNS,
                first_branch_kind=kind, first_branch_step=branch,
                actor_request_changed_at_first_branch=(True if kind in ('actor', 'simultaneous') else
                    False if kind == 'opponent' else None),
                actor_only_branch_attribution=('observed' if kind == 'actor' else
                    'no_branch' if kind == 'none' else 'unknown'),
                actor_request_components_differ=components, counts=counts,
                first_same_input_choice_difference=first_choice, post_branch_windows=windows,
                interpretation='Descriptive full-policy observations; no causal return attribution.')
