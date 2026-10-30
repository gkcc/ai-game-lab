"""Compare authenticated saved recordings without invoking either policy.

The caller must authenticate each replay, trace, game identity, official
acceptance and model provenance before invoking this function. Recorded model
inputs are checked against each other, not reconstructed from replay states.
No raw state, action, candidate, seed or private identity is returned. A
divergence in one historical pair is descriptive, not evidence of causation.
"""
from contextlib import ExitStack

from learning.task_pair_dev_metrics_v1.consumer import (
    CommonHistoryComparison, OFFICIAL_TURNS, _bytes, _turn,
)
from learning.task_pair_dev_metrics_v1.replay_pair import _replay, _trace_end, _trace_line
from learning.task_pair_historical_compare_v1.request_prefix import compare as request_prefix
from learning.task_request_gate_v1.policy import same_request


def _guard(check):
    if check() is not True:
        raise ValueError('Analysis resource guard stopped')


def _choice_detail(left, right, left_action, right_action, side):
    """Return the first different selection with strictly identical inputs."""
    for a, b in zip(left, right):
        if _bytes(a['inputs']) != _bytes(b['inputs']):
            raise ValueError('Selection inputs diverged before choice')
        if a['chosen'] != b['chosen']:
            return dict(
                baseline_op=a['op'], control_op=b['op'],
                same_unit=a['unit'] == b['unit'], same_target=a['target'] == b['target'],
                selected_unit_request_ordered_equal=_bytes(a['unit_request']) == _bytes(b['unit_request']),
                complete_actor_request_equal=same_request(left_action[side], right_action[side]),
                complete_opponent_request_equal=same_request(left_action[1-side], right_action[1-side]),
                both_model_callbacks=a['source'] == b['source'] == 'callback',
            )
    raise ValueError('Expected a recorded choice difference')


def compare(reference_replay, candidate_replay, reference_trace, candidate_trace, *, seed, side, check):
    """Return an allowlisted comparison after consuming both traces through EOF.

    The full two-player frame at index t must match before selection records at
    t can be compared. Frame t+1 stores the submitted requests for that turn.
    Comparison stops at the first ordered frame, model-input, selection-count,
    selection or joint-request difference; every subsequent trace turn is still
    validated and consumed, including iterator EOF integrity checks. The
    independent request-prefix helper uses semantic request equality, which can
    be less strict than this ordered joint representation comparison. The
    matched-selection count includes the first record whose choice differs.

    The caller's iterator must validate compressed identities at EOF. The
    caller's guard is sampled on each turn and around the request-prefix helper;
    one helper call is not itself interrupted. No filesystem IO is performed.
    """
    if (type(side) is not int or side not in (0, 1) or not callable(check)
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('Invalid comparison parameters')
    _guard(check)
    ref = _replay(reference_replay, seed)
    new = _replay(candidate_replay, seed)
    if _bytes(reference_replay['configuration']) != _bytes(candidate_replay['configuration']):
        raise ValueError('Replay configurations differ')

    comparison = CommonHistoryComparison(side)
    stop_step, first_choice = None, None
    # Closing each iterator also closes the file owned by its caller on error.
    with ExitStack() as stack:
        ref_trace, new_trace = iter(reference_trace), iter(candidate_trace)
        for iterator in (ref_trace, new_trace):
            close = getattr(iterator, 'close', None)
            if callable(close):
                stack.callback(close)
        for step in range(OFFICIAL_TURNS):
            _guard(check)
            ref_line, new_line = _trace_line(ref_trace, step), _trace_line(new_trace, step)
            # Unlike the existing pairing helper, validate every later record
            # too, even once comparison has stopped on a common-history split.
            left = _turn(ref_line, step, side)
            right = _turn(new_line, step, side)
            if not comparison.stopped:
                ref_actions = [state['action'] for state in ref[step + 1]]
                new_actions = [state['action'] for state in new[step + 1]]
                comparison.observe(step, ref[step], new[step], ref_line, new_line,
                                   ref_actions, new_actions)
                if comparison.stopped:
                    stop_step = step
                    if comparison.reason == 'choice_difference':
                        first_choice = _choice_detail(left, right, ref_actions, new_actions, side)
            _guard(check)
        _trace_end(ref_trace)
        _trace_end(new_trace)
        _guard(check)

    comparison_result = comparison.result()
    if not comparison_result['comparison_complete']:
        raise ValueError('Incomplete common-history comparison')
    _guard(check)
    requests = request_prefix(reference_replay, candidate_replay, side=side)
    _guard(check)
    return dict(
        scope='strict_common_recorded_history', trace_turns_per_game=OFFICIAL_TURNS,
        matched_selection_records=comparison_result['matched_calls'],
        stop_reason=comparison_result['stop_reason'], first_stop_step=stop_step,
        first_different_recorded_choice=first_choice,
        complete_request_prefix=requests,
    )
