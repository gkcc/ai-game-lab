"""First complete-request difference on an authenticated common replay prefix.

Caller must authenticate both completed official replays, role/seat mapping and
their prior acceptance independently. This module neither opens files nor runs
models, games or the official replay. It returns no seed, action, observation,
path or identifier; a detected association is not a causal explanation. It
validates complete replay structure and every action; finite, order-aware state
comparison covers only the common prefix, including the terminal frame only
when there is no earlier split. The outer official acceptance is responsible
for the validity of states outside that prefix.
"""
from decision.semantics import clean
from learning.task_pair_dev_metrics_v1.consumer import OFFICIAL_TURNS, _bytes
from learning.task_pair_dev_metrics_v1.replay_pair import _replay
from learning.task_request_gate_v1.policy import same_request


def _require(value, message):
    if not value:
        raise ValueError(message)


def _action(value):
    _require(type(value) is dict and set(value) == {'farmer', 'hands', 'market'}
             and all(type(value[name]) is list for name in ('farmer', 'hands', 'market')),
             'Invalid complete request')
    _bytes(value)  # finite JSON-domain values; no request is returned
    return value


def _result(status, reason, step=None):
    return dict(scope='same_complete_history', status=status, reason=reason, step=step)


def compare(reference_replay, candidate_replay, *, side):
    """Compare one baseline/control replay pair, without revealing its inputs.

    The official replay frame at index ``step`` is the shared pre-action state;
    that turn's requests are in frame ``step + 1``. Strict typed/order-aware
    state comparison excludes framework overage time. Request comparison uses
    the existing same_request semantics: mapping key order is immaterial, but
    scalar types and every list's order (including market orders) are retained.

    Return ``observed`` only for an actor-request difference when the opponent
    request and all preceding complete frames/requests still match. An earlier
    state split, opponent split or simultaneous split is ``unknown``. ``no_difference``
    means all 719 joint requests and the final frame matched in this one pair;
    it does not prove the policies equivalent in other contexts. Incomplete
    replay structure, mismatched configuration and invalid actions raise.
    State finiteness and ordering are checked only in the common prefix; the
    terminal state is checked only if no earlier split occurred. Official
    acceptance by the caller must validate all remaining states.
    """
    _require(type(side) is int and side in (0, 1), 'Invalid actor seat')
    _require(type(reference_replay) is dict and type(candidate_replay) is dict,
             'Expected replay objects')
    reference_info, candidate_info = reference_replay.get('info'), candidate_replay.get('info')
    _require(type(reference_info) is dict and type(candidate_info) is dict
             and type(reference_info.get('seed')) is int and 0 <= reference_info['seed'] < 2**32
             and type(candidate_info.get('seed')) is int
             and candidate_info['seed'] == reference_info['seed'], 'Replay roles differ')
    reference = _replay(reference_replay, reference_info['seed'])
    candidate = _replay(candidate_replay, reference_info['seed'])
    _require(_bytes(reference_replay['configuration']) == _bytes(candidate_replay['configuration']),
             'Replay configurations differ')
    for frames in (reference, candidate):
        for frame in frames[1:]:
            for state in frame:
                _action(state['action'])
    for step in range(OFFICIAL_TURNS):
        # This intentionally preserves ordered objects/inventories, so the
        # result may become unknown for order-only frame differences.
        if _bytes(clean(reference[step])) != _bytes(clean(candidate[step])):
            return _result('unknown', 'frame_difference')
        before = [state['action'] for state in reference[step + 1]]
        after = [state['action'] for state in candidate[step + 1]]
        actor_changed = not same_request(before[side], after[side])
        opponent_changed = not same_request(before[1 - side], after[1 - side])
        if opponent_changed:
            reason = 'simultaneous_request_difference' if actor_changed else 'opponent_request_difference'
            return _result('unknown', reason)
        if actor_changed:
            return _result('observed', 'actor_request_difference', step)
    if _bytes(clean(reference[-1])) != _bytes(clean(candidate[-1])):
        return _result('unknown', 'terminal_frame_difference')
    return _result('no_difference', 'all_requests_equal')
