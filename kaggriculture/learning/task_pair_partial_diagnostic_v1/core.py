"""Project one independently authenticated, already accepted replay pair.

This module executes no agents, official transitions or training and performs
no filesystem IO. Its caller must authenticate both metrics, replays, traces,
model/opponent provenance and the previous failed-stage evidence. Output is
strictly allowlisted; no input seed, identity, path, action or observation is
returned. The result describes one consumed role/seat, not a holdout or a test
of statistical significance. The two full-game task and request distributions
describe different trajectories after divergence: they are not same-input or
causal comparisons. Request counts cover the selected tasks, not every unit.
"""
from contextlib import closing
import math
import re

from learning.task_pair_dev_metrics_v1.panel import _game
from learning.task_pair_dev_metrics_v1.replay_pair import compare_replays
from learning.task_pair_historical_compare_v1.projection import _decision_counts, _common
from learning.task_pair_historical_compare_v1.request_prefix import compare as compare_requests


_HEX = re.compile(r'[0-9a-f]{64}\Z')
_STOPS = frozenset((None, 'frame_difference', 'input_difference', 'choice_difference',
                    'callback_count_difference', 'joint_action_difference'))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Invalid economic value')
    return value


def _identity(value):
    _require(type(value) is str and _HEX.fullmatch(value) is not None, 'Invalid local identity')
    return value


def _coordinates(rows):
    _require(type(rows) is dict and set(rows) == {'baseline', 'control'}, 'Expected one policy pair')
    for arm in ('baseline', 'control'):
        row = rows[arm]
        _require(type(row) is dict and row.get('schema') == 'task-pair-dev-game-metrics-v1'
                 and row.get('game') == f'dev-S02-market-0-{arm}'
                 and row.get('seed_alias') == 'S02' and row.get('opponent') == 'market'
                 and type(row.get('side')) is int and row['side'] == 0 and row.get('arm') == arm
                 and type(row.get('seed')) is int and 0 <= row['seed'] < 2**32,
                 'Unexpected game coordinate')
        for key in ('task_checkpoint_sha256', 'crop_checkpoint_sha256', 'trace_sha256', 'replay_sha256'):
            _identity(row.get(key))
    reference, candidate = rows['baseline'], rows['control']
    _require(reference['seed'] == candidate['seed']
             and reference['crop_checkpoint_sha256'] == candidate['crop_checkpoint_sha256']
             and reference['task_checkpoint_sha256'] != candidate['task_checkpoint_sha256'],
             'Game pair identity differs')


def _financial(row, replay):
    metrics = _game(row)
    _require(type(replay) is dict and type(replay.get('steps')) is list
             and len(replay['steps']) == 720, 'Incomplete replay')
    first, last = replay['steps'][0], replay['steps'][-1]
    _require(type(first) is list and len(first) == 2 and type(last) is list and len(last) == 2
             and all(type(state) is dict for state in first + last)
             and [state.get('status') for state in last] == ['DONE', 'DONE'], 'Invalid replay boundary')
    initial = [_number(farm['money']) for farm in first[0]['observation']['farms']]
    final = [_number(farm['money']) for farm in last[0]['observation']['farms']]
    _require(len(initial) == len(final) == 2
             and [_number(state['reward']) for state in last] == final,
             'Replay rewards and balances differ')
    own_net, other_net = final[0] - initial[0], final[1] - initial[1]
    margin = final[0] - final[1]
    score = 1. if margin > 0 else .5 if margin == 0 else 0.
    recorded = row['result']['economy']
    _require((recorded['own']['initial_cash'], recorded['opponent']['initial_cash'],
              recorded['own']['final_cash'], recorded['opponent']['final_cash'])
             == (initial[0], initial[1], final[0], final[1]),
             'Replay and accepted cash balances differ')
    _require((metrics['own_net_profit'], metrics['opponent_net_profit'], metrics['margin'], metrics['score'])
             == (own_net, other_net, margin, score), 'Replay and accepted economy differ')
    return dict(own_final_cash=final[0], opponent_final_cash=final[1], own_net_profit=own_net,
                opponent_net_profit=other_net, margin=margin, score=score,
                outcome='win' if score == 1. else 'draw' if score == .5 else 'loss')


def _matched(value, decisions, rows):
    result = value.get('result') if type(value) is dict else None
    _require(type(result) is dict and result.get('comparison_complete') is True
             and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
             and type(result.get('stopped')) is bool and result.get('stop_reason') in _STOPS
             and result['stopped'] == (result['stop_reason'] is not None), 'Invalid common-history result')
    calls, opportunities = result['matched_calls'], result.get('matched_multi_unit_opportunities')
    _require(type(opportunities) is int and 0 <= opportunities <= calls
             and result.get('support') == ('observed' if calls else 'unknown'),
             'Invalid common-history opportunities')
    for key in ('first_choice_disagreement', 'matched_call_request_differences',
                'matched_selected_multi_unit', 'matched_selected_different_unit'):
        _require(type(result.get(key)) is int and 0 <= result[key] <= calls,
                 'Invalid common-history count')
    _require(result['first_choice_disagreement'] == int(result['stop_reason'] == 'choice_difference')
             and result['matched_selected_different_unit'] <= result['matched_selected_multi_unit'] <= opportunities,
             'Inconsistent common-history result')
    for arm in ('baseline', 'control'):
        _require(calls <= decisions[arm]['total']
                 and opportunities <= rows[arm]['result']['selection_counts']
                 ['model_callback_multi_unit_opportunities'], 'Matched calls exceed game selections')
    return _common(result)


def _guarded(iterator, check):
    for item in iterator:
        _require(check() is True, 'Analysis resource check failed')
        yield item
    _require(check() is True, 'Analysis resource check failed')


def analyze(*, rows, replays, traces, check):
    """Return only a safe descriptive result for one preauthenticated pair.

    ``rows``/``replays``/``traces`` each map baseline and control; traces are
    bounded, hash-checking iterators and are consumed to EOF and closed.
    ``check`` enforces the caller's resource/time guard and must return True.
    Identity and official 719-step replay acceptance remain caller duties.
    """
    _require(callable(check) and check() is True, 'Analysis resource check failed')
    _coordinates(rows)
    _require(type(replays) is dict and set(replays) == {'baseline', 'control'}
             and type(traces) is dict and set(traces) == {'baseline', 'control'},
             'Incomplete paired inputs')
    decisions = {arm: _decision_counts(rows[arm]['result'], 0) for arm in ('baseline', 'control')}
    _require(all(decisions[arm]['by_source']['teacher'] == 0 for arm in decisions),
             'Unexpected rule-only task source')
    games = {arm: _financial(rows[arm], replays[arm]) for arm in ('baseline', 'control')}
    for arm in ('baseline', 'control'):
        games[arm]['selected_task_counts'] = dict(decisions[arm]['selected_task'])
        games[arm]['submitted_unit_request_counts'] = dict(decisions[arm]['submitted_unit_request'])
    _require(check() is True, 'Analysis resource check failed')
    with closing(traces['baseline']), closing(traces['control']):
        compared = compare_replays(replays['baseline'], replays['control'],
            _guarded(traces['baseline'], check), _guarded(traces['control'], check),
            reference='baseline', candidate='control', seed=rows['baseline']['seed'],
            seed_alias='S02', opponent='market', side=0)
    _require(check() is True, 'Analysis resource check failed')
    matched = _matched(compared, decisions, rows)
    request = compare_requests(replays['baseline'], replays['control'], side=0)
    _require(check() is True, 'Analysis resource check failed')
    delta = {key: games['control'][key] - games['baseline'][key]
             for key in ('own_net_profit', 'opponent_net_profit', 'margin', 'score')}
    return dict(schema='task-pair-partial-posthoc-result-v1', scope='consumed_development_posthoc',
                seed_alias='S01', opponent='market', side=0, accepted_games=2, paired_games=1,
                reference='baseline', candidate='control', games=games, delta=delta,
                task_common_history=matched, complete_request_prefix=request,
                decision_distribution_scope='selected_tasks_full_game_different_states_descriptive',
                original_official_replays_reused=2, new_games=0, new_official_transitions=0,
                new_training_updates=0, independently_verified_improvement=False,
                statistical_significance='unknown', promotion=False)
