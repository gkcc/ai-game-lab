"""Strict, in-memory projection of an authenticated eight-game development panel.

The caller must authenticate the fixed registration, role mapping, each accepted
game and each persisted comparison receipt, including both model and replay
identities. This module neither reads files nor independently proves provenance,
official execution or genuine historical seed identity. It never publishes raw
seeds, paths, identities, observations, events, candidate data or requests.

The old common-history comparator stops immediately on the first different task
choice. Its output cannot establish whether the *complete* submitted request at
that same turn changed, or identify the stop turn. Such cases remain unknown.
"""
from __future__ import annotations

import json
import math
import re
from statistics import mean

from learning.task_pair_dev_metrics_v1.panel import METRICS, _game, paired
from learning.task_pair_dev_campaign_v1.projection import _decision_counts


INTERNAL_TO_HISTORY = {'S02': 'S01', 'S03': 'S04'}
SIDES = (0, 1)
ARMS = ('baseline', 'control')
OPPONENT = 'market'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_STOPS = frozenset((None, 'frame_difference', 'input_difference', 'choice_difference',
                    'callback_count_difference', 'joint_action_difference'))
_ECONOMIC = ('own_net_profit', 'opponent_net_profit', 'margin', 'score',
             'own_sales_units', 'opponent_sales_units', 'own_sales_revenue',
             'opponent_sales_revenue', 'own_ending_warehouse_units',
             'opponent_ending_warehouse_units')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    _require(type(value) is str and _HEX.fullmatch(value) is not None, 'Invalid bound identity')
    return value


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric value')
    return value


def _typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_typed_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _json_snapshot(value):
    """Match persisted JSON while rejecting ambiguous stringified dict keys."""
    def check(item):
        if type(item) is dict:
            _require(all(type(key) in (str, int) for key in item), 'Invalid summary key')
            keys = [str(key) for key in item]
            _require(len(keys) == len(set(keys)), 'Ambiguous summary key')
            for child in item.values():
                check(child)
        elif type(item) in (list, tuple):
            for child in item:
                check(child)
    try:
        check(value)
        return json.loads(json.dumps(value, allow_nan=False, separators=(',', ':')))
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Invalid paired result') from None


def _seeds(seeds):
    _require(type(seeds) is dict and set(seeds) == set(INTERNAL_TO_HISTORY)
             and all(type(value) is int and 0 <= value < 2**32 for value in seeds.values())
             and len(set(seeds.values())) == 2, 'Invalid bound roles')
    return [seeds[alias] for alias in INTERNAL_TO_HISTORY]


def _rows(rows, seeds):
    _require(type(rows) is list and len(rows) == 8, 'Eight accepted metrics required')
    found, economics, decisions, identities = {}, {}, {}, {}
    for row in rows:
        _require(type(row) is dict and type(row.get('schema')) is str
                 and row['schema'] == 'task-pair-dev-game-metrics-v1',
                 'Invalid metrics row')
        alias, opponent, side, arm = (row.get('seed_alias'), row.get('opponent'),
                                      row.get('side'), row.get('arm'))
        _require(type(alias) is str and alias in INTERNAL_TO_HISTORY and type(opponent) is str
                 and opponent == OPPONENT
                 and type(side) is int and side in SIDES and type(arm) is str and arm in ARMS
                 and type(row.get('seed')) is int and row['seed'] == seeds[alias],
                 'Unexpected metrics coordinate')
        key = alias, side, arm
        _require(key not in found and type(row.get('game')) is str
                 and row['game'] == f'dev-{alias}-{opponent}-{side}-{arm}',
                 'Duplicate or misplaced metrics')
        for name in ('task_checkpoint_sha256', 'crop_checkpoint_sha256', 'trace_sha256', 'replay_sha256'):
            _digest(row.get(name))
        identities.setdefault(arm, set()).add(row['task_checkpoint_sha256'])
        identities.setdefault('crop', set()).add(row['crop_checkpoint_sha256'])
        result = row.get('result')
        _require(type(result) is dict and type(result.get('official_turns')) is int
                 and result['official_turns'] == 719, 'Incomplete official metrics')
        decision = _decision_counts(result, side)
        _require(decision['by_source']['teacher'] == 0, 'Unexpected rule-only task source')
        metrics = _game(row)
        _require(set(metrics) == set(METRICS), 'Incomplete economic metrics')
        for perspective in ('own', 'opponent'):
            _number(result['economy'][perspective]['initial_cash'])
            _number(result['economy'][perspective]['final_cash'])
        found[key], economics[key], decisions[key] = row, metrics, decision
    expected = {(alias, side, arm) for alias in INTERNAL_TO_HISTORY for side in SIDES for arm in ARMS}
    _require(set(found) == expected and all(len(identities[key]) == 1 for key in (*ARMS, 'crop'))
             and identities['baseline'] != identities['control'], 'Incomplete or inconsistent model roles')
    return found, economics, decisions


def _comparisons(receipts, seeds, rows, decisions):
    _require(type(receipts) is list and len(receipts) == 4, 'Four comparison receipts required')
    found, internal = {}, []
    for receipt in receipts:
        _require(type(receipt) is dict and type(receipt.get('schema')) is str
                 and receipt['schema'] == 'task-pair-dev-comparison-v1'
                 and type(receipt.get('comparison')) is dict, 'Invalid comparison receipt')
        item = receipt['comparison']
        alias, side = item.get('seed_alias'), item.get('side')
        _require(type(alias) is str and alias in INTERNAL_TO_HISTORY and type(side) is int and side in SIDES
                 and type(item.get('seed')) is int and item['seed'] == seeds[alias]
                 and type(item.get('opponent')) is str and item['opponent'] == OPPONENT
                 and type(item.get('reference')) is str and item['reference'] == 'baseline'
                 and type(item.get('candidate')) is str and item['candidate'] == 'control',
                 'Comparison coordinate differs')
        key = alias, side
        _require(key not in found, 'Duplicate comparison')
        reference, candidate = rows[(alias, side, 'baseline')], rows[(alias, side, 'control')]
        _require(receipt.get('reference_game') == reference['game']
                 and receipt.get('candidate_game') == candidate['game']
                 and receipt.get('reference_replay_sha256') == reference['replay_sha256']
                 and receipt.get('candidate_replay_sha256') == candidate['replay_sha256']
                 and receipt.get('reference_trace_sha256') == reference['trace_sha256']
                 and receipt.get('candidate_trace_sha256') == candidate['trace_sha256'],
                 'Comparison receipt is not bound to both games')
        result = item.get('result')
        _require(type(result) is dict and result.get('comparison_complete') is True
                 and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
                 and type(result.get('stopped')) is bool and result.get('stop_reason') in _STOPS
                 and result['stopped'] == (result['stop_reason'] is not None),
                 'Incomplete common history')
        calls, opportunities = result['matched_calls'], result.get('matched_multi_unit_opportunities')
        _require(type(opportunities) is int and 0 <= opportunities <= calls, 'Invalid opportunities')
        for name in ('first_choice_disagreement', 'matched_call_request_differences',
                     'matched_selected_multi_unit', 'matched_selected_different_unit'):
            _require(type(result.get(name)) is int and 0 <= result[name] <= calls,
                     'Invalid comparison count')
        _require(result['first_choice_disagreement'] == int(result['stop_reason'] == 'choice_difference')
                 and result['matched_selected_different_unit'] <= result['matched_selected_multi_unit'] <= opportunities
                 and result.get('support') == ('observed' if calls else 'unknown'),
                 'Comparison support contradicts counts')
        for arm in ARMS:
            decision = decisions[(alias, side, arm)]
            _require(calls <= decision['total']
                     and opportunities <= rows[(alias, side, arm)]['result']['selection_counts']
                     ['model_callback_multi_unit_opportunities'], 'Comparison exceeds observed opportunities')
        found[key] = result
        internal.append(item)
    _require(set(found) == {(alias, side) for alias in INTERNAL_TO_HISTORY for side in SIDES},
             'Incomplete comparison coordinates')
    return found, internal


def _public_game(row, metrics, decision):
    result = row['result']
    output = {name: _number(metrics[name]) for name in _ECONOMIC}
    output['own_final_cash'] = _number(result['economy']['own']['final_cash'])
    output['opponent_final_cash'] = _number(result['economy']['opponent']['final_cash'])
    output['outcome'] = 'win' if output['score'] == 1 else 'draw' if output['score'] == .5 else 'loss'
    output['decision_counts'] = decision
    return output


def _common(result):
    calls, reason = result['matched_calls'], result['stop_reason']
    choice = ('observed' if reason == 'choice_difference' else
              'none_in_matched_calls' if calls else 'unknown')
    # An earlier choice/input/frame divergence can preempt this check even at
    # the very same turn; only the joint request is available from this API.
    # The old comparator is deliberately more strict than same_request:
    # e.g. mapping key order alone can produce joint_action_difference.
    # A representation difference therefore cannot prove a semantic request
    # difference, or which player differed.
    joint = 'none_in_complete_history' if reason is None else 'unknown'
    ordered_difference = (True if reason == 'joint_action_difference' else
                          False if reason is None else None)
    actor = 'none_in_complete_history' if reason is None else 'unknown'
    return dict(scope='strict_common_history_only', matched_selection_records=calls,
                first_task_choice_difference=choice,
                matching_record_unit_request_differences=result['matched_call_request_differences'],
                matched_multi_unit_opportunities=result['matched_multi_unit_opportunities'],
                matched_selected_multi_unit=result['matched_selected_multi_unit'],
                matched_selected_different_unit=result['matched_selected_different_unit'],
                stop_reason=reason, first_joint_request_difference=joint,
                ordered_joint_representation_difference=ordered_difference,
                first_actor_request_difference=actor, first_difference_step=None)


def project(paired_result, *, seeds, rows, comparisons):
    """Project eight accepted metric objects and four bound comparison receipts.

    ``seeds`` maps the old Panel's internal S02/S03 to caller-authenticated
    actual values corresponding to historical S01/S04. ``paired_result`` is
    either the direct result of ``panel.paired`` or its decoded JSON snapshot.
    ``comparisons`` are the four outer receipts produced by LocalDriver.compare.
    The caller must bind their inner objects to the coordinator's saved
    comparisons. Missing, duplicated, inconsistent or fabricated inputs raise,
    not fill zero.
    """
    values = _seeds(seeds)
    indexed, metrics, decisions = _rows(rows, seeds)
    comparison_index, comparisons_inner = _comparisons(comparisons, seeds, indexed, decisions)
    recomputed = paired(rows, seeds=values, opponents=[OPPONENT], reference='baseline',
                        candidate='control', comparisons=comparisons_inner)
    _require(_typed_equal(_json_snapshot(paired_result), _json_snapshot(recomputed))
             and recomputed['expected_games'] == recomputed['observed_games'] == 8
             and recomputed['seed_clusters'] == 2 and recomputed['statistical_significance'] == 'unknown'
             and recomputed['independently_verified_improvement'] is False and recomputed['promotion'] is False,
             'Paired result differs from accepted rows')

    pairs = []
    for alias, public_alias in INTERNAL_TO_HISTORY.items():
        for side in SIDES:
            reference = _public_game(indexed[(alias, side, 'baseline')], metrics[(alias, side, 'baseline')],
                                     decisions[(alias, side, 'baseline')])
            candidate = _public_game(indexed[(alias, side, 'control')], metrics[(alias, side, 'control')],
                                     decisions[(alias, side, 'control')])
            delta = {key: candidate[key] - reference[key] for key in _ECONOMIC}
            pairs.append(dict(seed_alias=public_alias, opponent=OPPONENT, side=side,
                              reference=reference, candidate=candidate, delta=delta,
                              common_history=_common(comparison_index[(alias, side)])))
    mean_keys = ('own_net_profit', 'opponent_net_profit', 'margin', 'score')
    def summary(items):
        _require(bool(items), 'Empty summary')
        outcomes = {arm: {value: sum(pair[arm]['outcome'] == value for pair in items)
                          for value in ('win', 'draw', 'loss')}
                    for arm in ('reference', 'candidate')}
        return dict(pairs=len(items), mean_delta={key: mean(pair['delta'][key] for pair in items)
                                                  for key in mean_keys}, outcomes=outcomes)
    return dict(schema='task-pair-historical-comparison-public-v1', scope='consumed_development',
                reference='baseline', candidate='control', opponent=OPPONENT, games=8,
                seed_clusters=2, pairs=pairs, overall=summary(pairs),
                by_seed={alias: summary([item for item in pairs if item['seed_alias'] == alias])
                         for alias in INTERNAL_TO_HISTORY.values()},
                by_side={str(side): summary([item for item in pairs if item['side'] == side]) for side in SIDES},
                statistical_significance='unknown', independently_verified_improvement=False,
                promotion=False, metric_scope='descriptive; selected-task telemetry only')
