"""Project an authenticated private development campaign to public numbers.

The caller must authenticate the campaign, all 24 accepted metrics and all 16
comparison records against their original receipts and local files. This module
does not open files, replay games, or establish provenance. It validates their
coordinates and recomputes the recorded summaries before constructing a fresh
allowlisted output. Input seed values never become output identifiers.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from statistics import mean

from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS, POLICIES
from learning.task_pair_dev_metrics_v1.panel import METRICS, _game, paired


COMPARISONS = (('baseline', 'control'), ('control', 'auxiliary'))
SIDES = (0, 1)
_STOPS = frozenset((None, 'frame_difference', 'input_difference', 'choice_difference',
                    'callback_count_difference', 'joint_action_difference'))
_TASKS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
_REQUEST_GROUPS = _TASKS + ('movement', 'PASS', 'other')
_STATUSES = ('executed', 'partial', 'no_effect', 'blocked', 'pass')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric value')
    return value


def _json_snapshot(value):
    """Normalize integer keys exactly as the private JSON writer does."""
    try:
        return json.loads(json.dumps(value, allow_nan=False, separators=(',', ':')))
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Invalid summary value') from None


def _typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (left.keys() == right.keys()
                and all(_typed_equal(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _seed_roles(seeds):
    _require(type(seeds) is dict and set(seeds) == set(ALIASES), 'Invalid seed roles')
    values = [seeds[alias] for alias in ALIASES]
    _require(all(type(seed) is int and 0 <= seed < 2**32 for seed in values)
             and len(set(values)) == len(values), 'Invalid seed roles')
    return values, dict(zip(values, ALIASES))


def _coordinates(alias, opponent, side, arm):
    _require(type(alias) is str and alias in ALIASES and type(opponent) is str
             and opponent in OPPONENTS and type(side) is int and side in SIDES
             and type(arm) is str and arm in POLICIES, 'Invalid game coordinate')
    return alias, opponent, side, arm


def _action_group(value):
    """Map arbitrary requested/effective opcodes to fixed public categories."""
    if value in _TASKS:
        return value
    if value in ('NORTH', 'SOUTH', 'EAST', 'WEST'):
        return 'movement'
    if value == 'PASS':
        return 'PASS'
    return 'other'


def _decision_counts(result, side):
    """Reconcile and project events for selected tasks only, never events."""
    events, counts = result.get('events'), result.get('selection_counts')
    _require(type(events) is list and type(counts) is dict
             and all(type(k) is str and type(v) is int and v >= 0 for k, v in counts.items()),
             'Invalid selection counts')
    callback_count = counts.get('model_callbacks')
    opportunities = counts.get('model_callback_multi_unit_opportunities')
    selected_opportunities = counts.get('model_callback_selected_multi_unit_opportunities')
    _require(type(callback_count) is int and type(opportunities) is int
             and type(selected_opportunities) is int
             and 0 <= selected_opportunities <= opportunities <= callback_count,
             'Invalid model opportunity counts')
    expected = Counter(dict(model_callbacks=0,
                            model_callback_multi_unit_opportunities=opportunities,
                            model_callback_selected_multi_unit_opportunities=selected_opportunities))
    selected = {op: 0 for op in _TASKS}
    requests = {op: 0 for op in _REQUEST_GROUPS}
    effective = {op: 0 for op in _REQUEST_GROUPS}
    statuses = {status: 0 for status in _STATUSES}
    source = {'model_callback': 0, 'teacher': 0}
    target = {'observed_true': 0, 'observed_false': 0,
              'unknown_with_label': 0, 'unknown_unmatched': 0}
    target_by_task = {op: dict.fromkeys(target, 0) for op in _TASKS}
    seen, matched, unmatched = set(), 0, 0
    for event in events:
        _require(type(event) is dict and type(event.get('step')) is int
                 and 0 <= event['step'] < 719 and type(event.get('player')) is int
                 and event['player'] == side and type(event.get('unit')) is int
                 and event['unit'] >= 0, 'Invalid selected task coordinate')
        coordinate = event['step'], event['player'], event['unit']
        _require(coordinate not in seen, 'Duplicate selected task')
        seen.add(coordinate)
        origin, op, request = event.get('source'), event.get('selected_op'), event.get('request_op')
        _require(origin in ('callback', 'teacher') and type(op) is str and op in _TASKS
                 and type(request) is str and bool(request), 'Invalid selected task')
        selected[op] += 1
        requests[_action_group(request)] += 1
        expected['selected:' + op] += 1
        expected['request:' + request] += 1
        if origin == 'callback':
            source['model_callback'] += 1
            expected['model_callbacks'] += 1
        else:
            source['teacher'] += 1
        label, actual, status, effect = (event.get('label_support'), event.get('effective_op'),
                                         event.get('status'), event.get('same_target_atomic_effect'))
        if label == 'unknown':
            _require(actual is None and status is None and effect is None, 'Unmatched label has an effect')
            unmatched += 1
            expected['unmatched_unit_label'] += 1
            target['unknown_unmatched'] += 1
            target_by_task[op]['unknown_unmatched'] += 1
        else:
            _require(label == 'observed' and type(actual) is str and bool(actual)
                     and status in _STATUSES and (effect is None or type(effect) is bool),
                     'Invalid official unit label')
            _require(not (actual == 'invalid' and effect is not None), 'Invalid target effect')
            if effect is True:
                _require(actual == op and status in ('executed', 'partial'), 'Contradictory target effect')
                expected['same_target_atomic_effect'] += 1
                target['observed_true'] += 1
                target_by_task[op]['observed_true'] += 1
            elif effect is False:
                target['observed_false'] += 1
                target_by_task[op]['observed_false'] += 1
            else:
                expected['target_effect_unknown'] += 1
                target['unknown_with_label'] += 1
                target_by_task[op]['unknown_with_label'] += 1
            matched += 1
            statuses[status] += 1
            effective[_action_group(actual)] += 1
            expected['unit_status:' + status] += 1
    coverage = 'observed' if events and not unmatched else 'unknown'
    _require(result.get('execution_support') == coverage and dict(expected) == counts,
             'Selection, request or execution counts differ')
    return dict(scope='selected_tasks_only', total=len(events), by_source=source,
                selected_task=selected, submitted_unit_request=requests,
                official_unit_labels=dict(coverage=coverage, matched=matched, unmatched=unmatched,
                                          status=statuses, effective_action=effective,
                                          same_target_atomic_effect=target,
                                          same_target_atomic_effect_by_selected_task=target_by_task))


def _validated_rows(rows, seeds):
    _require(type(rows) is list and len(rows) == 24, 'Expected 24 metrics rows')
    indexed, economic, decisions = {}, {}, {}
    for row in rows:
        _require(type(row) is dict and row.get('schema') == 'task-pair-dev-game-metrics-v1',
                 'Invalid metrics row')
        alias, opponent, side, arm = _coordinates(row.get('seed_alias'), row.get('opponent'),
                                                  row.get('side'), row.get('arm'))
        key = alias, opponent, side, arm
        game = f'dev-{alias}-{opponent}-{side}-{arm}'
        _require(type(row.get('seed')) is int and row['seed'] == seeds[alias]
                 and row.get('game') == game and key not in indexed, 'Metrics identity differs')
        result = row.get('result')
        _require(type(result) is dict and type(result.get('official_turns')) is int
                 and result['official_turns'] == 719, 'Invalid game result')
        decisions[key] = _decision_counts(result, side)
        metrics = _game(row)
        _require(set(metrics) == set(METRICS), 'Invalid economic fields')
        indexed[key], economic[key] = row, metrics
    expected = {(a, o, s, p) for a in ALIASES for o in OPPONENTS for s in SIDES for p in POLICIES}
    _require(set(indexed) == expected, 'Incomplete game coordinates')
    return indexed, economic, decisions


def _validated_comparisons(comparisons, seeds, rows):
    _require(type(comparisons) is list and len(comparisons) == 16,
             'Expected 16 common-history comparisons')
    indexed = {}
    for item in comparisons:
        _require(type(item) is dict, 'Invalid comparison')
        reference, candidate = item.get('reference'), item.get('candidate')
        _require((reference, candidate) in COMPARISONS, 'Invalid comparison arms')
        alias, opponent, side, _ = _coordinates(item.get('seed_alias'), item.get('opponent'),
                                                 item.get('side'), reference)
        key = reference, candidate, alias, opponent, side
        _require(type(item.get('seed')) is int and item['seed'] == seeds[alias] and key not in indexed,
                 'Comparison identity differs')
        result = item.get('result')
        _require(type(result) is dict and result.get('comparison_complete') is True
                 and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
                 and type(result.get('matched_multi_unit_opportunities')) is int
                 and 0 <= result['matched_multi_unit_opportunities'] <= result['matched_calls']
                 and result.get('stop_reason') in _STOPS and type(result.get('stopped')) is bool
                 and result['stopped'] == (result['stop_reason'] is not None),
                 'Incomplete common-history comparison')
        count = result['matched_multi_unit_opportunities']
        for arm in (reference, candidate):
            own_count = rows[(alias, opponent, side, arm)]['result']['selection_counts']
            _require(count <= own_count['model_callback_multi_unit_opportunities'],
                     'Matched opportunities exceed a single arm')
        for field in ('first_choice_disagreement', 'matched_call_request_differences',
                      'matched_selected_multi_unit', 'matched_selected_different_unit'):
            _require(type(result.get(field)) is int and 0 <= result[field] <= result['matched_calls'],
                     'Invalid comparison counts')
        _require(result['first_choice_disagreement'] <= 1
                 and result['first_choice_disagreement'] == int(result['stop_reason'] == 'choice_difference')
                 and result['matched_selected_different_unit'] <= result['matched_selected_multi_unit']
                 and result['matched_selected_multi_unit'] <= count,
                 'Comparison counts contradict opportunities')
        indexed[key] = item
    expected = {(r, c, a, o, s) for r, c in COMPARISONS
                for a in ALIASES for o in OPPONENTS for s in SIDES}
    _require(set(indexed) == expected, 'Incomplete comparison coordinates')
    return indexed


def _metrics(values):
    _require(type(values) is dict and all(key in values for key in METRICS), 'Missing metrics')
    return {key: _number(values[key]) for key in METRICS}


def _summary(value):
    _require(type(value) is dict and value.get('support') == 'observed'
             and type(value.get('available_pairs')) is int
             and value.get('available_pairs') == value.get('expected_pairs'),
             'Incomplete paired summary')
    return dict(support='observed', pairs=value['available_pairs'], values=_metrics(value['values']))


def _support(value, seeds):
    _require(type(value) is dict and type(value.get('per_seed')) is dict, 'Missing opportunity support')
    per_seed = {}
    for alias in ALIASES:
        count = value['per_seed'].get(seeds[alias])
        _require(type(count) is int and count >= 0, 'Missing opportunity count')
        per_seed[alias] = count
    total = sum(per_seed.values())
    supported = total >= 20 and all(value >= 5 for value in per_seed.values())
    _require(value.get('min_per_seed') == 5 and type(value.get('min_per_seed')) is int
             and value.get('min_total') == 20 and type(value.get('min_total')) is int
             and type(value.get('total')) is int and value['total'] == total
             and value.get('support') == ('supported' if supported else 'unknown'),
             'Opportunity support differs')
    return dict(support=value['support'], per_seed=per_seed, total=total,
                min_per_seed=5, min_total=20)


def project(campaign, *, seeds, rows, comparisons):
    """Return new public-safe descriptive data, never private identifiers.

    ``rows`` and ``comparisons`` are authenticated caller-supplied objects;
    this function cannot prove their source. A caller must not treat a passing
    projection as proof of correct worker execution or independent progress.
    """
    seed_values, aliases = _seed_roles(seeds)
    row_index, economics, decisions = _validated_rows(rows, seeds)
    comparison_index = _validated_comparisons(comparisons, seeds, row_index)
    _require(type(campaign) is dict and campaign.get('schema') == 'task-pair-dev-campaign-result-v1'
             and campaign.get('passed') is True and type(campaign.get('games')) is int
             and campaign['games'] == 24 and type(campaign.get('official_replay_passes')) is int
             and campaign['official_replay_passes'] == 24
             and type(campaign.get('seed_clusters')) is int and campaign['seed_clusters'] == 2
             and campaign.get('independent_improvement') is False and campaign.get('promotion') is False,
             'Campaign result differs')
    receipts = campaign.get('receipts')
    _require(type(receipts) is list and len(receipts) == 24
             and all(type(item) is dict and type(item.get('game')) is str for item in receipts)
             and {item['game'] for item in receipts} == {row['game'] for row in rows},
             'Campaign receipt coordinates differ')
    saved = campaign.get('summaries')
    _require(type(saved) is dict and set(saved) == {'baseline_to_control', 'control_to_auxiliary'},
             'Campaign comparisons differ')
    details, summaries = [], {}
    for reference, candidate in COMPARISONS:
        key = reference + '_to_' + candidate
        selected = [row for row in rows if row['arm'] in (reference, candidate)]
        selected_comparisons = [item for item in comparisons
                                if (item['reference'], item['candidate']) == (reference, candidate)]
        computed = paired(selected, seeds=seed_values, opponents=OPPONENTS,
                          reference=reference, candidate=candidate, comparisons=selected_comparisons)
        _require(_typed_equal(_json_snapshot(computed), saved[key]), 'Stored paired summary differs')
        _require(computed['expected_games'] == computed['observed_games'] == 16
                 and computed['seed_clusters'] == 2 and computed['statistical_significance'] == 'unknown'
                 and computed['independently_verified_improvement'] is False
                 and computed['promotion'] is False, 'Paired result differs')
        support = computed['opportunity_support']
        summaries[key] = dict(reference=reference, candidate=candidate,
            overall=_summary(computed['overall']),
            by_opponent={opponent: _summary(computed['by_opponent'][opponent]) for opponent in OPPONENTS},
            by_seed={alias: _summary(computed['by_seed'][seeds[alias]]) for alias in ALIASES},
            by_side={str(side): _summary(computed['by_side'][side]) for side in SIDES},
            opportunity_support={name: _support(support[name], seeds)
                                 for name in ('reference', 'candidate', 'same_input_pairs')})
        for item in computed['pairs']:
            _require(item.get('support') == 'observed' and type(item.get('seed')) is int
                     and item['seed'] in aliases, 'Incomplete paired coordinate')
            alias, opponent, side = aliases[item['seed']], item['opponent'], item['side']
            pair = comparison_index[(reference, candidate, alias, opponent, side)]['result']
            details.append(dict(comparison=key, seed_alias=alias, opponent=opponent, side=side,
                                reference=reference, candidate=candidate,
                                reference_metrics=_metrics(item['reference']),
                                candidate_metrics=_metrics(item['candidate']), delta=_metrics(item['delta']),
                                common_history=dict(matched_calls=pair['matched_calls'],
                                    matched_multi_unit_opportunities=pair['matched_multi_unit_opportunities'],
                                    first_choice_disagreement=pair['first_choice_disagreement'],
                                    matched_call_request_differences=pair['matched_call_request_differences'],
                                    matched_selected_multi_unit=pair['matched_selected_multi_unit'],
                                    matched_selected_different_unit=pair['matched_selected_different_unit'],
                                    stop_reason=pair['stop_reason'])))
    games = []
    for alias in ALIASES:
        for opponent in OPPONENTS:
            for side in SIDES:
                for arm in POLICIES:
                    key = alias, opponent, side, arm
                    counts = row_index[key]['result']['selection_counts']
                    games.append(dict(seed_alias=alias, opponent=opponent, side=side, arm=arm,
                                      metrics=_metrics(economics[key]),
                                      decision_counts=decisions[key],
                                      model_callbacks=counts['model_callbacks'],
                                      model_callback_multi_unit_opportunities=
                                          counts['model_callback_multi_unit_opportunities']))
    groups = {}
    for arm in POLICIES:
        groups[arm] = {}
        for label in ('all', *OPPONENTS):
            selected = [game for game in games if game['arm'] == arm
                        and (label == 'all' or game['opponent'] == label)]
            _require(len(selected) == (8 if label == 'all' else 4), 'Incomplete arm group')
            scores = [game['metrics']['score'] for game in selected]
            groups[arm][label] = dict(games=len(selected), wins=scores.count(1.),
                draws=scores.count(.5), losses=scores.count(0.),
                mean_metrics={key: mean(game['metrics'][key] for game in selected) for key in METRICS})
    return dict(schema='task-pair-dev-public-projection-v1', scope='consumed_development',
                games=games, paired=details, summaries=summaries, by_arm=groups,
                metric_scope=dict(decision_counts='selected_tasks_only',
                                  successful_plants='all_official_units'),
                seed_clusters=2, statistical_significance='unknown',
                independently_verified_improvement=False, promotion=False)
