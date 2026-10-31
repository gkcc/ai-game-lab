"""Allowlisted descriptive projection of two authenticated 16-game cohorts.

The caller must authenticate the private seed binding, games, file identities,
comparison receipts and official replay checks. This module never opens a file,
runs a game or proves novelty. Raw seeds, paths, identities and actions are not
returned. Four seed clusters are an exploratory sample, not a significance test.
"""
import math
import re

from learning.task_pair_dev_metrics_v1.panel import _game


COHORTS = {'C01': {'S02': 'F01', 'S03': 'F02'},
           'C02': {'S02': 'F03', 'S03': 'F04'}}
OPPONENTS = ('market', 'round2')
ARMS = ('baseline', 'control')
SIDES = (0, 1)
FIELDS = ('own_net_profit', 'opponent_net_profit', 'margin', 'score',
          'own_sales_units', 'opponent_sales_units', 'own_sales_revenue',
          'opponent_sales_revenue', 'own_ending_warehouse_units',
          'opponent_ending_warehouse_units')
PREFIX_REASONS = {'observed': {'actor_request_difference'},
                  'no_difference': {'all_requests_equal'},
                  'unknown': {'frame_difference', 'opponent_request_difference',
                              'simultaneous_request_difference', 'terminal_frame_difference'}}
STOPS = frozenset((None, 'frame_difference', 'input_difference', 'choice_difference',
                   'callback_count_difference', 'joint_action_difference'))
HEX = re.compile(r'[0-9a-f]{64}\Z')


def require(condition):
    if not condition:
        raise ValueError('Fresh multi-opponent projection differs')


def _finite(value):
    require(type(value) in (int, float) and math.isfinite(value))
    return value


def _digest(value):
    require(type(value) is str and HEX.fullmatch(value) is not None)
    return value


def _prefix(value):
    require(type(value) is dict and set(value) == {'scope', 'status', 'reason', 'step'}
            and value['scope'] == 'same_complete_history' and value['status'] in PREFIX_REASONS
            and value['reason'] in PREFIX_REASONS[value['status']]
            and ((value['status'] == 'observed' and type(value['step']) is int
                  and 0 <= value['step'] < 719)
                 or (value['status'] != 'observed' and value['step'] is None)))
    return dict(value)


def _comparison(value, *, seed, alias, opponent, side):
    require(type(value) is dict and value.get('reference') == 'baseline'
            and value.get('candidate') == 'control' and value.get('seed_alias') == alias
            and type(value.get('seed')) is int and value['seed'] == seed
            and value.get('opponent') == opponent and type(value.get('side')) is int
            and value['side'] == side)
    result = value.get('result')
    require(type(result) is dict and result.get('comparison_complete') is True
            and type(result.get('matched_calls')) is int and result['matched_calls'] >= 0
            and type(result.get('matched_multi_unit_opportunities')) is int
            and 0 <= result['matched_multi_unit_opportunities'] <= result['matched_calls']
            and type(result.get('stopped')) is bool and result.get('stop_reason') in STOPS
            and result['stopped'] == (result['stop_reason'] is not None))
    return dict(scope='task_comparator_until_first_difference',
                matched_calls=result['matched_calls'], stop_reason=result['stop_reason'])


def _summary(pairs):
    require(type(pairs) is list and pairs)
    n = len(pairs)
    means = {key: sum(pair['delta'][key] for pair in pairs) / n for key in FIELDS}
    outcomes = {}
    for arm in ARMS:
        scores = [pair[arm]['score'] for pair in pairs]
        require(all(score in (0.0, 0.5, 1.0) for score in scores))
        outcomes[arm] = dict(wins=scores.count(1.0), draws=scores.count(0.5), losses=scores.count(0.0))
    return dict(pairs=n, mean_delta=means, outcomes=outcomes)


def _exploratory_screen(pairs):
    """Fixed gate for spending a later evaluation budget, never promotion."""
    overall = _summary(pairs)['mean_delta']
    guards, request_difference, opponent_profit = {}, {}, {}
    for role in ('F01', 'F02', 'F03', 'F04'):
        group = [pair for pair in pairs if pair['seed_alias'] == role]
        require(len(group) == 4)
        reference = sum(pair['baseline']['own_net_profit'] for pair in group) / len(group)
        delta = sum(pair['delta']['own_net_profit'] for pair in group) / len(group)
        guards[role] = 'unknown' if reference <= 0 else delta >= -0.01 * reference
        statuses = [pair['complete_request_prefix']['status'] for pair in group]
        request_difference[role] = (True if 'observed' in statuses else
                                    False if all(value == 'no_difference' for value in statuses)
                                    else 'unknown')
    for opponent in OPPONENTS:
        group = [pair for pair in pairs if pair['opponent'] == opponent]
        require(len(group) == 8)
        opponent_profit[opponent] = _summary(group)['mean_delta']['own_net_profit'] >= 0
    conditions = [overall['own_net_profit'] > 0, overall['margin'] >= 0, overall['score'] >= 0,
                  *guards.values(), *request_difference.values(), *opponent_profit.values()]
    require(all(value is True or value is False or value == 'unknown' for value in conditions))
    status = ('failed' if any(value is False for value in conditions) else
              'unknown' if any(value == 'unknown' for value in conditions) else 'passed')
    return dict(status=status, passed=(status == 'passed'), per_seed_profit_guard=guards,
                per_seed_request_difference=request_difference, per_opponent_profit_guard=opponent_profit,
                interpretation='Exploratory screening only; no significance claim or promotion')


def project(cohorts, *, seed_roles, prefixes):
    """Return aggregate and paired economics, with aliases in place of seeds.

    Each cohort supplies 16 previously accepted metrics and eight authenticated
    comparison values. Prefix values are keyed by (cohort, internal alias,
    opponent, seat) and must have been computed from the matching replays.
    ``control - baseline`` is the only comparison. The caller owns all source,
    receipt and replay verification.
    """
    require(type(cohorts) is dict and set(cohorts) == set(COHORTS)
            and type(seed_roles) is dict and set(seed_roles) == set(COHORTS)
            and type(prefixes) is dict and len(prefixes) == 16)
    seeds = [seed_roles[c][a] for c in COHORTS for a in COHORTS[c]]
    require(all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
            and len(set(seeds)) == 4
            and all(type(seed_roles[c]) is dict and set(seed_roles[c]) == set(COHORTS[c])
                    for c in COHORTS))
    pairs, identities = [], {'baseline': set(), 'control': set(), 'crop': set()}
    for cohort, alias_map in COHORTS.items():
        source = cohorts[cohort]
        require(type(source) is dict and type(source.get('rows')) is list and len(source['rows']) == 16
                and type(source.get('comparisons')) is list and len(source['comparisons']) == 8)
        indexed = {}
        for row in source['rows']:
            require(type(row) is dict and row.get('schema') == 'task-pair-dev-game-metrics-v1')
            alias, opponent, side, arm = (row.get('seed_alias'), row.get('opponent'), row.get('side'), row.get('arm'))
            require(alias in alias_map and opponent in OPPONENTS and type(side) is int and side in SIDES
                    and arm in ARMS and type(row.get('seed')) is int and row['seed'] == seed_roles[cohort][alias]
                    and row.get('game') == f'dev-{alias}-{opponent}-{side}-{arm}')
            key = (alias, opponent, side, arm)
            require(key not in indexed)
            identities[arm].add(_digest(row.get('task_checkpoint_sha256')))
            identities['crop'].add(_digest(row.get('crop_checkpoint_sha256')))
            _digest(row.get('trace_sha256')); _digest(row.get('replay_sha256'))
            result = _game(row)
            require(all(key in result for key in FIELDS))
            indexed[key] = {key: _finite(result[key]) for key in FIELDS}
        expected_rows = {(a, o, s, arm) for a in alias_map for o in OPPONENTS for s in SIDES for arm in ARMS}
        require(set(indexed) == expected_rows)
        comparisons = {}
        for value in source['comparisons']:
            require(type(value) is dict)
            alias, opponent, side = value.get('seed_alias'), value.get('opponent'), value.get('side')
            require(alias in alias_map and opponent in OPPONENTS and type(side) is int and side in SIDES)
            key = (alias, opponent, side)
            require(key not in comparisons)
            comparisons[key] = _comparison(value, seed=seed_roles[cohort][alias], alias=alias,
                                           opponent=opponent, side=side)
        for alias in alias_map:
            for opponent in OPPONENTS:
                for side in SIDES:
                    key = (alias, opponent, side)
                    require(key in comparisons and (cohort, *key) in prefixes)
                    left, right = indexed[(*key, 'baseline')], indexed[(*key, 'control')]
                    pairs.append(dict(seed_alias=alias_map[alias], opponent=opponent, side=side,
                                      baseline=left, control=right,
                                      delta={field: right[field] - left[field] for field in FIELDS},
                                      common_task_history=comparisons[key],
                                      complete_request_prefix=_prefix(prefixes[(cohort, *key)])))
    require(len(pairs) == 16 and all(len(identities[k]) == 1 for k in identities)
            and identities['baseline'] != identities['control'])
    return dict(schema='task-pair-fresh-multi-public-v1', scope='fresh_seed_exploration',
                games=32, official_replays=32, paired_coordinates=16, seed_clusters=4,
                reference='baseline', candidate='control', pairs=pairs,
                overall=_summary(pairs),
                by_seed={role: _summary([p for p in pairs if p['seed_alias'] == role])
                         for role in ('F01', 'F02', 'F03', 'F04')},
                by_opponent={opponent: _summary([p for p in pairs if p['opponent'] == opponent])
                             for opponent in OPPONENTS},
                by_side={str(side): _summary([p for p in pairs if p['side'] == side]) for side in SIDES},
                exploratory_screen=_exploratory_screen(pairs),
                training=False, statistical_significance='not_assessed',
                independent_improvement=False, promotion=False)
