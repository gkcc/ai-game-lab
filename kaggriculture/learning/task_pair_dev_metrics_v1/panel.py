"""Private descriptive paired accounting; no significance or promotion gate."""
import math
from statistics import mean

from .consumer import OFFICIAL_TURNS

ECONOMIC = ('net_profit', 'successful_plants', 'seed_units_bought', 'sales_units',
            'sales_revenue', 'ending_seed_units', 'ending_warehouse_units')
METRICS = tuple('own_' + key for key in ECONOMIC) + tuple('opponent_' + key for key in ECONOMIC) + ('margin', 'score')
OPPORTUNITY_KEY = 'model_callback_multi_unit_opportunities'
MATCHED_OPPORTUNITY_KEY = 'matched_multi_unit_opportunities'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Invalid economic number')
    return value


def _game(row):
    result = row.get('result')
    _require(type(result) is dict and result.get('official_turns') == OFFICIAL_TURNS
             and type(result.get('actor')) is int and result['actor'] == row['side'],
             'Invalid completed-game result')
    e = result.get('economy')
    _require(type(e) is dict and type(e.get('own')) is dict
             and type(e.get('opponent')) is dict, 'Missing economic results')
    metrics = {}
    for perspective in ('own', 'opponent'):
        data = e[perspective]
        _require(data.get('cash_reconciled') is True, 'Unreconciled cash')
        initial, final = _number(data.get('initial_cash')), _number(data.get('final_cash'))
        for key in ECONOMIC:
            metrics[perspective + '_' + key] = _number(data.get(key))
        _require(metrics[perspective + '_net_profit'] == final - initial,
                 'Net profit differs from cash')
        _require(all(metrics[perspective + '_' + key] >= 0 for key in ECONOMIC[1:]),
                 'Negative economic quantity')
    margin = _number(result.get('margin'))
    _require(margin == e['own']['final_cash'] - e['opponent']['final_cash'],
             'Margin differs from cash')
    score = _number(result.get('score'))
    _require(score == (1. if margin > 0 else .5 if margin == 0 else 0.),
             'Score differs from margin')
    metrics.update(margin=margin, score=score)
    return metrics


def _summary(pairs, expected):
    if len(pairs) != expected or not pairs:
        return dict(support='unknown', values=None, available_pairs=len(pairs), expected_pairs=expected)
    return dict(support='observed', values={m: mean(p['delta'][m] for p in pairs) for m in METRICS},
                available_pairs=len(pairs), expected_pairs=expected)


def _support(counts, seeds):
    """Fixed descriptive availability gate; never a significance test."""
    known = all(seed in counts and counts[seed] is not None for seed in seeds)
    total = sum(counts[seed] for seed in seeds) if known else None
    supported = known and total >= 20 and all(counts[seed] >= 5 for seed in seeds)
    return dict(support='supported' if supported else 'unknown',
                per_seed={seed: counts.get(seed) for seed in seeds}, total=total,
                min_per_seed=5, min_total=20)


def paired(rows, *, seeds, opponents, reference, candidate, comparisons=None):
    """Compare frozen arms at every predeclared seed/opponent/seat coordinate.

    The caller is responsible for authenticating each source, replay, result
    and model identity. Both seats are always required. A missing game does not
    become a zero or disappear from a group mean. Result is private, including
    identities and per-game metrics. Two seeds do not support significance.
    """
    _require(type(rows) is list and type(seeds) in (tuple, list) and len(seeds) > 0
             and all(type(s) is int and s >= 0 for s in seeds)
             and len(set(seeds)) == len(seeds), 'Invalid expected seeds')
    _require(type(opponents) in (tuple, list) and len(opponents) > 0
             and all(type(o) is str and o for o in opponents)
             and len(set(opponents)) == len(opponents), 'Invalid expected opponents')
    _require(type(reference) is str and reference and type(candidate) is str
             and candidate and reference != candidate, 'Invalid arms')
    coordinates = [(seed, opponent, side) for seed in seeds for opponent in opponents for side in (0, 1)]
    expected = {(arm, seed, opponent, side) for arm in (reference, candidate)
                for seed, opponent, side in coordinates}
    indexed, opportunities = {}, {}
    for row in rows:
        _require(type(row) is dict and type(row.get('seed')) is int
                 and type(row.get('side')) is int, 'Invalid game row')
        key = (row.get('arm'), row['seed'], row.get('opponent'), row['side'])
        _require(key in expected and key not in indexed, 'Unexpected or duplicate game')
        indexed[key] = _game(row)
        counts = row['result'].get('selection_counts')
        if counts is None:
            opportunities[key] = None
        else:
            _require(type(counts) is dict and type(counts.get('model_callbacks')) is int
                     and type(counts.get(OPPORTUNITY_KEY)) is int
                     and 0 <= counts[OPPORTUNITY_KEY] <= counts['model_callbacks'],
                     'Invalid opportunity counts')
            opportunities[key] = counts[OPPORTUNITY_KEY]

    pairs = []
    for seed, opponent, side in coordinates:
        baseline = indexed.get((reference, seed, opponent, side))
        new = indexed.get((candidate, seed, opponent, side))
        if baseline is None or new is None:
            pairs.append(dict(seed=seed, opponent=opponent, side=side, support='unknown',
                              reference=baseline, candidate=new, delta=None))
        else:
            pairs.append(dict(seed=seed, opponent=opponent, side=side, support='observed',
                              reference=baseline, candidate=new,
                              delta={m: new[m] - baseline[m] for m in METRICS}))

    good = [p for p in pairs if p['support'] == 'observed']
    per_seed = {seed: _summary([p for p in good if p['seed'] == seed], 2 * len(opponents)) for seed in seeds}
    per_opponent = {opponent: _summary([p for p in good if p['opponent'] == opponent], 2 * len(seeds))
                    for opponent in opponents}
    per_side = {side: _summary([p for p in good if p['side'] == side], len(seeds) * len(opponents))
                for side in (0, 1)}
    arm_counts = {}
    for arm in (reference, candidate):
        arm_counts[arm] = {}
        for seed in seeds:
            cells = [opportunities.get((arm, seed, opponent, side))
                     for opponent in opponents for side in (0, 1)]
            arm_counts[arm][seed] = sum(cells) if all(type(v) is int for v in cells) else None

    _require(comparisons is None or type(comparisons) is list, 'Invalid matched comparisons')
    matches = {}
    for item in comparisons or []:
        _require(type(item) is dict and type(item.get('seed')) is int and type(item.get('side')) is int,
                 'Invalid comparison coordinate')
        _require(item.get('reference') == reference and item.get('candidate') == candidate,
                 'Comparison arms differ from paired games')
        key = (item['seed'], item.get('opponent'), item['side'])
        _require(key in coordinates and key not in matches, 'Unexpected or duplicate comparison')
        comparison = item.get('result')
        _require(type(comparison) is dict and type(comparison.get('comparison_complete')) is bool,
                 'Invalid comparison summary')
        count = comparison.get(MATCHED_OPPORTUNITY_KEY)
        game_pair_complete = ((reference, *key) in indexed and (candidate, *key) in indexed)
        if comparison['comparison_complete'] and game_pair_complete:
            _require(type(count) is int and count >= 0
                     and type(comparison.get('matched_calls')) is int
                     and count <= comparison['matched_calls'], 'Invalid matched opportunity count')
            matches[key] = count
        else:
            matches[key] = None
    matched_counts = {}
    for seed in seeds:
        cells = [matches.get((seed, opponent, side)) for opponent in opponents for side in (0, 1)]
        matched_counts[seed] = sum(cells) if all(type(v) is int for v in cells) else None
    return dict(pairs=pairs, overall=_summary(good, len(coordinates)),
                by_seed=per_seed, by_opponent=per_opponent, by_side=per_side,
                opportunity_support=dict(reference=_support(arm_counts[reference], seeds),
                                         candidate=_support(arm_counts[candidate], seeds),
                                         same_input_pairs=_support(matched_counts, seeds)),
                expected_games=len(expected), observed_games=len(indexed),
                seed_clusters=len(seeds), statistical_significance='unknown',
                independently_verified_improvement=False, promotion=False,
                interpretation='Descriptive paired development results only; not independent significance.')
