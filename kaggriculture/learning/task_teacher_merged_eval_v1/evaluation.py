"""Paired descriptive statistics for the fixed four-seed official screen.

Only aliases leave this module.  Four complete seed clusters are exploratory:
they cannot establish statistical significance, generalization or promotion.
"""
import math
import random

OPPONENTS = ('round2', 'market', 'teacher')
POLICIES = ('merged', 'unweighted', 'teacher_task')
REFERENCES = ('unweighted', 'teacher_task')
GATES = dict(pair_own_min=-2000, pair_margin_min=-2000,
             seed_opponent_own_min=-1000, seed_opponent_margin_min=-1000,
             overall_own_min=0, overall_margin_min=0, overall_score_min=0,
             bootstrap_seed=2026102837, bootstrap_replicates=10000,
             bootstrap_cluster='whole_seed', bootstrap_shared_indices=True,
             positive_own_strict_min=0, positive_margin_strict_min=0,
             positive_requires_gate=True)
METRICS = ('own_profit_difference', 'opponent_profit_difference', 'margin_difference', 'score_difference')


def plans(seeds):
    if (not isinstance(seeds, (list, tuple)) or len(seeds) != 4
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != 4):
        raise ValueError('Four unique uint32 seeds required')
    return [dict(game=f'holdout-S{index:02d}-{opponent}-{side}-{policy}',
                 seed_alias=f'S{index:02d}', seed=seed, opponent=opponent,
                 side=side, policy=policy, split='holdout', parity_kind=None)
            for index, seed in enumerate(seeds, 1) for opponent in OPPONENTS
            for side in (0, 1) for policy in POLICIES]


def number(value):
    if type(value) not in (int, float): raise ValueError('Invalid economic value')
    try: result = float(value)
    except (OverflowError, ValueError) as exc: raise ValueError('Invalid economic value') from exc
    if not math.isfinite(result): raise ValueError('Invalid economic value')
    return result


def difference(a, b):
    return number(number(a) - number(b))


def mean(values):
    values = list(values)
    if not values: raise ValueError('Empty mean')
    try: return number(math.fsum(number(value) / len(values) for value in values))
    except OverflowError as exc: raise ValueError('Nonfinite mean') from exc


def close(a, b):
    return abs(a-b) <= 1e-7


def quantile(sorted_values, q):
    pos = (len(sorted_values)-1)*q
    lo = int(pos); fraction = pos-lo
    return number(sorted_values[lo]*(1-fraction) + sorted_values[min(lo+1,len(sorted_values)-1)]*fraction)


def evaluate(rows, seeds):
    expected = plans(seeds)
    if not isinstance(rows, list) or len(rows) != len(expected): raise ValueError('Incomplete panel')
    keyed = {}
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Invalid row')
        if (type(row.get('seed')) is not int or not 0 <= row['seed'] < 2**32
            or type(row.get('side')) is not int or row['side'] not in (0, 1)):
            raise ValueError('Invalid seed or seat')
        key = (row.get('seed_alias'), row.get('opponent'), row['side'], row.get('policy'))
        if key in keyed: raise ValueError('Duplicate game')
        keyed[key] = row
    if set(keyed) != {(p['seed_alias'],p['opponent'],p['side'],p['policy']) for p in expected}:
        raise ValueError('Panel mismatch')
    for plan in expected:
        row = keyed[(plan['seed_alias'],plan['opponent'],plan['side'],plan['policy'])]
        if any(row.get(key) != value for key,value in plan.items()): raise ValueError('Game identity mismatch')
        if (row.get('passed') is not True or type(row.get('states')) is not int or row['states'] != 720
            or type(row.get('action_turns')) is not int or row['action_turns'] != 719
            or row.get('status') != ['DONE','DONE'] or type(row.get('agent_errors')) is not int or row['agent_errors'] != 0):
            raise ValueError('Game not accepted')
        try:
            money = {name:number(row[name]) for name in ('bank','initial_bank','opponent_bank',
                'opponent_initial_bank','net_profit','opponent_net_profit','margin','score')}
        except KeyError as exc: raise ValueError('Missing economic value') from exc
        if (not close(difference(money['bank'],money['initial_bank']),money['net_profit'])
            or not close(difference(money['opponent_bank'],money['opponent_initial_bank']),money['opponent_net_profit'])
            or not close(difference(money['bank'],money['opponent_bank']),money['margin'])):
            raise ValueError('Economics inconsistent')
        if money['score'] != (1.0 if money['margin'] > 0 else 0.0 if money['margin'] < 0 else 0.5):
            raise ValueError('Score inconsistent')

    pairs = {reference: [] for reference in REFERENCES}
    for alias in (f'S{i:02d}' for i in range(1, 5)):
        for opponent in OPPONENTS:
            for side in (0, 1):
                candidate = keyed[(alias,opponent,side,'merged')]
                for reference in REFERENCES:
                    baseline = keyed[(alias,opponent,side,reference)]
                    pairs[reference].append(dict(seed_alias=alias, opponent=opponent, side=side,
                        own_profit_difference=difference(candidate['net_profit'],baseline['net_profit']),
                        opponent_profit_difference=difference(candidate['opponent_net_profit'],baseline['opponent_net_profit']),
                        margin_difference=difference(candidate['margin'],baseline['margin']),
                        score_difference=difference(candidate['score'],baseline['score'])))

    clusters = {reference: [] for reference in REFERENCES}
    per_opponent = {reference: {} for reference in REFERENCES}
    for reference in REFERENCES:
        for alias in (f'S{i:02d}' for i in range(1, 5)):
            selected = [pair for pair in pairs[reference] if pair['seed_alias'] == alias]
            clusters[reference].append(dict(seed_alias=alias, **{metric:mean(p[metric] for p in selected) for metric in METRICS}))
        for opponent in OPPONENTS:
            selected = [pair for pair in pairs[reference] if pair['opponent'] == opponent]
            per_opponent[reference][opponent] = {metric:mean(p[metric] for p in selected) for metric in METRICS}
    overall = {reference:{metric:mean(c[metric] for c in clusters[reference]) for metric in METRICS}
               for reference in REFERENCES}

    failures = []
    for pair in pairs['unweighted']:
        for metric,limit in (('own_profit_difference',GATES['pair_own_min']),('margin_difference',GATES['pair_margin_min'])):
            if pair[metric] < limit:
                failures.append(dict(scope='pair',seed_alias=pair['seed_alias'],opponent=pair['opponent'],side=pair['side'],metric=metric))
    for alias in (f'S{i:02d}' for i in range(1, 5)):
        for opponent in OPPONENTS:
            selected = [p for p in pairs['unweighted'] if p['seed_alias']==alias and p['opponent']==opponent]
            for metric,limit in (('own_profit_difference',GATES['seed_opponent_own_min']),('margin_difference',GATES['seed_opponent_margin_min'])):
                if mean(p[metric] for p in selected) < limit:
                    failures.append(dict(scope='seed_opponent',seed_alias=alias,opponent=opponent,side=None,metric=metric))
    for metric,limit in (('own_profit_difference',GATES['overall_own_min']),
                         ('margin_difference',GATES['overall_margin_min']),('score_difference',GATES['overall_score_min'])):
        if overall['unweighted'][metric] < limit:
            failures.append(dict(scope='overall',seed_alias=None,opponent=None,side=None,metric=metric))

    summaries = []
    for opponent in OPPONENTS:
        for policy in POLICIES:
            sample = [row for row in rows if row['opponent']==opponent and row['policy']==policy]
            scores = [number(row['score']) for row in sample]
            summaries.append(dict(opponent=opponent,policy=policy,wins=scores.count(1.0),draws=scores.count(.5),
                losses=scores.count(0.0),**{f'mean_{name}':mean(row[name] for row in sample)
                    for name in ('net_profit','opponent_net_profit','margin','score')}))

    rng = random.Random(GATES['bootstrap_seed'])
    samples = {reference:{metric:[] for metric in METRICS} for reference in REFERENCES}
    for _ in range(GATES['bootstrap_replicates']):
        indices = [rng.randrange(4) for _ in range(4)]
        for reference in REFERENCES:
            for metric in METRICS:
                samples[reference][metric].append(mean(clusters[reference][i][metric] for i in indices))
    intervals = {reference:{metric:[quantile(sorted(values),.025),quantile(sorted(values),.975)]
                  for metric,values in metrics.items()} for reference,metrics in samples.items()}
    gate_passed = not failures
    positive = (gate_passed and overall['unweighted']['own_profit_difference'] > GATES['positive_own_strict_min']
                and overall['unweighted']['margin_difference'] > GATES['positive_margin_strict_min'])
    return dict(summaries=summaries, pairs=pairs, seed_clusters=clusters, per_opponent=per_opponent,
        overall=overall, bootstrap_intervals=intervals, gates=dict(GATES), gate_failures=failures,
        exploratory_gate_passed=gate_passed, exploratory_positive_signal=positive,
        seed_cluster_count=4, automatic_promotion=False,
        interpretation='Four whole-seed clusters; descriptive only, no significance, generalization or promotion claim')
