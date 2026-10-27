"""Pure descriptive analysis of an already audited three-seed panel."""
import math
import random

OPPONENTS = ('round2', 'market')
POLICIES = ('weighted', 'unweighted')
BOOTSTRAP_SEED = 2026102733
BOOTSTRAP_REPLICATES = 10000


def plans(seeds):
    if (not isinstance(seeds, (list, tuple)) or len(seeds) != 3
        or any(type(x) is not int or not 0 <= x <= 0xffffffff for x in seeds)
        or len(set(seeds)) != 3): raise ValueError('Three unique uint32 seeds required')
    return [dict(game=f'holdout-{seed}-{opponent}-{side}-{policy}', seed=seed, opponent=opponent,
                 side=side, policy=policy, split='holdout', parity_kind=None)
            for seed in seeds for opponent in OPPONENTS for side in (0, 1) for policy in POLICIES]


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value): raise ValueError('Invalid economic value')
    return float(value)


def close(a, b):
    return abs(a-b) <= 1e-7


def difference(a, b):
    return number(number(a)-number(b))


def mean(values):
    values=list(values)
    if not values: raise ValueError('Empty mean')
    try: return number(math.fsum(number(v)/len(values) for v in values))
    except OverflowError as exc: raise ValueError('Nonfinite mean') from exc


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
        if (type(row.get('seed')) is not int or not 0 <= row['seed'] <= 0xffffffff
            or type(row.get('side')) is not int or row['side'] not in (0, 1)):
            raise ValueError('Invalid seed or side')
        key = (row['seed'], row['opponent'], row['side'], row['policy'])
        if key in keyed: raise ValueError('Duplicate game')
        keyed[key] = row
    if set(keyed) != {(p['seed'],p['opponent'],p['side'],p['policy']) for p in expected}:
        raise ValueError('Panel mismatch')
    for p in expected:
        r = keyed[(p['seed'],p['opponent'],p['side'],p['policy'])]
        if any(r.get(k) != p[k] for k in p): raise ValueError('Game identity mismatch')
        if (r.get('passed') is not True or type(r.get('states')) is not int or r['states'] != 720
            or type(r.get('action_turns')) is not int or r['action_turns'] != 719
            or r.get('status') != ['DONE','DONE'] or type(r.get('agent_errors')) is not int or r['agent_errors'] != 0):
            raise ValueError('Game not accepted')
        v = {k:number(r[k]) for k in ('bank','initial_bank','opponent_bank','opponent_initial_bank',
                                      'net_profit','opponent_net_profit','margin','score')}
        if (not close(difference(v['bank'],v['initial_bank']),v['net_profit'])
            or not close(difference(v['opponent_bank'],v['opponent_initial_bank']),v['opponent_net_profit'])
            or not close(difference(v['bank'],v['opponent_bank']),v['margin'])): raise ValueError('Economics inconsistent')
        if v['score'] != (1.0 if v['margin'] > 0 else 0.0 if v['margin'] < 0 else 0.5):
            raise ValueError('Score inconsistent')
    pairs = []
    for seed in seeds:
        for opponent in OPPONENTS:
            for side in (0,1):
                a = keyed[(seed,opponent,side,'weighted')]; b = keyed[(seed,opponent,side,'unweighted')]
                pairs.append(dict(seed=seed,opponent=opponent,side=side,
                    own_profit_difference=difference(a['net_profit'],b['net_profit']),
                    opponent_profit_difference=difference(a['opponent_net_profit'],b['opponent_net_profit']),
                    margin_difference=difference(a['margin'],b['margin']),
                    score_difference=difference(a['score'],b['score'])))
    clusters=[]; gate_failures=[]
    for p in pairs:
        for metric in ('own_profit_difference','margin_difference'):
            if p[metric] < -2000: gate_failures.append(dict(seed=p['seed'],opponent=p['opponent'],side=p['side'],metric=metric))
    for seed in seeds:
        selected=[p for p in pairs if p['seed']==seed]
        clusters.append(dict(seed=seed, **{k:mean(p[k] for p in selected) for k in
            ('own_profit_difference','opponent_profit_difference','margin_difference','score_difference')}))
        for opponent in OPPONENTS:
            sides=[p for p in selected if p['opponent']==opponent]
            for metric in ('own_profit_difference','margin_difference'):
                if mean(p[metric] for p in sides) < -1000:
                    gate_failures.append(dict(seed=seed,opponent=opponent,side=None,metric=metric))
    summaries=[]
    for opponent in OPPONENTS:
        for policy in POLICIES:
            sample=[r for r in rows if r['opponent']==opponent and r['policy']==policy]
            scores=[number(r['score']) for r in sample]
            summaries.append(dict(opponent=opponent,policy=policy,wins=scores.count(1),draws=scores.count(.5),
                losses=scores.count(0), **{f'mean_{k}':mean(r[k] for r in sample)
                for k in ('net_profit','opponent_net_profit','margin','score')}))
    rng=random.Random(BOOTSTRAP_SEED)
    boot={k:[] for k in ('own_profit_difference','opponent_profit_difference','margin_difference','score_difference')}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected=[clusters[rng.randrange(3)] for _ in range(3)]
        for k in boot: boot[k].append(mean(c[k] for c in selected))
    intervals={k:[quantile(sorted(v),.025),quantile(sorted(v),.975)] for k,v in boot.items()}
    return dict(summaries=summaries,pairs=pairs,seed_clusters=clusters,bootstrap_intervals=intervals,
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=BOOTSTRAP_REPLICATES,
        exploratory_gate_passed=not gate_failures,gate_failures=gate_failures,
        interpretation='Three seed clusters; descriptive only, no significance, promotion, generalization or causation claim')
