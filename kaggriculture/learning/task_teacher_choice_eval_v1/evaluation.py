"""Pure fixed-panel comparisons; inputs must already be independently audited.

No source files, models, official engine or games are opened by this module.
"""
import math

POLICIES = ('candidate','baseline')


def plans(seeds):
    if (not isinstance(seeds,(list,tuple)) or len(seeds)!=2 or len(set(seeds))!=2
        or any(type(x) is not int or not 0 <= x <= 0xffffffff for x in seeds)):
        raise ValueError('Two unique reserved seeds required')
    return [dict(game=f'holdout-{seed}-round2-{side}-{policy}',seed=seed,opponent='round2',
                 side=side,policy=policy,split='holdout',parity_kind=None)
            for seed in seeds for side in (0,1) for policy in POLICIES]


def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value): raise ValueError('Invalid economic value')
    return float(value)


def evaluate(rows,seeds):
    expected = plans(seeds)
    if not isinstance(rows,list) or len(rows)!=8: raise ValueError('Incomplete panel')
    keyed = {}
    for row in rows:
        if not isinstance(row,dict): raise ValueError('Invalid row')
        key = (row['seed'],row['side'],row['policy'])
        if key in keyed: raise ValueError('Duplicate game')
        keyed[key] = row
    expected_keys={(p['seed'],p['side'],p['policy']) for p in expected}
    if set(keyed)!=expected_keys or len({r['game'] for r in rows})!=8: raise ValueError('Panel mismatch')
    for p in expected:
        r=keyed[(p['seed'],p['side'],p['policy'])]
        if any(r.get(k)!=p[k] for k in ('game','seed','side','opponent','policy')):
            raise ValueError('Game identity mismatch')
        if (r.get('passed') is not True or r.get('states')!=720 or r.get('action_turns')!=719
            or r.get('status')!=['DONE','DONE'] or r.get('agent_errors')!=0):
            raise ValueError('Game not accepted')
        for key in ('net_profit','opponent_net_profit','bank','opponent_bank','margin','score'): _number(r[key])
        if abs((_number(r['bank'])-_number(r['opponent_bank']))-_number(r['margin']))>1e-7:
            raise ValueError('Cash margin mismatch')
        expected_score = 1.0 if r['margin'] > 0 else 0.0 if r['margin'] < 0 else 0.5
        if _number(r['score']) != expected_score: raise ValueError('Outcome score mismatch')
    pairs=[]
    for seed in seeds:
        for side in (0,1):
            candidate=keyed[(seed,side,'candidate')]; baseline=keyed[(seed,side,'baseline')]
            own=_number(candidate['net_profit'])-_number(baseline['net_profit'])
            opponent=_number(candidate['opponent_net_profit'])-_number(baseline['opponent_net_profit'])
            margin=_number(candidate['margin'])-_number(baseline['margin'])
            def score(row):
                a,b=_number(row['bank']),_number(row['opponent_bank'])
                return 'win' if a>b else 'loss' if a<b else 'draw'
            pairs.append(dict(seed=seed,side=side,candidate_own_profit=_number(candidate['net_profit']),
                baseline_own_profit=_number(baseline['net_profit']),candidate_opponent_profit=_number(candidate['opponent_net_profit']),
                baseline_opponent_profit=_number(baseline['opponent_net_profit']),candidate_margin=_number(candidate['margin']),
                baseline_margin=_number(baseline['margin']),own_profit_difference=own,opponent_profit_difference=opponent,
                margin_difference=margin,candidate_outcome=score(candidate),baseline_outcome=score(baseline),
                candidate_score=_number(candidate['score']),baseline_score=_number(baseline['score'])))
    clusters=[dict(seed=seed,mean_own_profit_difference=sum(p['own_profit_difference'] for p in pairs if p['seed']==seed)/2,
                   mean_margin_difference=sum(p['margin_difference'] for p in pairs if p['seed']==seed)/2) for seed in seeds]
    failures=[]
    for p in pairs:
        if p['own_profit_difference'] < -2000: failures.append(dict(seed=p['seed'],side=p['side'],metric='own_profit'))
        if p['margin_difference'] < -2000: failures.append(dict(seed=p['seed'],side=p['side'],metric='cash_margin'))
    for cluster in clusters:
        if cluster['mean_own_profit_difference'] < -1000: failures.append(dict(seed=cluster['seed'],metric='mean_own_profit'))
        if cluster['mean_margin_difference'] < -1000: failures.append(dict(seed=cluster['seed'],metric='mean_cash_margin'))
    return dict(pairs=pairs,seed_groups=clusters,exploratory_gate_passed=not failures,gate_failures=failures,
        interpretation='Two seed clusters and one opponent; no statistical significance, competitive promotion or generalization established')
