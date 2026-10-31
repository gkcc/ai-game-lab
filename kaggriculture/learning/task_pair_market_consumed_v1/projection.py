"""Allowlisted projection of accepted market-development comparisons.

The historical arithmetic verifier is reused; its historical display labels
are replaced with this experiment's distinct role names before publication.
"""
from statistics import mean

from learning.task_pair_historical_compare_v1.projection import project as historical_project


ALIASES = {'S01': 'D01', 'S04': 'D02'}
INTERNAL = {'S02': 'D01', 'S03': 'D02'}
REASONS = {'observed': {'actor_request_difference'},
           'no_difference': {'all_requests_equal'},
           'unknown': {'frame_difference', 'opponent_request_difference',
                       'simultaneous_request_difference', 'terminal_frame_difference'}}


def require(value):
    if not value:
        raise ValueError('Market development projection differs')


def project(paired, *, seeds, rows, comparisons, prefixes):
    require(type(prefixes) is dict and set(prefixes) == {(alias, side) for alias in INTERNAL for side in (0, 1)})
    result = historical_project(paired, seeds=seeds, rows=rows, comparisons=comparisons)
    require(result['games'] == 8 and result['seed_clusters'] == 2
            and set(result['by_seed']) == set(ALIASES) and len(result['pairs']) == 4)
    result['schema'] = 'task-pair-market-consumed-public-v1'
    result['by_seed'] = {ALIASES[key]: value for key, value in result['by_seed'].items()}
    seen = set()
    for item in result['pairs']:
        alias = item['seed_alias']
        require(alias in ALIASES and type(item['side']) is int and item['side'] in (0, 1))
        public = ALIASES[alias]
        internal = next(k for k, v in INTERNAL.items() if v == public)
        key = (internal, item['side'])
        require(key not in seen)
        seen.add(key)
        prefix = prefixes[key]
        require(type(prefix) is dict and set(prefix) == {'scope', 'status', 'reason', 'step'}
                and prefix['scope'] == 'same_complete_history' and prefix['status'] in REASONS
                and prefix['reason'] in REASONS[prefix['status']]
                and ((prefix['status'] == 'observed' and type(prefix['step']) is int
                      and 0 <= prefix['step'] < 719)
                     or (prefix['status'] != 'observed' and prefix['step'] is None)))
        item['seed_alias'] = public
        item['complete_request_prefix'] = dict(prefix)
    require(len(seen) == 4)
    overall = result['overall']['mean_delta']
    per_seed, action_evidence = {}, {}
    for role in ('D01', 'D02'):
        items = [item for item in result['pairs'] if item['seed_alias'] == role]
        require(len(items) == 2)
        baseline = mean(item['reference']['own_net_profit'] for item in items)
        delta = mean(item['delta']['own_net_profit'] for item in items)
        per_seed[role] = ('unknown' if baseline <= 0 else delta >= -0.01 * baseline)
        action_evidence[role] = any(item['complete_request_prefix']['status'] == 'observed' for item in items)
    passed = (overall['own_net_profit'] > 0 and overall['margin'] >= 0 and overall['score'] >= 0
              and all(value is True for value in per_seed.values()) and all(action_evidence.values()))
    result['development_screen'] = dict(passed=passed, per_seed_profit_guard=per_seed,
                                        per_seed_request_difference=action_evidence,
                                        interpretation='exploratory only; no promotion or independent inference')
    return result
