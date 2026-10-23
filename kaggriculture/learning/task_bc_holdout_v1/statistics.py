"""Frozen two-arm, eight-seed capability-retention panel statistics.

Exactly 8 seeds x 3 opponents x 2 seats x 2 arms = 96 completed games,
48 paired comparisons. The cluster is one COMPLETE seed (six pairs).
PCG64 seed 2026102341 draws 20,000 arrays of eight seed indices. Each
replicate recomputes mean(trained net - reference net) / mean(reference
net), NOT a mean of per-game ratios or a fixed-denominator interval.
Two-sided 95% intervals use NumPy's linear quantile. Primary admission:
relative interval lower > -0.02, each opponent's relative point estimate
>= -0.05, and each opponent's score difference >= -0.05. These opponent
point guards are not statistical non-inferiority tests. Eight clusters
limit precision; an interval strictly above zero is panel evidence only.

The reference is teacher_task: rule tasks with the SAME frozen crop head,
not the original all-rule teacher. Every relative reference denominator,
including all bootstrap resamples, must be strictly positive; otherwise
the relative endpoint is undefined and admission fails closed.
"""
from collections import Counter
import math
from numbers import Real

import numpy as np


ARMS = ('trained', 'teacher_task')
METRICS = ('net_profit', 'opponent_net_profit', 'margin', 'score')
BOOTSTRAP = dict(replicates=20000, seed=2026102341, quantile_method='linear')
GATES = dict(relative_ci_lower_gt=-.02, opponent_relative_mean_min=-.05,
             opponent_score_delta_min=-.05)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
             and math.isfinite(float(value)), 'Invalid finite numeric ' + name)
    return float(value)


def _validated(ledger, cfg):
    seeds, opponents = cfg['test_seeds'], cfg['opponents']
    _require(isinstance(seeds, list) and len(seeds) == 8
             and all(type(s) is int for s in seeds) and len(set(seeds)) == 8,
             'Exactly eight unique integer seeds required')
    _require(isinstance(opponents, list) and len(opponents) == 3
             and all(isinstance(o, str) and o for o in opponents)
             and set(opponents) == {'v1', 'market', 'round2'},
             'Frozen v1, market and round2 opponents required')
    _require(cfg['bootstrap'] == BOOTSTRAP, 'Bootstrap differs from frozen protocol')
    _require(cfg['gates'] == GATES, 'Gates differ from frozen protocol')
    if 'arms' in cfg:
        _require(cfg['arms'] == list(ARMS), 'Arms differ from frozen protocol')
    _require(isinstance(ledger, list) and len(ledger) == 96,
             'Incomplete or excess panel: exactly 96 rows required')
    expected = {(s, o, side, arm) for s in seeds for o in opponents
                for side in (0, 1) for arm in ARMS}
    indexed = {}
    for row in ledger:
        _require(isinstance(row, dict), 'Each ledger row must be a mapping')
        _require(type(row['seed']) is int and type(row['side']) is int,
                 'Seed and side must be integers, not booleans')
        key = row['seed'], row['opponent'], row['side'], row['policy']
        _require(key in expected and key not in indexed, 'Unexpected or duplicate key: ' + repr(key))
        _require(row['passed'] is True and row['status'] == ['DONE', 'DONE'],
                 'Unaccepted or incomplete game: ' + repr(key))
        for field, required in (('states', 720), ('actions', 719), ('action_turns', 719)):
            if field in row:
                _require(row[field] == required, 'Wrong completed game ' + field)
        values = {name: _number(row[name], name) for name in
                  ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank', *METRICS)}
        _require(values['net_profit'] == values['bank'] - values['initial_bank'],
                 'Own net profit disagrees with bank change')
        _require(values['opponent_net_profit'] == values['opponent_bank'] - values['opponent_initial_bank'],
                 'Opponent net profit disagrees with bank change')
        _require(values['margin'] == values['bank'] - values['opponent_bank'],
                 'Margin disagrees with final bank difference')
        score = 1. if values['margin'] > 0 else 0. if values['margin'] < 0 else .5
        _require(values['score'] == score, 'Score disagrees with final bank outcome')
        for field in ('errors', 'agent_errors', 'timeouts', 'consumer_errors', 'fallbacks', 'load_errors', 'inference_errors'):
            if field in row:
                _require(row[field] == 0, 'Nonzero execution errors: ' + field)
        indexed[key] = row
    _require(set(indexed) == expected, 'Panel missing required keys')
    return seeds, opponents, indexed


def _arm_summary(rows):
    counts = Counter('win' if r['score'] == 1 else 'draw' if r['score'] == .5 else 'loss' for r in rows)
    return dict(games=len(rows), wins=counts['win'], draws=counts['draw'], losses=counts['loss'],
                means={k: float(np.mean([r[k] for r in rows])) for k in METRICS},
                opponent_wins=counts['loss'], opponent_draws=counts['draw'], opponent_losses=counts['win'],
                opponent_score=float(np.mean([1-r['score'] for r in rows])))


def _interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975], method='linear')]


def _paired_summary(pairs, draws=None):
    arms = {a: _arm_summary([p[a] for p in pairs]) for a in ARMS}
    deltas = {k: float(np.mean([p['delta'][k] for p in pairs])) for k in METRICS}
    denominator = arms['teacher_task']['means']['net_profit']
    relative = deltas['net_profit']/denominator if denominator > 0 else None
    result = dict(pairs=len(pairs), arms=arms, delta_means=deltas,
                  relative_net_profit_mean=relative, reference_net_profit_mean=denominator)
    if draws is None:
        return result
    seeds = list(dict.fromkeys(p['seed'] for p in pairs))
    # A cluster's seats and opponents remain together in every resample.
    groups = [[p for p in pairs if p['seed'] == s] for s in seeds]
    _require(len(groups) == 8 and len({len(g) for g in groups}) == 1,
             'Bootstrap requires eight equally complete seed clusters')
    seed_delta = {k: np.array([np.mean([p['delta'][k] for p in g]) for g in groups]) for k in METRICS}
    seed_reference = np.array([np.mean([p['teacher_task']['net_profit'] for p in g]) for g in groups])
    bootstrap_delta = {k: v[draws].mean(axis=1) for k, v in seed_delta.items()}
    bootstrap_reference = seed_reference[draws].mean(axis=1)
    positive = denominator > 0 and bool(np.all(bootstrap_reference > 0))
    result.update(delta_ci95={k: _interval(v) for k, v in bootstrap_delta.items()},
                  relative_net_profit_defined=positive,
                  nonpositive_bootstrap_reference_count=int(np.count_nonzero(bootstrap_reference <= 0)),
                  bootstrap_reference_min=float(bootstrap_reference.min()),
                  relative_net_profit_ci95=_interval(bootstrap_delta['net_profit']/bootstrap_reference) if positive else None)
    if not positive:
        result['relative_net_profit_mean'] = None
        result['relative_undefined_reason'] = 'Nonpositive point or resampled reference net profit'
    return result


def analyze(ledger, cfg):
    """Validate a completed frozen panel, then return JSON-safe paired evidence.

    Raises ValueError (or KeyError for an absent required field) for incomplete,
    duplicate, inconsistent or failed games. A stopped runner must retain its
    partial ledger and describe incompleteness rather than invoke this as if
    complete. ``passed`` is statistical capability admission, not a replay,
    privacy, source-identity or independent-review certificate.
    """
    seeds, opponents, indexed = _validated(ledger, cfg)
    pairs = []
    for seed in seeds:
        for opponent in opponents:
            for side in (0, 1):
                rows = {arm: indexed[seed, opponent, side, arm] for arm in ARMS}
                # Initial funds must match within each matched comparison.
                for field in ('initial_bank', 'opponent_initial_bank'):
                    _require(rows['trained'][field] == rows['teacher_task'][field],
                             'Paired initial funds differ')
                pairs.append(dict(seed=seed, opponent=opponent, side=side,
                    **{arm: {k: rows[arm][k] for k in
                       ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank', *METRICS)} for arm in ARMS},
                    delta={k: rows['trained'][k]-rows['teacher_task'][k] for k in METRICS},
                    relative_net_profit=(rows['trained']['net_profit']-rows['teacher_task']['net_profit'])
                        / rows['teacher_task']['net_profit'] if rows['teacher_task']['net_profit'] > 0 else None))
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP['seed'])).integers(0, 8, size=(20000, 8))
    overall = _paired_summary(pairs, draws)
    by_opponent = {o: _paired_summary([p for p in pairs if p['opponent'] == o], draws) for o in opponents}
    by_seed = [{'seed': s, **_paired_summary([p for p in pairs if p['seed'] == s])} for s in seeds]
    primary = overall['relative_net_profit_defined'] and overall['relative_net_profit_ci95'][0] > GATES['relative_ci_lower_gt']
    opponent_guards = {o: bool(v['relative_net_profit_defined']
                               and v['relative_net_profit_mean'] >= GATES['opponent_relative_mean_min']
                               and v['delta_means']['score'] >= GATES['opponent_score_delta_min'])
                       for o, v in by_opponent.items()}
    passed = bool(primary and all(opponent_guards.values()))
    return dict(schema='task-bc-holdout-statistics-v1', complete=True, games=96, pairs=48,
        cluster_unit='complete_seed', clusters=8, bootstrap=dict(BOOTSTRAP), gates=dict(GATES),
        arms=list(ARMS), reference='rule tasks with fixed learned crop head',
        overall=overall, by_opponent=by_opponent, by_seed=by_seed, paired_detail=pairs,
        primary_gate_passed=bool(primary), opponent_point_guards=opponent_guards,
        passed=passed,
        panel_improvement=bool(passed and overall['relative_net_profit_ci95'][0] > 0),
        limitations=['Only eight independent seed clusters; bootstrap coverage is approximate.',
                     'Opponent guards use point estimates, not statistical non-inferiority.',
                     'Admission does not promote a default agent or prove superiority to the original rule teacher.',
                     'Execution, replay, source, privacy and independent-review acceptance are separate.'])
