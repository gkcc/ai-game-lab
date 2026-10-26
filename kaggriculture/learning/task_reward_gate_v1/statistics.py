"""Frozen signal gates and whole-seed statistics for the reward-gate pilot.

Missing interventions remain population cells with an unknown reward target.
They are never changed into training zeros. In development, no opportunity
means the deployed chooser keeps the baseline and has realized uplift zero.
The holdout reference is the frozen *student*, not a rule teacher. These pure
statistics neither certify replay execution nor authorize model promotion.
"""
from collections import Counter
import math
from numbers import Real

import numpy as np


OPPONENTS = ('round2', 'market')
ARMS = ('baseline', 'gated')
METRICS = ('net_profit', 'opponent_net_profit', 'margin', 'score')
BOOTSTRAP = dict(replicates=20000, seed=2026102702, quantile_method='linear')
HOLDOUT = dict(max_pairs=32, relative_ci_lower_gt=0,
               opponent_relative_mean_min=-.02, opponent_score_delta_min=-.05,
               minimum_changed_request_games=4)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
             and math.isfinite(float(value)), 'Expected finite number: ' + name)
    return float(value)


def _seeds(values, count):
    _require(isinstance(values, (list, tuple)) and len(values) == count
             and all(type(s) is int and 0 <= s < 2**32 for s in values)
             and len(set(values)) == count, 'Unexpected seed population')
    return list(values)


def _collection(pairs, expected_seeds, count, *, predictions=False):
    seeds = _seeds(expected_seeds, count)
    expected = {(s, o, side) for s in seeds for o in OPPONENTS for side in (0, 1)}
    _require(isinstance(pairs, list) and len(pairs) == len(expected),
             'All registered population cells, including missing interventions, are required')
    indexed = {}
    for row in pairs:
        _require(isinstance(row, dict), 'Pair must be a mapping')
        _require(type(row['seed']) is int and type(row['side']) is int,
                 'Seed and seat must be integers')
        key = row['seed'], row['opponent'], row['side']
        _require(key in expected and key not in indexed, 'Unexpected or duplicate pair')
        _require(type(row['eligible']) is bool, 'Eligibility must be explicit')
        if row['eligible']:
            _number(row['delta_own'], 'delta_own')
            if row.get('x12') is not None:
                _require(isinstance(row['x12'], list) and len(row['x12']) == 12,
                         'Expected twelve gate features')
                for value in row['x12']:
                    _number(value, 'x12')
            if predictions:
                _number(row['prediction'], 'prediction')
        else:
            _require(row.get('delta_own') is None, 'Missing intervention has unknown target, not zero')
            _require(row.get('x12') is None and row.get('prediction') is None,
                     'Missing intervention cannot have gate features or prediction')
        indexed[key] = row
    _require(set(indexed) == expected, 'Incomplete registered pair population')
    ordered = [indexed[s, o, side] for s in seeds for o in OPPONENTS for side in (0, 1)]
    return seeds, ordered


def _signal_counts(rows):
    eligible = [r for r in rows if r['eligible']]
    nonzero = [r for r in eligible if r['delta_own'] != 0]
    return dict(population_cells=len(rows), eligible=len(eligible),
                missing=len(rows)-len(eligible), zero=len(eligible)-len(nonzero),
                nonzero=len(nonzero), positive=sum(r['delta_own'] > 0 for r in eligible),
                negative=sum(r['delta_own'] < 0 for r in eligible),
                nonzero_seed_count=len({r['seed'] for r in nonzero}),
                nonzero_seeds=sorted({r['seed'] for r in nonzero}))


def training_signal(pairs, expected_seeds):
    """Check whether 32 one-shot pairs support the sole preregistered fit."""
    seeds, rows = _collection(pairs, expected_seeds, 8)
    counts = _signal_counts(rows)
    checks = dict(eligible_ge_24=counts['eligible'] >= 24,
                  nonzero_ge_12=counts['nonzero'] >= 12,
                  positive_ge_4=counts['positive'] >= 4,
                  negative_ge_4=counts['negative'] >= 4,
                  nonzero_seeds_ge_6=counts['nonzero_seed_count'] >= 6)
    return dict(schema='task-reward-gate-training-signal-v1', passed=all(checks.values()),
                seeds=seeds, counts=counts, checks=checks,
                fit_population='Every eligible pair, including zero and negative targets; no missing target imputation',
                target='(intervention final own cash - baseline final own cash) / 1000',
                evidence_role='Training collection signal only; not independent improvement evidence')


def development_gate(pairs, expected_seeds):
    """One frozen gate decision per cell; no fitting or reselection on dev."""
    seeds, rows = _collection(pairs, expected_seeds, 2, predictions=True)
    counts = _signal_counts(rows)
    detail = []
    for row in rows:
        choose = bool(row['eligible'] and row['prediction'] > 0)
        realized = float(row['delta_own']) if choose else 0.0
        detail.append(dict(seed=row['seed'], opponent=row['opponent'], side=row['side'],
                           eligible=row['eligible'], observed_delta_own=row['delta_own'],
                           prediction=row.get('prediction'), choose_top2=choose,
                           realized_chooser_delta_own=realized))
    mean = float(np.mean([r['realized_chooser_delta_own'] for r in detail]))
    by_seed = {str(s): float(np.mean([r['realized_chooser_delta_own'] for r in detail
                                    if r['seed'] == s])) for s in seeds}
    choices = sum(r['choose_top2'] for r in detail)
    checks = dict(eligible_ge_6=counts['eligible'] >= 6,
                  nonzero_ge_4=counts['nonzero'] >= 4,
                  nonzero_in_both_seeds=counts['nonzero_seed_count'] == 2,
                  positive_predictions_ge_2=choices >= 2,
                  mean_realized_uplift_positive=mean > 0,
                  each_seed_mean_nonnegative=all(x >= 0 for x in by_seed.values()))
    return dict(schema='task-reward-gate-development-v1', passed=all(checks.values()),
                seeds=seeds, counts=counts, checks=checks, positive_predictions=choices,
                mean_realized_delta_own=mean, by_seed_mean_realized_delta_own=by_seed,
                paired_detail=detail,
                missing_effect='No eligible opportunity keeps the baseline: realized chooser effect zero, target unknown',
                limitation='Single-opportunity development checks do not establish repeated-use holdout benefit')


def _validated_holdout(ledger, cfg):
    seeds = _seeds(cfg['test_seeds'], 8)
    _require(cfg['opponents'] == list(OPPONENTS), 'Opponent order differs from registration')
    _require(cfg['bootstrap'] == BOOTSTRAP, 'Bootstrap differs from frozen protocol')
    _require(cfg['holdout'] == HOLDOUT, 'Holdout gates differ from frozen protocol')
    if 'arms' in cfg:
        _require(cfg['arms'] == list(ARMS), 'Holdout arms differ from frozen protocol')
    expected = {(s, o, side, arm) for s in seeds for o in OPPONENTS
                for side in (0, 1) for arm in ARMS}
    _require(isinstance(ledger, list) and len(ledger) == 64,
             'Exactly 64 completed holdout games required')
    indexed = {}
    for row in ledger:
        _require(isinstance(row, dict), 'Ledger row must be a mapping')
        _require(type(row['seed']) is int and type(row['side']) is int,
                 'Seed and seat must be integers')
        key = row['seed'], row['opponent'], row['side'], row['policy']
        _require(key in expected and key not in indexed, 'Unexpected or duplicate game')
        _require(row['passed'] is True and row['status'] == ['DONE', 'DONE']
                 and type(row['states']) is int and row['states'] == 720
                 and type(row['action_turns']) is int and row['action_turns'] == 719,
                 'Incomplete or unaccepted official game')
        _require(type(row['agent_errors']) is int and row['agent_errors'] == 0,
                 'Official agent errors')
        for field in ('errors', 'timeouts', 'consumer_errors', 'fallbacks', 'load_errors', 'inference_errors'):
            if field in row:
                _require(type(row[field]) is int and row[field] == 0, 'Nonzero execution errors: '+field)
        values = {name: _number(row[name], name) for name in
                  ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank', *METRICS)}
        _require(values['net_profit'] == values['bank'] - values['initial_bank'],
                 'Own net differs from cash change')
        _require(values['opponent_net_profit'] == values['opponent_bank'] - values['opponent_initial_bank'],
                 'Opponent net differs from cash change')
        _require(values['margin'] == values['bank'] - values['opponent_bank'],
                 'Margin differs from terminal cash difference')
        score = 1.0 if values['margin'] > 0 else 0.0 if values['margin'] < 0 else .5
        _require(values['score'] == score, 'Score differs from terminal result')
        for field, expected_values in (
            ('banks', (row['bank'], row['opponent_bank'])),
            ('initial_banks', (row['initial_bank'], row['opponent_initial_bank'])),
            ('rewards', (row['bank'], row['opponent_bank'])),
        ):
            if field in row:
                actual = row[field]
                _require(isinstance(actual, list) and len(actual) == 2,
                         'Expected both player values: '+field)
                _require(actual[row['side']] == expected_values[0]
                         and actual[1-row['side']] == expected_values[1], 'Seat mismatch: '+field)
        changes = row['complete_request_changes']
        _require(type(changes) is int and 0 <= changes <= 3,
                 'Expected bounded same-observation complete-request changes')
        _require(row['policy'] != 'baseline' or changes == 0,
                 'Reference cannot contain gated request changes')
        indexed[key] = row
    _require(set(indexed) == expected, 'Missing registered games')
    return seeds, indexed


def _arm_summary(rows):
    counts = Counter('win' if r['score'] == 1 else 'draw' if r['score'] == .5 else 'loss' for r in rows)
    return dict(games=len(rows), wins=counts['win'], draws=counts['draw'], losses=counts['loss'],
                means={m: float(np.mean([r[m] for r in rows])) for m in METRICS})


def _interval(values):
    _require(bool(np.isfinite(values).all()), 'Nonfinite statistical result')
    return [float(x) for x in np.quantile(values, [.025, .975], method='linear')]


def _summary(pairs, *, seeds=None, draws=None):
    arms = {a: _arm_summary([p[a] for p in pairs]) for a in ARMS}
    deltas = {m: float(np.mean([p['delta'][m] for p in pairs])) for m in METRICS}
    reference = arms['baseline']['means']['net_profit']
    result = dict(pairs=len(pairs), arms=arms, delta_means=deltas,
                  reference_net_profit_mean=reference,
                  relative_net_profit_mean=deltas['net_profit']/reference if reference > 0 else None,
                  relative_net_profit_defined=reference > 0)
    if draws is None:
        return result
    groups = [[p for p in pairs if p['seed'] == seed] for seed in seeds]
    _require(len(groups) == 8 and len({len(g) for g in groups}) == 1 and all(groups),
             'Eight complete equal-sized seed clusters required')
    seed_delta = {m: np.array([np.mean([p['delta'][m] for p in g]) for g in groups]) for m in METRICS}
    seed_reference = np.array([np.mean([p['baseline']['net_profit'] for p in g]) for g in groups])
    resampled = {m: x[draws].mean(axis=1) for m, x in seed_delta.items()}
    denominator = seed_reference[draws].mean(axis=1)
    defined = bool(reference > 0 and np.all(denominator > 0))
    result.update(delta_ci95={m: _interval(x) for m, x in resampled.items()},
                  relative_net_profit_defined=defined,
                  nonpositive_bootstrap_reference_count=int(np.count_nonzero(denominator <= 0)),
                  bootstrap_reference_min=float(denominator.min()),
                  relative_net_profit_ci95=_interval(resampled['net_profit']/denominator) if defined else None)
    if not defined:
        result['relative_net_profit_mean'] = None
        result['relative_undefined_reason'] = 'Nonpositive point or resampled reference net profit'
    return result


def holdout_statistics(ledger, cfg):
    """Validate the full panel, then compare repeated gate use to the student.

    Eight seed clusters retain both opponents and both seats. Each bootstrap
    resample recomputes both numerator and baseline denominator. Per-opponent
    observed guards are not statistical non-inferiority guarantees.
    """
    seeds, indexed = _validated_holdout(ledger, cfg)
    pairs = []
    for seed in seeds:
        for opponent in OPPONENTS:
            for side in (0, 1):
                rows = {a: indexed[seed, opponent, side, a] for a in ARMS}
                for field in ('initial_bank', 'opponent_initial_bank'):
                    _require(rows['baseline'][field] == rows['gated'][field], 'Paired initial cash differs')
                pairs.append(dict(seed=seed, opponent=opponent, side=side,
                    **{a: {m: rows[a][m] for m in
                           ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank', *METRICS)} for a in ARMS},
                    delta={m: rows['gated'][m]-rows['baseline'][m] for m in METRICS},
                    complete_request_changes=rows['gated']['complete_request_changes']))
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP['seed'])).integers(0, 8, size=(20000, 8))
    overall = _summary(pairs, seeds=seeds, draws=draws)
    by_opponent = {o: _summary([p for p in pairs if p['opponent'] == o], seeds=seeds, draws=draws)
                   for o in OPPONENTS}
    by_seed = [dict(seed=s, **_summary([p for p in pairs if p['seed'] == s])) for s in seeds]
    primary = bool(overall['relative_net_profit_defined']
                   and overall['relative_net_profit_ci95'][0] > HOLDOUT['relative_ci_lower_gt'])
    opponent_guards = {o: bool(v['relative_net_profit_defined']
                              and v['relative_net_profit_mean'] >= HOLDOUT['opponent_relative_mean_min']
                              and v['delta_means']['score'] >= HOLDOUT['opponent_score_delta_min'])
                       for o, v in by_opponent.items()}
    changed_games = sum(p['complete_request_changes'] > 0 for p in pairs)
    effective = changed_games >= HOLDOUT['minimum_changed_request_games']
    return dict(schema='task-reward-gate-holdout-statistics-v1', complete=True, games=64, pairs=32,
                cluster_unit='complete_seed', clusters=8, bootstrap=dict(BOOTSTRAP), gates=dict(HOLDOUT),
                arms=list(ARMS), reference='Frozen task and crop student; not the rule teacher',
                overall=overall, by_opponent=by_opponent, by_seed=by_seed, paired_detail=pairs,
                primary_gate_passed=primary, opponent_point_guards=opponent_guards,
                complete_action_changed_games=changed_games, complete_action_effect_gate_passed=effective,
                research_positive=bool(primary and effective and all(opponent_guards.values())),
                limitations=['Only eight independent seed clusters; bootstrap coverage is approximate.',
                             'Opponent guards are observed point guards, not statistical non-inferiority.',
                             'No comparison with teacher_task or the original rule teacher; no teacher-parity claim.',
                             'Execution, source, privacy and independent review are separate acceptance conditions.',
                             'No default replacement or model admission follows from this pilot.'])
