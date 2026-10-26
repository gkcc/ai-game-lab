"""Preregistered margin-learning gates and a three-arm, whole-seed holdout.

Each collection cell is one same-reset teacher-task intervention followed by
student continuation. A missing disagreement has unknown outcome labels; it is
not an observed zero. The chooser's realized effect in that cell is zero because
it keeps the student. Repeated gate use is assessed only in the fresh holdout.
"""
from collections import Counter
import math
from numbers import Real

import numpy as np


OPPONENTS = ('round2', 'market')
ARMS = ('baseline', 'gated', 'teacher_task')
METRICS = ('net_profit', 'opponent_net_profit', 'margin', 'score')
DELTAS = ('delta_own', 'delta_opponent', 'delta_margin', 'delta_score')
BOOTSTRAP = dict(replicates=20000, seed=2026102704, quantile_method='linear')
HOLDOUT = dict(max_pairs=32, margin_ci_lower_gt=0,
               opponent_relative_mean_min=-.02, opponent_score_delta_min=-.05,
               minimum_changed_request_games=4, teacher_relative_ci_lower_gt=-.02,
               teacher_opponent_relative_mean_min=-.05,
               teacher_opponent_score_delta_min=-.05)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
             and math.isfinite(float(value)), 'Expected finite number: '+name)
    return float(value)


def _seeds(values, count):
    _require(isinstance(values, (list, tuple)) and len(values) == count
             and all(type(s) is int and 0 <= s < 2**32 for s in values)
             and len(set(values)) == count, 'Unexpected registered seed population')
    return list(values)


def _collection(pairs, expected_seeds, count, *, predictions=False):
    seeds = _seeds(expected_seeds, count)
    expected = {(s, o, side) for s in seeds for o in OPPONENTS for side in (0, 1)}
    _require(isinstance(pairs, list) and len(pairs) == len(expected),
             'Every registered cell, including missing disagreements, is required')
    indexed = {}
    for row in pairs:
        _require(isinstance(row, dict), 'Expected pair mapping')
        _require(type(row['seed']) is int and type(row['side']) is int,
                 'Seed and seat must be integers')
        key = row['seed'], row['opponent'], row['side']
        _require(key in expected and key not in indexed, 'Unexpected or duplicate pair')
        _require(type(row['eligible']) is bool, 'Eligibility must be explicit')
        _number(row['baseline_net_profit'], 'baseline_net_profit')
        if row['eligible']:
            values = {k: _number(row[k], k) for k in DELTAS}
            _require(values['delta_margin'] == values['delta_own']-values['delta_opponent'],
                     'Margin effect differs from own minus opponent cash effects')
            _require(values['delta_score'] in (-1., -.5, 0., .5, 1.), 'Invalid match-score difference')
            _require(isinstance(row['x8'], list) and len(row['x8']) == 8,
                     'Eligible disagreement requires eight finite features')
            for value in row['x8']:
                _number(value, 'x8')
            if predictions:
                _number(row['prediction'], 'prediction')
        else:
            _require(all(row[k] is None for k in DELTAS),
                     'Missing disagreement targets stay unknown, never imputed zero')
            _require(row['x8'] is None and row.get('prediction') is None,
                     'Missing disagreement cannot have features or prediction')
        indexed[key] = row
    _require(set(indexed) == expected, 'Incomplete registered pair population')
    return seeds, [indexed[s, o, side] for s in seeds for o in OPPONENTS for side in (0, 1)]


def _counts(rows):
    eligible = [r for r in rows if r['eligible']]
    nonzero = [r for r in eligible if r['delta_margin'] != 0]
    return dict(population_cells=len(rows), eligible=len(eligible), missing=len(rows)-len(eligible),
                zero_margin=len(eligible)-len(nonzero), nonzero_margin=len(nonzero),
                positive_margin=sum(r['delta_margin'] > 0 for r in eligible),
                negative_margin=sum(r['delta_margin'] < 0 for r in eligible),
                nonzero_seed_count=len({r['seed'] for r in nonzero}),
                nonzero_seeds=sorted({r['seed'] for r in nonzero}))


def early_signal(pairs, first_two_seeds):
    """The single chronological information check inside the training panel."""
    seeds, rows = _collection(pairs, first_two_seeds, 2)
    counts = _counts(rows)
    checks = dict(eligible_ge_4=counts['eligible'] >= 4,
                  any_nonzero_margin=counts['nonzero_margin'] >= 1)
    return dict(schema='task-request-gate-early-signal-v1', seeds=seeds, counts=counts,
                checks=checks, passed=all(checks.values()),
                role='First two registered training seeds, not an additional probe or replacement sample')


def training_signal(pairs, expected_seeds):
    seeds, rows = _collection(pairs, expected_seeds, 8)
    counts = _counts(rows)
    checks = dict(eligible_ge_24=counts['eligible'] >= 24,
                  nonzero_margin_ge_12=counts['nonzero_margin'] >= 12,
                  nonzero_seeds_ge_6=counts['nonzero_seed_count'] >= 6)
    return dict(schema='task-request-gate-training-signal-v1', seeds=seeds, counts=counts,
                checks=checks, passed=all(checks.values()),
                fit_population='All eligible pairs, including zero and negative margin targets; no sign quota or missing-label imputation',
                target='(intervention terminal cash margin - baseline terminal cash margin) / 1000',
                role='Training information gate; not independent improvement evidence')


def _equal_seed_mean(rows, seeds, field):
    per_seed = {str(s): float(np.mean([r[field] for r in rows if r['seed'] == s])) for s in seeds}
    return float(np.mean(list(per_seed.values()))), per_seed


def development_gate(pairs, expected_seeds):
    """Frozen one-turn choices must beat both constants without dev refitting."""
    seeds, rows = _collection(pairs, expected_seeds, 2, predictions=True)
    counts = _counts(rows)
    detail = []
    for row in rows:
        choose = bool(row['eligible'] and row['prediction'] > 0)
        detail.append(dict(seed=row['seed'], opponent=row['opponent'], side=row['side'],
            eligible=row['eligible'], prediction=row.get('prediction'), choose_teacher_task=choose,
            baseline_net_profit=row['baseline_net_profit'],
            observed_delta={k: row[k] for k in DELTAS},
            chosen_margin=float(row['delta_margin']) if choose else 0.,
            chosen_own=float(row['delta_own']) if choose else 0.,
            chosen_opponent=float(row['delta_opponent']) if choose else 0.,
            chosen_score=float(row['delta_score']) if choose else 0.,
            always_teacher_margin=float(row['delta_margin']) if row['eligible'] else 0.))
    means, by_seed = {}, {}
    for key in ('chosen_margin', 'chosen_own', 'chosen_opponent', 'chosen_score',
                'always_teacher_margin', 'baseline_net_profit'):
        means[key], by_seed[key] = _equal_seed_mean(detail, seeds, key)
    relative = means['chosen_own']/means['baseline_net_profit'] if means['baseline_net_profit'] > 0 else None
    checks = dict(eligible_ge_6=counts['eligible'] >= 6,
                  nonzero_margin_ge_4=counts['nonzero_margin'] >= 4,
                  nonzero_in_both_seeds=counts['nonzero_seed_count'] == 2,
                  beats_constant_student=means['chosen_margin'] > 0,
                  beats_constant_teacher=means['chosen_margin'] > means['always_teacher_margin'],
                  each_seed_selected_margin_nonnegative=all(x >= 0 for x in by_seed['chosen_margin'].values()),
                  own_relative_defined=relative is not None,
                  own_relative_ge_minus_2percent=relative is not None and relative >= -.02,
                  selected_score_delta_ge_minus_005=means['chosen_score'] >= -.05)
    return dict(schema='task-request-gate-development-v1', seeds=seeds, counts=counts, checks=checks,
                passed=all(checks.values()), chosen_teacher_cells=sum(r['choose_teacher_task'] for r in detail),
                equal_seed_means=means, by_seed_means=by_seed, selected_own_relative=relative,
                paired_detail=detail,
                missing_effect='No disagreement keeps the student: realized choice effect zero, observed targets unknown',
                boundary='Always teacher here means the one selected turn, not full teacher_task on every turn; repeated use requires holdout')


def _validated_holdout(ledger, cfg):
    seeds = _seeds(cfg['test_seeds'], 8)
    _require(cfg['opponents'] == list(OPPONENTS), 'Registered opponents differ')
    _require(cfg['bootstrap'] == BOOTSTRAP, 'Registered bootstrap differs')
    _require(cfg['holdout'] == HOLDOUT, 'Registered holdout gates differ')
    if 'arms' in cfg:
        _require(cfg['arms'] == list(ARMS), 'Registered holdout arms differ')
    expected = {(s, o, side, a) for s in seeds for o in OPPONENTS for side in (0, 1) for a in ARMS}
    _require(isinstance(ledger, list) and len(ledger) == 96, 'Exactly 96 completed three-arm games required')
    indexed = {}
    for row in ledger:
        _require(isinstance(row, dict), 'Expected ledger mapping')
        _require(type(row['seed']) is int and type(row['side']) is int, 'Seed and seat must be integers')
        key = row['seed'], row['opponent'], row['side'], row['policy']
        _require(key in expected and key not in indexed, 'Unexpected or duplicate game')
        _require(row['passed'] is True and row['status'] == ['DONE', 'DONE']
                 and type(row['states']) is int and row['states'] == 720
                 and type(row['action_turns']) is int and row['action_turns'] == 719,
                 'Incomplete or unaccepted official game')
        _require(type(row['agent_errors']) is int and row['agent_errors'] == 0, 'Official agent errors')
        for field in ('errors', 'timeouts', 'consumer_errors', 'fallbacks', 'load_errors', 'inference_errors'):
            if field in row:
                _require(type(row[field]) is int and row[field] == 0, 'Nonzero execution errors: '+field)
        values = {m: _number(row[m], m) for m in
                  ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank', *METRICS)}
        _require(values['net_profit'] == values['bank']-values['initial_bank'], 'Own cash change mismatch')
        _require(values['opponent_net_profit'] == values['opponent_bank']-values['opponent_initial_bank'],
                 'Opponent cash change mismatch')
        _require(values['margin'] == values['bank']-values['opponent_bank'], 'Terminal margin mismatch')
        score = 1. if values['margin'] > 0 else 0. if values['margin'] < 0 else .5
        _require(values['score'] == score, 'Match score mismatch')
        for field, expected_values in (
            ('banks', (row['bank'], row['opponent_bank'])),
            ('initial_banks', (row['initial_bank'], row['opponent_initial_bank'])),
            ('rewards', (row['bank'], row['opponent_bank'])),
        ):
            if field in row:
                actual = row[field]
                _require(isinstance(actual, list) and len(actual) == 2
                         and actual[row['side']] == expected_values[0]
                         and actual[1-row['side']] == expected_values[1], 'Player accounting mismatch: '+field)
        changes = row['complete_request_changes']
        _require(type(changes) is int and 0 <= changes <= (3 if row['policy'] == 'gated' else 719),
                 'Invalid complete-request change count')
        _require(row['policy'] != 'baseline' or changes == 0, 'Student baseline cannot have gated interventions')
        indexed[key] = row
    _require(set(indexed) == expected, 'Missing registered three-arm games')
    return seeds, indexed


def _arm_summary(rows):
    counts = Counter('win' if r['score'] == 1 else 'draw' if r['score'] == .5 else 'loss' for r in rows)
    return dict(games=len(rows), wins=counts['win'], draws=counts['draw'], losses=counts['loss'],
                means={m: float(np.mean([r[m] for r in rows])) for m in METRICS})


def _interval(values):
    _require(bool(np.isfinite(values).all()), 'Nonfinite statistical output')
    return [float(v) for v in np.quantile(values, [.025, .975], method='linear')]


def _paired_summary(pairs, reference, *, seeds=None, draws=None):
    arms = {a: _arm_summary([p[a] for p in pairs]) for a in (reference, 'gated')}
    deltas = {m: float(np.mean([p['gated'][m]-p[reference][m] for p in pairs])) for m in METRICS}
    denominator = arms[reference]['means']['net_profit']
    result = dict(pairs=len(pairs), reference=reference, arms=arms, delta_means=deltas,
                  reference_net_profit_mean=denominator,
                  relative_net_profit_mean=deltas['net_profit']/denominator if denominator > 0 else None,
                  relative_net_profit_defined=denominator > 0)
    if draws is None:
        return result
    groups = [[p for p in pairs if p['seed'] == s] for s in seeds]
    _require(len(groups) == 8 and len({len(g) for g in groups}) == 1 and all(groups),
             'Eight complete equally-sized seed clusters required')
    seed_deltas = {m: np.array([np.mean([p['gated'][m]-p[reference][m] for p in g]) for g in groups])
                   for m in METRICS}
    seed_reference = np.array([np.mean([p[reference]['net_profit'] for p in g]) for g in groups])
    resampled = {m: x[draws].mean(axis=1) for m, x in seed_deltas.items()}
    resampled_reference = seed_reference[draws].mean(axis=1)
    defined = bool(denominator > 0 and np.all(resampled_reference > 0))
    result.update(delta_ci95={m: _interval(x) for m, x in resampled.items()},
                  relative_net_profit_defined=defined,
                  nonpositive_bootstrap_reference_count=int(np.count_nonzero(resampled_reference <= 0)),
                  bootstrap_reference_min=float(resampled_reference.min()),
                  relative_net_profit_ci95=_interval(resampled['net_profit']/resampled_reference) if defined else None)
    if not defined:
        result['relative_net_profit_mean'] = None
        result['relative_undefined_reason'] = 'Nonpositive point or resampled reference own-net-profit denominator'
    return result


def holdout_statistics(ledger, cfg):
    """Separate student improvement, hybrid-teacher retention and superiority."""
    seeds, indexed = _validated_holdout(ledger, cfg)
    pairs = []
    for seed in seeds:
        for opponent in OPPONENTS:
            for side in (0, 1):
                rows = {a: indexed[seed, opponent, side, a] for a in ARMS}
                for field in ('initial_bank', 'opponent_initial_bank'):
                    _require(len({rows[a][field] for a in ARMS}) == 1, 'Three-arm initial cash differs')
                pairs.append(dict(seed=seed, opponent=opponent, side=side,
                    **{a: {m: rows[a][m] for m in
                           ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank', *METRICS)} for a in ARMS},
                    delta_vs_baseline={m: rows['gated'][m]-rows['baseline'][m] for m in METRICS},
                    delta_vs_teacher_task={m: rows['gated'][m]-rows['teacher_task'][m] for m in METRICS},
                    complete_request_changes=rows['gated']['complete_request_changes']))
    # One shared draw array for both comparisons and both opponent subgroups.
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP['seed'])).integers(0, 8, size=(20000, 8))
    comparisons = {}
    for reference in ('baseline', 'teacher_task'):
        comparisons[reference] = dict(
            overall=_paired_summary(pairs, reference, seeds=seeds, draws=draws),
            by_opponent={o: _paired_summary([p for p in pairs if p['opponent'] == o], reference,
                                            seeds=seeds, draws=draws) for o in OPPONENTS},
            by_seed=[dict(seed=s, **_paired_summary([p for p in pairs if p['seed'] == s], reference)) for s in seeds])
    student = comparisons['baseline']
    teacher = comparisons['teacher_task']
    student_primary = student['overall']['delta_ci95']['margin'][0] > HOLDOUT['margin_ci_lower_gt']
    student_guards = {o: bool(v['relative_net_profit_defined']
                             and v['relative_net_profit_mean'] >= HOLDOUT['opponent_relative_mean_min']
                             and v['delta_means']['score'] >= HOLDOUT['opponent_score_delta_min'])
                      for o, v in student['by_opponent'].items()}
    teacher_primary = bool(teacher['overall']['relative_net_profit_defined']
                           and teacher['overall']['relative_net_profit_ci95'][0] > HOLDOUT['teacher_relative_ci_lower_gt'])
    teacher_guards = {o: bool(v['relative_net_profit_defined']
                             and v['relative_net_profit_mean'] >= HOLDOUT['teacher_opponent_relative_mean_min']
                             and v['delta_means']['score'] >= HOLDOUT['teacher_opponent_score_delta_min'])
                      for o, v in teacher['by_opponent'].items()}
    changed_games = sum(p['complete_request_changes'] > 0 for p in pairs)
    effective = changed_games >= HOLDOUT['minimum_changed_request_games']
    research_positive = bool(student_primary and all(student_guards.values()) and effective)
    retention = bool(teacher_primary and all(teacher_guards.values()))
    teacher_margin_positive = teacher['overall']['delta_ci95']['margin'][0] > 0
    return dict(schema='task-request-gate-holdout-statistics-v1', complete=True,
                games=96, pairs=32, cluster_unit='complete_seed', clusters=8,
                bootstrap=dict(BOOTSTRAP), gates=dict(HOLDOUT), arms=list(ARMS),
                arm_summaries={a: _arm_summary([p[a] for p in pairs]) for a in ARMS},
                comparisons=comparisons, paired_detail=pairs,
                student_margin_gate_passed=bool(student_primary), student_opponent_point_guards=student_guards,
                complete_action_changed_games=changed_games, complete_action_effect_gate_passed=effective,
                teacher_retention_primary_passed=teacher_primary, teacher_opponent_point_guards=teacher_guards,
                teacher_margin_gate_passed=bool(teacher_margin_positive), research_positive=research_positive,
                teacher_retention=retention,
                teacher_superiority=bool(research_positive and retention and teacher_margin_positive),
                limitations=['The baseline is the frozen failed task/crop student.',
                             'teacher_task uses rule tasks plus the SAME frozen crop network, not the original all-rule teacher.',
                             'Only eight independent seed clusters and two known opponents; bootstrap coverage is approximate.',
                             'Per-opponent guards are observed point checks, not statistical non-inferiority guarantees.',
                             'These are research flags; source, execution, replay, privacy and independent review remain separate.',
                             'No default change or deployed-model promotion is authorized by these flags.'])
