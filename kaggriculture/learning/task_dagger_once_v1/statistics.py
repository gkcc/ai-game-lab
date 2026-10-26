"""Frozen supervision, development and whole-seed three-arm admission gates.

These pure consumers never load policies or observations. Candidate request
differences are evidence supplied by the official paired replay consumer.
"""
import numpy as np

from learning.task_request_gate_v1.statistics import (
    METRICS, _arm_summary, _interval, _number, _require, _seeds,
)

OPPONENTS = ('round2', 'market')
ARMS = ('baseline', 'candidate', 'teacher_task')
BOOTSTRAP = dict(replicates=20000, seed=2026102706, quantile_method='linear')
EARLY = dict(minimum_disagreements=128, minimum_games_with_disagreement=2)
DEVELOPMENT = dict(own_mean_delta_gt=0., overall_margin_delta_min=0.,
    overall_score_delta_min=0., opponent_relative_mean_min=-.02,
    opponent_score_delta_min=-.05, teacher_relative_mean_min=-.02,
    teacher_opponent_score_delta_min=-.05)
HOLDOUT = dict(own_ci_lower_gt=0., minimum_relative_gain=.005,
    overall_margin_delta_min=0., overall_score_delta_min=0.,
    opponent_relative_mean_min=-.02, opponent_score_delta_min=-.05,
    minimum_changed_request_games=4, teacher_relative_ci_lower_gt=-.02,
    teacher_opponent_relative_mean_min=-.05, teacher_opponent_score_delta_min=-.05)
CASH_FIELDS = ('bank', 'opponent_bank', 'initial_bank', 'opponent_initial_bank', *METRICS)


def _configuration(cfg):
    groups = {name: _seeds(cfg[name], n) for name, n in
              (('train_seeds', 4), ('validation_seeds', 2), ('test_seeds', 8))}
    _require(len(set(s for values in groups.values() for s in values)) == 14,
             'Training, development and holdout seeds must be disjoint')
    _require(cfg['opponents'] == list(OPPONENTS) and cfg['arms'] == list(ARMS),
             'Registered opponents or arms differ')
    _require(cfg['seats'] == [0, 1] and all(type(s) is int for s in cfg['seats']),
             'Registered seats differ')
    for name, value in (('bootstrap', BOOTSTRAP), ('early_screen', EARLY),
                        ('development', DEVELOPMENT), ('holdout', HOLDOUT)):
        _require(cfg[name] == value, 'Registered settings differ: '+name)
    return groups


def _complete(row):
    _require(row['passed'] is True and row['status'] == ['DONE', 'DONE']
             and type(row['states']) is int and row['states'] == 720
             and type(row['action_turns']) is int and row['action_turns'] == 719,
             'Incomplete or unaccepted official game')
    _require(type(row['agent_errors']) is int and row['agent_errors'] == 0,
             'Official agent errors')
    for field in ('errors', 'timeouts', 'consumer_errors', 'fallbacks',
                  'load_errors', 'inference_errors'):
        if field in row:
            _require(type(row[field]) is int and row[field] == 0,
                     'Nonzero execution errors: '+field)


def early_gate(summaries, cfg):
    """One first-seed screen; insufficient supervision is a normal false gate."""
    seed = _configuration(cfg)['train_seeds'][0]
    expected = {(seed, o, side) for o in OPPONENTS for side in (0, 1)}
    _require(isinstance(summaries, list) and len(summaries) == 4,
             'Exactly four first-training-seed game summaries required')
    indexed, games = {}, set()
    for row in summaries:
        _require(isinstance(row, dict), 'Expected summary mapping')
        _require(type(row['seed']) is int and type(row['side']) is int,
                 'Seed and seat must be integers')
        key = row['seed'], row['opponent'], row['side']
        _require(key in expected and key not in indexed, 'Unexpected or duplicate early game')
        _require(isinstance(row['game'], str) and row['game'] and row['game'] not in games,
                 'Missing or repeated game identity')
        _require(row['split'] == 'train' and row['policy'] == 'baseline',
                 'Only first-seed training student data belongs to this gate')
        _complete(row)
        _require(row['label_binding_passed'] is True, 'Current-callback label binding failed')
        for field in ('groups', 'candidate_rows', 'teacher_ID_disagreements'):
            _require(type(row[field]) is int and row[field] >= 0, 'Invalid supervision count')
        _require(row['candidate_rows'] >= row['groups'] >= row['teacher_ID_disagreements'],
                 'Inconsistent supervision counts')
        indexed[key] = row
        games.add(row['game'])
    counts = dict(games=4, groups=sum(r['groups'] for r in summaries),
        candidate_rows=sum(r['candidate_rows'] for r in summaries),
        teacher_ID_disagreements=sum(r['teacher_ID_disagreements'] for r in summaries),
        games_with_disagreement=sum(r['teacher_ID_disagreements'] > 0 for r in summaries))
    checks = dict(enough_disagreements=counts['teacher_ID_disagreements'] >= EARLY['minimum_disagreements'],
                  enough_games=counts['games_with_disagreement'] >= EARLY['minimum_games_with_disagreement'])
    return dict(schema='task-dagger-once-early-v1', seed=seed, counts=counts,
                gates=dict(EARLY), checks=checks, passed=all(checks.values()),
                role='Supervision quantity only; teacher-ID disagreement is not economic harm or unique optimality')


def _panel(rows, cfg, split):
    groups = _configuration(cfg)
    seeds = groups['validation_seeds' if split == 'dev' else 'test_seeds']
    expected = {(s, o, side, a) for s in seeds for o in OPPONENTS
                for side in (0, 1) for a in ARMS}
    _require(isinstance(rows, list) and len(rows) == len(expected),
             'Every registered three-arm game is required')
    indexed, names = {}, set()
    for row in rows:
        _require(isinstance(row, dict), 'Expected ledger mapping')
        _require(type(row['seed']) is int and type(row['side']) is int,
                 'Seed and seat must be integers')
        key = row['seed'], row['opponent'], row['side'], row['policy']
        _require(key in expected and key not in indexed, 'Unexpected or duplicate game')
        _require(row['split'] == split, 'Development and holdout cannot be mixed')
        _require(isinstance(row['game'], str) and row['game'] and row['game'] not in names,
                 'Missing or repeated game identity')
        _complete(row)
        values = {field: _number(row[field], field) for field in CASH_FIELDS}
        _require(values['net_profit'] == values['bank']-values['initial_bank'], 'Own cash change mismatch')
        _require(values['opponent_net_profit'] == values['opponent_bank']-values['opponent_initial_bank'],
                 'Opponent cash change mismatch')
        _require(values['margin'] == values['bank']-values['opponent_bank'], 'Terminal margin mismatch')
        score = 1. if values['margin'] > 0 else 0. if values['margin'] < 0 else .5
        _require(values['score'] == score, 'Match score mismatch')
        for field, cash in (('banks', ('bank', 'opponent_bank')),
                            ('initial_banks', ('initial_bank', 'opponent_initial_bank')),
                            ('rewards', ('bank', 'opponent_bank'))):
            if field in row:
                actual = row[field]
                _require(isinstance(actual, list) and len(actual) == 2, 'Expected two-player cash array')
                actual = [_number(v, field) for v in actual]
                _require(actual[row['side']] == values[cash[0]]
                         and actual[1-row['side']] == values[cash[1]], 'Player accounting mismatch: '+field)
        if row['policy'] == 'candidate':
            step = row['first_request_difference_step']
            _require(step is None or type(step) is int and 0 <= step < 719,
                     'Expected actual first request difference step or explicit null')
        else:
            _require(row.get('first_request_difference_step') is None,
                     'Only candidate rows carry paired control evidence')
        indexed[key] = row
        names.add(row['game'])
    pairs = []
    for s in seeds:
        for o in OPPONENTS:
            for side in (0, 1):
                cell = {a: indexed[s, o, side, a] for a in ARMS}
                for field in ('initial_bank', 'opponent_initial_bank'):
                    _require(len({r[field] for r in cell.values()}) == 1, 'Three-arm initial cash differs')
                pairs.append(dict(seed=s, opponent=o, side=side,
                    **{a: dict(game=cell[a]['game'], **{m: cell[a][m] for m in CASH_FIELDS}) for a in ARMS},
                    delta_vs_baseline={m: cell['candidate'][m]-cell['baseline'][m] for m in METRICS},
                    delta_vs_teacher_task={m: cell['candidate'][m]-cell['teacher_task'][m] for m in METRICS},
                    first_request_difference_step=cell['candidate']['first_request_difference_step']))
    return seeds, pairs


def _summary(pairs, reference, seeds, draws=None):
    arms = {a: _arm_summary([p[a] for p in pairs]) for a in (reference, 'candidate')}
    deltas = {m: float(np.mean([p['candidate'][m]-p[reference][m] for p in pairs])) for m in METRICS}
    denominator = arms[reference]['means']['net_profit']
    result = dict(pairs=len(pairs), reference=reference, arms=arms, delta_means=deltas,
        reference_net_profit_mean=denominator, relative_net_profit_defined=denominator > 0,
        relative_net_profit_mean=deltas['net_profit']/denominator if denominator > 0 else None)
    if draws is not None:
        groups = [[p for p in pairs if p['seed'] == s] for s in seeds]
        _require(len(groups) == 8 and all(groups) and len({len(g) for g in groups}) == 1,
                 'Eight complete equally-sized seed clusters required')
        seed_deltas = {m: np.array([np.mean([p['candidate'][m]-p[reference][m] for p in g])
                                   for g in groups]) for m in METRICS}
        seed_reference = np.array([np.mean([p[reference]['net_profit'] for p in g]) for g in groups])
        sampled = {m: v[draws].mean(axis=1) for m, v in seed_deltas.items()}
        sampled_reference = seed_reference[draws].mean(axis=1)
        defined = bool(denominator > 0 and np.all(sampled_reference > 0))
        result.update(delta_ci95={m: _interval(v) for m, v in sampled.items()},
            relative_net_profit_defined=defined,
            nonpositive_bootstrap_reference_count=int(np.count_nonzero(sampled_reference <= 0)),
            bootstrap_reference_min=float(sampled_reference.min()),
            relative_net_profit_ci95=_interval(sampled['net_profit']/sampled_reference) if defined else None)
        if not defined:
            result['relative_net_profit_mean'] = None
    if not result['relative_net_profit_defined']:
        result['relative_undefined_reason'] = 'Nonpositive point or resampled reference own-net-profit denominator'
    return result


def _comparisons(pairs, seeds, draws=None):
    return {reference: dict(overall=_summary(pairs, reference, seeds, draws),
        by_opponent={o: _summary([p for p in pairs if p['opponent'] == o], reference, seeds, draws)
                     for o in OPPONENTS},
        by_seed=[dict(seed=s, **_summary([p for p in pairs if p['seed'] == s], reference, [s]))
                 for s in seeds]) for reference in ('baseline', 'teacher_task')}


def _relative_at_least(summary, threshold):
    return bool(summary['relative_net_profit_defined'] and summary['relative_net_profit_mean'] >= threshold)


def development_gate(rows, cfg):
    """Fixed model economic screen; no model selection or statistical CI here."""
    seeds, pairs = _panel(rows, cfg, 'dev')
    comparisons = _comparisons(pairs, seeds)
    student, teacher = comparisons['baseline'], comparisons['teacher_task']
    mean = student['overall']['delta_means']
    checks = dict(own_mean_improved=mean['net_profit'] > DEVELOPMENT['own_mean_delta_gt'],
        overall_margin_not_lower=mean['margin'] >= DEVELOPMENT['overall_margin_delta_min'],
        overall_score_not_lower=mean['score'] >= DEVELOPMENT['overall_score_delta_min'],
        student_ratios_defined=student['overall']['relative_net_profit_defined'],
        teacher_mean_retained=_relative_at_least(teacher['overall'], DEVELOPMENT['teacher_relative_mean_min']))
    student_guards = {o: bool(_relative_at_least(v, DEVELOPMENT['opponent_relative_mean_min'])
                              and v['delta_means']['score'] >= DEVELOPMENT['opponent_score_delta_min'])
                      for o, v in student['by_opponent'].items()}
    teacher_guards = {o: bool(v['relative_net_profit_defined']
                              and v['delta_means']['score'] >= DEVELOPMENT['teacher_opponent_score_delta_min'])
                      for o, v in teacher['by_opponent'].items()}
    return dict(schema='task-dagger-once-development-v1', games=24, pairs=8, seeds=seeds,
        gates=dict(DEVELOPMENT), comparisons=comparisons, paired_detail=pairs, checks=checks,
        student_opponent_point_guards=student_guards, teacher_opponent_point_guards=teacher_guards,
        passed=bool(all(checks.values()) and all(student_guards.values()) and all(teacher_guards.values())),
        independent_holdout=False, confidence_intervals_computed=False,
        default_replacement=False, role='Fixed-checkpoint development screen, not independent improvement evidence')


def holdout_statistics(rows, cfg):
    """One shared whole-seed draw array for both comparisons and subgroups."""
    seeds, pairs = _panel(rows, cfg, 'holdout')
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP['seed'])).integers(0, 8, size=(20000, 8))
    comparisons = _comparisons(pairs, seeds, draws)
    student, teacher = comparisons['baseline'], comparisons['teacher_task']
    s, t = student['overall'], teacher['overall']
    checks = dict(own_profit_ci_positive=s['delta_ci95']['net_profit'][0] > HOLDOUT['own_ci_lower_gt'],
        meaningful_own_profit_gain=_relative_at_least(s, HOLDOUT['minimum_relative_gain']),
        overall_margin_not_lower=s['delta_means']['margin'] >= HOLDOUT['overall_margin_delta_min'],
        overall_score_not_lower=s['delta_means']['score'] >= HOLDOUT['overall_score_delta_min'])
    student_guards = {o: bool(_relative_at_least(v, HOLDOUT['opponent_relative_mean_min'])
                              and v['delta_means']['score'] >= HOLDOUT['opponent_score_delta_min'])
                      for o, v in student['by_opponent'].items()}
    teacher_primary = bool(t['relative_net_profit_defined']
        and t['relative_net_profit_ci95'][0] > HOLDOUT['teacher_relative_ci_lower_gt'])
    teacher_guards = {o: bool(_relative_at_least(v, HOLDOUT['teacher_opponent_relative_mean_min'])
                              and v['delta_means']['score'] >= HOLDOUT['teacher_opponent_score_delta_min'])
                      for o, v in teacher['by_opponent'].items()}
    changed = sum(p['first_request_difference_step'] is not None for p in pairs)
    control = changed >= HOLDOUT['minimum_changed_request_games']
    positive = bool(all(checks.values()) and all(student_guards.values()))
    retention = bool(teacher_primary and all(teacher_guards.values()))
    return dict(schema='task-dagger-once-holdout-statistics-v1', complete=True,
        games=96, pairs=32, seeds=seeds, cluster_unit='complete_seed', clusters=8,
        bootstrap=dict(BOOTSTRAP), gates=dict(HOLDOUT), arms=list(ARMS),
        arm_summaries={a: _arm_summary([p[a] for p in pairs]) for a in ARMS},
        comparisons=comparisons, paired_detail=pairs, student_checks=checks,
        student_opponent_point_guards=student_guards, teacher_retention_primary_passed=teacher_primary,
        teacher_opponent_point_guards=teacher_guards, complete_action_changed_games=changed,
        complete_action_effect_gate_passed=control, research_positive=positive,
        teacher_retention=retention, admission=bool(positive and retention and control),
        default_replacement=False, limitations=[
            'Eight independent seed clusters; 20000 resamples do not add independent games.',
            'Per-opponent point guards are not statistical noninferiority guarantees.',
            'The baseline task student previously failed retention; beating it alone is insufficient.',
            'teacher_task uses rule tasks plus the same crop network, not the original all-rule teacher.',
            'Request-difference evidence is supplied by paired official replay checks, not recomputed here.',
            'Conditional label accuracy, full-action change and competitive gain are distinct claims.'])
