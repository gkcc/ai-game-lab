"""Four consumed-source harvest probes, not an independent test or training run.

Selection is fixed before looking at counterfactual cash. No private files,
models or official environment are read or constructed by this module.
"""
import math


BASE_COMMIT = 'b58fda94abc06b377ca341bf4d17fead2eb6dff2'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
HORIZON = 48
WINDOW = (240, 648)
# Each role is a different already-consumed seed; seat is not a new sample.
CASES = (
    ('F01', 'C01', 'S02', 'market', 0, 'exploration'),
    ('F02', 'C01', 'S03', 'round2', 1, 'exploration'),
    ('F03', 'C02', 'S02', 'market', 1, 'diagnostic'),
    ('F04', 'C02', 'S03', 'round2', 0, 'diagnostic'),
)
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=6*GIB, between_available_bytes=6*GIB,
    running_available_bytes=4*GIB, readiness_samples=3, readiness_interval_seconds=10,
    readiness_timeout_seconds=300, max_rss_bytes=2*GIB, max_commit_bytes=3*GIB,
    job_wall_seconds=900, cleanup_seconds=10, poll_seconds=1,
    total_wall_seconds=2*60*60, disk_floor_bytes=4*GIB,
    per_case_output_bytes=128*MIB, per_case_reserve_bytes=128*MIB,
    total_output_bytes=512*MIB, private_result_bytes=16*MIB,
    max_cases=4, max_arms=8, max_official_transitions=384,
    max_environment_constructions=8, max_controller_calls=5752, max_rival_calls=5752,
    max_task_forwards=32000, max_crop_forwards=64000, max_neural_input_rows=1280000,
    max_attempts=1)


def require(value):
    if not value:
        raise ValueError('Harvest window development contract differs')


def sources():
    return [dict(role=r, cohort=c, seed_alias=a, opponent=o, side=s, purpose=p,
                 policy='baseline', game=f'dev-{a}-{o}-{s}-baseline')
            for r, c, a, o, s, p in CASES]


def specification():
    return dict(cases=sources(), horizon=HORIZON, selection_window=list(WINDOW),
        selection_order=['step', 'record_ordinal', 'unit', 'target_y', 'target_x', 'op', 'id'],
        source_scope='completed_32_only', selection_scope='existing_teacher_economic_candidates',
        same_unit=True, target_is_current_unit_position=True, operation='HARVEST',
        tile_kind='PLANT', positive_yield=True, officially_mature=True,
        original_operation_may_be_harvest=True, backpack_capacity_gate=False,
        no_hit='record_missing_without_replacement', horizon_endpoint='state[t + 48]',
        signal='delta_margin', training=False, new_seeds=False,
        complete_games=0, independent_evaluation=False, promotion=False)


def _cash(arm):
    require(type(arm) is dict and set(arm) == {'start', 'end'})
    for key in ('start', 'end'):
        require(type(arm[key]) is dict and set(arm[key]) == {'own', 'opponent'}
            and all(type(x) in (int, float) and math.isfinite(x) for x in arm[key].values()))
    return {key: arm['end'][key]-arm['start'][key] for key in ('own', 'opponent')}


def project(rows):
    """Descriptive 48-step cash signals; missing is never a zero outcome.

    All attempted pairs must have passed the caller's actual technical checks.
    A missing opportunity is a coverage fact, not an engine failure or a trial.
    The returned gate only concerns studying more labels, never model promotion.
    """
    require(type(rows) is list and len(rows) == len(CASES))
    pairs, missing = [], []
    for row, source in zip(rows, sources()):
        require(type(row) is dict and all(type(row.get(k)) is type(source[k])
                and row[k] == source[k] for k in ('role', 'opponent', 'side', 'purpose')))
        status = row.get('status')
        require(status in ('no_eligible_target', 'paired'))
        if status == 'no_eligible_target':
            require(row.get('target') is None and not any(k in row for k in
                ('baseline', 'intervention', 'request_changed', 'physical_changed', 'delta')))
            missing.append(row['role'])
            continue
        require(row.get('technical_passed') is True
            and type(row.get('request_changed')) is bool and type(row.get('physical_changed')) is bool
            and (not row['physical_changed'] or row['request_changed']))
        baseline, intervention = _cash(row['baseline']), _cash(row['intervention'])
        require(row['baseline']['start'] == row['intervention']['start'])
        delta = {k:intervention[k]-baseline[k] for k in baseline}
        delta['margin'] = delta['own']-delta['opponent']
        require(all(math.isfinite(x) for x in delta.values()))
        require(row['physical_changed'] or all(x == 0 for x in delta.values()))
        pairs.append(dict(role=row['role'], opponent=row['opponent'], side=row['side'],
            purpose=row['purpose'], baseline_window_cash=baseline,
            intervention_window_cash=intervention, delta=delta,
            request_changed=row['request_changed'], physical_changed=row['physical_changed']))
    effective = [r for r in pairs if r['request_changed'] and r['physical_changed']]
    nonzero = [r for r in effective if r['delta']['margin'] != 0]
    positive = [r for r in nonzero if r['delta']['margin'] > 0]
    negative = [r for r in nonzero if r['delta']['margin'] < 0]
    signal = (len(effective) >= 3 and len(nonzero) >= 2
        and any(r['purpose'] == 'diagnostic' for r in nonzero)
        and bool(positive) and bool(negative)
        and any(r['delta']['own'] >= 0 for r in positive))
    return dict(registered_roles=4, paired_roles=len(pairs), missing_roles=missing,
        pairs=pairs, effective_pairs=len(effective), nonzero_margin_roles=len(nonzero),
        positive_zero_negative={k:dict(positive=sum(r['delta'][k]>0 for r in pairs),
            zero=sum(r['delta'][k]==0 for r in pairs),negative=sum(r['delta'][k]<0 for r in pairs))
            for k in ('own', 'opponent', 'margin')},
        paired_only_mean_delta=({k:math.fsum(r['delta'][k] for r in pairs)/len(pairs)
            for k in ('own', 'opponent', 'margin')} if pairs else None),
        minimum_label_signal=signal, cash_horizon_steps=HORIZON,
        final_profit_evaluated=False, terminal_score_evaluated=False,
        independent_evaluation=False, statistical_significance=False,
        training=False, promotion=False)
