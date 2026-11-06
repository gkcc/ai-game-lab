"""Four consumed-source request interventions; no policy or training change."""
BASE_COMMIT = '967dbe9691b40c1b3f39eb5bf30b3727496f7e10'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
HORIZON, WINDOW = 48, (240,648)
CASES = (('F01','C01','S02','market',0,'exploration'),
         ('F02','C01','S03','round2',1,'exploration'),
         ('F03','C02','S02','market',1,'diagnostic'),
         ('F04','C02','S03','round2',0,'diagnostic'))
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=6*GIB,between_available_bytes=6*GIB,
    running_available_bytes=4*GIB,readiness_samples=3,readiness_interval_seconds=10,
    readiness_timeout_seconds=300,max_rss_bytes=2*GIB,max_commit_bytes=3*GIB,
    job_wall_seconds=900,cleanup_seconds=10,poll_seconds=1,total_wall_seconds=2*60*60,
    disk_floor_bytes=4*GIB,per_case_output_bytes=128*MIB,per_case_reserve_bytes=128*MIB,
    total_output_bytes=512*MIB,private_result_bytes=16*MIB,max_cases=4,max_arms=8,
    max_official_transitions=384,max_environment_constructions=8,max_controller_calls=5752,
    max_rival_calls=5752,max_task_forwards=32000,max_crop_forwards=64000,
    max_neural_input_rows=1280000,max_attempts=1)


def require(value):
    if not value:
        raise ValueError('Unexpressed harvest contract differs')


def sources():
    return [dict(role=r,cohort=c,seed_alias=a,opponent=o,side=s,purpose=p,
        policy='baseline',game=f'dev-{a}-{o}-{s}-baseline') for r,c,a,o,s,p in CASES]


def specification():
    return dict(cases=sources(), horizon=HORIZON, selection_window=list(WINDOW),
        selection_order=['step', 'unit', 'target_y', 'target_x'],
        source_scope='completed_32_only', observation_binding_callback=0,
        selection_scope='initially_available_unexpressed_direct_crop_harvest',
        unit_initially_available=True, own_harvest_absent_from_all_turn_candidates=True,
        site_has_no_final_harvest_request=True, target_is_current_unit_position=True,
        operation='HARVEST', tile_kind='PLANT', positive_yield=True, officially_mature=True,
        backpack_capacity_gate=False, no_hit='record_missing_without_replacement',
        metadata_binding='unit requests only; market absent from old trace',
        factual_binding='complete live original request and stable intent trace match historical replay/trace',
        intervention='replace exactly one submitted unit request after normal controller call',
        preserve_other_units=True, preserve_all_market_order_atoms_and_order=True,
        trace_is_original_intent=True, target_execution_trace='same target official step in both arms; no extra transitions',
        horizon_endpoint='state[t + 48]', signal='delta_margin',
        legacy_signal_field='minimum_label_signal',
        signal_scope='further research on positive and negative short-window outcomes only; not formal labels or promotion',
        training=False, formal_label_generation=False, new_seeds=False,
        complete_games=0, independent_evaluation=False, promotion=False)


def project(rows):
    """Keep the earlier fixed cash gate, with its restricted research meaning."""
    # Called after the execution bootstrap; snapshot/import itself is stdlib-only.
    from learning.task_harvest_window_v1.contract import project as prior_project
    value = prior_project(rows)
    value.update(signal_scope='further_short_window_outcome_research_only',
                 formal_label_generation=False)
    return value
