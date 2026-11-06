"""Technical reproduction of the same four frozen consumed-source points."""
BASE_COMMIT = '7f2deadcb1e3dc65d3073081d64177eee4354440'
PRIOR_COMMIT = '7bdf591cbdb4070be69f7a82d43e8202dadb661c'
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
        raise ValueError('Frozen-point runtime reproduction differs')


def sources():
    return [dict(role=r,cohort=c,seed_alias=a,opponent=o,side=s,purpose=p,
        policy='baseline',game=f'dev-{a}-{o}-{s}-baseline') for r,c,a,o,s,p in CASES]


def specification():
    return dict(cases=sources(), horizon=HORIZON, selection_window=list(WINDOW),
        scope='same_four_consumed_frozen_points_technical_repair_reproduction',
        prior_registration_commit=PRIOR_COMMIT,
        preparation='copy_authenticated_prior_source_and_selection_without_reselection',
        preparation_source_authentication='completed_32_registration_and_receipt_chain_plus_runtime_model_assets',
        preparation_historical_replay_or_trace_file_reads=0,
        preflight_and_run_source_authentication='complete_32_compressed_file_identities',
        prior_failed_inventory_files=18, prior_failed_accepted_cases=0,
        source_scope='completed_32_only', observation_binding_callback=0,
        selection_scope='inherited_initially_available_unexpressed_direct_crop_harvest',
        target_reselection=False, new_source_samples=False,
        runtime_preparation='trusted_runtime_and_target_tracer_before_restricted_event_writes',
        arbitrary_write_exception=False, model_or_environment_construction_during_runtime_preparation=False,
        metadata_binding='inherited unit requests only; factual complete request verified against replay',
        intervention='replace exactly one submitted unit request after normal controller call',
        preserve_other_units=True, preserve_all_market_order_atoms_and_order=True,
        trace_is_original_intent=True, target_execution_trace='same target official step in both arms; no extra transitions',
        horizon_endpoint='state[t + 48]', signal='delta_margin', legacy_signal_field='minimum_label_signal',
        signal_scope='further short-window outcome research only; not formal labels or promotion',
        training=False, formal_label_generation=False, new_seeds=False,
        complete_games=0, independent_evaluation=False, promotion=False)


def project(rows):
    # The snapshot is stdlib-only; execution bootstrap precedes this pure report.
    from learning.task_harvest_unexpressed_v1.contract import project as prior_project
    value = prior_project(rows)
    value['reproduction_scope'] = 'same_four_consumed_frozen_points_technical_repair'
    return value
