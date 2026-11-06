"""Four unchanged consumed points after a narrow official-container repair."""
BASE_COMMIT = '42e78af6f839fc6ebf694166707c43f3005928ac'
POINTS_COMMIT = '02f8ef2f9691e5451dcf247f8714797a04d6dcc5'
DIAGNOSTIC_COMMIT = 'c024d76c64903405905ebe3f8510b88ed42406a0'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
STRUCT_UTILS_SHA256 = '537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b'
HORIZON, WINDOW = 48, (240,648)
CASES = (('F01','C01','S02','market',0,'exploration'),
         ('F02','C01','S03','round2',1,'exploration'),
         ('F03','C02','S02','market',1,'diagnostic'),
         ('F04','C02','S03','round2',0,'diagnostic'))
GIB,MIB = 1024**3,1024**2
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
        raise ValueError('Official-container reproduction contract differs')


def sources():
    return [dict(role=r,cohort=c,seed_alias=a,opponent=o,side=s,purpose=p,
        policy='baseline',game=f'dev-{a}-{o}-{s}-baseline') for r,c,a,o,s,p in CASES]


def specification():
    return dict(cases=sources(),horizon=HORIZON,selection_window=list(WINDOW),
        scope='same_four_consumed_points_official_struct_compatibility_reproduction',
        points_registration_commit=POINTS_COMMIT,diagnostic_registration_commit=DIAGNOSTIC_COMMIT,
        target_reselection=False,new_source_samples=False,
        preparation_source_authentication='completed_32_receipts_plus_model_runtime_and_default_assets',
        preparation_historical_replay_or_trace_file_reads=0,
        preflight_and_run_source_authentication='32 complete compressed game identities',
        preserved_origin_inventories=[dict(role='runtime_failure',files=18,bytes=187746,official_steps=1),
            dict(role='located_diagnostic',files=18,bytes=246642,official_steps=1)],
        prior_lineage_official_steps=2,new_stage_official_step_limit=384,
        official_utils_sha256=STRUCT_UTILS_SHA256,
        compatibility='exact official Struct comparison copies only; original typed comparator retained',
        compiled_core='unchanged frozen bytes in an isolated namespace; only same_request rebound',
        scalar_types_and_finite_checks_unchanged=True,list_order_checks_unchanged=True,
        state_inventory_and_shed_order_checks_unchanged=True,official_state_and_requests_mutated=False,
        first_factual_step='part of F01 factual 48, no extra official probe',
        original_runtime_preparation_retained=True,
        factual_48_verified_before_intervention=True,one_request_patch_per_intervention=True,
        original_signal_gate_unchanged=True,signal_scope='short-window research only; no formal labels or promotion',
        training=False,gpu=False,new_seeds=False,complete_games=0,independent_evaluation=False,promotion=False)


def project(rows):
    from learning.task_harvest_unexpressed_v1.contract import project as prior_project
    value = prior_project(rows)
    value.update(reproduction_scope='same_four_consumed_points_official_struct_compatibility',
                 prior_lineage_official_steps=2)
    return value
