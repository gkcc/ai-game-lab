"""One frozen factual step; diagnostic acceptance is not game acceptance."""
BASE_COMMIT = '6d162b2652ae09e0ee739f64815820b741ee2ecf'
PRIOR_COMMIT = '02f8ef2f9691e5451dcf247f8714797a04d6dcc5'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
START_STEP, HORIZON, WINDOW = 240, 1, (240,240)
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=6*GIB,between_available_bytes=6*GIB,
    running_available_bytes=4*GIB,readiness_samples=3,readiness_interval_seconds=10,
    readiness_timeout_seconds=300,max_rss_bytes=2*GIB,max_commit_bytes=3*GIB,
    job_wall_seconds=900,cleanup_seconds=10,poll_seconds=1,total_wall_seconds=2*60*60,
    disk_floor_bytes=4*GIB,per_case_output_bytes=128*MIB,per_case_reserve_bytes=128*MIB,
    total_output_bytes=128*MIB,private_result_bytes=16*MIB,max_cases=1,max_arms=1,
    max_official_transitions=1,max_environment_constructions=1,max_controller_calls=241,
    max_rival_calls=241,max_task_forwards=4000,max_crop_forwards=8000,
    max_neural_input_rows=160000,max_request_patches=0,max_attempts=1)


def require(value):
    if not value:
        raise ValueError('First-step diagnostic contract differs')


def sources():
    return [dict(role='F01',cohort='C01',seed_alias='S02',opponent='market',side=0,
                 purpose='exploration',policy='baseline',game='dev-S02-market-0-baseline')]


def specification():
    return dict(cases=sources(),scope='one_consumed_frozen_F01_first_factual_step_diagnostic',
        prior_registration_commit=PRIOR_COMMIT,target_step=START_STEP,
        target_reselection=False,prepare_trace_rows_read=0,prepare_replay_decodes=0,
        preparation_source_authentication='32 completed receipt chains plus runtime/model/default assets',
        preflight_and_run_source_authentication='32 complete compressed game identities',
        prior_failed_inventory_files=18,prior_failed_inventory_bytes=187746,
        prefix_controller_calls=240,suffix_controller_calls=1,
        original_per_actor_model_limits_unchanged=True,frozen_core_horizon=48,
        diagnostic_horizon=1,second_official_step_forbidden=True,intervention_forbidden=True,
        request_patch_forbidden=True,other_roles_forbidden=True,full_horizon_verified=False,
        accepted_outcomes=['located_original_error','first_step_complete'],
        required_evidence='allowlisted registered source location and bounded context, or verified first suffix controlled stop',
        pre_location_or_resource_failure='seal diagnostic failure without retry',
        original_case_passed=False,accepted_pairs=0,economic_value_evaluated=False,
        training=False,gpu=False,new_seeds=False,complete_games=0,independent_evaluation=False,promotion=False)


def accept_diagnostic(checked):
    require(type(checked) is dict and type(checked.get('document')) is dict
        and checked['document'].get('passed') is False and checked['document'].get('status') == 'failed'
        and type(checked.get('receipt')) is dict and type(checked.get('diagnostic')) is dict
        and checked['diagnostic'].get('passed') is True
        and checked['diagnostic'].get('outcome') in ('located_original_error','first_step_complete'))
    return checked['diagnostic']
