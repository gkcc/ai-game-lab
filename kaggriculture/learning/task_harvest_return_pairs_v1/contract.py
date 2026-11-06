"""Stdlib-only registration and public projection for four consumed cases."""
import math


BASE = '317914328a81d5d26581791481140d9eea6438ea'
PRIOR_REG_COMMIT = '5f5d6bb8e37df9210d1f623ceb5c0c48f71ce2a6'
PRIOR_COMMITMENT = '6e4bd7a858454dd7fd673a77a9074bd1efbcf2eb81818c86e396653485335b6c'
PRIOR_FILES, PRIOR_BYTES = 145, 318604494
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
CORE_SHA = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
SPEC_SHA = 'a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867'
UTILS_SHA = '537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b'
AGENT_SHA = '0ac7baf1212b4ce1eabae3cc1cbb55c98effef5e0d0485522a8be87097c39a4f'
TEACHER_SHA = '0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff'
ROLES, ARMS = ('F01','F02','F03','F04'), ('H','W')
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=6*GIB, between_available_bytes=6*GIB,
    running_available_bytes=4*GIB, readiness_samples=3, readiness_interval_seconds=10,
    readiness_timeout_seconds=300, max_rss_bytes=2*GIB, max_commit_bytes=3*GIB,
    job_wall_seconds=900, cleanup_seconds=10, poll_seconds=1,
    total_wall_seconds=2*60*60, preparation_wall_seconds=900,
    disk_floor_bytes=4*GIB, total_output_bytes=GIB,
    per_case_output_bytes=192*MIB, per_case_decoded_bytes=GIB,
    private_result_bytes=16*MIB, closure_reserve_bytes=MIB, max_receipt_bytes=8*MIB,
    max_pairs=4, max_arms=8, max_attempts=1, max_environment_constructions=8,
    max_restores=8, max_explicit_resets=0, max_official_action_transitions=5752,
    max_controller_calls=5752, max_rival_calls=5752,
    max_task_forwards=32000, max_crop_forwards=64000, max_neural_input_rows=1280000,
    previous_inventory_files=PRIOR_FILES, previous_inventory_bytes=PRIOR_BYTES,
    preparation_source_trace_scans=4, execution_source_trace_scans=8,
    execution_replay_decodes=4, verification_replay_decodes=4,
    reconstructed_replays=8, verification_reconstructed_replay_decodes=8,
    reconstructed_replay_decoded_per_file=128*MIB,
    source_trace_decoded_per_file=512*MIB, source_replay_decoded_per_file=128*MIB)


def require(value, message='Frozen harvest return-pair contract differs'):
    if not value:
        raise ValueError(message)


def sources():
    cases=(('F01','C01','S02','market',0),('F02','C01','S03','round2',1),
           ('F03','C02','S02','market',1),('F04','C02','S03','round2',0))
    return [dict(role=r,cohort=c,seed_alias=a,opponent=o,side=s,
                 game=f'dev-{a}-{o}-{s}-harvest_model') for r,c,a,o,s in cases]


def specification():
    return dict(scope='four_consumed_student_states_single_selector_terminal_return_pair_pilot',
        cases=sources(), source_games=4, max_pairs=4,
        selection='earliest (step, ordinal, unit, y, x) actual virtual HARVEST with same-site WATER',
        missing_target='retain missing case; never replace; no successful four-pair claim',
        arms=list(ARMS), factual_arm_must_finish_before_counterfactual=True,
        override='one bound selector return only; no full-request patch',
        original_network_proposal_and_actual_choice_are_separate=True,
        continuation='same frozen expanded task model and same frozen crop model',
        target_after_callback_scheduling_and_market_changes_allowed=True,
        terminal_observation_step=719, prefix_is_model_reconstruction_not_official_execution=True,
        restore_path='make with authenticated nonempty steps; __set_state, not reset',
        primary_target='terminal margin(H) minus margin(W)',
        own_return_target_also_preserved=True, conflicting_objectives_not_collapsed=True,
        future_private_seed_and_opponent_identity_are_not_model_features=True,
        new_complete_from_zero_games=0, extra_official_replays=0,
        reconstructed_replays='source prefix plus actual official suffix; not new from-zero games',
        full_factual_suffix_matches_source_game=True,
        preparation_model_forwards=0, preparation_official_actions=0,
        training=False,gpu_training=False,new_seeds=False,independent_holdout=False,
        statistical_significance=False,promotion=False,retries=False,
        local_registration_not_third_party_timestamp=True)


def terminal_delta(harvest, water):
    names=('own_net_profit','opponent_net_profit','margin','score')
    require(all(type(x) is dict and set(x)==set(names) for x in (harvest,water)))
    require(all(type(v) in (int,float) and math.isfinite(v) for x in (harvest,water) for v in x.values()))
    for x in (harvest,water):
        require(x['score']==(1. if x['margin']>0 else .5 if x['margin']==0 else 0.))
    delta={k:harvest[k]-water[k] for k in names}
    require(delta['margin']==delta['own_net_profit']-delta['opponent_net_profit'])
    sign=lambda x:1 if x>0 else -1 if x<0 else 0
    return dict(H=dict(harvest),W=dict(water),H_minus_W=delta,
                objective_conflict=sign(delta['margin'])*sign(delta['own_net_profit'])<0,
                margin_sign=sign(delta['margin']),own_profit_sign=sign(delta['own_net_profit']),
                interpretation='fixed continuation policy and opponent; pilot targets, not optimal action values')


def projection(rows):
    """Only cash targets and aggregate execution counts cross this boundary."""
    require(type(rows) is list and len(rows)==4)
    output=[]
    for row,source in zip(rows,sources()):
        require(all(type(row.get(k)) is type(source[k]) and row[k]==source[k]
                    for k in ('role','cohort','opponent','side')))
        item={k:source[k] for k in ('role','cohort','opponent','side')}
        if row['status']=='no_eligible_target':
            item.update(status=row['status'],paired=False,returns=None)
        else:
            require(row['status']=='paired' and row['technical_passed'] is True)
            item.update(status='paired',paired=True,returns=terminal_delta(row['H'],row['W']))
        output.append(item)
    return dict(schema='task-harvest-return-pairs-public-summary-v1',cases=output,
        completed_pairs=sum(x['paired'] for x in output),
        four_pair_data_chain_passed=all(x['paired'] for x in output),
        training=False,gpu_training=False,new_seeds=False,independent_holdout=False,
        promotion=False,statistical_significance=False,
        scope='policy-conditioned single-selector terminal-return data pilot; not a trained candidate evaluation')
