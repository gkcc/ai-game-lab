"""Stdlib-only preregistration; no policy or official runtime imports."""
import math

BASE='8b7949b4ca29e3c2492cc01216d8391ecc7a2a1d'
PAIR_REG_COMMIT='f3a52cd96f23b514625767dfbfd9d27175f870ae'
PAIR_COMMITMENT='2cda99e072d723d24ab6e70a9270e709fcbb14b4bfd6c0faa5b7e25cffceba36'
PAIR_FILES,PAIR_BYTES=65,82682533
ENGINE_COMMIT='f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA='bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
CORE_SHA='0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
SPEC_SHA='a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867'
UTILS_SHA='537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b'
AGENT_SHA='0ac7baf1212b4ce1eabae3cc1cbb55c98effef5e0d0485522a8be87097c39a4f'
TEACHER_SHA='0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff'
ORIGINS=('F01','F02','F03','F04')
POINTS=('Q1','Q2','Q3')
ROLES=tuple(o+'-'+q for o in ORIGINS for q in POINTS)
ARMS=('H','W')
GIB,MIB=1024**3,1024**2
LIMITS=dict(initial_available_bytes=6*GIB,between_available_bytes=6*GIB,
    running_available_bytes=4*GIB,readiness_samples=3,readiness_interval_seconds=10,
    readiness_timeout_seconds=300,max_rss_bytes=2*GIB,max_commit_bytes=3*GIB,
    job_wall_seconds=900,cleanup_seconds=10,poll_seconds=1,total_wall_seconds=7200,
    preparation_wall_seconds=900,disk_floor_bytes=4*GIB,total_output_bytes=GIB,
    per_case_output_bytes=192*MIB,private_result_bytes=16*MIB,closure_reserve_bytes=MIB,
    max_receipt_bytes=8*MIB,max_pairs=12,max_arms=24,max_attempts=1,
    max_environment_constructions=24,max_restores=24,max_explicit_resets=0,
    max_official_action_transitions=11496,max_controller_calls=17256,max_rival_calls=17256,
    max_task_forwards=96000,max_crop_forwards=192000,max_neural_input_rows=3840000,
    max_analysis_samples=16,max_analysis_feature_rows=32,normalization_loads=1,
    preparation_source_trace_scans=4,execution_source_trace_scans=24,
    execution_replay_decodes=12,verification_replay_decodes=12,
    reconstructed_replays=24,verification_reconstructed_replay_decodes=24,
    old_reference_pair_executions=0,old_reference_sample_reads=4,
    source_trace_decoded_per_file=512*MIB,source_replay_decoded_per_file=128*MIB)


def require(value,message='Frozen quantile collection contract differs'):
    if not value:raise ValueError(message)


def split(origin):
    require(origin in ORIGINS)
    return 'exploration' if origin in ORIGINS[:2] else 'diagnostic'


def sources():
    values=(('F01','C01','S02','market',0),('F02','C01','S03','round2',1),
            ('F03','C02','S02','market',1),('F04','C02','S03','round2',0))
    return [dict(role=o,cohort=c,seed_alias=a,opponent=p,side=s,
        game=f'dev-{a}-{p}-{s}-harvest_model') for o,c,a,p,s in values]


def slots():
    return [dict(origin=s['role'],point=q,role=s['role']+'-'+q,split=split(s['role']),
                 cohort=s['cohort'],opponent=s['opponent'],side=s['side']) for s in sources() for q in POINTS]


def specification():
    return dict(scope='consumed_development_quantile_terminal_return_collection',sources=sources(),slots=slots(),
        quantile_indices='0, floor((N-1)/3), floor(2*(N-1)/3), N-1; N counts fully validated eligible callbacks',
        order=['step','ordinal','unit','y','x'],reference_index_zero='authenticate and reuse four sealed samples; never execute again',
        duplicate_or_missing='retain empty slot; no replacement',identity_mismatch='fail closed',
        independent_single_choice_branches=True,interventions_not_chained_or_returns_added=True,
        continuation='same frozen task and crop weights',primary_target='terminal margin(H) minus margin(W)',
        preserve_own_and_opponent_profit_score_and_objective_conflicts=True,
        group_by_actual_seed=True,exploration_origins=list(ORIGINS[:2]),diagnostic_origins=list(ORIGINS[2:]),
        grouping_is_not_blind_validation=True,old_results_already_seen=True,
        reuse='unchanged return_pairs core, policy, worker and official restoration',
        full_factual_reproduction_before_W=True,all_arms_reach_step_719_and_DONE=True,
        replay_exports='copied authenticated prefix plus actual official suffix; no from-zero replay claim',
        analysis='whole H and W 373-dimensional raw and normalized vectors; never just their difference',
        feature_rows_counted_separately_and_inside_total_input_row_cap=True,
        normalization='one identity-bound task NPZ mean/scale read in run only; no TaskNetwork construction or forward',
        duplicate_input_return_conflicts='report and retain; never delete or move across groups',
        preparation_model_forwards=0,preparation_official_actions=0,preparation_normalization_loads=0,
        training=False,gpu_training=False,training_admission=False,new_seeds=False,
        independent_holdout=False,statistical_significance=False,promotion=False,retries=False,
        local_registration_not_third_party_timestamp=True)


def returns(h,w):
    names=('own_net_profit','opponent_net_profit','margin','score')
    require(all(type(x) is dict and set(x)==set(names) for x in (h,w)))
    require(all(type(v) in (int,float) and math.isfinite(v) for x in (h,w) for v in x.values()))
    for x in (h,w):require(x['score']==(1. if x['margin']>0 else .5 if x['margin']==0 else 0.))
    delta={k:h[k]-w[k] for k in names}
    require(delta['margin']==delta['own_net_profit']-delta['opponent_net_profit'])
    sign=lambda x:1 if x>0 else -1 if x<0 else 0
    return dict(H=dict(h),W=dict(w),H_minus_W=delta,
        objective_conflict=sign(delta['own_net_profit'])*sign(delta['margin'])<0,
        margin_sign=sign(delta['margin']),own_profit_sign=sign(delta['own_net_profit']))


def projection(rows):
    """Sixteen fixed slots, including explicitly reused first points."""
    require(type(rows) is list and len(rows)==16)
    expected=[(source,q) for source in sources() for q in ('Q0',*POINTS)]
    output=[]
    for row,(source,point) in zip(rows,expected):
        origin=source['role']
        require(row.get('origin')==origin and row.get('point')==point and row.get('split')==split(origin))
        item=dict(origin=origin,point=point,split=split(origin),cohort=source['cohort'],
            opponent=source['opponent'],side=source['side'],reused=point=='Q0')
        if row['status']=='no_eligible_target':
            require(point!='Q0' and row.get('reason') in ('duplicate_index','no_eligible_target'))
            item.update(status=row['status'],reason=row['reason'],returns=None)
        else:
            require(row['status']=='paired' and row['technical_passed'] is True)
            item.update(status='paired',reason=None,returns=returns(row['H'],row['W']))
        output.append(item)
    new=sum(x['status']=='paired' and not x['reused'] for x in output)
    return dict(schema='task-harvest-quantile-pairs-public-ledger-v1',rows=output,
        reused_pairs=4,completed_new_pairs=new,all_twelve_new_pairs_completed=new==12,
        training=False,gpu_training=False,training_admission=False,new_seeds=False,
        independent_holdout=False,statistical_significance=False,promotion=False,
        scope='four consumed seed groups, at most sixteen policy-conditioned points, not sixteen independent samples')
