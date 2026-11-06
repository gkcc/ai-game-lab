"""Four consumed-source complete games with the original two frozen models."""
import math


BASE = 'bca1eb4b844cef2b407cbb15dff7e546b5ae1920'
PRIOR_REG_COMMIT = 'c4b60e5aad32406755c7570f4dee576ade16743c'
POLICY = 'harvest_model'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
CORE_SHA = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
SPEC_SHA = 'a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867'
UTILS_SHA = '537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b'
AGENT_SHA = '0ac7baf1212b4ce1eabae3cc1cbb55c98effef5e0d0485522a8be87097c39a4f'
TEACHER_SHA = '0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff'
COHORTS, ROLES = ('C01', 'C02'), ('F01', 'F02', 'F03', 'F04')
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=6*GIB, between_available_bytes=6*GIB,
    running_available_bytes=4*GIB, readiness_samples=3, readiness_interval_seconds=10,
    readiness_timeout_seconds=300, max_rss_bytes=2*GIB, max_commit_bytes=3*GIB,
    job_wall_seconds=900, cleanup_seconds=10, poll_seconds=1,
    total_wall_seconds=2*60*60, disk_floor_bytes=4*GIB,
    per_game_reserve_bytes=160*MIB, per_stage_output_bytes=GIB,
    total_output_bytes=GIB, private_result_bytes=16*MIB,
    max_games=4, max_official_replays=4, max_attempts=1,
    max_environment_constructions=8, max_initializations=12,
    max_live_action_transitions=2876, max_replay_action_transitions=2876,
    max_controller_calls=5760, max_rival_calls=5752,
    max_task_forwards=32264, max_crop_forwards=64512, max_neural_input_rows=1345536)
PRECHECK_LIMITS = dict(controller_calls=8, standalone_scores=8,
    task_forwards=264, crop_forwards=512, neural_input_rows=65536,
    score_rows=4096, atol=1e-10, rtol=1e-10)


def require(value):
    if not value:
        raise ValueError('Frozen harvest model development contract differs')


def sources():
    cases=(('F01','C01','S02','market',0,'exploration'),
           ('F02','C01','S03','round2',1,'exploration'),
           ('F03','C02','S02','market',1,'diagnostic'),
           ('F04','C02','S03','round2',0,'diagnostic'))
    return [dict(role=r,cohort=c,seed_alias=a,opponent=o,side=s,purpose=p,
        policy='baseline',game=f'dev-{a}-{o}-{s}-baseline') for r,c,a,o,s,p in cases]


class HarvestPanel:
    """Only policy name and the two selected plans per old cohort differ."""
    def __init__(self, base_panel, cohort):
        # Delayed import: the isolated registration snapshot must stay stdlib
        # only until the existing trusted runtime has bound project imports.
        from learning.task_joint_decoder_v1.contract import JointPanel
        require(cohort in COHORTS)
        JointPanel.__init__(self,base_panel)
        self._delegate=JointPanel
        self.cohort = cohort

    def __repr__(self):
        return '<FrozenHarvestModelPanel>'

    def plans(self):
        return tuple(dict(game=f"dev-{s['seed_alias']}-{s['opponent']}-{s['side']}-{POLICY}",
            seed_alias=s['seed_alias'], seed=self.base.seeds[('S02','S03').index(s['seed_alias'])],
            opponent=s['opponent'], side=s['side'], policy=POLICY, split='dev', parity_kind=None)
            for s in sources() if s['cohort'] == self.cohort)

    def validate_plan(self,plan,*,job=False):
        return self._delegate.validate_plan(self,plan,job=job)

    def task_model(self,plan):
        return self._delegate.task_model(self,plan)

    def worker_config(self):
        return self.base.worker_config()


def specification():
    return dict(scope='four_consumed_sources_frozen_dual_model_harvest_choice_development',
        cases=sources(), policy=POLICY, games=4, official_replays=4,
        source_completed_games=32, task_model_role='baseline', crop_model='same_frozen_student',
        precheck=PRECHECK_LIMITS, input_streams=4, input_stream_passes_each=1,
        input_stream_selection='same certified first factual suffix callback and complete request',
        historical_baseline_comparison=True, new_concurrent_baseline_games=0,
        full_original_baseline_reproduction=False, full_candidate_actor_reconstruction=True,
        original_candidates_and_shared_features_preserved=True,
        no_forced_new_action_or_economic_mask=True,
        old_teacher_labels_or_window_rewards_are_not_training_targets=True,
        training=False, gpu=False, new_seeds=False, independent_evaluation=False,
        statistical_significance=False, promotion=False, retries=False,
        local_registration_not_third_party_timestamp=True)


def metrics(row):
    result = dict(own=row['net_profit'], opponent=row['opponent_net_profit'],
                  margin=row['margin'], score=row['score'])
    require(all(type(v) in (int,float) and math.isfinite(v) for v in result.values())
        and result['score'] == (1. if result['margin']>0 else .5 if result['margin']==0 else 0.))
    return result


def public_choice(value):
    """Only aggregate counts cross this boundary; effects remain private."""
    require(type(value) is dict and type(value.get('counts')) is dict)
    names=('callbacks','callbacks_with_added_candidates','added_candidates','selected_virtual_harvest')
    counts={k:value['counts'][k] for k in names}
    require(all(type(v) is int and v>=0 for v in counts.values()))
    actual=value['selected_virtual_actual_executions']
    units=value['selected_virtual_harvested_melon_units']
    statuses=value['selected_virtual_execution_status_counts']
    require(type(actual) is int and actual>=0 and type(units) is int and units>=0
        and type(statuses) is dict and set(statuses)<={'executed','blocked','no_effect','partial','pass'}
        and all(type(v) is int and v>=0 for v in statuses.values()))
    return dict(counts=counts,selected_virtual_actual_executions=actual,
        selected_virtual_harvested_melon_units=units,
        selected_virtual_execution_status_counts=dict(statuses))


def projection(rows):
    """All four descriptive rows; a historical comparison is not a holdout."""
    require(type(rows) is list and len(rows)==4)
    output=[]
    for actual, source in zip(rows,sources()):
        require(all(actual[k]==source[k] for k in ('role','cohort','opponent','side')))
        old,new = metrics(actual['baseline']),metrics(actual['candidate'])
        output.append(dict(role=source['role'],cohort=source['cohort'],opponent=source['opponent'],
            side=source['side'],historical_baseline=old,candidate=new,
            delta={k:new[k]-old[k] for k in old},harvest_choice=public_choice(actual['harvest_choice'])))
    return dict(schema='task-harvest-model-dev-public-summary-v1',cases=output,games=4,
        official_replays=4,case_mean_delta={k:math.fsum(r['delta'][k] for r in output)/4
            for k in ('own','opponent','margin','score')},
        comparison='historical_consumed_development_not_concurrent_paired_or_holdout',
        training=False,gpu=False,independent_evaluation=False,statistical_significance=False,
        promotion=False)
