"""Independent completed-panel accounting and paired statistics; no model calls."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parent.parent
sys.path.insert(0,str(ROOT))
from learning.task_anchor_once_v1.storage import owned_writes

METRICS=('net_profit','opponent_net_profit','margin','score')
ARMS=('baseline','candidate','teacher_task')
REFS=('baseline','teacher_task')
used=0


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while data:=stream.read(1024*1024):h.update(data)
    return h.hexdigest()


def save(name,value):
    global used
    budget=owned_writes(HERE,8*1024**2-used,allowed_paths=[name])
    try:
        with budget:
            with (HERE/name).open('x',encoding='utf-8',newline='\n') as stream:
                json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    finally:used+=budget.used_bytes


def deadline():
    assert now()<datetime.fromisoformat(read(HERE/'numeric-reservation.json')['deadline_utc']), 'Independent review deadline'


def equal(a,b,path='value'):
    import numpy as np
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
        for k in a:equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,(list,tuple)) and len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    elif isinstance(a,(float,np.floating)):
        assert isinstance(b,(int,float)) and not isinstance(b,bool) and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9),(path,a,b)
    else:assert a==b,(path,a,b)


def prepare():
    deadline()
    assert not (HERE/'final-review-method.json').exists()
    cfg=read(OUT/'registration.json')
    save('final-review-method.json',dict(schema='independent-anchor-final-review-method-v1',created_utc=now().isoformat(),
        reviewer_source_sha256=sha(__file__),registration_sha256=sha(OUT/'registration.json'),freeze_sha256=sha(OUT/'freeze.json'),
        deadline_utc=read(HERE/'numeric-reservation.json')['deadline_utc'],output_limit_bytes=268435456,
        final_output_partition_bytes=8*1024**2,expected_games=120,expected_dev=24,expected_holdout=96,
        bootstrap=cfg['bootstrap'],common_draws_for_references_and_subgroups=True,
        paired_array_order='registered seed, registered opponent, seat, baseline/candidate/teacher_task, metric',
        complete_job_identity_checks=True,recorded_first_action_difference_checks=True,
        additional_neural_rows=0,additional_GPU_operations=0,additional_official_reexecution_transitions=0,
        original_deadline_not_extended=True,requires_closed_campaign_summary=True,
        comparison_atol=1e-9,comparison_rtol=1e-12,gate_thresholds_unchanged=True))
    print(json.dumps(dict(prepared=True,method_sha256=sha(HERE/'final-review-method.json'),source_sha256=sha(__file__))))


def statistics(rows,cfg,split):
    import numpy as np
    seeds=cfg['validation_seeds' if split=='dev' else 'test_seeds'];opponents=cfg['opponents']
    indexed={(r['seed'],r['opponent'],r['side'],r['policy']):r for r in rows}
    expected={(s,o,t,a) for s in seeds for o in opponents for t in (0,1) for a in ARMS}
    assert len(rows)==len(indexed)==len(expected) and set(indexed)==expected
    values=np.empty((len(seeds),2,2,3,4),np.float64)
    pairs=[]
    for i,s in enumerate(seeds):
        for j,o in enumerate(opponents):
            for t in (0,1):
                cell={a:indexed[s,o,t,a] for a in ARMS}
                assert len({r['initial_bank'] for r in cell.values()})==1
                assert len({r['opponent_initial_bank'] for r in cell.values()})==1
                for k,a in enumerate(ARMS):values[i,j,t,k]=[cell[a][m] for m in METRICS]
                pairs.append(dict(seed=s,opponent=o,side=t,first_request_difference_step=cell['candidate']['first_request_difference_step']))
    assert np.isfinite(values).all()
    draws=None
    if split=='holdout':
        assert cfg['bootstrap']==dict(replicates=20000,seed=2026102706,quantile_method='linear')
        draws=np.random.Generator(np.random.PCG64(2026102706)).integers(0,8,size=(20000,8))

    def arm(v,k):
        flattened=v[...,k,:].reshape(-1,4);score=flattened[:,3]
        return dict(games=len(flattened),wins=int((score==1).sum()),draws=int((score==.5).sum()),losses=int((score==0).sum()),
                    means={m:float(x) for m,x in zip(METRICS,flattened.mean(axis=0))})

    def summary(v,ref,sampled):
        k=ARMS.index(ref);delta=v[...,1,:]-v[...,k,:]
        dm=delta.reshape(-1,4).mean(axis=0);den=float(v[...,k,0].mean())
        result=dict(pairs=int(delta.size//4),reference=ref,arms={ref:arm(v,k),'candidate':arm(v,1)},
            delta_means={m:float(x) for m,x in zip(METRICS,dm)},reference_net_profit_mean=den,
            relative_net_profit_defined=den>0,relative_net_profit_mean=float(dm[0]/den) if den>0 else None)
        if sampled is not None:
            seed_delta=delta.mean(axis=(1,2));seed_ref=v[...,k,0].mean(axis=(1,2))
            bootstrap_delta=seed_delta[sampled].mean(axis=1);bootstrap_ref=seed_ref[sampled].mean(axis=1)
            defined=bool(den>0 and (bootstrap_ref>0).all())
            result.update(delta_ci95={m:np.quantile(bootstrap_delta[:,i],[.025,.975],method='linear').tolist() for i,m in enumerate(METRICS)},
                relative_net_profit_defined=defined,nonpositive_bootstrap_reference_count=int((bootstrap_ref<=0).sum()),
                bootstrap_reference_min=float(bootstrap_ref.min()),
                relative_net_profit_ci95=np.quantile(bootstrap_delta[:,0]/bootstrap_ref,[.025,.975],method='linear').tolist() if defined else None)
            if not defined:result['relative_net_profit_mean']=None
        if not result['relative_net_profit_defined']:result['relative_undefined_reason']='Nonpositive point or resampled reference own-net-profit denominator'
        return result

    comparisons={ref:dict(overall=summary(values,ref,draws),
        by_opponent={o:summary(values[:,j:j+1],ref,draws) for j,o in enumerate(opponents)},
        by_seed=[dict(seed=s,**summary(values[i:i+1],ref,None)) for i,s in enumerate(seeds)]) for ref in REFS}
    s,t=comparisons['baseline'],comparisons['teacher_task'];so,to=s['overall'],t['overall']
    rel=lambda v,x:bool(v['relative_net_profit_defined'] and v['relative_net_profit_mean']>=x)
    if split=='dev':
        gate=cfg['development'];checks=dict(own_mean_improved=so['delta_means']['net_profit']>gate['own_mean_delta_gt'],
            overall_margin_not_lower=so['delta_means']['margin']>=gate['overall_margin_delta_min'],
            overall_score_not_lower=so['delta_means']['score']>=gate['overall_score_delta_min'],
            student_ratios_defined=so['relative_net_profit_defined'],teacher_mean_retained=rel(to,gate['teacher_relative_mean_min']))
        sg={o:bool(rel(v,gate['opponent_relative_mean_min']) and v['delta_means']['score']>=gate['opponent_score_delta_min']) for o,v in s['by_opponent'].items()}
        tg={o:bool(v['relative_net_profit_defined'] and v['delta_means']['score']>=gate['teacher_opponent_score_delta_min']) for o,v in t['by_opponent'].items()}
        verdict=dict(checks=checks,student_opponent_point_guards=sg,teacher_opponent_point_guards=tg,
                     passed=bool(all(checks.values()) and all(sg.values()) and all(tg.values())))
    else:
        gate=cfg['holdout'];checks=dict(own_profit_ci_positive=so['delta_ci95']['net_profit'][0]>gate['own_ci_lower_gt'],
            meaningful_own_profit_gain=rel(so,gate['minimum_relative_gain']),
            overall_margin_not_lower=so['delta_means']['margin']>=gate['overall_margin_delta_min'],
            overall_score_not_lower=so['delta_means']['score']>=gate['overall_score_delta_min'])
        sg={o:bool(rel(v,gate['opponent_relative_mean_min']) and v['delta_means']['score']>=gate['opponent_score_delta_min']) for o,v in s['by_opponent'].items()}
        primary=bool(to['relative_net_profit_defined'] and to['relative_net_profit_ci95'][0]>gate['teacher_relative_ci_lower_gt'])
        tg={o:bool(rel(v,gate['teacher_opponent_relative_mean_min']) and v['delta_means']['score']>=gate['teacher_opponent_score_delta_min']) for o,v in t['by_opponent'].items()}
        changed=sum(p['first_request_difference_step'] is not None for p in pairs)
        control=changed>=gate['minimum_changed_request_games'];positive=bool(all(checks.values()) and all(sg.values()));retained=bool(primary and all(tg.values()))
        verdict=dict(student_checks=checks,student_opponent_point_guards=sg,teacher_retention_primary_passed=primary,
            teacher_opponent_point_guards=tg,complete_action_changed_games=changed,complete_action_effect_gate_passed=control,
            research_positive=positive,teacher_retention=retained,admission=bool(positive and retained and control))
    return dict(comparisons=comparisons,verdict=verdict,arm_summaries={a:arm(values,i) for i,a in enumerate(ARMS)},
        seed_order=seeds,pairs=len(pairs),bootstrap_draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest() if draws is not None else None)


def typed(value):
    if isinstance(value,dict):return ('dict',tuple((k,typed(value[k])) for k in sorted(value)))
    if isinstance(value,list):return ('list',tuple(map(typed,value)))
    assert value is None or type(value) in (bool,int,float,str)
    if type(value) is float:assert math.isfinite(value)
    return (type(value).__name__,value)


def verify_control(rows):
    # Stored-data comparison only: no actor, network, make(), or official.step().
    from decision.semantics import canonical
    indexed={(r['split'],r['seed'],r['opponent'],r['side'],r['policy']):r for r in rows}
    result=[]
    for candidate in [r for r in rows if r['policy']=='candidate']:
        deadline();side=candidate['side']
        baseline=indexed[candidate['split'],candidate['seed'],candidate['opponent'],side,'baseline']
        states=[]
        for row in (candidate,baseline):
            with gzip.open(OUT/'games'/row['game']/'replay.json.gz','rt',encoding='utf-8') as stream:
                replay=json.load(stream)
            assert replay['info']['seed']==row['seed'] and len(replay['steps'])==720
            assert [f['money'] for f in replay['steps'][-1][0]['observation']['farms']]==row['banks']
            states.append(replay['steps'])
        a,b=states;first=None
        for step in range(719):
            assert canonical(a[step])==canonical(b[step]),'Unexplained stored state divergence'
            assert typed(a[step+1][1-side]['action'])==typed(b[step+1][1-side]['action'])
            if typed(a[step+1][side]['action'])!=typed(b[step+1][side]['action']):first=step;break
        if first is None:assert canonical(a[-1])==canonical(b[-1])
        assert first==candidate['first_request_difference_step']
        result.append(dict(game=candidate['game'],first_request_difference_step=first))
    return result


def run():
    import numpy as np
    started=time.monotonic();deadline();method=read(HERE/'final-review-method.json')
    assert sha(__file__)==method['reviewer_source_sha256']
    assert not (HERE/'final-review-started.json').exists()
    required=['run-summary.json','campaign-storage.json','ledger.json','audit.json','development-gate.json','holdout-statistics.json']
    assert all((OUT/n).is_file() for n in required),'Closed complete panel required'
    cfg=read(OUT/'registration.json');f=read(OUT/'freeze.json');summary=read(OUT/'run-summary.json')
    assert summary['status']=='completed_holdout' and summary['completed_games']==summary['attempted_games']==120
    assert summary['all_attempts_completed'] and not (OUT/'failure.json').exists()
    input_hashes={n:sha(OUT/n) for n in required}
    assert sha(OUT/'registration.json')==method['registration_sha256']==f['registration_sha256']==summary['registration_sha256']
    assert sha(OUT/'freeze.json')==method['freeze_sha256']==summary['freeze_sha256']
    assert np.__version__==cfg['runtime_versions']['numpy']
    save('final-review-started.json',dict(created_utc=now().isoformat(),method_sha256=sha(HERE/'final-review-method.json'),
        input_sha256=input_hashes,neural_rows_reserved=0,official_transitions_reserved=0))
    try:
        commit=read(OUT/'pre-execution-commit.json')['commit'];public=ROOT.parent/'publication/ai-game-lab'
        for name in ('registration.json','freeze.json','pre-review.json'):
            assert subprocess.check_output(['git','show',commit+':review/evidence/'+OUT.name+'/'+name],cwd=public)==(OUT/name).read_bytes()
        checked_sources={p:v for p,v in f['runtime_files'].items() if p.startswith('learning/task_anchor_reentry_v1/')}
        for p in ('learning/task_anchor_once_v1/training.py','learning/task_anchor_spawn_repair_v1/launcher.py',
                  'learning/task_anchor_once_v1/games.py','learning/task_anchor_once_v1/statistics.py',
                  'learning/task_dagger_once_v1/statistics.py','decision/semantics.py'):
            checked_sources[p]=f['runtime_files'][p]
        assert all(sha(ROOT/p)==v for p,v in checked_sources.items())
        old_counts={}
        for dirname,key in [('task-anchor-once-v1-20261027','failed_anchor_artifact_sha256'),('task-anchor-spawn-repair-v1-20261027','repair_artifact_sha256')]:
            for p,v in cfg[key].items():assert sha(ROOT/'artifacts'/dirname/p)==v
            old_counts[key]=len(cfg[key])
        protection=read(OUT/'protection-before.json')['runtime_sha256']
        defaults={p:v for p,v in protection.items() if p in ('main.py','run_match.py') or p.startswith('bots/')}
        assert defaults and all(sha(ROOT/p)==v for p,v in defaults.items())
        for key in ('task_checkpoint','crop_checkpoint'):assert sha(ROOT/cfg[key])==cfg[key+'_sha256']
        final_sha=read(HERE/'training-review.json')['checkpoint_sha256']
        assert sha(OUT/'model/trained.npz')==final_sha
        rows=read(OUT/'ledger.json');audits=read(OUT/'audit.json')
        assert len(rows)==len(audits)==120 and len({r['game'] for r in rows})==120
        ai={a['game']:a for a in audits};totals=Counter();turns=0;job_bytes=0;job_evidence={}
        assert set(ai)=={r['game'] for r in rows}
        assert {p.stem for p in (OUT/'job-inputs').glob('*.json')}==set(ai)
        for row in rows:
            deadline();key=row['game'];prefix=OUT/'job-results'/key
            records={n:read(Path(str(prefix)+'.'+n+'.json')) for n in ('row','audit','complete','storage')}
            jr,ja,jc,js=(records[n] for n in ('row','audit','complete','storage'))
            assert {k:v for k,v in row.items() if k!='first_request_difference_step'}==jr
            assert jc['passed'] and jc['game']==key and jc['row_sha256']==sha(Path(str(prefix)+'.row.json'))
            assert jc['certificate_sha256']==sha(Path(str(prefix)+'.audit.json'))
            assert ja['passed'] and ja['errors']==0 and all(ai[key][k]==v for k,v in ja.items())
            assert ai[key]['storage_receipt_sha256']==sha(Path(str(prefix)+'.storage.json'))
            assert js['passed'] and js['error_kind'] is None and js['output']['refused']==0
            assert js['output']['limit_bytes']==cfg['storage']['game_write_bytes'] and js['output']['used_bytes']<=js['output']['limit_bytes']
            assert ai[key]['output_write_bytes']==js['output']['used_bytes'];job_bytes+=js['output']['used_bytes']
            assert ai[key]['capture']['reader_joined'] and ai[key]['capture']['child_exit_code']==0
            assert ai[key]['capture']['written_bytes']<=cfg['storage']['process_log_bytes']
            plan=read(OUT/'job-inputs'/(key+'.json'));assert all(row[k]==v for k,v in plan.items())
            assert row['passed'] and row['status']==['DONE','DONE'] and row['states']==720 and row['action_turns']==719 and row['agent_errors']==0
            assert row['engine_sha256']==cfg['engine_sha256'] and row['crop_checkpoint_sha256']==cfg['crop_checkpoint_sha256']
            assert row['task_checkpoint_sha256']=={'candidate':final_sha,'baseline':cfg['task_checkpoint_sha256'],'teacher_task':None}[row['policy']]
            assert sha(OUT/'agents'/(key+'.py'))==row['bot_sha256']
            assert sha(OUT/'sources'/(row['opponent']+'.py'))==row['rival_sha256']==cfg['source_sha256'][row['opponent']+'.py']
            assert sha(OUT/'games'/key/'replay.json.gz')==row['replay_gzip_sha256']==ja['replay_sha256']
            assert sha(OUT/'traces'/(key+'.jsonl.gz'))==row['trace_sha256']==ja['trace_sha256']
            result=read(OUT/'games'/key/'result.json')
            assert result['passed'] and result['done'] and result['statuses']==['DONE','DONE'] and not result['agent_errors']
            assert result['recorded_states']==720 and result['action_turns']==719 and result['seed']==row['seed'] and result['bot_side']==row['side']
            assert result['engine_matches_official_commit'] and result['engine_sha256']==cfg['engine_sha256'] and result['upstream_commit']==cfg['engine_commit']
            assert result['final_money']==result['rewards']==row['banks']==row['rewards']==ja['official']['final_rewards']
            for a,b in [('semantic_replay_sha256','semantic_sha256'),('semantic_replay_sha256_v2','semantic_sha256_v2')]:
                assert result[a]==row[a]==ja['official'][b]
            assert ja['official']['passed'] and ja['official']['official_transitions_reexecuted']==719
            assert ja['official']['both_public_and_private_states_match_each_step'] and ja['official']['inventory_order_checked_each_step']
            assert ja['official']['engine_sha256']==cfg['engine_sha256']
            assert ja['actor_actions_reproduced']==ja['rival_actions_reproduced']==ja['trace_turns_reproduced']==719
            assert ja['parity_kind'] is None and not ja['original_parity_model_counts'] and ja['original_controller_parity_turns']==0
            live=ja['live_model_counts'];primary=ja['primary_model_counts'];assert live==primary
            assert ja['total_model_counts']=={k:2*v for k,v in live.items()}
            assert live['task_forwards']<=4000 and live['crop_forwards']<=8000 and live['task_rows']+live['crop_rows']<=160000
            assert live['controller_calls']==719 and ja['policy_turns']==2*(719+719)
            if row['policy']=='teacher_task':assert live['task_forwards']==live['task_rows']==0
            totals.update(ja['total_model_counts']);turns+=ja['policy_turns']
            for metric in ('bank','opponent_bank','initial_bank','opponent_initial_bank',*METRICS):assert type(row[metric]) in (int,float) and math.isfinite(row[metric])
            assert row['net_profit']==row['bank']-row['initial_bank'] and row['opponent_net_profit']==row['opponent_bank']-row['opponent_initial_bank']
            assert row['margin']==row['bank']-row['opponent_bank'] and row['score']==(1. if row['margin']>0 else .5 if row['margin']==0 else 0.)
            side=row['side'];assert row['banks'][side]==row['bank'] and row['banks'][1-side]==row['opponent_bank']
            assert row['initial_banks'][side]==row['initial_bank'] and row['initial_banks'][1-side]==row['opponent_initial_bank']
            job_evidence[key]={n:sha(Path(str(prefix)+'.'+n+'.json')) for n in records}
        control=verify_control(rows)
        independent={}
        for split,file in [('dev','development-gate.json'),('holdout','holdout-statistics.json')]:
            subset=[r for r in rows if r['split']==split];actual=statistics(subset,cfg,split);producer=read(OUT/file)
            equal(actual['comparisons'],producer['comparisons'],split+'.comparisons')
            for key,value in actual['verdict'].items():equal(value,producer[key],split+'.'+key)
            if split=='holdout':equal(actual['arm_summaries'],producer['arm_summaries'],split+'.arms')
            assert producer['checkpoint_sha256']==final_sha
            independent[split]=actual
        hv=independent['holdout']['verdict'];assert independent['dev']['verdict']['passed']
        assert summary['research_positive']==hv['research_positive'] and summary['teacher_retention']==hv['teacher_retention'] and summary['candidate_admission']==hv['admission']
        assert dict(totals)==summary['game_model_counts'] and turns==summary['policy_turns']<=cfg['policy_turn_budget']
        assert totals['task_forwards']<=cfg['task_forward_budget'] and totals['crop_forwards']<=cfg['crop_forward_budget']
        assert totals['task_rows']+totals['crop_rows']<=cfg['neural_input_row_budget']
        assert summary['live_official_transitions']==summary['primary_official_transitions']==120*719
        replay=read(HERE/'replay-verification.json');numeric=read(HERE/'numeric-verification.json')
        assert replay['passed'] and replay['official_transitions_completed']==replay['official_transitions_debited']==2157
        assert numeric['passed'] and numeric['debited_rows']==numeric['completed_rows']==dict(cpu=6388,gpu=6388)
        assert summary['live_official_transitions']+summary['primary_official_transitions']+2157==cfg['total_official_transition_limit']==174717
        assert summary['fit_attempts']==1 and summary['fit_completed'] and summary['fit_result_sha256']==sha(OUT/'fit-result.json')
        assert summary['machine_seconds']<=cfg['campaign_seconds'] and not summary['default_changed'] and not summary['automatic_teacher_admission']
        storage=read(OUT/'campaign-storage.json');assert storage['passed'] and storage['output']['refused']==0
        assert storage['output']['used_bytes']<=storage['output']['limit_bytes']
        assert storage['output']['limit_bytes']+storage['initial_owned_bytes']+storage['foreign_receipts_reserved_bytes']==cfg['storage']['primary_parent_write_bytes']
        owned=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());independent_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
        assert owned<=cfg['storage']['total_bytes'] and independent_bytes+used<=cfg['storage']['independent_reserve_bytes']
        free=shutil.disk_usage(OUT).free;assert free>=cfg['storage']['free_floor_bytes']
        assert all(sha(OUT/n)==v for n,v in input_hashes.items());deadline()
        save('final-statistics.json',dict(passed=True,method_sha256=sha(HERE/'final-review-method.json'),
            development=independent['dev'],holdout=independent['holdout'],stored_action_control=control,
            all_job_evidence_sha256=job_evidence,additional_neural_rows=0,additional_official_transitions=0))
        report=dict(schema='independent-anchor-reentry-final-result-review-v1',review_passed=True,created_utc=now().isoformat(),
            reviewer_role='Independent result reviewer',registration_sha256=sha(OUT/'registration.json'),freeze_sha256=sha(OUT/'freeze.json'),
            local_preregistration_commit=commit,checkpoint_sha256=final_sha,source_scope='Seven new files and direct frozen consumers; no repeated whole-repository audit.',
            checked_source_sha256=checked_sources,old_artifacts_preserved=old_counts,protected_default_source_files=len(defaults),
            initial_and_crop_checkpoint_byte_identity_verified=True,all_registered_jobs_completed=True,completed_games=120,development_games=24,holdout_games=96,
            official_transitions=dict(live=86280,primary_reexecuted=86280,independent_reexecuted=2157,total=174717),
            registered_game_model_counts=dict(totals),registered_policy_turns=turns,game_output_debited_bytes=job_bytes,
            owned_output_bytes_at_review=owned,independent_output_bytes_before_final_reports=independent_bytes,free_disk_bytes=free,
            training_review_sha256=sha(HERE/'training-review.json'),numeric_verification_sha256=sha(HERE/'numeric-verification.json'),
            replay_verification_sha256=sha(HERE/'replay-verification.json'),final_statistics_sha256=sha(HERE/'final-statistics.json'),
            independent_added_neural_rows=0,independent_added_official_transitions=0,
            independent_total_neural_rows_debited=12776,independent_total_neural_rows_completed=12776,
            independent_total_official_transitions_debited=2157,independent_total_official_transitions_completed=2157,
            scientific_verdict=hv,default_bot_changed=False,automatic_teacher_admission=False,
            reviewer_execution_seconds=time.monotonic()-started,deadline_utc=method['deadline_utc'],
            conservative_elapsed_seconds=(now()-datetime.fromisoformat(read(HERE/'numeric-reservation.json')['review_started_upper_bound_utc'])).total_seconds(),
            limitations=['Eight independent holdout seed clusters; bootstrap resamples do not increase the number of independent seeds.',
              'Per-opponent point guards are not statistical noninferiority guarantees.',
              'Teacher-task combines rule tasks and the frozen crop network; it is not an entirely rule-based teacher.',
              'All 120 producer job certificates and identities were checked. Exactly three action-recorded official reexecutions were independently run, not another 120.',
              'The 40 candidate/baseline stored-state action differences were checked without policy or model calls; changed control alone is not an economic causal attribution.',
              'Passing evidence review does not imply every scientific gate passed and does not change the default Bot.'])
        save('final-result-review.json',report)
        save('final-verification.json',dict(passed=True,created_utc=now().isoformat(),
            report_sha256=sha(HERE/'final-result-review.json'),statistics_sha256=sha(HERE/'final-statistics.json'),
            method_sha256=sha(HERE/'final-review-method.json'),source_sha256=sha(__file__),
            final_output_write_bytes_before_receipt=used,owned_independent_bytes_before_receipt=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
            independent_output_limit_bytes=268435456,extra_neural_rows=0,extra_official_transitions=0,
            temporary_files_created=0,deadline_utc=method['deadline_utc']))
        print(json.dumps(dict(review_passed=True,scientific_verdict=hv,report_sha256=sha(HERE/'final-result-review.json'),
             statistics_sha256=sha(HERE/'final-statistics.json'),elapsed_seconds=time.monotonic()-started)),flush=True)
    except BaseException as exc:
        save('final-review-failure.json',dict(passed=False,error_kind=type(exc).__name__,created_utc=now().isoformat(),
            extra_neural_rows=0,extra_official_transitions=0,review_incomplete=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.command=='prepare' else run()
