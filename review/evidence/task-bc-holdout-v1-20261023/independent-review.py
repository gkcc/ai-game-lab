"""Independent whole-panel arithmetic and prespecified saved replay review.
Never dispatch games, update weights, or call production statistics.analyze.
"""
from pathlib import Path
import gzip,hashlib,json,math,subprocess,sys
from collections import Counter
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
ARMS=('trained','teacher_task');METRICS=('net_profit','opponent_net_profit','margin','score')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(values):return math.fsum(values)/len(values)
def interval(values):
    a=sorted(float(x) for x in values);result=[]
    for q in (.025,.975):
        at=(len(a)-1)*q;lo=math.floor(at);hi=math.ceil(at)
        result.append(a[lo]+(a[hi]-a[lo])*(at-lo))
    return result

def equal(actual,expected,path='root'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(path,'keys')
        for k,v in expected.items():equal(actual[k],v,path+'/'+str(k))
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,v in enumerate(expected):equal(actual[i],v,path+'/'+str(i))
    elif isinstance(expected,float):
        assert isinstance(actual,(int,float)) and math.isfinite(actual),path
        assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-10),(path,actual,expected)
    else:assert actual==expected,(path,actual,expected)

def arm_summary(rows):
    wins=sum(r['score']==1 for r in rows);draws=sum(r['score']==.5 for r in rows);losses=len(rows)-wins-draws
    return dict(games=len(rows),wins=wins,draws=draws,losses=losses,means={k:avg([r[k] for r in rows]) for k in METRICS},opponent_wins=losses,opponent_draws=draws,opponent_losses=wins,opponent_score=avg([1-r['score'] for r in rows]))

def summary(pairs,counts=None):
    arms={a:arm_summary([p[a] for p in pairs]) for a in ARMS}
    deltas={k:avg([p['delta'][k] for p in pairs]) for k in METRICS};den=arms['teacher_task']['means']['net_profit']
    r=dict(pairs=len(pairs),arms=arms,delta_means=deltas,relative_net_profit_mean=deltas['net_profit']/den if den>0 else None,reference_net_profit_mean=den)
    if counts is None:return r
    seeds=list(dict.fromkeys(p['seed'] for p in pairs));assert len(seeds)==8
    groups=[[p for p in pairs if p['seed']==s] for s in seeds];assert len({len(g) for g in groups})==1
    ds={k:np.array([avg([p['delta'][k] for p in g]) for g in groups]) for k in METRICS};refs=np.array([avg([p['teacher_task']['net_profit'] for p in g]) for g in groups])
    bd={k:counts@v/8 for k,v in ds.items()};br=counts@refs/8;valid=den>0 and all(x>0 for x in br)
    r.update(delta_ci95={k:interval(v) for k,v in bd.items()},relative_net_profit_defined=bool(valid),nonpositive_bootstrap_reference_count=sum(x<=0 for x in br),bootstrap_reference_min=float(min(br)),relative_net_profit_ci95=interval(bd['net_profit']/br) if valid else None)
    r['nonpositive_bootstrap_reference_count']=int(r['nonpositive_bootstrap_reference_count'])
    if not valid:r.update(relative_net_profit_mean=None,relative_undefined_reason='Nonpositive point or resampled reference net profit')
    return r

def main():
    from learning.task_bc_holdout_v1.experiment import verify
    cfg=verify(OUT);run=read(OUT/'run-summary.json');stats=read(OUT/'statistics.json');rows=read(OUT/'ledger.json');audits=read(OUT/'audit.json');attempts=read(OUT/'attempts.json')
    assert run['complete'] and run['full_gate_evaluated'] and not run['failures'] and not run['severe_regression']
    assert len(rows)==len(audits)==len(attempts)==96 and not read(OUT/'failures.json')
    assert read(OUT/'workers.json')['closed'] and read(OUT/'workers.json')['max_workers']==2
    assert run['statistics_sha256']==sha(OUT/'statistics.json') and run['official_reexecution_transitions']==96*719
    expected=[(s,o,side,a) for s in cfg['test_seeds'] for o in cfg['opponents'] for a in cfg['arms'] for side in (0,1)]
    key=lambda r:(r['seed'],r['opponent'],r['side'],r['policy'])
    assert [key(r) for r in rows]==expected==[key(r) for r in attempts]
    assert len({r['game'] for r in rows})==96 and [r['game'] for r in rows]==[r['game'] for r in audits]
    indexed={key(r):r for r in rows};auditindex={r['game']:r for r in audits};cashchecks=0
    for r,a in zip(rows,attempts):
        assert r['passed'] and r['official_filename_entry'] and r['status']==['DONE','DONE'] and r['states']==720 and r['action_turns']==719 and r['agent_errors']==0
        assert r['bot_sha256']==a['bot_sha256']==sha(ROOT/r['agent_relative']);assert r['rival_sha256']==a['rival_sha256']==cfg['source_sha256'][r['opponent']+'.py']==sha(OUT/'sources'/(r['opponent']+'.py'))
        assert r['task_checkpoint_sha256']==(cfg['task_checkpoint_sha256'] if r['policy']=='trained' else None) and r['crop_checkpoint_sha256']==cfg['crop_checkpoint_sha256']
        folder=OUT/'games'/r['game'];rp=folder/'replay.json.gz';assert sha(rp)==r['replay_gzip_sha256'];assert sha(OUT/'traces'/(r['game']+'.jsonl.gz'))==r['trace_sha256']
        with gzip.open(rp,'rt',encoding='utf-8') as f:replay=json.load(f)
        assert len(replay['steps'])==720 and replay['info']['seed']==r['seed'] and replay['configuration'].get('seed') is None
        first,last=replay['steps'][0],replay['steps'][-1];initial=[p['money'] for p in first[0]['observation']['farms']];final=[p['money'] for p in last[0]['observation']['farms']];side=r['side']
        assert initial==r['initial_banks'] and final==r['banks']==r['rewards']==[p['reward'] for p in last]
        assert [p['status'] for p in last]==['DONE','DONE'];assert r['bank']==final[side] and r['opponent_bank']==final[1-side]
        assert r['initial_bank']==initial[side] and r['opponent_initial_bank']==initial[1-side]
        assert r['net_profit']==final[side]-initial[side] and r['opponent_net_profit']==final[1-side]-initial[1-side]
        assert r['margin']==final[side]-final[1-side] and r['score']==(1. if r['margin']>0 else .5 if r['margin']==0 else 0.)
        cashchecks+=2
        actual=read(folder/'result.json');assert actual['passed'] and actual['done'] and actual['seed']==r['seed'] and actual['bot_side']==side and actual['engine_sha256']==cfg['engine_sha256'] and actual['upstream_commit']==cfg['engine_commit'] and not actual['agent_errors']
        assert actual['final_money']==actual['rewards']==final and actual['statuses']==['DONE','DONE'] and actual['recorded_states']==720 and actual['action_turns']==719
        assert actual['agent_file_sha256'][str(ROOT/r['agent_relative'])]==r['bot_sha256'];assert actual['agent_file_sha256'][str(OUT/'sources'/(r['opponent']+'.py'))]==r['rival_sha256']
        for k in ('semantic_replay_sha256','semantic_replay_sha256_v2'):assert actual[k]==r[k]
        marker=read(OUT/'job-results'/(r['game']+'.complete.json'));assert marker['passed'] and marker['game']==r['game']
        rf=OUT/'job-results'/(r['game']+'.row.json');af=OUT/'job-results'/(r['game']+'.audit.json');assert sha(rf)==marker['row_sha256'] and sha(af)==marker['certificate_sha256'] and read(rf)==r
        cert=auditindex[r['game']];assert read(af)==cert and cert['passed'] and cert['actor_actions_reproduced']==cert['rival_actions_reproduced']==cert['decision_traces_reproduced']==719
        assert cert['task_checkpoint_sha256']==r['task_checkpoint_sha256'] and cert['crop_checkpoint_sha256']==r['crop_checkpoint_sha256']
        assert cert['fallback']==cert['load_errors']==cert['inference_errors']==0
        official=cert['official'];assert official['passed'] and official['official_transitions_reexecuted']==719 and official['engine_sha256']==cfg['engine_sha256']
        assert official['semantic_sha256']==r['semantic_replay_sha256'] and official['semantic_sha256_v2']==r['semantic_replay_sha256_v2'] and official['final_rewards']==final
    safety=[]
    for seed in cfg['test_seeds']:
        a={arm:avg([r['bank'] for r in rows if r['seed']==seed and r['policy']==arm]) for arm in ARMS}
        assert a['trained']>=.5*a['teacher_task'];safety.append(dict(seed=seed,means=a))
    pairs=[]
    for seed in cfg['test_seeds']:
        for op in cfg['opponents']:
            for side in (0,1):
                rr={arm:indexed[seed,op,side,arm] for arm in ARMS}
                for f in ('initial_bank','opponent_initial_bank'):assert rr['trained'][f]==rr['teacher_task'][f]
                den=rr['teacher_task']['net_profit'];delta={k:rr['trained'][k]-rr['teacher_task'][k] for k in METRICS}
                pairs.append(dict(seed=seed,opponent=op,side=side,**{a:{k:rr[a][k] for k in ('bank','opponent_bank','initial_bank','opponent_initial_bank',*METRICS)} for a in ARMS},delta=delta,relative_net_profit=delta['net_profit']/den if den>0 else None))
    equal(stats['paired_detail'],pairs)
    draws=np.random.Generator(np.random.PCG64(2026102341)).integers(0,8,(20000,8));counts=np.zeros((20000,8),dtype=np.int64)
    np.add.at(counts,(np.repeat(np.arange(20000),8),draws.reshape(-1)),1);assert np.all(counts.sum(axis=1)==8)
    overall=summary(pairs,counts);byop={op:summary([p for p in pairs if p['opponent']==op],counts) for op in cfg['opponents']};byseed=[dict(seed=s,**summary([p for p in pairs if p['seed']==s])) for s in cfg['test_seeds']]
    equal(stats['overall'],overall);equal(stats['by_opponent'],byop);equal(stats['by_seed'],byseed)
    primary=overall['relative_net_profit_defined'] and overall['relative_net_profit_ci95'][0]>-.02
    guards={o:bool(v['relative_net_profit_defined'] and v['relative_net_profit_mean']>=-.05 and v['delta_means']['score']>=-.05) for o,v in byop.items()}
    passed=bool(primary and all(guards.values()));assert stats['primary_gate_passed']==primary and stats['opponent_point_guards']==guards and stats['passed']==passed and stats['panel_improvement']==bool(passed and overall['relative_net_profit_ci95'][0]>0)
    print('Independent96row/48pair/8cluster arithmetic complete; beginning fixed replay samples',flush=True)
    from learning.task_bc_holdout_v1.audit import audit
    sample_plan=read(OUT/'independent-sample-plan.json');samples=[]
    for sample in sample_plan['samples']:
        r=indexed[key(sample)];assert r['game']==sample['game'];got=audit(OUT,r,cfg);assert got==auditindex[r['game']];samples.append(got)
        print('Independent fixed replay verified',r['game'],flush=True)
    assert len(samples)==4
    protected=read(OUT/'protection-before.json');pub=ROOT.parent/'publication/ai-game-lab'
    for rel,h in protected['runtime'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in protected['public'].items():assert sha(pub/rel)==h,rel
    assert read(ROOT/'artifacts/task-bc-v1-20261023-run1/stage-outcome.json')['stage_acceptance_passed'] is False
    result=dict(passed=True,script_sha256=sha(Path(__file__)),freeze_sha256=sha(OUT/'freeze.json'),ledger_sha256=sha(OUT/'ledger.json'),audit_sha256=sha(OUT/'audit.json'),statistics_sha256=sha(OUT/'statistics.json'),run_summary_sha256=sha(OUT/'run-summary.json'),independent_sample_plan_sha256=sha(OUT/'independent-sample-plan.json'),games_checked=96,raw_final_player_cash_checks=cashchecks,paired_results=48,seed_clusters=8,bootstrap_draws=20000,bootstrap_method='PCG64 fixed draws converted to integer counts; weighted seed sums and explicit sorted linear interpolation, independent of production analyze.',statistics_gate_passed=passed,primary_gate_passed=bool(primary),opponent_guards=guards,overall=overall,by_opponent=byop,by_seed=byseed,safety_seed_cash=safety,root_official_transitions=96*719,independent_sample_official_transitions=4*719,independent_samples=samples,protected_runtime_files=len(protected['runtime']),protected_public_files=len(protected['public']),reviewer_new_games=0,reviewer_training_runs=0,limitations=['All96raw final cash and ledger/certificate/hash identities checked; independent full official/action/trace reexecution only four prespecified games.','Only eight sampled seed clusters and three known opponents;20kbootstrap draws do not add independent information.','Reference is rule tasks plus fixed crop model; point guards are not subgroup statistical noninferiority.','Original task training-stage failed entry outcome and default remain unchanged.'])
    p=OUT/'independent-review.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['independent_samples','overall','by_opponent','by_seed']}))
if __name__=='__main__':main()
