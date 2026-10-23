"""Independent preflight review using only retained old development observations."""
from pathlib import Path
from copy import deepcopy
import gzip,hashlib,json,subprocess,sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from learning.task_bc_v1.deployment import JointAgent
from learning.bc_crop.audit import stable
from decision.audit_market_teacher import observation

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def traces(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f]

def main():
    cfg=read(OUT/'registration.json');pre=read(OUT/'preflight.json');freeze=read(OUT/'freeze.json');commit=read(OUT/'pre-execution-commit.json')['commit'];pub=ROOT.parent/'publication/ai-game-lab'
    assert pre['passed'] and pre['freeze_sha256']==sha(OUT/'freeze.json') and pre['registration_sha256']==sha(OUT/'registration.json')
    assert pre['official_loader_calls']==2 and pre['official_agent_calls']==4 and pre['official_transitions']==pre['new_games']==pre['training_runs']==0
    for name in ['registration.json','freeze.json','pre-review.json']:
        assert subprocess.check_output(['git','-C',str(pub),'show',commit+':review/evidence/'+OUT.name+'/'+name])==(OUT/name).read_bytes()
    n=0
    for rel,h in freeze['runtime_files'].items():
        assert sha(ROOT/rel)==h
        if rel.startswith('learning/task_bc_holdout_v1/'):
            assert hashlib.sha256(subprocess.check_output(['git','-C',str(pub),'show',commit+':kaggriculture/'+rel])).hexdigest()==h;n+=1
    task=ROOT/'artifacts/task-bc-v1-20261023-run1';rows=read(task/'development-ledger.json')
    assert [x['arm'] for x in pre['rows']]==['trained','teacher_task']
    for r in pre['rows']:
        arm=r['arm'];old=next(x for x in rows if x['game']==r['source_game']);assert old['seed']==31205138 and old['side']==0 and old['policy']==arm
        rp=task/'games'/old['game']/'replay.json.gz';tp=task/'traces'/(old['game']+'.jsonl.gz');nt=OUT/'traces'/('preflight-'+arm+'.jsonl.gz')
        assert sha(rp)==old['replay_gzip_sha256']==r['source_replay_sha256'];assert sha(tp)==old['trace_sha256']==r['source_trace_sha256']
        assert sha(ROOT/r['wrapper_relative'])==r['wrapper_sha256'] and sha(nt)==r['trace_sha256']
        oldtrace=traces(tp)[:2];newtrace=traces(nt);assert len(newtrace)==2 and stable(oldtrace)==stable(newtrace)
        with gzip.open(rp,'rt',encoding='utf-8') as f:rep=json.load(f)
        def fresh():return JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],OUT/'model/trained.npz' if arm=='trained' else None,cfg['task_checkpoint_sha256'] if arm=='trained' else None)
        actor=fresh()
        for i,frame in enumerate(rep['steps'][:2]):
            assert actor(observation(frame,0),deepcopy(rep['configuration']))==rep['steps'][i+1][0]['action'];assert stable(actor.last)==stable(newtrace[i])
        obs=observation(rep['steps'][0],0);conf=deepcopy(rep['configuration']);altered=deepcopy(obs);ac=deepcopy(conf)
        ac['seed']=987654321;altered['seed']=123456789;altered['farms'][1]['private']={'shed':{'MELON':999999},'seeds':{'MELON':999999}};altered['market']['future_prices']={'MELON':999999}
        baseline,changed=fresh(),fresh();assert baseline(obs,conf)==changed(altered,ac);assert stable(baseline.last)==stable(changed.last)
        bad=deepcopy(obs);bad['info']={'seed':3}
        try:fresh()(bad,conf)
        except ValueError:pass
        else:raise AssertionError('Info not rejected')
    result=dict(passed=True,script_sha256=sha(Path(__file__)),preflight_sha256=sha(OUT/'preflight.json'),registration_sha256=sha(OUT/'registration.json'),freeze_sha256=sha(OUT/'freeze.json'),pre_execution_commit=commit,precommit_receipt_blobs=3,precommit_new_source_blobs=n,old_observation_decisions_reconstructed=4,direct_metadata_injection_comparisons=2,direct_info_rejections=2,official_loader_calls_by_reviewer=0,official_transitions_by_reviewer=0,new_games_by_reviewer=0,training_runs_by_reviewer=0,new_holdout_raw_inputs_read=0,limitations=['Actual official loader calls are root preflight evidence; reviewer checks its raw traces and directly reconstructs decisions and isolation on old observations only.'])
    p=OUT/'independent-preflight-review.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
