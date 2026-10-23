"""Independent review of fixed historical filename repair; never dispatches games."""
from pathlib import Path
import gzip,hashlib,json,sys,subprocess
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def traces(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def write(p,v):
    assert not p.exists()
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def regression():
    from learning.task_entry_repair_v1.experiment import verify
    from learning.task_bc_v1.deployment import JointAgent
    from decision.audit_market_teacher import observation
    from learning.bc_crop.audit import stable
    cfg=verify(OUT);reg=read(OUT/'regressions.json');freeze=read(OUT/'freeze.json')
    assert reg['passed'] and reg['freeze_sha256']==sha(OUT/'freeze.json') and reg['registration_sha256']==sha(OUT/'registration.json')
    assert reg['official_loader_sha256']==cfg['official_loader_sha256']
    assert reg['official_load_attempts']==2 and reg['worker_load_attempts']==1
    assert reg['official_agent_calls']==reg['worker_agent_calls']==reg['direct_calls']==2
    assert reg['official_transitions']==reg['complete_games']==reg['training_updates']==0
    assert reg['old_official_counterexample']=={'type':'InvalidArgument','cause':'NameError: __file__ is not defined'}
    for rel,h in reg['files_sha256'].items():assert sha(ROOT/rel)==h
    commit=read(OUT/'pre-execution-commit.json')['commit'];public=ROOT.parent/'publication/ai-game-lab'
    for name in ['registration.json','freeze.json','pre-review.json']:
        assert subprocess.check_output(['git','-C',str(public),'show',commit+':review/evidence/'+OUT.name+'/'+name])==(OUT/name).read_bytes()
    for rel,h in freeze['runtime_files'].items():
        if rel.startswith('learning/task_entry_repair_v1/'):
            raw=subprocess.check_output(['git','-C',str(public),'show',commit+':kaggriculture/'+rel])
            assert hashlib.sha256(raw).hexdigest()==h
    previous=ROOT/cfg['previous_root'];old=next(r for r in read(previous/'development-ledger.json') if r['seed']==cfg['seed'] and r['side']==0 and r['policy']=='trained')
    replay=gz(previous/'games'/old['game']/'replay.json.gz');oldtrace=traces(previous/'traces'/(old['game']+'.jsonl.gz'))[:2]
    official=traces(OUT/'traces/regression-official.jsonl.gz');worker=traces(OUT/'traces/regression-worker.jsonl.gz');direct=traces(OUT/'traces/regression-direct.jsonl.gz')
    assert len(official)==len(worker)==len(direct)==2
    assert stable(official)==stable(worker)==stable(direct)==stable(oldtrace)
    actions=read(OUT/'regression-local/official-direct-actions.json')
    assert actions==read(OUT/'regression-local/worker-actions.json')==[r[0]['action'] for r in replay['steps'][1:3]]
    actor=JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],OUT/'model/trained.npz',cfg['task_checkpoint_sha256'])
    for i,frame in enumerate(replay['steps'][:2]):
        assert actor(observation(frame,0),replay['configuration'])==actions[i]
        assert stable(actor.last)==stable(official[i])
    return dict(passed=True,script_sha256=sha(Path(__file__)),freeze_sha256=sha(OUT/'freeze.json'),registration_sha256=sha(OUT/'registration.json'),pre_execution_commit=commit,regressions_sha256=sha(OUT/'regressions.json'),all_regression_file_hashes=len(reg['files_sha256']),precommit_receipt_blobs_verified=3,precommit_new_source_blobs_verified=5,raw_trace_paths_compared=4,saved_observations_independently_reconstructed=2,reviewer_official_loader_calls=0,reviewer_official_transitions=0,reviewer_new_games=0,reviewer_training_runs=0,limitations=['Actual loader calls were made by the frozen root regression; reviewer verifies its raw actions/traces, source, identities and separately reconstructs two decisions.'])

def full():
    reg=regression();assert read(OUT/'independent-regression-review.json')==reg
    from learning.task_bc_v1.games import audit
    from learning.bc_crop.audit import stable
    cfg=read(OUT/'registration.json');previous=ROOT/cfg['previous_root'];rows=read(OUT/'entry-ledger.json');attempts=read(OUT/'attempts.json');certs=read(OUT/'audit.json');result=read(OUT/'result.json')
    assert result['passed'] and len(rows)==len(attempts)==len(certs)==2
    assert [a['side'] for a in attempts]==[0,1] and all(a['seed']==31205138 for a in attempts)
    assert result['training_runs']==result['new_economic_samples']==0 and result['prior_stage_acceptance_passed'] is False
    assert result['old_development_mean_increment']==-155 and result['root_official_reexecution_transitions']==1438
    for name,key in [('entry-ledger.json','ledger_sha256'),('audit.json','audit_sha256'),('regressions.json','regressions_sha256'),('freeze.json','freeze_sha256')]:assert sha(OUT/name)==result[key]
    checked=[]
    for side,row in enumerate(rows):
        assert row['side']==side and row['seed']==31205138 and row['policy']=='trained' and row['opponent']=='round2'
        assert row['game']==attempts[side]['game']==certs[side]['game'] and row['historical_reproduction']
        old=next(x for x in read(previous/'development-ledger.json') if x['game']==row['reference_game'])
        assert old['side']==side and old['policy']=='trained'
        folder=OUT/'games'/row['game'];replay=gz(folder/'replay.json.gz');original=gz(previous/'games'/old['game']/'replay.json.gz')
        assert sha(folder/'replay.json.gz')==row['replay_gzip_sha256'] and replay==read(folder/'replay.json')
        assert len(replay['steps'])==720 and row['action_turns']==719 and row['final_status']==['DONE','DONE']
        assert [[s['action'] for s in f] for f in replay['steps'][1:]]==[[s['action'] for s in f] for f in original['steps'][1:]]
        actualtrace=traces(OUT/'traces'/(row['game']+'.jsonl.gz'));priortrace=traces(previous/'traces'/(old['game']+'.jsonl.gz'))
        assert len(actualtrace)==719 and stable(actualtrace)==stable(priortrace)
        assert sha(OUT/'traces'/(row['game']+'.jsonl.gz'))==row['trace_sha256']
        final=replay['steps'][-1];cash=[f['money'] for f in final[0]['observation']['farms']]
        assert cash==row['banks']==old['banks']==[p['reward'] for p in final]==[41430.,37919.] if side==0 else cash==row['banks']==old['banks']==[p['reward'] for p in final]==[37919.,41430.]
        rawresult=read(folder/'result.json')
        assert rawresult['passed'] and rawresult['done'] and rawresult['statuses']==['DONE','DONE'] and rawresult['recorded_states']==720 and rawresult['action_turns']==719 and not rawresult['agent_errors']
        assert rawresult['final_money']==rawresult['rewards']==cash and rawresult['engine_sha256']==cfg['engine_sha256']
        assert rawresult['upstream_commit']==cfg['engine_commit']
        assert rawresult['agent_file_sha256'][str(ROOT/row['agent_relative'])]==row['bot_sha256']==sha(ROOT/row['agent_relative'])
        assert rawresult['agent_file_sha256'][str(OUT/'sources/round2.py')]==row['rival_sha256']==sha(OUT/'sources/round2.py')
        for name in ['semantic_replay_sha256','semantic_replay_sha256_v2']:assert rawresult[name]==row[name]==old[name]
        got=audit(OUT,row,cfg);assert got==certs[side]
        checked.append(got);print('Independent historical entry reexecuted',row['game'],flush=True)
    protected=read(OUT/'protection-before.json');public=ROOT.parent/'publication/ai-game-lab'
    for rel,h in protected['runtime'].items():assert sha(ROOT/rel)==h
    for rel,h in protected['public'].items():assert sha(public/rel)==h
    assert read(previous/'stage-outcome.json')['stage_acceptance_passed'] is False
    v=dict(passed=True,script_sha256=sha(Path(__file__)),result_sha256=sha(OUT/'result.json'),freeze_sha256=sha(OUT/'freeze.json'),regression_review_sha256=sha(OUT/'independent-regression-review.json'),complete_entries=2,all_actor_and_rival_actions_compared=2876,complete_decision_traces_compared=1438,all_final_player_cash_checks=4,root_official_reexecution_transitions=1438,independent_official_reexecution_transitions=1438,independent_audits=checked,protected_runtime_files=len(protected['runtime']),protected_public_files=len(protected['public']),prior_stage_acceptance_passed=False,reviewer_new_games=0,reviewer_training_runs=0,new_economic_samples=0,limitations=['Two fixed historical reproductions establish filename compatibility for the retained model and replay paths; no new economic evidence or generalization claim.','The prior task_bc_v1 failed outcome remains unchanged.'])
    write(OUT/'independent-review.json',v);print(json.dumps({k:v for k,v in v.items() if k!='independent_audits'}))

if __name__=='__main__':
    if '--regression-only' in sys.argv:
        r=regression();write(OUT/'independent-regression-review.json',r);print(json.dumps(r))
    else:full()
