"""Review completed development attempts and reexecute preselected saved games only."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    assert not p.exists()
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    cfg=read(OUT/'registration.json');ledger=read(OUT/'development-ledger.json')
    entries=read(OUT/'entry-ledger.json') if (OUT/'entry-ledger.json').exists() else []
    audits=read(OUT/'audit.json');summary=read(OUT/'game-summary.json');attempts=read(OUT/'attempts.json')
    admission=read(OUT/'admission.json');review=read(OUT/'post-training-review.json')
    assert admission['passed'] and review['passed'] and review['offline_entry_passed']
    assert admission['post_training_review_sha256']==sha(OUT/'post-training-review.json')
    assert admission['reload_sha256']==sha(OUT/'reload.json')
    for arm in ('initial','trained'):assert sha(OUT/'model'/(arm+'.npz'))==admission[arm+'_sha256']
    assert len(ledger) in (6,12) and len(entries)==(2 if len(ledger)==12 else 0)
    expected=[(seed,arm,side,False) for seed in cfg['validation_seeds'][:len(ledger)//6] for arm in ('trained','teacher_task','initial') for side in (0,1)]
    if entries:expected.extend((cfg['validation_seeds'][0],'trained',side,True) for side in (0,1))
    assert [(r['seed'],r['policy'],r['side'],r['entry']) for r in attempts]==expected
    assert len({r['game'] for r in attempts})==len(expected)==len(ledger)+len(entries)
    assert len(audits)==len(expected) and {a['game'] for a in audits}=={r['game'] for r in attempts}
    by_audit={a['game']:a for a in audits};rows=ledger+entries
    assert [r['game'] for r in rows]==[r['game'] for r in attempts]
    models={'initial':admission['initial_sha256'],'trained':admission['trained_sha256'],'teacher_task':None}
    raw_cash_checks=0
    for r in rows:
        assert r['opponent']=='round2' and r['passed'] and r['final_status']==['DONE','DONE']
        assert r['states']==720 and r['action_turns']==719 and r['agent_errors']==0
        assert r['engine_sha256']==cfg['engine_sha256']
        assert r['rival_sha256']==cfg['source_sha256']['round2.py']==sha(OUT/'sources/round2.py')
        assert sha(ROOT/r['agent_relative'])==r['bot_sha256']
        folder=OUT/'games'/r['game'];raw=folder/'replay.json.gz'
        assert sha(raw)==r['replay_gzip_sha256']
        trace=OUT/'traces'/(r['game']+'.jsonl.gz');assert sha(trace)==r['trace_sha256']
        with gzip.open(raw,'rt',encoding='utf-8') as f:replay=json.load(f)
        assert len(replay['steps'])==720
        final=replay['steps'][-1]
        cash=[p['money'] for p in final[0]['observation']['farms']]
        assert cash==r['banks']==r['rewards']==[p['reward'] for p in final]
        assert [p['status'] for p in final]==['DONE','DONE']
        assert r['bank']==cash[r['side']] and r['opponent_bank']==cash[1-r['side']]
        assert r['net_profit']==r['bank']-3000 and r['margin']==r['bank']-r['opponent_bank']
        raw_cash_checks+=2
        audit=by_audit[r['game']]
        assert audit['passed'] and audit['task_checkpoint_sha256']==models[r['policy']]
        assert audit['crop_checkpoint_sha256']==cfg['crop_checkpoint_sha256']
        assert audit['actor_actions_reproduced']==audit['rival_actions_reproduced']==audit['decision_traces_reproduced']==719
        assert audit['fallback']==audit['load_errors']==audit['inference_errors']==0
        official=audit['official'];assert official['passed'] and official['official_transitions_reexecuted']==719
        assert official['semantic_sha256']==r['semantic_replay_sha256'] and official['semantic_sha256_v2']==r['semantic_replay_sha256_v2']
        assert official['final_rewards']==cash
    first=cfg['validation_seeds'][0]
    def mean_cash(seed,arm):
        group=[r for r in ledger if r['seed']==seed and r['policy']==arm]
        assert len(group)==2 and {r['side'] for r in group}=={0,1}
        return sum(r['bank'] for r in group)/2
    first_student=mean_cash(first,'trained');first_teacher=mean_cash(first,'teacher_task')
    first_stop=first_student<.75*first_teacher
    assert first_stop==(len(ledger)==6)
    final_seed=cfg['validation_seeds'][len(ledger)//6-1]
    assert summary['severe_regression']==(mean_cash(final_seed,'trained')<.75*mean_cash(final_seed,'teacher_task'))
    assert summary['completed_development_games']==len(ledger) and summary['completed_entry_repeats']==len(entries)
    assert summary['official_reexecution_transitions']==719*len(rows) and not summary['independent_holdout']
    independently={}
    for arm in ('teacher_task','initial','trained'):
        group=[r for r in ledger if r['policy']==arm];n=len(group)
        v=dict(games=n,mean_net_profit=sum(r['bank']-3000 for r in group)/n,
            mean_margin=sum(r['bank']-r['opponent_bank'] for r in group)/n,
            wins=sum(r['bank']>r['opponent_bank'] for r in group),draws=sum(r['bank']==r['opponent_bank'] for r in group),losses=sum(r['bank']<r['opponent_bank'] for r in group))
        assert v==summary['arms'][arm];independently[arm]=v
    for e in entries:
        r=next(r for r in ledger if r['seed']==first and r['policy']=='trained' and r['side']==e['side'])
        assert e['banks']==r['banks'] and e['semantic_replay_sha256_v2']==r['semantic_replay_sha256_v2']
    assert read(OUT/'task-development-workers.json')['closed']
    # The samples were fixed before results; absent entry repeats are not replaced.
    samples=[r for r in ledger if r['seed']==first and r['policy']=='trained']+entries
    assert len(samples)<=4
    from learning.task_bc_v1.games import audit
    receipts=[]
    for r in samples:
        got=audit(OUT,r,cfg);assert got==by_audit[r['game']]
        receipts.append(got)
        print('independent old game reexecuted',r['game'],flush=True)
    protected=read(OUT/'protection-before.json');public=ROOT.parent/'publication/ai-game-lab'
    for rel,h in protected['runtime'].items():assert sha(ROOT/rel)==h
    for rel,h in protected['public'].items():assert sha(public/rel)==h
    result=dict(passed=True,script_sha256=sha(Path(__file__)),freeze_sha256=sha(OUT/'freeze.json'),
        development_ledger_sha256=sha(OUT/'development-ledger.json'),audit_sha256=sha(OUT/'audit.json'),summary_sha256=sha(OUT/'game-summary.json'),
        development_games=len(ledger),entry_repeats=len(entries),all_raw_player_final_cash_checks=raw_cash_checks,
        first_seed_trained_mean_cash=first_student,first_seed_hybrid_reference_mean_cash=first_teacher,
        first_seed_cash_stop_triggered=first_stop,arms=independently,
        root_official_reexecution_transitions=719*len(rows),independent_sample_reexecution_transitions=719*len(samples),
        independent_samples=receipts,unavailable_entry_samples=2-len(entries),
        protected_runtime_files=len(protected['runtime']),protected_public_files=len(protected['public']),
        reviewer_new_games=0,reviewer_training_runs=0,
        limitations=['Complete ledger, final cash, identities, stop order and aggregates independently checked; full action/trace/official reexecution repeated only for preselected saved samples.',
                    'Two previously consumed development seeds and one opponent are not independent-test evidence or competitive generalization.',
                    'Hybrid reference uses epoch6 crop decisions; it is not the original frozen teacher.'])
    save(OUT/'independent-game-review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='independent_samples'}))

if __name__=='__main__':main()
