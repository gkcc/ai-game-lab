"""Offline identity/control audit on top of the existing ordered official replay."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time
from .agent import CropAgent, ROOT, sha
from .experiment import read, save, integrity
from decision.audit_market_teacher import observation
from decision.full_audit import reexecute


def load(path):
    ns={'__name__':'frozen_opponent','__file__':str(path)}
    exec(path.read_bytes(),ns);return ns['agent']


def stable(record):
    if isinstance(record,dict):return {k:stable(v) for k,v in record.items() if k!='inference_seconds'}
    if isinstance(record,list):return [stable(v) for v in record]
    return record


def check(job):
    out,row=job;out=Path(out);side=row['side'];policy=row['policy']
    folder=out/'games'/row['game']
    raw=folder/'replay.json.gz';assert sha(raw)==row['replay_gzip_sha256']
    with gzip.open(raw,'rt',encoding='utf-8') as f:replay=json.load(f)
    assert sha(ROOT/row['agent_relative'])==row['bot_sha256']
    assert sha(out/'sources'/(row['opponent']+'.py'))==row['rival_sha256']
    model=None
    if policy in ('initial','trained','smoke'):
        model=out/('smoke-model' if policy=='smoke' else 'model')/('initial.npz' if policy=='initial' else 'trained.npz')
    agent=(load(out/'sources/teacher.py') if policy=='original' else
           CropAgent('model' if model else 'teacher',model,sha(model) if model else None))
    rival=load(out/'sources'/(row['opponent']+'.py'))
    teacher=load(out/'sources/teacher.py')
    turns=None
    if policy!='original':
        trace=out/'traces'/(row['game']+'.jsonl.gz');assert sha(trace)==row['trace_sha256']
        with gzip.open(trace,'rt',encoding='utf-8') as f:turns=[json.loads(l) for l in f]
        assert len(turns)==719
    counts=Counter();choice_counts=Counter();plant_counts=Counter();linked={};inference=0.
    for step,(before,after) in enumerate(zip(replay['steps'],replay['steps'][1:])):
        o=observation(before,side);cfg=replay['configuration']
        predicted=agent(o,cfg)
        assert predicted==after[side]['action'],'Deployed weights/source failed action reproduction'
        assert rival(observation(before,1-side),cfg)==after[1-side]['action']
        if turns is None:continue
        assert stable(turns[step])==stable(agent.last),'Decision-time trace differs from reload'
        assert agent.last['checkpoint_sha256']==(sha(model) if model else None)
        assert not any(agent.last[k] for k in ('fallbacks','load_errors','inference_errors'))
        counts['actor_turns']+=1
        shadow=teacher(deepcopy(o),cfg)
        counts['same_observation_teacher_action_differences']+=shadow!=predicted
        for r in turns[step]['records']:
            counts['selector_calls']+=1;counts[r['outcome']]+=1
            counts['model_calls']+=r['source']=='model'
            counts['different_from_conditional_teacher']+=r['chosen']!=r['teacher']
            choice_counts[r['chosen']]+=1;inference+=r['inference_seconds']
            key=(step,r['unit']);assert key not in linked;linked[key]=r
            if r['outcome']=='plant_request':assert r['unit_request']==['PLANT',r['chosen']]
            if r['outcome']=='economic_reject':assert r['unit_request']==['PASS']
        for u,request in enumerate([predicted['farmer'],*predicted['hands']]):
            if request[0]=='PLANT':
                assert (step,u) in linked and linked[step,u]['chosen']==request[1]
                counts['plant_requests_linked']+=1
        counts['buy_seed_orders_changed_vs_teacher']+=sum(a[0]=='BUY_SEED' for a in predicted['market']) if (
            [a for a in predicted['market'] if a[0]=='BUY_SEED'] != [a for a in shadow['market'] if a[0]=='BUY_SEED']) else 0
    def sink(event):
        if event['player']!=side:return
        if event['phase']=='unit':
            counts['unit_'+event['status']]+=1
            if event.get('successful_plant'):
                if turns is not None:
                    r=linked[event['step'],event['unit']]
                    assert r['outcome']=='plant_request' and event['effective']==['PLANT',r['chosen']]
                    plant_counts[r['chosen']]+=1
                counts['official_successful_plants']+=1
            elif isinstance(event['requested'],list) and event['requested'] and event['requested'][0]=='PLANT':
                counts['plant_request_no_success']+=1
        elif event['phase']=='market':
            counts['market_'+event['status']]+=1
            if event['requested'][0]=='BUY_SEED':
                if turns is not None:
                    assert any(r['outcome']=='request_seed' and r['chosen']==event['requested'][1]
                               for r in turns[event['step']]['records'])
                counts['seed_units_actually_bought']+=event.get('executed_units',0)
    audit=reexecute(replay,trace=True,label_sink=sink)
    assert audit['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    assert audit['semantic_sha256']==row['semantic_replay_sha256']
    assert audit['final_rewards']==row['banks']
    return dict(game=row['game'],seed=row['seed'],side=side,opponent=row['opponent'],policy=policy,
                audit=audit,trace_sha256=row['trace_sha256'],replay_sha256=row['replay_gzip_sha256'],
                checkpoint_sha256=sha(model) if model else None,bot_sha256=row['bot_sha256'],
                reproduced_actor_actions=719,reproduced_rival_actions=719,raw_decision_traces_reproduced=turns is not None,
                counts=dict(counts),selected_crops=dict(choice_counts),official_planted_crops=dict(plant_counts),
                original_inference_seconds=inference,fallbacks=0,load_errors=0,inference_errors=0)


def run(out, stages, workers=4):
    integrity(out);rows=sum((read(out/(s+'-ledger.json')) for s in stages),[])
    name='audit-'+'-'.join(stages)+'.json';assert not (out/name).exists()
    cert=[];tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(check,(str(out),r)) for r in rows]
        for f in as_completed(futures):
            cert.append(f.result())
            if len(cert)%8==0 or len(cert)==len(rows):print(f'official audit: {len(cert)}/{len(rows)}',flush=True)
    save(out/name,dict(passed=True,games=len(cert),official_transitions=len(cert)*719,
        rows=sorted(cert,key=lambda r:r['game']),elapsed_seconds=time.perf_counter()-tick))
    integrity(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stages',nargs='+',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.out.resolve(),a.stages,a.workers)
