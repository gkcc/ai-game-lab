"""Real official filename games and bounded replay checks; no replacement server."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import runpy
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.audit import stable, load
from learning.task_bc_holdout_v1.games import write_new
from learning.task_bc_v1.deployment import JointAgent
from learning.task_reward_gate_v1.experiment import read, verify, gate_identity
from learning.task_reward_gate_v1.policy import GateAgent, GateModel
from decision.audit_market_teacher import observation
from decision.full_audit import reexecute
from decision.semantics import canonical


def actor(out,plan,cfg,trace=None):
    model=GateModel.load(Path(out)/'gate-model.json',gate_identity(out)) if plan['policy']=='gated' else None
    return GateAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],
                     ROOT/cfg['task_checkpoint'],cfg['task_checkpoint_sha256'],
                     mode=plan['policy'],assigned_window=plan['window'],model=model,
                     trace=trace,limits=cfg['game_limits'])


def wrapper(out,plan,cfg):
    out=Path(out);key=plan['game'];path=out/'agents'/(key+'.py')
    trace=out/'traces'/(key+'.jsonl.gz')
    assert not trace.exists() and not path.exists()
    code=('from pathlib import Path\nimport sys\n'
          '_p=globals().get("__file__",sys._getframe().f_code.co_filename)\n'
          'ROOT=next(p for p in Path(_p).resolve().parents if (p/"learning/task_reward_gate_v1/policy.py").is_file())\n'
          'sys.path.insert(0,str(ROOT))\n'
          'from learning.task_reward_gate_v1.policy import GateAgent,GateModel\n')
    # Only deployment code is imported: neither registration nor replay has a
    # path into policy inputs. Seed/opponent are not arguments to the agent.
    model=('GateModel.load(ROOT/'+repr((out/'gate-model.json').relative_to(ROOT).as_posix())+','+repr(gate_identity(out))+')'
           if plan['policy']=='gated' else 'None')
    code+=('_runner=GateAgent(ROOT/'+repr(cfg['crop_checkpoint'])+','+repr(cfg['crop_checkpoint_sha256'])+
           ',ROOT/'+repr(cfg['task_checkpoint'])+','+repr(cfg['task_checkpoint_sha256'])+
           ',mode='+repr(plan['policy'])+',assigned_window='+repr(plan['window'])+',model='+model+
           ',trace=ROOT/'+repr(trace.relative_to(ROOT).as_posix())+',limits='+repr(cfg['game_limits'])+')\n'
           'def agent(observation,configuration):\n    return _runner(observation,configuration)\n'
           'submission_agent=agent\n')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(code)
    return path


def model_counts(last):
    return dict(task_forwards=len(last['records']),
                task_rows=sum(len(r['inputs']['candidates']) for r in last['records']),
                crop_forwards=len(last['crop_records']),crop_rows=len(last['crop_records']),gate_evaluations=0)


def audit(out,row,cfg):
    out=Path(out);side=row['side'];key=row['game']
    rp=out/'games'/key/'replay.json.gz';tp=out/'traces'/(key+'.jsonl.gz')
    assert sha(rp)==row['replay_gzip_sha256'] and sha(tp)==row['trace_sha256']
    with gzip.open(rp,'rt',encoding='utf-8') as f: replay=json.load(f)
    agent=actor(out,row,cfg);rival=load(out/'sources'/(row['opponent']+'.py'))
    original=(JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],
                       ROOT/cfg['task_checkpoint'],cfg['task_checkpoint_sha256'])
              if row['baseline_parity'] else None)
    shadow_counts=Counter();parity_counts=Counter();opportunities=[];shadow_turns=0
    with gzip.open(tp,'rt',encoding='utf-8') as trace:
        for step,(before,after) in enumerate(zip(replay['steps'],replay['steps'][1:])):
            line=trace.readline();assert line,'Missing decision trace'
            stored=json.loads(line);o=observation(before,side)
            request=agent(o,replay['configuration'])
            assert canonical(request)==canonical(after[side]['action'])
            assert canonical(rival(observation(before,1-side),replay['configuration']))==canonical(after[1-side]['action'])
            assert stable(agent.last)==stable(stored),'Reloaded decision trace differs'
            assert all(agent.last[k]==0 for k in ('fallbacks','load_errors','inference_errors'))
            if original is not None:
                assert canonical(original(o,replay['configuration']))==canonical(request),'Zero-gate baseline changed old student'
                parity_counts.update(model_counts(original.last))
            events=agent.last['gate_events']
            for event in events:
                record=dict(event,complete_request_changed=False)
                if event['changed_id']:
                    shadow=actor(out,{**row,'policy':'baseline'},cfg)
                    other=shadow(o,replay['configuration']);shadow_turns+=1
                    assert len(shadow.last['records'])<=32 and len(shadow.last['crop_records'])<=32
                    sc=model_counts(shadow.last)
                    assert sc['task_rows']+sc['crop_rows']<=2000
                    shadow_counts.update(sc)
                    record['complete_request_changed']=canonical(other)!=canonical(request)
                opportunities.append(record)
        assert not trace.readline(),'Extra decision trace'
    execution=Counter()
    def sink(event):
        execution[event['phase']+':'+event['status']]+=1
    official=reexecute(replay,trace=True,label_sink=sink)
    assert official['engine_sha256']==cfg['engine_sha256']
    assert official['semantic_sha256']==row['semantic_replay_sha256']
    assert official['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    assert official['final_rewards']==row['banks']==row['rewards']
    counts=agent.last['gate_counts']
    total=Counter(counts);total.update(counts);total.update(shadow_counts);total.update(parity_counts)
    return dict(game=key,passed=True,official=official,opportunities=opportunities,
        complete_request_changes=sum(e['complete_request_changed'] for e in opportunities),
        live_model_counts=counts,primary_model_counts=counts,shadow_model_counts=dict(shadow_counts),
        baseline_parity_model_counts=dict(parity_counts),total_model_counts=dict(total),
        policy_turns=4*719+shadow_turns+(719 if original else 0),
        original_student_parity_turns=719 if original else 0, shadow_turns=shadow_turns,
        replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'],
        actor_actions_reproduced=719,rival_actions_reproduced=719,trace_turns_reproduced=719,
        execution_counts=dict(execution),errors=0)


def job(out,key):
    out=Path(out).resolve();cfg=verify(out);plan=read(out/'job-inputs'/(key+'.json'))
    bot=out/'agents'/(key+'.py');rival=out/'sources'/(plan['opponent']+'.py')
    assert plan['game']==key and sha(bot)==plan['bot_sha256']
    assert sha(rival)==plan['rival_sha256']==cfg['source_sha256'][rival.name]
    if plan['policy']=='gated': assert gate_identity(out)==plan['gate_model_sha256']
    folder=out/'games'/key;assert not folder.exists()
    saved_argv,saved_cwd=sys.argv,Path.cwd()
    try:
        sys.argv=[str(ROOT/'run_match.py'),'--bot',str(bot),'--opponent',str(rival),
                  '--seed',str(plan['seed']),'--side',str(plan['side']),'--name',key,'--output-root',str(out/'games')]
        os.chdir(ROOT.parent)
        try:runpy.run_path(str(ROOT/'run_match.py'),run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (0,None):raise RuntimeError('Official filename entry failed') from exc
    finally:sys.argv=saved_argv;os.chdir(saved_cwd)
    result=read(folder/'result.json');replay=read(folder/'replay.json')
    assert result['passed'] and result['done'] and result['statuses']==['DONE','DONE']
    assert result['recorded_states']==720 and result['action_turns']==719 and not result['agent_errors']
    assert result['engine_matches_official_commit'] and result['engine_sha256']==cfg['engine_sha256']
    assert result['upstream_commit']==cfg['engine_commit']
    assert result['agent_file_sha256'][str(bot)]==sha(bot)==plan['bot_sha256']
    assert result['agent_file_sha256'][str(rival)]==sha(rival)==plan['rival_sha256']
    assert result['seed']==replay['info']['seed']==plan['seed'] and result['bot_side']==plan['side']
    assert len(replay['steps'])==720
    initial=[f['money'] for f in replay['steps'][0][0]['observation']['farms']]
    banks=[f['money'] for f in replay['steps'][-1][0]['observation']['farms']]
    assert banks==result['rewards']==result['final_money']
    with (folder/'replay.json.gz').open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1) as stream:
            stream.write((folder/'replay.json').read_bytes())
    side=plan['side'];margin=banks[side]-banks[1-side]
    row=dict(plan,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
        initial_banks=initial,banks=banks,rewards=result['rewards'],bank=banks[side],opponent_bank=banks[1-side],
        initial_bank=initial[side],opponent_initial_bank=initial[1-side],net_profit=banks[side]-initial[side],
        opponent_net_profit=banks[1-side]-initial[1-side],margin=margin,score=1. if margin>0 else .5 if margin==0 else 0.,
        semantic_replay_sha256=result['semantic_replay_sha256'],semantic_replay_sha256_v2=result['semantic_replay_sha256_v2'],
        replay_gzip_sha256=sha(folder/'replay.json.gz'),trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')),
        engine_sha256=cfg['engine_sha256'],task_checkpoint_sha256=cfg['task_checkpoint_sha256'],crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'])
    write_new(out/'job-results'/(key+'.row.json'),row)
    certificate=audit(out,row,cfg);verify(out)
    write_new(out/'job-results'/(key+'.audit.json'),certificate)
    write_new(out/'job-results'/(key+'.complete.json'),dict(game=key,passed=True,
        row_sha256=sha(out/'job-results'/(key+'.row.json')),certificate_sha256=sha(out/'job-results'/(key+'.audit.json'))))


def pair(out,a,b):
    """One-shot causal label with identical reset and pre-intervention prefix."""
    out=Path(out);cert=read(out/'job-results'/(a['game']+'.audit.json'))
    events=cert['opportunities'];assert len(events)<=1
    base=dict(seed=a['seed'],opponent=a['opponent'],side=a['side'],baseline_game=a['game'],
              intervention_game=b['game'] if b else None,eligible=bool(events),delta_own=None,x12=None,
              prefix_verified=False,complete_request_changed=False)
    if not events:
        assert b is None
        return base
    assert b is not None
    bc=read(out/'job-results'/(b['game']+'.audit.json'))
    assert len(bc['opportunities'])==1
    ae,be=events[0],bc['opportunities'][0]
    for name in ('step','window','top1','top2','x12'):
        assert ae[name]==be[name],name
    replay=[]
    for row in (a,b):
        with gzip.open(out/'games'/row['game']/'replay.json.gz','rt',encoding='utf-8') as f:replay.append(json.load(f))
    step=ae['step']
    assert canonical(replay[0]['steps'][:step+1])==canonical(replay[1]['steps'][:step+1]),'Pre-intervention trajectories differ'
    actual_changed=canonical(replay[0]['steps'][step+1][a['side']]['action'])!=canonical(replay[1]['steps'][step+1][a['side']]['action'])
    assert actual_changed==be['complete_request_changed']
    if not actual_changed:
        assert canonical(replay[0]['steps'])==canonical(replay[1]['steps']),'Identical one-shot request diverged later'
    assert a['initial_banks']==b['initial_banks']
    return dict(base,delta_own=b['bank']-a['bank'],x12=ae['x12'],prefix_verified=True,
                complete_request_changed=actual_changed,intervention_step=step,
                delta_opponent=b['opponent_bank']-a['opponent_bank'],delta_margin=b['margin']-a['margin'],
                delta_score=b['score']-a['score'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--key',required=True)
    args=p.parse_args();job(args.out,args.key)
