"""Whole-request interventions through the real official filename entry."""
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

from learning.bc_crop.agent import ROOT,sha
from learning.bc_crop.audit import stable,load
from learning.task_bc_holdout_v1.games import write_new
from learning.task_bc_v1.deployment import JointAgent
from learning.task_choice_v1.agent import TaskAgent
from learning.task_request_gate_v1.experiment import read,verify,gate_identity
from learning.task_request_gate_v1.policy import RequestAgent,RequestModel,same_request,request_digest,_Budget
from decision.audit_market_teacher import observation
from decision.full_audit import reexecute
from decision.semantics import canonical


def actor(out,plan,cfg,trace=None):
    model=RequestModel.load(Path(out)/'gate-model.json',gate_identity(out)) if plan['policy']=='gated' else None
    return RequestAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],
        ROOT/cfg['task_checkpoint'],cfg['task_checkpoint_sha256'],mode=plan['policy'],
        assigned_window=plan['window'],model=model,trace=trace,limits=cfg['game_limits'])


def wrapper(out,plan,cfg):
    out=Path(out);key=plan['game'];path=out/'agents'/(key+'.py');trace=out/'traces'/(key+'.jsonl.gz')
    assert not path.exists() and not trace.exists()
    code=('from pathlib import Path\nimport sys\n'
        '_p=globals().get("__file__",sys._getframe().f_code.co_filename)\n'
        'ROOT=next(p for p in Path(_p).resolve().parents if (p/"learning/task_request_gate_v1/policy.py").is_file())\n'
        'sys.path.insert(0,str(ROOT))\nfrom learning.task_request_gate_v1.policy import RequestAgent,RequestModel\n')
    model=('RequestModel.load(ROOT/'+repr((out/'gate-model.json').relative_to(ROOT).as_posix())+','+repr(gate_identity(out))+')'
        if plan['policy']=='gated' else 'None')
    code+=('_runner=RequestAgent(ROOT/'+repr(cfg['crop_checkpoint'])+','+repr(cfg['crop_checkpoint_sha256'])+
        ',ROOT/'+repr(cfg['task_checkpoint'])+','+repr(cfg['task_checkpoint_sha256'])+
        ',mode='+repr(plan['policy'])+',assigned_window='+repr(plan['window'])+',model='+model+
        ',trace=ROOT/'+repr(trace.relative_to(ROOT).as_posix())+',limits='+repr(cfg['game_limits'])+')\n'
        'def agent(observation,configuration):\n    return _runner(observation,configuration)\nsubmission_agent=agent\n')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(code)
    return path


def model_counts(last,task_network):
    return dict(task_forwards=len(last['records']) if task_network else 0,
        task_rows=sum(len(r['inputs']['candidates']) for r in last['records']) if task_network else 0,
        crop_forwards=len(last['crop_records']),crop_rows=len(last['crop_records']),
        gate_evaluations=0,controller_calls=1,teacher_proposals=0)


def bound_original(controller,task_network,limits):
    """Count the unchanged reference forwards before executing parity work."""
    budget=_Budget(limits)
    if task_network:
        original=controller.task_selector
        def task(inputs):
            budget.forward('task',len(inputs['candidates']))
            return original(inputs)
        controller.task_selector=task
    logits=controller.crop.network.logits
    def crop(x,names):
        budget.forward('crop',1)
        return logits(x,names)
    controller.crop.network.logits=crop
    return budget


def audit(out,row,cfg):
    out=Path(out);side=row['side'];key=row['game']
    rp=out/'games'/key/'replay.json.gz';tp=out/'traces'/(key+'.jsonl.gz')
    assert sha(rp)==row['replay_gzip_sha256'] and sha(tp)==row['trace_sha256']
    with gzip.open(rp,'rt',encoding='utf-8') as f:replay=json.load(f)
    agent=actor(out,row,cfg);rival=load(out/'sources'/(row['opponent']+'.py'))
    original=None
    if row['parity_kind']=='student':
        original=JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],ROOT/cfg['task_checkpoint'],cfg['task_checkpoint_sha256'])
    elif row['parity_kind']=='teacher_task':
        original=TaskAgent(None,'model',ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'])
    else:assert row['parity_kind'] is None
    parity=Counter();opportunities=[]
    parity_budget=bound_original(original,row['parity_kind']=='student',cfg['game_limits']) if original else None
    with gzip.open(tp,'rt',encoding='utf-8') as trace:
        for step,(before,after) in enumerate(zip(replay['steps'],replay['steps'][1:])):
            line=trace.readline();assert line
            stored=json.loads(line);o=observation(before,side);request=agent(o,replay['configuration'])
            assert same_request(request,after[side]['action'])
            assert same_request(rival(observation(before,1-side),replay['configuration']),after[1-side]['action'])
            assert stable(agent.last)==stable(stored),'Reloaded proposal/selection trace differs'
            assert all(agent.last['selected_trace'][k]==0 for k in ('fallbacks','load_errors','inference_errors'))
            if original is not None:
                parity_budget.begin_turn();parity_budget.controller()
                assert same_request(original(o,replay['configuration']),request),'Original controller parity failed'
                parity.update(model_counts(original.last,row['parity_kind']=='student'))
                assert dict(parity)==parity_budget.counts,'Reference forward counters disagree with records'
            for event in agent.last['request_events']:
                proposals=agent.last['proposals']
                assert not same_request(proposals['student'],proposals['teacher'])
                chosen=proposals['teacher'] if event['selected_teacher'] else proposals['student']
                assert same_request(chosen,request)
                assert event['changed_request']==event['selected_teacher']
                opportunities.append(dict(event,
                    student_request_sha256=request_digest(proposals['student']),
                    teacher_request_sha256=request_digest(proposals['teacher'])))
        assert not trace.readline()
    execution=Counter()
    def sink(event):execution[event['phase']+':'+event['status']]+=1
    official=reexecute(replay,trace=True,label_sink=sink)
    assert official['engine_sha256']==cfg['engine_sha256']
    assert official['semantic_sha256']==row['semantic_replay_sha256']
    assert official['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    assert official['final_rewards']==row['banks']==row['rewards']
    counts=agent.last['request_counts'];total=Counter(counts);total.update(counts);total.update(parity)
    assert row['policy']!='teacher_task' or counts['task_forwards']==counts['task_rows']==0
    return dict(game=key,passed=True,official=official,opportunities=opportunities,
        complete_request_changes=sum(e['changed_request'] for e in opportunities),
        live_model_counts=counts,primary_model_counts=counts,original_parity_model_counts=dict(parity),
        total_model_counts=dict(total),policy_turns=2*(counts['controller_calls']+719)+(719 if original else 0),
        original_controller_parity_turns=719 if original else 0,parity_kind=row['parity_kind'],
        replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'],
        actor_actions_reproduced=719,rival_actions_reproduced=719,trace_turns_reproduced=719,
        execution_counts=dict(execution),errors=0)


def job(out,key):
    out=Path(out).resolve();cfg=verify(out);plan=read(out/'job-inputs'/(key+'.json'))
    bot=out/'agents'/(key+'.py');rival=out/'sources'/(plan['opponent']+'.py')
    assert plan['game']==key and sha(bot)==plan['bot_sha256']
    assert sha(rival)==plan['rival_sha256']==cfg['source_sha256'][rival.name]
    if plan['policy']=='gated':assert gate_identity(out)==plan['gate_model_sha256']
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
        with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1) as stream:stream.write((folder/'replay.json').read_bytes())
    side=plan['side'];margin=banks[side]-banks[1-side]
    row=dict(plan,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
        initial_banks=initial,banks=banks,rewards=result['rewards'],bank=banks[side],opponent_bank=banks[1-side],
        initial_bank=initial[side],opponent_initial_bank=initial[1-side],net_profit=banks[side]-initial[side],
        opponent_net_profit=banks[1-side]-initial[1-side],margin=margin,score=1. if margin>0 else .5 if margin==0 else 0.,
        semantic_replay_sha256=result['semantic_replay_sha256'],semantic_replay_sha256_v2=result['semantic_replay_sha256_v2'],
        replay_gzip_sha256=sha(folder/'replay.json.gz'),trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')),
        engine_sha256=cfg['engine_sha256'],
        task_checkpoint_sha256=cfg['task_checkpoint_sha256'] if plan['policy']!='teacher_task' else None,
        crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'])
    write_new(out/'job-results'/(key+'.row.json'),row)
    certificate=audit(out,row,cfg);verify(out)
    write_new(out/'job-results'/(key+'.audit.json'),certificate)
    write_new(out/'job-results'/(key+'.complete.json'),dict(game=key,passed=True,
        row_sha256=sha(out/'job-results'/(key+'.row.json')),certificate_sha256=sha(out/'job-results'/(key+'.audit.json'))))


def pair(out,a,b):
    out=Path(out);ae=read(out/'job-results'/(a['game']+'.audit.json'))['opportunities'];assert len(ae)<=1
    base=dict(seed=a['seed'],opponent=a['opponent'],side=a['side'],baseline_game=a['game'],
        intervention_game=b['game'] if b else None,baseline_net_profit=a['net_profit'],eligible=bool(ae),
        delta_own=None,delta_opponent=None,delta_margin=None,delta_score=None,x8=None,
        prefix_verified=False,complete_request_changed=False)
    if not ae:
        assert b is None
        return base
    assert b is not None
    be=read(out/'job-results'/(b['game']+'.audit.json'))['opportunities'];assert len(be)==1
    ae,be=ae[0],be[0]
    for key in ('step','window','x8','student_request_sha256','teacher_request_sha256'):assert ae[key]==be[key],key
    assert not ae['selected_teacher'] and be['selected_teacher'] and be['changed_request']
    replays=[]
    for row in (a,b):
        with gzip.open(out/'games'/row['game']/'replay.json.gz','rt',encoding='utf-8') as f:replays.append(json.load(f))
    step=ae['step']
    assert canonical(replays[0]['steps'][:step+1])==canonical(replays[1]['steps'][:step+1]),'Prefix diverged before intervention'
    assert not same_request(replays[0]['steps'][step+1][a['side']]['action'],replays[1]['steps'][step+1][a['side']]['action'])
    assert a['initial_banks']==b['initial_banks']
    return dict(base,x8=ae['x8'],prefix_verified=True,complete_request_changed=True,intervention_step=step,
        delta_own=b['bank']-a['bank'],delta_opponent=b['opponent_bank']-a['opponent_bank'],
        delta_margin=b['margin']-a['margin'],delta_score=b['score']-a['score'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--key',required=True)
    a=p.parse_args();job(a.out,a.key)
