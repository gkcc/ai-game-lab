"""Scoped teacher-task official worker adapter for eight development games."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import threading
from learning.task_anchor_once_v1 import games as core
from . import experiment

_LOCK=threading.RLock()
_PLAN=frozenset(('game','policy','split','seed','opponent','side','parity_kind'))
_JOB=_PLAN|frozenset(('bot_sha256','rival_sha256','task_checkpoint_sha256'))


def digest(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def plans(cfg):
    seeds=cfg['train_seeds']; reserved=cfg['reserved_test_seeds']
    if (not isinstance(seeds,list) or len(seeds)!=2 or any(type(s) is not int or not 0<=s<2**32 for s in seeds)
        or len(set(seeds))!=2 or not isinstance(reserved,list) or not 2<=len(reserved)<=4
        or any(type(s) is not int or not 0<=s<2**32 for s in reserved) or len(set(reserved))!=len(reserved)
        or set(seeds)&set(reserved) or cfg['opponents']!=['round2','market'] or cfg['seats']!=[0,1]):
        raise ValueError('Invalid frozen development panel')
    return [dict(game=f'dev-{seed}-{opponent}-{side}-teacher_task',policy='teacher_task',split='dev',
        seed=seed,opponent=opponent,side=side,parity_kind=None)
        for seed in seeds for opponent in ('round2','market') for side in (0,1)]


def task_identity(out,plan,cfg):
    if plan.get('policy')!='teacher_task': raise ValueError('Only teacher-task allowed')
    return None,None


def validate_plan(plan,cfg,job=False):
    if not isinstance(plan,dict) or set(plan)!=(_JOB if job else _PLAN): raise ValueError('Invalid plan fields')
    if type(plan['seed']) is not int or type(plan['side']) is not int: raise ValueError('Invalid seed or seat')
    if {key:plan[key] for key in _PLAN} not in plans(cfg): raise ValueError('Plan outside registered panel')
    if job and (plan['task_checkpoint_sha256'] is not None or not digest(plan['bot_sha256'])
        or not digest(plan['rival_sha256']) or plan['rival_sha256']!=cfg['source_sha256'][plan['opponent']+'.py']):
        raise ValueError('Invalid job identities')


def validate_receipts(plan,row,audit,complete,storage,row_sha,audit_sha,cfg):
    """Caller supplies file hashes and must separately hash replay/trace and model."""
    validate_plan(plan,cfg,job=True)
    if not digest(row_sha) or not digest(audit_sha): raise ValueError('Invalid receipt hashes')
    if (any(row.get(k)!=v for k,v in plan.items()) or row.get('passed') is not True
        or row.get('states')!=720 or row.get('action_turns')!=719 or row.get('status')!=['DONE','DONE']
        or row.get('agent_errors')!=0 or row.get('engine_sha256')!=cfg['engine_sha256']
        or row.get('crop_checkpoint_sha256')!=cfg['crop_checkpoint_sha256']
        or any(not digest(row.get(k)) for k in ('replay_gzip_sha256','trace_sha256','semantic_replay_sha256','semantic_replay_sha256_v2'))):
        raise ValueError('Official row invalid')
    official=audit.get('official',{}); live=audit.get('live_model_counts'); primary=audit.get('primary_model_counts')
    fields={'task_forwards','task_rows','crop_forwards','crop_rows','gate_evaluations','controller_calls','teacher_proposals'}
    limits=cfg['game_limits']
    if (not isinstance(live,dict) or not isinstance(primary,dict) or set(live)!=fields or live!=primary
        or any(type(v) is not int or v<0 for v in live.values()) or live['controller_calls']!=719
        or live['controller_calls']>limits['controller_calls'] or live['task_forwards']!=0 or live['task_rows']!=0
        or live['crop_forwards']>limits['crop_forwards'] or live['crop_rows']<live['crop_forwards']
        or live['crop_rows']>limits['neural_input_rows'] or live['gate_evaluations']>limits['gate_evaluations']
        or live['teacher_proposals']>limits['teacher_proposals']): raise ValueError('Teacher execution counts invalid')
    if (audit.get('passed') is not True or audit.get('game')!=plan['game'] or audit.get('errors')!=0
        or audit.get('actor_actions_reproduced')!=719 or audit.get('rival_actions_reproduced')!=719
        or audit.get('trace_turns_reproduced')!=719 or audit.get('replay_sha256')!=row['replay_gzip_sha256']
        or audit.get('trace_sha256')!=row['trace_sha256'] or official.get('passed') is not True
        or official.get('official_transitions_reexecuted')!=719
        or official.get('engine_sha256')!=cfg['engine_sha256'] or official.get('semantic_sha256')!=row['semantic_replay_sha256']
        or official.get('semantic_sha256_v2')!=row['semantic_replay_sha256_v2'] or official.get('final_rewards')!=row.get('rewards')
        or official.get('both_public_and_private_states_match_each_step') is not True
        or official.get('inventory_order_checked_each_step') is not True or complete.get('passed') is not True
        or complete.get('game')!=plan['game'] or complete.get('row_sha256')!=row_sha or complete.get('certificate_sha256')!=audit_sha
        or storage.get('passed') is not True or storage.get('error_kind') is not None or storage.get('output',{}).get('refused')!=0):
        raise ValueError('Official audit receipt invalid')


def _unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate plan field')
        result[key]=value
    return result


@contextmanager
def bindings(job_path=None,expected=None,cfg=None):
    with _LOCK:
        saved=core.verify,core.task_identity,core.read
        try:
            def verify(out):
                value=dict(experiment.verify(out)); value['validation_seeds']=list(value['train_seeds']); return value
            core.verify,core.task_identity=verify,task_identity
            if job_path is not None:
                def checked_read(path):
                    if Path(path).resolve()!=job_path: return saved[2](path)
                    with Path(path).open(encoding='utf-8-sig') as stream:
                        value=json.load(stream,object_pairs_hook=_unique_pairs)
                    validate_plan(value,cfg,job=True)
                    if value!=expected: raise ValueError('Job input changed')
                    return value
                core.read=checked_read
            yield
        finally: core.verify,core.task_identity,core.read=saved


def make_wrapper(out,plan,cfg):
    validate_plan(plan,cfg)
    with bindings(): return core.wrapper(out,plan,cfg)


def job(out,key):
    if not __debug__: raise RuntimeError('Optimized Python execution is not permitted')
    out=Path(out).resolve(); cfg=experiment.ready(out); experiment.verify_commit(out); experiment.deadline_check()
    if not isinstance(key,str) or Path(key).name!=key: raise ValueError('Invalid game key')
    path=(out/'job-inputs'/(key+'.json')).resolve()
    with path.open(encoding='utf-8-sig') as stream: plan=json.load(stream,object_pairs_hook=_unique_pairs)
    validate_plan(plan,cfg,job=True)
    if plan['game']!=key: raise ValueError('Job key mismatch')
    with bindings(path,plan,cfg): return core.job(out,key)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--key',required=True)
    args=parser.parse_args(); job(args.out,args.key)
