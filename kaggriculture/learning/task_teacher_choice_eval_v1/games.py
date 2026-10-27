"""Scoped official worker adapter for two fixed neural task policies."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import threading
from learning.task_anchor_once_v1 import games as core
from . import experiment
from .evaluation import plans

_LOCK = threading.RLock()
_PLAN = frozenset(('game','policy','split','seed','opponent','side','parity_kind'))
_JOB = _PLAN | frozenset(('bot_sha256','rival_sha256','task_checkpoint_sha256'))


def _digest(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def task_identity(out,plan,cfg):
    policy=plan.get('policy')
    if policy not in ('candidate','baseline'): raise ValueError('Unsupported policy')
    rel,digest = cfg[policy+'_checkpoint'],cfg[policy+'_checkpoint_sha256']
    if (not isinstance(rel,str) or not rel or '\\' in rel or not _digest(digest)
        or not _digest(cfg['candidate_checkpoint_sha256']) or not _digest(cfg['baseline_checkpoint_sha256'])
        or cfg['candidate_checkpoint_sha256'] == cfg['baseline_checkpoint_sha256']):
        raise ValueError('Invalid or identical checkpoint identities')
    posix=PurePosixPath(rel)
    if posix.is_absolute() or posix.as_posix()!=rel or any(part in ('.','..') or ':' in part for part in posix.parts):
        raise ValueError('Invalid checkpoint path')
    root=experiment.ROOT.resolve(); path=(root/rel).resolve()
    if not path.is_relative_to(root) or path==root: raise ValueError('Checkpoint outside root')
    return path,digest


def validate_plan(plan,cfg,job=False):
    if not isinstance(plan,dict) or set(plan) != (_JOB if job else _PLAN): raise ValueError('Invalid plan fields')
    if cfg['opponents'] != ['round2'] or cfg['seats'] != [0,1]: raise ValueError('Invalid panel')
    allowed=plans(cfg['test_seeds'])
    base={k:plan[k] for k in _PLAN}
    if base not in allowed: raise ValueError('Plan outside fixed panel')
    _,digest=task_identity(None,plan,cfg)
    if job and (plan['task_checkpoint_sha256'] != digest or not _digest(plan['bot_sha256']) or not _digest(plan['rival_sha256'])):
        raise ValueError('Job identity mismatch')


def validate_receipts(plan,row,certificate,complete,row_sha,certificate_sha,cfg):
    """Check worker's aggregate receipts; caller must hash files and bind ledger."""
    validate_plan(plan,cfg,job=True)
    if (any(row.get(k)!=v for k,v in plan.items()) or row.get('passed') is not True
        or row.get('states')!=720 or row.get('action_turns')!=719 or row.get('status')!=['DONE','DONE']
        or row.get('agent_errors')!=0 or row.get('engine_sha256')!=cfg['engine_sha256']
        or row.get('crop_checkpoint_sha256')!=cfg['crop_checkpoint_sha256']
        or not _digest(row.get('replay_gzip_sha256')) or not _digest(row.get('trace_sha256'))
        or not _digest(row.get('semantic_replay_sha256')) or not _digest(row.get('semantic_replay_sha256_v2'))):
        raise ValueError('Official game row mismatch')
    official=certificate.get('official',{})
    live,primary=certificate.get('live_model_counts'),certificate.get('primary_model_counts')
    fields={'task_forwards','task_rows','crop_forwards','crop_rows','gate_evaluations','controller_calls','teacher_proposals'}
    limits=cfg['game_limits']
    if (not isinstance(live,dict) or not isinstance(primary,dict) or set(live)!=fields or set(primary)!=fields
        or live != primary or any(type(v) is not int or v < 0 for v in live.values())
        or live['controller_calls'] != 719 or live['controller_calls'] > limits['controller_calls']
        or not 0 < live['task_forwards'] <= limits['task_forwards'] or live['task_rows'] < live['task_forwards']
        or live['crop_forwards'] > limits['crop_forwards'] or live['crop_rows'] < live['crop_forwards']
        or live['task_rows']+live['crop_rows'] > limits['neural_input_rows']
        or live['gate_evaluations'] > limits['gate_evaluations'] or live['teacher_proposals'] > limits['teacher_proposals']):
        raise ValueError('Model execution counts invalid')
    if (certificate.get('passed') is not True or certificate.get('game')!=plan['game'] or certificate.get('errors')!=0
        or certificate.get('actor_actions_reproduced')!=719 or certificate.get('rival_actions_reproduced')!=719
        or certificate.get('trace_turns_reproduced')!=719 or certificate.get('replay_sha256')!=row['replay_gzip_sha256']
        or certificate.get('trace_sha256')!=row['trace_sha256'] or official.get('official_transitions_reexecuted')!=719
        or official.get('engine_sha256')!=cfg['engine_sha256'] or official.get('semantic_sha256')!=row['semantic_replay_sha256']
        or official.get('semantic_sha256_v2')!=row['semantic_replay_sha256_v2'] or official.get('final_rewards')!=row.get('rewards')
        or complete.get('passed') is not True or complete.get('game')!=plan['game']
        or complete.get('row_sha256')!=row_sha or complete.get('certificate_sha256')!=certificate_sha):
        raise ValueError('Official audit receipt mismatch')


def unique_pairs(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError('Duplicate plan field')
        result[k]=v
    return result


@contextmanager
def bindings(job_path=None,expected=None,cfg=None):
    with _LOCK:
        saved=core.verify,core.task_identity,core.read
        try:
            core.verify,core.task_identity=experiment.verify,task_identity
            if job_path is not None:
                def checked_read(path):
                    if Path(path).resolve()!=job_path: return saved[2](path)
                    with Path(path).open(encoding='utf-8-sig') as stream:
                        value=json.load(stream,object_pairs_hook=unique_pairs)
                    validate_plan(value,cfg,job=True)
                    if value!=expected: raise ValueError('Job plan changed before worker read')
                    return value
                core.read=checked_read
            yield
        finally:
            core.verify,core.task_identity,core.read=saved


def make_wrapper(out,plan,cfg):
    validate_plan(plan,cfg)
    with bindings(): return core.wrapper(out,plan,cfg)


def job(out,key):
    out=Path(out).resolve(); cfg=experiment.ready(out); experiment.verify_commit(out); experiment.deadline_check()
    if not isinstance(key,str) or Path(key).name!=key: raise ValueError('Invalid game key')
    path=(out/'job-inputs'/(key+'.json')).resolve()
    with path.open(encoding='utf-8-sig') as stream: plan=json.load(stream,object_pairs_hook=unique_pairs)
    validate_plan(plan,cfg,job=True)
    if plan['game']!=key: raise ValueError('Game key mismatch')
    with bindings(path,plan,cfg): return core.job(out,key)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--key',required=True)
    args=parser.parse_args(); job(args.out,args.key)
