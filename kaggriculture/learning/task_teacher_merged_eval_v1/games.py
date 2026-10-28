"""Official worker adapter for the fixed three-arm, three-opponent panel.

The plan and worker receipts stay local until seed disclosure. Python-level
worker stdout/stderr are discarded; the launcher must separately account for
native/OS handles and must not treat their behavior as independently verified.
"""
import os

if __name__=='__main__':
    # Keep imports, argument parsing and worker exceptions inside the same
    # Python-output boundary; the interpreter must not print a traceback after
    # job() restores its own streams. Native handles are not covered here.
    try:
        with open(os.devnull,'w',encoding='utf-8') as sink:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(sink), redirect_stderr(sink):
                from learning.task_teacher_merged_eval_v1.games import main
                main()
    except BaseException:
        raise SystemExit(1) from None
    raise SystemExit(0)

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
from pathlib import Path, PurePosixPath
import threading
from learning.task_anchor_once_v1 import games as core
from . import experiment
from .evaluation import plans

_LOCK = threading.RLock()
_PLAN = frozenset(('game','seed_alias','seed','opponent','side','policy','split','parity_kind'))
_JOB = _PLAN | frozenset(('bot_sha256','rival_sha256','task_checkpoint_sha256'))
NATIVE_OUTPUT_SCOPE = 'Python stdout/stderr discarded; native handles not independently verified'


def digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def task_identity(out, plan, cfg):
    policy=plan.get('policy')
    if policy=='teacher_task': return None,None
    if policy not in ('merged','unweighted'): raise ValueError('Unsupported model')
    relative,expected=cfg[policy+'_checkpoint'],cfg[policy+'_checkpoint_sha256']
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative or not digest(expected):
        raise ValueError('Invalid checkpoint identity')
    path=PurePosixPath(relative)
    if path.is_absolute() or path.as_posix()!=relative or any(part in ('.','..') for part in path.parts):
        raise ValueError('Invalid checkpoint path')
    root=experiment.ROOT.resolve(); target=(root/relative).resolve()
    if not target.is_relative_to(root) or target==root: raise ValueError('Checkpoint outside project')
    return target,expected


def validate_plan(plan, cfg, job=False, out=None):
    if not isinstance(plan,dict) or set(plan)!=(_JOB if job else _PLAN): raise ValueError('Invalid plan fields')
    if type(plan['seed']) is not int or type(plan['side']) is not int: raise ValueError('Invalid seed or seat')
    directory=experiment.OUT if out is None else Path(out)
    expected=plans(experiment.load_seeds(directory,cfg))
    if {key:plan[key] for key in _PLAN} not in expected: raise ValueError('Plan outside registered panel')
    _,task_sha=task_identity(directory,plan,cfg)
    if job and (plan['task_checkpoint_sha256']!=task_sha or not digest(plan['bot_sha256'])
        or not digest(plan['rival_sha256']) or plan['rival_sha256']!=cfg['source_sha256'][plan['opponent']+'.py']):
        raise ValueError('Invalid game identities')


def validate_receipts(plan,row,audit,complete,storage,row_sha,audit_sha,cfg,out=None):
    """Check official aggregates; caller separately hashes local artifacts/models."""
    validate_plan(plan,cfg,job=True,out=out)
    if not digest(row_sha) or not digest(audit_sha): raise ValueError('Invalid receipt hash')
    if (any(row.get(key)!=value for key,value in plan.items()) or row.get('passed') is not True
        or type(row.get('states')) is not int or row['states']!=720
        or type(row.get('action_turns')) is not int or row['action_turns']!=719
        or row.get('status')!=['DONE','DONE'] or type(row.get('agent_errors')) is not int or row['agent_errors']!=0
        or row.get('engine_sha256')!=cfg['engine_sha256'] or row.get('crop_checkpoint_sha256')!=cfg['crop_checkpoint_sha256']
        or any(not digest(row.get(key)) for key in ('replay_gzip_sha256','trace_sha256','semantic_replay_sha256','semantic_replay_sha256_v2'))):
        raise ValueError('Official game row invalid')
    official=audit.get('official',{}); live=audit.get('live_model_counts'); primary=audit.get('primary_model_counts')
    fields={'task_forwards','task_rows','crop_forwards','crop_rows','gate_evaluations','controller_calls','teacher_proposals'}
    limits=cfg['game_limits']
    if (not isinstance(live,dict) or not isinstance(primary,dict) or set(live)!=fields or live!=primary
        or any(type(v) is not int or v<0 for v in live.values()) or live['controller_calls']!=719
        or live['controller_calls']>limits['controller_calls'] or live['task_forwards']>limits['task_forwards']
        or live['task_rows']<live['task_forwards'] or live['crop_forwards']>limits['crop_forwards']
        or live['crop_rows']<live['crop_forwards'] or live['task_rows']+live['crop_rows']>limits['neural_input_rows']
        or live['gate_evaluations']>limits['gate_evaluations'] or live['teacher_proposals']>limits['teacher_proposals']):
        raise ValueError('Model execution counts invalid')
    if (plan['policy']=='teacher_task' and (live['task_forwards']!=0 or live['task_rows']!=0)):
        raise ValueError('Teacher unexpectedly used task model')
    if plan['policy']!='teacher_task' and live['task_forwards']==0:
        raise ValueError('Task model was never called')
    budget=storage.get('output'); registered_limit=cfg['storage']['game_write_bytes']
    budget_fields=('limit_bytes','used_bytes','remaining_bytes','created_count','refused')
    if (type(registered_limit) is not int or registered_limit<0 or not isinstance(budget,dict)
        or any(type(budget.get(name)) is not int or budget[name]<0 for name in budget_fields)
        or budget['limit_bytes']!=registered_limit or budget['used_bytes']>registered_limit
        or budget['remaining_bytes']!=registered_limit-budget['used_bytes'] or budget['refused']!=0):
        raise ValueError('Storage budget receipt invalid')
    if (audit.get('passed') is not True or audit.get('game')!=plan['game'] or audit.get('errors')!=0
        or audit.get('actor_actions_reproduced')!=719 or audit.get('rival_actions_reproduced')!=719
        or audit.get('trace_turns_reproduced')!=719 or audit.get('replay_sha256')!=row['replay_gzip_sha256']
        or audit.get('trace_sha256')!=row['trace_sha256'] or official.get('passed') is not True
        or official.get('official_transitions_reexecuted')!=719 or official.get('engine_sha256')!=cfg['engine_sha256']
        or official.get('semantic_sha256')!=row['semantic_replay_sha256']
        or official.get('semantic_sha256_v2')!=row['semantic_replay_sha256_v2']
        or official.get('final_rewards')!=row.get('rewards')
        or official.get('both_public_and_private_states_match_each_step') is not True
        or official.get('inventory_order_checked_each_step') is not True
        or complete.get('passed') is not True or complete.get('game')!=plan['game']
        or complete.get('row_sha256')!=row_sha or complete.get('certificate_sha256')!=audit_sha
        or storage.get('game')!=plan['game'] or storage.get('native_output_accounted_by')!=NATIVE_OUTPUT_SCOPE
        or storage.get('passed') is not True or storage.get('error_kind') is not None):
        raise ValueError('Official audit receipt invalid')


def _unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate plan field')
        result[key]=value
    return result


@contextmanager
def bindings(job_path=None,expected=None,cfg=None,out=None):
    with _LOCK:
        saved=core.verify,core.task_identity,core.read,core.write_new
        try:
            def verified(directory):
                value=dict(experiment.verify(directory))
                value['test_seeds']=list(experiment.load_seeds(directory,value))
                return value
            core.verify,core.task_identity=verified,task_identity
            if job_path is not None:
                if expected is None or cfg is None or out is None:
                    raise ValueError('Incomplete job binding')
                validate_plan(expected,cfg,job=True,out=out)
                root=Path(out).resolve()
                job_path=Path(job_path).resolve()
                if job_path!=(root/'job-inputs'/(expected['game']+'.json')).resolve():
                    raise ValueError('Job binding path mismatch')
                storage_path=(root/'job-results'/(expected['game']+'.storage.json')).resolve()
                def checked_read(path):
                    if Path(path).resolve()!=job_path: return saved[2](path)
                    with Path(path).open(encoding='utf-8-sig') as stream:
                        value=json.load(stream,object_pairs_hook=_unique_pairs)
                    validate_plan(value,cfg,job=True,out=out)
                    if value!=expected: raise ValueError('Job input changed')
                    return value
                def checked_write(path,value):
                    target=Path(path).resolve()
                    if target==storage_path:
                        if (not isinstance(value,dict) or value.get('game')!=expected['game']
                            or value.get('native_output_accounted_by')!='bounded parent process log'):
                            raise ValueError('Unexpected storage receipt')
                        value={**value,'native_output_accounted_by':NATIVE_OUTPUT_SCOPE}
                    elif target.parent==storage_path.parent and target.name.endswith('.storage.json'):
                        raise ValueError('Storage receipt outside job binding')
                    return saved[3](path,value)
                core.read=checked_read
                core.write_new=checked_write
            yield
        finally:
            core.verify,core.task_identity,core.read,core.write_new=saved


def make_wrapper(out,plan,cfg):
    validate_plan(plan,cfg,out=out)
    with bindings(): return core.wrapper(out,plan,cfg)


def job(out,key):
    if not __debug__: raise RuntimeError('Optimized Python execution is not permitted')
    # run_match.main prints the complete result, including the private seed.
    # Discard Python output without an unbounded in-memory capture.
    with open(os.devnull,'w',encoding='utf-8') as sink, redirect_stdout(sink), redirect_stderr(sink):
        directory=Path(out).resolve(); cfg=experiment.ready(directory); experiment.verify_commit(directory); experiment.deadline_check()
        if not isinstance(key,str) or Path(key).name!=key: raise ValueError('Invalid game key')
        path=(directory/'job-inputs'/(key+'.json')).resolve()
        with path.open(encoding='utf-8-sig') as stream: plan=json.load(stream,object_pairs_hook=_unique_pairs)
        validate_plan(plan,cfg,job=True,out=directory)
        if plan['game']!=key: raise ValueError('Job key mismatch')
        with bindings(path,plan,cfg,directory): return core.job(directory,key)


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--key',required=True)
    args=parser.parse_args(argv); job(args.out,args.key)
