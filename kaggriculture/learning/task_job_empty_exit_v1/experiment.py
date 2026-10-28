"""One-shot registration and execution gate for an artificial exit timing test."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from . import scenario

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
SOURCE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/task-job-empty-exit-v1-20261028'
PARENT=ROOT/'artifacts/task-teacher-weighted-eval-v1-20261027'
BASE='5eebafaa4ca481f1529fdde0d731ebd4e3c30c0d'
SHARED=('learning/task_job_sampling_race_v1/sampling.py','learning/task_training_job_supervision_v1/job.py')
EVIDENCE=('root-failure-acceptance.json','pre-execution-commit.json','freeze.json')
PREFLIGHT=dict(samples=3,min_available_bytes=4*1024**3,interval_seconds=10)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip()!=expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def deadline_check():
    deadline=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc)>=deadline: raise TimeoutError('Expired deadline')


def environment():
    exe=ROOT/'.venv/Scripts/python.exe'; venv=ROOT/'.venv'
    try:
        if (os.name!='nt' or sys.flags.optimize!=0 or 'PYTHONOPTIMIZE' in os.environ or not os.path.samefile(sys.executable,exe)
            or not os.path.samefile(sys.prefix,venv) or os.path.samefile(sys.prefix,sys.base_prefix)):
            raise ValueError('Project Windows venv and non-optimized Python required')
    except OSError as exc: raise ValueError('Environment identity unavailable') from exc
    return dict(python=list(sys.version_info[:3]),psutil=psutil.__version__,interpreter_sha256=sha(exe),optimized=False)


def check_parent():
    for name in EVIDENCE:
        if git('show',BASE+':review/evidence/'+PARENT.name+'/'+name)!=(PARENT/name).read_bytes():
            raise ValueError('Published parent evidence differs')
    result=read(PARENT/EVIDENCE[0]); previous=read(PARENT/'pre-execution-commit.json')
    if (result.get('passed') is not True or result.get('error_kind')!='EmptySnapshot'
        or result.get('partial_panel_scored') is not False or result.get('retry_or_replacement') is not False
        or result.get('pre_execution_commit')!=previous.get('commit') or result.get('freeze_sha256')!=sha(PARENT/'freeze.json')):
        raise ValueError('Parent failure evidence inconsistent')
    return {name:sha(PARENT/name) for name in EVIDENCE}


def prepare(out=OUT):
    out=Path(out).resolve()
    if out!=OUT or os.name!='nt' or any((out/n).exists() for n in ('registration.json','freeze.json','started.json','result.json')):
        raise ValueError('Invalid registration state')
    deadline_check(); decision=read(out/'root-decision.json')
    if (decision.get('baseline_head')!=BASE or decision.get('synthetic_only') is not True
        or decision.get('limits')!=scenario.LIMITS or decision.get('host_preflight')!=PREFLIGHT):
        raise ValueError('Invalid registration decision')
    clean_head(BASE); parent=check_parent(); env=environment()
    new={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    shared={rel:sha(ROOT/rel) for rel in SHARED}
    for rel in SHARED:
        if git('show',BASE+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Shared source changed')
    cfg=dict(baseline_head=BASE,synthetic_only=True,limits=scenario.LIMITS,host_preflight=PREFLIGHT,
        parent_evidence=parent,environment=env,attempts=1,retry_allowed=False,
        workload=dict(python_sleep_seconds=2,artificial_wait_ms=scenario.INJECTION_WAIT_MS,network=False),
        limitations='Artificial exit timing; five second wait pauses sampling; no hard memory cap or historical cause claim')
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        new_source_sha256=new,shared_source_sha256=shared))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT: raise ValueError('Unexpected output')
    deadline_check(); cfg=read(out/'registration.json'); freeze=read(out/'freeze.json')
    receipt=out/'pre-execution-commit.json'
    clean_head(read(receipt)['commit'] if receipt.exists() else BASE)
    if (cfg['baseline_head']!=BASE or cfg['synthetic_only'] is not True or cfg['limits']!=scenario.LIMITS
        or cfg['host_preflight']!=PREFLIGHT or cfg['environment']!=environment()
        or type(cfg.get('attempts')) is not int or cfg['attempts']!=1 or cfg.get('retry_allowed') is not False
        or cfg['workload']!=dict(python_sleep_seconds=2,artificial_wait_ms=scenario.INJECTION_WAIT_MS,network=False)
        or sha(out/'registration.json')!=freeze['registration_sha256'] or sha(out/'root-decision.json')!=freeze['root_decision_sha256']):
        raise ValueError('Frozen settings changed')
    if set(freeze['new_source_sha256'])!={p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} or set(freeze['shared_source_sha256'])!=set(SHARED):
        raise ValueError('Frozen source set changed')
    for rel,digest in {**freeze['new_source_sha256'],**freeze['shared_source_sha256']}.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Frozen source changed')
    if check_parent()!=cfg['parent_evidence']: raise ValueError('Parent evidence changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out); receipt=read(out/'pre-execution-commit.json'); commit=receipt['commit']
    clean_head(commit)
    git('merge-base','--is-ancestor',BASE,commit)
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes(): raise ValueError('Committed evidence changed')
    freeze=read(out/'freeze.json')
    for rel in (*freeze['new_source_sha256'],*freeze['shared_source_sha256']):
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg


def ready(out=OUT):
    cfg=verify_commit(out); review=read(Path(out)/'pre-review.json'); receipt=read(Path(out)/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('registration_sha256')!=sha(Path(out)/'registration.json')): raise ValueError('Review identity changed')
    return cfg


def memory_ready(cfg,available=lambda:psutil.virtual_memory().available,sleep=time.sleep):
    limits=cfg['host_preflight']
    if limits!=PREFLIGHT: raise ValueError('Preflight settings changed')
    for i in range(limits['samples']):
        deadline_check(); value=available()
        if type(value) is not int or value<limits['min_available_bytes']: return False
        if i+1<limits['samples']: sleep(limits['interval_seconds'])
    return True


def run(out=OUT):
    from learning.task_training_job_supervision_v1.job import WindowsAPI
    out=Path(out).resolve(); cfg=ready(out)
    if any((out/name).exists() for name in ('started.json','result.json','result.local.json')): raise ValueError('Attempt consumed')
    if not memory_ready(cfg): raise RuntimeError('Insufficient preflight memory')
    ready(out)
    marker=dict(created=False,complete=False); result=None; error=None; commit=None
    names=('started.json','freeze.json','registration.json','pre-execution-commit.json')
    identities={name:None for name in names}
    try:
        path=out/'started.json'
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created']=True
            json.dump(dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),stream); stream.write('\n')
        marker['complete']=True
        for name in names: identities[name]=sha(out/name)
        commit=read(out/'pre-execution-commit.json')['commit']
        result=scenario.run_owned(WindowsAPI(),sys.executable,ROOT,preflight=lambda:True)
        if (result.get('passed') is not True or not scenario.accepted(result.get('supervision',{}),result.get('artificial_exit_timing_arrangements'))):
            raise RuntimeError('Synthetic test failed')
        ready(out)
        if any(sha(out/name)!=digest for name,digest in identities.items()) or read(out/'pre-execution-commit.json')['commit']!=commit:
            raise ValueError('Execution identity changed')
    except BaseException as exc: error=type(exc).__name__
    if marker['created']:
        local=dict(passed=error is None,marker=marker,error_kind=error,supervision=result,identities=identities,retry_allowed=False)
        supervision=result.get('supervision') if isinstance(result,dict) else None
        supervision=supervision if isinstance(supervision,dict) else {}
        observed={name:supervision.get(name) for name in ('rss_sampling_attempts','rss_sampling_retries',
            'rss_complete_samples','rss_discarded_samples','confirmed_empty_exits','launcher_reaped',
            'job_members_zero','exit_code','error_kind','cleanup_error_kind','stop_reason')}
        summary=dict(passed=error is None,error_kind=error,marker_complete=marker['complete'],retry_allowed=False,
            identities=identities,pre_execution_commit=commit,observed=observed,
            synthetic_only=True,artificial_timing=True,identity_check_limit='Transient replacement followed by restoration is not excluded',
            sampling_limit='Artificial wait pauses sampling for up to five seconds; no hard memory cap')
        try:
            write_new(out/'result.local.json',local); write_new(out/'result.json',summary)
        except BaseException as exc: raise RuntimeError('Receipt failure: '+str(error)+'/'+type(exc).__name__) from None
    if error: raise RuntimeError('Synthetic attempt failed: '+error)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','verify','ready','verify_commit','run'))
    args=parser.parse_args(); result=globals()[args.command]()
    print(json.dumps(dict(passed=result.get('passed',True))))
