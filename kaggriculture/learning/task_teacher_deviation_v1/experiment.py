"""One-shot preregistration for an already-consumed training diagnostic."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-teacher-deviation-v1-20261027'
PARENT=ROOT/'artifacts/task-teacher-choice-eval-v2-20261027'
TRAIN=ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
SOURCE=Path(__file__).resolve().parent
TRAIN_EVIDENCE=('root-result-acceptance.json','registration.json','freeze.json','train-data-result.json','model-seal.json','fit-result.json')
PARENT_EVIDENCE=('root-result-acceptance.json','registration.json','freeze.json','independent-result-assessment.json')
LIMITS=dict(max_archive_bytes=128*1024**2,max_uncompressed_bytes=512*1024**2,max_groups=2253,max_candidates=1200000,
            selected_groups=200,selected_rows=120000,seconds=300)
JOB_LIMITS=dict(max_rss_bytes=1536*1024**2,max_commit_bytes=2*1024**3,min_available_bytes=2*1024**3,
                wall_seconds=360,poll_seconds=.25,cleanup_seconds=10)
PROBE_JOB_LIMITS=dict(max_rss_bytes=512*1024**2,max_commit_bytes=1024**3,min_available_bytes=2*1024**3,
                     wall_seconds=30,poll_seconds=.25,cleanup_seconds=10)
THREAD_LIMITS=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()
def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def deadline_check():
    deadline=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc)>=deadline: raise TimeoutError('Deadline expired or invalid')
def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip()!=expected or git('status','--porcelain').strip(): raise ValueError('Dirty or changed HEAD')
def environment():
    venv=ROOT/'.venv'; executable=venv/'Scripts/python.exe'
    try:
        if not os.path.samefile(sys.executable,executable) or not os.path.samefile(sys.prefix,venv) or os.path.samefile(sys.prefix,sys.base_prefix):
            raise ValueError('Project venv required')
    except OSError as exc: raise ValueError('Venv identity unavailable') from exc
    return dict(python_version=list(sys.version_info[:3]),psutil_version=psutil.__version__,interpreter_sha256=sha(executable))


def check_frozen_dependencies(runtime,paths,hash_file):
    for rel in paths:
        if rel not in runtime or hash_file(ROOT/rel)!=runtime[rel]:
            raise ValueError('Frozen dependency missing or changed')


def prepare(out=OUT):
    out=Path(out).resolve()
    if os.name!='nt' or out!=OUT or any((out/n).exists() for n in ('registration.json','freeze.json','started.json','result.json')):
        raise ValueError('Registration unavailable')
    deadline_check(); decision=read(out/'root-decision.json'); baseline=decision['previous_commit']
    if (decision.get('status')!='approved_for_implementation_not_execution' or decision.get('training_only') is not True
        or decision.get('limits')!=LIMITS or decision.get('job_limits')!=JOB_LIMITS
        or decision.get('probe_job_limits')!=PROBE_JOB_LIMITS or not isinstance(baseline,str)
        or len(baseline)!=40 or any(c not in '0123456789abcdef' for c in baseline)):
        raise ValueError('Invalid decision')
    clean_head(baseline)
    for folder,names in ((PARENT,PARENT_EVIDENCE),(TRAIN,TRAIN_EVIDENCE)):
        for name in names:
            if git('show',baseline+':review/evidence/'+folder.name+'/'+name)!=(folder/name).read_bytes():
                raise ValueError('Published evidence changed')
    prior=read(PARENT/'root-result-acceptance.json'); train=read(TRAIN/'root-result-acceptance.json')
    if not prior['passed'] or not train['passed']: raise ValueError('Parent stages not accepted')
    for folder,result,names in ((PARENT,prior,PARENT_EVIDENCE),(TRAIN,train,TRAIN_EVIDENCE)):
        for name in names[1:]:
            if name in result['evidence_sha256'] and sha(folder/name)!=result['evidence_sha256'][name]: raise ValueError('Evidence chain changed')
    data=read(TRAIN/'train-data-result.json'); seal=read(TRAIN/'model-seal.json'); fit=read(TRAIN/'fit-result.json')
    if (not data['passed'] or data['groups']!=2253 or not seal['passed'] or seal['checkpoint_sha256']!=fit['checkpoint_sha256']
        or seal['fit_result_sha256']!=sha(TRAIN/'fit-result.json') or fit['train_data_sha256']!=sha(TRAIN/'train-data-result.json')):
        raise ValueError('Training seal changed')
    cache=TRAIN/'private/train.npz'; model=TRAIN/'private/model-final.npz'
    if cache.stat().st_size>LIMITS['max_archive_bytes']: raise ValueError('Archive budget exceeded')
    # Only stat archive here. A single immutable, SHA-checked snapshot is made by the worker.
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime=dict(read(PARENT/'freeze.json')['runtime_files']); runtime.update(source)
    check_frozen_dependencies(runtime,('learning/task_teacher_choice_bc_v1/numeric.py','learning/task_bc_v1/model.py',
                'learning/task_job_sampling_race_v1/sampling.py','learning/task_training_job_supervision_v1/job.py'),sha)
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    parent_cfg=read(PARENT/'registration.json')
    old_rel=parent_cfg['baseline_checkpoint']; old_sha=parent_cfg['baseline_checkpoint_sha256']
    if not isinstance(old_rel,str) or not (ROOT/old_rel).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Old model path outside root')
    cfg=dict(experiment='task-teacher-deviation-v1',baseline_head=baseline,limits=LIMITS,job_limits=JOB_LIMITS,
        probe_job_limits=PROBE_JOB_LIMITS,thread_limits=THREAD_LIMITS,
        preflight='One synthetic venv Job membership probe; no training or model input',
        train_seeds=read(TRAIN/'registration.json')['train_seeds'],cache_sha256=data['array_sha256'],cache_bytes=cache.stat().st_size,
        checkpoint_sha256=seal['checkpoint_sha256'],old_checkpoint=old_rel,old_checkpoint_sha256=old_sha,
        environment=environment(),training_only=True,
        parent_evidence={n:sha(PARENT/n) for n in PARENT_EVIDENCE},train_evidence={n:sha(TRAIN/n) for n in TRAIN_EVIDENCE},
        interpretation='Descriptive training-set comparison, not independent holdout or causal evidence',retry_allowed=False)
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        runtime_files=runtime,new_source_sha256=source,baseline_head=baseline))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT: raise ValueError('Unexpected output')
    deadline_check(); cfg=read(out/'registration.json'); freeze=read(out/'freeze.json')
    receipt=out/'pre-execution-commit.json'; clean_head(read(receipt)['commit'] if receipt.exists() else cfg['baseline_head'])
    if sha(out/'registration.json')!=freeze['registration_sha256'] or sha(out/'root-decision.json')!=freeze['root_decision_sha256']:
        raise ValueError('Registration changed')
    if (cfg['limits']!=LIMITS or cfg['job_limits']!=JOB_LIMITS or cfg['probe_job_limits']!=PROBE_JOB_LIMITS
        or cfg['thread_limits']!=THREAD_LIMITS or cfg['environment']!=environment() or cfg['training_only'] is not True):
        raise ValueError('Limits or environment changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}!=set(freeze['new_source_sha256']): raise ValueError('Source set changed')
    for rel,digest in freeze['runtime_files'].items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    for folder,key in ((PARENT,'parent_evidence'),(TRAIN,'train_evidence')):
        for name,digest in cfg[key].items():
            if sha(folder/name)!=digest: raise ValueError('Evidence changed')
    if (TRAIN/'private/train.npz').stat().st_size!=cfg['cache_bytes']: raise ValueError('Cache size changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out); commit=read(out/'pre-execution-commit.json')['commit']
    clean_head(commit)
    if git('merge-base','--is-ancestor',cfg['baseline_head'],commit)!=b'': raise ValueError('Invalid ancestry')
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes(): raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg


def ready(out=OUT):
    cfg=verify(out); review=read(Path(out)/'pre-review.json'); receipt=read(Path(out)/'pre-execution-commit.json')
    if (not review['passed'] or review['freeze_sha256']!=sha(Path(out)/'freeze.json')
        or receipt['freeze_sha256']!=sha(Path(out)/'freeze.json') or receipt['registration_sha256']!=sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    verify_commit(out); return cfg
