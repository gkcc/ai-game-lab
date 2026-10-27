"""Preregister one development-trajectory diagnostic without opening traces."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[2]; PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-teacher-dev-deviation-v1-20261027'
PARENT=ROOT/'artifacts/task-teacher-deviation-v1-20261027'
TRAIN=ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
ACTION=ROOT/'artifacts/task-action-observation-v1-20261027'
SOURCE=Path(__file__).resolve().parent
GAMES=('dev-3911611436-market-0-teacher_task','dev-2328454240-market-0-teacher_task')
LIMITS=dict(compressed_bytes=32*1024**2,decoded_bytes=1024**3,max_line_bytes=16*1024**2,
    max_candidates_per_group=4096,max_retained_bytes_per_game=128*1024**2,selected_groups=200,
    selected_rows=120000,seconds=300)
JOB_LIMITS=dict(max_rss_bytes=1536*1024**2,max_commit_bytes=2*1024**3,min_available_bytes=2*1024**3,
                wall_seconds=360,poll_seconds=.25,cleanup_seconds=10)
PROBE_JOB_LIMITS=dict(max_rss_bytes=512*1024**2,max_commit_bytes=1024**3,min_available_bytes=2*1024**3,
                     wall_seconds=30,poll_seconds=.25,cleanup_seconds=10)
THREAD_LIMITS=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
PARENT_EVIDENCE=('root-result-acceptance.json','registration.json','freeze.json','independent-result-assessment.json')
TRAIN_EVIDENCE=('root-result-acceptance.json','registration.json','freeze.json','source-manifest.json','schema.json','model-seal.json')


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
    venv=ROOT/'.venv'; exe=venv/'Scripts/python.exe'
    try:
        if not os.path.samefile(sys.executable,exe) or not os.path.samefile(sys.prefix,venv) or os.path.samefile(sys.prefix,sys.base_prefix):
            raise ValueError('Project venv required')
    except OSError as exc: raise ValueError('Venv identity unavailable') from exc
    return dict(python_version=list(sys.version_info[:3]),psutil_version=psutil.__version__,interpreter_sha256=sha(exe))


def bind_sources(train_sources,manifest,audits,audit_sha):
    if (not isinstance(train_sources,list) or len(train_sources)!=8 or len({s['game'] for s in train_sources})!=8
        or any(s.get('audit_sha256')!=audit_sha for s in train_sources)):
        raise ValueError('Training audit binding changed')
    games=manifest['games']; selected=[]
    for game,seed in zip(GAMES,(3911611436,2328454240)):
        rows=[x for x in games if x.get('key')==game]
        reports=[x for x in audits if x.get('game')==game]
        if len(rows)!=1 or len(reports)!=1 or game in {s['game'] for s in train_sources}: raise ValueError('Target source not unique or trained')
        row,report=rows[0],reports[0]
        if (row['seed']!=seed or row['side']!=0 or row['opponent']!='market' or row['arm']!='teacher_task'
            or report['passed'] is not True or report['errors']!=0 or report['actor_actions_reproduced']!=719
            or report['rival_actions_reproduced']!=719 or report['trace_turns_reproduced']!=719
            or report['trace_sha256']!=row['trace']['sha256'] or report['replay_sha256']!=row['replay']['sha256']):
            raise ValueError('Target audit identity changed')
        selected.append(dict(game=game,seed=seed,side=0,trace=row['trace'],replay_sha256=row['replay']['sha256'],audit_sha256=audit_sha))
    return selected


def prepare(out=OUT):
    out=Path(out).resolve()
    if os.name!='nt' or out!=OUT or any((out/n).exists() for n in ('registration.json','freeze.json','started.json','result.json')):
        raise ValueError('Registration unavailable')
    deadline_check(); decision=read(out/'root-decision.json'); baseline=decision['previous_commit']
    if (decision.get('status')!='approved_for_implementation_not_execution' or decision.get('development_only') is not True
        or decision.get('limits')!=LIMITS or decision.get('job_limits')!=JOB_LIMITS or decision.get('probe_job_limits')!=PROBE_JOB_LIMITS
        or not isinstance(baseline,str) or len(baseline)!=40 or any(c not in '0123456789abcdef' for c in baseline)):
        raise ValueError('Invalid decision')
    clean_head(baseline)
    for folder,names in ((PARENT,PARENT_EVIDENCE),(TRAIN,TRAIN_EVIDENCE),(ACTION,('input-manifest.json',))):
        for name in names:
            if git('show',baseline+':review/evidence/'+folder.name+'/'+name)!=(folder/name).read_bytes(): raise ValueError('Published evidence changed')
    prior=read(PARENT/'root-result-acceptance.json'); train=read(TRAIN/'root-result-acceptance.json')
    if not prior['passed'] or not train['passed']: raise ValueError('Prior stage not accepted')
    for folder,result,names in ((PARENT,prior,PARENT_EVIDENCE),(TRAIN,train,TRAIN_EVIDENCE)):
        for name in names[1:]:
            if name in result['evidence_sha256'] and sha(folder/name)!=result['evidence_sha256'][name]: raise ValueError('Evidence chain changed')
    training_cfg=read(TRAIN/'registration.json'); source_manifest=read(TRAIN/'source-manifest.json')
    if (sha(TRAIN/'registration.json')!=read(TRAIN/'freeze.json')['registration_sha256']
        or sha(TRAIN/'source-manifest.json')!=training_cfg['source_manifest_sha256']
        or sha(TRAIN/'schema.json')!=training_cfg['schema_sha256'] or source_manifest!=training_cfg['attested_sources']):
        raise ValueError('Training source manifest changed')
    audit_sha=sha(ACTION/'audit.json')
    sources=bind_sources(source_manifest,read(ACTION/'input-manifest.json'),read(ACTION/'audit.json'),audit_sha)
    prior_cfg=read(PARENT/'registration.json'); seal=read(TRAIN/'model-seal.json')
    if not seal['passed'] or seal['checkpoint_sha256']!=prior_cfg['checkpoint_sha256']: raise ValueError('New model seal changed')
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime=dict(read(PARENT/'freeze.json')['runtime_files'])
    for rel in ('learning/task_teacher_choice_bc_v1/collect.py','learning/task_teacher_choice_bc_v1/extract.py',
                'learning/task_completion_bc_v1/data.py','learning/task_bc_v1/model.py',
                'learning/task_job_sampling_race_v1/sampling.py','learning/task_training_job_supervision_v1/job.py'):
        if rel not in runtime or sha(ROOT/rel)!=runtime[rel]: raise ValueError('Frozen dependency changed')
    runtime.update(source)
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    cfg=dict(experiment='task-teacher-dev-deviation-v1',baseline_head=baseline,development_only=True,limits=LIMITS,
        job_limits=JOB_LIMITS,probe_job_limits=PROBE_JOB_LIMITS,thread_limits=THREAD_LIMITS,environment=environment(),
        sources=sources,audit_sha256=audit_sha,action_manifest_sha256=sha(ACTION/'input-manifest.json'),
        checkpoint_sha256=seal['checkpoint_sha256'],old_checkpoint=prior_cfg['old_checkpoint'],old_checkpoint_sha256=prior_cfg['old_checkpoint_sha256'],
        schema_sha256=sha(TRAIN/'schema.json'),parent_evidence={n:sha(PARENT/n) for n in PARENT_EVIDENCE},
        train_evidence={n:sha(TRAIN/n) for n in TRAIN_EVIDENCE},
        interpretation='Development trajectories from training seeds, not independent holdout or causal evidence',retry_allowed=False)
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
        or cfg['thread_limits']!=THREAD_LIMITS or cfg['environment']!=environment() or cfg['development_only'] is not True):
        raise ValueError('Limits or environment changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}!=set(freeze['new_source_sha256']): raise ValueError('Source set changed')
    for rel,digest in freeze['runtime_files'].items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    for folder,key in ((PARENT,'parent_evidence'),(TRAIN,'train_evidence')):
        for name,digest in cfg[key].items():
            if sha(folder/name)!=digest: raise ValueError('Evidence changed')
    if sha(ACTION/'audit.json')!=cfg['audit_sha256'] or sha(ACTION/'input-manifest.json')!=cfg['action_manifest_sha256']:
        raise ValueError('Source metadata changed')
    if bind_sources(read(TRAIN/'source-manifest.json'),read(ACTION/'input-manifest.json'),read(ACTION/'audit.json'),cfg['audit_sha256'])!=cfg['sources']:
        raise ValueError('Source bindings changed')
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


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','verify','ready','verify_commit'))
    args=parser.parse_args(); result=globals()[args.command]()
    print(json.dumps(dict(experiment=result['experiment'],passed=True)))
