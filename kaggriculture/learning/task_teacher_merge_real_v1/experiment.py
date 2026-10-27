"""Freeze one old-cache plus synthetic-group compatibility check."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-teacher-merge-real-v1-20261027'
SOURCE=Path(__file__).resolve().parent
BC=ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
EVAL=ROOT/'artifacts/task-teacher-choice-eval-v2-20261027'
REFRESH=ROOT/'artifacts/task-teacher-refresh-v1-20261027'
BASE='6c6ccf2d6a03bb91b8f0c1ccbb5326ccbdf81991'
MIB,GIB=1024**2,1024**3
PROBE_JOB=dict(max_rss_bytes=GIB,max_commit_bytes=1536*MIB,min_available_bytes=2*GIB,
               wall_seconds=120,poll_seconds=.25,cleanup_seconds=10)
WORKER_JOB=dict(max_rss_bytes=1536*MIB,max_commit_bytes=2*GIB,min_available_bytes=2*GIB,
                wall_seconds=300,poll_seconds=.25,cleanup_seconds=10)
REQUIRED_RUNTIME={'learning/task_teacher_choice_bc_v1/numeric.py','learning/task_teacher_choice_bc_v1/training.py',
                  'learning/task_choice_v1/features.py','learning/task_training_job_supervision_v1/job.py',
                  'learning/task_job_sampling_race_v1/sampling.py','learning/task_bc_v1/model.py'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(MIB),b''): h.update(part)
    return h.hexdigest()


def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Output outside owned stage')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def deadline_check():
    end=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if end.tzinfo is None or datetime.now(timezone.utc)>=end: raise TimeoutError('Stage deadline')


def environment():
    venv=ROOT/'.venv'; python=venv/'Scripts/python.exe'
    if (os.name!='nt' or not os.path.samefile(sys.executable,python) or not os.path.samefile(sys.prefix,venv)
        or os.path.samefile(sys.prefix,sys.base_prefix) or sys.flags.optimize!=0 or 'PYTHONOPTIMIZE' in os.environ):
        raise ValueError('Expected non-optimized project Windows interpreter')
    return dict(interpreter_sha256=sha(python),optimization_level=0,python_optimize_env_absent=True)


def _clean_head(expected):
    if git('rev-parse','HEAD').decode().strip()!=expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty publication HEAD')


def _public_equal(folder,names):
    for name in names:
        if git('show',BASE+':review/evidence/'+folder.name+'/'+name)!=(folder/name).read_bytes():
            raise ValueError('Published predecessor differs')


def trusted_runtime(local_bytes,published_bytes,acceptance):
    """Bind the predecessor's runtime list to its published acceptance."""
    if (local_bytes!=published_bytes or not isinstance(acceptance,dict) or acceptance.get('panel_completed') is not True
        or hashlib.sha256(local_bytes).hexdigest()!=acceptance.get('evidence_sha256',{}).get('freeze.json')):
        raise ValueError('Predecessor runtime freeze changed')
    runtime=json.loads(local_bytes.decode('utf-8-sig')).get('runtime_files')
    if (not isinstance(runtime,dict) or not REQUIRED_RUNTIME.issubset(runtime)
        or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64
               or any(c not in '0123456789abcdef' for c in v) for k,v in runtime.items())):
        raise ValueError('Incomplete predecessor runtime')
    return dict(runtime)


def prepare(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve() or (out/'registration.json').exists(): raise ValueError('Stage already registered')
    deadline_check(); _clean_head(BASE)
    decision=read(out/'root-decision.json')
    if decision.get('previous_commit')!=BASE or decision.get('status')!='approved_for_implementation_not_execution':
        raise ValueError('Missing prospective decision')
    _public_equal(BC,('registration.json','freeze.json','train-data-result.json','root-result-acceptance.json','schema.json'))
    _public_equal(REFRESH,('registration.json','games-failure.json','root-result-acceptance.json'))
    _public_equal(EVAL,('freeze.json','root-result-acceptance.json'))
    old,bc_cfg,accepted=read(BC/'train-data-result.json'),read(BC/'registration.json'),read(BC/'root-result-acceptance.json')
    if (old.get('passed') is not True or old.get('sealed') is not True or accepted.get('passed') is not True
        or accepted.get('evidence_sha256',{}).get('train-data-result.json')!=sha(BC/'train-data-result.json')
        or accepted.get('selected_groups')!=old['groups'] or accepted.get('candidate_rows')!=old['candidate_rows']):
        raise ValueError('Old data seal not accepted')
    schema=BC/'schema.json'
    if sha(schema)!=bc_cfg['schema_sha256']: raise ValueError('Old schema identity changed')
    cache=BC/'private/train.npz'
    if not cache.is_file() or not 0<cache.stat().st_size<=128*MIB:
        raise ValueError('Old archive missing or over budget')
    prior=read(REFRESH/'registration.json')
    inventory=read(REFRESH/'seed-inventory.local.json')
    if sha(REFRESH/'seed-inventory.local.json')!=prior['seed_inventory_sha256']:
        raise ValueError('Prior seed inventory changed')
    train=list(bc_cfg['train_seeds'])
    excluded=set(inventory['excluded_uint32'])|set(prior['train_seeds'])|set(prior['reserved_test_seeds'])
    holdout=sorted(excluded-set(train))
    index=0
    while True:
        candidate=int.from_bytes(hashlib.sha256(f'task-teacher-merge-real-v1:synthetic:{index}'.encode('ascii')).digest()[:4],'big')
        index+=1
        if candidate not in excluded and candidate not in train: break
    if not holdout: raise ValueError('Missing conservative holdout exclusion')
    env=environment()
    eval_freeze=(EVAL/'freeze.json').read_bytes()
    eval_public=git('show',BASE+':review/evidence/'+EVAL.name+'/freeze.json')
    runtime=trusted_runtime(eval_freeze,eval_public,read(EVAL/'root-result-acceptance.json'))
    for rel in ('learning/task_probe_failure_reporting_v1/report.py','learning/task_teacher_dev_deviation_v1/membership.py',
                'learning/task_teacher_merge_v1/merge.py','learning/task_teacher_merge_v1/test_merge.py'):
        data=git('show',BASE+':kaggriculture/'+rel)
        if data!=(ROOT/rel).read_bytes(): raise ValueError('Published dependency changed')
        runtime[rel]=hashlib.sha256(data).hexdigest()
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Frozen dependency changed')
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    if not source: raise ValueError('Missing source')
    write_new(out/'seed-roles.local.json',dict(holdout_seeds=holdout,synthetic_seed=candidate,
        scope='Prior metadata exclusions plus the stopped panel development and reserved seeds; old training seeds removed',
        limitation='Does not cover private, deleted, unavailable or external seed history'))
    cfg=dict(experiment='task-teacher-merge-real-v1',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        environment=env,cache_path=cache.relative_to(ROOT).as_posix(),cache_sha256=old['array_sha256'],
        schema_path=schema.relative_to(ROOT).as_posix(),schema_sha256=bc_cfg['schema_sha256'],
        old_groups=old['groups'],old_candidate_rows=old['candidate_rows'],old_train_seeds=train,
        synthetic_seed=candidate,seed_roles_sha256=sha(out/'seed-roles.local.json'),
        seed_role_scope='Previous metadata exclusions plus stopped-panel seed roles; synthetic seed is not a game',
        seed_history_limitation='Private, deleted, unavailable or external seed history cannot be excluded',
        max_groups=4000,max_candidates=250000,max_archive_bytes=128*MIB,max_uncompressed_bytes=128*MIB,
        merge_max_groups=4002,merge_max_candidates=250003,synthetic_groups=2,synthetic_candidate_rows=3,
        probe_job_supervision=PROBE_JOB,worker_job_supervision=WORKER_JOB,
        preflight_min_available_bytes=4*GIB,preflight_samples=3,preflight_interval_seconds=10,
        archive_snapshot_limit=1,output='aggregate only; no merged archive',retry_allowed=False,
        no_training=True,no_model_reads=True,no_official_games=True)
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'),runtime_files=runtime,new_source_sha256=source,
        parent_commit=BASE,old_acceptance_sha256=sha(BC/'root-result-acceptance.json'),
        old_result_sha256=sha(BC/'train-data-result.json'),old_registration_sha256=sha(BC/'registration.json'),
        parent_eval_freeze_sha256=sha(EVAL/'freeze.json'),parent_eval_acceptance_sha256=sha(EVAL/'root-result-acceptance.json'),
        prior_refresh_registration_sha256=sha(REFRESH/'registration.json'),prior_inventory_sha256=sha(REFRESH/'seed-inventory.local.json')))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve(): raise ValueError('Unexpected output')
    deadline_check(); cfg,freeze=read(out/'registration.json'),read(out/'freeze.json')
    if (sha(out/'registration.json')!=freeze['registration_sha256'] or sha(out/'root-decision.json')!=freeze['root_decision_sha256']
        or sha(out/'seed-roles.local.json')!=cfg['seed_roles_sha256']
        or sha(BC/'root-result-acceptance.json')!=freeze['old_acceptance_sha256']
        or sha(BC/'train-data-result.json')!=freeze['old_result_sha256']
        or sha(BC/'registration.json')!=freeze['old_registration_sha256']
        or sha(EVAL/'freeze.json')!=freeze['parent_eval_freeze_sha256']
        or sha(EVAL/'root-result-acceptance.json')!=freeze['parent_eval_acceptance_sha256']
        or sha(REFRESH/'registration.json')!=freeze['prior_refresh_registration_sha256']
        or sha(REFRESH/'seed-inventory.local.json')!=freeze['prior_inventory_sha256']):
        raise ValueError('Stage or predecessor identity changed')
    if (cfg['probe_job_supervision']!=PROBE_JOB or cfg['worker_job_supervision']!=WORKER_JOB
        or cfg['preflight_min_available_bytes']!=4*GIB or cfg['preflight_samples']!=3
        or cfg['preflight_interval_seconds']!=10 or cfg['archive_snapshot_limit']!=1
        or cfg['synthetic_groups']!=2 or cfg['synthetic_candidate_rows']!=3 or cfg['retry_allowed'] is not False):
        raise ValueError('Stage budget changed')
    roles=read(out/'seed-roles.local.json'); old=set(cfg['old_train_seeds']); holdout=set(roles['holdout_seeds'])
    if (roles['synthetic_seed']!=cfg['synthetic_seed'] or not holdout or old&holdout
        or cfg['synthetic_seed'] in (old|holdout)):
        raise ValueError('Seed roles overlap')
    if sha(ROOT/cfg['schema_path'])!=cfg['schema_sha256']: raise ValueError('Schema changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}!=set(freeze['new_source_sha256']):
        raise ValueError('Stage source set changed')
    for rel,digest in {**freeze['runtime_files'],**freeze['new_source_sha256']}.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    if os.name=='nt' and environment()!=cfg['environment']: raise ValueError('Interpreter changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out); receipt=read(out/'pre-execution-commit.json'); commit=receipt['commit']
    _clean_head(commit); git('merge-base','--is-ancestor',cfg['baseline_head'],commit)
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes():
            raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg


def ready(out=OUT):
    cfg=verify_commit(out); review=read(Path(out)/'pre-review.json'); receipt=read(Path(out)/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('registration_sha256')!=sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    return cfg
