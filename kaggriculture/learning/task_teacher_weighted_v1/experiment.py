"""Freeze a single capped operation-weighted BC fit on an accepted old cache."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-teacher-weighted-v1-20261027'
SOURCE=Path(__file__).resolve().parent
BC=ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
EVAL=ROOT/'artifacts/task-teacher-choice-eval-v2-20261027'
MERGE=ROOT/'artifacts/task-teacher-merge-real-v1-20261027'
BASE='f6fa40595bcc698b52c27e3b4476fbe9d736d34a'
MIB,GIB=1024**2,1024**3
OPS_COUNTS={'DIG':8,'FERTILIZE':84,'HARVEST':199,'PLANT':146,'WATER':1816}
TRAINING=dict(training_seed=2026102711,updates=128,work_seconds=870,process_seconds=900,
              cancellation_wait_seconds=20,cleanup_seconds=10)
DATA_LIMITS=dict(max_groups=4000,max_candidates=250000,max_archive_bytes=128*MIB,max_uncompressed_bytes=128*MIB)
HOST_LIMITS=dict(preflight_min_available_bytes=4*GIB,preflight_samples=3,preflight_interval_seconds=10,
                 min_available_bytes=2*GIB,poll_seconds=.25)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(MIB),b''): h.update(part)
    return h.hexdigest()


def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Output outside owned experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def deadline_check():
    end=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if end.tzinfo is None or datetime.now(timezone.utc)>=end: raise TimeoutError('Stage deadline')


def windows_environment():
    venv=ROOT/'.venv'; python=venv/'Scripts/python.exe'
    if (os.name!='nt' or not os.path.samefile(sys.executable,python) or not os.path.samefile(sys.prefix,venv)
        or os.path.samefile(sys.prefix,sys.base_prefix) or sys.flags.optimize!=0 or 'PYTHONOPTIMIZE' in os.environ):
        raise ValueError('Expected non-optimized project Windows interpreter')
    return dict(interpreter_sha256=sha(python),python='.'.join(map(str,sys.version_info[:3])),
                numpy=importlib.metadata.version('numpy'),optimization_level=0,python_optimize_env_absent=True)


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip()!=expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def published_equal(folder,names):
    for name in names:
        if git('show',BASE+':review/evidence/'+folder.name+'/'+name)!=(folder/name).read_bytes():
            raise ValueError('Published predecessor differs')


def expected_weights(counts):
    if counts!=OPS_COUNTS: raise ValueError('Unexpected registered operation counts')
    total=sum(counts.values())
    return {op:min(2.,math.sqrt(total/(len(counts)*value))) for op,value in counts.items()}


def prepare(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve() or (out/'registration.json').exists(): raise ValueError('Experiment already registered')
    deadline_check(); clean_head(BASE)
    decision=read(out/'root-decision.json')
    if decision.get('previous_commit')!=BASE or decision.get('status')!='approved_for_implementation_not_execution':
        raise ValueError('Missing prospective decision')
    published_equal(BC,('registration.json','freeze.json','train-data-result.json','root-result-acceptance.json','schema.json','model-seal.json'))
    published_equal(EVAL,('freeze.json','root-result-acceptance.json'))
    published_equal(MERGE,('registration.json','freeze.json','root-result-acceptance.json'))
    bc,old,acceptance,merge=(read(p) for p in (BC/'registration.json',BC/'train-data-result.json',
        BC/'root-result-acceptance.json',MERGE/'root-result-acceptance.json'))
    if (old.get('passed') is not True or old.get('sealed') is not True or acceptance.get('passed') is not True
        or acceptance.get('evidence_sha256',{}).get('train-data-result.json')!=sha(BC/'train-data-result.json')
        or old.get('selected_operations')!=OPS_COUNTS or old.get('groups')!=2253 or old.get('candidate_rows')!=118955
        or merge.get('passed') is not True or merge.get('old_groups')!=old['groups']):
        raise ValueError('Accepted training source or merge evidence changed')
    if bc['training']['training_seed']!=TRAINING['training_seed'] or bc['training']['updates']!=128:
        raise ValueError('Prior training seed or updates changed')
    schema=BC/'schema.json'; cache=BC/'private/train.npz'
    if sha(schema)!=bc['schema_sha256'] or not cache.is_file() or not 0<cache.stat().st_size<=DATA_LIMITS['max_archive_bytes']:
        raise ValueError('Schema or archive metadata changed')
    model=BC/'private/model-final.npz'; model_seal=read(BC/'model-seal.json')
    if model_seal.get('passed') is not True or sha(model)!=model_seal['checkpoint_sha256']:
        raise ValueError('Frozen reference checkpoint changed')
    crop=ROOT/bc['crop_checkpoint']
    if sha(crop)!=bc['crop_checkpoint_sha256']: raise ValueError('Frozen crop checkpoint changed')
    launch=BC/'gpu-launch.local.json'
    if sha(launch)!=bc['gpu_launch_config_sha256']: raise ValueError('Private GPU launch identity changed')
    env=windows_environment()
    if env['python']!=bc['runtime_versions']['python'] or env['numpy']!=bc['runtime_versions']['numpy']:
        raise ValueError('Windows Python or NumPy changed')
    from learning.task_teacher_merge_real_v1.experiment import trusted_runtime
    runtime=trusted_runtime((EVAL/'freeze.json').read_bytes(),
        git('show',BASE+':review/evidence/'+EVAL.name+'/freeze.json'),read(EVAL/'root-result-acceptance.json'))
    for rel in ('learning/task_teacher_merge_real_v1/experiment.py',
                'learning/task_probe_failure_reporting_v1/report.py','learning/task_teacher_dev_deviation_v1/membership.py'):
        data=git('show',BASE+':kaggriculture/'+rel)
        if data!=(ROOT/rel).read_bytes(): raise ValueError('Published dependency changed')
        runtime[rel]=hashlib.sha256(data).hexdigest()
    for rel,digest in runtime.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime dependency changed')
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    if not source: raise ValueError('Missing weighted source')
    roles=read(MERGE/'seed-roles.local.json'); merge_cfg=read(MERGE/'registration.json')
    if sha(MERGE/'seed-roles.local.json')!=merge_cfg['seed_roles_sha256']:
        raise ValueError('Prior seed role inventory changed')
    excluded=set(roles['holdout_seeds'])|set(bc['train_seeds'])|{merge_cfg['synthetic_seed']}
    selected=[]; index=0
    while len(selected)<6:
        value=int.from_bytes(hashlib.sha256(f'task-teacher-weighted-v1:20261027:holdout:{index}'.encode('ascii')).digest()[:4],'big')
        index+=1
        if value not in excluded and value not in selected: selected.append(value)
    with (out/'gpu-launch.local.json').open('xb') as f: f.write(launch.read_bytes())
    (out/'private').mkdir(exist_ok=False)
    write_new(out/'seed-inventory.local.json',dict(excluded_uint32=sorted(excluded),reserved_eval_seeds=selected,
        scope='Prior accepted local metadata exclusion list plus old training and synthetic marker seeds',
        limitation='Private, deleted, unavailable, external or later unlisted seed history cannot be excluded'))
    cfg=dict(experiment='task-teacher-weighted-v1',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        environment=env,wsl_software_versions=bc['wsl_software_versions'],gpu_launch_config_sha256=sha(out/'gpu-launch.local.json'),
        cache_path=cache.relative_to(ROOT).as_posix(),cache_sha256=old['array_sha256'],
        schema_path=schema.relative_to(ROOT).as_posix(),schema_sha256=bc['schema_sha256'],
        old_train_seeds=list(bc['train_seeds']),expected_operations=dict(OPS_COUNTS),
        expected_class_weights=expected_weights(OPS_COUNTS),weight_formula='min(2,sqrt(N/(5*n_selected_operation)))',
        reference_checkpoint=model.relative_to(ROOT).as_posix(),reference_checkpoint_sha256=model_seal['checkpoint_sha256'],
        crop_checkpoint=bc['crop_checkpoint'],crop_checkpoint_sha256=bc['crop_checkpoint_sha256'],
        reserved_eval_seeds=selected,seed_inventory_sha256=sha(out/'seed-inventory.local.json'),
        seed_scope='Prior accepted metadata exclusions plus old training and synthetic marker seeds',
        seed_history_limitation='Private, deleted, unavailable, external or later unlisted seed history cannot be excluded',
        training=dict(TRAINING),data_limits=dict(DATA_LIMITS),host_limits=dict(HOST_LIMITS),process_log_bytes=MIB,
        retry_allowed=False,no_official_games=True,no_automatic_promotion=True,
        interpretation='One weighted fit on reused training examples; independent competitive improvement is not proven')
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'),parent_commit=BASE,runtime_files=runtime,new_source_sha256=source,
        old_result_sha256=sha(BC/'train-data-result.json'),old_acceptance_sha256=sha(BC/'root-result-acceptance.json'),
        old_registration_sha256=sha(BC/'registration.json'),old_model_seal_sha256=sha(BC/'model-seal.json'),
        parent_eval_freeze_sha256=sha(EVAL/'freeze.json'),parent_eval_acceptance_sha256=sha(EVAL/'root-result-acceptance.json'),
        merge_acceptance_sha256=sha(MERGE/'root-result-acceptance.json'),merge_roles_sha256=sha(MERGE/'seed-roles.local.json')))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve(): raise ValueError('Unexpected output')
    deadline_check(); cfg,freeze=read(out/'registration.json'),read(out/'freeze.json')
    if (sha(out/'registration.json')!=freeze['registration_sha256'] or sha(out/'root-decision.json')!=freeze['root_decision_sha256']
        or sha(BC/'train-data-result.json')!=freeze['old_result_sha256'] or sha(BC/'root-result-acceptance.json')!=freeze['old_acceptance_sha256']
        or sha(BC/'registration.json')!=freeze['old_registration_sha256'] or sha(BC/'model-seal.json')!=freeze['old_model_seal_sha256']
        or sha(EVAL/'freeze.json')!=freeze['parent_eval_freeze_sha256'] or sha(EVAL/'root-result-acceptance.json')!=freeze['parent_eval_acceptance_sha256']
        or sha(MERGE/'root-result-acceptance.json')!=freeze['merge_acceptance_sha256']
        or sha(MERGE/'seed-roles.local.json')!=freeze['merge_roles_sha256']):
        raise ValueError('Stage or predecessor identity changed')
    if (cfg['training']!=TRAINING or cfg['data_limits']!=DATA_LIMITS or cfg['host_limits']!=HOST_LIMITS
        or cfg['expected_operations']!=OPS_COUNTS or cfg['expected_class_weights']!=expected_weights(OPS_COUNTS)
        or cfg['retry_allowed'] is not False or cfg['no_official_games'] is not True or cfg['no_automatic_promotion'] is not True):
        raise ValueError('Training policy or budget changed')
    inventory=read(out/'seed-inventory.local.json'); seeds=cfg['reserved_eval_seeds']
    if (sha(out/'seed-inventory.local.json')!=cfg['seed_inventory_sha256'] or len(seeds)!=6 or len(set(seeds))!=6
        or seeds!=inventory['reserved_eval_seeds'] or any(type(s) is not int or s in inventory['excluded_uint32'] for s in seeds)):
        raise ValueError('Reserved seed identity changed')
    if (sha(out/'gpu-launch.local.json')!=cfg['gpu_launch_config_sha256'] or sha(ROOT/cfg['schema_path'])!=cfg['schema_sha256']
        or sha(ROOT/cfg['reference_checkpoint'])!=cfg['reference_checkpoint_sha256'] or sha(ROOT/cfg['crop_checkpoint'])!=cfg['crop_checkpoint_sha256']):
        raise ValueError('Private model, schema or launch identity changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}!=set(freeze['new_source_sha256']):
        raise ValueError('Stage source set changed')
    for rel,digest in {**freeze['runtime_files'],**freeze['new_source_sha256']}.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime source changed')
    if os.name=='nt' and windows_environment()!=cfg['environment']: raise ValueError('Windows interpreter changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out); receipt=read(out/'pre-execution-commit.json'); commit=receipt['commit']
    clean_head(commit); git('merge-base','--is-ancestor',cfg['baseline_head'],commit)
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
