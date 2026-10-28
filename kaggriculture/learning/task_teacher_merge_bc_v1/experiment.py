"""Prospective admission for a single merged fit; no execution on import.

Archive hashes attest bytes, not independent reconstruction of their features.
The fit worker must perform the frozen numeric readback under its guard.
"""
from datetime import datetime, timezone
import hashlib
import json
import importlib.metadata
import sys
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-teacher-merge-bc-v1-20261028'
SOURCE=Path(__file__).resolve().parent
BC=ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
REFRESH=ROOT/'artifacts/task-teacher-refresh-v2-20261028'
BASE='27fddfceca3e376f22f0cc24b67c82f88e46b7d9'
MIB,GIB=1024**2,1024**3
TRAINING=dict(training_seed=2026102711,updates=128,work_seconds=870,process_seconds=900,cancellation_wait_seconds=20,cleanup_seconds=10)
HOST_LIMITS=dict(preflight_min_available_bytes=4*GIB,preflight_samples=3,preflight_interval_seconds=10,min_available_bytes=2*GIB,poll_seconds=.25)
DATA_LIMITS=dict(max_groups=8000,max_candidates=400000,max_archive_bytes=256*MIB,max_uncompressed_bytes=256*MIB)
REUSED=('learning/task_teacher_choice_bc_v1/numeric.py','learning/task_teacher_choice_bc_v1/training.py',
        'learning/task_teacher_merge_v1/merge.py','learning/task_choice_v1/features.py',
        'learning/task_dagger_once_v1/gpu_guard.py','learning/task_anchor_once_v1/processes.py',
        'learning/task_action_observation_v1/campaign.py','learning/task_launcher_contract_repair_v1/observer.py')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(MIB),b''): h.update(chunk)
    return h.hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)
def clean_head(commit):
    if git('rev-parse','HEAD').decode().strip()!=commit or git('status','--porcelain').strip(): raise ValueError('Public HEAD changed or dirty')
def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def prepare_private_checkpoint(out, *, create=False):
    """Validate an empty owned directory; only the preflight may create it."""
    def linked(path):
        return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()

    if type(create) is not bool: raise ValueError('Invalid creation flag')
    configured=Path(OUT)
    if linked(configured): raise ValueError('Experiment output is a link')
    workspace=ROOT.resolve(strict=True); stage=configured.resolve(strict=True)
    if (not stage.is_dir() or stage==workspace or not stage.is_relative_to(workspace)
        or Path(out).resolve(strict=True)!=stage): raise ValueError('Unexpected experiment output')
    private=stage/'private'
    if linked(private): raise ValueError('Private output is a link')
    if not private.exists():
        if not create: raise FileNotFoundError('Private output was not prepared')
        private.mkdir()
    if linked(private) or not private.is_dir() or private.resolve(strict=True)!=private:
        raise ValueError('Private output outside experiment')
    if next(private.iterdir(),None) is not None:
        raise FileExistsError('Private output directory is not empty')
    return private/'model-final.npz'
def deadline_check():
    end=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if end.tzinfo is None or datetime.now(timezone.utc)>=end: raise TimeoutError('Deadline')
def published(folder,names):
    for name in names:
        if git('show',BASE+':review/evidence/'+folder.name+'/'+name)!=(folder/name).read_bytes(): raise ValueError('Historical publication differs')
def good_job(job):
    return (isinstance(job,dict) and all(job.get(k) is True for k in ('passed','assigned','resumed','launcher_reaped','job_members_zero'))
            and type(job.get('exit_code')) is int and job['exit_code']==0
            and all(job.get(k) is None for k in ('error_kind','cleanup_error_kind','stop_reason')))


def historical_admission():
    old_names=('registration.json','freeze.json','pre-execution-commit.json','root-result-acceptance.json','train-data-result.json','data-supervision-result.json','model-seal.json','schema.json')
    new_names=('registration.json','freeze.json','pre-execution-commit.json','root-result-acceptance.json','data-result.json','data-seal.json','data-worker-result.json','other-job-receipts.public.json')
    published(BC,old_names); published(REFRESH,new_names)
    old=read(BC/'train-data-result.json'); oa=read(BC/'root-result-acceptance.json'); oc=read(BC/'registration.json')
    new=read(REFRESH/'data-worker-result.json'); na=read(REFRESH/'root-result-acceptance.json'); nc=read(REFRESH/'registration.json')
    result=read(REFRESH/'data-result.json'); seal=read(REFRESH/'data-seal.json'); jobs=read(REFRESH/'other-job-receipts.public.json')
    for folder,receipt,acceptance in ((BC,read(BC/'pre-execution-commit.json'),oa),(REFRESH,read(REFRESH/'pre-execution-commit.json'),na)):
        git('merge-base','--is-ancestor',receipt['commit'],BASE)
        for n in ('registration.json','freeze.json'):
            if git('show',receipt['commit']+':review/evidence/'+folder.name+'/'+n)!=(folder/n).read_bytes(): raise ValueError('Historical commitment differs')
    if (oa.get('passed') is not True or old.get('passed') is not True or old.get('sealed') is not True
        or oa['evidence_sha256']['train-data-result.json']!=sha(BC/'train-data-result.json')
        or oa['evidence_sha256']['data-supervision-result.json']!=sha(BC/'data-supervision-result.json')
        or not good_job(read(BC/'data-supervision-result.json')['supervision'])
        or na.get('passed') is not True or na['pre_execution_commit']!=read(REFRESH/'pre-execution-commit.json')['commit']
        or result.get('passed') is not True or result.get('status')!='collected' or result.get('data_seal_sha256')!=sha(REFRESH/'data-seal.json')
        or seal.get('passed') is not True or seal.get('completion_state')!='pending_result_confirmation'
        or na['data_result_sha256']!=sha(REFRESH/'data-result.json') or na['data_seal_sha256']!=sha(REFRESH/'data-seal.json')
        or na['worker_result_sha256']!=sha(REFRESH/'data-worker-result.json') or seal['worker_result_sha256']!=sha(REFRESH/'data-worker-result.json')
        or jobs['data_supervision_sha256']!=seal['supervision_sha256'] or jobs['data_cleanup_state']!='recorded' or not good_job(jobs['data_job'])
        or new.get('passed') is not True or new.get('status')!='collected'):
        raise ValueError('Historical admission failed')
    paths={'old':BC/'private/train.npz','new':REFRESH/'private/new-teacher.npz'}
    hashes={'old':old['array_sha256'],'new':seal['array_sha256']}
    if new['array_sha256']!=hashes['new'] or na['archive_sha256']!=hashes['new'] or any(not p.is_file() or sha(p)!=hashes[k] for k,p in paths.items()): raise ValueError('Archive identity changed')
    if (BC/'schema.json').read_bytes()!=(REFRESH/'schema.json').read_bytes() or sha(BC/'schema.json')!=oc['schema_sha256'] or sha(REFRESH/'schema.json')!=nc['schema_sha256']: raise ValueError('Schema mismatch')
    return oc,nc,old,new,paths,hashes


def validate_roles(old,new,holdout):
    if any(not isinstance(x,list) for x in (old,new,holdout)): raise ValueError('Seed roles invalid')
    all_seeds=old+new+holdout
    if not old or not new or any(type(x) is not int or not 0<=x<2**32 for x in all_seeds) or len(set(all_seeds))!=len(all_seeds): raise ValueError('Seed overlap')


def role_values(value, active=False, development=False):
    """Treat validation as held out unless a source-specific development binding is proven."""
    result=set()
    if isinstance(value,dict):
        for key,item in value.items():
            label=str(key).lower()
            if (label=='excluded_uint32' or label.endswith(('_seed_count','_seeds_count'))
                or (development and label=='validation_seeds')): continue
            selected=(('seed' in label and any(word in label for word in ('test','holdout','reserved','validation'))) or label in ('test','holdout','reserved','validation'))
            nested=active and label in ('values','seed','seeds','test','holdout','reserved','validation')
            result.update(role_values(item,selected or nested,False))
    elif isinstance(value,list):
        for item in value: result.update(role_values(item,active,False))
    elif active and type(value) is int and 0<=value<2**32: result.add(value)
    return result


def development_record(label,raw):
    """Allow only the published observation registrations used by old BC sources."""
    prefix='task-action-observation-v1-20261027/'
    accepted={prefix+'registration.json',prefix+'pre-review-initial/registration.json'}
    if label.startswith('local/'):
        rel=label[len('local/'):]
    elif label.startswith('public/review/evidence/'):
        rel=label[len('public/review/evidence/'):]
    else: return False
    if rel not in accepted: return False
    if raw!=git('show',BASE+':review/evidence/'+rel): raise ValueError('Development source publication differs')
    value=json.loads(raw.decode('utf-8-sig')); old=read(BC/'registration.json')
    if (value.get('experiment')!='task-action-observation-v1'
        or value.get('data_role')!='Fresh development observation only; no promotion, tuning, confidence interval or independent holdout claim'
        or old.get('source_root')!='artifacts/task-action-observation-v1-20261027'
        or value.get('validation_seeds')!=old.get('train_seeds')):
        raise ValueError('Development source role differs')
    return True


def seed_inventory():
    """Recognized local/public non-private metadata; unavailable or deleted history remains unknown."""
    names={'registration.json','root-decision.json','seeds.json','seed-roles.local.json','seed-inventory.local.json','seed-inventory.json'}
    paths=[]
    for root,prefix in ((ROOT/'artifacts','local'),(PUBLIC,'public')):
        if root.exists():
            paths.extend((prefix+'/'+p.relative_to(root).as_posix(),p) for p in root.rglob('*.json')
                if p.name.lower() in names and 'private' not in {x.lower() for x in p.relative_to(root).parts}
                and not p.is_relative_to(OUT))
    identities=[]; found=set(); total=0
    for label,path in sorted(paths):
        raw=path.read_bytes(); total+=len(raw)
        if total>256*MIB: raise ValueError('Seed inventory budget exceeded')
        found.update(role_values(json.loads(raw.decode('utf-8-sig')),development=development_record(label,raw))); identities.append([label,hashlib.sha256(raw).hexdigest()])
    return dict(holdout_seeds=sorted(found),identities=identities,metadata_bytes=total,
        scope='Recognized non-private local artifact and public repository seed-role metadata',
        limitation='Deleted, private, unavailable, external, or unrecognized role metadata cannot be excluded')


def runtime_files():
    # Historic freeze authenticates local bytes; only the four observed line-ending
    # conversions may differ from the committed publication.
    previous=read(REFRESH/'freeze.json')['runtime_files']; runtime={}; public={}
    line_endings={'decision/execution.py','decision/full_audit.py','decision/verify_audit.py','learning/round2_worker.py'}
    for rel in sorted(set(previous)|set(REUSED)):
        data=git('show',BASE+':kaggriculture/'+rel); local=(ROOT/rel).read_bytes()
        local_sha=hashlib.sha256(local).hexdigest()
        if rel in previous and local_sha!=previous[rel]: raise ValueError('Frozen runtime differs')
        if data!=local and (rel not in line_endings or rel not in previous
            or data.replace(b'\r\n',b'\n')!=local.replace(b'\r\n',b'\n')):
            raise ValueError('Published runtime differs')
        runtime[rel]=local_sha; public[rel]=hashlib.sha256(data).hexdigest()
    return runtime,public


def protected(oc):
    acceptance=read(REFRESH/'root-result-acceptance.json'); expected=acceptance['existing_model_sha256']
    paths={'default_bot':'main.py','crop':oc['crop_checkpoint'],
           'unweighted':'artifacts/task-teacher-choice-bc-v1-20261027/private/model-final.npz',
           'weighted':'artifacts/task-teacher-weighted-v1-20261027/private/model-final.npz',
           'reference':oc['reference_checkpoint']}
    hashes={'default_bot':acceptance['default_bot_sha256'],'crop':expected['crop'],
            'unweighted':expected['unweighted'],'weighted':expected['weighted'],'reference':oc['reference_checkpoint_sha256']}
    if any(sha(ROOT/paths[k])!=hashes[k] for k in paths): raise ValueError('Protected model or bot changed')
    return paths,hashes


def environment(oc):
    from learning.task_teacher_weighted_v1.experiment import windows_environment
    env=windows_environment()
    if env['interpreter_sha256']!=oc['windows_venv_interpreter_sha256'] or env['python']!=oc['runtime_versions']['python'] or env['numpy']!=oc['runtime_versions']['numpy']:
        raise ValueError('Windows environment changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    files=installed_files()
    if sha(files['rules'])!=oc['engine_sha256'] or sha(files['loader'])!=oc['official_loader_sha256']: raise ValueError('Official engine changed')
    return env


def build_config(config,oc,nc,old,new,paths,hashes,inventory,env,launch_sha,protected_paths,protected_sha):
    old_seeds=oc['train_seeds']; new_seeds=nc['train_seeds']; holdout=inventory['holdout_seeds']
    validate_roles(old_seeds,new_seeds,holdout)
    if config.get('old_train_seeds')!=old_seeds or config.get('new_train_seeds')!=new_seeds or config.get('holdout_seed_count')!=len(holdout):
        raise ValueError('Explicit role inventory differs')
    if config.get('training_seed')!=TRAINING['training_seed'] or oc['training']['training_seed']!=TRAINING['training_seed']:
        raise ValueError('Training seed changed')
    old_ops=old['selected_operations']; new_ops=new['coverage']['retained_selected_operations']
    if (sum(old_ops.values())!=old['groups'] or sum(new_ops.values())!=new['coverage']['groups']
        or new['sample_summary']['groups']!=new['coverage']['groups'] or new['sample_summary']['selected_operations']!=new_ops):
        raise ValueError('Historical counts inconsistent')
    groups=dict(old=old['groups'],new=new['coverage']['groups']); candidates=dict(old=old['candidate_rows'],new=new['full_stream']['retained_candidate_rows'])
    if sum(groups.values())>DATA_LIMITS['max_groups'] or sum(candidates.values())>DATA_LIMITS['max_candidates']: raise ValueError('Merged data exceeds budget')
    return dict(experiment='task-teacher-merge-bc-v1',baseline_head=BASE,deadline_utc=config['deadline_utc'],
        old_train_seeds=old_seeds,new_train_seeds=new_seeds,holdout_seed_count=len(holdout),training_seed=TRAINING['training_seed'],
        archive_path={k:v.relative_to(ROOT).as_posix() for k,v in paths.items()},archive_sha256=hashes,
        schema_path={'old':(BC/'schema.json').relative_to(ROOT).as_posix(),'new':(REFRESH/'schema.json').relative_to(ROOT).as_posix()},
        schema_sha256=oc['schema_sha256'],expected_groups=groups,expected_candidates=candidates,expected_operations=dict(old=old_ops,new=new_ops),
        gpu_launch_config_sha256=launch_sha,wsl_software_versions=oc['wsl_software_versions'],environment=env,
        engine_commit=oc['engine_commit'],engine_sha256=oc['engine_sha256'],official_loader_sha256=oc['official_loader_sha256'],
        protected_paths=protected_paths,protected_sha256=protected_sha,training=dict(TRAINING),host_limits=dict(HOST_LIMITS),data_limits=dict(DATA_LIMITS),
        process_log_bytes=MIB,retry_allowed=False,no_official_games=True,no_automatic_promotion=True,
        seed_scope=inventory['scope'],seed_history_limitation=inventory['limitation'],
        admission_limitation='Archive SHA binds bytes; feature derivation is not independently recomputed from hashes')


def prepare(config,out=OUT):
    """Register only upon explicit call, after a separately created prospective decision."""
    out=Path(out).resolve()
    if out!=OUT.resolve() or (out/'registration.json').exists(): raise ValueError('Unexpected output or existing registration')
    deadline_check(); clean_head(BASE); decision=read(out/'root-decision.json')
    if decision.get('previous_commit')!=BASE or decision.get('status')!='approved_for_implementation_not_execution' or config.get('deadline_utc')!=decision['deadline_utc']: raise ValueError('Missing prospective decision')
    oc,nc,old,new,paths,hashes=historical_admission(); inventory=seed_inventory(); env=environment(oc); runtime,public_runtime=runtime_files()
    protected_paths,protected_sha=protected(oc); launch=BC/'gpu-launch.local.json'
    if sha(launch)!=oc['gpu_launch_config_sha256']: raise ValueError('GPU launch changed')
    launch_data=read(launch)
    if any(not isinstance(launch_data.get(k),str) or not launch_data[k].strip()
           for k in ('expected_device','expected_device_kind')): raise ValueError('GPU device not registered')
    cfg=build_config(config,oc,nc,old,new,paths,hashes,inventory,env,sha(launch),protected_paths,protected_sha)
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    with (out/'gpu-launch.local.json').open('xb') as f: f.write(launch.read_bytes())
    write_new(out/'seed-inventory.local.json',inventory)
    cfg['seed_inventory_sha256']=sha(out/'seed-inventory.local.json')
    write_new(out/'registration.json',cfg)
    evidence={folder.name+'/'+n:sha(folder/n) for folder,names in ((BC,('registration.json','freeze.json','pre-execution-commit.json','root-result-acceptance.json','train-data-result.json','data-supervision-result.json','model-seal.json','schema.json')),(REFRESH,('registration.json','freeze.json','pre-execution-commit.json','root-result-acceptance.json','data-result.json','data-seal.json','data-worker-result.json','other-job-receipts.public.json','schema.json'))) for n in names}
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),parent_commit=BASE,runtime_files=runtime,public_runtime_sha256=public_runtime,new_source_sha256=source,historical_evidence_sha256=evidence))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve(): raise ValueError('Unexpected output')
    deadline_check(); cfg=read(out/'registration.json'); freeze=read(out/'freeze.json')
    if (freeze['parent_commit']!=BASE or freeze['registration_sha256']!=sha(out/'registration.json')
        or freeze['root_decision_sha256']!=sha(out/'root-decision.json') or cfg['baseline_head']!=BASE
        or cfg['training']!=TRAINING or cfg['host_limits']!=HOST_LIMITS or cfg['data_limits']!=DATA_LIMITS
        or cfg['retry_allowed'] is not False or cfg['no_official_games'] is not True or cfg['no_automatic_promotion'] is not True): raise ValueError('Registration changed')
    for name,digest in freeze['historical_evidence_sha256'].items():
        if sha(ROOT/'artifacts'/name)!=digest: raise ValueError('Historical evidence changed')
    oc,nc,old,new,paths,hashes=historical_admission(); inventory=read(out/'seed-inventory.local.json')
    if sha(out/'seed-inventory.local.json')!=cfg['seed_inventory_sha256']: raise ValueError('Seed inventory changed')
    # Re-scan so newly recognizable historical holdouts cannot be silently ignored.
    current=seed_inventory()
    if set(current['holdout_seeds'])-set(inventory['holdout_seeds']): raise ValueError('New historical holdout discovered')
    protected_paths,protected_sha=protected(oc); env=environment(oc) if os.name=='nt' else cfg['environment']
    rebuilt=build_config(cfg,oc,nc,old,new,paths,hashes,inventory,env,sha(out/'gpu-launch.local.json'),protected_paths,protected_sha)
    rebuilt['seed_inventory_sha256']=cfg['seed_inventory_sha256']
    if rebuilt!=cfg: raise ValueError('Frozen input identity changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')}!=set(freeze['new_source_sha256']): raise ValueError('Source set changed')
    local_runtime,public_runtime=runtime_files()
    if local_runtime!=freeze['runtime_files'] or public_runtime!=freeze['public_runtime_sha256']: raise ValueError('Runtime publication binding changed')
    for rel,digest in {**freeze['runtime_files'],**freeze['new_source_sha256']}.items():
        if sha(ROOT/rel)!=digest: raise ValueError('Runtime changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out); receipt=read(out/'pre-execution-commit.json'); commit=receipt['commit']
    clean_head(commit); git('merge-base','--is-ancestor',BASE,commit)
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes(): raise ValueError('Committed evidence differs')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed source differs')
    return cfg


def ready(out=OUT):
    out=Path(out).resolve(); cfg=verify_commit(out); review=read(out/'pre-review.json'); receipt=read(out/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256')!=sha(out/'freeze.json')
        or receipt.get('freeze_sha256')!=sha(out/'freeze.json') or receipt.get('registration_sha256')!=sha(out/'registration.json')):
        raise ValueError('Review or commit binding invalid')
    return cfg
