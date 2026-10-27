"""Preregister a bounded conditional-teacher data and fit experiment.

Preparation reads source/evidence metadata and hashes, never historical trace
contents or future test results. The public local commit is not a trusted clock.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .collect import SAMPLING_ALGORITHM

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication/ai-game-lab'
OUT = ROOT/'artifacts/task-teacher-choice-bc-v1-20261027'
SOURCE = Path(__file__).resolve().parent
ACTION = ROOT/'artifacts/task-action-observation-v1-20261027'
OLD = ROOT/'artifacts/task-completion-bc-v1-20261027'
MODEL = ROOT/'artifacts/task-completion-bc-reentry-v1-20261027'
COVERAGE = ROOT/'artifacts/task-completion-training-coverage-v1-20261027'
JOB = ROOT/'artifacts/task-training-job-supervision-v1-20261027'
ECONOMY = ROOT/'artifacts/task-completion-economy-v1-20261027'
BASE = 'fd05badc54f409b2c9bee35038e124f819316e06'
MIB, GIB = 1024**2, 1024**3
TRAIN_SEEDS = (3911611436,2328454240)
COLLECTION = dict(compressed_bytes=128*MIB,decoded_bytes=2*GIB,max_line_bytes=16*MIB,
                  max_groups=4000,max_candidates=250000,max_candidates_per_group=8192)
DATA = dict(max_groups=4000,max_candidates=250000,max_archive_bytes=128*MIB,max_uncompressed_bytes=128*MIB,
            min_groups=256,min_groups_per_seed=32,min_selected_operations=3)
SUPERVISION = dict(max_rss_bytes=1536*MIB,max_commit_bytes=2*GIB,min_available_bytes=2*GIB,
                   wall_seconds=900,poll_seconds=.25,cleanup_seconds=10)
TRAINING = dict(training_seed=2026102711,work_seconds=1180,process_seconds=1200,updates=128)
WSL_SOFTWARE = dict(python='3.12.3',numpy='2.5.4',jax='0.4.38',optax='0.2.4')
EVALUATION = dict(new_games=8,live_transitions=8*719,primary_replay_transitions=8*719,
                  independent_replay_transitions=719,total_transitions=17*719,opponents=['round2'],seats=[0,1],
                  seed_mean_own_min=-1000,seed_mean_margin_min=-1000,seat_own_min=-2000,seat_margin_min=-2000)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(MIB),b''): h.update(block)
    return h.hexdigest()


def write_new(path,value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def deadline_check():
    deadline = datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline: raise TimeoutError('Invalid or expired deadline')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip() != expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def windows_venv_sha():
    venv = ROOT/'.venv'
    executable = venv/'Scripts/python.exe'
    try:
        if (not os.path.samefile(sys.executable,executable) or not os.path.samefile(sys.prefix,venv)
            or os.path.samefile(sys.prefix,sys.base_prefix)):
            raise ValueError('Project Windows virtual environment required')
    except OSError as exc:
        raise ValueError('Project Windows virtual environment identity unavailable') from exc
    return sha(executable)


def public_evidence(folder,names):
    for name in names:
        if git('show',BASE+':review/evidence/'+folder.name+'/'+name) != (folder/name).read_bytes():
            raise ValueError('Published evidence changed')


def seed_values(value, active=False):
    """Collect seed metadata, never interpret score or outcome numbers as seeds."""
    found = set()
    containers = {'train','training','test','holdout','validation','eval','evaluation',
                  'development','dev','reserved','used','unused','selected','excluded','values'}
    if isinstance(value,dict):
        for key, item in value.items():
            label = str(key).lower()
            if active and label.isdigit():
                if int(label) < 2**32: found.add(int(label))
                found.update(seed_values(item,False))
                continue
            key_active = 'seed' in label or key == 'excluded_uint32' or (active and label in containers)
            found.update(seed_values(item,key_active))
    elif isinstance(value,list):
        for item in value: found.update(seed_values(item,False if isinstance(item,dict) else active))
    elif active and type(value) is int and 0 <= value < 2**32:
        found.add(value)
    elif active and isinstance(value,str) and value.isdecimal() and int(value) < 2**32:
        found.add(int(value))
    return found


def verified_coverage_runtime(local_bytes,published_bytes,acceptance):
    if local_bytes != published_bytes or hashlib.sha256(local_bytes).hexdigest() != acceptance['evidence_sha256']['freeze.json']:
        raise ValueError('Prior coverage freeze differs from the accepted publication')
    if acceptance['resource_supervision_accepted'] is not False:
        raise ValueError('Prior negative resource finding changed')
    runtime = json.loads(local_bytes.decode('utf-8-sig'))['runtime_files']
    if (not isinstance(runtime,dict) or not runtime
        or any(not isinstance(k,str) or not k or not isinstance(v,str) or len(v) != 64
               or any(c not in '0123456789abcdef' for c in v) for k,v in runtime.items())):
        raise ValueError('Prior runtime identity invalid')
    return dict(runtime)


def inventory():
    """Metadata-only collision exclusion; no trace, result or test rows."""
    names = {'registration.json','input-manifest.json','seed-inventory.local.json','seed-inventory.json','seeds.json'}
    tracked = git('ls-files','-z').decode('utf-8').split('\0')
    files = [(f'public/{n}',PUBLIC/n) for n in tracked if n and Path(n).name.lower() in names]
    artifacts = ROOT/'artifacts'
    files += [('local/'+p.relative_to(artifacts).as_posix(),p) for p in sorted(artifacts.rglob('*.json'))
              if not p.is_relative_to(OUT) and 'private' not in p.relative_to(artifacts).parts and p.name.lower() in names]
    seen, identities, total = set(TRAIN_SEEDS) | {TRAINING['training_seed']},[],0
    for label,path in files:
        deadline_check()
        total += path.stat().st_size
        if total > 256*MIB: raise ValueError('Seed metadata budget exceeded')
        data = path.read_bytes()
        obj = json.loads(data.decode('utf-8-sig'))
        seen.update(seed_values(obj,path.name.lower() == 'seeds.json'))
        identities.append([label,hashlib.sha256(data).hexdigest()])
    selected,index = [],0
    while len(selected) < 2:
        value = int.from_bytes(hashlib.sha256(f'task-teacher-choice-bc-v1:20261027:{index}'.encode()).digest()[:4],'big')
        if value not in seen and value not in selected: selected.append(value)
        index += 1
    return dict(test_seeds=selected,metadata_files=len(files),metadata_bytes=total,
                identities=identities,excluded_seeds=sorted(seen),draws=index,
                scope='Seed-named values in registration, input-manifest and seed-inventory metadata only',
                limitation='Unlisted metadata structures, unscanned, deleted, unavailable or external seed history cannot be excluded')


def source_attestation(old_registration, audits, manifest):
    rows = old_registration['source_groups']
    by_game = {r['game']:r for r in audits}
    if len(rows) != 8 or len(by_game) != len(audits): raise ValueError('Historical source count differs')
    audit_sha = sha(ACTION/'audit.json')
    if audit_sha != manifest['audit_sha256']: raise ValueError('Historical actor audit changed')
    sources, arms = [],{}
    for row in rows:
        audit = by_game[row['key']]
        if (not audit['passed'] or audit['replay_sha256'] != row['replay']['sha256']
            or audit['trace_sha256'] != row['trace']['sha256'] or audit['actor_actions_reproduced'] != 719
            or audit['rival_actions_reproduced'] != 719 or audit['trace_turns_reproduced'] != 719
            or audit['errors'] != 0):
            raise ValueError('Historical actor reconstruction differs')
        sources.append(dict(game=row['key'],side=row['side'],seed=row['seed'],trace=row['trace'],
                            replay_sha256=row['replay']['sha256'],audit_sha256=audit_sha))
        arms[row['key']] = row['arm']
    if len(arms) != 8 or set(x['seed'] for x in sources) != set(TRAIN_SEEDS):
        raise ValueError('Historical source identity differs')
    return sources,arms


def prepare(out=OUT):
    out = Path(out).resolve()
    if os.name != 'nt' or out != OUT.resolve(): raise ValueError('Registration requires the project Windows environment')
    if any((out/n).exists() for n in ('registration.json','freeze.json','data-started.json','fit-started.json')):
        raise ValueError('Registration or attempt already exists')
    deadline_check()
    decision = read(out/'root-decision.json')
    if decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution':
        raise ValueError('Invalid decision')
    clean_head(BASE)
    public_evidence(JOB,('root-result-acceptance.json','registration.json','freeze.json','normal-result.json',
                         'rss_stop-result.json','independent-result-assessment.json','root-next-learning-decision.json'))
    public_evidence(OLD,('registration.json','freeze.json','train-data-result.json','schema.json'))
    public_evidence(ACTION,('registration.json','input-manifest.json'))
    public_evidence(COVERAGE,('freeze.json','root-result-acceptance.json','analysis-result.json'))
    public_evidence(MODEL,('registration.json','freeze.json','root-result-acceptance.json','fit-result.json','model-seal.json'))
    public_evidence(ECONOMY,('registration.json','root-result-acceptance.json'))
    if not read(JOB/'root-result-acceptance.json')['passed'] or not read(JOB/'independent-result-assessment.json')['passed']:
        raise ValueError('Synthetic Job evidence not accepted')
    windows_python_sha = windows_venv_sha()
    if windows_python_sha != read(JOB/'registration.json')['environment']['interpreter_sha256']:
        raise ValueError('Windows Job interpreter differs')
    coverage = read(COVERAGE/'root-result-acceptance.json')
    coverage_public = git('show',BASE+':review/evidence/'+COVERAGE.name+'/freeze.json')
    coverage_runtime = verified_coverage_runtime((COVERAGE/'freeze.json').read_bytes(),coverage_public,coverage)
    model = read(MODEL/'root-result-acceptance.json')
    if not model['passed'] or not model['training_accepted']: raise ValueError('Reference model not accepted')
    for name in ('registration.json','freeze.json','fit-result.json','model-seal.json'):
        if model['evidence_sha256'][name] != sha(MODEL/name): raise ValueError('Reference model evidence changed')
    old,old_freeze = read(OLD/'registration.json'),read(OLD/'freeze.json')
    if sha(OLD/'registration.json') != old_freeze['registration_sha256'] or sha(OLD/'schema.json') != old_freeze['schema_sha256']:
        raise ValueError('Prior registration or schema changed')
    action_manifest = read(ACTION/'input-manifest.json')
    sources,arms = source_attestation(old,read(ACTION/'audit.json'),action_manifest)
    economy = read(ECONOMY/'registration.json')
    if (read(MODEL/'fit-result.json')['checkpoint_sha256'] != economy['candidate_checkpoint_sha256']
        or read(MODEL/'model-seal.json')['checkpoint_sha256'] != economy['candidate_checkpoint_sha256']):
        raise ValueError('Reference checkpoint identity differs')
    if economy['engine_commit'] != 'f994f2050d58d84f649329c1e35ce09ada7007c5':
        raise ValueError('Unexpected official rules commit')
    from learning.task_entry_repair_v1.experiment import installed_files
    from learning.task_reward_gate_v1.experiment import versions
    installed = installed_files()
    if (versions() != economy['runtime_versions'] or sha(installed['rules']) != economy['engine_sha256']
        or sha(installed['loader']) != economy['official_loader_sha256']):
        raise ValueError('Official runtime changed')
    if shutil.disk_usage(ROOT).free < 8*GIB: raise ValueError('Insufficient reserved disk space')
    schema_data = (OLD/'schema.json').read_bytes()
    launch_data = (MODEL/'gpu-launch.local.json').read_bytes()
    preflight = read(out/'wsl-dependency-preflight.json')
    if (not preflight['passed'] or preflight['versions'] != WSL_SOFTWARE
        or preflight['explicit_gpu_operations'] != 0 or preflight['training_data_reads'] != 0):
        raise ValueError('WSL dependency preflight differs')
    runtime = coverage_runtime
    runtime.update(read(JOB/'freeze.json')['new_source_sha256'])
    source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    runtime.update(source)
    for rel,digest in runtime.items():
        if sha(ROOT/rel) != digest: raise ValueError('Frozen runtime changed')
    inventory_result = inventory()
    write_new(out/'seed-inventory.local.json',inventory_result)
    write_new(out/'source-manifest.json',sources)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'schema.json').open('xb') as f: f.write(schema_data)
    with (out/'gpu-launch.local.json').open('xb') as f: f.write(launch_data)
    cfg = dict(experiment='task-teacher-choice-bc-v1',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        source_root=ACTION.relative_to(ROOT).as_posix(),attested_sources=sources,source_arms=arms,
        source_manifest_sha256=sha(out/'source-manifest.json'),schema_sha256=sha(out/'schema.json'),
        seed_inventory_sha256=sha(out/'seed-inventory.local.json'),train_seeds=list(TRAIN_SEEDS),
        reserved_test_seeds=inventory_result['test_seeds'],evaluation_plan=dict(EVALUATION),
        sampling_algorithm=SAMPLING_ALGORITHM,collection_limits=dict(COLLECTION),data_limits=dict(DATA),
        collection_supervision=dict(SUPERVISION),training=dict(TRAINING),storage=dict(process_log_bytes=16*MIB),
        gpu_launch_config_sha256=sha(out/'gpu-launch.local.json'),
        wsl_software_versions=dict(WSL_SOFTWARE),wsl_preflight_sha256=sha(out/'wsl-dependency-preflight.json'),
        windows_venv_interpreter_sha256=windows_python_sha,
        reference_checkpoint=economy['candidate_checkpoint'],reference_checkpoint_sha256=economy['candidate_checkpoint_sha256'],
        crop_checkpoint=economy['crop_checkpoint'],crop_checkpoint_sha256=economy['crop_checkpoint_sha256'],
        engine_commit=economy['engine_commit'],engine_sha256=economy['engine_sha256'],
        official_loader_sha256=economy['official_loader_sha256'],runtime_versions=economy['runtime_versions'],
        input_role='Eight mixed-controller development traces; conditional teacher choices, not teacher trajectories',
        no_old_test_data=True,no_new_games_this_stage=True,retries=False)
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        runtime_files=runtime,new_source_sha256=source,parent_commit=BASE))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    if out != OUT.resolve(): raise ValueError('Unexpected output')
    deadline_check()
    receipt = out/'pre-execution-commit.json'
    clean_head(read(receipt)['commit'] if receipt.exists() else BASE)
    cfg,freeze = read(out/'registration.json'),read(out/'freeze.json')
    if sha(out/'registration.json') != freeze['registration_sha256'] or sha(out/'root-decision.json') != freeze['root_decision_sha256']:
        raise ValueError('Registration changed')
    if (cfg['collection_limits'] != COLLECTION or cfg['data_limits'] != DATA or cfg['collection_supervision'] != SUPERVISION
        or cfg['training'] != TRAINING or cfg['evaluation_plan'] != EVALUATION or cfg['sampling_algorithm'] != SAMPLING_ALGORITHM):
        raise ValueError('Bounded configuration changed')
    if set(cfg['train_seeds']) != set(TRAIN_SEEDS) or len(cfg['reserved_test_seeds']) != 2 or set(cfg['reserved_test_seeds']) & set(TRAIN_SEEDS):
        raise ValueError('Seed role changed')
    for name,key in (('source-manifest.json','source_manifest_sha256'),('schema.json','schema_sha256'),
                     ('seed-inventory.local.json','seed_inventory_sha256'),('gpu-launch.local.json','gpu_launch_config_sha256'),
                     ('wsl-dependency-preflight.json','wsl_preflight_sha256')):
        if sha(out/name) != cfg[key]: raise ValueError('Bound input changed')
    if read(out/'source-manifest.json') != cfg['attested_sources']: raise ValueError('Source list changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']):
        raise ValueError('Source set changed')
    for rel,digest in freeze['runtime_files'].items():
        if sha(ROOT/rel) != digest: raise ValueError('Runtime changed')
    if os.name == 'nt':
        if windows_venv_sha() != cfg['windows_venv_interpreter_sha256']:
            raise ValueError('Windows virtual environment changed')
        from learning.task_entry_repair_v1.experiment import installed_files
        from learning.task_reward_gate_v1.experiment import versions
        installed = installed_files()
        if (versions() != cfg['runtime_versions'] or sha(installed['rules']) != cfg['engine_sha256']
            or sha(installed['loader']) != cfg['official_loader_sha256']): raise ValueError('Official runtime changed')
    return cfg


def ready(out=OUT):
    cfg = verify(out)
    review,receipt = read(Path(out)/'pre-review.json'),read(Path(out)/'pre-execution-commit.json')
    if (not review['passed'] or review['freeze_sha256'] != sha(Path(out)/'freeze.json')
        or receipt['freeze_sha256'] != sha(Path(out)/'freeze.json') or receipt['registration_sha256'] != sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    verify_commit(out)
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve()
    cfg = verify(out)
    commit = read(out/'pre-execution-commit.json')['commit']
    clean_head(commit)
    if git('merge-base','--is-ancestor',BASE,commit) != b'': raise ValueError('Invalid ancestry')
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json','source-manifest.json','schema.json','wsl-dependency-preflight.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes(): raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg
