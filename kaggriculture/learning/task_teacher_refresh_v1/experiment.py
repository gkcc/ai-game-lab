"""Prospective identities and budgets for a small teacher-data collection.

Registration inventories seed metadata only. Real games and trace reads require
the separate, reviewed local commitment produced before execution.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / 'publication/ai-game-lab'
OUT = ROOT / 'artifacts/task-teacher-refresh-v1-20261027'
SOURCE = Path(__file__).resolve().parent
BASE = '4562f2b1a5424c8e611052e042aabeb8612284c8'
EVAL = ROOT / 'artifacts/task-teacher-choice-eval-v2-20261027'
BC = ROOT / 'artifacts/task-teacher-choice-bc-v1-20261027'
MIB, GIB = 1024**2, 1024**3
JOB = dict(max_rss_bytes=2*GIB, max_commit_bytes=3*GIB, min_available_bytes=2*GIB,
           wall_seconds=900, poll_seconds=.25, cleanup_seconds=10)
PROBE_JOB = dict(max_rss_bytes=GIB, max_commit_bytes=1536*MIB, min_available_bytes=2*GIB,
                 wall_seconds=120, poll_seconds=.25, cleanup_seconds=10)
DATA_JOB = dict(max_rss_bytes=1536*MIB, max_commit_bytes=2*GIB, min_available_bytes=2*GIB,
                wall_seconds=900, poll_seconds=.25, cleanup_seconds=10)
COLLECTION = dict(compressed_bytes=96*MIB, decoded_bytes=GIB, max_line_bytes=16*MIB,
                  max_groups=2500, max_candidates=150000, max_candidates_per_group=8192)
DATA = dict(min_groups=256, min_groups_per_seed=32, min_selected_operations=3,
            max_groups=2500, max_candidates=150000, max_archive_bytes=96*MIB,
            max_uncompressed_bytes=96*MIB)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Output outside owned experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def deadline_check():
    deadline = datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('Invalid or expired deadline')


def environment():
    venv = ROOT/'.venv'
    executable = venv/'Scripts/python.exe'
    if os.name != 'nt' or not os.path.samefile(sys.executable, executable) or not os.path.samefile(sys.prefix, venv):
        raise ValueError('Project Windows virtual environment required')
    if os.path.samefile(sys.prefix, sys.base_prefix):
        raise ValueError('Base interpreter is not the project virtual environment')
    return dict(interpreter_sha256=sha(executable))


def metadata_path_allowed(relative):
    """Exclude private path components even when capitalization differs."""
    return 'private' not in {part.casefold() for part in Path(relative).parts}


def inventory():
    """Seed-key metadata only; no raw replay, trace, tensor, or result rows."""
    from learning.task_teacher_choice_bc_v1.experiment import seed_values
    names = {'registration.json','input-manifest.json','root-decision.json',
             'seed-inventory.local.json','seed-inventory.json','seeds.json'}
    tracked = [n for n in git('ls-files','-z').decode('utf-8').split('\0') if n]
    files = [('public/'+n,PUBLIC/n) for n in tracked
             if Path(n).name.lower() in names and metadata_path_allowed(n)]
    artifacts = ROOT/'artifacts'
    files += [('local/'+p.relative_to(artifacts).as_posix(),p)
              for p in sorted(artifacts.rglob('*.json'))
              if not p.is_relative_to(OUT) and metadata_path_allowed(p.relative_to(artifacts))
              and (p.name.lower() in names or p.parent.name == 'job-inputs')]
    seen, identities, used = set(), [], 0
    for label, path in files:
        deadline_check()
        used += path.stat().st_size
        if used > 256*MIB:
            raise ValueError('Seed metadata budget exceeded')
        raw = path.read_bytes()
        seen.update(seed_values(json.loads(raw.decode('utf-8-sig')), path.name.lower() == 'seeds.json'))
        identities.append([label,hashlib.sha256(raw).hexdigest()])
    selected, index = [], 0
    while len(selected) < 4:
        digest = hashlib.sha256(f'task-teacher-refresh-v1:20261027:{index}'.encode('ascii')).digest()
        value = int.from_bytes(digest[:4], 'big')
        index += 1
        if value not in seen and value not in selected:
            selected.append(value)
    return dict(train_seeds=selected[:2], reserved_test_seeds=selected[2:], draws=index,
        metadata_files=len(files), metadata_bytes=used, identities=identities, excluded_uint32=sorted(seen),
        scope='Tracked and local non-private registration, seed, root-decision, input-manifest and job-input metadata',
        limitation='Deleted, private, unavailable, external and unlisted seed history cannot be excluded')


def _published_equal(folder, names):
    for name in names:
        if git('show',BASE+':review/evidence/'+folder.name+'/'+name) != (folder/name).read_bytes():
            raise ValueError('Prior evidence differs from published bytes')


def _clean_head(expected):
    if git('rev-parse','HEAD').decode().strip() != expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def prepare(out=OUT):
    out = Path(out).resolve()
    if out != OUT.resolve() or (out/'registration.json').exists():
        raise ValueError('Invalid or already registered experiment')
    deadline_check()
    decision = read(out/'root-decision.json')
    if decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution':
        raise ValueError('Missing prospective decision')
    _clean_head(BASE)
    _published_equal(EVAL,('registration.json','freeze.json','root-result-acceptance.json'))
    _published_equal(BC,('registration.json','freeze.json','train-data-result.json','model-seal.json'))
    eval_cfg, old = read(EVAL/'registration.json'), read(BC/'registration.json')
    if not read(EVAL/'root-result-acceptance.json')['panel_completed'] or not read(BC/'model-seal.json')['passed']:
        raise ValueError('Prior evidence not accepted')
    env = environment()
    if env['interpreter_sha256'] != eval_cfg['windows_venv_interpreter_sha256']:
        raise ValueError('Interpreter changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    from learning.task_reward_gate_v1.experiment import versions
    installed = installed_files()
    if (versions() != eval_cfg['runtime_versions'] or sha(installed['rules']) != eval_cfg['engine_sha256']
        or sha(installed['loader']) != eval_cfg['official_loader_sha256']):
        raise ValueError('Official environment changed')
    if shutil.disk_usage(ROOT).free < 8*GIB:
        raise ValueError('Insufficient free disk')
    crop = ROOT/eval_cfg['crop_checkpoint']
    if sha(crop) != eval_cfg['crop_checkpoint_sha256']:
        raise ValueError('Frozen crop checkpoint changed')
    rival = EVAL/'sources/round2.py'
    if sha(rival) != eval_cfg['source_sha256']['round2.py']:
        raise ValueError('Frozen opponent changed')
    schema = BC/'schema.json'
    if sha(schema) != old['schema_sha256']:
        raise ValueError('Teacher feature schema changed')
    runtime = dict(read(EVAL/'freeze.json')['runtime_files'])
    extras = ['learning/task_probe_failure_reporting_v1/report.py',
              'learning/task_teacher_dev_deviation_v1/membership.py']
    for rel in extras:
        data = git('show',BASE+':kaggriculture/'+rel)
        if data != (ROOT/rel).read_bytes():
            raise ValueError('Published dependency differs')
        runtime[rel] = hashlib.sha256(data).hexdigest()
    for rel,digest in runtime.items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Frozen dependency changed')
    source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    if not source:
        raise ValueError('Missing stage source')
    inv = inventory()
    target = out/'sources/round2.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        stream.write(rival.read_bytes())
    with (out/'schema.json').open('xb') as stream:
        stream.write(schema.read_bytes())
    write_new(out/'seed-inventory.local.json',inv)
    old_data = read(BC/'train-data-result.json')
    cfg = dict(experiment='task-teacher-refresh-v1', baseline_head=BASE,
        deadline_utc=decision['deadline_utc'],train_seeds=inv['train_seeds'],
        reserved_test_seeds=inv['reserved_test_seeds'],seed_inventory_sha256=sha(out/'seed-inventory.local.json'),
        seed_scope=inv['scope'],seed_history_limitation=inv['limitation'],opponents=['round2'],seats=[0,1],
        policies=['teacher_task'],crop_checkpoint=eval_cfg['crop_checkpoint'],
        crop_checkpoint_sha256=eval_cfg['crop_checkpoint_sha256'],source_sha256=dict(eval_cfg['source_sha256']),
        engine_commit=eval_cfg['engine_commit'],engine_sha256=eval_cfg['engine_sha256'],
        official_loader_sha256=eval_cfg['official_loader_sha256'],runtime_versions=eval_cfg['runtime_versions'],
        environment=env,schema_sha256=sha(out/'schema.json'),game_limits=eval_cfg['game_limits'],
        storage=eval_cfg['storage'],official_job_supervision=JOB,probe_job_supervision=PROBE_JOB,
        data_job_supervision=DATA_JOB,collection_limits=COLLECTION,data_limits=DATA,
        sampling_algorithm='sha256-compact-json-game-step-ordinal-first16-lt8192',
        official_limits=dict(new_games=4,live_transitions=4*719,primary_replay_transitions=4*719),
        followup_fit_recipe=dict(old_array_sha256=old_data['array_sha256'],old_groups=old_data['groups'],
            new_and_old_data='combine both complete sealed samples',sampling='existing variable-width packed uniform permutations',
            updates=128,training_seed=2026102722,automatic_fit_this_stage=False),
        retries=False,interpretation='Rule task selector plus frozen crop model; teacher choice is not task completion')
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'),runtime_files=runtime,
        new_source_sha256=source,parent_commit=BASE,parent_eval_sha256=sha(EVAL/'root-result-acceptance.json'),
        parent_bc_sha256=sha(BC/'train-data-result.json')))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    if out != OUT.resolve():
        raise ValueError('Unexpected output')
    deadline_check()
    cfg, freeze = read(out/'registration.json'), read(out/'freeze.json')
    if (sha(out/'registration.json') != freeze['registration_sha256']
        or sha(out/'root-decision.json') != freeze['root_decision_sha256']):
        raise ValueError('Registration identity changed')
    if (cfg['opponents'] != ['round2'] or cfg['seats'] != [0,1] or cfg['policies'] != ['teacher_task']
        or cfg['official_job_supervision'] != JOB or cfg['probe_job_supervision'] != PROBE_JOB
        or cfg['data_job_supervision'] != DATA_JOB or cfg['collection_limits'] != COLLECTION
        or cfg['data_limits'] != DATA or cfg['official_limits'] != dict(new_games=4,live_transitions=4*719,primary_replay_transitions=4*719)):
        raise ValueError('Plan or budget changed')
    inv = read(out/'seed-inventory.local.json')
    seeds = cfg['train_seeds']+cfg['reserved_test_seeds']
    if (sha(out/'seed-inventory.local.json') != cfg['seed_inventory_sha256'] or len(seeds) != 4
        or len(set(seeds)) != 4 or any(type(s) is not int or not 0 <= s < 2**32 or s in inv['excluded_uint32'] for s in seeds)
        or cfg['train_seeds'] != inv['train_seeds'] or cfg['reserved_test_seeds'] != inv['reserved_test_seeds']):
        raise ValueError('Seed identity changed')
    if (sha(out/'sources/round2.py') != cfg['source_sha256']['round2.py']
        or sha(out/'schema.json') != cfg['schema_sha256'] or sha(ROOT/cfg['crop_checkpoint']) != cfg['crop_checkpoint_sha256']
        or sha(EVAL/'root-result-acceptance.json') != freeze['parent_eval_sha256']
        or sha(BC/'train-data-result.json') != freeze['parent_bc_sha256']):
        raise ValueError('Prior identity changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']):
        raise ValueError('Stage source set changed')
    for rel,digest in {**freeze['runtime_files'],**freeze['new_source_sha256']}.items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Runtime identity changed')
    if os.name == 'nt':
        if environment() != cfg['environment']:
            raise ValueError('Interpreter changed')
        from learning.task_entry_repair_v1.experiment import installed_files
        from learning.task_reward_gate_v1.experiment import versions
        installed = installed_files()
        if (versions() != cfg['runtime_versions'] or sha(installed['rules']) != cfg['engine_sha256']
            or sha(installed['loader']) != cfg['official_loader_sha256']):
            raise ValueError('Official runtime changed')
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve()
    cfg = verify(out)
    receipt = read(out/'pre-execution-commit.json')
    commit = receipt['commit']
    _clean_head(commit)
    git('merge-base','--is-ancestor',cfg['baseline_head'],commit)
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes():
            raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes():
            raise ValueError('Committed source changed')
    return cfg


def ready(out=OUT):
    cfg = verify_commit(out)
    review = read(Path(out)/'pre-review.json')
    receipt = read(Path(out)/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256') != sha(Path(out)/'freeze.json')
        or receipt.get('freeze_sha256') != sha(Path(out)/'freeze.json')
        or receipt.get('registration_sha256') != sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    return cfg
