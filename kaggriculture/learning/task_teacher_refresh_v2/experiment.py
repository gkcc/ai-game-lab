"""Prospective, single-use identities and limits for new teacher demonstrations."""
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
OUT = ROOT / 'artifacts/task-teacher-refresh-v2-20261028'
SOURCE = Path(__file__).resolve().parent
BASE = '6f848b40d295649adb58162a0ad1d7035913647b'
PANEL = ROOT / 'artifacts/task-teacher-weighted-eval-v2-20261028'
BC = ROOT / 'artifacts/task-teacher-choice-bc-v1-20261027'
OLD_REFRESH = ROOT / 'artifacts/task-teacher-refresh-v1-20261027'
JOB_TEST = ROOT / 'artifacts/task-job-empty-exit-v1-r2-20261028'
MIB, GIB = 1024**2, 1024**3
JOB = dict(max_rss_bytes=2*GIB, max_commit_bytes=3*GIB, min_available_bytes=2*GIB,
           wall_seconds=900, poll_seconds=.25, cleanup_seconds=10)
DATA_JOB = dict(max_rss_bytes=1536*MIB, max_commit_bytes=2*GIB, min_available_bytes=2*GIB,
                wall_seconds=900, poll_seconds=.25, cleanup_seconds=10)
PREFLIGHT = dict(samples=3, min_available_bytes=4*GIB, interval_seconds=10)
COLLECTION = dict(compressed_bytes=192*MIB, decoded_bytes=2*GIB, max_line_bytes=16*MIB,
                  max_groups=8000, max_candidates=400000, max_candidates_per_group=8192,
                  max_total_callbacks=40000, max_total_candidate_rows=2000000)
DATA = dict(min_groups=256, min_groups_per_seed=32, min_selected_operations=3,
            min_dig=32, min_fertilize=100, min_dig_per_seed=1,
            max_groups=8000, max_candidates=400000,
            max_archive_bytes=256*MIB, max_uncompressed_bytes=256*MIB)
INDEPENDENT = dict(compressed_bytes=128*MIB, decoded_bytes=GIB, seconds=600, sample_count=20)
OFFICIAL = dict(new_games=8, live_transitions=8*719, primary_replay_transitions=8*719,
                independent_replay_transitions=719, total_transitions=17*719)
SAMPLING = 'DIG/FERTILIZE:all; other teacher selections:sha256-first16-lt8192'
EXTRAS = (
    'learning/task_job_empty_exit_v1/sampling.py',
    'learning/task_job_sampling_race_v1/sampling.py',
    'learning/task_training_job_supervision_v1/job.py',
    'learning/task_teacher_choice_bc_v1/collect.py',
    'learning/task_teacher_choice_bc_v1/extract.py',
    'learning/task_teacher_choice_bc_v1/numeric.py',
    'learning/task_teacher_choice_bc_v1/data_worker.py',
    'learning/task_completion_bc_v1/data.py',
    'learning/task_action_impact_diagnostic_v1/analysis.py',
    'learning/task_action_observation_v1/analysis.py',
    'learning/task_teacher_dev_deviation_v1/membership.py',
    'learning/task_probe_failure_reporting_v1/report.py',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Output outside owned experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def deadline_check():
    end = datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if end.tzinfo is None or datetime.now(timezone.utc) >= end:
        raise TimeoutError('Stage deadline')


def clean_head(expected):
    if git('rev-parse', 'HEAD').decode().strip() != expected or git('status', '--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def public_equal(folder, names):
    for name in names:
        if git('show', BASE+':review/evidence/'+folder.name+'/'+name) != (folder/name).read_bytes():
            raise ValueError('Published predecessor differs')


def environment():
    venv, executable = ROOT/'.venv', ROOT/'.venv/Scripts/python.exe'
    if (os.name != 'nt' or not os.path.samefile(sys.executable, executable)
        or not os.path.samefile(sys.prefix, venv) or os.path.samefile(sys.prefix, sys.base_prefix)
        or sys.flags.optimize != 0 or 'PYTHONOPTIMIZE' in os.environ):
        raise ValueError('Expected non-optimized project Windows interpreter')
    from learning.task_reward_gate_v1.experiment import versions
    return dict(interpreter_sha256=sha(executable), versions=versions(),
                optimization_level=0, python_optimize_env_absent=True)


def inventory():
    """Recognized seed metadata only, without traces, tensors or outcome rows."""
    from learning.task_teacher_choice_bc_v1.experiment import seed_values
    names = {'registration.json', 'input-manifest.json', 'root-decision.json', 'seeds.json',
             'seed-inventory.json', 'seed-inventory.local.json'}
    def allowed(path):
        return 'private' not in {p.casefold() for p in Path(path).parts}
    tracked = [n for n in git('ls-files', '-z').decode('utf-8').split('\0') if n]
    files = [('public/'+n, PUBLIC/n) for n in tracked if allowed(n) and OUT.name not in Path(n).parts
             and (Path(n).name.lower() in names or Path(n).parent.name == 'job-inputs')]
    artifacts = ROOT/'artifacts'
    files += [('local/'+p.relative_to(artifacts).as_posix(), p) for p in sorted(artifacts.rglob('*.json'))
              if not p.is_relative_to(OUT) and allowed(p.relative_to(artifacts))
              and (p.name.lower() in names or p.parent.name == 'job-inputs')]
    seen, identities, used = set(), [], 0
    for label, path in files:
        deadline_check()
        used += path.stat().st_size
        if used > 256*MIB:
            raise ValueError('Seed metadata budget exceeded')
        data = path.read_bytes()
        seen.update(seed_values(json.loads(data.decode('utf-8-sig')), path.name.lower() == 'seeds.json'))
        identities.append([label, hashlib.sha256(data).hexdigest()])
    return dict(excluded_uint32=sorted(seen), identities=identities, metadata_files=len(files), metadata_bytes=used,
        scope='Recognized public and local non-private seed metadata, including prior reservations and failed panels',
        limitation='Private, deleted, unavailable, external or unlisted history cannot be excluded')


def validate_seed_roles(train, reserved, excluded):
    if (not isinstance(train, list) or len(train) != 2 or not isinstance(reserved, list) or len(reserved) != 4
        or any(type(s) is not int or not 0 <= s < 2**32 for s in train+reserved)
        or len(set(train+reserved)) != 6 or set(train+reserved) & set(excluded)):
        raise ValueError('Invalid or previously recognized seed roles')


def prepare(out=OUT):
    out = Path(out).resolve()
    if out != OUT.resolve() or (out/'registration.json').exists():
        raise ValueError('Already registered or unexpected output')
    deadline_check(); clean_head(BASE)
    decision = read(out/'root-decision.json')
    if decision.get('previous_commit') != BASE or decision.get('status') != 'approved_for_implementation_not_execution':
        raise ValueError('Missing prospective decision')
    public_equal(PANEL, ('registration.json', 'freeze.json', 'root-result-acceptance.json'))
    public_equal(BC, ('registration.json', 'model-seal.json', 'train-data-result.json'))
    public_equal(OLD_REFRESH, ('registration.json', 'freeze.json', 'root-result-acceptance.json'))
    public_equal(JOB_TEST, ('registration.json', 'freeze.json', 'root-result-acceptance.json'))
    panel, panel_freeze, acceptance = (read(PANEL/n) for n in ('registration.json', 'freeze.json', 'root-result-acceptance.json'))
    if acceptance.get('passed') is not True or acceptance.get('freeze_sha256') != sha(PANEL/'freeze.json'):
        raise ValueError('Prior panel identity changed')
    old_bc = read(BC/'registration.json')
    if read(BC/'model-seal.json').get('passed') is not True:
        raise ValueError('Prior schema source not sealed')
    schema = BC/'schema.json'
    if sha(schema) != old_bc['schema_sha256']:
        raise ValueError('Feature schema changed')
    env = environment()
    if env != panel['environment']:
        raise ValueError('Interpreter or dependencies changed')
    import psutil
    job_cfg, job_acceptance, job_freeze = (read(JOB_TEST/n) for n in
        ('registration.json', 'root-result-acceptance.json', 'freeze.json'))
    tested_sampler = 'learning/task_job_empty_exit_v1/sampling.py'
    if (job_acceptance.get('passed') is not True or job_acceptance.get('confirmed_empty_exits') != 1
        or job_acceptance.get('freeze_sha256') != sha(JOB_TEST/'freeze.json')
        or job_freeze['new_source_sha256'].get(tested_sampler) != sha(ROOT/tested_sampler)
        or job_cfg['environment']['psutil'] != psutil.__version__):
        raise ValueError('Verified Job sampler or dependency changed')
    from learning.task_entry_repair_v1.experiment import installed_files
    installed = installed_files()
    if sha(installed['rules']) != panel['engine_sha256'] or sha(installed['loader']) != panel['official_loader_sha256']:
        raise ValueError('Official runtime changed')
    if shutil.disk_usage(ROOT).free < 8*GIB:
        raise ValueError('Insufficient free disk')
    if sha(ROOT/panel['crop_checkpoint']) != panel['crop_checkpoint_sha256']:
        raise ValueError('Crop model changed')
    source_sha = {}
    for name in ('round2.py', 'market.py'):
        source_sha[name] = sha(PANEL/'sources'/name)
        if source_sha[name] != panel['source_sha256'][name]:
            raise ValueError('Opponent changed')
    runtime = dict(panel_freeze['runtime_files'])
    for rel in EXTRAS:
        data = git('show', BASE+':kaggriculture/'+rel)
        if data != (ROOT/rel).read_bytes():
            raise ValueError('Published dependency changed')
        runtime[rel] = hashlib.sha256(data).hexdigest()
    for rel, digest in runtime.items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Runtime identity changed')
    inv = inventory()
    train, reserved = decision.get('train_seeds'), decision.get('reserved_test_seeds')
    validate_seed_roles(train, reserved, inv['excluded_uint32'])
    source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    if not source:
        raise ValueError('Missing stage source')
    (out/'sources').mkdir(exist_ok=False)
    for name in source_sha:
        with (out/'sources'/name).open('xb') as stream:
            stream.write((PANEL/'sources'/name).read_bytes())
    with (out/'schema.json').open('xb') as stream:
        stream.write(schema.read_bytes())
    write_new(out/'seed-inventory.local.json', inv)
    cfg = dict(experiment='task-teacher-refresh-v2', baseline_head=BASE, deadline_utc=decision['deadline_utc'],
        train_seeds=train, reserved_test_seeds=reserved, seed_inventory_sha256=sha(out/'seed-inventory.local.json'),
        seed_scope=inv['scope'], seed_history_limitation=inv['limitation'], opponents=['round2','market'], seats=[0,1],
        policies=['teacher_task'], crop_checkpoint=panel['crop_checkpoint'], crop_checkpoint_sha256=panel['crop_checkpoint_sha256'],
        source_sha256=source_sha, engine_commit=panel['engine_commit'], engine_sha256=panel['engine_sha256'],
        official_loader_sha256=panel['official_loader_sha256'], environment=env, schema_sha256=sha(out/'schema.json'),
        sampler_psutil_version=psutil.__version__,
        game_limits=panel['game_limits'], storage=panel['storage'], official_job_supervision=JOB,
        data_job_supervision=DATA_JOB, independent_job_supervision=JOB, host_preflight=PREFLIGHT,
        collection_limits=COLLECTION, data_limits=DATA, independent_limits=INDEPENDENT, official_limits=OFFICIAL,
        sampling_algorithm=SAMPLING, retry_allowed=False, no_training=True, no_automatic_promotion=True,
        independent_selection='first train seed, round2, seat 0; deterministic 20 callback sample',
        interpretation='Conditional teacher choices; not proof of execution, completion or improved performance')
    write_new(out/'registration.json', cfg)
    write_new(out/'freeze.json', dict(registration_sha256=sha(out/'registration.json'),
        root_decision_sha256=sha(out/'root-decision.json'), parent_commit=BASE, runtime_files=runtime,
        new_source_sha256=source, prior_panel_acceptance_sha256=sha(PANEL/'root-result-acceptance.json'),
        prior_panel_freeze_sha256=sha(PANEL/'freeze.json'), bc_registration_sha256=sha(BC/'registration.json'),
        old_refresh_acceptance_sha256=sha(OLD_REFRESH/'root-result-acceptance.json'),
        job_test_registration_sha256=sha(JOB_TEST/'registration.json'),
        job_test_acceptance_sha256=sha(JOB_TEST/'root-result-acceptance.json'),
        job_test_freeze_sha256=sha(JOB_TEST/'freeze.json')))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    if out != OUT.resolve():
        raise ValueError('Unexpected output')
    deadline_check(); cfg, freeze = read(out/'registration.json'), read(out/'freeze.json')
    if (sha(out/'registration.json') != freeze['registration_sha256'] or sha(out/'root-decision.json') != freeze['root_decision_sha256']
        or sha(PANEL/'root-result-acceptance.json') != freeze['prior_panel_acceptance_sha256']
        or sha(PANEL/'freeze.json') != freeze['prior_panel_freeze_sha256']
        or sha(BC/'registration.json') != freeze['bc_registration_sha256']
        or sha(OLD_REFRESH/'root-result-acceptance.json') != freeze['old_refresh_acceptance_sha256']
        or sha(JOB_TEST/'registration.json') != freeze['job_test_registration_sha256']
        or sha(JOB_TEST/'root-result-acceptance.json') != freeze['job_test_acceptance_sha256']
        or sha(JOB_TEST/'freeze.json') != freeze['job_test_freeze_sha256']):
        raise ValueError('Registration or predecessor changed')
    if (cfg['baseline_head'] != BASE or cfg['opponents'] != ['round2','market'] or cfg['seats'] != [0,1]
        or cfg['policies'] != ['teacher_task'] or cfg['official_job_supervision'] != JOB
        or cfg['data_job_supervision'] != DATA_JOB or cfg['independent_job_supervision'] != JOB
        or cfg['host_preflight'] != PREFLIGHT or cfg['collection_limits'] != COLLECTION
        or cfg['data_limits'] != DATA or cfg['independent_limits'] != INDEPENDENT or cfg['official_limits'] != OFFICIAL
        or cfg['sampling_algorithm'] != SAMPLING or cfg['retry_allowed'] is not False
        or cfg['no_training'] is not True or cfg['no_automatic_promotion'] is not True):
        raise ValueError('Panel or budget changed')
    inv, decision = read(out/'seed-inventory.local.json'), read(out/'root-decision.json')
    if sha(out/'seed-inventory.local.json') != cfg['seed_inventory_sha256']:
        raise ValueError('Seed inventory changed')
    validate_seed_roles(cfg['train_seeds'], cfg['reserved_test_seeds'], inv['excluded_uint32'])
    if cfg['train_seeds'] != decision['train_seeds'] or cfg['reserved_test_seeds'] != decision['reserved_test_seeds']:
        raise ValueError('Seed roles changed')
    for name, digest in cfg['source_sha256'].items():
        if sha(out/'sources'/name) != digest:
            raise ValueError('Opponent source changed')
    if sha(out/'schema.json') != cfg['schema_sha256'] or sha(ROOT/cfg['crop_checkpoint']) != cfg['crop_checkpoint_sha256']:
        raise ValueError('Schema or crop model changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']):
        raise ValueError('Stage source set changed')
    for rel, digest in {**freeze['runtime_files'], **freeze['new_source_sha256']}.items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Runtime source changed')
    if os.name == 'nt':
        import psutil
        if psutil.__version__ != cfg['sampler_psutil_version']:
            raise ValueError('Job sampling dependency changed')
        if environment() != cfg['environment']:
            raise ValueError('Windows runtime changed')
        from learning.task_entry_repair_v1.experiment import installed_files
        installed = installed_files()
        if sha(installed['rules']) != cfg['engine_sha256'] or sha(installed['loader']) != cfg['official_loader_sha256']:
            raise ValueError('Official runtime changed')
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve(); cfg = verify(out)
    receipt = read(out/'pre-execution-commit.json'); commit = receipt['commit']
    clean_head(commit); git('merge-base', '--is-ancestor', cfg['baseline_head'], commit)
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show', commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes():
            raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show', commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes():
            raise ValueError('Committed source changed')
    return cfg


def ready(out=OUT):
    cfg = verify_commit(out); review = read(Path(out)/'pre-review.json'); receipt = read(Path(out)/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256') != sha(Path(out)/'freeze.json')
        or receipt.get('freeze_sha256') != sha(Path(out)/'freeze.json')
        or receipt.get('registration_sha256') != sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    return cfg
