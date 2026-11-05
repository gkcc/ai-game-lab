"""One bounded teacher-task diagnostic on four already completed seed roles.

Run this file with the game venv's ``python -I -B``. Preparation reads only
completed evidence; it starts no match. The registered run uses the original
official Worker and its one main replay, with the old teacher-specific verifier.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
PRIVATE = REPO.parent / 'private' / 'task-teacher-fresh-panel-v1-20261105'
EVIDENCE = 'review/evidence/task-teacher-fresh-panel-v1-20261105'
REGISTRATION = EVIDENCE + '/registration.json'
BASE = '77127e1ce63924196e5660f346224e1f4f7a5770'
PREFIX = 'kaggriculture/learning/task_teacher_fresh_panel_v1/'
OLD_REGISTRATION = 'review/evidence/task-pair-fresh-multi-v1-20261031/registration.json'
OLD_COMMIT = '40f3f393819bbdc62af1007877cd5fbdf3a9a42f'
IDENTITY = 'kaggriculture/learning/task_pair_dev_campaign_v1/identity.py'
BOOTSTRAP = 'kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py'
BASE_ENTRY = 'kaggriculture/learning/task_pair_market_consumed_v1/entry.py'
STAGES = {c: GAME / 'artifacts' / f'task-teacher-fresh-panel-{c.lower()}-v1-20261105'
          for c in ('C01', 'C02')}
MAX_SMALL = 512 * 1024
MAX_METADATA = 16 * 1024**2
SCHEMA = 'task-teacher-fresh-panel-prereg-v1'


def require(value):
    if not value:
        raise ValueError('Registered teacher diagnostic differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out)
            out[k] = v
        return out
    def bad(_):
        raise ValueError('Nonfinite metadata')
    require(type(raw) is bytes and 0 < len(raw) <= MAX_METADATA)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    require(type(value) is dict)
    return value


def read(path, maximum=MAX_METADATA):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file())
    for item in (path, *path.parents):
        info = item.lstat()
        require(not item.is_symlink() and not (getattr(info, 'st_file_attributes', 0) & 0x400))
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum)
    return raw


def write(path, value, maximum=MAX_METADATA):
    raw = (json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(raw) <= maximum)
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent and not os.path.lexists(path))
    with path.open('xb') as stream:
        require(stream.write(raw) == len(raw))
        stream.flush()
        os.fsync(stream.fileno())
    require(read(path, maximum) == raw)
    return sha(raw)


def git(*args):
    result = subprocess.run(['git', '-C', str(REPO), *args], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0 and len(result.stdout) <= MAX_METADATA)
    return result.stdout


def snapshot(relative, expected, commit=None):
    raw = read(REPO / relative)
    require(sha(raw) == expected and (commit is None or raw == git('show', commit + ':' + relative)))
    module = ModuleType('_teacher_snapshot_' + Path(relative).stem)
    module.__file__ = str(REPO / relative)
    exec(compile(raw, module.__file__, 'exec', dont_inherit=True), module.__dict__)
    return module


def _required_sources(old):
    result = set(old['public_sources'])
    for relative in ('kaggriculture/learning/task_student_continuation_v1/__init__.py',
                     'kaggriculture/learning/task_student_continuation_v1/sources.py',
                     'kaggriculture/learning/task_student_continuation_v1/samples.py',
                     'kaggriculture/learning/task_teacher_market_dev_v1/__init__.py',
                     'kaggriculture/learning/task_teacher_market_dev_v1/entry.py'):
        result.add(relative)
    result.update(p.relative_to(REPO).as_posix() for p in (REPO / PREFIX).glob('*.py'))
    note = EVIDENCE + '/REGISTRATION.zh-CN.md'
    if (REPO / note).is_file():
        result.add(note)
    require({PREFIX + n for n in ('__init__.py', 'contract.py', 'sources.py', 'runner.py')}.issubset(result))
    return result


def validate_registration(reg, contract):
    require(reg.get('schema') == SCHEMA and reg.get('base_commit') == BASE
        and json.dumps(reg.get('limits'), sort_keys=True, allow_nan=False)
            == json.dumps(contract.LIMITS, sort_keys=True, allow_nan=False)
        and reg.get('engine_commit') == 'f994f2050d58d84f649329c1e35ce09ada7007c5'
        and reg.get('engine_sha256') == 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
        and reg.get('panel') == dict(cohorts=contract.COHORTS, opponents=list(contract.OPPONENTS),
             sides=list(contract.SIDES), policy='teacher_task', games=16, official_replays=16,
             source_student_games=32, seed_clusters=4, scope='consumed_development')
        and reg.get('evaluation') == dict(training=False, gpu=False, new_seeds=False,
             independent_evaluation=False, statistical_significance=False, promotion=False,
             transaction_cash_reconciliation=False, retries=False)
        and type(reg.get('binding_commitment_sha256')) is str
        and re.fullmatch(r'[0-9a-f]{64}', reg['binding_commitment_sha256']) is not None
        and reg.get('old_result_commit') == '0e9734a0109a0ee4822ef1310e8b56fac21eaa48')


def authenticate(commit):
    """Stdlib-only source/runtime verification before any private source read."""
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
        and type(commit) is str and re.fullmatch(r'[0-9a-f]{40}', commit) is not None
        and git('rev-parse', 'HEAD').decode().strip() == commit
        and not git('status', '--porcelain', '--untracked-files=all'))
    git('merge-base', '--is-ancestor', BASE, commit)
    raw = read(REPO / REGISTRATION)
    require(raw == git('show', commit + ':' + REGISTRATION))
    reg = decode(raw)
    identities = reg['public_sources']
    contract = snapshot(PREFIX + 'contract.py', identities[PREFIX + 'contract.py'], commit)
    validate_registration(reg, contract)
    old = decode(git('show', OLD_COMMIT + ':' + OLD_REGISTRATION))
    required = _required_sources(old)
    require(required.issubset(identities))
    identity = snapshot(IDENTITY, identities[IDENTITY], commit)
    identity.REGISTRATION = REGISTRATION
    identity._REQUIRED = frozenset(required)
    identity._panel = lambda value: validate_registration(value, contract)
    verified, environment, freeze = identity.verify_public(repo=REPO, game_root=GAME,
                                                           registration_commit=commit)
    require(verified == reg and reg['versions'] == _versions()
        and reg['versions']['python'] == environment['environment']['versions']['python'])
    for prefix in ('task_teacher_fresh_panel_v1', 'task_teacher_market_dev_v1',
                   'task_student_continuation_v1', 'task_pair_fresh_multi_v1'):
        require(not os.path.lexists(GAME / 'learning' / prefix))
    return reg, contract, identity, snapshot(BOOTSTRAP, identities[BOOTSTRAP], commit), snapshot(
        BASE_ENTRY, identities[BASE_ENTRY], commit)


def _versions():
    return dict(python='.'.join(map(str, sys.version_info[:3])),
        numpy=importlib.metadata.version('numpy'), psutil=importlib.metadata.version('psutil'),
        kaggle_environments=importlib.metadata.version('kaggle-environments'))


def preparation_runtime(identity, head):
    """Check the existing frozen runtime without fabricating a committed plan."""
    identity._historical_lineage(REPO, head)
    freeze = decode(read(REPO / identity.HISTORICAL_FREEZE))
    old_reg = decode(read(REPO / identity.HISTORICAL_REGISTRATION))
    local, public = freeze['runtime_files'], freeze['public_runtime_sha256']
    supplemental = freeze['supplemental_runtime_sha256']
    require(type(local) is dict and type(public) is dict and len(local) == len(public) == 297
        and set(local) == set(public) and type(supplemental) is dict and len(supplemental) == 23)
    for rel, expected in local.items():
        identity._check_digest(GAME, rel, expected)
        identity._check_digest(REPO / 'kaggriculture', rel, public[rel])
    for rel, expected in supplemental.items():
        identity._check_digest(GAME, rel, expected['local_sha256'])
        identity._check_digest(REPO / 'kaggriculture', rel, expected['public_sha256'])
    environment = old_reg['environment']
    require(sha(read(sys.executable)) == environment['interpreter_sha256']
        and all(_versions()[key] == environment['versions'][key]
                for key in ('python', 'numpy', 'kaggle_environments')))
    spec = importlib.util.find_spec('kaggle_environments')
    require(spec is not None and spec.submodule_search_locations is not None
        and len(spec.submodule_search_locations) == 1)
    package = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
    rules = read(package / 'envs/kaggriculture/kaggriculture.py')
    require(sha(rules) == identity.ENGINE_SHA and sha(read(package / 'agent.py')) == old_reg['official_loader_sha256']
        and sha(read(package / 'core.py')) == old_reg['official_core_sha256']
        and identity._git(GAME / 'upstream', 'rev-parse', 'HEAD').decode().strip() == identity.ENGINE_COMMIT
        and identity._git(GAME / 'upstream', 'show', identity.ENGINE_COMMIT +
            ':kaggle_environments/envs/kaggriculture/kaggriculture.py') == rules)


def _runtime(bootstrap, reg):
    modules = bootstrap._runtime(GAME, reg)
    result = dict(modules)
    for key, name in (('sources', 'task_teacher_fresh_panel_v1.sources'),
                      ('teacher', 'task_teacher_market_dev_v1.entry'),
                      ('fresh', 'task_pair_fresh_multi_v1.entry'),
                      ('worker', 'task_pair_dev_games_v1.worker')):
        module = importlib.import_module('learning.' + name)
        require(Path(module.__file__).resolve(strict=True).is_relative_to(REPO / 'kaggriculture'))
        result[key] = module
    return result


def prepare():
    """Freeze sources and completed metadata; do not generate seeds or games."""
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
    head = git('rev-parse', 'HEAD').decode().strip()
    git('merge-base', '--is-ancestor', BASE, head)
    require(not os.path.lexists(PRIVATE) and not os.path.lexists(REPO / REGISTRATION)
        and all(not os.path.lexists(s) for s in STAGES.values()))
    old = decode(git('show', OLD_COMMIT + ':' + OLD_REGISTRATION))
    source_map = {p: sha(read(REPO / p)) for p in sorted(_required_sources(old))}
    contract = snapshot(PREFIX + 'contract.py', source_map[PREFIX + 'contract.py'])
    base = snapshot(BASE_ENTRY, source_map[BASE_ENTRY])
    deadline = time.monotonic() + 900
    def check():
        if base.physical_available() < contract.LIMITS['running_available_bytes']:
            raise MemoryError('Preparation memory floor')
        if time.monotonic() >= deadline:
            raise TimeoutError('Preparation deadline')
        return True
    contract.await_memory(base.physical_available)
    identity = snapshot(IDENTITY, source_map[IDENTITY])
    preparation_runtime(identity, head)
    reg = dict(schema=SCHEMA, base_commit=BASE, old_result_commit='0e9734a0109a0ee4822ef1310e8b56fac21eaa48',
        engine_commit=identity.ENGINE_COMMIT, engine_sha256=identity.ENGINE_SHA,
        public_sources=source_map, versions=_versions(), limits=contract.LIMITS,
        panel=dict(cohorts=contract.COHORTS, opponents=list(contract.OPPONENTS),
             sides=list(contract.SIDES), policy='teacher_task', games=16, official_replays=16,
             source_student_games=32, seed_clusters=4, scope='consumed_development'),
        evaluation=dict(training=False, gpu=False, new_seeds=False, independent_evaluation=False,
             statistical_significance=False, promotion=False, transaction_cash_reconciliation=False,
             retries=False))
    bootstrap = snapshot(BOOTSTRAP, source_map[BOOTSTRAP])
    modules = _runtime(bootstrap, reg)
    context = modules['sources'].load(guard=check)
    for manifest in context.manifests.values():
        require(base.validate_assets(manifest, modules['driver']))
    binding = dict(schema='task-teacher-fresh-panel-binding-v1', base_commit=BASE,
        salt=secrets.token_hex(32), completed_sources=context.identities,
        default_sha256=sha(read(GAME / 'main.py')), source_parent_commit=head)
    require(all(sha(read(REPO / p)) == h for p, h in source_map.items()) and check())
    PRIVATE.mkdir(exist_ok=False)
    commitment = write(PRIVATE / 'binding.local.json', binding)
    reg['binding_commitment_sha256'] = commitment
    validate_registration(reg, contract)
    (REPO / EVIDENCE).mkdir(exist_ok=True, parents=True)
    write(REPO / REGISTRATION, reg)
    return dict(status='prepared_no_games', frozen_games=16, old_games_authenticated=32,
                source_replay_decoding=False, training=False)


def marker(kind, commit, binding_sha, cohort=None, game=None):
    require(kind in ('attempt', 'permit', 'started', 'consumed'))
    value = dict(schema='task-teacher-fresh-panel-marker-v1', kind=kind,
        registration_commit=commit, binding_sha256=binding_sha, retry_allowed=False)
    if kind != 'attempt':
        require(cohort in STAGES and type(game) is str)
        value.update(cohort=cohort, game=game)
    return value


def _bound(reg):
    raw = read(PRIVATE / 'binding.local.json')
    value = decode(raw)
    require(sha(raw) == reg['binding_commitment_sha256']
        and value.get('schema') == 'task-teacher-fresh-panel-binding-v1'
        and value.get('base_commit') == BASE)
    return raw, value


@contextmanager
def hidden_official_entry(module, contract):
    """Patch the actual run_match parser used inside the frozen Worker."""
    run_match = importlib.import_module('run_match')
    require(Path(run_match.__file__).resolve(strict=True) == (GAME / 'run_match.py').resolve(strict=True))
    original = module.official_entry
    def invoke():
        with contract.hide_parsed_argv(run_match):
            return original()
    module.official_entry = invoke
    try:
        yield
    finally:
        module.official_entry = original


def execute_child(modules, manifest, stage, plan, check, contract):
    dm, worker, teacher = modules['driver'], modules['worker'], modules['teacher']
    source = dm._read(GAME, GAME / worker.WORKER_RELATIVE, worker.MAX_WORKER_BYTES)
    require(sha(source) == manifest.panel.worker_sha)
    worker._single_python_thread()
    with worker._quiet_output(), contract.teacher_adapter(teacher):
        with teacher.bound_worker(worker, source, manifest, GAME, stage, plan, check) as (module, cfg, expected):
            bot = module.wrapper(stage, plan, cfg)
            require(bot == stage / 'agents' / (plan['game'] + '.py') and check())
            rival = dm._hash(stage, stage / 'sources' / (plan['opponent'] + '.py'), dm.MAX_SOURCE)
            require(rival == manifest.opponent(plan['opponent'])[1])
            job = dict(plan, bot_sha256=dm._hash(stage, bot, dm.MAX_SOURCE),
                       rival_sha256=rival, task_checkpoint_sha256=None)
            contract.validate_plan(job, plan, job=True)
            expected[0] = job
            dm._write_new(stage, stage / 'job-inputs' / (plan['game'] + '.json'), job,
                          limit=worker.MAX_JOB_BYTES)
            with hidden_official_entry(module, contract):
                module.job(stage, plan['game'])
            require(check())


def verify_game_files(modules, manifest, stage, plan, row, contract):
    dm, worker, acceptance = modules['driver'], modules['worker'], modules['acceptance']
    from learning.task_pair_dev_games_v1 import capture
    job = acceptance._json(dm._read(stage, stage / 'job-inputs' / (plan['game'] + '.json'),
                                    worker.MAX_JOB_BYTES), worker.MAX_JOB_BYTES)
    contract.validate_plan(job, plan, job=True)
    require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in job.items()))
    source, digest = manifest.opponent(plan['opponent'])
    checks = ((GAME, GAME / manifest.panel.crop[0], dm.MAX_MODEL, manifest.panel.crop[1]),
        (GAME, GAME / source, dm.MAX_SOURCE, digest),
        (stage, stage / 'sources' / (plan['opponent'] + '.py'), dm.MAX_SOURCE, digest),
        (GAME, GAME / worker.WORKER_RELATIVE, worker.MAX_WORKER_BYTES, manifest.panel.worker_sha),
        (stage, stage / 'agents' / (plan['game'] + '.py'), dm.MAX_SOURCE, job['bot_sha256']),
        (stage, stage / 'traces' / (plan['game'] + '.jsonl.gz'), capture.MAX_COMPRESSED_BYTES, row['trace_sha256']),
        (stage, stage / 'games' / plan['game'] / 'replay.json.gz', capture.MAX_COMPRESSED_BYTES, row['replay_gzip_sha256']))
    require(all(dm._hash(root, path, maximum) == expected for root, path, maximum, expected in checks)
        and job['rival_sha256'] == digest)
    return True


def closing_memory(error, latest_memory, reader):
    """Keep the last pre-cleanup observation separate from a later close read.

    Samples are discrete observations. An arbitrary exception may occur after
    the last observation; this does not diagnose its allocation failure cause.
    """
    before = dict(latest_memory)
    boundary = None if error is None else dict(last_observation_before_close=before,
        gate_last_available_bytes=getattr(error, 'last_available_bytes', None),
        scope='discrete_sample_not_continuous_monitoring_or_exception_cause')
    try:
        reader()
        close = dict(latest_memory, sample_error=None)
    except BaseException as sample_error:
        close = dict(available_bytes=None, sample_monotonic=None,
            sample_error=type(sample_error).__name__, prior_sample=before)
    return dict(failure_boundary_observation=boundary, memory_at_close=close)


def observed_api_type(base, floor, observations, clock=time.monotonic):
    """Observe the same available() value used by the existing Job loop.

    This does not change that loop or its cleanup. Keep three bounded samples,
    not a per-second log. The stop reason remains the supervisor's own field.
    """
    class ObservedAPI(base):
        def available(self):
            try:
                value = super().available()
                require(type(value) is int and value >= 0)
                sample = dict(available_bytes=value, sample_monotonic=clock())
            except BaseException as error:
                observations['sample_error'] = dict(error_kind=type(error).__name__,
                                                    sample_monotonic=clock())
                raise
            observations['count'] += 1
            observations['last'] = sample
            minimum = observations['minimum']
            if minimum is None or value < minimum['available_bytes']:
                observations['minimum'] = sample
            if value < floor and observations['first_below_floor'] is None:
                observations['first_below_floor'] = sample
            return value
    return ObservedAPI


def supervise_recorded(campaign, command, cwd, limits):
    """Reuse the exact existing owned-Job backend, recording its memory reads."""
    from learning.task_training_job_supervision_v1.job import WindowsAPI
    from learning.task_job_empty_exit_v1.sampling import supervise
    from learning.panel_eval_v3.launch import WindowsJob
    observations = dict(count=0, last=None, minimum=None, first_below_floor=None,
        sample_error=None, scope='discrete_supervisor_available_calls_not_continuous_or_exception_cause')
    quiet = campaign.quiet_api_type(WindowsAPI, WindowsJob.create_suspended)
    api = observed_api_type(quiet, limits['min_available_bytes'], observations)()
    result = supervise(api, command, cwd, dict(limits))
    result['available_memory_samples'] = observations
    return result


def run(commit, *, preflight=False, cohort=None, game=None):
    reg, contract, identity, bootstrap, base = authenticate(commit)
    binding_raw, binding = _bound(reg)
    child = game is not None
    require((cohort in STAGES if child else cohort is None) and not (preflight and child))
    limits, commitment = contract.LIMITS, sha(binding_raw)
    started = time.monotonic()
    latest_memory = {'available_bytes': None, 'sample_monotonic': None}
    def available():
        value = base.physical_available()
        latest_memory.update(available_bytes=value, sample_monotonic=time.monotonic())
        return value
    import psutil
    owner = (decode(read(PRIVATE / 'owner.local.json')) if child else
        dict(schema='task-pair-fresh-multi-owner-v1', registration_commit=commit,
            binding_sha256=commitment, pid=os.getpid(),
            created_us=int(psutil.Process().create_time() * 1_000_000),
            deadline=started + limits['total_wall_seconds']))
    owned, accepted, supervision_receipts = False, [], []
    stages = dict(STAGES)
    def check():
        if available() < limits['running_available_bytes']:
            raise MemoryError('Runtime memory floor')
        if time.monotonic() >= owner['deadline']:
            raise TimeoutError('Campaign deadline')
        require(read(PRIVATE / 'binding.local.json') == binding_raw
            and not os.path.lexists(PRIVATE / 'closed.local.json'))
        return True
    def ready():
        remaining = owner['deadline'] - time.monotonic()
        return contract.await_memory(available, timeout=min(limits['readiness_timeout_seconds'], remaining))
    try:
        if not child:
            require(not os.path.lexists(PRIVATE / 'attempt.local.json')
                and not os.path.lexists(PRIVATE / 'owner.local.json')
                and not os.path.lexists(PRIVATE / 'closed.local.json')
                and all(not os.path.lexists(s) for s in stages.values()))
            if not preflight:
                write(PRIVATE / 'attempt.local.json', marker('attempt', commit, commitment))
                owned = True
                write(PRIVATE / 'owner.local.json', owner)
            ready()
        else:
            check()
        modules = _runtime(bootstrap, reg)
        dm, acceptance, teacher = modules['driver'], modules['acceptance'], modules['teacher']
        if child:
            modules['fresh'].verify_live_owner(owner, commit=commit, binding_sha=commitment,
                current_pid=os.getpid(), process=psutil.Process, executable=sys.executable,
                samefile=os.path.samefile, now=time.monotonic())
            require(decode(read(PRIVATE / 'attempt.local.json')) == marker('attempt', commit, commitment))
        context = modules['sources'].load(guard=check, verify_game_files=not child)
        require(context.identities == binding['completed_sources']
            and sha(read(GAME / 'main.py')) == binding['default_sha256'])
        for manifest in context.manifests.values():
            require(base.validate_assets(manifest, dm))
        require(shutil.disk_usage(GAME / 'artifacts').free >= limits['disk_floor_bytes']
                + limits['per_game_reserve_bytes'] + limits['private_result_bytes'])
        if preflight:
            return dict(status='ready', games_started=0, old_games_authenticated=32,
                        source_replay_decoding=False)
        if child:
            stage, manifest = stages[cohort], context.manifests[cohort]
            plans = [p for p in contract.plans(manifest.panel) if p['game'] == game]
            require(len(plans) == 1 and read(stage / 'manifest.local.json') == binding_raw
                and decode(read(stage / 'job-results' / (game + '.permit.local.json'))) ==
                    marker('permit', commit, commitment, cohort, game))
            for kind in ('started', 'consumed'):
                write(stage / 'job-results' / (game + '.' + kind + '.local.json'),
                      marker(kind, commit, commitment, cohort, game))
            execute_child(modules, manifest, stage, plans[0], check, contract)
            require(check() and sha(read(GAME / 'main.py')) == binding['default_sha256'])
            identity.verify_public(repo=REPO, game_root=GAME, registration_commit=commit)
            return None
        for current, stage in stages.items():
            manifest = context.manifests[current]
            driver = dm.LocalDriver(manifest=manifest, game_root=GAME, stage=stage,
                commit=commit, command_for=lambda _name: [], check=check)
            require(check() and driver.claim())
            # The reused driver's max_games=24 receipt is its generic capacity;
            # the immutable registration and loop below allow only eight here.
            write(stage / 'manifest.local.json', binding)
            for plan in contract.plans(manifest.panel):
                gate = ready()
                require(check() and len(accepted) < limits['max_games']
                    and owner['deadline'] - time.monotonic() > limits['job_wall_seconds'] + limits['cleanup_seconds']
                    and dm._owned_size(stage) + limits['per_game_reserve_bytes'] <= limits['per_stage_output_bytes']
                    and sum(dm._owned_size(s) for s in stages.values() if s.exists())
                        + limits['per_game_reserve_bytes'] <= limits['total_output_bytes']
                    and shutil.disk_usage(stage).free >= limits['disk_floor_bytes'] + limits['per_game_reserve_bytes'])
                name = plan['game']
                write(stage / 'job-results' / (name + '.readiness.local.json'), gate)
                write(stage / 'job-results' / (name + '.permit.local.json'),
                      marker('permit', commit, commitment, current, name))
                command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                           '--registration-commit', commit, '--cohort', current, '--game', name]
                supervision = supervise_recorded(modules['campaign'], command, GAME,
                    dict(max_rss_bytes=limits['max_rss_bytes'], max_commit_bytes=limits['max_commit_bytes'],
                         min_available_bytes=limits['running_available_bytes'], wall_seconds=limits['job_wall_seconds'],
                         cleanup_seconds=limits['cleanup_seconds'], poll_seconds=limits['poll_seconds']))
                supervision_path = stage / 'job-results' / (name + '.supervision.local.json')
                supervision_sha = write(supervision_path,
                    dict(schema='task-teacher-fresh-panel-supervision-v1', cohort=current, game=name,
                         supervision=supervision, retry_allowed=False), MAX_SMALL)
                supervision_receipts.append(dict(cohort=current, game=name, sha256=supervision_sha,
                    launcher_reaped=supervision.get('launcher_reaped'),
                    job_members_zero=supervision.get('job_members_zero'),
                    cleanup_error_kind=supervision.get('cleanup_error_kind'),
                    stop_reason=supervision.get('stop_reason'),
                    available_memory_samples=supervision.get('available_memory_samples')))
                require(acceptance.supervision_ok(supervision) and check())
                bundle, hashes = teacher._load_bundle(dm, acceptance, stage, plan)
                job_raw = dm._read(stage, stage / 'job-inputs' / (name + '.json'), 16384)
                job = acceptance._json(job_raw, 16384)
                with contract.teacher_adapter(teacher):
                    row = teacher.verify_bundle(plan, job, bundle, True, crop_sha=manifest.panel.crop[1],
                        engine_sha=reg['engine_sha256'],
                        storage_limit=manifest.panel.worker_config()['storage']['game_write_bytes'],
                        native_scope=modules['worker'].NATIVE_OUTPUT_SCOPE, receipt_hashes=hashes,
                        verify_files=lambda p, r: verify_game_files(modules, manifest, stage, p, r, contract))
                    require(row['initial_banks'] == modules['sources'].initial_cash_for(context, current, plan))
                    def guarded_trace(*args):
                        for value in dm._trace(*args):
                            require(check())
                            yield value
                    trace = teacher.trace_summary(SimpleNamespace(_trace=guarded_trace), stage, plan, row,
                                                  manifest.panel.crop[1])
                for kind in ('permit', 'started', 'consumed'):
                    require(decode(read(stage / 'job-results' / (name + '.' + kind + '.local.json'))) ==
                        marker(kind, commit, commitment, current, name))
                receipt = dict(schema='task-teacher-fresh-panel-accepted-v1', cohort=current, game=name,
                    receipt_sha256=hashes, supervision_sha256=supervision_sha, job_input_sha256=sha(job_raw),
                    trace_summary=trace, passed=True, retry_allowed=False,
                    final_cash_arithmetic=True, transaction_cash_reconciliation=False)
                receipt_sha = write(stage / 'job-results' / (name + '.accepted.local.json'), receipt, MAX_SMALL)
                accepted.append(dict(cohort=current, row=row, acceptance_sha256=receipt_sha))
        require(len(accepted) == limits['max_games'] == limits['max_official_replays'] and check())
        again = modules['sources'].load(guard=check)
        require(again.identities == context.identities == binding['completed_sources']
            and sha(read(GAME / 'main.py')) == binding['default_sha256'])
        identity.verify_public(repo=REPO, game_root=GAME, registration_commit=commit)
        ledger, projected = contract.projection(accepted, context.prior_public, commit)
        write(PRIVATE / 'ledger.local.json', ledger)
        write(PRIVATE / 'projected-summary.local.json', projected)
        write(PRIVATE / 'result.local.json', dict(schema='task-teacher-fresh-panel-result-v1',
            registration_commit=commit, binding_sha256=commitment, games=16, official_replays=16,
            accepted=accepted, projection_sha256=sha(read(PRIVATE / 'projected-summary.local.json')),
            ledger_sha256=sha(read(PRIVATE / 'ledger.local.json'))))
        require(sum(dm._owned_size(s) for s in stages.values()) <= limits['total_output_bytes']
            and dm._owned_size(PRIVATE) <= limits['private_result_bytes'] and check())
        return dict(status='passed', games=16, official_replays=16, seed_clusters=4,
                    training=False, gpu=False, independent_evaluation=False, promotion=False)
    finally:
        if owned:
            error = sys.exc_info()[1]
            memory = closing_memory(error, latest_memory, available)
            write(PRIVATE / 'closed.local.json', dict(schema='task-teacher-fresh-panel-closed-v1',
                registration_commit=commit, binding_sha256=commitment,
                outcome='completed' if error is None else 'failed',
                error_kind=None if error is None else type(error).__name__, accepted_games=len(accepted),
                **memory, child_cleanup_receipts=supervision_receipts,
                elapsed_seconds=time.monotonic()-started, retry_allowed=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--registration-commit')
    parser.add_argument('--cohort')
    parser.add_argument('--game')
    args = parser.parse_args()
    try:
        require(not (args.prepare and (args.preflight or args.registration_commit or args.cohort or args.game)))
        result = prepare() if args.prepare else run(args.registration_commit,
            preflight=args.preflight, cohort=args.cohort, game=args.game)
        if result is not None:
            print(json.dumps(result))
        return 0
    except BaseException as exc:
        kind = ('memory_gate' if isinstance(exc, MemoryError) else 'time_gate' if isinstance(exc, TimeoutError)
                else 'io_error' if isinstance(exc, OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed', category=kind, retry_allowed=False)), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
