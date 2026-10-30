"""Single-use entry for an eight-game consumed-seed round2 comparison.

The previous market attempt is never reopened. Public registration, rules and
runtime are authenticated before private historical roles and assets are bound.
No raw seed, model identity, request or exception text is printed.
"""
import argparse
import copy
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import ModuleType


REPO = Path(__file__).resolve().parents[3]
GAME_SOURCE = REPO / 'kaggriculture'
PACKAGE = 'task_pair_round2_compare_v1'
PREFIX = 'kaggriculture/learning/' + PACKAGE + '/'
REGISTRATION = 'review/evidence/task-pair-round2-compare-v1-20261031/registration.json'
REGISTRATION_NOTE = 'review/evidence/task-pair-round2-compare-v1-20261031/REGISTRATION.zh-CN.md'
STAGE = 'task-pair-round2-compare-v1-20261031'
SCHEMA = 'task-pair-round2-compare-prereg-v1'
OLD_COMMIT = 'e458efa11b4320b1d8982abea34dba3a2bf377d2'
OLD_REGISTRATION = 'review/evidence/task-pair-historical-compare-v1-20261030/registration.json'
OLD_FAILURE_COMMIT = 'b04bc64b388ad050bf46ca25a47b0e4af37c0948'
OLD_FAILURE_NOTE = 'review/evidence/task-pair-historical-compare-v1-failure-20261030/FAILURE.zh-CN.md'
OLD_FAILURE_CERTIFICATE = 'review/evidence/task-pair-historical-compare-v1-failure-20261030/certificate.json'
NEW_FILES = frozenset(PREFIX + name + '.py' for name in ('__init__', 'entry', 'coordinator', 'projection'))
SAVED_PREFIX = 'kaggriculture/learning/task_pair_saved_compare_v1/'
SAVED_FILES = frozenset(SAVED_PREFIX + name + '.py' for name in ('__init__', 'analysis'))
EXTRA_REQUIRED = NEW_FILES | SAVED_FILES | frozenset((REGISTRATION_NOTE, OLD_REGISTRATION,
                                                       OLD_FAILURE_NOTE, OLD_FAILURE_CERTIFICATE))
PANEL = dict(actual_seed_roles=['S01', 'S04'], internal_seed_roles=['S02', 'S03'],
             opponents=['round2'], policies=['baseline', 'control'], sides=[0, 1],
             games=8, scope='consumed_development_only')
EVALUATION = dict(official_games=8, official_replays=8, comparisons=4,
                  new_seed_generation=False, training_updates=0, retries=False,
                  independent_evaluation=False, statistical_significance=False, promotion=False)
GAMES = tuple(f'dev-{seed}-round2-{side}-{arm}' for seed in ('S02', 'S03')
              for side in (0, 1) for arm in ('baseline', 'control'))
INITIAL = 5 * 1024**3
BETWEEN = 4 * 1024**3
JOB_MINIMUM = 3 * 1024**3
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_PHASES = frozenset(('snapshot', 'public_authentication', 'preflight', 'claim', 'binding',
                     'child_authorization', 'child_run', 'campaign', 'returned'))
_CATEGORIES = frozenset(('memory_gate', 'time_gate', 'missing_file', 'io_error',
                         'invalid_input', 'runtime_error', 'other_exception'))


def _require(condition):
    if not condition:
        raise ValueError('Round2 verification failed')


def _adapt_auth(auth):
    """Configure a fresh historical verifier instance for this registration."""
    auth.REGISTRATION, auth.STAGE_NAME, auth.SCHEMA = REGISTRATION, STAGE, SCHEMA
    auth.PANEL, auth.EVALUATION = copy.deepcopy(PANEL), copy.deepcopy(EVALUATION)
    auth.REQUIRED = auth.REQUIRED | EXTRA_REQUIRED
    auth.PRIOR = (*auth.PRIOR, (OLD_COMMIT, (OLD_REGISTRATION,)),
                  (OLD_FAILURE_COMMIT, (OLD_FAILURE_NOTE, OLD_FAILURE_CERTIFICATE)))
    return auth


def _validate_inheritance(reg, previous, typed_equal):
    """Compare to immutable original registration; no caller-defined limits."""
    _require(type(reg) is dict and type(previous) is dict and callable(typed_equal))
    old_sources, sources = previous.get('public_sources'), reg.get('public_sources')
    _require(type(old_sources) is dict and type(sources) is dict
             and EXTRA_REQUIRED.issubset(sources)
             and all(sources.get(key) == digest for key, digest in old_sources.items()
                     if key != 'review/PROGRESS.md'))
    for key in ('engine_commit', 'engine_sha256', 'previous_training_result_commit',
                'historical_panel_result_commit'):
        _require(type(previous.get(key)) is str and reg.get(key) == previous[key])
    limits = copy.deepcopy(previous.get('limits'))
    _require(type(limits) is dict and type(limits.get('job')) is dict)
    limits['initial_available_bytes'] = INITIAL
    limits['between_available_bytes'] = BETWEEN
    limits['job']['min_available_bytes'] = JOB_MINIMUM
    _require(typed_equal(reg.get('limits'), limits) is True)


def _historical():
    """Construct old code only from the verified committed-byte launcher."""
    raw = globals().get('_AUTH_HISTORY_ENTRY_BYTES')
    _require(type(raw) is bytes and 0 < len(raw) <= 4 * 1024**2)
    path = GAME_SOURCE / 'learning/task_pair_historical_compare_v1/entry.py'
    old = ModuleType('_round2_historical_entry')
    old.__file__ = str(path)
    for key in ('_AUTH_IDENTITY_BYTES', '_AUTH_BOOTSTRAP_BYTES', '_AUTH_ONCE_AUTH_BYTES',
                '_AUTH_ONCE_ENTRY_BYTES', '_AUTH_COMPARISON_BYTES'):
        value = globals().get(key)
        _require(type(value) is bytes and 0 < len(value) <= 4 * 1024**2)
        old.__dict__[key] = value
    exec(compile(raw, str(path), 'exec', dont_inherit=True), old.__dict__)
    old.GAMES = GAMES
    snapshots = old._snapshots

    def configured_snapshots():
        identity, bootstrap, once_auth, once_entry, auth = snapshots()
        return identity, bootstrap, once_auth, once_entry, _adapt_auth(auth)

    old._snapshots = configured_snapshots
    return old


def _check_one_package(root, reg, package, names):
    """Reject local shadows and verify the exact registered module bytes."""
    private, public = root / 'learning', GAME_SOURCE / 'learning'
    suffixes = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    _require(not os.path.lexists(private / package))
    for base in (private, public):
        _require(not any(os.path.lexists(base / (package + suffix)) for suffix in suffixes))
    folder = public / package
    _require(folder.is_dir() and folder.resolve(strict=True) == folder
             and not os.path.lexists(folder / '__pycache__'))
    identity = reg['public_sources']
    import hashlib
    for name in names:
        path = folder / (name + '.py')
        _require(path.resolve(strict=True) == path and path.is_file())
        with path.open('rb') as stream:
            data = stream.read(4 * 1024**2 + 1)
        _require(0 < len(data) <= 4 * 1024**2
                 and hashlib.sha256(data).hexdigest() == identity[
                     'kaggriculture/learning/' + package + '/' + name + '.py']
                 and not os.path.lexists(folder / name)
                 and not any(os.path.lexists(folder / (name + suffix))
                             for suffix in suffixes if suffix != '.py'))
    prefix = 'learning.' + package
    _require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules))


def _check_package(root, reg):
    _check_one_package(root, reg, PACKAGE, ('__init__', 'entry', 'coordinator', 'projection'))
    _check_one_package(root, reg, 'task_pair_saved_compare_v1', ('__init__', 'analysis'))


def _load(name):
    _require(name in ('coordinator', 'projection'))
    module = importlib.import_module('learning.' + PACKAGE + '.' + name)
    expected = GAME_SOURCE / 'learning' / PACKAGE / (name + '.py')
    _require(Path(module.__file__).resolve(strict=True) == expected)
    return module


def _context(game_root, commit):
    old = _historical()
    ctx = old._context(game_root, commit)
    identity, reg = ctx['identity'], ctx['reg']
    original_raw = identity._git(REPO, 'show', OLD_COMMIT + ':' + OLD_REGISTRATION)
    original = identity._json(original_raw)
    _validate_inheritance(reg, original, ctx['once_auth']._typed_equal)
    _check_package(ctx['root'], reg)
    coordinator, projection = _load('coordinator'), _load('projection')
    _require(callable(getattr(coordinator, 'Coordinator', None))
             and callable(getattr(coordinator, 'limits_snapshot', None))
             and callable(getattr(projection, 'save_result', None))
             and ctx['once_auth']._typed_equal(ctx['limits'], coordinator.limits_snapshot(reg['limits'])))
    ctx['coordinator'], ctx['new_projection'] = coordinator, projection
    return old, ctx


def _sample(ctx):
    import psutil
    value = psutil.virtual_memory().available
    _require(type(value) is int and value >= 0)
    ctx['last_available_bytes'] = value
    return value


def _category(exc):
    if isinstance(exc, MemoryError):
        return 'memory_gate'
    if isinstance(exc, TimeoutError):
        return 'time_gate'
    if isinstance(exc, FileNotFoundError):
        return 'missing_file'
    if isinstance(exc, OSError):
        return 'io_error'
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError)):
        return 'invalid_input'
    if isinstance(exc, RuntimeError):
        return 'runtime_error'
    return 'other_exception'


def _event(ctx, value):
    """Copy only the coordinator's explicitly whitelisted failure fields."""
    module = ctx['coordinator']
    if (type(value) is dict and set(value) == {'phase', 'category', 'attempted', 'accepted'}
            and value['phase'] in module.FAILURE_PHASES and value['category'] in module.FAILURE_CATEGORIES
            and type(value['attempted']) is int and type(value['accepted']) is int
            and 0 <= value['accepted'] <= value['attempted'] <= 8):
        ctx['coordinator_failure'] = dict(value)


def _failure_record(ctx, phase, exc):
    _require(phase in _PHASES)
    result = dict(schema='task-pair-round2-failure-v1', status='failed', phase=phase,
                  category=_category(exc), retry_allowed=False,
                  last_available_bytes=ctx.get('last_available_bytes'))
    event = ctx.get('coordinator_failure')
    if event is not None:
        result['coordinator'] = event
    return result


def _write_failure(ctx, record):
    """Best effort; never open or overwrite an existing old or new record."""
    path = ctx['stage'].with_name(ctx['stage'].name + '.diagnostic.local.json')
    data = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode()
    _require(path.parent == ctx['stage'].parent and len(data) <= 16384
             and shutil.disk_usage(path.parent).free >= ctx['limits']['disk_free_floor_bytes'] + len(data))
    with path.open('xb') as stream:
        _require(stream.write(data) == len(data))
        stream.flush()
        os.fsync(stream.fileno())
    _require(path.read_bytes() == data)


def command(root, commit, *, game=None, preflight=False):
    """Build an isolated launcher argv without running it or printing paths."""
    _require(type(commit) is str and _COMMIT.fullmatch(commit)
             and type(preflight) is bool and (game is None or game in GAMES)
             and not (preflight and game is not None))
    argv = [sys.executable, '-I', '-B', '-c', SNAPSHOT_STUB, str(REPO), commit,
            '--game-root', str(root), '--registration-commit', commit]
    if preflight:
        argv.append('--preflight')
    if game is not None:
        argv.extend(('--game', game))
    return argv


def _command(root, commit, game):
    _require(game in GAMES)
    return command(root, commit, game=game)


def _unclaimed(ctx):
    stage = ctx['stage']
    return (not os.path.lexists(ctx['attempt_path']) and not os.path.lexists(ctx['failure_path'])
            and not os.path.lexists(stage)
            and not os.path.lexists(stage.with_name(stage.name + '.diagnostic.local.json')))


def _preflight_binding(ctx):
    """Rebuild the old binding read-only, without impersonating an attempt."""
    bootstrap, identity, once = ctx['bootstrap'], ctx['identity'], ctx['once_entry']
    root, reg, old_reg, freeze = ctx['root'], ctx['reg'], ctx['old_reg'], ctx['freeze']
    _require(ctx['public_check']() is True and ctx['time_check']() is True and _unclaimed(ctx))
    status_path, current, old_dev, prior = bootstrap._private_status()
    _require(type(current) is dict and ctx['once_auth']._typed_equal(
        old_dev, current.get('current_pair_dev_diagnostic')) and old_dev.get('state') == 'failed_no_retry')
    status_raw = bootstrap._read(status_path)
    binding_path = REPO.parent / 'private' / bootstrap.PREBIND
    binding_raw = bootstrap._read(binding_path, 65536)
    once._verify_private_snapshot(bootstrap, ctx['once_auth'], current, prior, status_raw, binding_raw,
                                  old_dev.get('prebinding_record_sha256'))
    seed_path = root / bootstrap.OLD_PANEL / 'private-seeds.local.json'
    seed_raw = bootstrap._read(seed_path, 16384)
    fingerprint = once._fingerprint(reg, old_reg, freeze, prior)

    def check():
        if _sample(ctx) < INITIAL:
            raise MemoryError('Resource preflight failed')
        return (ctx['time_check']() is True and _unclaimed(ctx) and ctx['public_check']() is True
                and bootstrap._read(status_path) == status_raw
                and bootstrap._read(binding_path, 65536) == binding_raw
                and bootstrap._read(seed_path, 16384) == seed_raw
                and once._fingerprint(reg, old_reg, freeze, prior) == fingerprint)

    manifest, previous, roles = ctx['bindings'].build(
        modules=ctx['modules'], bootstrap=bootstrap, identity=identity, reg=reg, old_reg=old_reg,
        freeze=freeze, game_root=root, repo=REPO, status_path=status_path, prior_binding=prior, check=check)
    _require(check() is True and ctx['bindings'].verify_mapping(manifest, previous, roles) is True
             and manifest.opponent('round2') == previous.opponent('round2'))
    indexed = {plan['game']: plan for plan in manifest.panel.plans()}
    _require(len(indexed) == len(manifest.panel.plans()) and all(game in indexed for game in GAMES))
    for game in GAMES:
        plan = manifest.panel.validate_plan(indexed[game])
        _require(plan['game'] == game and plan['opponent'] == 'round2')
    import hashlib
    source = identity._read(root / 'learning/task_anchor_once_v1/games.py')
    _require(hashlib.sha256(source).hexdigest() == manifest.panel.worker_sha)
    bootstrap._verify_assets(manifest, root, ctx['modules']['history'])
    _require(check() is True and ctx['bindings'].verify_mapping(manifest, previous, roles) is True)


def public_preflight(*, game_root, registration_commit):
    """Recheck identity and resources without claiming or executing a game."""
    ctx, phase = {}, 'public_authentication'
    try:
        started = globals().get('_AUTH_ENTRY_STARTED')
        if started is None:
            started = time.monotonic()
        _, ctx = _context(game_root, registration_commit)
        once = ctx['once_entry']
        sampled = once._starting_clock(started, time.monotonic)
        sampled()

        def time_check():
            once._check_entry_time(started, ctx['limits']['campaign_wall_seconds'], sampled)
            return True

        ctx['time_check'] = time_check
        phase = 'preflight'
        available = _sample(ctx)
        if available < INITIAL:
            raise MemoryError('Resource preflight failed')
        _require(ctx['public_check']() is True and _unclaimed(ctx)
                 and shutil.disk_usage(ctx['stage'].parent).free >=
                 ctx['limits']['disk_free_floor_bytes'] + ctx['limits']['per_game_reserve_bytes'])
        phase = 'binding'
        _preflight_binding(ctx)
        return dict(status='ready', scope='read_only_identity_preflight', games_started=0,
                    last_available_bytes=ctx['last_available_bytes'])
    except BaseException as exc:
        print(json.dumps(_failure_record(ctx, phase, exc), sort_keys=True), file=sys.stderr)
        raise


def execute(*, game_root, registration_commit, game=None, clock=time.monotonic):
    """Authenticate and execute one new stage or one exclusively permitted child."""
    ctx, old, phase = {}, None, 'public_authentication'
    started = globals().get('_AUTH_ENTRY_STARTED')
    if started is None:
        started = clock()
    try:
        _require(game is None or game in GAMES)
        old, ctx = _context(game_root, registration_commit)
        once = ctx['once_entry']
        sampled = once._starting_clock(started, clock)
        sampled()

        def time_check():
            once._check_entry_time(started, ctx['limits']['campaign_wall_seconds'], sampled)
            return True

        ctx['time_check'] = time_check
        phase = 'preflight'
        available = _sample(ctx)
        if available < (INITIAL if game is None else JOB_MINIMUM):
            raise MemoryError('Resource preflight failed')
        _require(time_check() is True)
        if game is not None:
            phase = 'child_authorization'
            old._start_child(ctx, game)
            phase = 'binding'
            old._bind(ctx)
            phase = 'child_run'
            old._run_child(ctx, game)
            return None

        _require(shutil.disk_usage(ctx['stage'].parent).free >=
                 ctx['limits']['disk_free_floor_bytes'] + ctx['limits']['per_game_reserve_bytes'])
        _require(_unclaimed(ctx))
        ctx['_attempt_created'], ctx['_attempt_open_failed'] = False, False
        phase = 'claim'
        old._claim(ctx)
        phase = 'binding'
        old._bind(ctx)
        prepared = [False]

        def check():
            if _sample(ctx) < JOB_MINIMUM:
                raise MemoryError('Resource floor reached')
            return old._check(ctx, prepared[0])

        ctx['check'] = check
        ctx['marker'] = lambda name, kind: old._marker(ctx, name, kind)
        driver = ctx['modules']['driver'].LocalDriver(manifest=ctx['manifest'], game_root=ctx['root'],
            stage=ctx['stage'], commit=registration_commit,
            command_for=lambda name: _command(ctx['root'], registration_commit, name),
            check=check, available=lambda: _sample(ctx))
        original_claim = driver.claim

        def claim():
            _require(original_claim() is True)
            once._write_stage(ctx, old.MANIFEST, ctx['manifest_raw'])
            prepared[0] = True
            _require(check() is True)
            return True

        driver.claim = claim
        phase = 'campaign'
        result = ctx['coordinator'].Coordinator().run(
            manifest=ctx['manifest'], plans=ctx['plans'], limits=ctx['limits'], driver=driver,
            check=check, authorize_game=lambda plan: old._authorize(ctx, plan),
            save_result=lambda payload: ctx['new_projection'].save_result(ctx, driver, payload),
            on_failure=lambda event: _event(ctx, event), clock=once._coordinator_clock(started, sampled))
        _require(type(result) is dict and result.get('passed') is True and result.get('games') == 8
                 and result.get('official_replay_passes') == 8 and result.get('comparisons') == 4
                 and result.get('promotion') is False and check() is True and time_check() is True)
        phase = 'returned'
        old._write_stage(ctx, 'comparison-returned.local.json',
                         dict(schema='task-pair-round2-returned-v1', runner_returned=True,
                              accepted=8, official_replay_passes=8, observed_outer_exit=False))
        _require(check() is True and time_check() is True)
        return dict(status='passed', games=8, official_replay_passes=8, comparisons=4,
                    observed_outer_exit=False, training=False, promotion=False)
    except BaseException as exc:
        record = _failure_record(ctx, phase, exc)
        if old is not None and (ctx.get('_attempt_created') or ctx.get('_attempt_open_failed')):
            try:
                old._failure(ctx)
            except Exception:
                pass
            try:
                _write_failure(ctx, record)
            except Exception:
                pass
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        raise


SNAPSHOT_STUB = '''import pathlib, re, subprocess, sys, time
started=time.monotonic()
def start():
 repo=pathlib.Path(sys.argv.pop(1)); commit=sys.argv.pop(1)
 if re.fullmatch(r"[0-9a-f]{40}",commit) is None: raise ValueError()
 def snapshot(rel):
  path=repo.joinpath(*rel.split("/"))
  with path.open("rb") as stream: data=stream.read(4194305)
  if not data or len(data)>4194304: raise ValueError()
  saved=subprocess.run(["git","-C",str(repo),"show",commit+":"+rel],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30)
  if data!=saved.stdout: raise ValueError()
  return path,data
 path,data=snapshot("kaggriculture/learning/task_pair_round2_compare_v1/entry.py")
 _,history=snapshot("kaggriculture/learning/task_pair_historical_compare_v1/entry.py")
 _,identity=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/identity.py")
 _,bootstrap=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py")
 _,once_auth=snapshot("kaggriculture/learning/task_control_once_v1/auth.py")
 _,once_entry=snapshot("kaggriculture/learning/task_control_once_v1/entry.py")
 _,auth=snapshot("kaggriculture/learning/task_pair_historical_compare_v1/auth.py")
 exec(compile(data,str(path),"exec",dont_inherit=True),{"__name__":"__main__","__file__":str(path),"_AUTH_HISTORY_ENTRY_BYTES":history,"_AUTH_IDENTITY_BYTES":identity,"_AUTH_BOOTSTRAP_BYTES":bootstrap,"_AUTH_ONCE_AUTH_BYTES":once_auth,"_AUTH_ONCE_ENTRY_BYTES":once_entry,"_AUTH_COMPARISON_BYTES":auth,"_AUTH_ENTRY_STARTED":started})
try: start()
except Exception: raise SystemExit("Round2 snapshot stopped.") from None
'''


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-root', required=True)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--game')
    parser.add_argument('--preflight', action='store_true')
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        if args.preflight:
            _require(args.game is None)
            result = public_preflight(game_root=args.game_root, registration_commit=args.registration_commit)
        else:
            result = execute(game_root=args.game_root, registration_commit=args.registration_commit, game=args.game)
        if result is not None:
            print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
