"""One registered eight-game market development panel, using existing Worker.

Private seed/model identities, observations, logs and replays stay local.
The coordinator runs each official game and its existing official replay once;
this entry does not claim an independent holdout or promotion.
"""
import argparse
import ctypes
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import ModuleType


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
PRIVATE = REPO.parent / 'private' / 'task-pair-market-consumed-v1-20261031'
STAGE = 'task-pair-historical-compare-v1-consumed-market-20261031'
REGISTRATION = 'review/evidence/task-pair-market-consumed-v1-20261031/registration.json'
NOTE = 'review/evidence/task-pair-market-consumed-v1-20261031/REGISTRATION.zh-CN.md'
PREFIX = 'kaggriculture/learning/task_pair_market_consumed_v1/'
BASE = 'c70bdf406a9eb4d9a3de66d37fee65bc102a9e63'
OLD_LIMIT_COMMIT = '7ab3f32532b29443ea2b97fd9e7b2d04dd651a43'
OLD_LIMITS = 'review/evidence/task-pair-round2-compare-v1-20261031/registration.json'
SCHEMA = 'task-pair-market-consumed-prereg-v1'
PANEL = dict(actual_seed_roles=['D01', 'D02'], internal_seed_roles=['S02', 'S03'],
             opponents=['market'], policies=['baseline', 'control'], sides=[0, 1],
             games=8, scope='consumed_development_only')
EVALUATION = dict(official_games=8, official_replays=8, comparisons=4,
                  new_seed_generation=False, training_updates=0, retries=False,
                  independent_evaluation=False, statistical_significance=False, promotion=False)
GAMES = tuple(f'dev-{alias}-market-{side}-{arm}' for alias in ('S02', 'S03')
              for side in (0, 1) for arm in ('baseline', 'control'))
HEX = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
MAX_METADATA = 8 * 1024**2
INITIAL_MEMORY = 5 * 1024**3
CHILD_MEMORY = 3 * 1024**3


def require(value):
    if not value:
        raise ValueError('Registered market comparison check failed')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def physical_available():
    """Read Windows available physical memory before importing game packages."""
    require(os.name == 'nt')
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                    ('avail_extended_virtual', ctypes.c_ulonglong)]
    value = Status()
    value.length = ctypes.sizeof(value)
    api = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
    api.argtypes, api.restype = [ctypes.POINTER(Status)], ctypes.c_int
    if not api(ctypes.byref(value)):
        raise OSError('Windows memory sample failed')
    return int(value.avail_phys)


def enforce_start_memory(*, child, reader=None):
    require(type(child) is bool)
    available = (reader or physical_available)()
    require(type(available) is int and available >= 0)
    if available < (CHILD_MEMORY if child else INITIAL_MEMORY):
        raise MemoryError('Initial memory floor')
    return True


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result)
        result[key] = value
    return result


def _bad(_):
    raise ValueError('Nonfinite JSON')


def decode(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_METADATA)
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_pairs, parse_constant=_bad)
    require(type(value) is dict)
    return value


def read(path, limit=MAX_METADATA):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
            and not (getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)
            and path.stat().st_size <= limit)
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit)
    return raw


def git(*args):
    with open(os.devnull, 'rb') as null:
        process = subprocess.run(['git', '-C', str(REPO), *args], stdin=null,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(process.returncode == 0 and len(process.stdout) <= MAX_METADATA
            and len(process.stderr) <= MAX_METADATA)
    return process.stdout


def _snapshot(relative, commit, sources):
    path = REPO.joinpath(*relative.split('/'))
    raw = read(path)
    require(relative in sources and sha(raw) == sources[relative]
            and raw == git('show', commit + ':' + relative))
    module = ModuleType('_market_' + path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def no_extra_shadow():
    """Prevent game-root modules from replacing the three public packages."""
    private = GAME / 'learning'
    public = REPO / 'kaggriculture/learning'
    suffixes = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    for name in ('task_pair_historical_compare_v1', 'task_control_once_v1', 'task_pair_market_consumed_v1'):
        require(not os.path.lexists(private / name))
        for base in (private, public):
            require(not any(os.path.lexists(base / (name + suffix)) for suffix in suffixes))
        folder = public / name
        require(folder.is_dir() and folder.resolve(strict=True) == folder
                and not os.path.lexists(folder / '__pycache__'))
        for source in folder.glob('*.py'):
            require(source.is_file() and source.resolve(strict=True) == source)
            stem = source.stem
            other = folder / stem
            require(not any(os.path.lexists(folder / (stem + suffix)) for suffix in suffixes[1:])
                    and not (other.is_dir() and any(os.path.lexists(other / ('__init__' + suffix))
                                                   for suffix in suffixes)))


def public_context(commit):
    """Authenticate the registration, rule engine, code and runtime first."""
    require(os.name == 'nt' and sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__
            and type(commit) is str and COMMIT.fullmatch(commit))
    require(git('rev-parse', 'HEAD').decode().strip() == commit
            and not git('status', '--porcelain', '--untracked-files=all'))
    git('merge-base', '--is-ancestor', BASE, commit)
    raw = read(REPO / REGISTRATION)
    require(raw == git('show', commit + ':' + REGISTRATION))
    reg = decode(raw)
    require(reg.get('base_commit') == BASE and type(reg.get('public_sources')) is dict)
    sources = reg['public_sources']
    identity = _snapshot('kaggriculture/learning/task_pair_dev_campaign_v1/identity.py', commit, sources)
    once_auth = _snapshot('kaggriculture/learning/task_control_once_v1/auth.py', commit, sources)
    auth = _snapshot('kaggriculture/learning/task_pair_historical_compare_v1/auth.py', commit, sources)
    bootstrap = _snapshot('kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py', commit, sources)
    auth.REGISTRATION, auth.STAGE_NAME, auth.SCHEMA = REGISTRATION, STAGE, SCHEMA
    auth.PANEL, auth.EVALUATION = PANEL, EVALUATION
    auth.REQUIRED = auth.REQUIRED | frozenset((NOTE, PREFIX + '__init__.py', PREFIX + 'prepare.py',
                                                PREFIX + 'entry.py', PREFIX + 'projection.py'))
    verified, _, _ = auth.verify_public(identity, once_auth, repo=REPO, game_root=GAME,
                                         registration_commit=commit)
    require(verified == reg)
    previous = decode(git('show', OLD_LIMIT_COMMIT + ':' + OLD_LIMITS))
    require(reg.get('limits') == previous.get('limits'))
    no_extra_shadow()
    modules = bootstrap._runtime(GAME, reg)
    import importlib
    prep = importlib.import_module('learning.task_pair_market_consumed_v1.prepare')
    projection = importlib.import_module('learning.task_pair_market_consumed_v1.projection')
    runner = importlib.import_module('learning.task_pair_historical_compare_v1.runner')
    prefix = importlib.import_module('learning.task_pair_historical_compare_v1.request_prefix')
    for module in (prep, projection, runner, prefix):
        require(Path(module.__file__).resolve(strict=True).is_relative_to(REPO))

    def verify_again():
        current, _, _ = auth.verify_public(identity, once_auth, repo=REPO, game_root=GAME,
                                             registration_commit=commit)
        return typed_equal(current, reg)

    return reg, modules, prep, projection, runner, prefix, verify_again


def binding_context(prep, expected_sha):
    require(type(expected_sha) is str and HEX.fullmatch(expected_sha))
    raw = read(PRIVATE / 'binding.local.json', 65536)
    require(sha(raw) == expected_sha)
    document = decode(raw)
    require(document.get('schema') == 'task-pair-market-consumed-binding-v1'
            and document.get('base_commit') == BASE
            and prep.collect(document['old_binding_sha256'], document['salt']) == document)
    return raw, document


def plans_for(manifest):
    panel = manifest.panel
    indexed = {(p['seed_alias'], p['opponent'], p['side'], p['policy']): p for p in panel.plans()}
    result = [indexed[(alias, 'market', side, arm)] for alias in ('S02', 'S03')
              for side in (0, 1) for arm in ('baseline', 'control')]
    require([p['game'] for p in result] == list(GAMES))
    return result


def validate_assets(manifest, driver_module):
    """Hash all frozen task/crop models, both opponents and the Worker."""
    assets = [(path, digest, driver_module.MAX_MODEL) for path, digest in (*manifest.panel.models, manifest.panel.crop)]
    assets += [(path, digest, driver_module.MAX_SOURCE) for path, digest in manifest.opponent_sources]
    assets.append(('learning/task_anchor_once_v1/games.py', manifest.panel.worker_sha, 4 * 1024**2))
    for relative, expected, limit in assets:
        require(driver_module._hash(GAME, GAME / relative, limit) == expected)
    return True


def preflight(commit, binding_sha):
    """Read-only runtime, source and private-binding check; starts no games."""
    enforce_start_memory(child=False)
    reg, modules, prep, _projection, _runner, _prefix, _verify = public_context(commit)
    import psutil
    if psutil.virtual_memory().available < reg['limits']['initial_available_bytes']:
        raise MemoryError('Initial memory floor')
    stage = GAME / 'artifacts' / STAGE
    modules['driver']._no_links(GAME, stage)
    require(not stage.exists() and not stage.is_symlink()
            and not (PRIVATE / 'attempt.local.json').exists()
            and shutil.disk_usage(GAME / 'artifacts').free >=
                reg['limits']['disk_free_floor_bytes'] + reg['limits']['per_game_reserve_bytes'])
    _raw, document = binding_context(prep, binding_sha)
    manifest = modules['manifest'].build_manifest(seeds=document['seeds'], models=document['models'],
        crop=document['crop'], opponents=document['opponents'], worker_sha256=document['worker_sha256'])
    require(len(plans_for(manifest)) == 8 and validate_assets(manifest, modules['driver']))
    return dict(status='ready', scope='read_only_preflight', games_started=0)


def _write(path, raw):
    require(type(raw) is bytes and 0 < len(raw) <= 65536)
    with Path(path).open('xb') as stream:
        require(stream.write(raw) == len(raw))
        stream.flush()
        os.fsync(stream.fileno())
    require(read(path, 65536) == raw)


def _json(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def marker(kind, *, commit, binding_sha, game=None):
    require(kind in ('attempt', 'permit', 'started', 'consumed') and COMMIT.fullmatch(commit)
            and HEX.fullmatch(binding_sha) and (game is None if kind == 'attempt' else game in GAMES))
    value = dict(schema='task-pair-market-consumed-marker-v1', kind=kind,
                 registration_commit=commit, binding_sha256=binding_sha, retry_allowed=False)
    if game is not None:
        value['game'] = game
    return _json(value)


def typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(typed_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def clocks(started, clock):
    from learning.task_control_once_v1.entry import _coordinator_clock, _starting_clock
    sampled = _starting_clock(started, clock)
    require(sampled() == started)
    return sampled, _coordinator_clock(started, sampled)


def verify_comparison_receipt(receipt, comparison):
    require(type(receipt) is dict and type(comparison) is dict
            and typed_equal(receipt.get('comparison'), comparison))
    return True


def compact_result(*, payload, commit, binding_sha, ledger, comparison_ledger, prefixes):
    """Reference existing per-game receipts without duplicating large events."""
    require(type(payload) is dict and type(payload.get('paired')) is dict
            and type(ledger) is list and len(ledger) == 8
            and type(comparison_ledger) is list and len(comparison_ledger) == 4
            and type(prefixes) is dict and len(prefixes) == 4)
    return dict(schema='task-pair-market-consumed-private-result-v1', registration_commit=commit,
                binding_sha256=binding_sha, games=8, official_replay_passes=8, comparisons=4,
                game_receipts=ledger, comparison_receipts=comparison_ledger,
                request_prefixes=[dict(seed_alias=alias, side=side, **value)
                                  for (alias, side), value in sorted(prefixes.items())],
                paired=payload['paired'], training=False, independent_improvement=False, promotion=False)


def save_result(*, payload, manifest, driver, driver_module, limits, binding_sha,
                projection, prefix, check, remaining):
    require(type(payload) is dict and payload.get('attempted') == 8
            and payload.get('official_replay_passes') == 8 and payload.get('training') is False
            and payload.get('promotion') is False and payload.get('independent_improvement') is False)
    rows, comparisons = payload['rows'], payload['comparisons']
    require(type(rows) is list and len(rows) == 8 and type(comparisons) is list and len(comparisons) == 4)
    by_game = {row['game']: row for row in rows}
    require(set(by_game) == set(GAMES) and len(by_game) == 8)
    by_pair = {(item['seed_alias'], item['side']): item for item in comparisons}
    require(set(by_pair) == {(alias, side) for alias in ('S02', 'S03') for side in (0, 1)}
            and len(by_pair) == 4)
    acceptance = __import__('learning.task_pair_dev_campaign_v1.acceptance', fromlist=['acceptance'])
    worker = __import__('learning.task_pair_dev_games_v1.worker', fromlist=['worker'])
    ledger = []
    for plan in plans_for(manifest):
        name = plan['game']
        require(check() is True and driver.preflight_comparison(limits, remaining()) is True)
        bundle = driver.read_bundle(plan)
        supervision_path = driver.stage / 'job-results' / (name + '.supervision.local.json')
        acceptance_path = driver.stage / 'job-results' / (name + '.acceptance.local.json')
        supervision_raw = driver_module._read(driver.stage, supervision_path, acceptance.MAX_RECEIPT_BYTES)
        acceptance_raw = driver_module._read(driver.stage, acceptance_path, acceptance.MAX_RECEIPT_BYTES)
        supervision = acceptance._json(supervision_raw, acceptance.MAX_RECEIPT_BYTES)
        accepted = acceptance._json(acceptance_raw, acceptance.MAX_RECEIPT_BYTES)
        require(supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
                and supervision.get('game') == name and supervision.get('retry_allowed') is False
                and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
                and accepted.get('game') == name and accepted.get('passed') is True
                and accepted.get('retry_allowed') is False
                and accepted.get('supervision_sha256') == sha(supervision_raw)
                and accepted.get('job_input_sha256') == driver_module._hash(driver.stage,
                    driver.stage / 'job-inputs' / (name + '.json'), worker.MAX_JOB_BYTES)
                and accepted.get('receipt_sha256') == {key: sha(value) for key, value in bundle.items()})
        row = acceptance.accept(manifest.panel, plan, bundle, supervision.get('supervision'), driver.verify_files)
        require(typed_equal(row, by_game[name]))
        markers = {}
        for kind in ('permit', 'started', 'consumed'):
            raw = driver_module._read(driver.stage,
                driver.stage / 'job-results' / (name + '.' + kind + '.local.json'), 65536)
            require(raw == marker(kind, commit=driver.commit, binding_sha=binding_sha, game=name))
            markers[kind] = sha(raw)
        ledger.append(dict(game=name, supervision_sha256=sha(supervision_raw),
                           acceptance_sha256=sha(acceptance_raw), marker_sha256=markers))
    receipts, prefixes, comparison_ledger = [], {}, []
    for alias in ('S02', 'S03'):
        for side in (0, 1):
            require(check() is True and driver.preflight_comparison(limits, remaining()) is True)
            left_name = f'dev-{alias}-market-{side}-baseline'
            right_name = f'dev-{alias}-market-{side}-control'
            left, right = by_game[left_name], by_game[right_name]
            path = driver_module._comparison_file(driver.stage, left_name, 'control')
            raw = driver_module._read(driver.stage, path, acceptance.MAX_RECEIPT_BYTES)
            receipt = acceptance._json(raw, acceptance.MAX_RECEIPT_BYTES)
            require(verify_comparison_receipt(receipt, by_pair[(alias, side)]))
            receipts.append(receipt)
            comparison_ledger.append(dict(reference_game=left_name, sha256=sha(raw)))
            a = driver_module._replay(driver.stage,
                driver.stage / 'games' / left_name / 'replay.json.gz', left['replay_sha256'])
            b = driver_module._replay(driver.stage,
                driver.stage / 'games' / right_name / 'replay.json.gz', right['replay_sha256'])
            prefixes[(alias, side)] = prefix.compare(a, b, side=side)
            del a, b
            require(check() is True and remaining() > 0)
    seeds = dict(zip(('S02', 'S03'), manifest.panel.seeds))
    projected = projection.project(payload['paired'], seeds=seeds, rows=rows,
                                   comparisons=receipts, prefixes=prefixes)
    projected['registration_commit'] = driver.commit
    private = compact_result(payload=payload, commit=driver.commit, binding_sha=binding_sha,
                             ledger=ledger, comparison_ledger=comparison_ledger, prefixes=prefixes)
    driver_module._write_new(driver.stage, driver.stage / 'comparison-result.local.json', private)
    driver_module._write_new(driver.stage, driver.stage / 'projected-summary.local.json', projected)
    return True


def run(commit, binding_sha, game=None):
    started = time.monotonic()
    enforce_start_memory(child=game is not None)
    reg, modules, prep, projection, runner, prefix, verify_again = public_context(commit)
    require(game is None or game in GAMES)
    import psutil
    sampled, coordinator_clock = clocks(started, time.monotonic)

    def remaining():
        return reg['limits']['campaign_wall_seconds'] - (sampled() - started)

    if remaining() <= 0:
        raise TimeoutError('Campaign deadline')
    threshold = reg['limits']['initial_available_bytes'] if game is None else reg['limits']['job']['min_available_bytes']
    if psutil.virtual_memory().available < threshold:
        raise MemoryError('Initial memory floor')
    stage = GAME / 'artifacts' / STAGE
    modules['driver']._no_links(GAME, stage)
    default_before = modules['driver']._hash(GAME, GAME / 'main.py', modules['driver'].MAX_SOURCE)
    attempt = PRIVATE / 'attempt.local.json'
    attempt_raw = marker('attempt', commit=commit, binding_sha=binding_sha)
    if game is None:
        require(not stage.exists() and not stage.is_symlink())
        _write(attempt, attempt_raw)
    else:
        require(read(attempt, 65536) == attempt_raw and stage.is_dir())
        permit = stage / 'job-results' / (game + '.permit.local.json')
        require(read(permit, 65536) == marker('permit', commit=commit, binding_sha=binding_sha, game=game))
        _write(stage / 'job-results' / (game + '.started.local.json'),
               marker('started', commit=commit, binding_sha=binding_sha, game=game))
    raw, document = binding_context(prep, binding_sha)
    manifest = modules['manifest'].build_manifest(seeds=document['seeds'], models=document['models'],
        crop=document['crop'], opponents=document['opponents'], worker_sha256=document['worker_sha256'])
    require(validate_assets(manifest, modules['driver']))
    plans = plans_for(manifest)
    if game is not None:
        _write(stage / 'job-results' / (game + '.consumed.local.json'),
               marker('consumed', commit=commit, binding_sha=binding_sha, game=game))
    source = modules['driver']._read(GAME, GAME / 'learning/task_anchor_once_v1/games.py', 4 * 1024**2)
    require(sha(source) == manifest.panel.worker_sha)
    prepared = [game is not None]

    def check():
        if psutil.virtual_memory().available < reg['limits']['job']['min_available_bytes']:
            raise MemoryError('Runtime memory floor')
        if remaining() <= 0:
            raise TimeoutError('Campaign deadline')
        require(read(PRIVATE / 'binding.local.json', 65536) == raw
                and read(attempt, 65536) == attempt_raw)
        if prepared[0]:
            require(read(stage / 'manifest.local.json', 65536) == raw)
        return True

    if game is not None:
        plan = next((p for p in plans if p['game'] == game), None)
        require(plan is not None and check())
        modules['child'].run_one(manifest=manifest, worker_source=source, game_root=GAME,
                                 stage=stage, plan=plan, check=check)
        require(check() and validate_assets(manifest, modules['driver']) and verify_again()
                and modules['driver']._hash(GAME, GAME / 'main.py', modules['driver'].MAX_SOURCE) == default_before
                and check())
        return None

    def command_for(name):
        require(name in GAMES)
        return [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                '--registration-commit', commit, '--binding-sha256', binding_sha, '--game', name]

    driver_module = modules['driver']
    driver = driver_module.LocalDriver(manifest=manifest, game_root=GAME, stage=stage, commit=commit,
                                      command_for=command_for, check=check)
    original_claim = driver.claim

    def claim():
        require(original_claim() is True)
        _write(stage / 'manifest.local.json', raw)
        prepared[0] = True
        return check()

    driver.claim = claim

    def authorize(plan):
        name = plan['game']
        require(plan in plans and name in GAMES and check())
        _write(stage / 'job-results' / (name + '.permit.local.json'),
               marker('permit', commit=commit, binding_sha=binding_sha, game=name))
        return True

    result = runner.Coordinator().run(manifest=manifest, plans=plans, limits=reg['limits'],
        driver=driver, check=check, authorize_game=authorize,
        save_result=lambda payload: save_result(payload=payload, manifest=manifest, driver=driver,
            driver_module=driver_module, limits=reg['limits'], binding_sha=binding_sha,
            projection=projection, prefix=prefix, check=check, remaining=remaining), clock=coordinator_clock)
    require(result.get('passed') is True and result.get('games') == 8
            and result.get('official_replay_passes') == 8 and result.get('promotion') is False
            and check())
    # Repeat the existing complete source, installed engine and runtime check.
    require(verify_again() and validate_assets(manifest, driver_module)
            and driver_module._hash(GAME, GAME / 'main.py', driver_module.MAX_SOURCE) == default_before
            and binding_context(prep, binding_sha)[0] == raw and check())
    return dict(status='passed', games=8, official_replay_passes=8, comparisons=4,
                training=False, independent_improvement=False, promotion=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--game')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        require(not (args.preflight and args.game is not None))
        result = (preflight(args.registration_commit, args.binding_sha256) if args.preflight
                  else run(args.registration_commit, args.binding_sha256, args.game))
        if result is not None:
            print(json.dumps(result))
        return 0
    except BaseException as exc:
        category = ('memory_gate' if isinstance(exc, MemoryError) else
                    'time_gate' if isinstance(exc, TimeoutError) else
                    'missing_file' if isinstance(exc, FileNotFoundError) else
                    'io_error' if isinstance(exc, OSError) else 'invalid_or_runtime')
        print(json.dumps({'status': 'failed', 'category': category, 'retry_allowed': False}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
