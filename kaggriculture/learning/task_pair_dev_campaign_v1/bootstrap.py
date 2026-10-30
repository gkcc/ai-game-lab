"""Registered Windows entry for the consumed-seed development diagnostic.

The standard-library snapshot stub checks this bootstrap and its identity
helper against committed bytes before executing either. The public
registration and runtime bytes are then checked before private metadata or
other project modules are loaded. Private identities, observations and output
paths are never printed. This is a local reproducibility check, not a file
lock, third-party timestamp or protection against a hostile same-user process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType
import uuid


PUBLIC_ROOT = Path(__file__).resolve().parents[3]
GAME_SOURCE = PUBLIC_ROOT / 'kaggriculture'
STAGE_NAME = 'task-pair-dev-games-v1-20261030'
OLD_PANEL = 'artifacts/task-teacher-merged-eval-v3-formal-20261030'
SUPERVISOR = 'artifacts/task-pair-wsl-training-supervisor-v1-20261030/supervisor-result.local.json'
MODEL_DIR = 'artifacts/task-pair-wsl-training-v1-20261030'
PREBIND = 'task-pair-dev-v1-prebinding.local.json'
STATUS = 'CURRENT_PANEL.json'
MANIFEST = 'task-pair-dev-v1-manifest.local.json'
EXPORT = 'review/evidence/task-bc-v1-20261023-run1/model/selected-export.json'
PREFIX = 'kaggriculture/learning/task_prefix_diagnostic_v2/runner.py'
GUARD = 'kaggriculture/learning/task_single_intervention_v1/trusted_imports.py'
MAX_PRIVATE = 4 * 1024 * 1024


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate private metadata key')
        result[key] = value
    return result


def _constant(_value):
    raise ValueError('Nonfinite private metadata')


def _read(path, maximum=MAX_PRIVATE):
    with Path(path).open('rb') as stream:
        data = stream.read(maximum + 1)
    _require(0 < len(data) <= maximum, 'Private metadata size limit')
    return data


def _json(data):
    value = json.loads(data.decode('utf-8-sig'), object_pairs_hook=_pairs, parse_constant=_constant)
    _require(type(value) is dict, 'Private metadata must be an object')
    return value


def _module(name, source, path):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def _identity():
    # The stdlib snapshot stub checked these exact bytes against the committed
    # Git blob before this bootstrap ran. Never re-read and execute a new copy.
    raw = globals().get('_AUTH_IDENTITY_BYTES')
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_PRIVATE, 'Identity snapshot missing')
    path = Path(__file__).with_name('identity.py')
    return _module('_task_pair_dev_identity', raw, path)


def _absolute(path):
    path = Path(path)
    _require(path.is_absolute() and path.resolve(strict=True) == path, 'Invalid local root')
    return path


@contextmanager
def _quiet_imports():
    """Discard Python streams and fd 1/2 while loading known dependencies."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = []
    with open(os.devnull, 'w', encoding='utf-8') as sink:
        try:
            saved.append(os.dup(1))
            saved.append(os.dup(2))
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            failed = False
            for number, descriptor in enumerate(saved, 1):
                try:
                    os.dup2(descriptor, number)
                except OSError:
                    failed = True
                finally:
                    os.close(descriptor)
            if failed:
                raise RuntimeError('Import output restoration failed')


def _configure_thread_environment():
    """Match the previous coordinator before NumPy loads in this process.

    Child processes inherit these process-local values. This does not impose an
    OS thread limit or modify the user's persistent environment.
    """
    _require('numpy' not in sys.modules, 'NumPy was imported before thread configuration')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'


def _preload_dependencies(reg):
    """Load the known top-level runtime packages before adding project paths."""
    _configure_thread_environment()
    site = (Path(sys.prefix) / 'Lib/site-packages').resolve(strict=True)
    packages = {}
    with _quiet_imports():
        for name in ('numpy', 'psutil', 'kaggle_environments'):
            spec = importlib.util.find_spec(name)
            _require(spec is not None and spec.origin is not None
                     and Path(spec.origin).resolve(strict=True).is_relative_to(site),
                     'Dependency origin differs before import')
            module = importlib.import_module(name)
            _require(Path(module.__file__).resolve(strict=True).is_relative_to(site),
                     'Dependency origin differs after import')
            packages[name] = module
        rules = importlib.import_module('kaggle_environments.envs.kaggriculture.kaggriculture')
        rule_path = Path(packages['kaggle_environments'].__file__).resolve(strict=True).parent / 'envs/kaggriculture/kaggriculture.py'
        _require(Path(rules.__file__).resolve(strict=True) == rule_path
                 and _sha(_read(rule_path)) == reg.get('engine_sha256'), 'Official rule origin differs after import')


def _check_new_packages(game_root, reg):
    """Reject local/public module shadows before importing new packages."""
    private = game_root / 'learning'
    public = GAME_SOURCE / 'learning'
    suffixes = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    packages = {
        'task_pair_dev_games_v1': ('panel', 'worker', 'capture'),
        'task_pair_dev_metrics_v1': ('consumer', 'panel', 'replay_pair'),
        'task_pair_dev_campaign_v1': ('manifest', 'history', 'child', 'campaign',
                                      'acceptance', 'driver', 'evidence', 'projection',
                                      'identity', 'bootstrap'),
    }
    for name, modules in packages.items():
        _require(not os.path.lexists(private / name), 'New package exists in game root')
        for base in (private, public):
            _require(not any(os.path.lexists(base / (name + ext)) for ext in suffixes),
                     'New package module shadow')
        folder = public / name
        _require(folder.is_dir() and folder.resolve(strict=True) == folder, 'New package path differs')
        init = folder / '__init__.py'
        if name == 'task_pair_dev_campaign_v1':
            _require(not os.path.lexists(init), 'Unexpected campaign initializer')
        else:
            relative = 'kaggriculture/learning/' + name + '/__init__.py'
            _require(_sha(_read(init)) == reg['public_sources'][relative], 'Package initializer differs')
        _require(not any(os.path.lexists(folder / ('__init__' + ext)) for ext in suffixes if ext != '.py'),
                 'Package initializer shadow')
        for module in modules:
            relative = 'kaggriculture/learning/' + name + '/' + module + '.py'
            _require(_sha(_read(folder / (module + '.py'))) == reg['public_sources'][relative],
                     'New module source differs')
            _require(not os.path.lexists(folder / module)
                     and not any(os.path.lexists(folder / (module + ext)) for ext in suffixes if ext != '.py'),
                     'New module shadow')


def _runtime(game_root, reg):
    """Bind old code to the game root and only new packages to the checkout."""
    _preload_dependencies(reg)
    _check_new_packages(game_root, reg)
    guard_path = PUBLIC_ROOT / GUARD
    raw = _read(guard_path)
    _require(_sha(raw) == reg['public_sources'][GUARD], 'Import guard differs')
    guard = _module('_task_pair_dev_import_guard', raw, guard_path)
    guard.bind(game_root)
    namespace = sys.modules['learning']
    namespace.__path__.append(str(GAME_SOURCE / 'learning'))
    loaded = {}
    for name in ('manifest', 'history', 'child', 'campaign', 'acceptance', 'driver',
                 'projection', 'evidence'):
        module = importlib.import_module('learning.task_pair_dev_campaign_v1.' + name)
        _require(Path(module.__file__).resolve().is_relative_to(GAME_SOURCE), 'New package origin differs')
        loaded[name] = module
    for name in ('learning.task_pair_dev_games_v1.worker',
                 'learning.task_pair_dev_metrics_v1.consumer',
                 'learning.task_pair_dev_metrics_v1.replay_pair'):
        module = importlib.import_module(name)
        _require(Path(module.__file__).resolve().is_relative_to(GAME_SOURCE), 'New package origin differs')
    crop = importlib.import_module('learning.bc_crop.agent')
    decision = importlib.import_module('decision.semantics')
    _require(Path(crop.ROOT).resolve() == game_root
             and Path(decision.__file__).resolve().is_relative_to(game_root), 'Game code origin differs')
    return loaded


def _private_status():
    directory = PUBLIC_ROOT.parent / 'private'
    status_path, binding_path = directory / STATUS, directory / PREBIND
    current = _json(_read(status_path))
    dev = current.get('current_pair_dev_diagnostic')
    _require(type(dev) is dict, 'Prior private binding missing')
    binding_raw = _read(binding_path, 65536)
    _require(_sha(binding_raw) == dev.get('prebinding_record_sha256'), 'Prior private binding differs')
    binding = _json(binding_raw)
    prior = current.get('current_pair_training', {}).get('wsl_contracts', {})
    _require(binding.get('schema') == 'task-pair-dev-prebinding-v1'
             and binding.get('prior_training_result_commit') == prior.get('public_result_commit')
             and binding.get('prior_training_result_commit') == '5a66216f99a8e467818978c96e707f6045834709'
             and binding.get('historical_result_commit') == '3a3e3172b0403a171a38f6e78e37d07f387be2cb'
             and binding.get('seed_roles') == ['S02', 'S03']
             and binding.get('seed_scope') == 'previously_consumed_development'
             and prior.get('attempt_status') == 'completed_exit_observed'
             and type(prior.get('operator_observed_wrapper_exit_code')) is int
             and prior['operator_observed_wrapper_exit_code'] == 0,
             'Previous execution or private binding differs')
    return status_path, current, dev, binding


def _atomic_status(path, original, dev):
    updated = dict(original)
    updated['current_pair_dev_diagnostic'] = dict(dev)
    raw = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _require(_read(pending) == raw, 'Private status readback differs')
    os.replace(pending, path)
    _require(_json(_read(path)) == updated, 'Private status replacement differs')


def _manifest_document(value, commit, prebinding_sha):
    from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS, POLICIES
    panel = value.panel
    return dict(schema='task-pair-dev-private-manifest-v1', registration_commit=commit,
        prebinding_sha256=prebinding_sha,
        seeds={role: panel.seeds[i] for i, role in enumerate(ALIASES)},
        models={role: dict(path=panel.models[i][0], sha256=panel.models[i][1])
                for i, role in enumerate(POLICIES)},
        crop=dict(path=panel.crop[0], sha256=panel.crop[1]),
        opponents={role: dict(path=value.opponent(role)[0], sha256=value.opponent(role)[1])
                   for role in OPPONENTS}, worker_sha256=panel.worker_sha)


def _stage(game_root):
    root = game_root / 'artifacts'
    _require(root.is_dir() and root.resolve(strict=True) == root, 'Invalid artifacts root')
    return root / STAGE_NAME


def prepare(*, game_root, registration_commit):
    """Create and privately bind a single pre-registered development attempt."""
    game_root = _absolute(game_root)
    identity = _identity()
    reg, old_reg, freeze = identity.verify_public(repo=PUBLIC_ROOT, game_root=game_root,
                                                  registration_commit=registration_commit)
    modules = _runtime(game_root, reg)
    modules['campaign'].limits_snapshot(reg.get('limits'))
    status_path, current, dev, binding = _private_status()
    _require(dev.get('state') == 'preparing_sources_no_games', 'Preparation already consumed')
    stage = _stage(game_root)
    manifest_path = PUBLIC_ROOT.parent / 'private' / MANIFEST
    _require(not os.path.lexists(stage) and not os.path.lexists(manifest_path),
             'Development attempt already exists')
    dev = dict(dev, state='binding_started_no_games', private_reads_may_have_started=True)
    _atomic_status(status_path, current, dev)
    def check():
        identity.verify_public(repo=PUBLIC_ROOT, game_root=game_root, registration_commit=registration_commit)
        _, _, fresh, prior = _private_status()
        return fresh == dev and prior == binding
    historical = game_root / OLD_PANEL
    source = _read(PUBLIC_ROOT / PREFIX)
    seeds = modules['history'].historical_roles(source=source, source_sha256=reg['public_sources'][PREFIX],
        source_path=PUBLIC_ROOT/PREFIX, panel_path=historical, status_path=status_path, check=check)
    supervisor = _read(game_root / SUPERVISOR, 65536)
    export = _read(PUBLIC_ROOT / EXPORT, 65536)
    model_paths = {role: MODEL_DIR + '/' + role + '.npz' for role in ('baseline', 'control', 'auxiliary')}
    value = modules['history'].bind_prior(seeds=seeds, supervisor_raw=supervisor,
        supervisor_sha256=binding['prior_supervisor_sha256'], selected_export_raw=export,
        selected_export_sha256=reg['public_sources'][EXPORT], historical_registration=old_reg,
        historical_freeze=freeze, historical_seed_raw=_read(historical/'private-seeds.local.json', 16384),
        game_root=game_root, historical_panel_relative=OLD_PANEL,
        model_paths=model_paths, check=check)
    document = _manifest_document(value, registration_commit, dev['prebinding_record_sha256'])
    data = (json.dumps(document, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with manifest_path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _require(_read(manifest_path) == data, 'Private manifest readback differs')
    status_path, current, fresh, latest_binding = _private_status()
    _require(fresh == dev and latest_binding == binding, 'Private binding changed during preparation')
    dev = dict(dev, state='prepared_no_games', registration_commit=registration_commit,
               private_seed_or_model_loaded=True, model_arrays_loaded=False,
               manifest_sha256=_sha(data), stage_name=STAGE_NAME)
    _atomic_status(status_path, current, dev)


def _load_prepared(*, game_root, registration_commit, identity, modules, allowed_states):
    status_path, current, dev, binding = _private_status()
    _require(dev.get('state') in allowed_states and dev.get('registration_commit') == registration_commit
             and dev.get('stage_name') == STAGE_NAME, 'Private attempt state differs')
    raw = _read(PUBLIC_ROOT.parent / 'private' / MANIFEST)
    _require(_sha(raw) == dev.get('manifest_sha256'), 'Private manifest differs')
    document = _json(raw)
    fields = {'schema', 'registration_commit', 'prebinding_sha256', 'seeds', 'models',
              'crop', 'opponents', 'worker_sha256'}
    _require(set(document) == fields and document.get('schema') == 'task-pair-dev-private-manifest-v1'
             and document.get('registration_commit') == registration_commit
             and document.get('prebinding_sha256') == dev['prebinding_record_sha256'],
             'Private manifest binding differs')
    manifest = modules['manifest'].build_manifest(seeds=document['seeds'], models=document['models'],
        crop=document['crop'], opponents=document['opponents'], worker_sha256=document['worker_sha256'])
    source = _read(game_root / 'learning/task_anchor_once_v1/games.py', 4 * 1024 * 1024)
    _require(_sha(source) == manifest.panel.worker_sha, 'Worker snapshot differs')
    return status_path, current, dev, binding, raw, manifest, source


def _verify_assets(manifest, game_root, history):
    """Rehash each frozen input; no model arrays or replay are opened."""
    for relative, expected in (*manifest.panel.models, manifest.panel.crop):
        path = history._safe_relative_file(game_root, relative, '.npz', artifacts=True)
        _require(history._file_digest(path, history.MODEL_LIMIT) == expected, 'Model bytes changed')
    for role in ('market', 'round2'):
        relative, expected = manifest.opponent(role)
        path = history._safe_relative_file(game_root, relative, '.py', artifacts=True)
        _require(history._file_digest(path, history.METADATA_LIMIT) == expected, 'Opponent bytes changed')
    path = history._safe_relative_file(game_root, 'learning/task_anchor_once_v1/games.py', '.py')
    _require(history._file_digest(path, history.METADATA_LIMIT) == manifest.panel.worker_sha,
             'Worker bytes changed')


def _recheck(*, identity, game_root, registration_commit, manifest_raw, binding,
             required_state):
    # Full source/engine verification occurs once when this process starts.
    # These repeated checks bind its running inputs; they are not file locks.
    if identity._git(PUBLIC_ROOT, 'rev-parse', 'HEAD').decode().strip() != registration_commit:
        return False
    if identity._git(PUBLIC_ROOT, 'status', '--porcelain', '--untracked-files=all'):
        return False
    status_path, _, dev, latest = _private_status()
    if (dev.get('state') != required_state or dev.get('registration_commit') != registration_commit
            or dev.get('stage_name') != STAGE_NAME or latest != binding
            or _sha(manifest_raw) != dev.get('manifest_sha256')):
        return False
    return _read(PUBLIC_ROOT.parent / 'private' / MANIFEST) == manifest_raw


_SNAPSHOT_STUB = '''import pathlib, re, subprocess, sys
repo=pathlib.Path(sys.argv.pop(1)); commit=sys.argv.pop(1)
if re.fullmatch(r"[0-9a-f]{40}",commit) is None: raise SystemExit(1)
def snapshot(rel):
 path=repo.joinpath(*rel.split("/"))
 with path.open("rb") as stream: data=stream.read(4194305)
 if not data or len(data)>4194304: raise SystemExit(1)
 size=subprocess.run(["git","-C",str(repo),"cat-file","-s",commit+":"+rel],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True,timeout=30).stdout
 if len(size)>32 or not size.strip().isdigit() or int(size)!=len(data): raise SystemExit(1)
 result=subprocess.run(["git","-C",str(repo),"show",commit+":"+rel],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True,timeout=30)
 if data!=result.stdout: raise SystemExit(1)
 return path,data
path,data=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py")
_,identity=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/identity.py")
exec(compile(data,str(path),"exec",dont_inherit=True),{"__name__":"__main__","__file__":str(path),"_AUTH_IDENTITY_BYTES":identity})
'''


def _command(game_root, registration_commit, mode, game=None):
    command = [sys.executable, '-I', '-B', '-c', _SNAPSHOT_STUB, str(PUBLIC_ROOT), registration_commit,
               '--game-root', str(game_root), '--registration-commit', registration_commit]
    if mode == 'game':
        _require(type(game) is str and game, 'Invalid game alias')
        command.extend(['--game', game])
    elif mode in ('run', 'prepare', 'project'):
        command.append('--' + mode)
    else:
        raise ValueError('Invalid command mode')
    return command


def execute(*, game_root, registration_commit, game=None):
    """Enter the authenticated full campaign or one supervised child."""
    game_root = _absolute(game_root)
    identity = _identity()
    reg, _, _ = identity.verify_public(repo=PUBLIC_ROOT, game_root=game_root,
                                       registration_commit=registration_commit)
    modules = _runtime(game_root, reg)
    limits = modules['campaign'].limits_snapshot(reg.get('limits'))
    allowed = ('running',) if game is not None else ('prepared_no_games',)
    status_path, current, dev, binding, raw, manifest, source = _load_prepared(
        game_root=game_root, registration_commit=registration_commit, identity=identity,
        modules=modules, allowed_states=allowed)
    _verify_assets(manifest, game_root, modules['history'])
    stage = _stage(game_root)
    if game is not None:
        choices = [plan for plan in manifest.panel.plans() if plan['game'] == game]
        _require(len(choices) == 1, 'Unknown game alias')
        check = lambda: _recheck(identity=identity, game_root=game_root,
            registration_commit=registration_commit, manifest_raw=raw, binding=binding,
            required_state='running')
        modules['child'].run_one(manifest=manifest, worker_source=source, game_root=game_root,
                                 stage=stage, plan=choices[0], check=check)
        return None
    _require(not os.path.lexists(stage), 'Development campaign already started')
    dev = dict(dev, state='running', new_games_may_have_started=True)
    _atomic_status(status_path, current, dev)
    check = lambda: _recheck(identity=identity, game_root=game_root,
        registration_commit=registration_commit, manifest_raw=raw, binding=binding,
        required_state='running')
    driver = modules['driver'].LocalDriver(manifest=manifest, game_root=game_root, stage=stage,
        commit=registration_commit, command_for=lambda alias: _command(game_root, registration_commit, 'game', alias),
        check=check)
    try:
        result = modules['campaign'].run(manifest=manifest, limits=limits, driver=driver, check=check)
        identity.verify_public(repo=PUBLIC_ROOT, game_root=game_root, registration_commit=registration_commit)
        _verify_assets(manifest, game_root, modules['history'])
    except BaseException:
        path, fresh, existing, prior = _private_status()
        if existing.get('state') == 'running' and prior == binding:
            _atomic_status(path, fresh, dict(existing, state='failed_no_retry'))
        raise
    path, fresh, existing, prior = _private_status()
    _require(existing.get('state') == 'running' and prior == binding, 'Private status changed')
    _atomic_status(path, fresh, dict(existing, state='finished_no_promotion',
                                    complete_games=result['games'], promotion=False))
    return result


def project_evidence(*, game_root, registration_commit):
    """Recheck completed local evidence and write only the allowlisted projection."""
    game_root = _absolute(game_root)
    identity = _identity()
    reg, _, _ = identity.verify_public(repo=PUBLIC_ROOT, game_root=game_root,
                                       registration_commit=registration_commit)
    modules = _runtime(game_root, reg)
    limits = modules['campaign'].limits_snapshot(reg.get('limits'))
    _, _, _, binding, raw, manifest, _ = _load_prepared(
        game_root=game_root, registration_commit=registration_commit, identity=identity,
        modules=modules, allowed_states=('finished_no_promotion',))
    _verify_assets(manifest, game_root, modules['history'])
    deadline = time.monotonic() + limits['campaign_wall_seconds']
    check = lambda: (time.monotonic() < deadline and _recheck(
        identity=identity, game_root=game_root, registration_commit=registration_commit,
        manifest_raw=raw, binding=binding, required_state='finished_no_promotion'))
    stage = _stage(game_root)
    local_driver = modules['driver'].LocalDriver(manifest=manifest, game_root=game_root,
        stage=stage, commit=registration_commit, command_for=lambda _alias: [], check=check)
    # The completed stage is fixed by the private attempt; this enables the
    # existing read-only resource preflight, not stage creation or adoption.
    local_driver.claimed = True
    _require(local_driver.preflight_comparison(limits, deadline-time.monotonic()) is True,
             'Projection resource preflight failed')
    projected = modules['evidence'].verify_and_project(manifest=manifest,
                                                        local_driver=local_driver, check=check)
    identity.verify_public(repo=PUBLIC_ROOT, game_root=game_root, registration_commit=registration_commit)
    _verify_assets(manifest, game_root, modules['history'])
    _require(check() is True, 'Projection identity changed')
    destination = stage / 'public-projection.local.json'
    if os.path.lexists(destination):
        existing = modules['driver']._read(stage, destination, modules['driver'].MAX_SUMMARY)
        _require(modules['acceptance']._json(existing, modules['driver'].MAX_SUMMARY) == projected,
                 'Existing projection differs')
    else:
        modules['driver']._write_new(stage, destination, projected)
    _require(check() is True, 'Projection identity changed after write')


def _startup():
    _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__,
             'Use isolated non-optimized Python without bytecode writes')


def main(argv=None):
    _startup()
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-root', required=True, type=Path)
    parser.add_argument('--registration-commit', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--project', action='store_true')
    mode.add_argument('--game')
    args = parser.parse_args(argv)
    if args.prepare:
        prepare(game_root=args.game_root, registration_commit=args.registration_commit)
    elif args.project:
        project_evidence(game_root=args.game_root, registration_commit=args.registration_commit)
    else:
        result = execute(game_root=args.game_root, registration_commit=args.registration_commit,
                         game=args.game)
        if result is not None:
            print(json.dumps(result, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        raise SystemExit('Registered development entry failed; inspect private state') from None
