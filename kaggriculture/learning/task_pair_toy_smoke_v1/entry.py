"""Authenticated entry for a separate, untrained two-game toy smoke.

Invoke through SNAPSHOT_STUB, not by executing this file directly.  The caller
must commit a separately reviewed toy registration before running this entry.
The original failed 24-game attempt and its private status are never accessed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import ModuleType


PUBLIC_ROOT = Path(__file__).resolve().parents[3]
GAME_SOURCE = PUBLIC_ROOT / 'kaggriculture'
TOY = 'kaggriculture/learning/task_pair_toy_smoke_v1/'
CAMPAIGN = 'kaggriculture/learning/task_pair_dev_campaign_v1/'
CROP_SCHEMA = 'review/evidence/bc-crop-v1-20261023/data-manifest.json'
TASK_SCHEMA = 'review/evidence/task-bc-v1-20261023-run1/cache/schema.json'
_ALLOWED = ('dev-S02-market-0-baseline', 'dev-S02-market-0-control')
_MANIFEST = 'toy-manifest.local.json'
_RESULT = 'toy-smoke-result.local.json'
_RETURNED = 'toy-runner-returned.local.json'
_MAX_SMALL = 512 * 1024
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _module(name, raw, path):
    _require(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024,
             'Authenticated source snapshot missing')
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def _snapshot_modules():
    """Execute only the three in-memory snapshots supplied by the stub."""
    identity = _module('_toy_identity_snapshot', globals().get('_AUTH_IDENTITY_BYTES'),
                       PUBLIC_ROOT / CAMPAIGN / 'identity.py')
    bootstrap = _module('_toy_bootstrap_snapshot', globals().get('_AUTH_BOOTSTRAP_BYTES'),
                        PUBLIC_ROOT / CAMPAIGN / 'bootstrap.py')
    auth = _module('_toy_auth_snapshot', globals().get('_AUTH_TOY_AUTH_BYTES'),
                   PUBLIC_ROOT / TOY / 'toy_auth.py')
    return identity, bootstrap, auth


def _check_toy_package(game_root, reg):
    """Reject unregistered local, bytecode and native toy package shadows."""
    name = 'task_pair_toy_smoke_v1'
    private, public = game_root / 'learning', GAME_SOURCE / 'learning'
    extensions = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    _require(not os.path.lexists(private / name), 'Toy package exists in game root')
    for base in (private, public):
        _require(not any(os.path.lexists(base / (name + ext)) for ext in extensions),
                 'Toy package module shadow')
    folder = public / name
    _require(folder.is_dir() and folder.resolve(strict=True) == folder
             and not os.path.lexists(folder / '__pycache__'), 'Toy package path differs')
    for module in ('__init__', 'toy_auth', 'toy_weights', 'runner', 'entry'):
        relative = TOY + module + '.py'
        source = folder / (module + '.py')
        with source.open('rb') as stream:
            data = stream.read(4 * 1024 * 1024 + 1)
        _require(0 < len(data) <= 4 * 1024 * 1024
                 and _sha(data) == reg['public_sources'][relative],
                 'Toy module source differs')
        _require(not os.path.lexists(folder / module)
                 and not any(os.path.lexists(folder / (module + ext))
                             for ext in extensions if ext != '.py'), 'Toy module shadow')
    prefix = 'learning.' + name
    _require(not any(loaded == prefix or loaded.startswith(prefix + '.') for loaded in sys.modules),
             'Toy package was imported early')


def _load_toy(name):
    module = importlib.import_module('learning.task_pair_toy_smoke_v1.' + name)
    expected = GAME_SOURCE / 'learning' / 'task_pair_toy_smoke_v1' / (name + '.py')
    _require(Path(module.__file__).resolve(strict=True) == expected, 'Toy import origin differs')
    return module


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _game_root(bootstrap, value):
    """Bind this smoke to the game project beside this publication workspace."""
    root = bootstrap._absolute(value)
    expected = PUBLIC_ROOT.parent.parent / 'kaggriculture'
    _require(expected.is_absolute() and expected.resolve(strict=True) == expected and root == expected,
             'Toy game project root differs')
    return root


def _build_context(game_root, registration_commit):
    """Authenticate sources, bind the old runtime, and create only in-memory toys."""
    _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__,
             'Isolated non-optimized Python without bytecode writes required')
    _require(type(registration_commit) is str and _COMMIT.fullmatch(registration_commit),
             'Invalid toy registration commit')
    identity, bootstrap, auth = _snapshot_modules()
    game_root = _game_root(bootstrap, game_root)
    reg, _old, freeze = auth.verify_public(identity, repo=PUBLIC_ROOT, game_root=game_root,
                                           registration_commit=registration_commit)
    _check_toy_package(game_root, reg)
    modules = bootstrap._runtime(game_root, reg)
    weights, runner = _load_toy('toy_weights'), _load_toy('runner')
    limits = runner.limits_snapshot(reg.get('limits'))
    crop_doc = identity._json(identity._read(PUBLIC_ROOT / CROP_SCHEMA))
    task_doc = identity._json(identity._read(PUBLIC_ROOT / TASK_SCHEMA))
    crop = {key: crop_doc[key] for key in ('features', 'schema_version', 'teacher_sha256', 'schema_sha256')}
    crop['crops'] = list(weights.CROPS)
    archives = weights.build(crop_schema=crop, task_schema=task_doc)
    expected = reg['toy_model_sha256']
    _require(type(archives) is dict and archives.get('trained') is False
             and archives['crop']['sha256'] == expected['crop']
             and _sha(archives['crop']['data']) == expected['crop']
             and all(archives['tasks'][role]['sha256'] == expected[role]
                     and _sha(archives['tasks'][role]['data']) == expected[role]
                     for role in ('baseline', 'control', 'auxiliary')),
             'Toy checkpoint identities differ')
    stage = game_root / 'artifacts' / auth.STAGE_NAME
    opponent_paths = {'market': 'bots/heuristic_v1_market.py', 'round2': 'bots/heuristic_v1.py'}
    opponents = {}
    for role, relative in opponent_paths.items():
        public = 'kaggriculture/' + relative
        digest = reg['public_sources'][public]
        _require(_sha(identity._read(game_root / relative)) == digest, 'Toy opponent bytes differ')
        opponents[role] = dict(path=relative, sha256=digest)
    source_relative = 'learning/task_anchor_once_v1/games.py'
    source = identity._read(game_root / source_relative)
    worker_sha = freeze['runtime_files'][source_relative]
    _require(_sha(source) == worker_sha, 'Toy worker bytes differ')
    relative_stage = 'artifacts/' + auth.STAGE_NAME + '/models/'
    models = {role: dict(path=relative_stage + role + '.npz', sha256=expected[role])
              for role in ('baseline', 'control', 'auxiliary')}
    crop_binding = dict(path=relative_stage + 'crop.npz', sha256=expected['crop'])
    manifest = modules['manifest'].build_manifest(seeds={'S02': 11017, 'S03': 22029},
        models=models, crop=crop_binding, opponents=opponents, worker_sha256=worker_sha)
    plans = [next(plan for plan in manifest.panel.plans() if plan['game'] == alias) for alias in _ALLOWED]
    doc = dict(schema='task-pair-toy-smoke-manifest-v1', registration_commit=registration_commit,
               stage_name=auth.STAGE_NAME, synthetic_seeds={'S02': 11017, 'S03': 22029},
               models=expected, opponents={role: opponents[role]['sha256'] for role in opponents},
               worker_sha256=worker_sha, trained=False)
    attempt = dict(schema='task-pair-toy-smoke-attempt-v1', registration_commit=registration_commit,
                   stage_name=auth.STAGE_NAME, scope='synthetic_toy', retry_allowed=False)
    reg_raw = identity._read(PUBLIC_ROOT / auth.REGISTRATION)
    _require(identity._git(PUBLIC_ROOT, 'show', registration_commit + ':' + auth.REGISTRATION) == reg_raw,
             'Toy registration changed')
    return dict(identity=identity, auth=auth, bootstrap=bootstrap, modules=modules, runner=runner,
                game_root=game_root, stage=stage, commit=registration_commit, reg_raw=reg_raw,
                manifest=manifest, manifest_raw=_json_bytes(doc), plans=plans, source=source,
                attempt_path=game_root / 'artifacts' / (auth.STAGE_NAME + '.attempt.local.json'),
                attempt_raw=_json_bytes(attempt), archives=archives, task_schema=task_doc, limits=limits)


def _git_check(ctx):
    identity, commit, auth = ctx['identity'], ctx['commit'], ctx['auth']
    if identity._git(PUBLIC_ROOT, 'rev-parse', 'HEAD').decode().strip() != commit:
        return False
    if identity._git(PUBLIC_ROOT, 'status', '--porcelain', '--untracked-files=all'):
        return False
    return identity._read(PUBLIC_ROOT / auth.REGISTRATION) == ctx['reg_raw']


def _stage_check(ctx):
    """Recheck only this stage's public-synthetic manifest and bound bytes."""
    driver, stage, root = ctx['modules']['driver'], ctx['stage'], ctx['game_root']
    driver._no_links(root, stage)
    if not stage.is_dir() or os.path.lexists(stage / 'campaign-failure.local.json'):
        return False
    if driver._read(stage, stage / _MANIFEST, _MAX_SMALL) != ctx['manifest_raw']:
        return False
    for relative, digest in (*ctx['manifest'].panel.models, ctx['manifest'].panel.crop):
        if driver._hash(root, root / relative, driver.MAX_MODEL) != digest:
            return False
    for role in ('market', 'round2'):
        _, digest = ctx['manifest'].opponent(role)
        if driver._hash(stage, stage / 'sources' / (role + '.py'), driver.MAX_SOURCE) != digest:
            return False
    return True


def _attempt_check(ctx):
    driver = ctx['modules']['driver']
    return driver._read(ctx['game_root'], ctx['attempt_path'], _MAX_SMALL) == ctx['attempt_raw']


def _check(ctx, prepared):
    return _git_check(ctx) and _attempt_check(ctx) and (not prepared or _stage_check(ctx))


def _claim_attempt(ctx):
    """Create the one-use local marker before stage creation or preflight."""
    root, marker = ctx['game_root'], ctx['attempt_path']
    driver = ctx['modules']['driver']
    driver._no_links(root, marker)
    _require(type(ctx['attempt_raw']) is bytes and 0 < len(ctx['attempt_raw']) <= _MAX_SMALL
             and not os.path.lexists(marker) and not os.path.lexists(ctx['stage']),
             'Toy attempt already exists')
    with marker.open('xb') as stream:
        if stream.write(ctx['attempt_raw']) != len(ctx['attempt_raw']):
            raise OSError('Incomplete toy attempt')
        stream.flush()
        os.fsync(stream.fileno())
    _require(_attempt_check(ctx), 'Toy attempt readback differs')


def _write_exact(path, data):
    with path.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('Incomplete toy output')


def _ensure_room(ctx, size):
    """Sample the stage limit and disk floor before a small receipt write."""
    _require(type(size) is int and 0 < size <= _MAX_SMALL, 'Invalid toy receipt size')
    used = ctx['modules']['driver']._owned_size(ctx['stage'])
    free = shutil.disk_usage(ctx['stage']).free
    limits = ctx['limits']
    _require(type(used) is int and used >= 0 and type(free) is int
             and used + size <= limits['total_output_bytes']
             and free >= limits['disk_free_floor_bytes'] + size, 'Toy receipt resource limit')


def _game_marker(ctx, game, kind):
    _require(game in _ALLOWED and kind in ('permit', 'consumed'), 'Invalid toy game marker')
    path = ctx['stage'] / 'job-results' / (game + '.' + kind + '.local.json')
    raw = _json_bytes(dict(schema='task-pair-toy-game-' + kind + '-v1', game=game,
                           registration_commit=ctx['commit'], attempt_sha256=_sha(ctx['attempt_raw']),
                           manifest_sha256=_sha(ctx['manifest_raw']), retry_allowed=False))
    return path, raw


def _write_game_marker(ctx, game, kind):
    """Write one exclusive receipt under a separately bounded stage budget."""
    path, raw = _game_marker(ctx, game, kind)
    _ensure_room(ctx, len(raw))
    from learning.task_anchor_once_v1.storage import owned_writes
    relative = 'job-results/' + path.name
    with owned_writes(ctx['stage'], _MAX_SMALL, allowed_paths=[relative]):
        _write_exact(path, raw)
    _require(ctx['modules']['driver']._read(ctx['stage'], path, _MAX_SMALL) == raw,
             'Toy game marker readback differs')


def _authorize_game(ctx, plan):
    """Grant one registered game immediately before its supervised launch."""
    _require(type(plan) is dict and plan in ctx['plans'] and _check(ctx, True),
             'Toy game is not authorized')
    ctx['manifest'].panel.validate_plan(plan)
    game = plan['game']
    permit, _ = _game_marker(ctx, game, 'permit')
    consumed, _ = _game_marker(ctx, game, 'consumed')
    _require(not os.path.lexists(permit) and not os.path.lexists(consumed),
             'Toy game authorization already exists')
    _write_game_marker(ctx, game, 'permit')


def _consume_game(ctx, game):
    """Consume the parent's authorization before entering the game worker.

    This prevents a second consumption. It does not prove the parent Job is
    alive, and is not a defense against a hostile same-user process racing it.
    """
    permit, expected = _game_marker(ctx, game, 'permit')
    consumed, _ = _game_marker(ctx, game, 'consumed')
    _require(not os.path.lexists(consumed), 'Toy game authorization already consumed')
    _require(ctx['modules']['driver']._read(ctx['stage'], permit, _MAX_SMALL) == expected,
             'Toy game authorization missing or changed')
    _write_game_marker(ctx, game, 'consumed')


def _prepare(ctx, manifest, local_driver, allowance):
    """Write authenticated synthetic input bytes only in the claimed stage."""
    _require(manifest is ctx['manifest'] and local_driver.stage == ctx['stage']
             and local_driver.claimed and type(allowance) is int and allowance > 0,
             'Invalid toy preparation context')
    stage = ctx['stage']
    models = stage / 'models'
    _require(not os.path.lexists(models) and not os.path.lexists(stage / _MANIFEST),
             'Toy inputs already exist')
    model_data = {role: ctx['archives']['tasks'][role]['data']
                  for role in ('baseline', 'control', 'auxiliary')}
    model_data['crop'] = ctx['archives']['crop']['data']
    _require(all(type(value) is bytes and value for value in model_data.values()), 'Invalid toy bytes')
    _require(sum(map(len, model_data.values())) + len(ctx['manifest_raw']) <= allowance,
             'Toy input budget exceeded')
    models.mkdir(exist_ok=False)
    from learning.task_anchor_once_v1.storage import owned_writes
    paths = ['models/' + role + '.npz' for role in model_data] + [_MANIFEST]
    with owned_writes(stage, allowance, allowed_paths=paths):
        for role, data in model_data.items():
            _write_exact(models / (role + '.npz'), data)
        _write_exact(stage / _MANIFEST, ctx['manifest_raw'])
    driver = ctx['modules']['driver']
    _require(driver._read(stage, stage / _MANIFEST, _MAX_SMALL) == ctx['manifest_raw'],
             'Toy manifest readback differs')
    for role, data in model_data.items():
        _require(driver._read(stage, models / (role + '.npz'), driver.MAX_MODEL) == data,
                 'Toy model readback differs')
    _validate_model_choice(ctx)
    return True


def _validate_model_choice(ctx):
    """Load two actual NPZ files and compare their decisions on one toy input."""
    model = importlib.import_module('learning.task_bc_v1.model')
    schema = ctx['task_schema']
    local, shared = list(schema['feature_names']), list(schema['shared_feature_names'])
    left, right = [0.] * len(local), [0.] * len(local)
    index = local.index('target.x')
    left[index], right[index] = 1., 2.
    inputs = dict(schema_version=schema['schema_version'], schema_sha256=schema['schema_sha256'],
                  feature_names=local, shared_feature_names=shared, shared_x=[0.] * len(shared),
                  candidates=[dict(id='left', x=left), dict(id='right', x=right)])
    baseline = model.TaskNetwork(ctx['stage'] / 'models' / 'baseline.npz',
                                 ctx['manifest'].panel.models[0][1])
    control = model.TaskNetwork(ctx['stage'] / 'models' / 'control.npz',
                                ctx['manifest'].panel.models[1][1])
    _require(baseline(deepcopy(inputs)) == 'right' and control(deepcopy(inputs)) == 'left',
             'Toy task heads do not make distinct decisions')


def _child_ready(ctx, game):
    """Check stage ownership/progress; this is not a cross-process lock."""
    _require(game in _ALLOWED and not os.path.lexists(ctx['stage'] / _RESULT)
             and not os.path.lexists(ctx['stage'] / _RETURNED)
             and _attempt_check(ctx) and _stage_check(ctx), 'Toy child stage is not ready')
    driver, stage = ctx['modules']['driver'], ctx['stage']
    started = ctx['identity']._json(driver._read(stage, stage / 'campaign-started.local.json', _MAX_SMALL))
    _require(started.get('schema') == 'task-pair-dev-campaign-start-v1'
             and started.get('registration_commit') == ctx['commit']
             and type(started.get('max_games')) is int and started['max_games'] == 24
             and type(started.get('attempt')) is int and started['attempt'] == 1
             and started.get('retry_allowed') is False, 'Toy stage claim differs')
    _require(not os.path.lexists(stage / 'agents' / (game + '.py')),
             'Toy game has already started')
    if game == _ALLOWED[1]:
        key = _ALLOWED[0]
        path = stage / 'job-results' / (key + '.acceptance.local.json')
        receipt = ctx['identity']._json(driver._read(stage, path, _MAX_SMALL))
        supervision = driver._read(stage, stage / 'job-results' / (key + '.supervision.local.json'), _MAX_SMALL)
        supervision_doc = ctx['identity']._json(supervision)
        _require(receipt.get('schema') == 'task-pair-dev-job-acceptance-v1'
                 and receipt.get('game') == key and receipt.get('passed') is True
                 and receipt.get('retry_allowed') is False
                 and receipt.get('supervision_sha256') == _sha(supervision)
                 and supervision_doc.get('schema') == 'task-pair-dev-job-supervision-v1'
                 and supervision_doc.get('game') == key and supervision_doc.get('retry_allowed') is False
                 and ctx['modules']['acceptance'].supervision_ok(supervision_doc.get('supervision')),
                 'Baseline toy game was not accepted')


def _save(ctx, payload):
    """Persist counts and fixed receipt references, never private event rows."""
    _require(type(payload) is dict and payload.get('scope') == 'synthetic_toy'
             and payload.get('attempted') == 2 and payload.get('official_replay_passes') == 2
             and payload.get('comparisons') == 1 and payload.get('promotion') is False
             and [row.get('game') for row in payload.get('games', [])] == list(_ALLOWED),
             'Toy result differs')
    driver, stage = ctx['modules']['driver'], ctx['stage']
    games = []
    for game in _ALLOWED:
        sup = stage / 'job-results' / (game + '.supervision.local.json')
        accepted = stage / 'job-results' / (game + '.acceptance.local.json')
        permit_path, permit_raw = _game_marker(ctx, game, 'permit')
        consumed_path, consumed_raw = _game_marker(ctx, game, 'consumed')
        _require(driver._read(stage, permit_path, _MAX_SMALL) == permit_raw
                 and driver._read(stage, consumed_path, _MAX_SMALL) == consumed_raw,
                 'Toy game authorization receipts differ')
        games.append(dict(game=game, supervision_sha256=driver._hash(stage, sup, _MAX_SMALL),
                          acceptance_sha256=driver._hash(stage, accepted, _MAX_SMALL),
                          permit_sha256=_sha(permit_raw), consumed_sha256=_sha(consumed_raw)))
    comparison = driver._comparison_file(stage, _ALLOWED[0], 'control')
    result = dict(schema='task-pair-toy-smoke-result-v1', scope='synthetic_toy',
                  phase='intermediate_requires_runner_return_and_observed_parent_exit',
                  games=games, official_replay_passes=2, comparisons=1,
                  comparison_sha256=driver._hash(stage, comparison, _MAX_SMALL),
                  generic_driver_claim_capacity=24, registered_game_limit=2,
                  independent_improvement=False, promotion=False)
    data = _json_bytes(result)
    _ensure_room(ctx, len(data))
    from learning.task_anchor_once_v1.storage import owned_writes
    with owned_writes(stage, _MAX_SMALL, allowed_paths=[_RESULT]):
        _write_exact(stage / _RESULT, data)
    _require(driver._read(stage, stage / _RESULT, _MAX_SMALL) == data, 'Toy result readback differs')
    return True


def _mark_returned(ctx, result):
    """Record inner return only; the outer process exit must be observed separately."""
    _require(type(result) is dict and result.get('passed') is True
             and result.get('scope') == 'synthetic_toy' and result.get('games') == 2
             and result.get('official_replay_passes') == 2 and result.get('comparisons') == 1
             and result.get('promotion') is False and _check(ctx, True), 'Toy runner did not return')
    driver, stage = ctx['modules']['driver'], ctx['stage']
    intermediate = driver._read(stage, stage / _RESULT, _MAX_SMALL)
    record = dict(schema='task-pair-toy-smoke-runner-return-v1', phase='runner_returned',
                  intermediate_sha256=_sha(intermediate), observed_outer_exit=False,
                  registered_game_limit=2, promotion=False)
    data = _json_bytes(record)
    _ensure_room(ctx, len(data))
    from learning.task_anchor_once_v1.storage import owned_writes
    with owned_writes(stage, _MAX_SMALL, allowed_paths=[_RETURNED]):
        _write_exact(stage / _RETURNED, data)
    _require(driver._read(stage, stage / _RETURNED, _MAX_SMALL) == data,
             'Toy return marker differs')


SNAPSHOT_STUB = '''import pathlib, re, subprocess, sys
repo=pathlib.Path(sys.argv.pop(1)); commit=sys.argv.pop(1)
if re.fullmatch(r"[0-9a-f]{40}",commit) is None: raise SystemExit(1)
def snapshot(rel):
 path=repo.joinpath(*rel.split("/"))
 with path.open("rb") as stream: data=stream.read(4194305)
 if not data or len(data)>4194304: raise SystemExit(1)
 size=subprocess.run(["git","-C",str(repo),"cat-file","-s",commit+":"+rel],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True,timeout=30).stdout
 if len(size)>32 or not size.strip().isdigit() or int(size)!=len(data): raise SystemExit(1)
 found=subprocess.run(["git","-C",str(repo),"show",commit+":"+rel],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True,timeout=30)
 if data!=found.stdout: raise SystemExit(1)
 return path,data
path,entry=snapshot("kaggriculture/learning/task_pair_toy_smoke_v1/entry.py")
_,identity=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/identity.py")
_,bootstrap=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py")
_,auth=snapshot("kaggriculture/learning/task_pair_toy_smoke_v1/toy_auth.py")
exec(compile(entry,str(path),"exec",dont_inherit=True),{"__name__":"__main__","__file__":str(path),"_AUTH_IDENTITY_BYTES":identity,"_AUTH_BOOTSTRAP_BYTES":bootstrap,"_AUTH_TOY_AUTH_BYTES":auth})
'''


def _command(game_root, commit, game):
    _require(game in _ALLOWED and type(commit) is str and _COMMIT.fullmatch(commit),
             'Invalid toy child command')
    return [sys.executable, '-I', '-B', '-c', SNAPSHOT_STUB, str(PUBLIC_ROOT), commit,
            '--game-root', str(game_root), '--registration-commit', commit, '--game', game]


def execute(*, game_root, registration_commit, game=None):
    _require(game is None or (type(game) is str and game in _ALLOWED), 'Unknown toy game')
    ctx = _build_context(game_root, registration_commit)
    if game is not None:
        _require(_git_check(ctx) and _attempt_check(ctx), 'Toy registration changed')
        _child_ready(ctx, game)
        selected = [plan for plan in ctx['plans'] if plan['game'] == game]
        _require(len(selected) == 1, 'Unknown toy plan')
        _consume_game(ctx, game)
        check = lambda: _check(ctx, True)
        ctx['modules']['child'].run_one(manifest=ctx['manifest'], worker_source=ctx['source'],
            game_root=ctx['game_root'], stage=ctx['stage'], plan=selected[0], check=check)
        return None
    _claim_attempt(ctx)
    prepared = [False]
    check = lambda: _check(ctx, prepared[0])
    driver = ctx['modules']['driver'].LocalDriver(manifest=ctx['manifest'], game_root=ctx['game_root'],
        stage=ctx['stage'], commit=registration_commit,
        command_for=lambda alias: _command(ctx['game_root'], registration_commit, alias), check=check)
    original_launch = driver.launch
    def launch(plan, limits):
        _authorize_game(ctx, plan)
        return original_launch(plan, limits)
    driver.launch = launch
    def prepare(manifest, local_driver, allowance):
        result = _prepare(ctx, manifest, local_driver, allowance)
        prepared[0] = True
        return result
    result = ctx['runner'].run(manifest=ctx['manifest'], plans=ctx['plans'], limits=ctx['limits'],
        driver=driver, check=check, prepare_inputs=prepare, save_result=lambda payload: _save(ctx, payload))
    _mark_returned(ctx, result)
    # This process has not exited yet. Only the outer observer can attest its
    # natural zero exit; do not turn the inner runner result into final success.
    return dict(scope='synthetic_toy', runner_completed=True, observed_outer_exit=False,
                games=2, official_replay_passes=2, comparisons=1,
                registered_game_limit=2, promotion=False)


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid toy entry arguments')


def main(argv=None):
    parser = _Parser()
    parser.add_argument('--game-root', type=Path, required=True)
    parser.add_argument('--registration-commit', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--game')
    args = parser.parse_args(argv)
    result = execute(game_root=args.game_root, registration_commit=args.registration_commit, game=args.game)
    if result is not None:
        print(json.dumps(result, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        raise SystemExit('Toy smoke entry stopped; inspect its local stage') from None
