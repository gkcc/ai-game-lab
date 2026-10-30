"""Registered, supervised one-game control-head development entry.

This is not an evaluation panel or a retry entry. Public source is checked
before a fixed attempt marker; private historical bindings are checked after
the marker. Game inputs, raw receipts, trace and replay remain local.
"""
import argparse
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import ModuleType


PUBLIC_ROOT = Path(__file__).resolve().parents[3]
GAME_SOURCE = PUBLIC_ROOT / 'kaggriculture'
PACKAGE = 'task_control_once_v1'
REL = 'kaggriculture/learning/' + PACKAGE + '/'
GAME = 'dev-S02-market-0-control'
MANIFEST = 'control-manifest.local.json'
MAX_SMALL = 16384
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _module(name, data, path):
    _require(type(data) is bytes and 0 < len(data) <= 4 * 1024 * 1024, 'Authenticated snapshot missing')
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def _snapshots():
    campaign = 'kaggriculture/learning/task_pair_dev_campaign_v1/'
    identity = _module('_control_identity_snapshot', globals().get('_AUTH_IDENTITY_BYTES'),
                       PUBLIC_ROOT / campaign / 'identity.py')
    bootstrap = _module('_control_bootstrap_snapshot', globals().get('_AUTH_BOOTSTRAP_BYTES'),
                        PUBLIC_ROOT / campaign / 'bootstrap.py')
    auth = _module('_control_auth_snapshot', globals().get('_AUTH_CONTROL_BYTES'),
                   PUBLIC_ROOT / REL / 'auth.py')
    return identity, bootstrap, auth


def _fingerprint(*values):
    data = json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).digest()


def _verify_private_snapshot(bootstrap, auth, current, prior, status_raw, binding_raw, expected_binding_sha):
    _require(auth._typed_equal(bootstrap._json(status_raw), current)
             and auth._typed_equal(bootstrap._json(binding_raw), prior)
             and type(expected_binding_sha) is str and _sha(binding_raw) == expected_binding_sha,
             'Private identity changed while binding')


def _check_package(root, reg):
    private, public = root / 'learning', GAME_SOURCE / 'learning'
    suffixes = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    _require(not os.path.lexists(private / PACKAGE), 'Control package shadows the registered source')
    for base in (private, public):
        _require(not any(os.path.lexists(base / (PACKAGE + suffix)) for suffix in suffixes),
                 'Control package shadow')
    folder = public / PACKAGE
    _require(folder.is_dir() and folder.resolve(strict=True) == folder
             and not os.path.lexists(folder / '__pycache__'), 'Control package path differs')
    for name in ('__init__', 'auth', 'bindings', 'budget_observer', 'child_scope', 'runner', 'entry'):
        source = folder / (name + '.py')
        with source.open('rb') as stream:
            data = stream.read(4 * 1024 * 1024 + 1)
        _require(0 < len(data) <= 4 * 1024 * 1024 and _sha(data) == reg['public_sources'][REL + name + '.py'],
                 'Control module source differs')
        _require(not os.path.lexists(folder / name)
                 and not any(os.path.lexists(folder / (name + suffix)) for suffix in suffixes if suffix != '.py'),
                 'Control module shadow')
    prefix = 'learning.' + PACKAGE
    _require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules),
             'Control package imported before authentication')


def _load(name):
    module = importlib.import_module('learning.' + PACKAGE + '.' + name)
    expected = GAME_SOURCE / 'learning' / PACKAGE / (name + '.py')
    _require(Path(module.__file__).resolve(strict=True) == expected, 'Control import origin differs')
    return module


def _json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _manifest_document(value, commit, prior_binding_sha):
    panel = value.panel
    return dict(schema='task-control-once-private-manifest-v1', registration_commit=commit,
                prior_binding_sha256=prior_binding_sha,
                seeds={'S02': panel.seeds[0], 'S03': panel.seeds[1]},
                models={role: dict(path=panel.models[i][0], sha256=panel.models[i][1])
                        for i, role in enumerate(('baseline', 'control', 'auxiliary'))},
                crop=dict(path=panel.crop[0], sha256=panel.crop[1]),
                opponents={role: dict(path=value.opponent(role)[0], sha256=value.opponent(role)[1])
                           for role in ('market', 'round2')}, worker_sha256=panel.worker_sha)


def _build_context(game_root, registration_commit):
    """Authenticate public inputs only; no private role or model is read."""
    _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__, 'Isolated Python required')
    _require(type(registration_commit) is str and _COMMIT.fullmatch(registration_commit), 'Invalid registration')
    identity, bootstrap, auth = _snapshots()
    root = bootstrap._absolute(game_root)
    expected_root = PUBLIC_ROOT.parent.parent / 'kaggriculture'
    _require(expected_root.resolve(strict=True) == expected_root and root == expected_root, 'Game root differs')
    reg, old_reg, freeze = auth.verify_public(identity, repo=PUBLIC_ROOT, game_root=root,
                                               registration_commit=registration_commit)
    public_fingerprint = _fingerprint(reg, old_reg, freeze)
    _check_package(root, reg)
    modules = bootstrap._runtime(root, reg)
    bindings, runner = _load('bindings'), _load('runner')
    _require(_fingerprint(reg, old_reg, freeze) == public_fingerprint, 'Public metadata changed')
    # Validate all published resource limits before opening private status,
    # historical seed records or model identities.
    limits = runner.limits_snapshot(reg.get('limits'))
    reg_raw = identity._read(PUBLIC_ROOT / auth.REGISTRATION)
    stage = root / 'artifacts' / auth.STAGE_NAME
    _require(stage.parent.resolve(strict=True) == stage.parent and stage.resolve(strict=False) == stage,
             'Control stage path differs')
    attempt = _json(dict(schema='task-control-once-attempt-v1', registration_commit=registration_commit,
                         max_games=1, retry_allowed=False))

    def public_check():
        if (identity._git(PUBLIC_ROOT, 'rev-parse', 'HEAD').decode().strip() != registration_commit
                or identity._git(PUBLIC_ROOT, 'status', '--porcelain', '--untracked-files=all')
                or identity._read(PUBLIC_ROOT / auth.REGISTRATION) != reg_raw
                or _fingerprint(reg, old_reg, freeze) != public_fingerprint
                or not auth._typed_equal(limits, runner.limits_snapshot(reg['limits']))):
            return False
        return True
    return dict(identity=identity, bootstrap=bootstrap, auth=auth, bindings=bindings, runner=runner,
                modules=modules, root=root, stage=stage, commit=registration_commit, reg=reg,
                old_reg=old_reg, freeze=freeze, limits=limits, public_check=public_check,
                attempt_path=stage.with_name(stage.name + '.attempt.local.json'),
                failure_path=stage.with_name(stage.name + '.failure.local.json'), attempt_raw=attempt)


def _bind_private(ctx):
    """Bind prior private identities only after this new attempt is claimed."""
    _require(ctx['time_check']() is True and _not_failed(ctx)
             and ctx['public_check']() is True and _attempt_check(ctx),
             'Control attempt is not bound')
    identity, bootstrap, bindings = ctx['identity'], ctx['bootstrap'], ctx['bindings']
    reg, old_reg, freeze, root = ctx['reg'], ctx['old_reg'], ctx['freeze'], ctx['root']
    _require(ctx['time_check']() is True, 'Control entry time limit')
    status_path, current, old_dev, prior_binding = bootstrap._private_status()
    _require(type(current) is dict and ctx['auth']._typed_equal(old_dev, current.get('current_pair_dev_diagnostic'))
             and old_dev.get('state') == 'failed_no_retry', 'Prior closed attempt status differs')
    _require(ctx['time_check']() is True, 'Control entry time limit')
    status_raw = bootstrap._read(status_path)
    binding_path = PUBLIC_ROOT.parent / 'private' / bootstrap.PREBIND
    _require(ctx['time_check']() is True, 'Control entry time limit')
    binding_raw = bootstrap._read(binding_path, 65536)
    _verify_private_snapshot(bootstrap, ctx['auth'], current, prior_binding, status_raw, binding_raw,
                             old_dev.get('prebinding_record_sha256'))
    old_seed_path = root / bootstrap.OLD_PANEL / 'private-seeds.local.json'
    _require(ctx['time_check']() is True, 'Control entry time limit')
    old_seed_raw = bootstrap._read(old_seed_path, 16384)
    initial_fingerprint = _fingerprint(reg, old_reg, freeze, prior_binding)
    _require(bindings.context_identity(reg, old_reg, freeze, prior_binding) == initial_fingerprint,
             'Authenticated metadata changed')

    def precheck():
        if (ctx['time_check']() is not True or not _not_failed(ctx)
                or ctx['public_check']() is not True or not _attempt_check(ctx)
                or bootstrap._read(status_path) != status_raw
                or bootstrap._read(binding_path, 65536) != binding_raw
                or bootstrap._read(old_seed_path, 16384) != old_seed_raw
                or bindings.context_identity(reg, old_reg, freeze, prior_binding) != initial_fingerprint):
            return False
        return ctx['time_check']() is True

    manifest, previous, seed_s01, teacher_relative = bindings.build(modules=ctx['modules'], bootstrap=bootstrap,
        identity=identity, reg=reg, old_reg=old_reg, freeze=freeze, game_root=root, repo=PUBLIC_ROOT,
        status_path=status_path, prior_binding=prior_binding, check=precheck)
    _require(precheck() is True, 'Registration changed during binding')
    plans = [plan for plan in manifest.panel.plans() if plan['game'] == GAME]
    _require(len(plans) == 1, 'Control plan differs')
    source = identity._read(root / 'learning/task_anchor_once_v1/games.py')
    _require(_sha(source) == manifest.panel.worker_sha, 'Worker identity differs')
    document = _manifest_document(manifest, ctx['commit'], _sha(binding_raw))
    ctx.update(manifest=manifest, previous=previous, seed=seed_s01, teacher_relative=teacher_relative,
               manifest_raw=_json(document), source=source, plan=plans[0], precheck=precheck)


def _attempt_check(ctx):
    return ctx['bootstrap']._read(ctx['attempt_path'], MAX_SMALL) == ctx['attempt_raw']


def _not_failed(ctx):
    return (not os.path.lexists(ctx['failure_path'])
            and not os.path.lexists(ctx['stage'] / 'campaign-failure.local.json'))


def _check(ctx, prepared):
    if not (_not_failed(ctx) and ctx['precheck']() is True and _attempt_check(ctx)
            and ctx['auth']._typed_equal(ctx['limits'], ctx['runner'].limits_snapshot(ctx['reg']['limits']))):
        return False
    ctx['bootstrap']._verify_assets(ctx['manifest'], ctx['root'], ctx['modules']['history'])
    ctx['bindings'].verify_mapping(ctx['manifest'], ctx['previous'], ctx['seed'], ctx['teacher_relative'])
    if prepared:
        driver = ctx['modules']['driver']
        if driver._read(ctx['stage'], ctx['stage'] / MANIFEST, MAX_SMALL) != ctx['manifest_raw']:
            return False
        for role in ('market', 'round2'):
            if driver._hash(ctx['stage'], ctx['stage'] / 'sources' / (role + '.py'), driver.MAX_SOURCE) != ctx['manifest'].opponent(role)[1]:
                return False
    return ctx['time_check']() is True


def _write_exact(path, data):
    with path.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('Incomplete bounded receipt')


def _claim_attempt(ctx):
    _require(ctx['public_check']() is True and not os.path.lexists(ctx['stage'])
             and not os.path.lexists(ctx['attempt_path']) and not os.path.lexists(ctx['failure_path']),
             'Control attempt already exists')
    try:
        with ctx['attempt_path'].open('xb') as stream:
            ctx['_attempt_created'] = True
            _require(stream.write(ctx['attempt_raw']) == len(ctx['attempt_raw']), 'Attempt marker write failed')
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise
    except OSError:
        if not ctx['_attempt_created'] and not os.path.lexists(ctx['attempt_path']):
            ctx['_attempt_open_failed'] = True
        raise
    _require(_attempt_check(ctx), 'Attempt marker readback differs')


def _attempt_failure(ctx):
    """Best-effort parent receipt; do not guess whether a game was accepted."""
    data = _json(_failure_record(ctx))
    _require(0 < len(data) <= MAX_SMALL
             and shutil.disk_usage(ctx['stage'].parent).free >= ctx['limits']['disk_free_floor_bytes'] + len(data),
             'Failure receipt resource limit')
    with ctx['failure_path'].open('xb') as stream:
        _require(stream.write(data) == len(data), 'Failure receipt write failed')
        stream.flush()
        os.fsync(stream.fileno())
    _require(ctx['bootstrap']._read(ctx['failure_path'], MAX_SMALL) == data, 'Failure receipt readback differs')


def _failure_record(ctx):
    return dict(schema='task-control-once-attempt-failure-v1', registration_commit=ctx['commit'],
                expected_attempt_sha256=_sha(ctx['attempt_raw']), stage_created=ctx['stage'].is_dir(),
                accepted_games=None, retry_allowed=False)


def _write_stage(ctx, relative, data):
    _require(type(data) is bytes and 0 < len(data) <= MAX_SMALL, 'Invalid receipt size')
    driver, stage, limits = ctx['modules']['driver'], ctx['stage'], ctx['limits']
    used, free = driver._owned_size(stage), shutil.disk_usage(stage).free
    _require(type(used) is int and type(free) is int and used + len(data) <= limits['total_output_bytes']
             and free >= limits['disk_free_floor_bytes'] + len(data), 'Receipt resource limit')
    path = stage.joinpath(*relative.split('/'))
    from learning.task_anchor_once_v1.storage import owned_writes
    with owned_writes(stage, MAX_SMALL, allowed_paths=[relative]):
        _write_exact(path, data)
    _require(driver._read(stage, path, MAX_SMALL) == data, 'Receipt readback differs')


def _marker_name(kind):
    _require(kind in ('permit', 'started', 'consumed'), 'Unknown game marker')
    return 'job-results/' + GAME + '.' + kind + '.local.json'


def _game_marker(ctx, kind):
    name = _marker_name(kind)
    raw = _json(dict(schema='task-control-once-game-' + kind + '-v1', game=GAME,
                     registration_commit=ctx['commit'], attempt_sha256=_sha(ctx['attempt_raw']),
                     manifest_sha256=_sha(ctx['manifest_raw']), retry_allowed=False))
    return name, raw


def _authorize(ctx, plan):
    _require(plan == ctx['plan'] and _check(ctx, True), 'Control game is not authorized')
    permit, raw = _game_marker(ctx, 'permit')
    consumed, _ = _game_marker(ctx, 'consumed')
    started = _marker_name('started')
    _require(not os.path.lexists(ctx['stage'] / permit) and not os.path.lexists(ctx['stage'] / consumed)
             and not os.path.lexists(ctx['stage'] / started),
             'Control game authorization already exists')
    _write_stage(ctx, permit, raw)


def _start_child(ctx):
    """Claim the one child start before any private identity read."""
    permit, started, consumed = (_marker_name(kind) for kind in ('permit', 'started', 'consumed'))
    _require(ctx['public_check']() is True and _not_failed(ctx) and _attempt_check(ctx)
             and ctx['stage'].is_dir() and os.path.lexists(ctx['stage'] / permit)
             and not os.path.lexists(ctx['stage'] / started) and not os.path.lexists(ctx['stage'] / consumed),
             'Control child authorization missing')
    data = _json(dict(schema='task-control-once-child-started-v1', registration_commit=ctx['commit'],
                      attempt_sha256=_sha(ctx['attempt_raw']), retry_allowed=False))
    _write_stage(ctx, started, data)


def _consume(ctx):
    permit, raw = _game_marker(ctx, 'permit')
    consumed, data = _game_marker(ctx, 'consumed')
    started = _marker_name('started')
    expected_start = _json(dict(schema='task-control-once-child-started-v1', registration_commit=ctx['commit'],
                                attempt_sha256=_sha(ctx['attempt_raw']), retry_allowed=False))
    _require(_not_failed(ctx) and not os.path.lexists(ctx['stage'] / consumed)
             and ctx['modules']['driver']._read(ctx['stage'], ctx['stage'] / started, MAX_SMALL) == expected_start
             and ctx['modules']['driver']._read(ctx['stage'], ctx['stage'] / permit, MAX_SMALL) == raw,
             'Control game authorization missing')
    _write_stage(ctx, consumed, data)


def _diagnostic(ctx, result):
    _write_stage(ctx, 'job-results/' + GAME + '.diagnostic.local.json', _json(result))
    return True


def _run_child(ctx):
    _require(_check(ctx, True), 'Control child context differs')
    _consume(ctx)
    observer_module, scope = _load('budget_observer'), _load('child_scope')
    from learning.task_dagger_once_v1.policy import BoundedAgent
    from learning.task_request_gate_v1.policy import _Budget
    _require(BoundedAgent.__init__.__globals__.get('_Budget') is _Budget, 'Budget class origin differs')
    observer = observer_module.BudgetObserver(_Budget, BoundedAgent,
        expected_forward=_Budget.forward, expected_controller=_Budget.controller,
        expected_agent_init=BoundedAgent.__init__, expected_limits=ctx['manifest'].panel.worker_config()['game_limits'])
    child, worker = ctx['modules']['child'], ctx['modules']['child'].worker
    scope.run(worker=worker, observer=observer,
        child_call=lambda: child.run_one(manifest=ctx['manifest'], worker_source=ctx['source'],
            game_root=ctx['root'], stage=ctx['stage'], plan=ctx['plan'], check=lambda: _check(ctx, True)),
        save_observation=lambda value: _diagnostic(ctx, value), identity_attested=True, phase_attested=True)


SNAPSHOT_STUB = '''import pathlib, re, subprocess, sys, time
started=time.monotonic()
def start():
 repo=pathlib.Path(sys.argv.pop(1)); commit=sys.argv.pop(1)
 if re.fullmatch(r"[0-9a-f]{40}",commit) is None: raise ValueError("Invalid snapshot")
 def snapshot(rel):
  path=repo.joinpath(*rel.split("/"))
  with path.open("rb") as stream: data=stream.read(4194305)
  if not data or len(data)>4194304: raise ValueError("Invalid snapshot")
  result=subprocess.run(["git","-C",str(repo),"show",commit+":"+rel],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30)
  if data!=result.stdout: raise ValueError("Invalid snapshot")
  return path,data
 path,data=snapshot("kaggriculture/learning/task_control_once_v1/entry.py")
 _,identity=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/identity.py")
 _,bootstrap=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py")
 _,auth=snapshot("kaggriculture/learning/task_control_once_v1/auth.py")
 exec(compile(data,str(path),"exec",dont_inherit=True),{"__name__":"__main__","__file__":str(path),"_AUTH_IDENTITY_BYTES":identity,"_AUTH_BOOTSTRAP_BYTES":bootstrap,"_AUTH_CONTROL_BYTES":auth,"_AUTH_ENTRY_STARTED":started})
try: start()
except Exception: raise SystemExit("Control snapshot stopped.") from None
'''


def _command(root, commit, game):
    _require(game == GAME and type(commit) is str and _COMMIT.fullmatch(commit), 'Invalid child command')
    return [sys.executable, '-I', '-B', '-c', SNAPSHOT_STUB, str(PUBLIC_ROOT), commit,
            '--game-root', str(root), '--registration-commit', commit, '--game', game]


def _starting_clock(started, clock):
    first = [True]
    last = [None]
    def read():
        if first[0]:
            first[0] = False
            value = started
        else:
            value = clock()
        _require(type(value) in (int, float) and math.isfinite(value)
                 and (last[0] is None or value >= last[0]), 'Unreliable entry clock')
        last[0] = value
        return value
    return read


def _coordinator_clock(started, sampled_clock):
    """Anchor the coordinator to the original entry while sharing later samples."""
    first = [True]
    def read():
        if first[0]:
            first[0] = False
            return started
        return sampled_clock()
    return read


def _check_entry_time(started, seconds, clock):
    now = clock()
    _require(type(started) in (int, float) and math.isfinite(started)
             and type(now) in (int, float) and math.isfinite(now)
             and type(seconds) is int and seconds > 0 and started <= now < started + seconds,
             'Control entry time limit')


def execute(*, game_root, registration_commit, game=None, clock=time.monotonic):
    started = globals().get('_AUTH_ENTRY_STARTED')
    if started is None:
        started = clock()
    _require(game is None or game == GAME, 'Unexpected control game')
    ctx = _build_context(game_root, registration_commit)
    sampled_clock = _starting_clock(started, clock)
    sampled_clock()
    def time_check():
        _check_entry_time(started, ctx['limits']['campaign_wall_seconds'], sampled_clock)
        return True
    ctx['time_check'] = time_check
    if game is not None:
        _start_child(ctx)
        _require(time_check() is True, 'Control entry time limit')
        _bind_private(ctx)
        _run_child(ctx)
        return None
    ctx['_attempt_created'] = False
    ctx['_attempt_open_failed'] = False
    try:
        _claim_attempt(ctx)
        _require(time_check() is True, 'Control entry time limit')
        _bind_private(ctx)
        prepared = [False]
        check = lambda: _check(ctx, prepared[0])
        driver = ctx['modules']['driver'].LocalDriver(manifest=ctx['manifest'], game_root=ctx['root'],
            stage=ctx['stage'], commit=registration_commit,
            command_for=lambda alias: _command(ctx['root'], registration_commit, alias), check=check)
        original_claim, original_launch = driver.claim, driver.launch
        def claim():
            result = original_claim()
            _require(result is True, 'Control stage claim failed')
            _write_stage(ctx, MANIFEST, ctx['manifest_raw'])
            prepared[0] = True
            _require(check() is True, 'Prepared control stage differs')
            return True
        def launch(plan, limits):
            _authorize(ctx, plan)
            return original_launch(plan, limits)
        driver.claim, driver.launch = claim, launch
        def save(value):
            _write_stage(ctx, 'control-result.local.json', _json(value))
            return True
        result = ctx['runner'].Coordinator().run(manifest=ctx['manifest'], plan=ctx['plan'],
            limits=ctx['limits'], driver=driver, check=check, save_result=save,
            clock=_coordinator_clock(started, sampled_clock))
        _require(type(result) is dict and type(result.get('accepted')) is int and result['accepted'] == 1
                 and type(result.get('official_replay_passes')) is int and result['official_replay_passes'] == 1
                 and result.get('promotion') is False and result.get('observed_outer_exit') is False
                 and check() is True, 'Control coordinator result differs')
        _check_entry_time(started, ctx['limits']['campaign_wall_seconds'], sampled_clock)
        _write_stage(ctx, 'control-returned.local.json', _json(dict(schema='task-control-once-returned-v1',
            runner_returned=True, accepted=1, official_replay_passes=1, observed_outer_exit=False)))
        _require(check() is True, 'Control context changed after coordinator return')
        _check_entry_time(started, ctx['limits']['campaign_wall_seconds'], sampled_clock)
    except BaseException:
        if ctx['_attempt_created'] or ctx['_attempt_open_failed']:
            try:
                _attempt_failure(ctx)
            except Exception:
                pass
        raise
    return dict(scope='one_control_game', runner_completed=True, observed_outer_exit=False,
                games=result['accepted'], promotion=False, independent_improvement=False)


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid control entry arguments')


def main(argv=None):
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--game')
        args = parser.parse_args(argv)
        execute(game_root=args.game_root, registration_commit=args.registration_commit, game=args.game)
    except SystemExit:
        raise SystemExit('Control entry stopped without a normal return.') from None
    except Exception:
        raise SystemExit('Control entry stopped; inspect local evidence.') from None


if __name__ == '__main__':
    main()
