"""One registered, serial comparison on two already consumed seed roles.

No new training or seed generation. Official Worker, replay and per-game
acceptance are reused. Original games, private identities and receipts stay
local; this entry never retries an attempted game or resumes an old campaign.
"""
import argparse
import hashlib
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


PUBLIC_ROOT = Path(__file__).resolve().parents[3]
GAME_SOURCE = PUBLIC_ROOT / 'kaggriculture'
PACKAGE = 'task_pair_historical_compare_v1'
REL = 'kaggriculture/learning/' + PACKAGE + '/'
MANIFEST = 'comparison-manifest.local.json'
MAX_SMALL = 16384
GAMES = tuple(f'dev-{seed}-market-{side}-{policy}'
              for seed in ('S02', 'S03') for side in (0, 1)
              for policy in ('baseline', 'control'))
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _module(name, raw, path):
    _require(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024, 'Authenticated source missing')
    result = ModuleType(name)
    result.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), result.__dict__)
    return result


def _snapshots():
    campaign = 'kaggriculture/learning/task_pair_dev_campaign_v1/'
    once = 'kaggriculture/learning/task_control_once_v1/'
    return (
        _module('_historical_identity', globals().get('_AUTH_IDENTITY_BYTES'), PUBLIC_ROOT / campaign / 'identity.py'),
        _module('_historical_bootstrap', globals().get('_AUTH_BOOTSTRAP_BYTES'), PUBLIC_ROOT / campaign / 'bootstrap.py'),
        _module('_historical_once_auth', globals().get('_AUTH_ONCE_AUTH_BYTES'), PUBLIC_ROOT / once / 'auth.py'),
        _module('_historical_once_entry', globals().get('_AUTH_ONCE_ENTRY_BYTES'), PUBLIC_ROOT / once / 'entry.py'),
        _module('_historical_auth', globals().get('_AUTH_COMPARISON_BYTES'), PUBLIC_ROOT / REL / 'auth.py'),
    )


def _check_package(root, reg):
    private, public = root / 'learning', GAME_SOURCE / 'learning'
    suffixes = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    _require(not os.path.lexists(private / PACKAGE), 'Comparison package shadows public source')
    for base in (private, public):
        _require(not any(os.path.lexists(base / (PACKAGE + suffix)) for suffix in suffixes),
                 'Comparison package shadow')
    folder = public / PACKAGE
    _require(folder.is_dir() and folder.resolve(strict=True) == folder
             and not os.path.lexists(folder / '__pycache__'), 'Comparison package path differs')
    for name in ('__init__', 'auth', 'bindings', 'entry', 'projection', 'request_prefix', 'runner'):
        source = folder / (name + '.py')
        with source.open('rb') as stream:
            data = stream.read(4 * 1024 * 1024 + 1)
        _require(0 < len(data) <= 4 * 1024 * 1024 and _sha(data) == reg['public_sources'][REL + name + '.py'],
                 'Comparison module source differs')
        _require(not os.path.lexists(folder / name)
                 and not any(os.path.lexists(folder / (name + suffix)) for suffix in suffixes if suffix != '.py'),
                 'Comparison module shadow')
    prefix = 'learning.' + PACKAGE
    _require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules),
             'Comparison package imported too early')


def _load(name):
    module = importlib.import_module('learning.' + PACKAGE + '.' + name)
    expected = GAME_SOURCE / 'learning' / PACKAGE / (name + '.py')
    _require(Path(module.__file__).resolve(strict=True) == expected, 'Comparison import origin differs')
    return module


def _context(game_root, commit):
    _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__, 'Isolated Python required')
    _require(type(commit) is str and _COMMIT.fullmatch(commit), 'Invalid registration')
    identity, bootstrap, once_auth, once_entry, auth = _snapshots()
    root = bootstrap._absolute(game_root)
    expected = PUBLIC_ROOT.parent.parent / 'kaggriculture'
    _require(expected.resolve(strict=True) == expected and root == expected, 'Game root differs')
    reg, old_reg, freeze = auth.verify_public(identity, once_auth, repo=PUBLIC_ROOT,
                                               game_root=root, registration_commit=commit)
    fingerprint = once_entry._fingerprint(reg, old_reg, freeze)
    once_entry._check_package(root, reg)
    _check_package(root, reg)
    modules = bootstrap._runtime(root, reg)
    bindings, runner, projection, request_prefix = (
        _load(name) for name in ('bindings', 'runner', 'projection', 'request_prefix'))
    _require(once_entry._fingerprint(reg, old_reg, freeze) == fingerprint, 'Public metadata changed')
    limits = runner.limits_snapshot(reg.get('limits'))
    reg_raw = identity._read(PUBLIC_ROOT / auth.REGISTRATION)
    stage = root / 'artifacts' / auth.STAGE_NAME
    _require(stage.parent.resolve(strict=True) == stage.parent and stage.resolve(strict=False) == stage,
             'Comparison stage path differs')
    attempt_raw = _json(dict(schema='task-pair-historical-attempt-v1', registration_commit=commit,
                             max_games=8, retry_allowed=False))

    def public_check():
        return (identity._git(PUBLIC_ROOT, 'rev-parse', 'HEAD').decode().strip() == commit
                and not identity._git(PUBLIC_ROOT, 'status', '--porcelain', '--untracked-files=all')
                and identity._read(PUBLIC_ROOT / auth.REGISTRATION) == reg_raw
                and once_entry._fingerprint(reg, old_reg, freeze) == fingerprint
                and once_auth._typed_equal(limits, runner.limits_snapshot(reg['limits'])))
    return dict(identity=identity, bootstrap=bootstrap, once_auth=once_auth, once_entry=once_entry,
                auth=auth, modules=modules, bindings=bindings, runner=runner, projection=projection,
                request_prefix=request_prefix,
                root=root, stage=stage, commit=commit, reg=reg, old_reg=old_reg, freeze=freeze,
                limits=limits, public_check=public_check, attempt_raw=attempt_raw,
                attempt_path=stage.with_name(stage.name + '.attempt.local.json'),
                failure_path=stage.with_name(stage.name + '.failure.local.json'))


def _attempt_check(ctx):
    return ctx['bootstrap']._read(ctx['attempt_path'], MAX_SMALL) == ctx['attempt_raw']


def _not_failed(ctx):
    return (not os.path.lexists(ctx['failure_path'])
            and not os.path.lexists(ctx['stage'] / 'campaign-failure.local.json'))


def _claim(ctx):
    _require(ctx['public_check']() is True and not os.path.lexists(ctx['stage'])
             and not os.path.lexists(ctx['attempt_path']) and not os.path.lexists(ctx['failure_path']),
             'Comparison attempt already exists')
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


def _failure(ctx):
    data = _json(dict(schema='task-pair-historical-attempt-failure-v1', registration_commit=ctx['commit'],
                      stage_created=ctx['stage'].is_dir(), accepted_games=None, retry_allowed=False))
    _require(shutil.disk_usage(ctx['stage'].parent).free >= ctx['limits']['disk_free_floor_bytes'] + len(data),
             'Failure receipt resource limit')
    with ctx['failure_path'].open('xb') as stream:
        _require(stream.write(data) == len(data), 'Failure receipt write failed')
        stream.flush()
        os.fsync(stream.fileno())
    _require(ctx['bootstrap']._read(ctx['failure_path'], MAX_SMALL) == data, 'Failure receipt differs')


def _bind(ctx):
    _require(ctx['time_check']() is True and _not_failed(ctx)
             and ctx['public_check']() is True and _attempt_check(ctx), 'Comparison attempt is not bound')
    bootstrap, identity, once = ctx['bootstrap'], ctx['identity'], ctx['once_entry']
    root, reg, old_reg, freeze = ctx['root'], ctx['reg'], ctx['old_reg'], ctx['freeze']
    _require(ctx['time_check']() is True, 'Entry time limit')
    status_path, current, old_dev, prior = bootstrap._private_status()
    _require(type(current) is dict and ctx['once_auth']._typed_equal(old_dev, current.get('current_pair_dev_diagnostic'))
             and old_dev.get('state') == 'failed_no_retry', 'Previous closed status differs')
    _require(ctx['time_check']() is True, 'Entry time limit')
    status_raw = bootstrap._read(status_path)
    binding_path = PUBLIC_ROOT.parent / 'private' / bootstrap.PREBIND
    _require(ctx['time_check']() is True, 'Entry time limit')
    binding_raw = bootstrap._read(binding_path, 65536)
    once._verify_private_snapshot(bootstrap, ctx['once_auth'], current, prior, status_raw, binding_raw,
                                  old_dev.get('prebinding_record_sha256'))
    seed_path = root / bootstrap.OLD_PANEL / 'private-seeds.local.json'
    _require(ctx['time_check']() is True, 'Entry time limit')
    seed_raw = bootstrap._read(seed_path, 16384)
    fingerprint = once._fingerprint(reg, old_reg, freeze, prior)

    def precheck():
        if not (ctx['time_check']() is True and _not_failed(ctx) and ctx['public_check']() is True
                and _attempt_check(ctx) and ctx['time_check']() is True
                and bootstrap._read(status_path) == status_raw and ctx['time_check']() is True
                and bootstrap._read(binding_path, 65536) == binding_raw and ctx['time_check']() is True
                and bootstrap._read(seed_path, 16384) == seed_raw
                and once._fingerprint(reg, old_reg, freeze, prior) == fingerprint):
            return False
        return ctx['time_check']() is True

    manifest, previous, roles = ctx['bindings'].build(
        modules=ctx['modules'], bootstrap=bootstrap, identity=identity, reg=reg, old_reg=old_reg,
        freeze=freeze, game_root=root, repo=PUBLIC_ROOT, status_path=status_path, prior_binding=prior, check=precheck)
    _require(precheck() is True and ctx['bindings'].verify_mapping(manifest, previous, roles) is True,
             'Historical binding changed')
    indexed = {plan['game']: plan for plan in manifest.panel.plans()}
    _require(all(game in indexed for game in GAMES), 'Comparison plans missing')
    source = identity._read(root / 'learning/task_anchor_once_v1/games.py')
    _require(_sha(source) == manifest.panel.worker_sha, 'Worker identity differs')
    document = once._manifest_document(manifest, ctx['commit'], _sha(binding_raw))
    document['schema'] = 'task-pair-historical-private-manifest-v1'
    document['actual_seed_roles'] = {'S02': 'S01', 'S03': 'S04'}
    ctx.update(manifest=manifest, previous=previous, roles=roles, source=source,
               plans=[indexed[game] for game in GAMES], manifest_raw=_json(document), precheck=precheck)


def _check(ctx, prepared):
    if not (_not_failed(ctx) and ctx['precheck']() is True and _attempt_check(ctx)
            and ctx['once_auth']._typed_equal(ctx['limits'], ctx['runner'].limits_snapshot(ctx['reg']['limits']))):
        return False
    ctx['bootstrap']._verify_assets(ctx['manifest'], ctx['root'], ctx['modules']['history'])
    if ctx['bindings'].verify_mapping(ctx['manifest'], ctx['previous'], ctx['roles']) is not True:
        return False
    if prepared:
        driver = ctx['modules']['driver']
        if driver._read(ctx['stage'], ctx['stage'] / MANIFEST, MAX_SMALL) != ctx['manifest_raw']:
            return False
        for role in ('market', 'round2'):
            if driver._hash(ctx['stage'], ctx['stage'] / 'sources' / (role + '.py'), driver.MAX_SOURCE) != ctx['manifest'].opponent(role)[1]:
                return False
    return ctx['time_check']() is True


def _write_stage(ctx, relative, value):
    ctx['once_entry']._write_stage(ctx, relative, _json(value))


def _marker(ctx, game, kind):
    _require(game in GAMES and kind in ('permit', 'started', 'consumed'), 'Invalid game marker')
    name = 'job-results/' + game + '.' + kind + '.local.json'
    data = dict(schema='task-pair-historical-' + kind + '-v1', game=game, registration_commit=ctx['commit'],
                attempt_sha256=_sha(ctx['attempt_raw']), retry_allowed=False)
    if kind != 'started':
        data['manifest_sha256'] = _sha(ctx['manifest_raw'])
    return name, data


def _authorize(ctx, plan):
    game = plan['game']
    _require(game in GAMES and plan in ctx['plans'] and _check(ctx, True), 'Game is not authorized')
    for kind in ('permit', 'started', 'consumed'):
        name, _ = _marker(ctx, game, kind)
        _require(not os.path.lexists(ctx['stage'] / name), 'Game authorization already exists')
    name, data = _marker(ctx, game, 'permit')
    _write_stage(ctx, name, data)
    return True


def _start_child(ctx, game):
    _require(game in GAMES and ctx['public_check']() is True and _not_failed(ctx) and _attempt_check(ctx)
             and ctx['stage'].is_dir(), 'Child authorization missing')
    permit = 'job-results/' + game + '.permit.local.json'
    started = 'job-results/' + game + '.started.local.json'
    consumed = 'job-results/' + game + '.consumed.local.json'
    _require(os.path.lexists(ctx['stage'] / permit) and not os.path.lexists(ctx['stage'] / started)
             and not os.path.lexists(ctx['stage'] / consumed), 'Child authorization already used')
    # No private identity is read before this exclusive child-start marker.
    data = dict(schema='task-pair-historical-started-v1', game=game, registration_commit=ctx['commit'],
                attempt_sha256=_sha(ctx['attempt_raw']), retry_allowed=False)
    _write_stage(ctx, started, data)


def _run_child(ctx, game):
    _require(_check(ctx, True), 'Child context differs')
    driver = ctx['modules']['driver']
    for kind in ('permit', 'started'):
        name, data = _marker(ctx, game, kind)
        _require(driver._read(ctx['stage'], ctx['stage'] / name, MAX_SMALL) == _json(data),
                 'Child marker differs')
    name, data = _marker(ctx, game, 'consumed')
    _require(not os.path.lexists(ctx['stage'] / name), 'Child authorization already consumed')
    _write_stage(ctx, name, data)
    plans = [plan for plan in ctx['plans'] if plan['game'] == game]
    _require(len(plans) == 1, 'Child plan differs')
    ctx['modules']['child'].run_one(manifest=ctx['manifest'], worker_source=ctx['source'],
        game_root=ctx['root'], stage=ctx['stage'], plan=plans[0], check=lambda: _check(ctx, True))


def _save_results(ctx, driver, payload):
    """Recheck eight local receipt chains and persist the bounded projection.

    This checks saved receipts and runs the existing arithmetic projection; it
    does not re-execute the official games or provide external attestation.
    """
    _require(type(payload) is dict and payload.get('attempted') == 8
             and payload.get('official_replay_passes') == 8
             and payload.get('training') is False and payload.get('promotion') is False
             and payload.get('independent_improvement') is False
             and type(payload.get('rows')) is list and len(payload['rows']) == 8
             and type(payload.get('comparisons')) is list and len(payload['comparisons']) == 4,
             'Private result scope differs')
    stage, dmod = ctx['stage'], ctx['modules']['driver']
    acceptance = ctx['modules']['acceptance']
    ledger, rows = [], []
    for plan, prior in zip(ctx['plans'], payload['rows']):
        _require(_check(ctx, True), 'Comparison context changed during result read')
        game = plan['game']
        bundle = driver.read_bundle(plan)
        supervision_path = stage / 'job-results' / (game + '.supervision.local.json')
        acceptance_path = stage / 'job-results' / (game + '.acceptance.local.json')
        input_path = stage / 'job-inputs' / (game + '.json')
        supervision_raw = dmod._read(stage, supervision_path, acceptance.MAX_RECEIPT_BYTES)
        acceptance_raw = dmod._read(stage, acceptance_path, acceptance.MAX_RECEIPT_BYTES)
        supervision = acceptance._json(supervision_raw, acceptance.MAX_RECEIPT_BYTES)
        accepted = acceptance._json(acceptance_raw, acceptance.MAX_RECEIPT_BYTES)
        _require(supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
                 and supervision.get('game') == game and supervision.get('retry_allowed') is False
                 and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
                 and accepted.get('game') == game and accepted.get('passed') is True
                 and accepted.get('retry_allowed') is False
                 and accepted.get('supervision_sha256') == _sha(supervision_raw)
                 and accepted.get('job_input_sha256') == dmod._hash(stage, input_path,
                     ctx['modules']['child'].worker.MAX_JOB_BYTES)
                 and accepted.get('receipt_sha256') == {name: _sha(raw) for name, raw in bundle.items()},
                 'Accepted receipt chain differs')
        row = acceptance.accept(ctx['manifest'].panel, plan, bundle, supervision.get('supervision'),
                                driver.verify_files)
        _require(ctx['once_auth']._typed_equal(row, prior), 'Saved metrics differ')
        rows.append(row)
        markers = {}
        for kind in ('permit', 'started', 'consumed'):
            name, expected = _marker(ctx, game, kind)
            raw = dmod._read(stage, stage / name, MAX_SMALL)
            _require(raw == _json(expected), 'Game authorization receipt differs')
            markers[kind] = _sha(raw)
        ledger.append(dict(game=game, supervision_sha256=_sha(supervision_raw),
                           acceptance_sha256=_sha(acceptance_raw), marker_sha256=markers))
    receipts, comparison_ledger, request_prefixes = [], [], {}
    indexed = {row['game']: row for row in rows}
    for item in payload['comparisons']:
        _require(type(item) is dict and item.get('seed_alias') in ('S02', 'S03')
                 and type(item.get('side')) is int and item['side'] in (0, 1),
                 'Comparison coordinate differs')
        game = f"dev-{item['seed_alias']}-market-{item['side']}-baseline"
        path = dmod._comparison_file(stage, game, 'control')
        raw = dmod._read(stage, path, acceptance.MAX_RECEIPT_BYTES)
        receipt = acceptance._json(raw, acceptance.MAX_RECEIPT_BYTES)
        _require(ctx['once_auth']._typed_equal(receipt.get('comparison'), item),
                 'Saved comparison differs')
        receipts.append(receipt)
        comparison_ledger.append(dict(reference_game=game, sha256=_sha(raw)))
        candidate_game = f"dev-{item['seed_alias']}-market-{item['side']}-control"
        _require(game in indexed and candidate_game in indexed and game not in request_prefixes
                 and ctx['time_check']() is True
                 and driver.preflight_comparison(ctx['limits'], 1.0) is True,
                 'Request comparison resources or identity differ')
        reference = indexed[game]
        candidate = indexed[candidate_game]
        left = dmod._replay(stage, stage / 'games' / game / 'replay.json.gz', reference['replay_sha256'])
        right = dmod._replay(stage, stage / 'games' / candidate_game / 'replay.json.gz',
                             candidate['replay_sha256'])
        request_prefixes[game] = ctx['request_prefix'].compare(left, right, side=item['side'])
        del left, right
        _require(ctx['time_check']() is True, 'Request comparison time limit')
    seeds = dict(zip(('S02', 'S03'), ctx['manifest'].panel.seeds))
    projected = ctx['projection'].project(payload['paired'], seeds=seeds, rows=rows, comparisons=receipts)
    for pair in projected['pairs']:
        internal = {'S01': 'S02', 'S04': 'S03'}[pair['seed_alias']]
        game = f"dev-{internal}-market-{pair['side']}-baseline"
        _require(game in request_prefixes, 'Request comparison missing')
        pair['complete_request_prefix'] = request_prefixes[game]
    private = dict(schema='task-pair-historical-result-v1', passed=True, games=8,
                   official_replay_passes=8, comparisons=4, seeds=seeds, ledger=ledger,
                   comparison_receipts=comparison_ledger, request_prefixes=request_prefixes,
                   paired=payload['paired'],
                   training=False, independent_improvement=False, promotion=False)
    combined = len(_json(private)) + len(_json(projected))
    used, free = dmod._owned_size(stage), shutil.disk_usage(stage).free
    _require(type(used) is int and type(free) is int and combined <= dmod.MAX_SUMMARY
             and used + combined <= ctx['limits']['total_output_bytes']
             and free >= ctx['limits']['disk_free_floor_bytes'] + combined,
             'Result reserve exceeded')
    dmod._write_new(stage, stage / 'comparison-result.local.json', private)
    dmod._write_new(stage, stage / 'projected-summary.local.json', projected)
    return True


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
 path,data=snapshot("kaggriculture/learning/task_pair_historical_compare_v1/entry.py")
 _,identity=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/identity.py")
 _,bootstrap=snapshot("kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py")
 _,once_auth=snapshot("kaggriculture/learning/task_control_once_v1/auth.py")
 _,once_entry=snapshot("kaggriculture/learning/task_control_once_v1/entry.py")
 _,auth=snapshot("kaggriculture/learning/task_pair_historical_compare_v1/auth.py")
 exec(compile(data,str(path),"exec",dont_inherit=True),{"__name__":"__main__","__file__":str(path),"_AUTH_IDENTITY_BYTES":identity,"_AUTH_BOOTSTRAP_BYTES":bootstrap,"_AUTH_ONCE_AUTH_BYTES":once_auth,"_AUTH_ONCE_ENTRY_BYTES":once_entry,"_AUTH_COMPARISON_BYTES":auth,"_AUTH_ENTRY_STARTED":started})
try: start()
except Exception: raise SystemExit("Historical comparison snapshot stopped.") from None
'''


def _command(root, commit, game):
    _require(game in GAMES and type(commit) is str and _COMMIT.fullmatch(commit), 'Invalid child command')
    return [sys.executable, '-I', '-B', '-c', SNAPSHOT_STUB, str(PUBLIC_ROOT), commit,
            '--game-root', str(root), '--registration-commit', commit, '--game', game]


def execute(*, game_root, registration_commit, game=None, clock=time.monotonic):
    started = globals().get('_AUTH_ENTRY_STARTED')
    if started is None:
        started = clock()
    _require(game is None or game in GAMES, 'Unexpected comparison game')
    ctx = _context(game_root, registration_commit)
    once = ctx['once_entry']
    sampled = once._starting_clock(started, clock)
    sampled()
    def time_check():
        once._check_entry_time(started, ctx['limits']['campaign_wall_seconds'], sampled)
        return True
    ctx['time_check'] = time_check
    if game is not None:
        _start_child(ctx, game)
        _require(time_check() is True, 'Entry time limit')
        _bind(ctx)
        _run_child(ctx, game)
        return None
    ctx['_attempt_created'] = False
    ctx['_attempt_open_failed'] = False
    try:
        _claim(ctx)
        _require(time_check() is True, 'Entry time limit')
        _bind(ctx)
        prepared = [False]
        check = lambda: _check(ctx, prepared[0])
        driver = ctx['modules']['driver'].LocalDriver(manifest=ctx['manifest'], game_root=ctx['root'],
            stage=ctx['stage'], commit=registration_commit,
            command_for=lambda name: _command(ctx['root'], registration_commit, name), check=check)
        original_claim = driver.claim
        def claim():
            _require(original_claim() is True, 'Comparison stage claim failed')
            once._write_stage(ctx, MANIFEST, ctx['manifest_raw'])
            prepared[0] = True
            _require(check() is True, 'Prepared comparison stage differs')
            return True
        driver.claim = claim

        def save(payload):
            return _save_results(ctx, driver, payload)

        result = ctx['runner'].Coordinator().run(manifest=ctx['manifest'], plans=ctx['plans'],
            limits=ctx['limits'], driver=driver, check=check, authorize_game=lambda plan: _authorize(ctx, plan),
            save_result=save, clock=once._coordinator_clock(started, sampled))
        _require(type(result) is dict and result.get('games') == 8 and result.get('official_replay_passes') == 8
                 and result.get('promotion') is False and check() is True, 'Coordinator result differs')
        time_check()
        _write_stage(ctx, 'comparison-returned.local.json', dict(schema='task-pair-historical-returned-v1',
            runner_returned=True, accepted=8, official_replay_passes=8, observed_outer_exit=False))
        _require(check() is True and time_check() is True, 'Comparison context changed after return')
    except BaseException:
        if ctx['_attempt_created'] or ctx['_attempt_open_failed']:
            try:
                _failure(ctx)
            except Exception:
                pass
        raise
    return dict(scope='historical_development', games=8, official_replay_passes=8,
                observed_outer_exit=False, independent_improvement=False, promotion=False)


def public_preflight(*, game_root, registration_commit):
    """Authenticate committed public inputs only, without claiming or binding."""
    ctx = _context(game_root, registration_commit)
    _require(ctx['public_check']() is True and not os.path.lexists(ctx['attempt_path'])
             and not os.path.lexists(ctx['failure_path']) and not os.path.lexists(ctx['stage']),
             'Public preflight is not ready')
    return True


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid comparison arguments')


def main(argv=None):
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--game')
        parser.add_argument('--public-preflight', action='store_true')
        args = parser.parse_args(argv)
        if args.public_preflight:
            _require(args.game is None, 'Preflight cannot start a game')
            public_preflight(game_root=args.game_root, registration_commit=args.registration_commit)
        else:
            execute(game_root=args.game_root, registration_commit=args.registration_commit, game=args.game)
    except SystemExit:
        raise SystemExit('Historical comparison stopped without a normal return.') from None
    except Exception:
        raise SystemExit('Historical comparison stopped; inspect local evidence.') from None


if __name__ == '__main__':
    main()
