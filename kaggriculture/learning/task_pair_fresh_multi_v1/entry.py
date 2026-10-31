"""Two frozen-model cohorts against two opponents on four fresh seed roles.

This is a one-attempt exploratory panel, not training or a promotion test.
Seed values, model identities, raw replays and diagnostics remain local.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
PRIVATE = REPO.parent / 'private' / 'task-pair-fresh-multi-v1-20261031'
REGISTRATION = 'review/evidence/task-pair-fresh-multi-v1-20261031/registration.json'
NOTE = 'review/evidence/task-pair-fresh-multi-v1-20261031/REGISTRATION.zh-CN.md'
SCHEMA = 'task-pair-fresh-multi-prereg-v1'
BASE_COMMIT = '8f6448e641db98b521dc5336d3b0b2d9217739d4'
BASE_ENTRY = 'kaggriculture/learning/task_pair_market_consumed_v1/entry.py'
PREFIX = 'kaggriculture/learning/task_pair_fresh_multi_v1/'
STAGE = 'task-pair-fresh-multi-v1-20261031'
STAGES = {'C01': 'task-pair-fresh-multi-c01-v1-20261031',
          'C02': 'task-pair-fresh-multi-c02-v1-20261031'}
COHORTS = {'C01': {'S02': 'F01', 'S03': 'F02'},
           'C02': {'S02': 'F03', 'S03': 'F04'}}
PANEL = dict(seed_roles=['F01', 'F02', 'F03', 'F04'], cohorts=COHORTS,
             opponents=['market', 'round2'], policies=['baseline', 'control'],
             sides=[0, 1], games=32, scope='fresh_seed_exploration')
EVALUATION = dict(official_games=32, official_replays=32, comparisons=16, prefix_comparisons=16,
                  new_seed_generation=True, training_updates=0, retries=False,
                  independent_evaluation=False, statistical_significance=False, promotion=False)
FRESH_LIMITS = dict(max_games=32, max_replays=32, max_comparisons=16, max_prefix_comparisons=16, max_attempts=1,
                    cohort_max_games=16, cohort_max_replays=16, cohort_max_comparisons=8,
                    cohort_output_bytes=3 * 1024**3, cohort_total_output_bytes=6 * 1024**3,
                    private_results_bytes=16 * 1024**2)
OPPONENTS = ('market', 'round2')
ARMS = ('baseline', 'control')
ALIASES = ('S02', 'S03')
GAMES = tuple(f'dev-{alias}-{opponent}-{side}-{arm}' for alias in ALIASES
              for opponent in OPPONENTS for side in (0, 1) for arm in ARMS)
MAX_SMALL = 512 * 1024


def require(value):
    if not value:
        raise ValueError('Fresh multi-opponent exploration check failed')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_limits(reg):
    value = reg.get('fresh_limits')
    require(type(value) is dict and set(value) == set(FRESH_LIMITS)
            and all(type(value[k]) is int and value[k] == v for k, v in FRESH_LIMITS.items()))
    return dict(value)


def validate_commitment(reg, binding_sha):
    require(type(binding_sha) is str and len(binding_sha) == 64
            and all(char in '0123456789abcdef' for char in binding_sha)
            and reg.get('binding_commitment_sha256') == binding_sha)
    return True


def shared_remaining(started, seconds, clock):
    """Keep one non-resettable monotonic budget across both cohorts."""
    require(type(started) in (int, float) and math.isfinite(started)
            and type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
            and callable(clock))
    previous = [started]

    def sample():
        current = clock()
        require(type(current) in (int, float) and math.isfinite(current) and current >= previous[0])
        previous[0] = current
        return seconds - (current - started)

    return sample


def cohort_limits(legacy, coordinator, registered):
    """Preserve old resource values, replacing only this stage's explicit scope."""
    require(type(legacy) is dict and validate_limits({'fresh_limits': registered}) == FRESH_LIMITS)
    value = {**legacy, 'job': dict(legacy['job']),
             'max_games': registered['cohort_max_games'],
             'max_replays': registered['cohort_max_replays'],
             'max_comparisons': registered['cohort_max_comparisons'],
             'max_attempts': registered['max_attempts'],
             'total_output_bytes': registered['cohort_output_bytes']}
    return coordinator.limits_snapshot(value)


def _base_snapshot(commit):
    require(type(commit) is str and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit))
    path = REPO / BASE_ENTRY
    raw = path.read_bytes()
    require(0 < len(raw) <= 4 * 1024**2)
    with open(os.devnull, 'rb') as null:
        proc = subprocess.run(['git', '-C', str(REPO), 'show', commit + ':' + BASE_ENTRY],
            stdin=null, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(proc.returncode == 0 and proc.stdout == raw and len(proc.stderr) <= MAX_SMALL)
    module = ModuleType('_fresh_multi_previous_entry')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    module.REGISTRATION, module.NOTE, module.STAGE = REGISTRATION, NOTE, STAGE
    module.SCHEMA, module.PANEL, module.EVALUATION = SCHEMA, PANEL, EVALUATION
    return module


def _context(commit, binding_sha, *, child):
    base = _base_snapshot(commit)
    base.enforce_start_memory(child=child)
    reg, modules, old_prepare, _projection, _runner, prefix, verify_again = base.public_context(commit)
    base.git('merge-base', '--is-ancestor', BASE_COMMIT, commit)
    require(reg.get('fresh_base_commit') == BASE_COMMIT
            and validate_commitment(reg, binding_sha) and validate_limits(reg))
    sources = reg['public_sources']
    needed = tuple(PREFIX + name for name in ('__init__.py', 'prepare.py', 'test_prepare.py',
                   'entry.py', 'test_entry.py', 'coordinator.py', 'test_coordinator.py',
                   'projection.py', 'test_projection.py'))
    for relative in needed:
        raw = base.read(REPO / relative)
        require(sources.get(relative) == sha(raw) and raw == base.git('show', commit + ':' + relative))
    fresh = base._snapshot(PREFIX + 'prepare.py', commit, sources)
    coordinator = base._snapshot(PREFIX + 'coordinator.py', commit, sources)
    projection = base._snapshot(PREFIX + 'projection.py', commit, sources)
    raw, document, inventory = fresh.verify_binding(binding_sha)
    market_raw, market = fresh.load_verified_market_binding()
    old_raw, old_doc = base.binding_context(old_prepare, document['market_binding_sha256'])
    require(market_raw == old_raw and base.typed_equal(market, old_doc)
            and document.get('base_commit') == BASE_COMMIT
            and document.get('cohorts') == COHORTS
            and sha(market_raw) == document['market_binding_sha256'])
    roles = document['seed_roles']
    require(type(roles) is dict and set(roles) == {'F01', 'F02', 'F03', 'F04'}
            and all(type(v) is int and 0 <= v < 2**32 for v in roles.values())
            and len(set(roles.values())) == 4)
    manifests = {}
    for cohort, aliases in COHORTS.items():
        seeds = {alias: roles[role] for alias, role in aliases.items()}
        manifests[cohort] = modules['manifest'].build_manifest(seeds=seeds, models=market['models'],
            crop=market['crop'], opponents=market['opponents'], worker_sha256=market['worker_sha256'])
        require(base.validate_assets(manifests[cohort], modules['driver'])
                and len(plans_for(manifests[cohort])) == 16)
    limits = cohort_limits(reg['limits'], coordinator, validate_limits(reg))
    return (base, reg, modules, prefix, verify_again, fresh, coordinator, projection,
            raw, document, inventory, market_raw, manifests, limits)


def plans_for(manifest):
    plans = [plan for plan in manifest.panel.plans() if plan['policy'] in ARMS]
    require(len(plans) == 16 and [plan['game'] for plan in plans] == list(GAMES)
            and all(manifest.panel.validate_plan(plan) == plan for plan in plans))
    return plans


def marker(kind, *, commit, binding_sha, cohort=None, game=None):
    require(kind in ('attempt', 'permit', 'started', 'consumed')
            and type(commit) is str and len(commit) == 40
            and type(binding_sha) is str and len(binding_sha) == 64)
    require((cohort is None and game is None) if kind == 'attempt'
            else (cohort in COHORTS and game in GAMES))
    value = dict(schema='task-pair-fresh-multi-marker-v1', kind=kind,
                 registration_commit=commit, binding_sha256=binding_sha, retry_allowed=False)
    if kind != 'attempt':
        value.update(cohort=cohort, game=game)
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def owner_marker(*, commit, binding_sha, pid, created_us, deadline):
    require(type(pid) is int and pid > 0 and type(created_us) is int and created_us > 0
            and type(deadline) in (int, float) and math.isfinite(deadline))
    value = dict(schema='task-pair-fresh-multi-owner-v1', registration_commit=commit,
                 binding_sha256=binding_sha, pid=pid, created_us=created_us, deadline=deadline)
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def verify_live_owner(owner, *, commit, binding_sha, current_pid, process, executable, samefile, now):
    """Require the active coordinator or its verified venv launcher as parent."""
    require(type(owner) is dict and set(owner) == {'schema', 'registration_commit', 'binding_sha256',
                                                   'pid', 'created_us', 'deadline'}
            and owner['schema'] == 'task-pair-fresh-multi-owner-v1'
            and owner['registration_commit'] == commit and owner['binding_sha256'] == binding_sha
            and type(owner['pid']) is int and owner['pid'] > 0
            and type(owner['created_us']) is int and owner['created_us'] > 0
            and type(owner['deadline']) in (int, float) and math.isfinite(owner['deadline'])
            and type(now) in (int, float) and math.isfinite(now) and now < owner['deadline']
            and type(current_pid) is int and current_pid > 0 and callable(process) and callable(samefile))
    parent = process(current_pid).parent()
    require(parent is not None and parent.is_running())
    if parent.pid != owner['pid']:
        require(samefile(parent.exe(), executable))
        parent = parent.parent()
        require(parent is not None and parent.is_running())
    require(parent.pid == owner['pid'] and int(parent.create_time() * 1_000_000) == owner['created_us'])
    return True


def verify_open_attempt(stages, *, exists=os.path.lexists):
    require(callable(exists) and not exists(PRIVATE / 'closed.local.json')
            and all(not exists(stage / 'campaign-failure.local.json') for stage in stages.values()))
    return True


def _finish_attempt(commit, binding_sha, outcome):
    """Close only an attempt owned by this coordinator; retain all evidence."""
    require(outcome in ('completed', 'failed'))
    attempt_path, owner_path = PRIVATE / 'attempt.local.json', PRIVATE / 'owner.local.json'
    if not os.path.lexists(attempt_path) or not os.path.lexists(owner_path):
        return False
    require(PRIVATE.resolve(strict=True) == PRIVATE and attempt_path.resolve(strict=True) == attempt_path
            and owner_path.resolve(strict=True) == owner_path and not attempt_path.is_symlink()
            and not owner_path.is_symlink()
            and attempt_path.read_bytes() == marker('attempt', commit=commit, binding_sha=binding_sha))
    owner = json.loads(owner_path.read_text(encoding='utf8'))
    import psutil
    current = psutil.Process(os.getpid())
    require(type(owner) is dict and owner.get('pid') == os.getpid()
            and owner.get('created_us') == int(current.create_time() * 1_000_000)
            and owner.get('registration_commit') == commit and owner.get('binding_sha256') == binding_sha)
    value = dict(schema='task-pair-fresh-multi-closed-v1', registration_commit=commit,
                 binding_sha256=binding_sha, outcome=outcome, retry_allowed=False)
    raw = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    target = PRIVATE / 'closed.local.json'
    if os.path.lexists(target):
        return False
    with target.open('xb') as stream:
        require(stream.write(raw) == len(raw))
        stream.flush()
        os.fsync(stream.fileno())
    require(target.read_bytes() == raw)
    return True


def verify_comparison_receipt(base, receipt, comparison, left, right):
    require(type(receipt) is dict and type(comparison) is dict
            and receipt.get('schema') == 'task-pair-dev-comparison-v1'
            and receipt.get('reference_game') == left['game']
            and receipt.get('candidate_game') == right['game']
            and receipt.get('reference_replay_sha256') == left['replay_sha256']
            and receipt.get('candidate_replay_sha256') == right['replay_sha256']
            and receipt.get('reference_trace_sha256') == left['trace_sha256']
            and receipt.get('candidate_trace_sha256') == right['trace_sha256']
            and base.typed_equal(receipt.get('comparison'), comparison))
    return True


def preflight(commit, binding_sha):
    """Authenticate frozen inputs and budgets without creating a match attempt."""
    context = _context(commit, binding_sha, child=False)
    base, _reg, modules, _prefix, verify_again, fresh, _coordinator, _projection, _raw, _doc, inventory, _market, _manifests, limits = context
    import psutil
    for name in STAGES.values():
        stage = GAME / 'artifacts' / name
        modules['driver']._no_links(GAME, stage)
        require(not stage.exists() and not stage.is_symlink())
    require(not (PRIVATE / 'attempt.local.json').exists() and not (PRIVATE / 'owner.local.json').exists()
            and verify_open_attempt({key: GAME / 'artifacts' / value for key, value in STAGES.items()})
            and psutil.virtual_memory().available >= limits['initial_available_bytes']
            and shutil.disk_usage(GAME / 'artifacts').free >=
                limits['disk_free_floor_bytes'] + FRESH_LIMITS['cohort_total_output_bytes']
                + FRESH_LIMITS['private_results_bytes']
            and fresh.verify_inputs(inventory) is True and verify_again())
    return dict(status='ready', scope='read_only_preflight', games_started=0, registered_games=32)


def _game_evidence(*, base, modules, manifest, driver, plan, row, cohort, commit, binding_sha):
    dm, acceptance = modules['driver'], modules['acceptance']
    name, stage = plan['game'], driver.stage
    bundle = driver.read_bundle(plan)
    supervision_raw = dm._read(stage, stage / 'job-results' / (name + '.supervision.local.json'), MAX_SMALL)
    accepted_raw = dm._read(stage, stage / 'job-results' / (name + '.acceptance.local.json'), MAX_SMALL)
    supervision = acceptance._json(supervision_raw, MAX_SMALL)
    accepted = acceptance._json(accepted_raw, MAX_SMALL)
    require(supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
            and supervision.get('game') == name and supervision.get('retry_allowed') is False
            and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
            and accepted.get('game') == name and accepted.get('passed') is True
            and accepted.get('retry_allowed') is False and accepted.get('supervision_sha256') == sha(supervision_raw)
            and accepted.get('job_input_sha256') == dm._hash(stage, stage / 'job-inputs' / (name + '.json'), 16384)
            and accepted.get('receipt_sha256') == {key: sha(value) for key, value in bundle.items()})
    actual = acceptance.accept(manifest.panel, plan, bundle, supervision.get('supervision'), driver.verify_files)
    require(base.typed_equal(actual, row))
    markers = {}
    for kind in ('permit', 'started', 'consumed'):
        raw = dm._read(stage, stage / 'job-results' / (name + '.' + kind + '.local.json'), 65536)
        require(raw == marker(kind, commit=commit, binding_sha=binding_sha, cohort=cohort, game=name))
        markers[kind] = sha(raw)
    return dict(cohort=cohort, game=name, supervision_sha256=sha(supervision_raw),
                acceptance_sha256=sha(accepted_raw), marker_sha256=markers)


def _run_impl(commit, binding_sha, cohort=None, game=None):
    started = time.monotonic()
    require((cohort is None and game is None) or (cohort in COHORTS and game in GAMES))
    (base, reg, modules, prefix, verify_again, fresh, coordinator, projection,
     raw, document, inventory, market_raw, manifests, limits) = _context(commit, binding_sha, child=game is not None)
    import psutil
    dm, acceptance = modules['driver'], modules['acceptance']
    stages = {key: GAME / 'artifacts' / value for key, value in STAGES.items()}
    attempt_path = PRIVATE / 'attempt.local.json'
    owner_path = PRIVATE / 'owner.local.json'
    attempt = marker('attempt', commit=commit, binding_sha=binding_sha)
    default_before = dm._hash(GAME, GAME / 'main.py', dm.MAX_SOURCE)
    prepared = {key: game is not None and key == cohort for key in COHORTS}
    if game is None:
        owner_process = psutil.Process(os.getpid())
        require(owner_process.is_running())
        owner_raw = owner_marker(commit=commit, binding_sha=binding_sha, pid=os.getpid(),
            created_us=int(owner_process.create_time() * 1_000_000),
            deadline=started + limits['campaign_wall_seconds'])
        owner = base.decode(owner_raw)
    else:
        owner_raw = base.read(owner_path, 65536)
        owner = base.decode(owner_raw)
        verify_live_owner(owner, commit=commit, binding_sha=binding_sha, current_pid=os.getpid(),
                          process=psutil.Process, executable=sys.executable,
                          samefile=os.path.samefile, now=time.monotonic())
    remaining = shared_remaining(started, owner['deadline'] - started, time.monotonic)

    def check():
        if psutil.virtual_memory().available < limits['job']['min_available_bytes']:
            raise MemoryError('Runtime memory floor')
        if remaining() <= 0:
            raise TimeoutError('Campaign deadline')
        require(base.read(PRIVATE / 'binding.local.json', 8 * 1024**2) == raw
                and base.read(attempt_path, 65536) == attempt
                and base.read(owner_path, 65536) == owner_raw and verify_open_attempt(stages))
        if game is not None:
            verify_live_owner(owner, commit=commit, binding_sha=binding_sha, current_pid=os.getpid(),
                              process=psutil.Process, executable=sys.executable,
                              samefile=os.path.samefile, now=time.monotonic())
        for key, is_prepared in prepared.items():
            if is_prepared:
                require(base.read(stages[key] / 'manifest.local.json', 8 * 1024**2) == raw)
        return True

    for stage in stages.values():
        dm._no_links(GAME, stage)
    if game is not None:
        stage, manifest = stages[cohort], manifests[cohort]
        require(verify_open_attempt(stages) and stage.is_dir() and base.read(attempt_path, 65536) == attempt
                and base.read(stage / 'job-results' / (game + '.permit.local.json'), 65536)
                == marker('permit', commit=commit, binding_sha=binding_sha, cohort=cohort, game=game))
        plan = next((p for p in plans_for(manifest) if p['game'] == game), None)
        require(plan is not None and check() and base.validate_assets(manifest, dm))
        for kind in ('started', 'consumed'):
            base._write(stage / 'job-results' / (game + '.' + kind + '.local.json'),
                        marker(kind, commit=commit, binding_sha=binding_sha, cohort=cohort, game=game))
        source = dm._read(GAME, GAME / 'learning/task_anchor_once_v1/games.py', 4 * 1024**2)
        require(sha(source) == manifest.panel.worker_sha)
        modules['child'].run_one(manifest=manifest, worker_source=source, game_root=GAME,
                                 stage=stage, plan=plan, check=check)
        require(check() and verify_again() and base.validate_assets(manifest, dm)
                and dm._hash(GAME, GAME / 'main.py', dm.MAX_SOURCE) == default_before)
        return None

    require(psutil.virtual_memory().available >= limits['initial_available_bytes']
            and all(not s.exists() and not s.is_symlink() for s in stages.values())
            and not attempt_path.exists() and not owner_path.exists() and verify_open_attempt(stages)
            and shutil.disk_usage(GAME / 'artifacts').free >=
                limits['disk_free_floor_bytes'] + FRESH_LIMITS['cohort_total_output_bytes']
                + FRESH_LIMITS['private_results_bytes']
            and fresh.verify_inputs(inventory) is True)
    base._write(attempt_path, attempt)
    base._write(owner_path, owner_raw)
    results, drivers, receipts = {}, {}, []
    for current in COHORTS:
        manifest, stage = manifests[current], stages[current]

        def command_for(name, current=current):
            require(name in GAMES)
            return [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                    '--registration-commit', commit, '--binding-sha256', binding_sha,
                    '--cohort', current, '--game', name]

        driver = dm.LocalDriver(manifest=manifest, game_root=GAME, stage=stage, commit=commit,
                                command_for=command_for, check=check)
        drivers[current] = driver

        def claim(current=current, driver=driver, stage=stage):
            require(driver.claim() is True)
            base._write(stage / 'manifest.local.json', raw)
            prepared[current] = True
            return check()

        def authorize(plan, current=current, stage=stage, manifest=manifest):
            require(plan in plans_for(manifest) and check())
            base._write(stage / 'job-results' / (plan['game'] + '.permit.local.json'),
                        marker('permit', commit=commit, binding_sha=binding_sha,
                               cohort=current, game=plan['game']))
            return True

        results[current] = coordinator.Coordinator().run(manifest=manifest, limits=limits, driver=driver,
            check=check, authorize_game=authorize, claim=claim, remaining=remaining)
        require(len(results[current]['rows']) == 16 and len(results[current]['comparisons']) == 8
                and check())

    prefixes, comparisons = {}, []
    for current in COHORTS:
        manifest, driver, stage = manifests[current], drivers[current], stages[current]
        rows = results[current]['rows']
        indexed = {(r['seed_alias'], r['opponent'], r['side'], r['arm']): r for r in rows}
        paired = {(v['seed_alias'], v['opponent'], v['side']): v for v in results[current]['comparisons']}
        require(len(indexed) == 16 and len(paired) == 8)
        for plan in plans_for(manifest):
            key = (plan['seed_alias'], plan['opponent'], plan['side'], plan['policy'])
            require(check() and driver.preflight_comparison(limits, remaining()))
            receipts.append(_game_evidence(base=base, modules=modules, manifest=manifest, driver=driver,
                plan=plan, row=indexed[key], cohort=current, commit=commit, binding_sha=binding_sha))
        for alias in ALIASES:
            for opponent in OPPONENTS:
                for side in (0, 1):
                    require(check() and driver.preflight_comparison(limits, remaining())
                            and len(comparisons) < FRESH_LIMITS['max_comparisons']
                            and len(prefixes) < FRESH_LIMITS['max_prefix_comparisons'])
                    left = indexed[(alias, opponent, side, 'baseline')]
                    right = indexed[(alias, opponent, side, 'control')]
                    path = dm._comparison_file(stage, left['game'], 'control')
                    data = dm._read(stage, path, MAX_SMALL)
                    receipt = acceptance._json(data, MAX_SMALL)
                    verify_comparison_receipt(base, receipt, paired[(alias, opponent, side)], left, right)
                    comparisons.append(dict(cohort=current, reference_game=left['game'], sha256=sha(data)))
                    a = dm._replay(stage, stage / 'games' / left['game'] / 'replay.json.gz', left['replay_sha256'])
                    b = dm._replay(stage, stage / 'games' / right['game'] / 'replay.json.gz', right['replay_sha256'])
                    prefixes[(current, alias, opponent, side)] = prefix.compare(a, b, side=side)
                    del a, b
                    require(check() and remaining() > 0)
    require(len(receipts) == FRESH_LIMITS['max_games']
            and len(comparisons) == FRESH_LIMITS['max_comparisons']
            and len(prefixes) == FRESH_LIMITS['max_prefix_comparisons'])
    seeds = {current: {alias: document['seed_roles'][role] for alias, role in mapping.items()}
             for current, mapping in COHORTS.items()}
    projected = projection.project(results, seed_roles=seeds, prefixes=prefixes)
    projected['registration_commit'] = commit
    dm._write_new(PRIVATE, PRIVATE / 'projected-summary.local.json', projected)
    public_raw = dm._read(PRIVATE, PRIVATE / 'projected-summary.local.json', 8 * 1024**2)
    private = dict(schema='task-pair-fresh-multi-private-result-v1', registration_commit=commit,
                   binding_sha256=binding_sha, games=32, official_replay_passes=32, comparisons=16,
                   prefix_comparisons=16,
                   receipts=receipts, comparison_receipts=comparisons, projection_sha256=sha(public_raw),
                   training=False, independent_improvement=False, promotion=False)
    dm._write_new(PRIVATE, PRIVATE / 'campaign-result.local.json', private)
    private_size = len(public_raw) + (PRIVATE / 'campaign-result.local.json').stat().st_size
    require(private_size <= FRESH_LIMITS['private_results_bytes']
            and sum(dm._owned_size(stage) for stage in stages.values()) <= FRESH_LIMITS['cohort_total_output_bytes']
            and check() and verify_again() and all(base.validate_assets(m, dm) for m in manifests.values())
            and dm._hash(GAME, GAME / 'main.py', dm.MAX_SOURCE) == default_before
            and fresh.verify_binding(binding_sha)[0] == raw
            and fresh.load_verified_market_binding()[0] == market_raw)
    return dict(status='passed', games=32, official_replay_passes=32, comparisons=16, prefix_comparisons=16,
                seed_clusters=4, training=False, independent_improvement=False, promotion=False)


def run(commit, binding_sha, cohort=None, game=None):
    try:
        result = _run_impl(commit, binding_sha, cohort, game)
    except BaseException:
        if cohort is None and game is None:
            _finish_attempt(commit, binding_sha, 'failed')
        raise
    if cohort is None and game is None:
        require(_finish_attempt(commit, binding_sha, 'completed'))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--cohort')
    parser.add_argument('--game')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        require(not (args.preflight and (args.game is not None or args.cohort is not None)))
        result = preflight(args.registration_commit, args.binding_sha256) if args.preflight else run(
            args.registration_commit, args.binding_sha256, args.cohort, args.game)
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
