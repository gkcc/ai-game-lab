"""Four teacher-task games against the previously frozen market opponent.

The seeds have already been consumed. No training, holdout or promotion occurs.
Private identities, observations, native output and original replays stay local.
"""
import argparse
from contextlib import contextmanager
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
PRIVATE = REPO.parent / 'private' / 'task-teacher-market-dev-v1-20261031'
STAGE = 'task-teacher-market-dev-v1-20261031'
REGISTRATION = 'review/evidence/task-teacher-market-dev-v1-20261031/registration.json'
NOTE = 'review/evidence/task-teacher-market-dev-v1-20261031/REGISTRATION.zh-CN.md'
SCHEMA = 'task-teacher-market-dev-prereg-v1'
PREVIOUS_COMMIT = 'be65a2533e2373d0ebf54b1eedfd792a662b40ac'
MARKET_STAGE = 'task-pair-historical-compare-v1-consumed-market-20261031'
MARKET_LEDGER = 'review/evidence/task-pair-market-consumed-v1-result-20261031/ledger.json'
BASE_ENTRY = 'kaggriculture/learning/task_pair_market_consumed_v1/entry.py'
BIND_SOURCE = 'kaggriculture/learning/task_teacher_market_dev_v1/bind.py'
OWN_SOURCE = 'kaggriculture/learning/task_teacher_market_dev_v1/entry.py'
PANEL = dict(actual_seed_roles=['D01', 'D02'], internal_seed_roles=['S02', 'S03'],
             opponents=['market'], policies=['teacher_task'], sides=[0, 1],
             games=4, scope='consumed_development_only')
EVALUATION = dict(official_games=4, official_replays=4, comparisons=8,
                  new_seed_generation=False, training_updates=0, retries=False,
                  independent_evaluation=False, statistical_significance=False, promotion=False)
TEACHER_LIMITS = dict(max_teacher_games=4, max_official_replays=4,
                      max_prefix_comparisons=8, max_attempts=1)
ALIASES = {'S02': 'D01', 'S03': 'D02'}
GAMES = tuple(f'dev-{alias}-market-{side}-teacher_task' for alias in ALIASES for side in (0, 1))
PLAN_KEYS = frozenset(('game', 'seed_alias', 'seed', 'opponent', 'side', 'policy', 'split', 'parity_kind'))
JOB_KEYS = PLAN_KEYS | frozenset(('bot_sha256', 'rival_sha256', 'task_checkpoint_sha256'))
RECEIPT_SUFFIXES = {'row': '.row.json', 'audit': '.audit.json',
                    'complete': '.complete.json', 'storage': '.storage.json'}
MAX_SMALL = 512 * 1024


def require(value):
    if not value:
        raise ValueError('Teacher market diagnostic check failed')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def validate_teacher_limits(reg):
    value = reg.get('teacher_limits')
    require(type(value) is dict and set(value) == set(TEACHER_LIMITS)
            and all(type(value[key]) is int and value[key] == expected
                    for key, expected in TEACHER_LIMITS.items()))
    return dict(value)


def _base_snapshot(commit):
    """Execute the prior entry only after comparison with the registered blob."""
    require(type(commit) is str and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit))
    path = REPO / BASE_ENTRY
    raw = path.read_bytes()
    require(0 < len(raw) <= 4 * 1024**2)
    with open(os.devnull, 'rb') as null:
        proc = subprocess.run(['git', '-C', str(REPO), 'show', commit + ':' + BASE_ENTRY],
            stdin=null, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(proc.returncode == 0 and proc.stdout == raw and len(proc.stderr) <= MAX_SMALL)
    module = ModuleType('_teacher_market_prior_entry')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    module.REGISTRATION, module.NOTE, module.STAGE = REGISTRATION, NOTE, STAGE
    module.SCHEMA, module.PANEL, module.EVALUATION = SCHEMA, PANEL, EVALUATION
    return module


def _public_context(commit, *, child):
    base = _base_snapshot(commit)
    base.enforce_start_memory(child=child)
    reg, modules, prep, _projection, _runner, prefix, verify_again = base.public_context(commit)
    base.git('merge-base', '--is-ancestor', PREVIOUS_COMMIT, commit)
    require(reg.get('prior_market_result_commit') == PREVIOUS_COMMIT and validate_teacher_limits(reg))
    sources = reg['public_sources']
    for relative in (OWN_SOURCE, BIND_SOURCE, MARKET_LEDGER):
        raw = base.read(REPO / relative)
        require(sources.get(relative) == sha(raw) and raw == base.git('show', commit + ':' + relative))
    require(base.read(REPO / MARKET_LEDGER) == base.git('show', PREVIOUS_COMMIT + ':' + MARKET_LEDGER))
    binder = base._snapshot(BIND_SOURCE, commit, sources)
    return base, reg, modules, prep, prefix, verify_again, binder


def _binding(base, binder, prep, expected):
    require(type(expected) is str and base.HEX.fullmatch(expected))
    raw = base.read(PRIVATE / 'binding.local.json', 65536)
    doc = base.decode(raw)
    require(sha(raw) == expected and doc.get('schema') == 'task-teacher-market-dev-binding-v1'
            and binder.collect(doc['salt']) == doc)
    previous_raw, previous = base.binding_context(prep, doc['market_binding_sha256'])
    require(sha(previous_raw) == doc['market_binding_sha256'])
    return raw, doc, previous_raw, previous


def plans(panel):
    require(type(panel.seeds) is tuple and len(panel.seeds) == 2)
    result = [dict(game=f'dev-{alias}-market-{side}-teacher_task', seed_alias=alias,
                   seed=panel.seeds[index], opponent='market', side=side,
                   policy='teacher_task', split='dev', parity_kind=None)
              for index, alias in enumerate(ALIASES) for side in (0, 1)]
    require([item['game'] for item in result] == list(GAMES)
            and all(type(item['seed']) is int and 0 <= item['seed'] < 2**32 for item in result))
    return result


def validate_plan(plan, expected, *, job=False):
    require(type(plan) is dict and set(plan) == (JOB_KEYS if job else PLAN_KEYS)
            and type(expected) is dict and set(expected) == PLAN_KEYS
            and all(type(plan[k]) is type(expected[k]) and plan[k] == expected[k] for k in PLAN_KEYS)
            and plan['policy'] == 'teacher_task' and plan['opponent'] == 'market'
            and type(plan['side']) is int and plan['side'] in (0, 1))
    if job:
        require(plan['task_checkpoint_sha256'] is None
                and all(type(plan[k]) is str and len(plan[k]) == 64
                        and all(c in '0123456789abcdef' for c in plan[k])
                        for k in ('bot_sha256', 'rival_sha256')))
    return dict(plan)


def marker(kind, commit, binding_sha, game=None):
    require(kind in ('attempt', 'permit', 'started', 'consumed')
            and type(commit) is str and len(commit) == 40
            and type(binding_sha) is str and len(binding_sha) == 64
            and (game is None if kind == 'attempt' else game in GAMES))
    value = dict(schema='task-teacher-market-dev-marker-v1', kind=kind,
                 registration_commit=commit, binding_sha256=binding_sha, retry_allowed=False)
    if game is not None:
        value['game'] = game
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


@contextmanager
def bound_worker(worker, source, manifest, root, stage, plan, check):
    """Adapt one authenticated worker snapshot without changing official rules."""
    module = worker._load_snapshot(source, manifest.panel.worker_sha, root, check)
    cfg = manifest.panel.worker_config()
    expected_job = [None]
    original = module.verify, module.task_identity, module.read, module.stable, module.write_new
    job_path = stage / 'job-inputs' / (plan['game'] + '.json')
    storage_path = stage / 'job-results' / (plan['game'] + '.storage.json')

    def verify(directory):
        require(Path(directory).resolve(strict=False) == stage and check() is True)
        return manifest.panel.worker_config()

    def task_identity(directory, value, actual_cfg):
        require(Path(directory).resolve(strict=False) == stage and actual_cfg == cfg
                and type(value) is dict and PLAN_KEYS.issubset(value)
                and validate_plan({k: value[k] for k in PLAN_KEYS}, plan) == plan)
        result = original[1](directory, value, actual_cfg)
        require(result == (None, None))
        return result

    def checked_read(path):
        if Path(path).resolve(strict=False) == job_path:
            require(expected_job[0] is not None)
            value = worker._read_job(path)
            require(validate_plan(value, plan, job=True) == expected_job[0])
            return value
        return original[2](path)

    from learning.task_bc_error_diagnostics_v1.shadow import _typed
    def stable(value):
        return _typed(original[3](value))

    def checked_write(path, value):
        target = Path(path).resolve(strict=False)
        if target == storage_path:
            require(type(value) is dict and value.get('game') == plan['game']
                    and value.get('native_output_accounted_by') == 'bounded parent process log')
            value = dict(value, native_output_accounted_by=worker.NATIVE_OUTPUT_SCOPE)
        elif target.parent == storage_path.parent and target.name.endswith('.storage.json'):
            raise ValueError('Unexpected teacher storage receipt')
        return original[4](path, value)

    try:
        module.verify, module.task_identity, module.read, module.stable, module.write_new = (
            verify, task_identity, checked_read, stable, checked_write)
        yield module, cfg, expected_job
    finally:
        module.verify, module.task_identity, module.read, module.stable, module.write_new = original


def _num(value):
    require(type(value) in (int, float) and math.isfinite(value))
    return value


def verify_bundle(plan, job, bundle, supervision, *, crop_sha, engine_sha, storage_limit,
                  native_scope, verify_files, receipt_hashes):
    """Check real teacher receipts; this does not itself execute a replay."""
    require(validate_plan(job, plan, job=True) == job and type(bundle) is dict
            and set(bundle) == set(RECEIPT_SUFFIXES) and callable(verify_files))
    row, audit, complete, storage = (bundle[k] for k in ('row', 'audit', 'complete', 'storage'))
    require(all(type(value) is dict for value in (row, audit, complete, storage))
            and all(type(row.get(key)) is type(value) and row[key] == value for key, value in job.items())
            and row.get('passed') is True and type(row.get('states')) is int and row['states'] == 720
            and type(row.get('action_turns')) is int and row['action_turns'] == 719
            and row.get('status') == ['DONE', 'DONE'] and type(row.get('agent_errors')) is int
            and row['agent_errors'] == 0
            and row.get('crop_checkpoint_sha256') == crop_sha and row.get('engine_sha256') == engine_sha)
    for key in ('trace_sha256', 'replay_gzip_sha256', 'semantic_replay_sha256', 'semantic_replay_sha256_v2'):
        digest = row.get(key)
        require(type(digest) is str and len(digest) == 64
                and all(char in '0123456789abcdef' for char in digest))
    require(type(row.get('banks')) is list and len(row['banks']) == 2
            and type(row.get('initial_banks')) is list and len(row['initial_banks']) == 2
            and row.get('rewards') == row['banks'])
    banks, initial = ([_num(value) for value in row[key]] for key in ('banks', 'initial_banks'))
    side = plan['side']; other = 1 - side; margin = banks[side] - banks[other]
    require(_num(row['bank']) == banks[side] and _num(row['opponent_bank']) == banks[other]
            and _num(row['initial_bank']) == initial[side] and _num(row['opponent_initial_bank']) == initial[other]
            and _num(row['net_profit']) == banks[side] - initial[side]
            and _num(row['opponent_net_profit']) == banks[other] - initial[other]
            and _num(row['margin']) == margin
            and _num(row['score']) == (1.0 if margin > 0 else .5 if margin == 0 else 0.0))
    official = audit.get('official')
    counts = audit.get('primary_model_counts')
    require(audit.get('game') == plan['game'] and audit.get('passed') is True
            and type(audit.get('errors')) is int and audit['errors'] == 0
            and all(type(audit.get(k)) is int and audit[k] == 719 for k in
                    ('actor_actions_reproduced', 'rival_actions_reproduced', 'trace_turns_reproduced'))
            and audit.get('trace_sha256') == row.get('trace_sha256')
            and audit.get('replay_sha256') == row.get('replay_gzip_sha256')
            and type(official) is dict and official.get('passed') is True
            and type(official.get('official_transitions_reexecuted')) is int
            and official['official_transitions_reexecuted'] == 719
            and official.get('both_public_and_private_states_match_each_step') is True
            and official.get('inventory_order_checked_each_step') is True
            and official.get('engine_sha256') == engine_sha
            and official.get('semantic_sha256') == row.get('semantic_replay_sha256')
            and official.get('semantic_sha256_v2') == row.get('semantic_replay_sha256_v2')
            and official.get('final_rewards') == banks)
    require(type(counts) is dict and counts == audit.get('live_model_counts')
            and type(counts.get('task_forwards')) is int and counts['task_forwards'] == 0
            and type(counts.get('task_rows')) is int and counts['task_rows'] == 0
            and type(counts.get('crop_forwards')) is int and counts['crop_forwards'] > 0
            and type(counts.get('controller_calls')) is int and counts['controller_calls'] == 719)
    require(type(receipt_hashes) is dict and set(receipt_hashes) == set(RECEIPT_SUFFIXES)
            and complete.get('row_sha256') == receipt_hashes['row']
            and complete.get('certificate_sha256') == receipt_hashes['audit']
            and complete.get('game') == plan['game'] and complete.get('passed') is True
            and storage.get('game') == plan['game'] and storage.get('passed') is True
            and storage.get('error_kind') is None and storage.get('native_output_accounted_by') == native_scope)
    output = storage.get('output')
    require(type(output) is dict and type(output.get('limit_bytes')) is int and output['limit_bytes'] == storage_limit
            and type(output.get('used_bytes')) is int and 0 <= output['used_bytes'] <= storage_limit
            and type(output.get('remaining_bytes')) is int
            and output['remaining_bytes'] == storage_limit - output['used_bytes']
            and type(output.get('refused')) is int and output['refused'] == 0
            and verify_files(plan, row) is True and supervision is True)
    return row


def project(rows, old_ledger, prefixes, commit):
    """Publish only role aliases, cash metrics and the eight prefix outcomes."""
    require(type(rows) is list and len(rows) == 4 and type(old_ledger) is dict
            and type(old_ledger.get('pairs')) is list and len(old_ledger['pairs']) == 4
            and type(prefixes) is dict and len(prefixes) == 8)
    previous = {(x['seed_alias'], x['side']): x for x in old_ledger['pairs']}
    require(set(previous) == {(role, side) for role in ALIASES.values() for side in (0, 1)})
    records, seen = [], set()
    keys = ('own_net_profit', 'opponent_net_profit', 'margin', 'score')
    reasons = {'observed': {'actor_request_difference'}, 'no_difference': {'all_requests_equal'},
               'unknown': {'frame_difference', 'opponent_request_difference',
                           'simultaneous_request_difference', 'terminal_frame_difference'}}
    for row in rows:
        role, side = ALIASES[row['seed_alias']], row['side']
        require((role, side) not in seen and (role, side) in previous)
        seen.add((role, side))
        teacher = dict(own_net_profit=_num(row['net_profit']),
                       opponent_net_profit=_num(row['opponent_net_profit']),
                       margin=_num(row['margin']), score=_num(row['score']))
        for arm, field in (('baseline', 'reference'), ('control', 'candidate')):
            student = {key: _num(previous[(role, side)][field][key]) for key in keys}
            prefix = prefixes[(row['seed_alias'], side, arm)]
            require(type(prefix) is dict and set(prefix) == {'scope', 'status', 'reason', 'step'}
                    and prefix['scope'] == 'same_complete_history' and prefix['status'] in reasons
                    and prefix['reason'] in reasons[prefix['status']]
                    and ((prefix['status'] == 'observed' and type(prefix['step']) is int
                          and 0 <= prefix['step'] < 719)
                         or (prefix['status'] != 'observed' and prefix['step'] is None)))
            records.append(dict(seed_alias=role, side=side, arm=arm, teacher=teacher, student=student,
                                teacher_minus_student={key: teacher[key] - student[key] for key in keys},
                                complete_request_prefix=prefix))
    require(len(seen) == 4 and len(records) == 8)
    means = {arm: {key: sum(x['teacher_minus_student'][key] for x in records if x['arm'] == arm) / 4
                   for key in keys} for arm in ('baseline', 'control')}
    return dict(schema='task-teacher-market-dev-public-v1', registration_commit=commit,
                scope='consumed_development_only', teacher_games=4, previous_student_games=8,
                seed_clusters=2, teacher_official_replays=4, pairs=records, mean_teacher_minus_student=means,
                training=False, independent_improvement=False, statistical_significance=False, promotion=False)


def trace_summary(driver, stage, plan, row, crop_sha):
    """Consume the complete hashed trace; reveal only the two decision counts."""
    task_count = crop_count = 0
    for turn in driver._trace(stage, stage / 'traces' / (plan['game'] + '.jsonl.gz'), row['trace_sha256']):
        require('task_checkpoint_sha256' in turn and turn['task_checkpoint_sha256'] is None
                and turn.get('checkpoint_sha256') == crop_sha
                and type(turn.get('records')) is list and type(turn.get('crop_records')) is list)
        for record in turn['records']:
            require(type(record) is dict and record.get('source') == 'teacher'
                    and type(record.get('chosen')) is str and record['chosen'] == record.get('teacher_label'))
            task_count += 1
        for record in turn['crop_records']:
            require(type(record) is dict and record.get('source') == 'model')
            crop_count += 1
    require(task_count > 0 and crop_count > 0)
    return dict(turns=719, task_teacher_choices=task_count, crop_model_choices=crop_count)


def _load_bundle(driver, acceptance, stage, plan):
    raw = {name: driver._read(stage, stage / 'job-results' / (plan['game'] + suffix), MAX_SMALL)
           for name, suffix in RECEIPT_SUFFIXES.items()}
    values = {name: acceptance._json(value, MAX_SMALL) for name, value in raw.items()}
    return values, {name: sha(value) for name, value in raw.items()}


def _verify_game_files(modules, manifest, stage, plan, row):
    driver, panel = modules['driver'], manifest.panel
    worker = __import__('learning.task_pair_dev_games_v1.worker', fromlist=['worker'])
    capture = __import__('learning.task_pair_dev_games_v1.capture', fromlist=['capture'])
    job_path = stage / 'job-inputs' / (plan['game'] + '.json')
    job = modules['acceptance']._json(driver._read(stage, job_path, worker.MAX_JOB_BYTES), worker.MAX_JOB_BYTES)
    require(validate_plan(job, plan, job=True) == job
            and all(type(row.get(k)) is type(v) and row[k] == v for k, v in job.items()))
    source, source_sha = manifest.opponent('market')
    checks = ((GAME, GAME / panel.crop[0], driver.MAX_MODEL, panel.crop[1]),
              (GAME, GAME / source, driver.MAX_SOURCE, source_sha),
              (stage, stage / 'sources' / 'market.py', driver.MAX_SOURCE, source_sha),
              (GAME, GAME / worker.WORKER_RELATIVE, worker.MAX_WORKER_BYTES, panel.worker_sha),
              (stage, stage / 'agents' / (plan['game'] + '.py'), driver.MAX_SOURCE, job['bot_sha256']),
              (stage, stage / 'traces' / (plan['game'] + '.jsonl.gz'), capture.MAX_COMPRESSED_BYTES, row['trace_sha256']),
              (stage, stage / 'games' / plan['game'] / 'replay.json.gz', capture.MAX_COMPRESSED_BYTES, row['replay_gzip_sha256']))
    require(all(driver._hash(root, path, maximum) == digest for root, path, maximum, digest in checks)
            and job['rival_sha256'] == source_sha and job['task_checkpoint_sha256'] is None)
    return True


def _prior_rows(base, modules, manifest):
    """Recheck prior accepted game identities and cash against the public ledger."""
    driver, acceptance = modules['driver'], modules['acceptance']
    old_stage = GAME / 'artifacts' / MARKET_STAGE
    previous = base.decode(base.read(old_stage / 'comparison-result.local.json'))
    require(previous.get('binding_sha256') == sha(base.read(old_stage / 'manifest.local.json', 65536))
            and previous.get('games') == 8 and previous.get('official_replay_passes') == 8)
    ledger = base.decode(base.read(REPO / MARKET_LEDGER))
    pairs = {(p['seed_alias'], p['side']): p for p in ledger['pairs']}
    receipts = {item['game']: item for item in previous['game_receipts']}
    result = {}
    for plan in base.plans_for(manifest):
        name = plan['game']; item = receipts[name]
        accepted_raw = driver._read(old_stage, old_stage / 'job-results' / (name + '.acceptance.local.json'), MAX_SMALL)
        require(sha(accepted_raw) == item['acceptance_sha256'])
        accepted = acceptance._json(accepted_raw, MAX_SMALL)
        job_raw = driver._read(old_stage, old_stage / 'job-inputs' / (name + '.json'), 16384)
        row_raw = driver._read(old_stage, old_stage / 'job-results' / (name + '.row.json'), MAX_SMALL)
        require(accepted.get('passed') is True and accepted.get('game') == name
                and accepted.get('job_input_sha256') == sha(job_raw)
                and accepted.get('receipt_sha256', {}).get('row') == sha(row_raw))
        job = acceptance._json(job_raw, 16384)
        row = acceptance._json(row_raw, MAX_SMALL)
        require(manifest.panel.validate_plan(job, job=True) == job
                and {k: job[k] for k in PLAN_KEYS} == plan
                and all(type(row.get(k)) is type(v) and row[k] == v for k, v in job.items())
                and row.get('passed') is True and row.get('states') == 720 and row.get('action_turns') == 719
                and row.get('status') == ['DONE', 'DONE'] and row.get('agent_errors') == 0
                and row.get('crop_checkpoint_sha256') == manifest.panel.crop[1]
                and row.get('engine_sha256') == manifest.panel.worker_config()['engine_sha256'])
        role, side, arm = ALIASES[plan['seed_alias']], plan['side'], plan['policy']
        economic = pairs[(role, side)]['reference' if arm == 'baseline' else 'candidate']
        for old_key, public_key in (('net_profit', 'own_net_profit'), ('opponent_net_profit', 'opponent_net_profit'),
                                    ('margin', 'margin'), ('score', 'score')):
            require(_num(row[old_key]) == _num(economic[public_key]))
        result[(plan['seed_alias'], side, arm)] = row
    require(len(result) == 8)
    return ledger, result


def _context(commit, binding_sha, *, child):
    base, reg, modules, prep, prefix, verify_again, binder = _public_context(commit, child=child)
    raw, document, market_raw, previous = _binding(base, binder, prep, binding_sha)
    manifest = modules['manifest'].build_manifest(seeds=previous['seeds'], models=previous['models'],
        crop=previous['crop'], opponents=previous['opponents'], worker_sha256=previous['worker_sha256'])
    require(base.validate_assets(manifest, modules['driver']) and len(plans(manifest.panel)) == 4)
    return base, reg, modules, prefix, verify_again, binder, raw, document, market_raw, previous, manifest


def preflight(commit, binding_sha):
    """Read-only preregistration, identity and host resource check."""
    base, reg, modules, _prefix, verify_again, _binder, _raw, _doc, _market, _previous, manifest = (
        _context(commit, binding_sha, child=False))
    import psutil
    stage = GAME / 'artifacts' / STAGE
    modules['driver']._no_links(GAME, stage)
    require(not stage.exists() and not stage.is_symlink() and not (PRIVATE / 'attempt.local.json').exists()
            and psutil.virtual_memory().available >= reg['limits']['initial_available_bytes']
            and shutil.disk_usage(GAME / 'artifacts').free >=
                reg['limits']['disk_free_floor_bytes'] + reg['limits']['per_game_reserve_bytes']
            and verify_again())
    return dict(status='ready', scope='read_only_preflight', games_started=0, registered_games=4)


def _execute_child(*, base, modules, manifest, stage, plan, check):
    worker = __import__('learning.task_pair_dev_games_v1.worker', fromlist=['worker'])
    driver = modules['driver']
    source = driver._read(GAME, GAME / worker.WORKER_RELATIVE, worker.MAX_WORKER_BYTES)
    require(sha(source) == manifest.panel.worker_sha)
    worker._single_python_thread()
    with worker._quiet_output():
        with bound_worker(worker, source, manifest, GAME, stage, plan, check) as (module, cfg, expected_job):
            bot = module.wrapper(stage, plan, cfg)
            require(bot == stage / 'agents' / (plan['game'] + '.py') and check())
            bot_sha = driver._hash(stage, bot, driver.MAX_SOURCE)
            rival_sha = driver._hash(stage, stage / 'sources' / 'market.py', driver.MAX_SOURCE)
            require(rival_sha == manifest.opponent('market')[1])
            job = dict(plan, bot_sha256=bot_sha, rival_sha256=rival_sha, task_checkpoint_sha256=None)
            require(validate_plan(job, plan, job=True) == job)
            expected_job[0] = job
            driver._write_new(stage, stage / 'job-inputs' / (plan['game'] + '.json'), job, limit=worker.MAX_JOB_BYTES)
            module.job(stage, plan['game'])
            require(check())


def run(commit, binding_sha, game=None):
    started = time.monotonic()
    require(game is None or game in GAMES)
    base, reg, modules, prefix, verify_again, binder, raw, doc, market_raw, previous, manifest = (
        _context(commit, binding_sha, child=game is not None))
    import psutil
    driver_module, acceptance = modules['driver'], modules['acceptance']
    runner = __import__('learning.task_pair_historical_compare_v1.runner', fromlist=['runner'])
    worker = __import__('learning.task_pair_dev_games_v1.worker', fromlist=['worker'])
    limits = runner.limits_snapshot(reg['limits'])
    teacher_limits = validate_teacher_limits(reg)
    stage = GAME / 'artifacts' / STAGE
    attempt_path = PRIVATE / 'attempt.local.json'
    attempt = marker('attempt', commit, binding_sha)
    prepared = [game is not None]
    selected = plans(manifest.panel)
    require(len(selected) == teacher_limits['max_teacher_games'] == teacher_limits['max_official_replays'])
    default_before = driver_module._hash(GAME, GAME / 'main.py', driver_module.MAX_SOURCE)

    def remaining():
        delta = time.monotonic() - started
        require(math.isfinite(delta) and delta >= 0)
        return limits['campaign_wall_seconds'] - delta

    def check():
        if psutil.virtual_memory().available < limits['job']['min_available_bytes']:
            raise MemoryError('Runtime memory floor')
        if remaining() <= 0:
            raise TimeoutError('Campaign deadline')
        require(base.read(PRIVATE / 'binding.local.json', 65536) == raw
                and base.read(attempt_path, 65536) == attempt)
        if prepared[0]:
            require(base.read(stage / 'manifest.local.json', 65536) == raw)
        return True

    driver_module._no_links(GAME, stage)
    if game is not None:
        require(stage.is_dir() and base.read(attempt_path, 65536) == attempt)
        permit = stage / 'job-results' / (game + '.permit.local.json')
        require(base.read(permit, 65536) == marker('permit', commit, binding_sha, game))
        plan = next(p for p in selected if p['game'] == game)
        require(check() and base.validate_assets(manifest, driver_module))
        base._write(stage / 'job-results' / (game + '.started.local.json'), marker('started', commit, binding_sha, game))
        base._write(stage / 'job-results' / (game + '.consumed.local.json'), marker('consumed', commit, binding_sha, game))
        _execute_child(base=base, modules=modules, manifest=manifest, stage=stage, plan=plan, check=check)
        require(check() and verify_again() and base.validate_assets(manifest, driver_module)
                and driver_module._hash(GAME, GAME / 'main.py', driver_module.MAX_SOURCE) == default_before)
        return None

    require(psutil.virtual_memory().available >= limits['initial_available_bytes']
            and not stage.exists() and not stage.is_symlink() and not attempt_path.exists())
    base._write(attempt_path, attempt)
    driver = driver_module.LocalDriver(manifest=manifest, game_root=GAME, stage=stage, commit=commit,
        command_for=lambda name: [], check=check)
    require(driver.preflight(None, limits, remaining()) and driver.claim())
    base._write(stage / 'manifest.local.json', raw)
    prepared[0] = True
    accepted = []
    for plan in selected:
        require(len(accepted) < teacher_limits['max_teacher_games'] and check()
                and remaining() > limits['job']['wall_seconds'] + limits['job']['cleanup_seconds']
                and driver.preflight(plan, limits, remaining()) and base.validate_assets(manifest, driver_module))
        name = plan['game']
        base._write(stage / 'job-results' / (name + '.permit.local.json'), marker('permit', commit, binding_sha, name))
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--registration-commit', commit,
                   '--binding-sha256', binding_sha, '--game', name]
        supervision = modules['campaign'].supervise_windows(command, GAME, dict(limits['job']))
        supervision_record = dict(schema='task-teacher-market-dev-job-supervision-v1', game=name,
                                  supervision=supervision, retry_allowed=False)
        driver_module._write_new(stage, stage / 'job-results' / (name + '.supervision.local.json'),
                                 supervision_record, limit=MAX_SMALL)
        require(acceptance.supervision_ok(supervision))
        bundle, hashes = _load_bundle(driver_module, acceptance, stage, plan)
        job = acceptance._json(driver_module._read(stage, stage / 'job-inputs' / (name + '.json'), 16384), 16384)
        row = verify_bundle(plan, job, bundle, True, crop_sha=manifest.panel.crop[1],
            engine_sha=reg['engine_sha256'], storage_limit=manifest.panel.worker_config()['storage']['game_write_bytes'],
            native_scope=worker.NATIVE_OUTPUT_SCOPE, receipt_hashes=hashes,
            verify_files=lambda p, r: _verify_game_files(modules, manifest, stage, p, r))
        require(driver.preflight_comparison(limits, remaining()))
        trace = trace_summary(driver_module, stage, plan, row, manifest.panel.crop[1])
        for kind in ('permit', 'started', 'consumed'):
            require(base.read(stage / 'job-results' / (name + '.' + kind + '.local.json'), 65536)
                    == marker(kind, commit, binding_sha, name))
        driver_module._write_new(stage, stage / 'job-results' / (name + '.accepted.local.json'),
            dict(schema='task-teacher-market-dev-game-accepted-v1', game=name, receipt_sha256=hashes,
                 supervision_sha256=driver_module._hash(stage,
                    stage / 'job-results' / (name + '.supervision.local.json'), MAX_SMALL),
                 job_input_sha256=sha(driver_module._read(stage, stage / 'job-inputs' / (name + '.json'), 16384)),
                 trace_summary=trace, passed=True, retry_allowed=False), limit=MAX_SMALL)
        accepted.append(row)
        require(len(accepted) <= teacher_limits['max_teacher_games']
                and driver_module._owned_size(stage) <= limits['total_output_bytes'])
    require(len(accepted) == teacher_limits['max_official_replays'] and check())
    ledger, prior = _prior_rows(base, modules, manifest)
    prefixes = {}
    prefix_count = 0
    old_stage = GAME / 'artifacts' / MARKET_STAGE
    for row in accepted:
        for arm in ('baseline', 'control'):
            require(prefix_count < teacher_limits['max_prefix_comparisons']
                    and driver.preflight_comparison(limits, remaining()) and check())
            older = prior[(row['seed_alias'], row['side'], arm)]
            a = driver_module._replay(stage, stage / 'games' / row['game'] / 'replay.json.gz', row['replay_gzip_sha256'])
            b = driver_module._replay(old_stage, old_stage / 'games' / older['game'] / 'replay.json.gz', older['replay_gzip_sha256'])
            prefixes[(row['seed_alias'], row['side'], arm)] = prefix.compare(a, b, side=row['side'])
            prefix_count += 1
            del a, b
    require(prefix_count == teacher_limits['max_prefix_comparisons'])
    projected = project(accepted, ledger, prefixes, commit)
    driver_module._write_new(stage, stage / 'projected-summary.local.json', projected)
    require(driver_module._owned_size(stage) <= limits['total_output_bytes'] and check() and verify_again()
            and base.validate_assets(manifest, driver_module)
            and driver_module._hash(GAME, GAME / 'main.py', driver_module.MAX_SOURCE) == default_before
            and _binding(base, binder, __import__('learning.task_pair_market_consumed_v1.prepare', fromlist=['prepare']), binding_sha)[0] == raw)
    return dict(status='passed', games=4, official_replay_passes=4, comparisons=8,
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
        result = preflight(args.registration_commit, args.binding_sha256) if args.preflight else run(
            args.registration_commit, args.binding_sha256, args.game)
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
