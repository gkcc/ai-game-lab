"""Bounded posthoc analysis of one accepted pair from a stopped experiment.

Use the isolated committed-byte snapshot launcher, not a direct project import.
All original artifacts remain read-only. This entry runs no game, official
transition, model inference, GPU training or old campaign entry. The evidence
ledger and its independently retained SHA remain private; the ledger was
recorded after the old process exited and is not a third-party timestamp.
"""
import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
from types import ModuleType


PUBLIC_ROOT = Path(__file__).resolve().parents[3]
GAME_SOURCE = PUBLIC_ROOT / 'kaggriculture'
REL = 'kaggriculture/learning/task_pair_partial_diagnostic_v1/'
REGISTRATION = 'review/evidence/task-pair-partial-diagnostic-v1-20261031/registration.json'
ORIGINAL_REGISTRATION = 'review/evidence/task-pair-historical-compare-v1-20261030/registration.json'
ORIGINAL_COMMIT = 'e458efa11b4320b1d8982abea34dba3a2bf377d2'
ORIGINAL_STAGE = 'task-pair-historical-compare-v1-20261030'
OUTPUT_STAGE = 'task-pair-partial-diagnostic-v1-20261031'
LEDGER_SCHEMA = 'task-pair-historical-compare-failure-private-evidence-v1'
FAILURE_SCHEMA = 'task-pair-historical-compare-public-failure-v1'
SCHEMA = 'task-pair-partial-diagnostic-prereg-v1'
SCOPE = dict(historical_seed_role='S01', internal_seed_role='S02', opponent='market', side=0,
             policies=['baseline', 'control'], source_accepted_games=3, input_games=2,
             paired_comparisons=1, new_games=0, new_official_transitions=0, new_training_updates=0,
             posthoc=True, independent_evaluation=False, promotion=False,
             metrics_fields_previously_read=True, replay_trace_previously_hashed_only=True,
             replay_trace_previously_parsed=False,
             pair_economic_comparison_before_registration=False)
LIMITS = dict(initial_available_bytes=4 * 1024**3, during_available_bytes=3 * 1024**3,
              disk_free_floor_bytes=4 * 1024**3, original_files=79,
              original_stage_bytes=2 * 1024**3, ledger_bytes=1024**2,
              output_bytes=128 * 1024, wall_seconds=1800)
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_REQUIRED = frozenset((
    REL + '__init__.py', REL + 'cli.py', REL + 'core.py',
    'kaggriculture/learning/task_control_once_v1/entry.py',
    'kaggriculture/learning/task_pair_historical_compare_v1/entry.py',
    'kaggriculture/learning/task_pair_historical_compare_v1/bindings.py',
    'kaggriculture/learning/task_pair_historical_compare_v1/request_prefix.py',
    'kaggriculture/learning/task_pair_historical_compare_v1/projection.py',
    ORIGINAL_REGISTRATION,
))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _json(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'Invalid bounded input')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'Duplicate input key')
            value[key] = item
        return value
    def invalid(_value):
        raise ValueError('Nonfinite input')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)
    _require(type(value) is dict, 'Invalid input document')
    return value


def _read(path, maximum):
    path = Path(path)
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not (getattr(info, 'st_file_attributes', 0) & 0x400)
             and 0 < info.st_size <= maximum, 'Invalid bounded file')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    _require(len(raw) == info.st_size and len(raw) <= maximum, 'Bounded file changed')
    return raw


def _git(repo, *args):
    try:
        with open(os.devnull, 'rb') as null:
            proc = subprocess.run(['git', '-C', str(repo), *args], stdin=null,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        raise ValueError('Public source verification failed') from None
    _require(proc.returncode == 0 and len(proc.stdout) <= 8 * 1024**2
             and len(proc.stderr) <= 8 * 1024**2, 'Public source verification failed')
    return proc.stdout


def _source(repo, commit, relative, expected=None):
    path = repo.joinpath(*relative.split('/'))
    _require(path.resolve(strict=True) == path and path.is_relative_to(repo), 'Source path differs')
    raw = _read(path, 4 * 1024**2)
    _require(_git(repo, 'show', commit + ':' + relative) == raw
             and (expected is None or _sha(raw) == expected), 'Source snapshot differs')
    return raw


def _module(name, raw, path):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def _registration(value):
    _require(type(value) is dict and value.get('schema') == SCHEMA
             and value.get('original_commit') == ORIGINAL_COMMIT
             and _same(value.get('scope'), SCOPE) and _same(value.get('limits'), LIMITS)
             and type(value.get('failure_commit')) is str and _COMMIT.fullmatch(value['failure_commit'])
             and type(value.get('failure_certificate')) is str
             and type(value.get('failure_certificate_sha256')) is str
             and _HEX.fullmatch(value['failure_certificate_sha256'])
             and type(value.get('public_sources')) is dict
             and _REQUIRED.issubset(value['public_sources']), 'Partial registration differs')


def _public(repo, root, commit):
    """Authenticate new registration and reuse the original official verifier."""
    _require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__ and os.name == 'nt',
             'Isolated Windows Python required')
    _require(type(commit) is str and _COMMIT.fullmatch(commit), 'Invalid registration commit')
    _require(_git(repo, 'rev-parse', 'HEAD').decode().strip() == commit
             and not _git(repo, 'status', '--porcelain', '--untracked-files=all'),
             'Registered checkout differs')
    raw_reg = _source(repo, commit, REGISTRATION)
    reg = _json(raw_reg, 4 * 1024**2)
    _registration(reg)
    sources = reg['public_sources']
    original_raw = _source(repo, ORIGINAL_COMMIT, ORIGINAL_REGISTRATION, sources[ORIGINAL_REGISTRATION])
    original = _json(original_raw, 4 * 1024**2)
    original_sources = original.get('public_sources')
    _require(type(original_sources) is dict, 'Original registration differs')
    _require(all(type(reg.get(key)) is str and reg[key] == original.get(key)
                 for key in ('engine_commit', 'engine_sha256', 'previous_training_result_commit',
                             'historical_panel_result_commit')), 'Original public identity differs')
    # Progress can describe the failure after the old registration; every
    # original frozen source/record otherwise remains byte-bound.
    _require(all(sources.get(key) == digest for key, digest in original_sources.items()
                 if key != 'review/PROGRESS.md'), 'Original frozen sources changed')
    _git(repo, 'merge-base', '--is-ancestor', ORIGINAL_COMMIT, reg['failure_commit'])
    _git(repo, 'merge-base', '--is-ancestor', reg['failure_commit'], commit)
    failure_path = reg['failure_certificate']
    _require(failure_path.startswith('review/evidence/') and failure_path.endswith('/certificate.json')
             and sources.get(failure_path) == reg['failure_certificate_sha256'], 'Failure certificate differs')
    failure = _json(_source(repo, reg['failure_commit'], failure_path,
                            reg['failure_certificate_sha256']), 4 * 1024**2)
    _require(failure.get('schema') == FAILURE_SCHEMA
             and failure.get('registration_commit') == ORIGINAL_COMMIT
             and type(failure.get('planned_games')) is int and failure['planned_games'] == 8
             and type(failure.get('started_games')) is int and failure['started_games'] == 4
             and type(failure.get('accepted_games')) is int and failure['accepted_games'] == 3
             and type(failure.get('outer_exit_code_observed_in_session')) is int
             and failure['outer_exit_code_observed_in_session'] == 1
             and failure.get('outer_exit_independent_persistent_receipt') is False
             and failure.get('outer_observation_transcribed_after_exit') is True
             and failure.get('completed_panel') is False and failure.get('retry_allowed') is False,
             'Failure certificate scope differs')
    def snapshot(relative, name):
        return _module(name, _source(repo, commit, relative, sources.get(relative)), repo / relative)
    identity = snapshot('kaggriculture/learning/task_pair_dev_campaign_v1/identity.py', '_partial_identity')
    bootstrap = snapshot('kaggriculture/learning/task_pair_dev_campaign_v1/bootstrap.py', '_partial_bootstrap')
    once_auth = snapshot('kaggriculture/learning/task_control_once_v1/auth.py', '_partial_once_auth')
    once = snapshot('kaggriculture/learning/task_control_once_v1/entry.py', '_partial_once')
    historical = snapshot('kaggriculture/learning/task_pair_historical_compare_v1/entry.py', '_partial_history')
    _source(repo, commit, REL + 'cli.py', sources[REL + 'cli.py'])
    old_registration, old_panel, old_required = identity.REGISTRATION, identity._panel, identity._REQUIRED
    try:
        identity.REGISTRATION = REGISTRATION
        identity._panel = _registration
        identity._REQUIRED = old_required | _REQUIRED
        checked, old_reg, freeze = identity.verify_public(repo=repo, game_root=root, registration_commit=commit)
    finally:
        identity.REGISTRATION, identity._panel, identity._REQUIRED = old_registration, old_panel, old_required
    _require(_same(checked, reg), 'Registered metadata changed')
    once._check_package(root, reg)
    historical._check_package(root, reg)
    _check_partial_package(root, reg)
    modules = bootstrap._runtime(root, reg)
    binding = importlib.import_module('learning.task_pair_historical_compare_v1.bindings')
    core = importlib.import_module('learning.task_pair_partial_diagnostic_v1.core')
    for module in (binding, core):
        _require(Path(module.__file__).resolve(strict=True).is_relative_to(GAME_SOURCE), 'Imported source differs')
    return dict(repo=repo, root=root, commit=commit, reg=reg, reg_raw=raw_reg, original=original,
                old_reg=old_reg, freeze=freeze, identity=identity, bootstrap=bootstrap,
                once=once, historical=historical, historical_auth=once_auth,
                modules=modules, binding=binding, core=core, failure=failure)


def _check_partial_package(root, reg):
    name = 'task_pair_partial_diagnostic_v1'
    private, public = root / 'learning', GAME_SOURCE / 'learning'
    suffixes = ('.py', '.pyc', *importlib.machinery.EXTENSION_SUFFIXES)
    _require(not os.path.lexists(private / name), 'Partial package shadow')
    for base in (private, public):
        _require(not any(os.path.lexists(base / (name + suffix)) for suffix in suffixes),
                 'Partial package shadow')
    folder = public / name
    _require(folder.is_dir() and folder.resolve(strict=True) == folder
             and not os.path.lexists(folder / '__pycache__'), 'Partial package path differs')
    for item in ('__init__', 'cli', 'core'):
        _require(_sha(_read(folder / (item + '.py'), 4 * 1024**2)) == reg['public_sources'][REL + item + '.py']
                 and not os.path.lexists(folder / item)
                 and not any(os.path.lexists(folder / (item + suffix)) for suffix in suffixes if suffix != '.py'),
                 'Partial source differs')
    prefix = 'learning.' + name
    _require(not any(key == prefix or key.startswith(prefix + '.') for key in sys.modules),
             'Partial package imported too early')


def _safe_relative(value):
    _require(type(value) is str and value and len(value) <= 4096
             and not any(ord(c) < 32 or ord(c) == 127 or c in '\\:<>' + '"|?*' for c in value),
             'Invalid ledger relative path')
    path = PurePosixPath(value)
    reserved = re.compile(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z', re.IGNORECASE)
    _require(not path.is_absolute() and path.as_posix() == value
             and all(part not in ('.', '..') and part.rstrip(' .') == part
                     and not reserved.fullmatch(part.split('.', 1)[0]) for part in path.parts),
             'Invalid ledger relative path')
    return value


def _metadata(value):
    _require(type(value) is dict and set(value) == {'bytes', 'sha256'}
             and type(value['bytes']) is int and 0 <= value['bytes'] <= LIMITS['original_stage_bytes']
             and type(value['sha256']) is str and _HEX.fullmatch(value['sha256']), 'Invalid ledger metadata')


def _ledger(value):
    fields = {'schema', 'registration_commit', 'stage_files', 'attempt', 'failure',
              'observed_outer_exit_code', 'observed_outer_exit_source', 'recorded_after_exit', 'retry_allowed'}
    _require(type(value) is dict and set(value) == fields and value['schema'] == LEDGER_SCHEMA
             and value['registration_commit'] == ORIGINAL_COMMIT
             and type(value['observed_outer_exit_code']) is int and value['observed_outer_exit_code'] != 0
             and value['observed_outer_exit_source'] == 'operator tool session'
             and value['recorded_after_exit'] is True and value['retry_allowed'] is False
             and type(value['stage_files']) is dict and len(value['stage_files']) == LIMITS['original_files'],
             'Original evidence ledger differs')
    paths = [_safe_relative(path) for path in value['stage_files']]
    _require(len({path.casefold() for path in paths}) == len(paths), 'Duplicate ledger paths')
    for metadata in (*value['stage_files'].values(), value['attempt'], value['failure']):
        _metadata(metadata)
    _require(sum(item['bytes'] for item in value['stage_files'].values()) <= LIMITS['original_stage_bytes'],
             'Original stage size differs')
    return value


class _Guard:
    """Time and system-memory samples; individual long calls are not interrupted."""
    def __init__(self, psutil, started, clock=time.monotonic):
        self.psutil, self.started, self.clock, self.last = psutil, started, clock, started

    def sample(self, *, initial=False):
        now = self.clock()
        _require(type(now) in (int, float) and math.isfinite(now) and self.last <= now
                 and now < self.started + LIMITS['wall_seconds'], 'Diagnostic time limit')
        self.last = now
        available = self.psutil.virtual_memory().available
        threshold = LIMITS['initial_available_bytes' if initial else 'during_available_bytes']
        _require(type(available) is int and available >= threshold, 'Diagnostic memory limit')
        return True


def _hash_file(root, path, metadata, driver, guard):
    driver._no_links(root, path)
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_size == metadata['bytes'], 'Original file differs')
    digest, total = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while True:
            guard()
            block = stream.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            _require(total <= metadata['bytes'], 'Original file changed')
            digest.update(block)
    _require(total == metadata['bytes'] and digest.hexdigest() == metadata['sha256'], 'Original file differs')


def _stage_snapshot(root, stage, evidence, driver, guard):
    """Compare all 79 regular files to the post-exit private ledger, read-only."""
    driver._no_links(root, stage)
    observed = set()
    def failed(_error):
        raise ValueError('Original stage scan failed')
    for directory, dirs, files in os.walk(stage, topdown=True, followlinks=False, onerror=failed):
        parent = Path(directory)
        for name in (*dirs, *files):
            path = parent / name
            driver._no_links(stage, path)
            info = path.lstat()
            _require(stat.S_ISDIR(info.st_mode) if name in dirs else stat.S_ISREG(info.st_mode),
                     'Unexpected original stage item')
            if name in files:
                observed.add(path.relative_to(stage).as_posix())
    _require(observed == set(evidence['stage_files']), 'Original stage file set changed')
    for relative in sorted(observed):
        _hash_file(stage, stage.joinpath(*relative.split('/')), evidence['stage_files'][relative], driver, guard)
    _hash_file(root, stage.with_name(stage.name + '.attempt.local.json'), evidence['attempt'], driver, guard)
    _hash_file(root, stage.with_name(stage.name + '.failure.local.json'), evidence['failure'], driver, guard)
    return True


def _marker(game, kind, attempt_raw, manifest_raw):
    _require(kind in ('permit', 'started', 'consumed'), 'Invalid original marker')
    value = dict(schema='task-pair-historical-' + kind + '-v1', game=game,
                 registration_commit=ORIGINAL_COMMIT, attempt_sha256=_sha(attempt_raw), retry_allowed=False)
    if kind != 'started':
        value['manifest_sha256'] = _sha(manifest_raw)
    return value


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _old_context(ctx, stage, guard):
    """Independently rebuild old roles from the previously pinned receipts."""
    bootstrap, identity, once = ctx['bootstrap'], ctx['identity'], ctx['once']
    root, repo, reg = ctx['root'], ctx['repo'], ctx['reg']
    guard()
    status_path, current, old_dev, prior = bootstrap._private_status()
    _require(type(current) is dict and _same(old_dev, current.get('current_pair_dev_diagnostic'))
             and old_dev.get('state') == 'failed_no_retry', 'Prior closed status differs')
    status_raw = bootstrap._read(status_path)
    binding_path = repo.parent / 'private' / bootstrap.PREBIND
    binding_raw = bootstrap._read(binding_path, 65536)
    once._verify_private_snapshot(bootstrap, ctx['historical_auth'], current, prior, status_raw, binding_raw,
                                  old_dev.get('prebinding_record_sha256'))
    seed_path = root / bootstrap.OLD_PANEL / 'private-seeds.local.json'
    seed_raw = bootstrap._read(seed_path, 16384)
    def check():
        guard()
        return (_git(repo, 'rev-parse', 'HEAD').decode().strip() == ctx['commit']
                and not _git(repo, 'status', '--porcelain', '--untracked-files=all')
                and _read(repo / REGISTRATION, 4 * 1024**2) == ctx['reg_raw']
                and bootstrap._read(status_path) == status_raw and bootstrap._read(binding_path, 65536) == binding_raw
                and bootstrap._read(seed_path, 16384) == seed_raw)
    manifest, previous, roles = ctx['binding'].build(
        modules=ctx['modules'], bootstrap=bootstrap, identity=identity, reg=reg,
        old_reg=ctx['old_reg'], freeze=ctx['freeze'], game_root=root, repo=repo,
        status_path=status_path, prior_binding=prior, check=check)
    _require(ctx['binding'].verify_mapping(manifest, previous, roles) is True and check() is True,
             'Previous roles differ')
    driver = ctx['modules']['driver']
    manifest_raw = driver._read(stage, stage / 'comparison-manifest.local.json', 16384)
    expected = once._manifest_document(manifest, ORIGINAL_COMMIT, _sha(binding_raw))
    expected['schema'] = 'task-pair-historical-private-manifest-v1'
    expected['actual_seed_roles'] = {'S02': 'S01', 'S03': 'S04'}
    _require(_same(_json(manifest_raw, 16384), expected), 'Original manifest differs')
    attempt_raw = _read(stage.with_name(stage.name + '.attempt.local.json'), 16384)
    _require(_same(_json(attempt_raw, 16384), dict(schema='task-pair-historical-attempt-v1',
                 registration_commit=ORIGINAL_COMMIT, max_games=8, retry_allowed=False)),
             'Original attempt differs')
    failure = _json(_read(stage.with_name(stage.name + '.failure.local.json'), 16384), 16384)
    _require(failure.get('schema') == 'task-pair-historical-attempt-failure-v1'
             and failure.get('registration_commit') == ORIGINAL_COMMIT and failure.get('retry_allowed') is False
             and failure.get('stage_created') is True and failure.get('accepted_games') is None,
             'Original failure differs')
    return manifest, attempt_raw, manifest_raw, check


def _accepted_pair(ctx, stage, manifest, attempt_raw, manifest_raw, check, guard):
    dmod, acceptance, child = (ctx['modules'][name] for name in ('driver', 'acceptance', 'child'))
    driver = dmod.LocalDriver(manifest=manifest, game_root=ctx['root'], stage=stage,
        commit=ORIGINAL_COMMIT, command_for=lambda _game: None, check=check)
    rows, plans = {}, {}
    for arm in ('baseline', 'control'):
        game = f'dev-S02-market-0-{arm}'
        matches = [p for p in manifest.panel.plans() if p['game'] == game]
        _require(len(matches) == 1, 'Original game plan differs')
        plan = manifest.panel.validate_plan(matches[0]); plans[arm] = plan
        for kind in ('permit', 'started', 'consumed'):
            raw = dmod._read(stage, stage / 'job-results' / (game + '.' + kind + '.local.json'), 16384)
            _require(raw == _encoded(_marker(game, kind, attempt_raw, manifest_raw)), 'Original marker differs')
        bundle = driver.read_bundle(plan)
        supervision_raw = dmod._read(stage, stage / 'job-results' / (game + '.supervision.local.json'),
                                     acceptance.MAX_RECEIPT_BYTES)
        accepted_raw = dmod._read(stage, stage / 'job-results' / (game + '.acceptance.local.json'),
                                 acceptance.MAX_RECEIPT_BYTES)
        supervision = acceptance._json(supervision_raw, acceptance.MAX_RECEIPT_BYTES)
        accepted = acceptance._json(accepted_raw, acceptance.MAX_RECEIPT_BYTES)
        _require(supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
                 and supervision.get('game') == game and supervision.get('retry_allowed') is False
                 and accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
                 and accepted.get('game') == game and accepted.get('passed') is True
                 and accepted.get('retry_allowed') is False
                 and accepted.get('supervision_sha256') == _sha(supervision_raw)
                 and accepted.get('job_input_sha256') == dmod._hash(stage, stage / 'job-inputs' / (game + '.json'),
                                                                    child.worker.MAX_JOB_BYTES)
                 and accepted.get('receipt_sha256') == {name: _sha(raw) for name, raw in bundle.items()},
                 'Original acceptance chain differs')
        row = acceptance.accept(manifest.panel, plan, bundle, supervision.get('supervision'), driver.verify_files)
        audit = acceptance._json(bundle['audit'], acceptance.MAX_RECEIPT_BYTES)
        counts = audit.get('primary_model_counts')
        _require(type(counts) is dict and all(type(counts.get(k)) is int and counts[k] > 0
                 for k in ('task_forwards', 'crop_forwards')), 'Original model counts differ')
        rows[arm] = row
        _require(check() is True and guard() is True, 'Original context changed')
    return driver, plans, rows


def _write_new(path, data, root):
    existing = list(root.iterdir())
    entries = [item.lstat() for item in existing]
    _require(all(stat.S_ISREG(item.st_mode) and not stat.S_ISLNK(item.st_mode)
                 and not (getattr(item, 'st_file_attributes', 0) & 0x400) for item in entries),
             'Unexpected diagnostic output item')
    used = sum(item.st_size for item in entries)
    _require(type(data) is bytes and 0 < len(data) <= LIMITS['output_bytes'] - used
             and shutil.disk_usage(root).free >= LIMITS['disk_free_floor_bytes'] + len(data),
             'Diagnostic output resource limit')
    with path.open('xb') as stream:
        _require(stream.write(data) == len(data), 'Diagnostic output write failed')
        stream.flush()
        os.fsync(stream.fileno())
    _require(_read(path, LIMITS['output_bytes']) == data, 'Diagnostic output readback differs')


def _success_receipts(result_bytes, registration_commit, evidence_sha256):
    """Bind the safe result and its receipt without claiming an external timestamp."""
    _require(type(result_bytes) is bytes and 0 < len(result_bytes) <= LIMITS['output_bytes'],
             'Invalid result bytes')
    receipt = dict(schema='task-pair-partial-local-receipt-v1', passed=True,
                   registration_commit=registration_commit, original_registration_commit=ORIGINAL_COMMIT,
                   input_ledger_sha256=evidence_sha256, result_sha256=_sha(result_bytes),
                   original_files=LIMITS['original_files'], accepted_games=2, pairs=1,
                   original_official_replays_reused=2, new_games=0, new_official_transitions=0,
                   new_training_updates=0, independence='operator_post_exit_snapshot', promotion=False)
    receipt_bytes = _encoded(receipt)
    returned_bytes = _encoded(dict(schema='task-pair-partial-returned-v1', returned=True,
                                   receipt_sha256=_sha(receipt_bytes), observed_outer_exit=False))
    return receipt_bytes, returned_bytes


def run(*, game_root, registration_commit, evidence_file, evidence_sha256):
    """Run a separately preregistered one-pair read-only diagnostic once."""
    started = time.monotonic()
    root, repo = Path(game_root), PUBLIC_ROOT
    _require(root.is_absolute() and root.resolve(strict=True) == root
             and root == PUBLIC_ROOT.parent.parent / 'kaggriculture', 'Project root differs')
    _require(type(evidence_sha256) is str and _HEX.fullmatch(evidence_sha256), 'Invalid private evidence identity')
    site = (Path(sys.prefix) / 'Lib/site-packages').resolve(strict=True)
    spec = importlib.util.find_spec('psutil')
    _require(spec is not None and spec.origin is not None
             and Path(spec.origin).resolve(strict=True).is_relative_to(site), 'Resource sampler origin differs')
    import psutil
    guard_object = _Guard(psutil, started)
    guard_object.sample(initial=True)
    guard = guard_object.sample
    ctx = _public(repo, root, registration_commit)
    guard()
    artifacts = root / 'artifacts'
    _require(artifacts.resolve(strict=True) == artifacts, 'Artifact root differs')
    output = artifacts / OUTPUT_STAGE
    original = artifacts / ORIGINAL_STAGE
    _require(original.resolve(strict=True) == original and original.is_dir()
             and not os.path.lexists(output), 'Diagnostic output or original stage differs')
    output.mkdir(exist_ok=False)
    try:
        _write_new(output / 'attempt.local.json', _encoded(dict(schema='task-pair-partial-attempt-v1',
                   registration_commit=registration_commit, retry_allowed=False)), output)
        guard()
        evidence_path = Path(evidence_file)
        _require(evidence_path.is_absolute() and evidence_path.resolve(strict=True) == evidence_path,
                 'Private evidence path differs')
        raw_evidence = _read(evidence_path, LIMITS['ledger_bytes'])
        _require(_sha(raw_evidence) == evidence_sha256, 'Private evidence identity differs')
        evidence = _ledger(_json(raw_evidence, LIMITS['ledger_bytes']))
        _require(evidence['observed_outer_exit_code'] == ctx['failure']['outer_exit_code_observed_in_session'],
                 'Original exit record differs')
        dmod = ctx['modules']['driver']
        _require(_stage_snapshot(root, original, evidence, dmod, guard), 'Original evidence differs')
        manifest, attempt_raw, manifest_raw, check = _old_context(ctx, original, guard)
        driver, plans, rows = _accepted_pair(ctx, original, manifest, attempt_raw, manifest_raw, check, guard)
        replays, traces = {}, {}
        try:
            for arm in ('baseline', 'control'):
                guard()
                game = plans[arm]['game']
                replays[arm] = dmod._replay(original, original / 'games' / game / 'replay.json.gz',
                                            rows[arm]['replay_sha256'])
                guard()
                traces[arm] = dmod._trace(original, original / 'traces' / (game + '.jsonl.gz'),
                                          rows[arm]['trace_sha256'])
            result = ctx['core'].analyze(rows=rows, replays=replays, traces=traces, check=guard)
        finally:
            for stream in traces.values():
                stream.close()
        del replays, traces, rows
        _require(check() is True and _stage_snapshot(root, original, evidence, dmod, guard)
                 and _sha(_read(evidence_path, LIMITS['ledger_bytes'])) == evidence_sha256,
                 'Original evidence changed')
        guard()
        result_bytes = _encoded(result)
        receipt_bytes, returned_bytes = _success_receipts(result_bytes, registration_commit, evidence_sha256)
        _write_new(output / 'result.json', result_bytes, output)
        _write_new(output / 'receipt.local.json', receipt_bytes, output)
        guard()
        _write_new(output / 'returned.local.json', returned_bytes, output)
        guard()
        return True
    except BaseException:
        try:
            _write_new(output / 'failure.local.json', _encoded(dict(schema='task-pair-partial-failure-v1',
                       passed=False, retry_allowed=False)), output)
        except Exception:
            pass
        raise


SNAPSHOT_STUB = '''import pathlib, re, subprocess, sys
try:
 repo=pathlib.Path(sys.argv.pop(1)); commit=sys.argv.pop(1)
 if re.fullmatch(r"[0-9a-f]{40}",commit) is None: raise ValueError()
 rel="kaggriculture/learning/task_pair_partial_diagnostic_v1/cli.py"
 path=repo.joinpath(*rel.split("/"))
 with path.open("rb") as stream: raw=stream.read(4194305)
 if not raw or len(raw)>4194304: raise ValueError()
 result=subprocess.run(["git","-C",str(repo),"show",commit+":"+rel],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30)
 if raw!=result.stdout: raise ValueError()
 exec(compile(raw,str(path),"exec",dont_inherit=True),{"__name__":"__main__","__file__":str(path)})
except Exception:
 raise SystemExit("Partial diagnostic snapshot stopped.") from None
'''


def command(*, repo, commit, game_root, evidence_file, evidence_sha256):
    """Build the isolated committed-byte CLI command; no execution occurs."""
    _require(type(commit) is str and _COMMIT.fullmatch(commit)
             and type(evidence_sha256) is str and _HEX.fullmatch(evidence_sha256), 'Invalid local command')
    return [sys.executable, '-I', '-B', '-c', SNAPSHOT_STUB, str(repo), commit,
            '--game-root', str(game_root), '--registration-commit', commit,
            '--evidence-file', str(evidence_file), '--evidence-sha256', evidence_sha256]


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid diagnostic arguments')


def main(argv=None):
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--evidence-file', required=True)
        parser.add_argument('--evidence-sha256', required=True)
        args = parser.parse_args(argv)
        run(game_root=args.game_root, registration_commit=args.registration_commit,
            evidence_file=args.evidence_file, evidence_sha256=args.evidence_sha256)
    except SystemExit:
        raise SystemExit('Partial diagnostic stopped without a normal return.') from None
    except Exception:
        raise SystemExit('Partial diagnostic stopped; inspect local evidence.') from None


if __name__ == '__main__':
    main()
