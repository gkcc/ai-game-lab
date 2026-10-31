"""Authenticate and stream the eight previously completed exploration games.

Nothing in this module starts games, trains a model, or publishes data.  All
objects returned by it are private.  The caller must bind the campaign-result
digest in the new registration *before* using it and enforce an external
wall-clock/memory budget.  The old binding is authenticated against its
published registration, not regenerated from observed outcomes.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_games_v1 import panel
from learning.task_pair_dev_metrics_v1.replay_pair import _replay as validate_replay
from learning.task_pair_fresh_multi_v1 import entry, prepare as fresh
from . import samples


OLD_REGISTRATION_COMMIT = '40f3f393819bbdc62af1007877cd5fbdf3a9a42f'
REPLAY_DECODED_MAX = 128 * 1024 * 1024
COMPRESSED_MAX = 128 * 1024 * 1024
MODEL_MAX = 16 * 1024 * 1024
SMALL_MAX = 512 * 1024
RESULT_MAX = 8 * 1024 * 1024
HEX = re.compile(r'[0-9a-f]{64}\Z')


class SourceError(ValueError):
    """A private source or its provenance failed validation."""


def _require(value, message):
    if not value:
        raise SourceError(message)


def _digest(value):
    _require(type(value) is str and HEX.fullmatch(value) is not None, 'invalid digest')
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _nonfinite(_value):
    raise SourceError('nonfinite JSON value')


def _json(raw, limit):
    _require(isinstance(raw, (bytes, bytearray)) and 0 < len(raw) <= limit, 'invalid JSON size')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError):
        raise SourceError('invalid JSON') from None
    _require(type(value) is dict, 'JSON root must be an object')
    return value


def _registration_blob():
    """Read the registration from the fixed earlier commit, not the worktree."""
    command = ['git', '-C', str(fresh.REPO), 'show',
               OLD_REGISTRATION_COMMIT + ':' + entry.REGISTRATION]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=False, timeout=30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        raise SourceError('registration lookup failed') from None
    _require(result.returncode == 0 and 0 < len(result.stdout) <= RESULT_MAX,
             'registration lookup failed')
    return result.stdout


def _stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@contextmanager
def _open_checked(root, path, maximum, guard):
    """Keep one descriptor and reject links or identity changes around a read."""
    guard()
    try:
        path = fresh._inside(root, path)
        before = path.lstat()
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
                 'invalid source size or type')
        stream = path.open('rb')
    except OSError:
        raise SourceError('private source could not be opened') from None
    with stream:
        try:
            opened = os.fstat(stream.fileno())
        except OSError:
            raise SourceError('private source identity could not be read') from None
        _require(_stat_identity(opened) == _stat_identity(before), 'opened source identity differs')
        # Exceptions from the caller, including a guard OSError, propagate as-is.
        yield stream
        guard()
        try:
            after = path.lstat()
            current = os.fstat(stream.fileno())
            resolved = fresh._inside(root, path)
        except OSError:
            raise SourceError('private source identity could not be reread') from None
        _require(resolved == path
                     and _stat_identity(current) == _stat_identity(before)
                     and _stat_identity(after) == _stat_identity(before),
                     'source identity changed during read')


def _small(root, path, maximum, guard):
    with _open_checked(root, path, maximum, guard) as stream:
        try:
            raw = stream.read(maximum + 1)
        except OSError:
            raise SourceError('private receipt could not be read') from None
        _require(0 < len(raw) <= maximum, 'private receipt exceeds limit')
        return raw


def _rewind(stream):
    try:
        stream.seek(0)
    except OSError:
        raise SourceError('private source could not be rewound') from None


def _hash_stream(stream, maximum, guard):
    _rewind(stream)
    digest, count = hashlib.sha256(), 0
    while True:
        guard()
        try:
            data = stream.read(65536)
        except OSError:
            raise SourceError('private source could not be read') from None
        if not data:
            break
        count += len(data)
        _require(count <= maximum, 'compressed source exceeds limit')
        digest.update(data)
    _require(count > 0, 'empty compressed source')
    return digest.hexdigest(), count


class _GuardedGzip:
    """Sanitize file read failures without intercepting the caller's guard."""
    def __init__(self, stream):
        self.stream = stream

    def read(self, size=-1):
        try:
            return self.stream.read(size)
        except OSError:
            raise SourceError('compressed source could not be read') from None

    def readline(self, size=-1):
        try:
            return self.stream.readline(size)
        except OSError:
            raise SourceError('compressed source could not be read') from None


def _rule_sha(guard):
    spec = importlib.util.find_spec('kaggle_environments')
    _require(spec is not None and spec.submodule_search_locations is not None
             and len(spec.submodule_search_locations) == 1, 'official package unavailable')
    root = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
    path = root / 'envs' / 'kaggriculture' / 'kaggriculture.py'
    with _open_checked(root, path, RESULT_MAX, guard) as raw:
        value, _ = _hash_stream(raw, RESULT_MAX, guard)
    return value


def _model_file(relative, expected, guard):
    """Return the authenticated absolute model path, never a cwd-relative one."""
    _digest(expected)
    _require(type(relative) is str and relative.startswith('artifacts/'), 'invalid model path')
    path = fresh.GAME.joinpath(*relative.split('/'))
    with _open_checked(fresh.GAME, path, MODEL_MAX, guard) as raw:
        actual, _ = _hash_stream(raw, MODEL_MAX, guard)
    _require(actual == expected and path.is_absolute(), 'model identity differs')
    return str(path)


def _verify_campaign(registration, binding_raw, market_raw, campaign, closed):
    commitment = _sha(binding_raw)
    _require(registration.get('schema') == entry.SCHEMA
             and registration.get('binding_commitment_sha256') == commitment
             and registration.get('engine_commit') == panel.ENGINE_COMMIT
             and registration.get('engine_sha256') == panel.ENGINE_SHA
             and registration.get('panel') == entry.PANEL
             and registration.get('evaluation') == entry.EVALUATION,
             'old registration or commitment differs')
    _require(type(campaign) is dict and campaign.get('schema') == 'task-pair-fresh-multi-private-result-v1'
             and campaign.get('registration_commit') == OLD_REGISTRATION_COMMIT
             and campaign.get('binding_sha256') == commitment
             and type(campaign.get('games')) is int and campaign['games'] == 32
             and type(campaign.get('official_replay_passes')) is int and campaign['official_replay_passes'] == 32
             and type(campaign.get('comparisons')) is int and campaign['comparisons'] == 16
             and type(campaign.get('prefix_comparisons')) is int and campaign['prefix_comparisons'] == 16
             and campaign.get('training') is False and campaign.get('promotion') is False,
             'campaign result differs')
    _require(type(closed) is dict and closed.get('schema') == 'task-pair-fresh-multi-closed-v1'
             and closed.get('registration_commit') == OLD_REGISTRATION_COMMIT
             and closed.get('binding_sha256') == commitment and closed.get('outcome') == 'completed'
             and closed.get('retry_allowed') is False, 'campaign did not close successfully')
    _digest(_sha(market_raw))
    receipts = campaign.get('receipts')
    _require(type(receipts) is list and len(receipts) == 32, 'campaign receipts are incomplete')
    expected = {(cohort, game) for cohort in entry.COHORTS for game in entry.GAMES}
    accepted = {}
    for receipt in receipts:
        _require(type(receipt) is dict, 'invalid campaign receipt')
        key = (receipt.get('cohort'), receipt.get('game'))
        _require(key in expected and key not in accepted, 'duplicate or unexpected campaign coordinate')
        accepted[key] = _digest(receipt.get('acceptance_sha256'))
    _require(set(accepted) == expected, 'campaign coordinates are incomplete')
    return accepted


@dataclass(repr=False)
class PrivateContext:
    """Internal snapshot; no values should be formatted for logs or publication."""
    manifests: dict
    role_map: dict
    acceptances: dict
    binding_sha256: str
    campaign_sha256: str
    active: bool = False

    def __repr__(self):
        return '<PrivateSourceContext>'


class PrivateGame:
    """One local replay; use as a context manager and discard all references."""
    def __init__(self, context, replay, selected, identity):
        self._context = context
        self.replay, self.selected, self.identity = replay, selected, identity

    def __repr__(self):
        return '<PrivateSourceGame>'

    def __enter__(self):
        return self

    def close(self):
        if self._context is not None:
            self.replay = self.selected = self.identity = None
            self._context.active = False
            self._context = None

    def __exit__(self, _type, _value, _traceback):
        self.close()


def open_context(*, campaign_sha256, guard):
    """Verify old public commitment and private completed-campaign identities.

    ``campaign_sha256`` must come from the new, independently frozen local
    registration.  The campaign digest itself was recorded after the earlier
    games; it is not a third-party precommitment. No game is opened here.
    """
    _digest(campaign_sha256)
    _require(callable(guard), 'resource guard required')
    guard()
    reg = _json(_registration_blob(), RESULT_MAX)
    expected_binding = _digest(reg.get('binding_commitment_sha256'))
    binding_raw, binding, inventory = fresh.verify_binding(expected_binding)
    guard()
    market_raw, market = fresh.load_verified_market_binding()
    _require(binding.get('market_binding_sha256') == _sha(market_raw)
             and inventory.get('market_binding_sha256') == _sha(market_raw),
             'old market binding differs')
    campaign_raw = _small(fresh.PRIVATE_ROOT, fresh.PRIVATE / 'campaign-result.local.json', RESULT_MAX, guard)
    closed_raw = _small(fresh.PRIVATE_ROOT, fresh.PRIVATE / 'closed.local.json', SMALL_MAX, guard)
    _require(_sha(campaign_raw) == campaign_sha256, 'campaign digest differs')
    campaign, closed = _json(campaign_raw, RESULT_MAX), _json(closed_raw, SMALL_MAX)
    acceptances = _verify_campaign(reg, binding_raw, market_raw, campaign, closed)
    _require(_rule_sha(guard) == panel.ENGINE_SHA, 'official rule bytes differ')
    _require(binding.get('cohorts') == entry.COHORTS and type(binding.get('seed_roles')) is dict
             and set(binding['seed_roles']) == set(samples.ROLES), 'seed roles differ')
    seeds = list(binding['seed_roles'].values())
    _require(all(type(s) is int and 0 <= s < 2**32 for s in seeds) and len(set(seeds)) == 4,
             'invalid seed roles')
    manifests, role_map = {}, {}
    for cohort, aliases in entry.COHORTS.items():
        cohort_seeds = {alias: binding['seed_roles'][role] for alias, role in aliases.items()}
        manifests[cohort] = build_manifest(seeds=cohort_seeds, models=market['models'], crop=market['crop'],
                                          opponents=market['opponents'], worker_sha256=market['worker_sha256'])
        for alias, role in aliases.items():
            role_map[role] = (cohort, alias, cohort_seeds[alias])
    guard()
    return PrivateContext(manifests, role_map, acceptances, expected_binding, campaign_sha256)


def _verified_receipt(stage, name, expected, guard):
    raw = _small(fresh.ARTIFACTS, stage / 'job-results' / (name + '.acceptance.local.json'), SMALL_MAX, guard)
    _require(_sha(raw) == expected, 'acceptance receipt digest differs')
    accepted = _json(raw, SMALL_MAX)
    _require(accepted.get('schema') == 'task-pair-dev-job-acceptance-v1'
             and accepted.get('game') == name and accepted.get('passed') is True
             and accepted.get('retry_allowed') is False and type(accepted.get('receipt_sha256')) is dict,
             'acceptance receipt differs')
    return accepted['receipt_sha256']


def _receipt(stage, name, kind, hashes, guard):
    raw = _small(fresh.ARTIFACTS, stage / 'job-results' / (name + '.' + kind + '.json'), SMALL_MAX, guard)
    _require(_sha(raw) == _digest(hashes.get(kind)), 'worker receipt digest differs')
    return _json(raw, SMALL_MAX), _sha(raw)


def _validate_row_audit(row, audit, complete, row_sha, audit_sha, manifest, plan):
    job = {key: row.get(key) for key in panel._JOB}
    _require(manifest.panel.validate_plan(job, job=True) == job
             and {key: job[key] for key in panel._PLAN} == plan,
             'worker plan differs from fixed coordinate')
    _require(row.get('passed') is True and type(row.get('states')) is int and row['states'] == 720
             and type(row.get('action_turns')) is int and row['action_turns'] == 719
             and row.get('status') == ['DONE', 'DONE'] and type(row.get('agent_errors')) is int
             and row['agent_errors'] == 0 and row.get('engine_sha256') == panel.ENGINE_SHA
             and row.get('crop_checkpoint_sha256') == manifest.panel.crop[1], 'worker game did not pass')
    banks = row.get('banks')
    _require(type(banks) is list and len(banks) == 2
             and all(type(x) in (int, float) and math.isfinite(x) for x in banks)
             and row.get('rewards') == banks, 'invalid official rewards')
    for key in ('replay_gzip_sha256', 'trace_sha256', 'semantic_replay_sha256', 'semantic_replay_sha256_v2'):
        _digest(row.get(key))
    official = audit.get('official')
    _require(audit.get('game') == plan['game'] and audit.get('passed') is True
             and type(audit.get('errors')) is int and audit['errors'] == 0
             and all(type(audit.get(k)) is int and audit[k] == 719 for k in
                     ('actor_actions_reproduced', 'rival_actions_reproduced', 'trace_turns_reproduced'))
             and audit.get('trace_sha256') == row['trace_sha256']
             and audit.get('replay_sha256') == row['replay_gzip_sha256']
             and type(official) is dict and official.get('passed') is True
             and type(official.get('official_transitions_reexecuted')) is int
             and official['official_transitions_reexecuted'] == 719
             and official.get('both_public_and_private_states_match_each_step') is True
             and official.get('inventory_order_checked_each_step') is True
             and official.get('engine_sha256') == panel.ENGINE_SHA
             and official.get('semantic_sha256') == row['semantic_replay_sha256']
             and official.get('semantic_sha256_v2') == row['semantic_replay_sha256_v2']
             and official.get('final_rewards') == row.get('banks') == row.get('rewards'),
             'official replay audit differs')
    _require(complete.get('game') == plan['game'] and complete.get('passed') is True
             and complete.get('row_sha256') == row_sha and complete.get('certificate_sha256') == audit_sha,
             'worker completion chain differs')
    counts = audit.get('primary_model_counts')
    _require(type(counts) is dict and counts == audit.get('live_model_counts')
             and type(counts.get('task_forwards')) is int and counts['task_forwards'] > 0
             and type(counts.get('controller_calls')) is int and counts['controller_calls'] == 719,
             'replayed model call counts differ')


def _read_replay(stage, name, expected, seed, guard):
    path = stage / 'games' / name / 'replay.json.gz'
    with _open_checked(fresh.ARTIFACTS, path, COMPRESSED_MAX, guard) as raw:
        before, compressed = _hash_stream(raw, COMPRESSED_MAX, guard)
        _require(before == expected, 'replay digest differs')
        _rewind(raw)
        data = bytearray()
        with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
            safe = _GuardedGzip(stream)
            while True:
                guard()
                piece = safe.read(min(1024 * 1024, REPLAY_DECODED_MAX + 1 - len(data)))
                if not piece:
                    break
                data.extend(piece)
                _require(len(data) <= REPLAY_DECODED_MAX, 'decoded replay exceeds limit')
        replay = _json(data, REPLAY_DECODED_MAX)
        after, again = _hash_stream(raw, COMPRESSED_MAX, guard)
        _require((after, again) == (before, compressed), 'replay changed during read')
    _require(replay.get('name') == 'kaggriculture' and replay.get('configuration', {}).get('episodeSteps') == 720
             and replay.get('configuration', {}).get('seed') is None, 'replay metadata differs')
    validate_replay(replay, seed)
    return replay


def _read_trace(stage, name, expected, task_sha, crop_sha, guard):
    path = stage / 'traces' / (name + '.jsonl.gz')
    with _open_checked(fresh.ARTIFACTS, path, COMPRESSED_MAX, guard) as raw:
        before, compressed = _hash_stream(raw, COMPRESSED_MAX, guard)
        _require(before == expected, 'trace digest differs')
        _rewind(raw)
        with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
            selection = samples.scan(_GuardedGzip(stream), task_sha256=task_sha, crop_sha256=crop_sha, guard=guard)
        after, again = _hash_stream(raw, COMPRESSED_MAX, guard)
        _require((after, again) == (before, compressed), 'trace changed during read')
    return selection, compressed


def read_game(context, *, role, opponent, guard):
    """Open the fixed role/opponent game; caller must close before the next.

    Same-file-descriptor hashes bracket the trace scan and replay decoding.
    ``game.replay`` and the selected trace documents may contain both players'
    private history. They are for local official verification only; pass only
    the legal own-player view into a deployed/model controller.
    """
    _require(isinstance(context, PrivateContext) and context.active is False and callable(guard),
             'source context is unavailable')
    coordinate = samples.source_coordinates(role, opponent)
    cohort, alias, seed = context.role_map[role]
    side, arm = coordinate['side'], coordinate['arm']
    manifest = context.manifests[cohort]
    plan = next((p for p in manifest.panel.plans() if p['seed_alias'] == alias
                 and p['opponent'] == opponent and p['side'] == side and p['policy'] == arm), None)
    _require(plan is not None and plan['seed'] == seed, 'fixed source coordinate differs')
    name, stage = plan['game'], fresh.ARTIFACTS / entry.STAGES[cohort]
    context.active = True
    try:
        guard()
        hashes = _verified_receipt(stage, name, context.acceptances[(cohort, name)], guard)
        row, row_sha = _receipt(stage, name, 'row', hashes, guard)
        audit, audit_sha = _receipt(stage, name, 'audit', hashes, guard)
        complete, _ = _receipt(stage, name, 'complete', hashes, guard)
        _validate_row_audit(row, audit, complete, row_sha, audit_sha, manifest, plan)
        task_relative, task_sha = manifest.panel.task_model(plan)
        crop_relative, crop_sha = manifest.panel.crop
        task_path = _model_file(task_relative, task_sha, guard)
        crop_path = _model_file(crop_relative, crop_sha, guard)
        replay = _read_replay(stage, name, row['replay_gzip_sha256'], seed, guard)
        selection, compressed = _read_trace(stage, name, row['trace_sha256'], task_sha, crop_sha, guard)
        identity = dict(role=role, opponent=opponent, side=side, arm=arm,
                        task_path=task_path, task_sha256=task_sha, crop_path=crop_path,
                        crop_sha256=crop_sha, game_limits=manifest.panel.worker_config()['game_limits'],
                        trace_decoded_bytes=selection['decoded_bytes'], trace_compressed_bytes=compressed)
        guard()
        return PrivateGame(context, replay, selection['selected'], identity)
    except BaseException:
        context.active = False
        raise
