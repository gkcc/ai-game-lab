"""Explicit local IO for a preregistered serial development campaign.

The caller supplies an authenticated manifest, source check and fixed child
command builder.  All created artifacts are private; no directory is removed.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

from learning.task_pair_dev_games_v1 import capture, worker
from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS
from learning.task_pair_dev_metrics_v1.replay_pair import compare_replays
from . import acceptance, campaign
from .manifest import PrivateManifest


MAX_SOURCE = 4 * 1024 * 1024
MAX_MODEL = 16 * 1024 * 1024
MAX_REPLAY_DECODED = campaign.REPLAY_DECODED_BYTES
MAX_SUMMARY = 8 * 1024 * 1024
RECEIPTS = {
    'row': '.row.json', 'audit': '.audit.json', 'complete': '.complete.json',
    'storage': '.storage.json', 'metrics': '.metrics.local.json',
    'metrics_receipt': '.metrics-receipt.local.json',
}
COMPARISON_ARMS = (('baseline', 'control'), ('control', 'auxiliary'))


def _comparison_file(stage, reference_game, candidate):
    return Path(stage) / 'job-results' / (reference_game + '-to-' + candidate + '.comparison.local.json')


def _no_links(root, path):
    root, path = Path(root), Path(path)
    if path != root and not path.is_relative_to(root):
        raise ValueError('Private path is outside its root')
    current = root
    for part in ((), *[path.relative_to(root).parts[:i]
                       for i in range(1, len(path.relative_to(root).parts) + 1)]):
        current = root.joinpath(*part)
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Private path contains a link')


def _read(root, path, limit):
    path = Path(path)
    _no_links(root, path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError('Invalid private file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size or len(raw) > limit:
        raise ValueError('Private file length differs')
    return raw


def _hash(root, path, limit):
    path = Path(path)
    _no_links(root, path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError('Invalid private file')
    digest, total = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while True:
            block = stream.read(65536)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise ValueError('Private file exceeds budget')
            digest.update(block)
    if total != info.st_size:
        raise ValueError('Private file length differs')
    return digest.hexdigest()


def _write_new(root, path, value, limit=MAX_SUMMARY):
    data = (json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')
    if len(data) > limit:
        raise ValueError('Private output exceeds budget')
    path = Path(path)
    _no_links(root, path)
    with path.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('Incomplete private output')
    if _read(root, path, limit) != data:
        raise ValueError('Private output readback differs')


def _owned_size(stage):
    """Inspect this new owned stage only; never scan its artifacts parent."""
    stage = Path(stage)
    total = 0
    def fail(_error):
        raise OSError('Owned stage inspection failed')
    for directory, dirs, files in os.walk(stage, topdown=True, followlinks=False, onerror=fail):
        parent = Path(directory)
        for name in (*dirs, *files):
            path = parent / name
            _no_links(stage, path)
            info = path.lstat()
            if name in files:
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('Unexpected owned stage object')
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError('Unexpected owned stage object')
    return total


def _replay(root, path, expected):
    if _hash(root, path, capture.MAX_COMPRESSED_BYTES) != expected:
        raise ValueError('Replay identity differs')
    with Path(path).open('rb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
            data = stream.read(MAX_REPLAY_DECODED + 1)
    if len(data) > MAX_REPLAY_DECODED:
        raise ValueError('Replay decoded budget exceeded')
    result = acceptance._json(data, MAX_REPLAY_DECODED)
    if _hash(root, path, capture.MAX_COMPRESSED_BYTES) != expected:
        raise ValueError('Replay identity changed')
    return result


def _trace(root, path, expected):
    """Yield bounded private trace lines, with a checked same-handle digest."""
    _no_links(root, path)
    with Path(path).open('rb') as raw:
        if capture._hash_stream(raw) != expected:
            raise ValueError('Trace identity differs')
        raw.seek(0)
        total = 0
        with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
            for step in range(capture.TURNS):
                line = stream.readline(capture.MAX_LINE_BYTES + 1)
                total += len(line)
                if not line or len(line) > capture.MAX_LINE_BYTES or total > capture.MAX_DECODED_BYTES:
                    raise ValueError('Trace budget differs')
                value = acceptance._json(line, capture.MAX_LINE_BYTES)
                if type(value.get('step')) is not int or value['step'] != step:
                    raise ValueError('Trace turn sequence differs')
                yield value
            if stream.read(1):
                raise ValueError('Trace has extra data')
        if capture._hash_stream(raw) != expected:
            raise ValueError('Trace identity changed')


class LocalDriver:
    """Concrete private-file driver; ``check`` and command come from bootstrap."""
    def __init__(self, *, manifest, game_root, stage, commit, command_for, check,
                 available=None, disk_usage=shutil.disk_usage, supervisor=campaign.supervise_windows):
        if (type(manifest) is not PrivateManifest or type(commit) is not str
                or re.fullmatch(r'[0-9a-f]{40}', commit) is None):
            raise ValueError('Invalid authenticated context')
        self.root, self.stage = worker._root_and_stage(game_root, stage)
        self.manifest, self.commit = manifest, commit
        self.command_for, self.check, self.supervisor = command_for, check, supervisor
        self.available, self.disk_usage = available, disk_usage
        self.claimed = False

    def claim(self):
        worker._registered(self.check)
        if self.claimed:
            raise ValueError('Campaign stage already claimed')
        _no_links(self.root, self.stage)
        created = False
        try:
            self.stage.mkdir(exist_ok=False)
            created = True
            for name in ('agents', 'sources', 'job-inputs', 'job-results', 'games', 'traces'):
                (self.stage / name).mkdir(exist_ok=False)
            for role in ('market', 'round2'):
                relative, digest = self.manifest.opponent(role)
                data = _read(self.root, self.root / relative, MAX_SOURCE)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('Frozen opponent differs')
                path = self.stage / 'sources' / (role + '.py')
                with path.open('xb') as stream:
                    if stream.write(data) != len(data):
                        raise OSError('Opponent copy failed')
                if _read(self.stage, path, MAX_SOURCE) != data:
                    raise ValueError('Opponent copy differs')
            _write_new(self.stage, self.stage / 'campaign-started.local.json',
                       dict(schema='task-pair-dev-campaign-start-v1', registration_commit=self.commit,
                            max_games=24, attempt=1, retry_allowed=False))
            worker._registered(self.check)
            self.claimed = True
            return True
        except BaseException:
            if created:
                try:
                    self.failure(None, 0)
                except BaseException:
                    pass
            raise

    def preflight(self, plan, limits, remaining):
        if self.available is None:
            import psutil
            available = psutil.virtual_memory().available
        else:
            available = self.available()
        required = limits['initial_available_bytes'] if plan is None else limits['between_available_bytes']
        if type(available) is not int or available < required or remaining <= 0:
            return False
        if plan is None:
            # An existing stage may belong to a previous attempt.  Do not
            # inspect its contents or adopt it as this run's output.
            if self.stage.exists() or self.stage.is_symlink():
                return False
            parent, used = self.root / 'artifacts', 0
        else:
            if not self.claimed or not self.stage.is_dir():
                return False
            parent, used = self.stage, _owned_size(self.stage)
        free = self.disk_usage(parent).free
        reserve = limits['per_game_reserve_bytes']
        return (type(free) is int and free >= limits['disk_free_floor_bytes'] + reserve
                and used + reserve <= limits['total_output_bytes'])

    def launch(self, plan, limits):
        self.manifest.panel.validate_plan(plan)
        command = self.command_for(plan['game'])
        if (type(command) is not list or len(command) < 2
                or any(type(value) is not str or not value for value in command)
                or command[-2:] != ['--game', plan['game']]):
            raise ValueError('Invalid registered child command')
        return self.supervisor(command, self.root, limits)

    def preflight_comparison(self, limits, remaining):
        if not self.claimed:
            return False
        if self.available is None:
            import psutil
            available = psutil.virtual_memory().available
        else:
            available = self.available()
        free = self.disk_usage(self.stage).free
        return (type(available) is int and available >= limits['between_available_bytes']
                and type(free) is int and free >= limits['disk_free_floor_bytes'] + MAX_SUMMARY
                and _owned_size(self.stage) + MAX_SUMMARY <= limits['total_output_bytes'] and remaining > 0)

    def read_bundle(self, plan):
        self.manifest.panel.validate_plan(plan)
        return {name: _read(self.stage, self.stage / 'job-results' / (plan['game'] + suffix),
                            capture.MAX_METRICS_BYTES if name == 'metrics' else acceptance.MAX_RECEIPT_BYTES)
                for name, suffix in RECEIPTS.items()}

    def record_supervision(self, plan, supervision):
        self.manifest.panel.validate_plan(plan)
        _write_new(self.stage, self.stage / 'job-results' / (plan['game'] + '.supervision.local.json'),
                   dict(schema='task-pair-dev-job-supervision-v1', game=plan['game'],
                        supervision=supervision, retry_allowed=False), limit=acceptance.MAX_RECEIPT_BYTES)
        return True

    def record_acceptance(self, plan, bundle):
        self.manifest.panel.validate_plan(plan)
        if type(bundle) is not dict or set(bundle) != acceptance.FILES:
            raise ValueError('Incomplete accepted game bundle')
        supervision = self.stage / 'job-results' / (plan['game'] + '.supervision.local.json')
        input_path = self.stage / 'job-inputs' / (plan['game'] + '.json')
        _write_new(self.stage, self.stage / 'job-results' / (plan['game'] + '.acceptance.local.json'),
                   dict(schema='task-pair-dev-job-acceptance-v1', game=plan['game'], passed=True,
                        job_input_sha256=_hash(self.stage, input_path, worker.MAX_JOB_BYTES),
                        supervision_sha256=_hash(self.stage, supervision, acceptance.MAX_RECEIPT_BYTES),
                        receipt_sha256={key: hashlib.sha256(raw).hexdigest() for key, raw in bundle.items()},
                        retry_allowed=False), limit=acceptance.MAX_RECEIPT_BYTES)
        return True

    def verify_files(self, plan, row):
        self.manifest.panel.validate_plan(plan)
        job_raw = _read(self.stage, self.stage / 'job-inputs' / (plan['game'] + '.json'), worker.MAX_JOB_BYTES)
        job = self.manifest.panel.validate_plan(acceptance._json(job_raw, worker.MAX_JOB_BYTES), job=True)
        if any(row.get(key) != value for key, value in job.items()):
            return False
        model, expected = self.manifest.panel.task_model(plan)
        if _hash(self.root, self.root / model, MAX_MODEL) != expected:
            return False
        crop, crop_sha = self.manifest.panel.crop
        if _hash(self.root, self.root / crop, MAX_MODEL) != crop_sha:
            return False
        rival, rival_sha = self.manifest.opponent(plan['opponent'])
        if (_hash(self.root, self.root / rival, MAX_SOURCE) != rival_sha
                or _hash(self.stage, self.stage / 'sources' / (plan['opponent'] + '.py'), MAX_SOURCE) != rival_sha
                or _hash(self.root, self.root / worker.WORKER_RELATIVE, worker.MAX_WORKER_BYTES) != self.manifest.panel.worker_sha
                or _hash(self.stage, self.stage / 'agents' / (plan['game'] + '.py'), MAX_SOURCE) != row['bot_sha256']
                or _hash(self.stage, self.stage / 'traces' / (plan['game'] + '.jsonl.gz'), capture.MAX_COMPRESSED_BYTES) != row['trace_sha256']
                or _hash(self.stage, self.stage / 'games' / plan['game'] / 'replay.json.gz', capture.MAX_COMPRESSED_BYTES) != row['replay_gzip_sha256']):
            return False
        return True

    def compare(self, left, right):
        keys = ('seed', 'seed_alias', 'opponent', 'side')
        if any(left[k] != right[k] for k in keys) or (left['arm'], right['arm']) not in COMPARISON_ARMS:
            raise ValueError('Comparison identity differs')
        for row in (left, right):
            plans = [plan for plan in self.manifest.panel.plans() if plan['game'] == row['game']]
            if (len(plans) != 1 or any(row[key] != plans[0][key] for key in keys)
                    or row['arm'] != plans[0]['policy']):
                raise ValueError('Comparison game is outside the registered panel')
        lrep = _replay(self.stage, self.stage / 'games' / left['game'] / 'replay.json.gz', left['replay_sha256'])
        rrep = _replay(self.stage, self.stage / 'games' / right['game'] / 'replay.json.gz', right['replay_sha256'])
        lt = _trace(self.stage, self.stage / 'traces' / (left['game'] + '.jsonl.gz'), left['trace_sha256'])
        rt = _trace(self.stage, self.stage / 'traces' / (right['game'] + '.jsonl.gz'), right['trace_sha256'])
        try:
            result = compare_replays(lrep, rrep, lt, rt, reference=left['arm'], candidate=right['arm'],
                                     seed=left['seed'], seed_alias=left['seed_alias'],
                                     opponent=left['opponent'], side=left['side'])
            _write_new(self.stage, _comparison_file(self.stage, left['game'], right['arm']),
                       dict(schema='task-pair-dev-comparison-v1', reference_game=left['game'],
                            candidate_game=right['game'],
                            reference_replay_sha256=left['replay_sha256'],
                            candidate_replay_sha256=right['replay_sha256'],
                            reference_trace_sha256=left['trace_sha256'],
                            candidate_trace_sha256=right['trace_sha256'], comparison=result),
                       limit=acceptance.MAX_RECEIPT_BYTES)
            return result
        finally:
            lt.close()
            rt.close()

    def complete(self, rows, summaries):
        worker._registered(self.check)
        plans = self.manifest.panel.plans()
        if (type(rows) is not list or len(rows) != 24
                or {row['game'] for row in rows} != {plan['game'] for plan in plans}
                or type(summaries) is not dict
                or set(summaries) != {'baseline_to_control', 'control_to_auxiliary'}):
            raise ValueError('Incomplete private game ledger')
        receipts = []
        for row in rows:
            key = row['game']
            supervision = self.stage / 'job-results' / (key + '.supervision.local.json')
            acceptance_path = self.stage / 'job-results' / (key + '.acceptance.local.json')
            receipts.append(dict(game=key,
                                 supervision_sha256=_hash(self.stage, supervision, acceptance.MAX_RECEIPT_BYTES),
                                 acceptance_sha256=_hash(self.stage, acceptance_path, acceptance.MAX_RECEIPT_BYTES)))
        comparisons = []
        for reference, candidate in COMPARISON_ARMS:
            for alias in ALIASES:
                for opponent in OPPONENTS:
                    for side in (0, 1):
                        reference_game = f'dev-{alias}-{opponent}-{side}-{reference}'
                        candidate_game = f'dev-{alias}-{opponent}-{side}-{candidate}'
                        path = _comparison_file(self.stage, reference_game, candidate)
                        comparisons.append(dict(reference_game=reference_game, candidate_game=candidate_game,
                                                sha256=_hash(self.stage, path, acceptance.MAX_RECEIPT_BYTES)))
        _write_new(self.stage, self.stage / 'campaign-result.local.json',
                   dict(schema='task-pair-dev-campaign-result-v1', passed=True, games=len(rows),
                        official_replay_passes=len(rows), seed_clusters=2, receipts=receipts,
                        comparison_receipts=comparisons, summaries=summaries,
                        independent_improvement=False, promotion=False))
        worker._registered(self.check)
        return True

    def failure(self, game, completed):
        if self.stage.is_dir():
            _write_new(self.stage, self.stage / 'campaign-failure.local.json',
                       dict(schema='task-pair-dev-campaign-failure-v1', game=game,
                            completed=completed, retry_allowed=False))
