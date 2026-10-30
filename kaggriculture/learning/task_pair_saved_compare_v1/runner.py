"""Inspect one accepted, previously consumed game pair without replaying it.

The output is descriptive and post hoc. The old diagnostic entry is never
called. Paths, seeds, hashes, private observations and requests stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
GAME_CODE = REPO / 'kaggriculture'
sys.path.insert(0, str(GAME_CODE))

from learning.task_pair_dev_campaign_v1 import acceptance, driver  # noqa: E402
from learning.task_pair_dev_games_v1.panel import _JOB, _digest, STORAGE  # noqa: E402
from learning.task_pair_dev_games_v1.worker import NATIVE_OUTPUT_SCOPE  # noqa: E402
from learning.task_pair_dev_metrics_v1.replay_pair import _replay  # noqa: E402
from learning.task_pair_saved_compare_v1.analysis import compare  # noqa: E402
import psutil  # noqa: E402


ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
ORIGINAL_COMMIT = 'e458efa11b4320b1d8982abea34dba3a2bf377d2'
ARMS = ('baseline', 'control')
INITIAL_MEMORY = 4 * 1024**3
DURING_MEMORY = 3 * 1024**3
MAX_SECONDS = 600
MAX_STAGE_BYTES = 2 * 1024**3
ORIGINAL_STAGE = 'task-pair-historical-compare-v1-20261030'
OUTPUT_STAGE = 'task-pair-saved-compare-v1-20261031'


def require(condition):
    if not condition:
        raise ValueError('Verification failed')


def git(*arguments):
    return subprocess.check_output(['git', '-C', str(REPO), *arguments],
                                   stderr=subprocess.DEVNULL, timeout=30).decode().strip()


def require_checkout(commit):
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit))
    require(git('rev-parse', 'HEAD') == commit)
    require(not git('status', '--porcelain', '--untracked-files=all'))


def read_json(path, maximum=1024 * 1024):
    path = Path(path)
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode)
            and not getattr(s, 'st_file_attributes', 0) & 0x400 and 0 < s.st_size <= maximum)
    raw = path.read_bytes()
    require(len(raw) == s.st_size)
    return raw, acceptance._json(raw, maximum)


class Guard:
    """Sample available physical memory; not a hard asynchronous memory cap."""
    def __init__(self):
        self.start = time.monotonic()
        self.minimum = None
        self.last = None

    def sample(self, initial=False):
        if time.monotonic() - self.start >= MAX_SECONDS:
            raise TimeoutError('Diagnostic time limit')
        available = psutil.virtual_memory().available
        if type(available) is not int:
            raise ValueError('Resource sampling failed')
        self.last = available
        self.minimum = available if self.minimum is None else min(self.minimum, available)
        if available < (INITIAL_MEMORY if initial else DURING_MEMORY):
            raise MemoryError('Diagnostic memory limit')
        return True


def hash_file(root, path, metadata, check):
    driver._no_links(root, path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and type(metadata) is dict
            and type(metadata.get('bytes')) is int and metadata['bytes'] == before.st_size
            and type(metadata.get('sha256')) is str and len(metadata['sha256']) == 64)
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while True:
            check()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            require(count <= before.st_size)
            digest.update(chunk)
    after = path.lstat()
    require(count == before.st_size and (before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns) and digest.hexdigest() == metadata['sha256'])


def snapshot(stage, evidence, check):
    """Recheck the original 79 files and both sibling markers, read only."""
    require(stage.is_dir() and stage.resolve(strict=True) == stage)
    driver._no_links(stage, stage)
    files = evidence['stage_files']
    require(type(files) is dict and len(files) == 79
            and sum(v['bytes'] for v in files.values()) <= MAX_STAGE_BYTES)
    found = set()
    def failed(_error):
        raise OSError('Original stage traversal failed')
    for parent, dirs, names in os.walk(stage, followlinks=False, onerror=failed):
        parent = Path(parent)
        for name in (*dirs, *names):
            path = parent / name
            driver._no_links(stage, path)
            info = path.lstat()
            require(stat.S_ISDIR(info.st_mode) if name in dirs else stat.S_ISREG(info.st_mode))
            if name in names:
                found.add(path.relative_to(stage).as_posix())
    require(found == set(files))
    for relative in sorted(found):
        hash_file(stage, stage.joinpath(*relative.split('/')), files[relative], check)
    for name in ('attempt', 'failure'):
        path = stage.with_name(stage.name + '.' + name + '.local.json')
        hash_file(stage.parent, path, evidence[name], check)


def accepted(stage, arm):
    game = 'dev-S02-market-0-' + arm
    base = stage / 'job-results'
    _, accepted_receipt = read_json(base / (game + '.acceptance.local.json'))
    raw_s, supervision = read_json(base / (game + '.supervision.local.json'))
    require(accepted_receipt.get('schema') == 'task-pair-dev-job-acceptance-v1'
            and accepted_receipt.get('passed') is True and accepted_receipt.get('game') == game
            and accepted_receipt.get('retry_allowed') is False
            and accepted_receipt.get('supervision_sha256') == hashlib.sha256(raw_s).hexdigest()
            and supervision.get('game') == game and acceptance.supervision_ok(supervision.get('supervision')))
    job_raw, job = read_json(stage / 'job-inputs' / (game + '.json'))
    require(accepted_receipt.get('job_input_sha256') == hashlib.sha256(job_raw).hexdigest())
    suffixes = driver.RECEIPTS
    require(set(accepted_receipt.get('receipt_sha256', {})) == set(suffixes))
    bundle, raw_bundle = {}, {}
    for name, suffix in suffixes.items():
        raw, parsed = read_json(base / (game + suffix), maximum=8 * 1024 * 1024)
        require(accepted_receipt['receipt_sha256'][name] == hashlib.sha256(raw).hexdigest())
        bundle[name] = parsed
        raw_bundle[name] = raw
    row, audit, complete, metrics, receipt = (bundle[n] for n in
                                              ('row', 'audit', 'complete', 'metrics', 'metrics_receipt'))
    require(row.get('game') == game and row.get('policy') == arm and row.get('seed_alias') == 'S02'
            and row.get('opponent') == 'market' and type(row.get('side')) is int and row['side'] == 0
            and type(row.get('seed')) is int and row.get('passed') is True
            and row.get('engine_sha256') == ENGINE_SHA and row.get('states') == 720
            and row.get('action_turns') == 719 and row.get('status') == ['DONE', 'DONE']
            and row.get('agent_errors') == 0 and row.get('rewards') == row.get('banks'))
    require(set(job) == _JOB and set(job).issubset(row) and all(job[key] == row[key] for key in job))
    for field in ('bot_sha256', 'rival_sha256', 'task_checkpoint_sha256', 'crop_checkpoint_sha256',
                  'trace_sha256', 'replay_gzip_sha256', 'semantic_replay_sha256',
                  'semantic_replay_sha256_v2'):
        _digest(row.get(field))
    official = audit.get('official')
    require(type(official) is dict and audit.get('passed') is True and audit.get('errors') == 0
            and audit.get('game') == game and audit.get('replay_sha256') == row.get('replay_gzip_sha256')
            and audit.get('trace_sha256') == row.get('trace_sha256') and official.get('passed') is True
            and all(audit.get(k) == 719 for k in
                    ('actor_actions_reproduced', 'rival_actions_reproduced', 'trace_turns_reproduced'))
            and official.get('engine_sha256') == ENGINE_SHA and official.get('official_transitions_reexecuted') == 719
            and official.get('both_public_and_private_states_match_each_step') is True
            and official.get('inventory_order_checked_each_step') is True
            and official.get('semantic_sha256') == row.get('semantic_replay_sha256')
            and official.get('semantic_sha256_v2') == row.get('semantic_replay_sha256_v2')
            and official.get('final_rewards') == row['banks']
            and complete.get('passed') is True and complete.get('game') == game
            and complete.get('row_sha256') == hashlib.sha256(raw_bundle['row']).hexdigest()
            and complete.get('certificate_sha256') == hashlib.sha256(raw_bundle['audit']).hexdigest()
            and receipt.get('schema') == 'task-pair-dev-game-metrics-receipt-v1'
            and receipt.get('official_replay_passes') == 1 and receipt.get('game') == game
            and receipt.get('metrics_bytes') == len(raw_bundle['metrics'])
            and receipt.get('metrics_sha256') == hashlib.sha256(raw_bundle['metrics']).hexdigest()
            and metrics.get('game') == game and metrics.get('seed') == row['seed']
            and metrics.get('arm') == arm and metrics.get('side') == 0
            and metrics.get('task_checkpoint_sha256') == row.get('task_checkpoint_sha256')
            and metrics.get('crop_checkpoint_sha256') == row.get('crop_checkpoint_sha256')
            and metrics.get('replay_sha256') == row.get('replay_gzip_sha256')
            and metrics.get('trace_sha256') == row.get('trace_sha256'))
    counts = audit.get('primary_model_counts')
    require(type(counts) is dict and counts == audit.get('live_model_counts')
            and type(counts.get('task_forwards')) is int and counts['task_forwards'] > 0
            and counts.get('controller_calls') == 719)
    storage = bundle['storage']
    out = storage.get('output')
    require(storage.get('passed') is True and storage.get('game') == game
            and storage.get('error_kind') is None and storage.get('native_output_accounted_by') == NATIVE_OUTPUT_SCOPE
            and type(out) is dict and type(out.get('limit_bytes')) is int
            and out['limit_bytes'] == STORAGE['game_write_bytes']
            and type(out.get('used_bytes')) is int and 0 <= out['used_bytes'] <= out['limit_bytes']
            and type(out.get('remaining_bytes')) is int
            and out['remaining_bytes'] == out['limit_bytes'] - out['used_bytes']
            and type(out.get('refused')) is int and out['refused'] == 0)
    return row


def economics(row, replay):
    frames = _replay(replay, row['seed'])
    initial = [farm['money'] for farm in frames[0][0]['observation']['farms']]
    final = [farm['money'] for farm in frames[-1][0]['observation']['farms']]
    require(initial == row['initial_banks'] and final == row['banks']
            and [p.get('reward') for p in frames[-1]] == final
            and row['net_profit'] == final[0] - initial[0]
            and row['opponent_net_profit'] == final[1] - initial[1]
            and row['margin'] == final[0] - final[1]
            and row['score'] == (1 if final[0] > final[1] else .5 if final[0] == final[1] else 0))
    return dict(own_net_profit=row['net_profit'], opponent_net_profit=row['opponent_net_profit'],
                margin=row['margin'], score=row['score'])


def classify(exc):
    if isinstance(exc, MemoryError):
        return 'memory_gate'
    if isinstance(exc, TimeoutError):
        return 'time_gate'
    if isinstance(exc, FileNotFoundError):
        return 'missing_file'
    if isinstance(exc, OSError):
        return 'io_error'
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError)):
        return 'invalid_or_changed_input'
    return 'other_exception'


def output_layout(stage, output):
    """Require a new sibling stage and a fixed output name, without aliases."""
    require(stage.is_absolute() and output.is_absolute() and stage.name == ORIGINAL_STAGE
            and output == stage.parent / OUTPUT_STAGE / 'result.local.json')


def prepare_output(stage, output):
    output_layout(stage, output)
    require(stage.is_dir() and stage.resolve(strict=True) == stage
            and stage.parent.resolve(strict=True) == stage.parent)
    directory = output.parent
    driver._no_links(stage.parent, directory)
    require(not os.path.lexists(directory))
    directory.mkdir(exist_ok=False)
    driver._no_links(stage.parent, directory)
    require(directory.is_dir() and directory.resolve(strict=True) == directory)


def run(stage, evidence_path, binding_path, output, commit):
    phase, guard = 'preflight', Guard()
    try:
        guard.sample(initial=True)
        require_checkout(commit)
        stage, output = Path(stage), Path(output)
        prepare_output(stage, output)
        phase = 'provenance'
        evidence_raw, evidence = read_json(evidence_path)
        binding_raw, binding = read_json(binding_path)
        require(binding.get('schema') == 'task-pair-partial-diagnostic-prebinding-v1'
                and binding.get('failure_evidence_sha256') == hashlib.sha256(evidence_raw).hexdigest()
                and evidence.get('schema') == 'task-pair-historical-compare-failure-private-evidence-v1'
                and evidence.get('registration_commit') == ORIGINAL_COMMIT
                and evidence.get('retry_allowed') is False and evidence.get('recorded_after_exit') is True)
        snapshot(stage, evidence, guard.sample)
        rows = {arm: accepted(stage, arm) for arm in ARMS}
        require(rows['baseline']['seed'] == rows['control']['seed']
                and rows['baseline']['crop_checkpoint_sha256'] == rows['control']['crop_checkpoint_sha256']
                and rows['baseline']['task_checkpoint_sha256'] != rows['control']['task_checkpoint_sha256']
                and rows['baseline']['rival_sha256'] == rows['control']['rival_sha256'])
        phase = 'replay_read'
        replays, results, traces = {}, {}, {}
        for arm in ARMS:
            guard.sample()
            game = rows[arm]['game']
            replays[arm] = driver._replay(stage, stage / 'games' / game / 'replay.json.gz',
                                          rows[arm]['replay_gzip_sha256'])
            guard.sample()
            results[arm] = economics(rows[arm], replays[arm])
            traces[arm] = driver._trace(stage, stage / 'traces' / (game + '.jsonl.gz'),
                                        rows[arm]['trace_sha256'])
        phase = 'comparison'
        try:
            comparison = compare(replays['baseline'], replays['control'], traces['baseline'], traces['control'],
                                 seed=rows['baseline']['seed'], side=0, check=guard.sample)
        finally:
            for trace in traces.values():
                trace.close()
        del replays, traces
        phase = 'postcheck'
        guard.sample()
        snapshot(stage, evidence, guard.sample)
        require(read_json(evidence_path)[0] == evidence_raw and read_json(binding_path)[0] == binding_raw)
        require_checkout(commit)
        delta = {key: results['control'][key] - results['baseline'][key] for key in results['baseline']}
        result = dict(schema='task-pair-saved-compare-v1', scope='consumed_development_posthoc',
                      historical_seed_role='S01', opponent='market', side=0, paired_games=1,
                      old_stage_files_rechecked=79, reexecution_this_run=0, new_games=0,
                      training_updates=0, baseline=results['baseline'], control=results['control'],
                      control_minus_baseline=delta, comparison=comparison,
                      independently_verified_improvement=False, statistical_significance='unknown')
        phase = 'output'
        raw = (json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True) + '\n').encode()
        require(len(raw) <= 65536)
        with output.open('xb') as stream:
            require(stream.write(raw) == len(raw))
            stream.flush()
            os.fsync(stream.fileno())
        require(output.read_bytes() == raw)
        print(json.dumps(dict(status='passed', analysis=result,
                              local_resource=dict(minimum_available_bytes=guard.minimum,
                                                  max_process_working_set_bytes=getattr(
                                                      psutil.Process().memory_info(), 'peak_wset', None))),
                         ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='failed', phase=phase, category=classify(exc),
                              last_available_bytes=guard.last), sort_keys=True), file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser()
    for arg in ('stage', 'evidence', 'binding', 'output', 'commit'):
        parser.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    return run(args.stage, args.evidence, args.binding, args.output, args.commit)


if __name__ == '__main__':
    raise SystemExit(main())
