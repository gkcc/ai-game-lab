"""Read-only diagnosis of four already accepted round2 game pairs.

This entry never invokes the official engine, a policy, a model or training.
Only its new, exclusive output directory is written. All identities and raw
game data remain private; errors are reported by fixed category only.
"""
import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time


ENTRY_STARTED = time.monotonic()
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'kaggriculture'))

from learning.task_pair_dev_campaign_v1 import acceptance, driver, manifest
from learning.task_pair_dev_games_v1 import capture, worker
from learning.task_pair_dev_metrics_v1.consumer import _turn
from learning.task_pair_dev_metrics_v1.panel import paired
from learning.task_pair_round2_compare_v1 import projection
from learning.task_action_impact_diagnostic_v1.analysis import compact_replay
from learning.task_pair_action_flow_v1 import analysis


SOURCE_MAP = 'review/evidence/task-pair-action-flow-v1-20261031/source-map.json'
REGISTRATION = 'review/evidence/task-pair-action-flow-v1-20261031/REGISTRATION.zh-CN.md'
OLD_RESULT_DIR = 'review/evidence/task-pair-round2-compare-v1-result-20261031/'
OLD_RESULT_FILES = tuple(OLD_RESULT_DIR + name for name in
                         ('ledger.json', 'RESULTS.zh-CN.md', 'REVIEW.zh-CN.md'))
BASE_COMMIT = '879b2a21dcb7f9e5113e66d5208502307d58eb1b'
OLD_COMMIT = '7ab3f32532b29443ea2b97fd9e7b2d04dd651a43'
OLD_STAGE = 'task-pair-round2-compare-v1-20261031'
NEW_STAGE = 'task-pair-action-flow-v1-20261031'
EXIT_RECORD = OLD_STAGE + '-campaign-exit.local.txt'
INITIAL_MEMORY = 5 * 1024**3
RUN_MEMORY = 4 * 1024**3
MAX_SECONDS = 1800
MAX_SMALL = 8 * 1024 * 1024
ALIASES = {'S02': 'S01', 'S03': 'S04'}
ARMS = ('baseline', 'control')
STATUSES = ('executed', 'partial', 'no_effect', 'blocked', 'pass', 'unknown')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
REQUIRED_SOURCES = frozenset({
    'kaggriculture/learning/task_pair_action_flow_v1/runner.py',
    'kaggriculture/learning/task_pair_action_flow_v1/analysis.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/manifest.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/driver.py',
    'kaggriculture/learning/task_pair_dev_campaign_v1/acceptance.py',
    'kaggriculture/learning/task_action_impact_diagnostic_v1/analysis.py',
    'kaggriculture/learning/task_pair_round2_compare_v1/projection.py',
})


def require(value):
    if not value:
        raise ValueError('Input validation failed')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jread(root, path, maximum=MAX_SMALL):
    raw = driver._read(root, path, maximum)
    return raw, acceptance._json(raw, maximum)


def git(*args):
    process = subprocess.run(['git', '-C', str(REPO), *args], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=30, check=False)
    if process.returncode != 0:
        raise ValueError('Git verification failed')
    return process.stdout


def clean_head(commit):
    require(type(commit) is str and COMMIT.fullmatch(commit) is not None)
    require(git('rev-parse', 'HEAD').strip().decode('ascii') == commit)
    require(not git('status', '--porcelain', '--untracked-files=all'))
    git('merge-base', '--is-ancestor', BASE_COMMIT, commit)
    git('merge-base', '--is-ancestor', OLD_COMMIT, BASE_COMMIT)
    return True


def public_file(relative, commit):
    require(type(relative) is str and relative and '\\' not in relative and ':' not in relative)
    posix = PurePosixPath(relative)
    require(not posix.is_absolute() and posix.as_posix() == relative
            and all(part not in ('.', '..') for part in posix.parts))
    path = REPO.joinpath(*posix.parts)
    raw = driver._read(REPO, path, MAX_SMALL)
    require(raw == git('show', commit + ':' + relative))
    return raw


def verify_sources(commit, check=lambda: True):
    require(check() is True)
    clean_head(commit)
    require(check() is True)
    note = public_file(REGISTRATION, commit)
    require(bool(note))
    source_raw = public_file(SOURCE_MAP, commit)
    source_map = acceptance._json(source_raw, MAX_SMALL)
    require(set(source_map) == {'schema', 'base_commit', 'public_sources'}
            and source_map['schema'] == 'task-pair-action-flow-source-map-v1'
            and source_map['base_commit'] == BASE_COMMIT)
    sources = source_map['public_sources']
    require(type(sources) is dict and REQUIRED_SOURCES.issubset(sources) and SOURCE_MAP not in sources)
    for path, expected in sources.items():
        require(check() is True)
        require(type(expected) is str and SHA.fullmatch(expected) is not None)
        require(sha(public_file(path, commit)) == expected)
        require(check() is True)
    # Every imported project module must be a preregistered source, rather
    # than a local shadow or an unlisted checkout file.
    for name, module in tuple(sys.modules.items()):
        if name == 'learning' or name.startswith('learning.') or name == 'decision' or name.startswith('decision.'):
            filename = getattr(module, '__file__', None)
            if filename is None:  # Namespace packages need no file identity.
                continue
            path = Path(filename).resolve(strict=True)
            require(path.is_relative_to(REPO))
            relative = path.relative_to(REPO).as_posix()
            require(relative in sources and path.suffix == '.py')
    for path in OLD_RESULT_FILES:
        require(check() is True)
        public_file(path, BASE_COMMIT)
    require(check() is True)
    return source_raw, note


class Guard:
    """Sampled checks; individual JSON decode calls cannot be interrupted."""
    def __init__(self, available=None, clock=time.monotonic, started=None):
        if available is None:
            import psutil
            available = lambda: psutil.virtual_memory().available
        self.available, self.clock = available, clock
        self.started = clock() if started is None else started
        self.last_available = None

    def check(self, initial=False):
        if self.clock() - self.started >= MAX_SECONDS:
            raise TimeoutError('Resource time limit')
        value = self.available()
        require(type(value) is int and value >= 0)
        self.last_available = value
        if value < (INITIAL_MEMORY if initial else RUN_MEMORY):
            raise MemoryError('Resource memory floor')
        return True


def _hash_handle(stream, check):
    stream.seek(0)
    total, digest = 0, hashlib.sha256()
    while True:
        check()
        chunk = stream.read(65536)
        if not chunk:
            break
        total += len(chunk)
        require(total <= capture.MAX_COMPRESSED_BYTES)
        digest.update(chunk)
    return digest.hexdigest()


def read_compact(stage, row, side, check):
    """Parse a single replay once; hash the same open handle before/after."""
    path = stage / 'games' / row['game'] / 'replay.json.gz'
    driver._no_links(stage, path)
    require(path.is_file())
    with path.open('rb') as raw:
        require(_hash_handle(raw, check) == row['replay_gzip_sha256'])
        raw.seek(0)
        pieces, size = [], 0
        with gzip.GzipFile(fileobj=raw, mode='rb') as data:
            while True:
                check()
                part = data.read(1024 * 1024)
                if not part:
                    break
                size += len(part)
                require(size <= driver.MAX_REPLAY_DECODED)
                pieces.append(part)
        encoded = b''.join(pieces)
        del pieces
        replay = acceptance._json(encoded, driver.MAX_REPLAY_DECODED)
        del encoded
        require(type(replay.get('info')) is dict and type(replay['info'].get('seed')) is int
                and replay['info']['seed'] == row['seed'])
        compact = compact_replay(replay, {'side': side, 'expected': row})
        del replay
        gc.collect()
        require(_hash_handle(raw, check) == row['replay_gzip_sha256'])
    check()
    return compact


def _op(request):
    return request[0] if type(request) is list and request and type(request[0]) is str and request[0] else 'invalid'


def capture_branch(iterator, branch, player, sink, completeness):
    """Retain just selected task descriptors on one authenticated turn."""
    try:
        for line in iterator:
            if line.get('step') == branch:
                records = _turn(line, branch, player)
                require(not sink)
                for record in records:
                    unit = record['unit']
                    require(unit not in sink)
                    sink[unit] = {'source': record['source'], 'selected_op': record['op'],
                                  'request_op': _op(record['unit_request'])}
            yield line
        completeness['eof'] = True
    finally:
        close = getattr(iterator, 'close', None)
        if callable(close):
            close()


def _events_at(metrics, step, side, selections):
    events = metrics.get('result', {}).get('events')
    require(type(events) is list)
    selected = {}
    for event in events:
        require(type(event) is dict)
        if event.get('step') != step or event.get('player') != side:
            continue
        unit = event.get('unit')
        require(type(unit) is int and unit >= 0 and unit not in selected)
        require(unit in selections)
        for key in ('source', 'selected_op', 'request_op'):
            require(event.get(key) == selections[unit][key])
        support, status = event.get('label_support'), event.get('status')
        require((support == 'unknown' and status is None)
                or (support == 'observed' and status in STATUSES[:-1]))
        selected[unit] = status if support == 'observed' else 'unknown'
    return selected


def execution_summary(reference, candidate, reference_metrics, candidate_metrics,
                      reference_selections, candidate_selections, step, side, market_changed):
    """Count changed unit requests, and only their saved selected-task labels.

    An absent selection/label, or an absent unit on either arm, is unknown.
    Blocked is the official tracer label, not a claim that nothing executed.
    """
    require(type(step) is int and 0 <= step < 719 and type(side) is int and side in (0, 1))
    require(type(market_changed) is bool)
    ref_units, new_units = reference['units'][step], candidate['units'][step]
    require(type(ref_units) is list and type(new_units) is list)
    ref_events = _events_at(reference_metrics, step, side, reference_selections)
    new_events = _events_at(candidate_metrics, step, side, candidate_selections)
    counts = {arm: Counter({status: 0 for status in STATUSES}) for arm in ARMS}
    changed = 0
    for index in range(max(len(ref_units), len(new_units))):
        ref_present, new_present = index < len(ref_units), index < len(new_units)
        if ref_present and new_present and ref_units[index] == new_units[index]:
            continue
        changed += 1
        for arm, present, events in (('baseline', ref_present, ref_events), ('control', new_present, new_events)):
            counts[arm][events.get(index, 'unknown') if present else 'unknown'] += 1
    return dict(changed_unit_requests=changed,
                unit_status={arm: dict(counts[arm]) for arm in ARMS},
                market_request_changed=market_changed,
                market_order_execution_status='unknown_not_saved_per_order',
                interpretation='Saved selected-task status only; missing is unknown, blocked may still execute, and execution is not task completion.')


def _same(left, right):
    return projection._same(left, right)


def read_context(root, campaign_exit, check):
    """Bind the original receipts and public projection without new games."""
    root = Path(root)
    require(root.is_absolute() and root.resolve(strict=True) == root
            and root == (REPO.parent.parent / 'kaggriculture').resolve(strict=True))
    stage = root / 'artifacts' / OLD_STAGE
    driver._no_links(root, stage)
    require(stage.is_dir())
    exit_path = Path(campaign_exit)
    private = REPO.parent / 'private'
    require(exit_path.is_absolute() and exit_path == private / EXIT_RECORD
            and driver._read(private, exit_path, 64).strip() == b'0')
    manifest_raw, doc = jread(stage, stage / 'comparison-manifest.local.json')
    require(set(doc) == {'schema', 'registration_commit', 'prior_binding_sha256', 'actual_seed_roles',
                         'seeds', 'models', 'crop', 'opponents', 'worker_sha256'}
            and doc['schema'] == 'task-pair-historical-private-manifest-v1'
            and doc['registration_commit'] == OLD_COMMIT and doc['actual_seed_roles'] == ALIASES
            and type(doc['prior_binding_sha256']) is str and SHA.fullmatch(doc['prior_binding_sha256']))
    private_manifest = manifest.build_manifest(seeds=doc['seeds'], models=doc['models'], crop=doc['crop'],
                                                opponents=doc['opponents'], worker_sha256=doc['worker_sha256'])
    plans = [p for p in private_manifest.panel.plans()
             if p['opponent'] == 'round2' and p['policy'] in ARMS]
    require(len(plans) == 8 and len({p['game'] for p in plans}) == 8)
    local = driver.LocalDriver(manifest=private_manifest, game_root=root, stage=stage,
                               commit=OLD_COMMIT, command_for=lambda _: None, check=check)
    attempt_path = stage.with_name(stage.name + '.attempt.local.json')
    attempt_raw, attempt = jread(root, attempt_path)
    require(attempt == {'schema': 'task-pair-historical-attempt-v1', 'registration_commit': OLD_COMMIT,
                        'max_games': 8, 'retry_allowed': False})
    _, started = jread(stage, stage / 'campaign-started.local.json')
    require(started == {'schema': 'task-pair-dev-campaign-start-v1', 'registration_commit': OLD_COMMIT,
                        'max_games': 24, 'attempt': 1, 'retry_allowed': False})
    _, returned = jread(stage, stage / 'comparison-returned.local.json')
    require(returned == {'schema': 'task-pair-round2-returned-v1', 'runner_returned': True,
                         'accepted': 8, 'official_replay_passes': 8, 'observed_outer_exit': False})
    _, result = jread(stage, stage / 'comparison-result.local.json')
    require(result.get('schema') == 'task-pair-round2-compare-private-result-v1'
            and result.get('passed') is True and result.get('games') == 8
            and result.get('comparisons') == 4 and result.get('official_replay_passes') == 8
            and result.get('seeds') == doc['seeds'] and result.get('training') is False
            and result.get('independent_improvement') is False and result.get('promotion') is False
            and type(result.get('ledger')) is list and len(result['ledger']) == 8)
    ledger = {item.get('game'): item for item in result['ledger'] if type(item) is dict}
    require(len(ledger) == 8 and set(ledger) == {p['game'] for p in plans})
    public_raw = public_file(OLD_RESULT_DIR + 'ledger.json', BASE_COMMIT)
    projected_raw, public = jread(stage, stage / 'projected-summary.local.json')
    require(projected_raw == public_raw)

    accepted, metric_rows = {}, []
    for plan in plans:
        check()
        game, old_ledger = plan['game'], ledger[plan['game']]
        require(set(old_ledger) == {'game', 'supervision_sha256', 'acceptance_sha256', 'marker_sha256'}
                and type(old_ledger['marker_sha256']) is dict
                and set(old_ledger['marker_sha256']) == {'permit', 'started', 'consumed'})
        bundle = local.read_bundle(plan)
        result_dir = stage / 'job-results'
        supervision_raw, supervision = jread(stage, result_dir / (game + '.supervision.local.json'),
                                             acceptance.MAX_RECEIPT_BYTES)
        accepted_raw, receipt = jread(stage, result_dir / (game + '.acceptance.local.json'),
                                       acceptance.MAX_RECEIPT_BYTES)
        require(sha(supervision_raw) == old_ledger['supervision_sha256']
                and sha(accepted_raw) == old_ledger['acceptance_sha256']
                and supervision.get('schema') == 'task-pair-dev-job-supervision-v1'
                and supervision.get('game') == game and supervision.get('retry_allowed') is False
                and receipt.get('schema') == 'task-pair-dev-job-acceptance-v1'
                and receipt.get('game') == game and receipt.get('passed') is True
                and receipt.get('retry_allowed') is False
                and receipt.get('supervision_sha256') == sha(supervision_raw)
                and receipt.get('receipt_sha256') == {k: sha(v) for k, v in bundle.items()}
                and receipt.get('job_input_sha256') == driver._hash(
                    stage, stage / 'job-inputs' / (game + '.json'), worker.MAX_JOB_BYTES))
        for kind in ('permit', 'started', 'consumed'):
            marker_raw, marker = jread(stage, result_dir / (game + '.' + kind + '.local.json'),
                                       acceptance.MAX_RECEIPT_BYTES)
            expected = {'schema': 'task-pair-historical-' + kind + '-v1', 'game': game,
                        'registration_commit': OLD_COMMIT, 'attempt_sha256': sha(attempt_raw),
                        'retry_allowed': False}
            if kind != 'started':
                expected['manifest_sha256'] = sha(manifest_raw)
            require(marker == expected and sha(marker_raw) == old_ledger['marker_sha256'][kind])
        metrics = acceptance.accept(private_manifest.panel, plan, bundle, supervision.get('supervision'),
                                    local.verify_files)
        check()
        row = acceptance._json(bundle['row'], acceptance.MAX_RECEIPT_BYTES)
        key = (plan['seed_alias'], plan['side'], plan['policy'])
        require(key not in accepted)
        accepted[key] = {'row': row, 'metrics': metrics}
        metric_rows.append(metrics)

    public_pairs = {}
    for pair in public.get('pairs', []):
        require(type(pair) is dict and pair.get('seed_alias') in ALIASES.values()
                and type(pair.get('side')) is int and pair['side'] in (0, 1)
                and pair.get('opponent') == 'round2')
        internal = next(key for key, value in ALIASES.items() if value == pair['seed_alias'])
        key = (internal, pair['side'])
        require(key not in public_pairs)
        public_pairs[key] = pair
    require(set(public_pairs) == {(a, s) for a in ALIASES for s in (0, 1)})
    comparisons, details = [], {}
    for alias in ALIASES:
        for side in (0, 1):
            check()
            left, right = (accepted[alias, side, arm] for arm in ARMS)
            path = driver._comparison_file(stage, left['row']['game'], 'control')
            _, receipt = jread(stage, path, acceptance.MAX_RECEIPT_BYTES)
            require(receipt.get('schema') == 'task-pair-dev-comparison-v1'
                    and receipt.get('reference_game') == left['row']['game']
                    and receipt.get('candidate_game') == right['row']['game']
                    and receipt.get('reference_replay_sha256') == left['row']['replay_gzip_sha256']
                    and receipt.get('candidate_replay_sha256') == right['row']['replay_gzip_sha256']
                    and receipt.get('reference_trace_sha256') == left['row']['trace_sha256']
                    and receipt.get('candidate_trace_sha256') == right['row']['trace_sha256'])
            item = receipt.get('comparison')
            require(type(item) is dict and item.get('seed_alias') == alias
                    and item.get('seed') == doc['seeds'][alias] and item.get('side') == side
                    and item.get('opponent') == 'round2' and item.get('reference') == 'baseline'
                    and item.get('candidate') == 'control')
            comparisons.append(item)
            details[alias, side] = public_pairs[alias, side]['comparison']
    expected_paired = paired(metric_rows, seeds=list(doc['seeds'].values()), opponents=['round2'],
                             reference='baseline', candidate='control', comparisons=comparisons)
    require(_same(expected_paired, result.get('paired')))
    reprojected = projection.project(rows=metric_rows, seeds=doc['seeds'],
                                     comparisons=comparisons, details=details)
    require(_same(reprojected, public))
    return local, stage, accepted, public_pairs


def compare_existing(stage, accepted, public_pairs, check):
    """Load one replay at a time and consume both authenticated traces to EOF."""
    rows = []
    for alias in ALIASES:
        for side in (0, 1):
            check()
            left, right = (accepted[alias, side, arm] for arm in ARMS)
            prefix = public_pairs[alias, side]['comparison']['complete_request_prefix']
            require(prefix.get('scope') == 'same_complete_history' and prefix.get('status') == 'observed'
                    and prefix.get('reason') == 'actor_request_difference'
                    and type(prefix.get('step')) is int and 0 <= prefix['step'] < 719)
            step = prefix['step']
            ref = read_compact(stage, left['row'], side, check)
            new = read_compact(stage, right['row'], side, check)
            raw_ref = driver._trace(stage, stage / 'traces' / (left['row']['game'] + '.jsonl.gz'),
                                    left['row']['trace_sha256'])
            raw_new = driver._trace(stage, stage / 'traces' / (right['row']['game'] + '.jsonl.gz'),
                                    right['row']['trace_sha256'])
            selections = {arm: {} for arm in ARMS}
            eof = {arm: {} for arm in ARMS}
            ref_trace = capture_branch(raw_ref, step, side, selections['baseline'], eof['baseline'])
            new_trace = capture_branch(raw_new, step, side, selections['control'], eof['control'])
            try:
                flow = analysis.compare_pair(ref, new, ref_trace, new_trace, side=side, check=check)
            finally:
                ref_trace.close()
                new_trace.close()
            require(eof['baseline'].get('eof') is True and eof['control'].get('eof') is True
                    and flow.get('scope') == 'v2_semantic_common_history'
                    and flow.get('trace_turns_per_game') == 719
                    and flow.get('first_branch_kind') == 'actor'
                    and flow.get('first_branch_step') == step
                    and flow.get('actor_request_changed_at_first_branch') is True
                    and flow.get('actor_only_branch_attribution') == 'observed')
            components = flow.get('actor_request_components_differ')
            require(type(components) is dict and set(components) == {'farmer', 'hands', 'market'}
                    and all(type(value) is bool for value in components.values()))
            labels = execution_summary(ref, new, left['metrics'], right['metrics'],
                selections['baseline'], selections['control'], step, side, components['market'])
            allowed = ('scope', 'trace_turns_per_game', 'first_branch_kind', 'first_branch_step',
                       'actor_request_changed_at_first_branch', 'actor_only_branch_attribution',
                       'actor_request_components_differ', 'counts', 'first_same_input_choice_difference',
                       'post_branch_windows', 'interpretation')
            require(set(flow) == set(allowed))
            rows.append({'seed_alias': ALIASES[alias], 'side': side, 'opponent': 'round2',
                         'flow': {key: flow[key] for key in allowed}, 'execution_labels': labels})
            del ref, new
            gc.collect()
            check()
            print(json.dumps({'phase': 'comparison', 'completed_pairs': len(rows)}), flush=True)
    require(len(rows) == 4)
    return rows


def _category(exc):
    if isinstance(exc, MemoryError):
        return 'memory_gate'
    if isinstance(exc, TimeoutError):
        return 'time_gate'
    if isinstance(exc, FileNotFoundError):
        return 'missing_file'
    if isinstance(exc, OSError):
        return 'io_error'
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError)):
        return 'invalid_input'
    if isinstance(exc, RuntimeError):
        return 'runtime_error'
    return 'other_exception'


def write_summary(output, result, check):
    """A post-write resource failure must not leave a valid passed summary."""
    path = output / 'summary.local.json'
    require(not path.exists())
    try:
        driver._write_new(output, path, result)
        require(check() is True)
    except BaseException:
        # The stage was exclusively created by this process. If deletion fails,
        # its failure record and nonzero exit explicitly supersede this file.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError('Invalid arguments')


def main(argv=None):
    phase, output, guard, claimed = 'arguments', None, None, False
    try:
        parser = Parser(description='Read-only accepted-round2 comparison')
        parser.add_argument('--game-root', required=True)
        parser.add_argument('--registration-commit', required=True)
        parser.add_argument('--campaign-exit-file', required=True)
        args = parser.parse_args(argv)
        require(sys.flags.isolated and sys.flags.dont_write_bytecode and __debug__)
        phase = 'public_snapshot'
        guard = Guard(started=ENTRY_STARTED)
        guard.check(initial=True)
        source_raw, note_raw = verify_sources(args.registration_commit, guard.check)
        guard.check()
        print(json.dumps({'phase': phase, 'status': 'passed'}), flush=True)
        phase = 'claim'
        root = Path(args.game_root)
        require(root.is_absolute() and root.resolve(strict=True) == root
                and root == (REPO.parent.parent / 'kaggriculture').resolve(strict=True))
        output = root / 'artifacts' / NEW_STAGE
        driver._no_links(root, output)
        output.mkdir(exist_ok=False)
        claimed = True
        phase = 'historical_binding'
        _, stage, accepted, public_pairs = read_context(root, args.campaign_exit_file, guard.check)
        print(json.dumps({'phase': phase, 'accepted_games': len(accepted)}), flush=True)
        phase = 'comparison'
        pairs = compare_existing(stage, accepted, public_pairs, guard.check)
        phase = 'final_snapshot'
        require(verify_sources(args.registration_commit, guard.check) == (source_raw, note_raw))
        guard.check()
        result = {'schema': 'task-pair-action-flow-v1', 'status': 'passed',
                  'registration_commit': args.registration_commit,
                  'public_source_map_sha256': sha(source_raw),
                  'scope': 'consumed_development_only', 'games': 8, 'pairs': pairs,
                  'original_official_replay_receipts_checked': 8, 'new_official_replays': 0,
                  'new_games': 0, 'training_updates': 0, 'independent_improvement': False,
                  'promotion': False, 'resource_guard': 'sampled_not_hard_realtime',
                  'interpretation': 'Descriptive trajectory association; no causal or independent improvement claim.'}
        write_summary(output, result, guard.check)
        print(json.dumps({'phase': 'complete', 'games': 8, 'pairs': 4, 'status': 'passed'}), flush=True)
        return 0
    except BaseException as exc:
        failure = {'schema': 'task-pair-action-flow-failure-v1', 'status': 'failed',
                   'phase': phase, 'category': _category(exc), 'retry_allowed': False,
                   'supersedes_summary': True}
        if claimed and output is not None and output.is_dir():
            try:
                driver._write_new(output, output / 'failure.local.json', failure)
            except BaseException:
                pass
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
