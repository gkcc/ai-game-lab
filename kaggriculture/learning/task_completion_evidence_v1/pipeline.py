"""Single-pass, private evidence extraction from the eight registered games.

The official engine is invoked only by run(), after registration verification.
Raw observations and atomic events never enter output files.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

from learning.task_action_observation_v1.analysis import raw_path, validate_manifest
from learning.task_action_impact_diagnostic_v1.analysis import (
    Meter, compact_replay, descriptor, read_json, trace_lines, validate_trace,
)
from .labels import EpisodeTracker

SOURCE_MANIFEST_SHA = 'b433ee9652f38bbb0cb39afb1e02dcdb3ad84183beabd678f3ecbc8c0570f4fa'
SOURCE = Path(__file__).resolve().parents[2] / 'artifacts/task-action-observation-v1-20261027'
ROW_FIELDS = frozenset(('game', 'side', 'first_step', 'last_step', 'first_ordinal',
                        'unit', 'target', 'op', 'chosen', 'status', 'reason',
                        'observed_steps', 'had_blocked', 'reason_counts'))


def select_games(source_manifest, selected):
    """Bind exact selected metadata to the previously accepted 24-game manifest."""
    indexed, _ = validate_manifest(SOURCE, source_manifest)
    games = list(indexed.values())
    by_key = {game['key']: game for game in games}
    if type(selected) is not list or len(selected) != 8:
        raise ValueError('Exactly eight fixed games required')
    keys = [item if isinstance(item, str) else item['key'] for item in selected]
    if len(set(keys)) != 8 or any(k not in by_key for k in keys):
        raise ValueError('Unknown or duplicate game selection')
    result = [by_key[k] for k in keys]
    for spec, game in zip(selected, result):
        if isinstance(spec, dict) and spec != game:
            raise ValueError('Selected game metadata differs from accepted manifest')
    if (len({g['seed'] for g in result}) != 2
            or {g['opponent'] for g in result} != {'round2', 'market'}
            or {g['side'] for g in result} != {0, 1}
            or {g['arm'] for g in result} != {'baseline', 'candidate', 'teacher_task'}):
        raise ValueError('Selected games do not cover registered dimensions')
    paths = [raw_path(SOURCE, g[k], k) for g in result for k in ('replay', 'trace')]
    if len(set(paths)) != 16:
        raise ValueError('Expected sixteen distinct raw inputs')
    return result


def selections_from_trace(stream, compact, check):
    """Discard callback inputs immediately after verification, before reexecution."""
    selections, desc = [], descriptor()
    for step in range(719):
        check()
        try:
            trace = next(stream)
        except StopIteration as exc:
            raise ValueError('Trace ended before 719 turns') from exc
        records = validate_trace(trace, compact, step, desc)
        selections.append([dict(step=step, ordinal=i, unit=r['unit'], target=r['target'],
                                op=r['op'], chosen=r['chosen'], unit_request=r['unit_request'])
                           for i, r in enumerate(records)])
    try:
        next(stream)
    except StopIteration:
        return selections
    raise ValueError('Trace contains extra turns')


class EventJoin:
    """Keep only one transition's actor events until its next transition."""
    def __init__(self, tracker, selections, days, side, check):
        self.tracker, self.selections, self.days = tracker, selections, days
        self.side, self.check, self.next_step, self.events = side, check, 0, []

    def flush_until(self, end):
        while self.next_step < end:
            self.check()
            self.tracker.step(self.next_step, self.selections[self.next_step],
                              self.events, self.days[self.next_step])
            self.events = []
            self.next_step += 1

    def sink(self, event):
        self.check()
        step = event['step']
        if type(step) is not int or not self.next_step <= step < 719:
            raise ValueError('Official event step out of order or range')
        self.flush_until(step)
        if event.get('phase') == 'unit' and event.get('player') == self.side:
            self.events.append(event)

    def finish(self):
        self.flush_until(719)
        return self.tracker.finish()


def validate_rows(rows, game, selection_count=None):
    if not isinstance(rows, list):
        raise ValueError('Invalid tracker output')
    identifiers = set()
    for row in rows:
        if (set(row) != ROW_FIELDS or row['game'] != game['key'] or row['side'] != game['side']
                or row['status'] not in ('completed_observed', 'unknown')
                or type(row['observed_steps']) is not int or row['observed_steps'] <= 0):
            raise ValueError('Unsafe or invalid tracker row')
        identity = (row['first_step'], row['first_ordinal'])
        if identity in identifiers:
            raise ValueError('Duplicate episode origin')
        identifiers.add(identity)
    if selection_count is not None and sum(row['observed_steps'] for row in rows) != selection_count:
        raise ValueError('Selection-to-episode count mismatch')
    return rows


def run(out):
    """Run once; per-game artifacts are immutable, and partial failures are retained."""
    from .experiment import read, sha, write_new, verify, ready, deadline_check
    from decision.full_audit import reexecute
    out = Path(out).resolve()
    cfg = verify(out)
    ready(out)
    limits = cfg['analysis_limits']
    meter = Meter(compressed_limit=min(limits['compressed_bytes'], 256 * 2**20),
                  decoded_limit=min(limits['decoded_bytes'], 4 * 2**30))
    completed, transitions, dataset, output_bytes = [], 0, [], 0
    started, cpu_started = time.monotonic(), time.process_time()
    join = None
    def check():
        deadline_check()
        if time.monotonic() - started > limits['work_seconds']:
            raise TimeoutError('Analysis wall limit exceeded')
        if time.process_time() - cpu_started > limits['cpu_seconds']:
            raise TimeoutError('Analysis CPU limit exceeded')
    try:
        old_manifest_path = SOURCE / 'input-manifest.json'
        if sha(old_manifest_path) != SOURCE_MANIFEST_SHA:
            raise ValueError('Accepted source manifest hash mismatch')
        original = read(old_manifest_path)
        selected_manifest = read(out / 'input-manifest.json')
        if (selected_manifest.get('schema') != 'task-completion-evidence-input-v1'
                or selected_manifest.get('source_manifest_sha256') != SOURCE_MANIFEST_SHA
                or selected_manifest.get('games_count') != 8 or selected_manifest.get('raw_files') != 16):
            raise ValueError('Invalid registered input manifest')
        games = select_games(original, selected_manifest['games'])
        ledger_path = SOURCE / 'ledger.json'
        if sha(ledger_path) != original['ledger_sha256']:
            raise ValueError('Accepted source ledger hash mismatch')
        ledger = {r['game']: r for r in read(ledger_path)}
        if len(ledger) != 24:
            raise ValueError('Invalid accepted source ledger')
        all_counts = Counter()
        (out / 'private' / 'games').mkdir(parents=True, exist_ok=True)
        (out / 'game-results').mkdir(parents=True, exist_ok=True)
        for game in games:
            check()
            row = ledger[game['key']]
            if (not row['passed'] or row['replay_gzip_sha256'] != game['replay']['sha256']
                    or row['trace_sha256'] != game['trace']['sha256']):
                raise ValueError('Source ledger identity mismatch')
            replay = read_json(raw_path(SOURCE, game['replay'], 'replay'), game['replay'], meter)
            compact = compact_replay(replay, game)
            stream = trace_lines(raw_path(SOURCE, game['trace'], 'trace'), game['trace'], meter)
            try:
                selections = selections_from_trace(stream, compact, check)
            finally:
                stream.close()
            join = EventJoin(EpisodeTracker(game['key'], game['side']), selections,
                             compact['days'], game['side'], check)
            audit = reexecute(replay, trace=True, label_sink=join.sink)
            transitions += audit['official_transitions_reexecuted']
            if (not audit['passed'] or audit['official_transitions_reexecuted'] != 719
                    or audit['semantic_sha256_v2'] != row['semantic_replay_sha256_v2']
                    or audit['engine_sha256'] != row['engine_sha256']):
                raise ValueError('Official audit differs from accepted ledger')
            labels = validate_rows(join.finish(), game, sum(map(len, selections)))
            counts = Counter((r['op'], r['status'], r['reason']) for r in labels)
            all_counts.update(counts)
            payload = b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':')).encode('utf-8') + b'\n' for r in labels)
            compressed = gzip.compress(payload, mtime=0)
            label_sha = hashlib.sha256(compressed).hexdigest()
            report = dict(game=game['key'], side=game['side'], rows=len(labels),
                          counts=[dict(op=k[0], status=k[1], reason=k[2], count=v)
                                  for k, v in sorted(counts.items())],
                          unknown_step_reasons=dict(sorted(sum(
                              (Counter(r['reason_counts']) for r in labels), Counter()).items())),
                          semantic_sha256_v2=audit['semantic_sha256_v2'],
                          ordered_effect_sha256=audit['execution']['ordered_effect_sha256'],
                          replay_sha256=game['replay']['sha256'], trace_sha256=game['trace']['sha256'],
                          labels_sha256=label_sha,
                          official_transitions=719)
            # Reserve ample bounded space for JSON formatting and final manifests.
            output_bytes += len(compressed) + len(json.dumps(report).encode()) + 4096
            if output_bytes + 2 * 2**20 > limits['output_bytes']:
                raise ValueError('Output budget exceeded')
            check()
            relative = 'private/games/' + game['key'] + '.episodes.jsonl.gz'
            with (out / relative).open('xb') as file:
                file.write(compressed)
            write_new(out / 'game-results' / (game['key'] + '.json'), report)
            dataset.append(dict(game=game['key'], path=relative, sha256=label_sha, rows=len(labels)))
            completed.append(report)
            del replay, compact, selections, join, labels, payload, compressed
            join = None
        if len(completed) != 8 or len(meter.receipts) != 16 or transitions != 5752:
            raise ValueError('Incomplete registered campaign')
        total = sum(all_counts.values())
        observed = sum(n for (_, status, _), n in all_counts.items() if status == 'completed_observed')
        check()
        write_new(out / 'game-ledger.json', completed)
        write_new(out / 'dataset-manifest.json', dict(games=dataset, complete=True,
                  join_keys=['game', 'first_step', 'first_ordinal'], private=True))
        result = dict(complete=True, games=8, raw_files=16, official_transitions=transitions,
                      episodes=total, observed=observed, unknown=total-observed,
                      coverage=observed / total if total else None, raw_reads=meter.snapshot(),
                      actor_projection_checked=True,
                      wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
                      memory_limit_kind='sampled externally; not a hard limit', gpu_seconds=0,
                      new_games=0, training_updates=0,
                      counts=[dict(op=k[0], status=k[1], reason=k[2], count=v)
                              for k, v in sorted(all_counts.items())],
                      scope='development feasibility only; no causal or statistical improvement claim')
        write_new(out / 'analysis-result.json', result)
        return result
    except BaseException as exc:
        write_new(out / 'failure.local.json', dict(error_type=type(exc).__name__,
                  message=str(exc)[:500], completed_games=len(completed),
                  completed_official_transitions=transitions, raw_reads=meter.snapshot(),
                  current_event_step_lower_bound=join.next_step if join is not None else None,
                  attempt_consumed=True, no_retry=True))
        raise
