"""Read a fixed, already accepted panel subset. No simulation or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

GAME_ROOT = Path(__file__).resolve().parents[2]
REPO = GAME_ROOT.parent
if str(GAME_ROOT) not in sys.path:
    sys.path.insert(0, str(GAME_ROOT))

from learning.task_action_impact_diagnostic_v1 import analysis
from learning.task_prefix_diagnostic_v1.pairing import PairComparison

LEDGER_SHA256 = '481d4bc82e3e705c0c4750d2dd8f984098e65eef334231d1225169c43bd459ba'
REGISTRATION_SHA256 = '293478899d30f47830add190e699b3077d0466806d2276cee0a4828b822bb245'
FREEZE_SHA256 = '249d72b0770bb97cb4a1cc66fc14586dd674c778185f6aebf8c50c73fad9cb4d'
PROJECTOR_SHA256 = '6409fe65dfa5f332edc52ee1d2edccf8694a08f27be20de9a3d460b0ef88a9b3'
RESULT_COMMIT = '3a3e3172b0403a171a38f6e78e37d07f387be2cb'
REGISTRATION = REPO/'review/evidence/task-teacher-merged-eval-v3-formal-20261030/registration.json'
LEDGER = REPO/'review/evidence/task-teacher-merged-eval-v3-formal-result-20261030/alias-ledger.json'
OPPONENTS = ('round2', 'market', 'teacher')
ARMS = ('merged', 'unweighted', 'teacher_task')
COMPRESSED_LIMIT, DECODED_LIMIT, WALL_SECONDS = 512*2**20, 16*2**30, 900


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lf_sha(data):
    return sha(data.replace(b'\r\n', b'\n'))


def read_small(path):
    with path.open('rb') as stream:
        raw = stream.read(4*2**20+1)
    if len(raw) > 4*2**20:
        raise ValueError('Metadata size limit')
    return raw


def load_small(path, expected=None, normalized=False):
    raw = read_small(path)
    if expected is not None and (lf_sha(raw) if normalized else sha(raw)) != expected:
        raise ValueError('Metadata identity mismatch')
    return json.loads(raw.decode('utf-8-sig'))


def safe_file(root, *parts):
    path = root.joinpath(*parts)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Panel input missing or outside panel')
    return path


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_deadline(started, clock=time.monotonic):
    """Sample elapsed wall time; this is not an OS-enforced hard deadline."""
    require(clock()-started <= WALL_SECONDS, 'Sampled analysis deadline exceeded')


def equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def verify_completion(panel, status_path):
    """Screen existing program receipts; they are not an independent replay."""
    status = load_small(status_path)
    require(status.get('state') == 'accepted_and_published' and status.get('result_commit') == RESULT_COMMIT
            and status.get('remote_main_verified') is True and status.get('process_exit_code') == 0,
            'Local panel completion status not accepted')
    data = read_small(REPO/'tools/panel_v3_alias_ledger.py')
    require(lf_sha(data) == PROJECTOR_SHA256, 'Completion checker identity differs')
    projector = types.ModuleType('historical_panel_completion_checker')
    exec(compile(data, 'historical-panel-completion-checker', 'exec', dont_inherit=True), projector.__dict__)
    files = (('games-result.json',), ('independent-runner-result.json',),
             ('independent-replay-verification.json',), ('results', 'coordinator-complete.json'),
             ('replays', 'launcher-result.json'))
    values = [load_small(safe_file(panel, *parts)) for parts in files]
    projector.require_non_economic_completion(*values, status['process_exit_code'])
    return files


def check_replay_seed(replay, row):
    require(isinstance(replay, dict) and isinstance(replay.get('info'), dict)
            and type(row.get('seed')) is int and type(replay['info'].get('seed')) is int
            and replay['info']['seed'] == row['seed'], 'Replay seed differs from accepted row')


def check_trace_identity(trace, row):
    require(isinstance(trace, dict) and 'checkpoint_sha256' in trace and 'task_checkpoint_sha256' in trace
            and trace['checkpoint_sha256'] == row['crop_checkpoint_sha256']
            and trace['task_checkpoint_sha256'] == row['task_checkpoint_sha256'], 'Trace model identity differs')


def check_counts(counts, limits, policy):
    fields = {'task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
              'gate_evaluations', 'controller_calls', 'teacher_proposals'}
    require(isinstance(counts, dict) and set(counts) == fields
            and all(type(v) is int and v >= 0 for v in counts.values())
            and counts['controller_calls'] == 719
            and counts['controller_calls'] <= limits['controller_calls']
            and counts['task_forwards'] <= limits['task_forwards']
            and counts['task_rows'] >= counts['task_forwards']
            and counts['crop_forwards'] <= limits['crop_forwards']
            and counts['crop_rows'] >= counts['crop_forwards']
            and counts['task_rows'] + counts['crop_rows'] <= limits['neural_input_rows']
            and counts['gate_evaluations'] <= limits['gate_evaluations']
            and counts['teacher_proposals'] <= limits['teacher_proposals'], 'Recorded model budget invalid')
    require((policy == 'teacher_task' and counts['task_forwards'] == counts['task_rows'] == 0)
            or (policy != 'teacher_task' and counts['task_forwards'] > 0), 'Recorded task model invalid')


def count_sources(arm, records, counts):
    require(arm in ARMS and isinstance(records, list), 'Invalid task selection records')
    expected = 'teacher' if arm == 'teacher_task' else 'callback'
    require(all(isinstance(r, dict) and r.get('source') == expected for r in records),
            'Task selection source does not match policy')
    counts['selection_records'] += len(records)
    counts[expected+'_records'] += len(records)
    counts['zero_selection_turns'] += not records
    counts['zero_callback_turns'] += not any(r['source'] == 'callback' for r in records)


def checked_rows(panel, status_path):
    """Bind 18 historical rows to their existing local acceptance receipts.

    The receipts are program records, not independent proof of raw gameplay.
    Raw hashes and private plan values stay inside this process and local run.
    """
    ledger = load_small(LEDGER, LEDGER_SHA256, normalized=True)
    cfg = load_small(REGISTRATION, REGISTRATION_SHA256, normalized=True)
    verify_completion(panel, status_path)
    load_small(safe_file(panel, 'registration.json'), REGISTRATION_SHA256, normalized=True)
    load_small(safe_file(panel, 'freeze.json'), FREEZE_SHA256, normalized=True)
    seeds = load_small(safe_file(panel, 'private-seeds.local.json'), cfg['seed_document_sha256'])
    require(isinstance(seeds, dict) and isinstance(seeds.get('seeds'), list)
            and len(seeds['seeds']) == 4 and all(type(v) is int and 0 <= v < 2**32 for v in seeds['seeds'])
            and len(set(seeds['seeds'])) == 4, 'Private seed document invalid')
    selected_seed = seeds['seeds'][0]
    game_result = load_small(safe_file(panel, 'games-result.json'))
    require(game_result.get('passed') is True and game_result.get('complete_games') == 72
            and game_result.get('live_official_transitions') == 72*719
            and game_result.get('primary_replay_transitions') == 72*719, 'Panel not accepted')
    audit_rows = load_small(safe_file(panel, 'game-audit-ledger.json'), game_result['audit_ledger_sha256'])
    require(isinstance(ledger, list) and len(ledger) == 72 and isinstance(audit_rows, list)
            and len(audit_rows) == 72, 'Panel ledger incomplete')
    audits = {r['game']: r for r in audit_rows}
    require(len(audits) == 72, 'Duplicate audit rows')
    selected = {}
    for opponent in OPPONENTS:
        for side in (0, 1):
            for policy in ARMS:
                key = f'holdout-S01-{opponent}-{side}-{policy}'
                found = [r for r in ledger if r.get('game') == key]
                require(len(found) == 1, 'Missing public alias row')
                published, audit_row = found[0], audits[key]
                acceptance_path = safe_file(panel, 'job-acceptance', key+'.json')
                acc = load_small(acceptance_path, audit_row['acceptance_receipt_sha256'])
                require(acc.get('passed') is True and acc.get('game') == key
                        and acc.get('retry_allowed') is False
                        and acc.get('freeze_sha256') == sha(read_small(safe_file(panel, 'freeze.json'))),
                        'Game acceptance missing')
                evidence = acc['evidence']
                supervision_doc = load_small(safe_file(panel, 'job-supervision', key+'.json'),
                                             acc['supervision_sha256'])
                supervision = supervision_doc['supervision']
                require(evidence.get('game') == key and evidence.get('official_transitions') == 719
                        and evidence.get('replay_sha256') == audit_row['replay_sha256']
                        and evidence.get('trace_sha256') == audit_row['trace_sha256']
                        and acc['supervision_sha256'] == audit_row['supervision_receipt_sha256']
                        and supervision_doc.get('game') == key
                        and supervision_doc.get('freeze_sha256') == acc['freeze_sha256']
                        and supervision_doc.get('job_input_sha256') == acc['job_input_sha256']
                        and supervision_doc.get('retry_allowed') is False
                        and equal(supervision, evidence.get('job_supervision'))
                        and equal(supervision, audit_row.get('job_supervision'))
                        and all(supervision.get(k) is True for k in ('passed', 'assigned', 'resumed',
                                                                    'launcher_reaped', 'job_members_zero'))
                        and supervision.get('exit_code') == 0
                        and all(supervision.get(k) is None for k in ('error_kind', 'cleanup_error_kind', 'stop_reason')),
                        'Acceptance mismatch')
                plan = load_small(safe_file(panel, 'job-inputs', key+'.json'), acc['job_input_sha256'])
                receipt_paths = {name: safe_file(panel, 'job-results', key+'.'+name+'.json')
                                 for name in ('row', 'audit', 'complete', 'storage')}
                receipts = {name: load_small(path, evidence['receipt_sha256'][name])
                            for name, path in receipt_paths.items()}
                row, audit, complete, storage = (receipts[n] for n in ('row', 'audit', 'complete', 'storage'))
                require(all(k in row and equal(row[k], v) for k, v in published.items())
                        and all(k in row and equal(row[k], v) for k, v in plan.items()), 'Public row or plan mismatch')
                require(row.get('game') == key and row.get('seed_alias') == 'S01'
                        and type(row.get('seed')) is int and row['seed'] == selected_seed
                        and row.get('opponent') == opponent and row.get('side') == side
                        and row.get('policy') == policy and row.get('states') == 720
                        and row.get('action_turns') == 719 and row.get('status') == ['DONE', 'DONE']
                        and row.get('agent_errors') == 0, 'Unexpected historical row')
                task = None if policy == 'teacher_task' else cfg[policy+'_checkpoint_sha256']
                require(row.get('task_checkpoint_sha256') == task
                        and evidence.get('task_checkpoint_sha256') == task
                        and row.get('crop_checkpoint_sha256') == cfg['crop_checkpoint_sha256']
                        and row.get('engine_sha256') == cfg['engine_sha256']
                        and row.get('rival_sha256') == cfg['source_sha256'][opponent+'.py']
                        and sha(read_small(safe_file(panel, 'agents', key+'.py'))) == row.get('bot_sha256')
                        and sha(read_small(safe_file(panel, 'sources', opponent+'.py'))) == row.get('rival_sha256'),
                        'Recorded model, engine or wrapper identity mismatch')
                official = audit.get('official', {})
                counts = audit.get('primary_model_counts')
                check_counts(counts, cfg['game_limits'], policy)
                output = storage.get('output', {})
                require(audit.get('passed') is True and audit.get('game') == key and audit.get('errors') == 0
                        and audit.get('actor_actions_reproduced') == 719
                        and audit.get('rival_actions_reproduced') == 719
                        and audit.get('trace_turns_reproduced') == 719
                        and audit.get('replay_sha256') == row['replay_gzip_sha256'] == evidence['replay_sha256']
                        and audit.get('trace_sha256') == row['trace_sha256'] == evidence['trace_sha256']
                        and official.get('passed') is True and official.get('official_transitions_reexecuted') == 719
                        and official.get('engine_sha256') == cfg['engine_sha256']
                        and official.get('both_public_and_private_states_match_each_step') is True
                        and official.get('inventory_order_checked_each_step') is True
                        and official.get('semantic_sha256') == row['semantic_replay_sha256']
                        and official.get('semantic_sha256_v2') == row['semantic_replay_sha256_v2']
                        and official.get('final_rewards') == row['rewards']
                        and complete.get('passed') is True and complete.get('game') == key
                        and complete.get('row_sha256') == evidence['receipt_sha256']['row']
                        and complete.get('certificate_sha256') == evidence['receipt_sha256']['audit']
                        and storage.get('passed') is True and storage.get('game') == key
                        and storage.get('error_kind') is None
                        and output.get('limit_bytes') == cfg['storage']['game_write_bytes']
                        and type(output.get('used_bytes')) is int and 0 <= output['used_bytes'] <= output['limit_bytes']
                        and output.get('remaining_bytes') == output['limit_bytes']-output['used_bytes']
                        and output.get('refused') == 0
                        and counts == audit.get('live_model_counts'), 'Historical replay receipt mismatch')
                replay = safe_file(panel, 'games', key, 'replay.json.gz')
                trace = safe_file(panel, 'traces', key+'.jsonl.gz')
                selected[(opponent, side, policy)] = dict(row=row, counts=counts,
                    replay=dict(path=f'games/{key}/replay.json.gz', size=replay.stat().st_size,
                                sha256=row['replay_gzip_sha256']),
                    trace=dict(path=f'traces/{key}.jsonl.gz', size=trace.stat().st_size,
                               sha256=row['trace_sha256']))
    require(len(selected) == 18, 'Wrong selected game count')
    return selected


def input_snapshot(panel, status_path, selected):
    """Freeze current expected identities privately, after previous acceptance.

    The snapshot is post-hoc and cannot prove what was present when the
    historical panel was first accepted. The raw files are checked when read.
    """
    metadata = ('games-result.json', 'game-audit-ledger.json', 'independent-runner-result.json',
                'independent-replay-verification.json', 'results/coordinator-complete.json',
                'replays/launcher-result.json')
    hashes = {name: sha(read_small(safe_file(panel, *name.split('/')))) for name in metadata}
    hashes['local-completion-status'] = sha(read_small(status_path))
    inputs = {game['row']['game']: {'replay_sha256': game['replay']['sha256'],
                                    'trace_sha256': game['trace']['sha256']}
              for game in selected.values()}
    require(len(inputs) == 18, 'Input snapshot incomplete')
    return dict(schema='task-prefix-private-input-snapshot-v1', result_commit=RESULT_COMMIT,
                metadata_sha256=hashes, expected_inputs=inputs)


def write_result_new(output, result):
    """Publish success only after a complete, closed, verified local write."""
    payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    pending, final = output/'result.pending.json', output/'result.json'
    with pending.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == payload and not final.exists(), 'Result write verification failed')
    os.replace(pending, final)


def run(panel, status_path, manifest, progress):
    started = time.monotonic()
    check_deadline(started)
    selected = checked_rows(panel, status_path)
    require(equal(manifest, input_snapshot(panel, status_path, selected)), 'Private input snapshot changed')
    check_deadline(started)
    meter = analysis.Meter(compressed_limit=COMPRESSED_LIMIT, decoded_limit=DECODED_LIMIT)
    pairs = []
    selection_counts = {arm: dict(selection_records=0, callback_records=0, teacher_records=0,
                                  zero_selection_turns=0, zero_callback_turns=0) for arm in ARMS}
    for opponent in OPPONENTS:
        for side in (0, 1):
            games = {arm: selected[(opponent, side, arm)] for arm in ARMS}
            compact = {}
            for arm, game in games.items():
                check_deadline(started)
                replay = panel/game['replay']['path']
                value = analysis.read_json(replay, game['replay'], meter)
                check_replay_seed(value, game['row'])
                compact[arm] = analysis.compact_replay(value, dict(side=side, expected=game['row']))
                del value
                check_deadline(started)
            comparisons = {ref: PairComparison(compact['merged'], compact[ref], side, opponent, ref)
                           for ref in ('unweighted', 'teacher_task')}
            streams = {arm: analysis.trace_lines(panel/game['trace']['path'], game['trace'], meter)
                       for arm, game in games.items()}
            descriptions = {arm: analysis.descriptor() for arm in ARMS}
            recorded = {arm: 0 for arm in ARMS}
            try:
                for step in range(719):
                    check_deadline(started)
                    records = {}
                    for arm in ARMS:
                        check_deadline(started)
                        try:
                            trace = next(streams[arm])
                        except StopIteration as exc:
                            raise ValueError('Trace ended early') from exc
                        check_deadline(started)
                        check_trace_identity(trace, games[arm]['row'])
                        records[arm] = analysis.validate_trace(trace, compact[arm], step, descriptions[arm])
                        recorded[arm] += len(records[arm])
                        count_sources(arm, records[arm], selection_counts[arm])
                    for ref, comparison in comparisons.items():
                        comparison.observe_turn(step, records['merged'], records[ref])
                for arm in ARMS:
                    check_deadline(started)
                    try:
                        next(streams[arm])
                    except StopIteration:
                        pass
                    else:
                        raise ValueError('Trace has extra turns')
                    if arm == 'teacher_task':
                        require(games[arm]['counts'].get('task_forwards') == 0, 'Unexpected teacher task model')
                    else:
                        require(recorded[arm] == games[arm]['counts'].get('task_forwards'),
                                'Recorded task calls differ from reconstructed count')
            finally:
                for stream in streams.values():
                    stream.close()
            pairs.extend(comparison.finish() for comparison in comparisons.values())
            check_deadline(started)
            progress['triads_completed'] += 1
            print(json.dumps({'triads_completed': progress['triads_completed']}), flush=True)
    require(len(pairs) == 12 and len(meter.receipts) == 36 and len(meter.opened) == 36,
            'Input or pairing coverage incomplete')
    check_deadline(started)
    return dict(schema='task-prefix-diagnostic-v1', passed=True, seed_alias='S01',
        input_games=18, paired_comparisons=12, pairs=pairs, selection_counts=selection_counts,
        raw_reads=meter.snapshot(), wall_deadline='sampled, not OS-enforced',
        decoded_budget_is_memory_limit=False,
        seconds=time.monotonic()-started, new_games=0, official_transitions=0,
        neural_forwards=0, training_updates=0, gpu_operations=0,
        interpretation='Exploratory historical prefix associations; no causal, significance or holdout claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', required=True, type=Path, help='Local accepted panel directory')
    parser.add_argument('--status', required=True, type=Path, help='Local panel completion status')
    parser.add_argument('--manifest', required=True, type=Path, help='Private input snapshot')
    parser.add_argument('--prepare-only', action='store_true', help='Create snapshot without decoding games')
    parser.add_argument('--output', type=Path, help='New local output directory')
    args = parser.parse_args()
    panel = args.panel.resolve(strict=True)
    status_path = args.status.resolve(strict=True)
    manifest_path = args.manifest.resolve()
    if manifest_path.is_relative_to(panel) or manifest_path.is_relative_to(REPO.resolve()):
        raise ValueError('Private manifest must be outside the source panel and public repository')
    if args.prepare_only:
        require(args.output is None, 'Preparation does not create an analysis output')
        selected = checked_rows(panel, status_path)
        snapshot = input_snapshot(panel, status_path, selected)
        with manifest_path.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        print(json.dumps({'passed': True, 'input_games': 18, 'raw_games_decoded': 0}), flush=True)
        return
    require(args.output is not None, 'Output directory required')
    manifest = load_small(manifest_path)
    output = args.output.resolve()
    if output.is_relative_to(panel) or output.is_relative_to(REPO.resolve()):
        raise ValueError('Output must be outside the source panel and public repository')
    output.mkdir(parents=False, exist_ok=False)
    progress = dict(triads_completed=0)
    try:
        result = run(panel, status_path, manifest, progress)
        write_result_new(output, result)
    except BaseException as exc:
        with (output/'failure.json').open('x', encoding='utf-8', newline='\n') as f:
            json.dump(dict(passed=False, error_type=type(exc).__name__, **progress), f, indent=2)
            f.write('\n')
        with (output/'failure.local.log').open('x', encoding='utf-8', newline='\n') as f:
            f.write(traceback.format_exc())
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__, **progress}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'passed': True, 'input_games': 18, 'paired_comparisons': 12}), flush=True)


if __name__ == '__main__':
    main()
