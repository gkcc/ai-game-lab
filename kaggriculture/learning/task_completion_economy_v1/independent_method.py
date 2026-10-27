"""Single-use callback projection audit using the shared official engine and validators."""
import hashlib
import json
import time
from pathlib import Path
from collections import Counter
from learning.task_choice_v1.features import OPS

from .experiment import OUT, ROOT, deadline_check, read, ready, sha, verify_commit, write_new


class ReceiptFailure(RuntimeError):
    def __init__(self, original, receipt):
        super().__init__('Independent failure receipt could not be written')
        self.original_error_kind = type(original).__name__
        self.receipt_error_kind = type(receipt).__name__


def sample_identities(identities, count=20):
    if len(identities) < count or len(set(identities)) != len(identities):
        raise ValueError('Insufficient or duplicate callback identities')
    def key(identity):
        payload = json.dumps(list(identity), ensure_ascii=True, separators=(',', ':')).encode('ascii')
        return hashlib.sha256(payload).digest()
    return sorted(identities, key=lambda x: (key(x), x))[:count]


def sample_operation_counts(selected, operations):
    counts = Counter(operations[identity] for identity in selected)
    if any(op not in OPS for op in counts) or sum(counts.values()) != len(selected):
        raise ValueError("Invalid sample operation counts")
    return {op: counts[op] for op in OPS}


def create_marker(out, state):
    path = (Path(out)/'independent-attempt.local.json').resolve()
    if not path.is_relative_to(OUT): raise ValueError('Marker outside frozen output')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        state['created'] = True
        json.dump(dict(attempt=1, retry_allowed=False, freeze_sha256=sha(out/'freeze.json')), stream)
        stream.write('\n')
    state['complete'] = True


def run(out=OUT):
    if not __debug__: raise RuntimeError('Optimized execution is not permitted')
    from learning.task_action_impact_diagnostic_v1.analysis import Meter, compact_replay, descriptor, read_json, trace_lines, validate_trace, ordered_hash, request_digest
    from learning.task_action_observation_v1.analysis import raw_path
    from learning.task_completion_bc_v1.collect import current_game
    from decision.full_audit import reexecute
    out = Path(out).resolve()
    if out != OUT: raise ValueError('Unexpected output root')
    cfg = ready(out)
    verify_commit(out)
    finished = read(out/'games-result.json')
    ledger = read(out/'game-ledger.json')
    audits = read(out/'game-audit-ledger.json')
    if (not finished['passed'] or not finished['all_owned_processes_reaped']
            or finished['game_ledger_sha256'] != sha(out/'game-ledger.json')
            or finished['audit_ledger_sha256'] != sha(out/'game-audit-ledger.json')):
        raise ValueError('Eight-game evidence not accepted')
    expected = {(seed, 'round2', side, arm) for seed in cfg['test_seeds'] for side in (0, 1) for arm in ('candidate', 'teacher_task')}
    if len(cfg['test_seeds']) != 2 or len(set(cfg['test_seeds'])) != 2: raise ValueError('Invalid fixed seed panel')
    if (len(ledger) != 8 or len(audits) != 8 or len({x['game'] for x in ledger}) != 8 or len({x['game'] for x in audits}) != 8
            or {x['game'] for x in audits} != {x['game'] for x in ledger}
            or {(x['seed'], x['opponent'], x['side'], x['policy']) for x in ledger} != expected
            or any(x['game'] != f"holdout-{x['seed']}-round2-{x['side']}-{x['policy']}" for x in ledger)):
        raise ValueError('Panel identity differs from registration')
    selected = [x for x in ledger if x['policy'] == 'candidate' and x['opponent'] == 'round2' and x['side'] == 0 and x['seed'] == cfg['test_seeds'][0]]
    if len(selected) != 1: raise ValueError('Selected game missing')
    row = selected[0]
    if row['task_checkpoint_sha256'] != cfg['candidate_checkpoint_sha256']:
        raise ValueError('Candidate checkpoint identity mismatch')
    game, certificate = current_game(out, row)
    entries = [x for x in audits if x['game'] == row['game']]
    if (len(entries) != 1 or entries[0]['certificate_sha256'] != sha(out/'job-results'/(row['game']+'.audit.json'))
            or entries[0]['replay_sha256'] != game['replay']['sha256']
            or entries[0]['trace_sha256'] != game['trace']['sha256']
            or entries[0]['official_transitions'] != 719):
        raise ValueError('Selected audit identity mismatch')
    if (out/'independent-attempt.local.json').exists(): raise ValueError('Attempt already consumed')
    marker = dict(created=False, complete=False)
    limits, meter, started = cfg['independent_limits'], None, time.monotonic()
    replay_started, transitions, last_event_step = False, 0, None
    def check():
        deadline_check()
        if time.monotonic() - started > limits['seconds']: raise TimeoutError('Independent budget exceeded')
    try:
        create_marker(out, marker)
        meter = Meter(limits['compressed_bytes'], limits['decoded_bytes'])
        check()
        replay = read_json(raw_path(out, game['replay'], 'replay'), game['replay'], meter)
        compact = compact_replay(replay, game)
        stream = trace_lines(raw_path(out, game['trace'], 'trace'), game['trace'], meter)
        identities, verified, operations = [], {}, {}
        desc = descriptor()
        try:
            for step in range(719):
                check()
                trace = next(stream)
                if (trace['task_checkpoint_sha256'] != cfg['candidate_checkpoint_sha256']
                        or trace['checkpoint_sha256'] != cfg['crop_checkpoint_sha256']):
                    raise ValueError('Trace model identity mismatch')
                records = validate_trace(trace, compact, step, desc)
                for ordinal, record in enumerate(records):
                    identity = (row['game'], step, ordinal)
                    identities.append(identity)
                    operations[identity] = record['op']
                    unit = record['unit']
                    verified[identity] = (record['source'] == 'callback'
                        and ordered_hash(record['inputs']['observation']) == compact['actor_observation'][step]
                        and ordered_hash(record['inputs']['configuration']) == compact['configuration'][step]
                        and request_digest(record['unit_request']) == compact['units'][step][unit])
            if next(stream, None) is not None: raise ValueError('Trace has extra turns')
        finally:
            stream.close()
        chosen = sample_identities(identities, limits['sample_count'])
        matched = sum(verified[x] for x in chosen)
        if matched != limits['sample_count']: raise ValueError('Sample projection or request mismatch')
        operation_counts = sample_operation_counts(chosen, operations)
        if sum(operation_counts.values()) != limits['sample_count']: raise ValueError('Sample count mismatch')
        def sink(event):
            nonlocal last_event_step
            check()
            last_event_step = event.get('step')
        replay_started = True
        audit = reexecute(replay, trace=True, label_sink=sink)
        transitions = audit['official_transitions_reexecuted']
        check()
        if (not audit['passed'] or transitions != 719 or audit['engine_sha256'] != cfg['engine_sha256']
                or audit['semantic_sha256'] != row['semantic_replay_sha256']
                or audit['semantic_sha256_v2'] != row['semantic_replay_sha256_v2']
                or audit['semantic_sha256_v2'] != certificate['official']['semantic_sha256_v2']):
            raise ValueError('Official replay identity mismatch')
        snapshot = meter.snapshot()
        if snapshot['files_opened'] != 2 or snapshot['files_completed'] != 2 or snapshot['raw_decode_passes_per_file'] != 1:
            raise ValueError('Raw read plan incomplete')
        selected_hash = hashlib.sha256(json.dumps(chosen, ensure_ascii=True, separators=(',', ':')).encode('ascii')).hexdigest()
        result = dict(passed=True, game=row['game'], official_transitions=719, sample_count=len(chosen), matched=matched,
            selection_rule=cfg['independent_selection'], sample_identity_sha256=selected_hash,
            sample_operation_counts=operation_counts, raw_reads=snapshot,
            checkpoint_sha256=cfg['candidate_checkpoint_sha256'], crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],
            engine_sha256=cfg['engine_sha256'], test_seeds=list(cfg['test_seeds']), freeze_sha256=sha(out/'freeze.json'),
            registration_sha256=sha(out/'registration.json'), games_result_sha256=sha(out/'games-result.json'),
            game_ledger_sha256=sha(out/'game-ledger.json'), audit_ledger_sha256=sha(out/'game-audit-ledger.json'),
            game_row_sha256=sha(out/'job-results'/(row['game']+'.row.json')),
            game_audit_sha256=sha(out/'job-results'/(row['game']+'.audit.json')),
            method_sha256=sha(Path(__file__)), no_model_inference=True, no_new_games=True, retry_allowed=False,
            limitations='Shared official engine and shared trace input validator; no independent game engine or score computation')
        write_new(out/'independent-replay-verification.json', result)
        review = dict(passed=True, freeze_sha256=sha(out/'freeze.json'),
            registration_sha256=sha(out/'registration.json'), games_result_sha256=sha(out/'games-result.json'),
            game_ledger_sha256=sha(out/'game-ledger.json'), game_audit_ledger_sha256=sha(out/'game-audit-ledger.json'),
            verification_sha256=sha(out/'independent-replay-verification.json'), method_sha256=sha(Path(__file__)),
            selected_game=row['game'], checkpoint_sha256=cfg['candidate_checkpoint_sha256'],
            crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'], engine_sha256=cfg['engine_sha256'],
            test_seeds=list(cfg['test_seeds']),
            official_transitions=719, sample_count=len(chosen), sample_matched=matched,
            limitations='Shared official engine and shared input validation, not independent engine or scoring')
        write_new(out/'independent-result-review.json', review)
        return result
    except BaseException as exc:
        if marker['created']:
            try:
                write_new(out/'independent-failure.local.json', dict(error_kind=type(exc).__name__, marker=marker,
                    official_transitions_completed=transitions, last_event_step=last_event_step,
                    replay_state='completed' if transitions else 'unknown_after_start' if replay_started else 'not_started',
                    partial_transition_count_is_unknown=bool(replay_started and not transitions),
                    raw_reads=meter.snapshot() if meter is not None else None, retry_allowed=False))
            except BaseException as receipt:
                raise ReceiptFailure(exc, receipt) from exc
        raise


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(passed=result['passed'], official_transitions=result['official_transitions'], matched=result['matched'])))
