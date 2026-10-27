"""Bounded post-hoc diagnosis of one fixed, already observed pair."""
import json
from pathlib import Path
import time
from learning.task_action_impact_diagnostic_v1.analysis import (
    Meter, compact_replay, compare_callbacks, descriptor, first_request_difference,
    read_json, trace_lines, validate_trace, request_digest)
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_completion_bc_v1.collect import current_game
from .experiment import OUT, PARENT, deadline_check, read, ready, sha, verify_commit, write_new

SEED, SIDE, OPPONENT = 3342200684, 0, 'round2'
OPS = frozenset(('DIG','FERTILIZE','HARVEST','PLANT','WATER','PASS','PICKUP','PLACE','DROP','MOVE','REST',
                 'NORTH','SOUTH','EAST','WEST'))


class ReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Failure receipt could not be saved')
        self.primary_error_kind, self.receipt_error_kind = primary_error_kind, receipt_error_kind


def choice_summary(left, right, request_different):
    comparison = compare_callbacks(left, right)
    choice = comparison['choice']
    if choice is None: return None, None
    i = choice['callback']
    if choice['candidate_op'] not in OPS or choice['reference_op'] not in OPS:
        raise ValueError('Invalid operation category')
    summary = dict(candidate_operation=choice['candidate_op'], teacher_operation=choice['reference_op'],
                   same_unit=choice['same_unit'], complete_request_different=bool(request_different),
                   emitted_unit_request_equal=choice['emitted_unit_request_equal'],
                   interpretation='Same-input choice and complete-request differences coincide; the choice alone is not proven to cause the request difference')
    return summary, (left[i]['unit'], request_digest(left[i]['unit_request']))


def event_summary(events, expected_request_digest=None):
    if len(events) != 1: return dict(observation='unknown')
    event = events[0]
    def operation(value):
        return value[0] if isinstance(value,list) and value and value[0] in OPS else 'unknown'
    requested, effective = operation(event.get('requested')), operation(event.get('effective'))
    if expected_request_digest is not None:
        if requested == 'unknown' or request_digest(event['requested']) != expected_request_digest:
            return dict(observation='unknown')
    status = event.get('status')
    if status not in ('executed','blocked','no_effect','error','pass','partial'): status = 'unknown'
    changed = (event.get('before') != event.get('after') or event.get('tile_before') != event.get('tile_after'))
    has_state = all(k in event for k in ('before','after','tile_before','tile_after'))
    return dict(observation='unique',requested_operation=requested,effective_operation=effective,
                rewritten=(event['requested'] != event['effective']) if requested != 'unknown' and effective != 'unknown' else None,
                status=status,atomic_change_observed=bool(changed) if has_state else None)


def transition_progress(replay_started, transitions):
    return dict(confirmed_transitions=(transitions if transitions is not None else 'unknown' if replay_started else 0),
                partial_replay_transitions='unknown' if replay_started and transitions is None else 0)


def validate_raw_identities(games, expected):
    if not isinstance(expected, dict) or set(expected) != {'candidate', 'teacher_task'}:
        raise ValueError('Invalid registered raw inputs')
    for arm in ('candidate', 'teacher_task'):
        if not isinstance(expected[arm], dict) or set(expected[arm]) != {'replay', 'trace'}:
            raise ValueError('Invalid registered raw inputs')
        for kind in ('replay', 'trace'):
            registered = expected[arm][kind]
            actual = games[arm][kind]
            if (not isinstance(registered, dict) or set(registered) != {'path', 'size', 'sha256'}
                    or not isinstance(actual, dict) or set(actual) != {'path', 'size', 'sha256'}
                    or any(actual[k] != registered[k] for k in ('path', 'size', 'sha256'))):
                raise ValueError('Registered raw identity changed')


def run(out=OUT):
    from decision.full_audit import reexecute
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if (cfg['seed'] != SEED or cfg['side'] != SIDE or cfg['opponent'] != OPPONENT
        or cfg['limits'] != dict(compressed_bytes=128*1024**2,decoded_bytes=1024**3,seconds=300,
                                 official_transitions=719,raw_files=4,decode_passes_per_file=1)):
        raise ValueError('Diagnostic scope changed')
    finished, ledger = read(PARENT/'games-result.json'), read(PARENT/'game-ledger.json')
    if (not finished['passed'] or finished['game_ledger_sha256'] != sha(PARENT/'game-ledger.json')
            or finished['audit_ledger_sha256'] != sha(PARENT/'game-audit-ledger.json')):
        raise ValueError('Parent panel identity changed')
    selected = {}
    for arm in ('candidate','teacher_task'):
        matches = [r for r in ledger if r['seed']==SEED and r['opponent']==OPPONENT and r['side']==SIDE and r['policy']==arm]
        if len(matches) != 1: raise ValueError('Fixed pair not unique')
        selected[arm] = matches[0]
    games, certificates = {}, {}
    for arm,row in selected.items(): games[arm], certificates[arm] = current_game(PARENT,row)
    validate_raw_identities(games,cfg['raw_inputs'])
    if (out/'analysis-started.json').exists(): raise ValueError('Attempt already consumed')
    marker = dict(created=False,complete=False)
    meter, started = None, time.monotonic()
    replay_started, replay_completed, transitions = False, False, None
    def check():
        deadline_check()
        if time.monotonic()-started > 300: raise TimeoutError('Analysis budget exceeded')
    try:
        path = (out/'analysis-started.json').resolve()
        if not path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json'),
                           parent_ledger_sha256=sha(PARENT/'game-ledger.json')),stream,indent=2)
            stream.write('\n')
        marker['complete'] = True
        meter = Meter(128*1024**2,1024**3)
        replay, compact = {}, {}
        for arm in ('candidate','teacher_task'):
            check()
            replay[arm] = read_json(raw_path(PARENT,games[arm]['replay'],'replay'),games[arm]['replay'],meter)
            check()
            compact[arm] = compact_replay(replay[arm],games[arm])
        first = first_request_difference(compact['candidate'],compact['teacher_task'],SIDE)
        streams = {arm:trace_lines(raw_path(PARENT,games[arm]['trace'],'trace'),games[arm]['trace'],meter) for arm in games}
        desc = {arm:descriptor() for arm in games}
        linked, selected_identity = None, None
        try:
            for step in range(719):
                check()
                records = {}
                for arm in games:
                    trace = next(streams[arm])
                    digest = cfg['candidate_checkpoint_sha256'] if arm == 'candidate' else None
                    if trace['task_checkpoint_sha256'] != digest or trace['checkpoint_sha256'] != cfg['crop_checkpoint_sha256']:
                        raise ValueError('Trace model identity changed')
                    records[arm] = validate_trace(trace,compact[arm],step,desc[arm])
                    expected = 'callback' if arm == 'candidate' else 'teacher'
                    if any(r['source'] != expected for r in records[arm]): raise ValueError('Unexpected selector source')
                if step == first and compact['candidate']['physical'][step] == compact['teacher_task']['physical'][step]:
                    linked, selected_identity = choice_summary(records['candidate'],records['teacher_task'],True)
            for stream in streams.values():
                if next(stream,None) is not None: raise ValueError('Extra trace records')
        finally:
            for stream in streams.values(): stream.close()
        snapshot = meter.snapshot()
        if snapshot['files_opened'] != 4 or snapshot['files_completed'] != 4 or snapshot['raw_decode_passes_per_file'] != 1:
            raise ValueError('Unexpected raw read count')
        events = []
        def sink(event):
            check()
            if linked is not None and event.get('step') == first and event.get('player') == SIDE and event.get('phase') == 'unit' and event.get('unit') == selected_identity[0]:
                events.append(event)
        check()
        replay_started = True
        audit = reexecute(replay['candidate'],trace=True,label_sink=sink)
        replay_completed = True
        transitions = audit['official_transitions_reexecuted']
        check()
        row = selected['candidate']
        if (not audit['passed'] or transitions != 719 or audit['engine_sha256'] != cfg['engine_sha256']
            or audit['semantic_sha256'] != row['semantic_replay_sha256']
            or audit['semantic_sha256_v2'] != row['semantic_replay_sha256_v2']
            or audit['final_rewards'] != row['rewards'] or row['status'] != ['DONE','DONE']):
            raise ValueError('Official replay identity changed')
        result = dict(passed=True,first_request_difference_step=first,linked_choice=linked,
            official_event=event_summary(events,selected_identity[1] if selected_identity else None),official_transitions=transitions,raw_reads=snapshot,
            parent_freeze_sha256=sha(PARENT/'freeze.json'),parent_ledger_sha256=sha(PARENT/'game-ledger.json'),
            parent_audit_ledger_sha256=sha(PARENT/'game-audit-ledger.json'),method_sha256=sha(Path(__file__)),
            post_hoc_not_holdout=True,causal_financial_effect_not_established=True,retry_allowed=False)
        write_new(out/'analysis-result.json',result)
        return result
    except BaseException as exc:
        if marker['created']:
            try:
                write_new(out/'analysis-failure.local.json',dict(error_kind=type(exc).__name__,marker=marker,
                    replay_started=replay_started,replay_completed=replay_completed,**transition_progress(replay_started,transitions),
                    raw_reads=meter.snapshot() if meter else None,retry_allowed=False,partial_work_not_refunded=True))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise
