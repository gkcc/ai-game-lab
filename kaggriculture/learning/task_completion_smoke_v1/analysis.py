"""One bounded, read-only comparison of four newly generated official games."""
import json
from pathlib import Path
import time

from learning.task_action_impact_diagnostic_v1.analysis import (
    Meter, compact_replay, compare_callbacks, descriptor, first_request_difference,
    read_json, trace_lines, validate_trace)
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_completion_bc_v1.collect import current_game
from .experiment import OUT, deadline_check, read, ready, sha, verify_commit, write_new


class AnalysisReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Analysis failure receipt could not be saved')
        self.primary_error_kind, self.receipt_error_kind = primary_error_kind, receipt_error_kind


def matched_choice(left, right, candidate_request, reference_request):
    """Compare the first differing choice only on equal full selector inputs."""
    comparison = compare_callbacks(left, right)
    choice = comparison['choice']
    return dict(matched_context_calls=comparison['matched_context_calls'],
        context_stop=comparison['context_stop'], choice_different=choice is not None,
        complete_request_different=candidate_request != reference_request,
        same_unit=choice['same_unit'] if choice else None,
        selected_unit_request_equal=choice['emitted_unit_request_equal'] if choice else None)


def first_request_linked(pair):
    """Require the same-turn model difference at the first history divergence."""
    return (pair['first_linked_change_step'] is not None and
            pair['first_linked_change_step'] == pair['first_request_difference_step'])


def check_budget(started, seconds):
    deadline_check()
    if time.monotonic()-started > seconds: raise TimeoutError('Analysis budget exceeded')


def run(out=OUT):
    from .campaign import verify_independent_review
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    verify_independent_review(out, cfg)
    finished = read(out/'games-result.json')
    ledger = read(out/'game-ledger.json')
    if (not finished['passed'] or finished['game_ledger_sha256'] != sha(out/'game-ledger.json')
            or finished['audit_ledger_sha256'] != sha(out/'game-audit-ledger.json')):
        raise ValueError('Accepted four-game panel required')
    seed, opponent = cfg['test_seeds'][0], cfg['opponents'][0]
    expected = {(seed, opponent, side, policy) for side in (0,1) for policy in ('candidate','teacher_task')}
    actual = {(r['seed'],r['opponent'],r['side'],r['policy']) for r in ledger}
    if len(ledger) != 4 or actual != expected or len({r['game'] for r in ledger}) != 4:
        raise ValueError('Panel differs from registration')
    if (out/'analysis-started.json').exists(): raise ValueError('Analysis attempt already consumed')
    limits = cfg['analysis_limits']
    marker = dict(created=False, complete=False)
    meter, tick, completed_pairs, rows = None, time.monotonic(), 0, []
    def check_time():
        check_budget(tick, limits['seconds'])
    try:
        path = (out/'analysis-started.json').resolve()
        if not path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json'),
                           games_result_sha256=sha(out/'games-result.json')),stream,indent=2)
            stream.write('\n')
        marker['complete'] = True
        meter = Meter(limits['compressed_bytes'], limits['decoded_bytes'])
        for side in (0,1):
            check_time()
            selected = {arm:next(r for r in ledger if r['side']==side and r['policy']==arm)
                        for arm in ('candidate','teacher_task')}
            games = {arm:current_game(out,row)[0] for arm,row in selected.items()}
            compact = {}
            for arm,g in games.items():
                check_time()
                replay = read_json(raw_path(out,g['replay'],'replay'),g['replay'],meter)
                check_time()
                compact[arm] = compact_replay(replay,g)
                check_time()
                del replay
            first_request = first_request_difference(compact['candidate'],compact['teacher_task'],side)
            descriptors = {arm:descriptor() for arm in games}
            streams = {arm:trace_lines(raw_path(out,g['trace'],'trace'),g['trace'],meter) for arm,g in games.items()}
            pair = dict(side=side,first_request_difference_step=first_request,common_physical_prefix_turns=0,
                        matched_context_calls=0,choice_difference_same_request_turns=0,
                        choice_difference_changed_request_turns=0,first_linked_change_step=None,
                        unmatched_context_turns=0,physical_prefix_ended=False,
                        candidate_callbacks=0,candidate_fertilizer_opportunities=0,
                        candidate_fertilizer_selections=0,candidate_fertilizer_work_requests=0)
            try:
                for step in range(719):
                    check_time()
                    traces = {arm:next(stream) for arm,stream in streams.items()}
                    records = {}
                    for arm,trace in traces.items():
                        digest = cfg['candidate_checkpoint_sha256'] if arm == 'candidate' else None
                        if (trace['task_checkpoint_sha256'] != digest
                                or trace['checkpoint_sha256'] != cfg['crop_checkpoint_sha256']):
                            raise ValueError('Trace model identity changed')
                        records[arm] = validate_trace(trace,compact[arm],step,descriptors[arm])
                        source = 'callback' if arm == 'candidate' else 'teacher'
                        if any(r['source'] != source for r in records[arm]):
                            raise ValueError('Unexpected task selector source')
                    for r in records['candidate']:
                        pair['candidate_callbacks'] += 1
                        pair['candidate_fertilizer_opportunities'] += int(any(c['op']=='FERTILIZE' for c in r['inputs']['candidates']))
                        pair['candidate_fertilizer_selections'] += int(r['op']=='FERTILIZE')
                        pair['candidate_fertilizer_work_requests'] += int(r['op']=='FERTILIZE' and bool(r['unit_request']) and r['unit_request'][0]=='FERTILIZE')
                    if pair['physical_prefix_ended']:
                        continue
                    a,b = compact['candidate'],compact['teacher_task']
                    if a['physical'][step] != b['physical'][step]:
                        pair['physical_prefix_ended'] = True
                        continue
                    pair['common_physical_prefix_turns'] += 1
                    c = matched_choice(records['candidate'],records['teacher_task'],a['requests'][step][side],b['requests'][step][side])
                    pair['matched_context_calls'] += c['matched_context_calls']
                    pair['unmatched_context_turns'] += int(c['context_stop'] is not None)
                    if c['choice_different']:
                        field = ('choice_difference_changed_request_turns' if c['complete_request_different']
                                 else 'choice_difference_same_request_turns')
                        pair[field] += 1
                        if c['complete_request_different'] and pair['first_linked_change_step'] is None:
                            pair['first_linked_change_step'] = step
                for stream in streams.values():
                    if next(stream,None) is not None: raise ValueError('Trace has extra turns')
            finally:
                for stream in streams.values(): stream.close()
            candidate_audit = read(out/'job-results'/(selected['candidate']['game']+'.audit.json'))
            pair['candidate_task_forwards'] = candidate_audit['live_model_counts']['task_forwards']
            pair['candidate_fertilizer_official_events'] = candidate_audit['official']['fertilizer_actor_request_status']
            pair['reference_task_forwards'] = read(out/'job-results'/(selected['teacher_task']['game']+'.audit.json'))['live_model_counts']['task_forwards']
            if pair['candidate_task_forwards'] <= 0 or pair['reference_task_forwards'] != 0:
                raise ValueError('Task model not active in registered arms')
            rows.append(pair)
            completed_pairs += 1
        check_time()
        snapshot = meter.snapshot()
        if (completed_pairs != 2 or snapshot['files_opened'] != 8 or snapshot['files_completed'] != 8
                or snapshot['raw_decode_passes_per_file'] != 1):
            raise ValueError('Analysis raw-file plan incomplete')
        linked = sum(first_request_linked(row) for row in rows)
        result = dict(passed=True,model_request_influence_observed=linked>0,first_request_changes_linked=linked,pairs=rows,
            raw_reads=snapshot,official_transitions=0,new_games=0,training_updates=0,
            own_model_inference_calls=0,completion_and_financial_causation='not_established',
            interpretation='Two seat-pairs on one seed are an interface smoke, not a competitive or statistical claim.',
            game_ledger_sha256=sha(out/'game-ledger.json'),freeze_sha256=sha(out/'freeze.json'),
            retry_allowed=False)
        write_new(out/'analysis-result.json',result)
        return result
    except BaseException as exc:
        if marker['created']:
            try:
                write_new(out/'analysis-failure.local.json',dict(error_kind=type(exc).__name__,marker=marker,
                    complete_pairs=completed_pairs,raw_reads=meter.snapshot() if meter else None,
                    partial_raw_reads_not_refunded=True,retry_allowed=False))
            except BaseException as receipt:
                raise AnalysisReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(passed=result['passed'],model_request_influence_observed=result['model_request_influence_observed'])))
