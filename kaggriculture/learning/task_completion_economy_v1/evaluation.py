"""Fixed two-seed economic screening with bounded trace aggregation."""
import json
import math
from pathlib import Path
import time

from learning.task_action_impact_diagnostic_v1.analysis import Meter, trace_lines
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_completion_bc_v1.collect import current_game
from .experiment import OUT, deadline_check, read, ready, sha, verify_commit, write_new


class ReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Evaluation failure receipt could not be saved')
        self.primary_error_kind, self.receipt_error_kind = primary_error_kind, receipt_error_kind


def check_budget(started, seconds):
    deadline_check()
    if time.monotonic()-started > seconds: raise TimeoutError('Trace aggregation budget exceeded')


def count_turn(trace, step, policy, task_sha, crop_sha):
    if (trace['step']!=step or trace['task_checkpoint_sha256']!=task_sha
            or trace['checkpoint_sha256']!=crop_sha
            or any(trace[k]!=0 for k in ('load_errors','inference_errors','fallbacks'))
            or type(trace['records']) is not list):
        raise ValueError('Trace identity or error mismatch')
    source = 'callback' if policy=='candidate' else 'teacher'
    result = dict(callbacks=0,fertilizer_opportunities=0,fertilizer_selections=0,fertilizer_work_requests=0)
    for row in trace['records']:
        if row['source']!=source or type(row['inputs']['candidates']) is not list:
            raise ValueError('Unexpected task selection source')
        action = row['unit_request']
        if type(action) is not list: raise ValueError('Malformed unit request')
        result['callbacks'] += 1
        result['fertilizer_opportunities'] += int(any(x['op']=='FERTILIZE' for x in row['inputs']['candidates']))
        result['fertilizer_selections'] += int(row['op']=='FERTILIZE')
        result['fertilizer_work_requests'] += int(row['op']=='FERTILIZE' and bool(action) and action[0]=='FERTILIZE')
    return result


def validate_panel(rows, cfg):
    seeds, opponent = cfg['test_seeds'], cfg['opponents'][0]
    expected = {(s,opponent,side,policy) for s in seeds for side in (0,1) for policy in ('candidate','teacher_task')}
    actual = {(r['seed'],r['opponent'],r['side'],r['policy']) for r in rows}
    if len(rows)!=8 or actual!=expected or len({r['game'] for r in rows})!=8:
        raise ValueError('Invalid fixed economic panel')


def paired_metrics(rows, cfg):
    validate_panel(rows,cfg)
    seeds = cfg['test_seeds']
    paired = []
    for seed in seeds:
        for side in (0,1):
            a = next(r for r in rows if r['seed']==seed and r['side']==side and r['policy']=='candidate')
            b = next(r for r in rows if r['seed']==seed and r['side']==side and r['policy']=='teacher_task')
            fields = ('net_profit','opponent_net_profit','margin','score')
            for row in (a,b):
                if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in fields):
                    raise ValueError('Nonfinite or malformed economic value')
            own = a['net_profit']-b['net_profit']
            rival = a['opponent_net_profit']-b['opponent_net_profit']
            margin = a['margin']-b['margin']
            if abs(margin-(own-rival))>1e-7: raise ValueError('Margin accounting mismatch')
            relative = own/abs(b['net_profit']) if abs(b['net_profit'])>=cfg['acceptance']['relative_denominator_abs_min'] else None
            paired.append(dict(seed=seed,side=side,candidate_own_profit=a['net_profit'],teacher_own_profit=b['net_profit'],
                candidate_opponent_profit=a['opponent_net_profit'],teacher_opponent_profit=b['opponent_net_profit'],
                own_profit_difference=own,opponent_profit_difference=rival,margin_difference=margin,
                score_difference=a['score']-b['score'],relative_own_profit_difference=relative))
    groups = [dict(seed=s,mean_own_profit_difference=sum(p['own_profit_difference'] for p in paired if p['seed']==s)/2,
                   mean_margin_difference=sum(p['margin_difference'] for p in paired if p['seed']==s)/2) for s in seeds]
    gate = cfg['acceptance']
    failures = []
    for p in paired:
        if p['own_profit_difference']<gate['pair_own_income_delta_min']: failures.append(dict(seed=p['seed'],side=p['side'],metric='own_profit'))
        if p['margin_difference']<gate['pair_cash_margin_delta_min']: failures.append(dict(seed=p['seed'],side=p['side'],metric='margin'))
    for group in groups:
        if group['mean_own_profit_difference']<gate['seed_mean_own_income_delta_min']: failures.append(dict(seed=group['seed'],metric='seed_mean_own_profit'))
        if group['mean_margin_difference']<gate['seed_mean_cash_margin_delta_min']: failures.append(dict(seed=group['seed'],metric='seed_mean_margin'))
    return dict(pairs=paired,seed_groups=groups,exploratory_gate_passed=not failures,gate_failures=failures)


def write_marker(out,name,payload,state):
    path = (Path(out)/name).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        state['created'] = True
        json.dump(payload,stream,indent=2,allow_nan=False)
        stream.write('\n')
    state['complete'] = True


def trace_summary(out=OUT):
    from .campaign import verify_independent_review
    out = Path(out).resolve(); cfg = ready(out); verify_commit(out)
    verify_independent_review(out,cfg)
    games, ledger = read(out/'games-result.json'),read(out/'game-ledger.json')
    if (not games['passed'] or games['game_ledger_sha256']!=sha(out/'game-ledger.json')
            or games['audit_ledger_sha256']!=sha(out/'game-audit-ledger.json')):
        raise ValueError('Accepted game panel required')
    validate_panel(ledger,cfg)
    if (out/'trace-summary-started.json').exists(): raise ValueError('Trace attempt already consumed')
    marker = dict(created=False,complete=False); meter = None; summaries = []; tick = time.monotonic()
    limits = cfg['analysis_limits']
    try:
        write_marker(out,'trace-summary-started.json',dict(attempt=1,retry_allowed=False,
            freeze_sha256=sha(out/'freeze.json'),games_result_sha256=sha(out/'games-result.json')),marker)
        meter = Meter(limits['compressed_bytes'],limits['decoded_bytes'])
        for row in ledger:
            check_budget(tick,limits['seconds'])
            game, certificate = current_game(out,row)
            stream = trace_lines(raw_path(out,game['trace'],'trace'),game['trace'],meter)
            counts = dict(callbacks=0,fertilizer_opportunities=0,fertilizer_selections=0,fertilizer_work_requests=0)
            try:
                for step in range(719):
                    check_budget(tick,limits['seconds'])
                    line = next(stream)
                    task = cfg['candidate_checkpoint_sha256'] if row['policy']=='candidate' else None
                    turn = count_turn(line,step,row['policy'],task,cfg['crop_checkpoint_sha256'])
                    for name,value in turn.items(): counts[name] += value
                if next(stream,None) is not None: raise ValueError('Trace has extra turns')
            finally:
                stream.close()
            model_counts = certificate['live_model_counts']
            if (row['policy']=='candidate' and counts['callbacks']!=model_counts['task_forwards']) or (
                    row['policy']=='teacher_task' and model_counts['task_forwards']!=0):
                raise ValueError('Task model callback count mismatch')
            official = certificate['official']['fertilizer_actor_request_status']
            summaries.append(dict(game=row['game'],seed=row['seed'],side=row['side'],policy=row['policy'],
                **counts,official_fertilizer_events=official,trace_sha256=game['trace']['sha256']))
        check_budget(tick,limits['seconds'])
        snapshot = meter.snapshot()
        if (len(summaries)!=8 or snapshot['files_opened']!=8 or snapshot['files_completed']!=8
                or snapshot['raw_decode_passes_per_file']!=1): raise ValueError('Trace plan incomplete')
        result = dict(passed=True,games=8,summaries=summaries,raw_reads=snapshot,official_transitions=0,
            new_games=0,model_inference=0,training_updates=0,game_ledger_sha256=sha(out/'game-ledger.json'),
            freeze_sha256=sha(out/'freeze.json'),retry_allowed=False,
            interpretation='Trace summaries count callbacks and requests; official status records atomic change, not long-term completion or profit causation.')
        write_new(out/'trace-summary.json',result)
        return result
    except BaseException as exc:
        if marker['created']:
            try:
                write_new(out/'trace-summary-failure.local.json',dict(error_kind=type(exc).__name__,marker=marker,
                    completed_games=len(summaries),raw_reads=meter.snapshot() if meter else None,
                    partial_raw_reads_not_refunded=True,retry_allowed=False))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


def evaluate(out=OUT):
    from .campaign import verify_independent_review
    out = Path(out).resolve(); cfg = ready(out); verify_commit(out)
    verify_independent_review(out,cfg)
    if (out/'evaluation-started.json').exists(): raise ValueError('Evaluation already consumed')
    games, summary = read(out/'games-result.json'),read(out/'trace-summary.json')
    if (not games['passed'] or not summary['passed'] or summary['game_ledger_sha256']!=sha(out/'game-ledger.json')
            or summary['freeze_sha256']!=sha(out/'freeze.json')):
        raise ValueError('Accepted game and trace data required')
    marker = dict(created=False,complete=False)
    try:
        write_marker(out,'evaluation-started.json',dict(attempt=1,retry_allowed=False,
            game_ledger_sha256=sha(out/'game-ledger.json'),trace_summary_sha256=sha(out/'trace-summary.json')),marker)
        metrics = paired_metrics(read(out/'game-ledger.json'),cfg)
        result = dict(passed=True,**metrics,game_ledger_sha256=sha(out/'game-ledger.json'),
            trace_summary_sha256=sha(out/'trace-summary.json'),independent_review_sha256=sha(out/'independent-result-review.json'),
            interpretation='Two seed clusters screen preset declines only; no statistical significance, competitive gain or automatic promotion.',
            retry_allowed=False)
        write_new(out/'evaluation-result.json',result)
        return result
    except BaseException as exc:
        if marker['created']:
            try: write_new(out/'evaluation-failure.local.json',dict(error_kind=type(exc).__name__,marker=marker,retry_allowed=False))
            except BaseException as receipt: raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('traces','evaluate'))
    args=parser.parse_args()
    result = trace_summary() if args.command=='traces' else evaluate()
    print(json.dumps(dict(passed=result['passed'],command=args.command)))
