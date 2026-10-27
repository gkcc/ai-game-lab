"""Run eight preregistered official task-selector evaluation games once."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from learning.task_anchor_once_v1.processes import CapturedProcess
from learning.task_action_observation_v1.campaign import observe_processes
from learning.task_launcher_contract_repair_v1.observer import reap
from learning.task_completion_bc_reentry_v1.campaign import FitFailure, require_cleanup, storage_ready
from .experiment import OUT, ROOT, deadline_check, read, ready, sha, verify_commit, write_new


def plans(cfg):
    from .games import validate_plan
    opponent = cfg['opponents'][0]
    queue = [dict(game=f'holdout-{seed}-{opponent}-{side}-{policy}',policy=policy,split='holdout',
                  seed=seed,opponent=opponent,side=side,parity_kind=None)
             for seed in cfg['test_seeds'] for side in (0,1) for policy in ('candidate','teacher_task')]
    for plan in queue: validate_plan(plan,cfg)
    if len(queue)!=8 or len({p['game'] for p in queue})!=8:
        raise ValueError('Unexpected game plan')
    return queue


def write_marker(out, value, state):
    path = (Path(out)/'games-started.json').resolve()
    if not path.is_relative_to(OUT): raise ValueError('Start marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        state['created'] = True
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    state['complete'] = True


def verify_independent_review(out, cfg):
    out = Path(out)
    review = read(out/'independent-result-review.json')
    replay = read(out/'independent-replay-verification.json')
    key = f"holdout-{cfg['test_seeds'][0]}-{cfg['opponents'][0]}-0-candidate"
    required = {'freeze_sha256':sha(out/'freeze.json'), 'games_result_sha256':sha(out/'games-result.json'),
        'game_ledger_sha256':sha(out/'game-ledger.json'), 'game_audit_ledger_sha256':sha(out/'game-audit-ledger.json'),
        'verification_sha256':sha(out/'independent-replay-verification.json'),
        'method_sha256':sha(ROOT/'learning/task_completion_economy_v1/independent_method.py')}
    raw, limits = replay.get('raw_reads',{}), cfg['independent_limits']
    if (not review['passed'] or any(review.get(k)!=v for k,v in required.items())
            or not replay['passed'] or replay['game']!=key or replay['official_transitions']!=719
            or replay['sample_count']!=20 or replay['matched']!=20
            or replay['selection_rule']!=cfg['independent_selection']
            or raw.get('files_opened')!=2 or raw.get('files_completed')!=2
            or raw.get('raw_decode_passes_per_file')!=1
            or not 0<raw.get('compressed_input_read_bytes',0)<=limits['compressed_bytes']
            or not 0<raw.get('decoded_input_bytes',0)<=limits['decoded_bytes']
            or review.get('official_transitions')!=719 or review.get('sample_count')!=20
            or review.get('sample_matched')!=20):
        raise ValueError('Independent replay does not match registered panel')
    return review


def games(out=OUT):
    from .games import make_wrapper
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if (out/'games-started.json').exists(): raise ValueError('Game attempt already consumed')
    queue = plans(cfg)
    rows, certs, cleanups, attempted = [], [], [], 0
    error, child, known = None, None, {}
    marker = dict(created=False,complete=False)
    phase, failure_phase, cleanup_error = 'start_marker', None, None
    spawn_entered, active_plan, last_cleanup = False, None, None
    environment = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        write_marker(out,dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json'),fixed_plan=queue),marker)
        for plan in queue:
            phase = 'job_configuration'
            deadline_check(); ready(out)
            storage_ready(out,cfg,(8-len(rows))*cfg['storage']['game_write_bytes']+256*1024**2)
            bot = make_wrapper(out,plan,cfg)
            rival = out/'sources'/(plan['opponent']+'.py')
            checkpoint = cfg['candidate_checkpoint_sha256'] if plan['policy']=='candidate' else None
            item = dict(plan,bot_sha256=sha(bot),rival_sha256=sha(rival),task_checkpoint_sha256=checkpoint)
            write_new(out/'job-inputs'/(plan['game']+'.json'),item)
            attempted += 1
            active_plan = plan['game']
            tick, known, last_cleanup = time.monotonic(), {}, None
            phase, spawn_entered = 'spawn', True
            child = CapturedProcess([sys.executable,'-B','-X','utf8','-m',
                'learning.task_completion_economy_v1.games','--out',out.relative_to(ROOT).as_posix(),
                '--key',plan['game']],cwd=ROOT,env=environment,
                log=out/'job-local'/(plan['game']+'.log'),limit_bytes=cfg['storage']['process_log_bytes'])
            phase = 'monitor'
            while child.process.poll() is None:
                deadline_check(); observe_processes(child,known); child.check()
                if time.monotonic()-tick >= cfg['official_limits']['per_job_seconds']:
                    raise TimeoutError('Official job deadline exceeded')
                time.sleep(.2)
            capture = child.finish()
            phase = 'cleanup'
            cleanup = reap(child,known); last_cleanup = cleanup; cleanups.append(cleanup); child = None
            require_cleanup(cleanup)
            phase = 'job_acceptance'
            if capture['child_exit_code'] != 0: raise RuntimeError('Official worker failed')
            key = plan['game']
            files = {name:out/'job-results'/(key+'.'+name+'.json') for name in ('row','audit','complete','storage')}
            row,audit,complete,storage = (read(files[name]) for name in ('row','audit','complete','storage'))
            counts = audit['live_model_counts']
            fertilizer = audit['official']['fertilizer_actor_request_status']
            if (not complete['passed'] or complete['game']!=key or complete['row_sha256']!=sha(files['row'])
                    or complete['certificate_sha256']!=sha(files['audit']) or not storage['passed']
                    or storage['error_kind'] is not None or storage['output']['refused']!=0 or not row['passed']
                    or any(row[k]!=v for k,v in item.items()) or row['states']!=720 or row['action_turns']!=719
                    or row['status']!=['DONE','DONE'] or row['agent_errors']!=0
                    or not audit['passed'] or audit['errors']!=0 or audit['actor_actions_reproduced']!=719
                    or audit['rival_actions_reproduced']!=719 or audit['trace_turns_reproduced']!=719
                    or audit['official']['official_transitions_reexecuted']!=719
                    or audit['official']['engine_sha256']!=cfg['engine_sha256']
                    or audit['official']['final_rewards']!=row['banks']
                    or fertilizer['requested'] != sum(fertilizer[k] for k in
                        ('executed_with_observed_change','blocked','no_effect','error','other'))
                    or counts['crop_forwards']<=0
                    or (plan['policy']=='candidate' and (counts['task_forwards']<=0 or counts['task_rows']<=0))
                    or (plan['policy']=='teacher_task' and (counts['task_forwards']!=0 or counts['task_rows']!=0))):
                raise ValueError('Official job acceptance failed')
            rows.append(row)
            certs.append(dict(game=key,certificate_sha256=sha(files['audit']),
                replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'],
                official_transitions=719,cleanup=cleanup))
            active_plan, spawn_entered = None, False
            print(json.dumps(dict(completed_games=len(rows),attempted_games=attempted)),flush=True)
        if len(rows)!=8 or len(certs)!=8: raise ValueError('Incomplete panel')
        write_new(out/'game-ledger.json',rows)
        write_new(out/'game-audit-ledger.json',certs)
        result = dict(passed=True,new_games=8,complete_games=8,live_official_transitions=8*719,
            primary_replay_transitions=8*719,model_deployed_in_games=True,
            candidate_games=4,teacher_task_games=4,all_owned_processes_reaped=True,
            game_ledger_sha256=sha(out/'game-ledger.json'),audit_ledger_sha256=sha(out/'game-audit-ledger.json'),
            no_retry=True)
        write_new(out/'games-result.json',result)
        return result
    except BaseException as exc:
        error, failure_phase = type(exc).__name__, phase
    finally:
        if child is not None:
            try:
                observe_processes(child,known)
                last_cleanup = reap(child,known); cleanups.append(last_cleanup)
                require_cleanup(last_cleanup)
            except BaseException as exc:
                cleanup_error = type(exc).__name__
        if (error or cleanup_error) and marker['created']:
            receipt = dict(error_kind=error,failure_phase=failure_phase,cleanup_error_kind=cleanup_error,
                attempted_games=attempted,completed_games=len(rows),retry_allowed=False,
                marker_created=marker['created'],marker_complete=marker['complete'],
                child_lifecycle=('handle_acquired' if child is not None else
                    'unknown_after_spawn_attempt' if spawn_entered and last_cleanup is None else
                    'cleanup_recorded' if last_cleanup is not None else 'not_started'),
                uncertain_game=active_plan,possible_official_steps='unknown' if active_plan else 0,
                last_cleanup=last_cleanup,cleanup_results=cleanups,partial_work_not_refunded=True)
            try: write_new(out/'games-failure.local.json',receipt)
            except BaseException as exc: raise FitFailure(error or cleanup_error,failure_phase or 'cleanup',type(exc).__name__) from None
    if error or cleanup_error: raise FitFailure(error or cleanup_error,failure_phase or 'cleanup')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('games',))
    args = parser.parse_args()
    try:
        result = games()
        print(json.dumps(dict(passed=result['passed'],command=args.command)),flush=True)
    except BaseException as exc:
        print(json.dumps(dict(passed=False,command=args.command,
            error_kind=getattr(exc,'primary_error_kind',None) or type(exc).__name__,retry_allowed=False)),flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__': main()
