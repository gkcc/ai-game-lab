"""Bounded orchestration of training, one fixed collection panel and evaluation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time

from learning.task_anchor_once_v1.processes import CapturedProcess
from learning.task_action_observation_v1.campaign import observe_processes, owned_size
from learning.task_launcher_contract_repair_v1.observer import reap
from .experiment import OUT, ROOT, deadline_check, read, ready, sha, verify_commit, write_new


def storage_ready(out, cfg, reserve):
    if owned_size(out) + reserve > cfg['storage']['total_bytes']:
        raise ValueError('Registered output budget would be exceeded')
    if shutil.disk_usage(out).free < cfg['storage']['free_floor_bytes'] + reserve:
        raise ValueError('Insufficient reserved free space')


def require_cleanup(cleanup):
    if (not isinstance(cleanup, dict) or cleanup.get('owned_instances_remaining') != 0
            or cleanup.get('reader_joined') is not True or cleanup.get('launched_process_reaped') is not True):
        raise RuntimeError('Official worker cleanup not proven')


def verify_model_seal(out):
    out = Path(out)
    seal, fit = read(out/'model-seal.json'), read(out/'fit-result.json')
    transport, source = read(out/'fit-transport-result.json'), read(out/'train-data-result.json')
    summary = fit['training_summary']
    parity = summary['inference_parity']
    if (not seal['passed'] or seal['freeze_sha256'] != sha(out/'freeze.json')
            or seal['fit_result_sha256'] != sha(out/'fit-result.json')
            or seal['fit_transport_sha256'] != sha(out/'fit-transport-result.json')
            or not transport['passed'] or not source['passed']
            or fit['train_data_sha256'] != sha(out/'train-data-result.json')
            or not fit['passed'] or fit['updates'] != 128 or summary['synchronized_updates'] != 128
            or not summary['gpu_verified'] or not summary['weights_changed']
            or summary['nonzero_gradient_updates'] <= 0 or parity['groups'] != 16
            or parity['candidate_rows'] < 16 or parity['synchronized'] is not True
            or parity['requested_matmul_precision'] != 'highest'
            or parity['hardware_precision_measured'] is not False
            or seal['checkpoint_sha256'] != fit['checkpoint_sha256']
            or sha(out/'private/model-final.npz') != seal['checkpoint_sha256']):
        raise ValueError('Model, fit, baseline or source seal changed')
    return seal, fit


def verify_independent_review(out, cfg):
    out = Path(out)
    review = read(out/'independent-result-review.json')
    replay = read(out/'independent-replay-verification.json')
    first = cfg['test_seeds'][0]
    selected = f'holdout-{first}-round2-0-teacher_task'
    required = {
        'freeze_sha256': sha(out/'freeze.json'),
        'model_seal_sha256': sha(out/'model-seal.json'),
        'games_result_sha256': sha(out/'games-result.json'),
        'game_ledger_sha256': sha(out/'game-ledger.json'),
        'game_audit_ledger_sha256': sha(out/'game-audit-ledger.json'),
        'test_data_result_sha256': sha(out/'test-data-result.json'),
        'test_label_ledger_sha256': sha(out/'test-label-ledger.json'),
        'verification_sha256': sha(out/'independent-replay-verification.json'),
        'method_sha256': sha(ROOT/'learning/task_completion_bc_reentry_v1/independent_method.py')}
    reads = replay.get('raw_reads', {})
    limits = cfg['independent_limits']
    if (not review['passed'] or any(review.get(k) != v for k, v in required.items())
            or not replay['passed'] or replay['game'] != selected
            or replay['official_transitions'] != 719 or replay['sample_count'] != 20
            or replay['matched'] != 20 or replay['selection_rule'] != cfg['independent_sample']
            or reads.get('files_opened') != 2 or reads.get('files_completed') != 2
            or reads.get('raw_decode_passes_per_file') != 1
            or not 0 < reads.get('compressed_input_read_bytes', 0) <= limits['compressed_bytes']
            or not 0 < reads.get('decoded_input_bytes', 0) <= limits['decoded_bytes']
            or review.get('official_transitions') != 719 or review.get('sample_count') != 20
            or review.get('sample_matched') != 20):
        raise ValueError('Independent review does not bind this exact panel and sample')
    return review


class FitFailure(RuntimeError):
    def __init__(self, primary_error_kind, failure_phase, receipt_error_kind=None):
        super().__init__('Fit failed; attempt consumed')
        self.primary_error_kind = primary_error_kind
        self.failure_phase = failure_phase
        self.receipt_error_kind = receipt_error_kind


def write_fit_marker(out, value, state):
    path = (Path(out)/'fit-started.json').resolve()
    if not path.is_relative_to(OUT): raise ValueError('Start marker outside experiment')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        state['created'] = True
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    state['complete'] = True


def gpu_fit(out=OUT):
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if not read(out/'train-data-result.json')['passed'] or (out/'fit-started.json').exists():
        raise ValueError('Training not ready or attempt already consumed')
    storage_ready(out, cfg, 32*1024**2)
    child, known, error, cleanup, capture = None, {}, None, None, None
    phase, failure_phase = 'start_marker', None
    marker, spawn_entered = {'created': False, 'complete': False}, False
    cancel_error = guard_read_error = cleanup_error = None
    tick = time.monotonic()
    try:
        write_fit_marker(out, dict(attempt=1, retry_allowed=False,
            freeze_sha256=sha(out/'freeze.json'), train_data_sha256=sha(out/'train-data-result.json'),
            test_data_allowed=False), marker)
        phase = 'configuration'
        launch = read(out/'gpu-launch.local.json')
        command = ['wsl.exe', '-d', launch['distro'], '--cd', launch['workdir'], '--', launch['python'],
                   '-B', '-m', 'learning.task_completion_bc_reentry_v1.gpu_entry', 'guard',
                   '--out', out.relative_to(ROOT).as_posix()]
        phase = 'transport_start'
        spawn_entered = True
        child = CapturedProcess(command, cwd=ROOT, log=out/'gpu-process.local.log',
                                limit_bytes=cfg['storage']['process_log_bytes'])
        phase = 'transport_monitor'
        while child.process.poll() is None:
            deadline_check()
            observe_processes(child, known)
            child.check()
            if time.monotonic()-tick >= cfg['training']['process_seconds']-20:
                raise TimeoutError('GPU transport work allowance exhausted')
            time.sleep(.2)
        capture = child.finish()
        if child.process.returncode != 0:
            raise RuntimeError('GPU transport failed')
        phase = 'guard_result'
        guard = read(out/'gpu-guard-result.json')
        if not guard['passed']:
            raise RuntimeError('GPU guard did not accept fit')
    except BaseException as exc:
        error = type(exc).__name__
        failure_phase = phase
        if child is not None:
            try:
                if child.process.poll() is None:
                    if not (out/'gpu-cancel.local.json').exists():
                        write_new(out/'gpu-cancel.local.json', dict(cancel=True, reason='owned transport stopping'))
                    child.process.wait(timeout=20)
            except BaseException as secondary:
                cancel_error = type(secondary).__name__
    finally:
        if child is not None:
            try:
                observe_processes(child, known)
                cleanup = reap(child, known)
            except BaseException as exc:
                cleanup_error = type(exc).__name__
        try:
            guard = read(out/'gpu-guard-result.json') if child is not None and (out/'gpu-guard-result.json').exists() else None
        except BaseException as exc:
            guard, guard_read_error = None, type(exc).__name__
        flags = ('owned_process_group_signalled','direct_child_reaped','no_live_owned_session_members_before_reap')
        supervision = guard.get('supervision') if isinstance(guard, dict) else None
        linux_cleanup = all(supervision.get(k) is True for k in flags) if isinstance(supervision, dict) else None
        try:
            require_cleanup(cleanup)
            windows_clean = True
        except RuntimeError:
            windows_clean = False
        passed = (error is None and cleanup_error is None and cancel_error is None and guard_read_error is None
                  and windows_clean and linux_cleanup is True
                  and time.monotonic()-tick <= cfg['training']['process_seconds'])
        result = dict(passed=passed, error_kind=error, failure_phase=failure_phase,
                      cleanup_error_kind=cleanup_error, cancel_error_kind=cancel_error, guard_read_error_kind=guard_read_error,
                      marker_created=marker['created'], marker_complete=marker['complete'],
                      child_handle_acquired=child is not None,
                      child_lifecycle=('handle_acquired' if child is not None else 'unknown_after_spawn_attempt' if spawn_entered else 'not_started'),
                      windows_cleanup=cleanup, linux_session_cleanup_proven=linux_cleanup,
                      capture=capture, elapsed_seconds=time.monotonic()-tick, retry_allowed=False)
        if marker['created']:
            try:
                write_new(out/'fit-transport-result.json', result)
            except BaseException as exc:
                raise FitFailure(error, failure_phase, type(exc).__name__) from None
    if not result['passed']:
        raise FitFailure(error or 'CleanupFailure', failure_phase or 'cleanup')
    try:
        return _seal_fit(out)
    except BaseException as exc:
        try:
            write_new(out/'fit-seal-failure.json', dict(error_kind=type(exc).__name__, retry_allowed=False))
        except BaseException as receipt_exc:
            raise FitFailure(type(exc).__name__, 'seal', type(receipt_exc).__name__) from None
        raise


def _seal_fit(out):
    fit = read(out/'fit-result.json')
    summary = fit['training_summary']
    model = out/'private/model-final.npz'
    parity = summary['inference_parity']
    if (not fit['passed'] or fit['updates'] != 128 or summary['synchronized_updates'] != 128
            or not summary['gpu_verified'] or not summary['weights_changed']
            or summary['nonzero_gradient_updates'] <= 0 or parity['groups'] != 16
            or parity['candidate_rows'] < 16 or parity['synchronized'] is not True
            or parity['requested_matmul_precision'] != 'highest'
            or parity['hardware_precision_measured'] is not False
            or sha(model) != fit['checkpoint_sha256']):
        raise ValueError('Fit result failed training identity checks')
    write_new(out/'model-seal.json', dict(passed=True, checkpoint_sha256=fit['checkpoint_sha256'],
        fit_result_sha256=sha(out/'fit-result.json'), fit_transport_sha256=sha(out/'fit-transport-result.json'),
        freeze_sha256=sha(out/'freeze.json'),
        training_closed_before_test_games=True, no_checkpoint_selection=True,
        checkpoint_is_private=True, retry_allowed=False))
    return read(out/'model-seal.json')


def plans(cfg):
    from .games import validate_plan
    result = [dict(game=f'holdout-{seed}-{opponent}-{side}-teacher_task',
                   policy='teacher_task', split='holdout', seed=seed, opponent=opponent,
                   side=side, parity_kind=None)
              for seed in cfg['test_seeds'] for opponent in cfg['opponents'] for side in cfg['seats']]
    if len(result) != 8 or len({p['game'] for p in result}) != 8:
        raise ValueError('Unexpected collection plan')
    for plan in result:
        validate_plan(plan, cfg)
    return result


def write_games_marker(out, value, state):
    path = (Path(out)/'games-started.json').resolve()
    if not path.is_relative_to(OUT): raise ValueError('Start marker outside experiment')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        state['created'] = True
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    state['complete'] = True


def games(out=OUT):
    from .games import make_wrapper
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    seal, _ = verify_model_seal(out)
    if (out/'games-started.json').exists():
        raise ValueError('Test collection attempt already consumed')
    queue = plans(cfg)
    rows, certs, cleanups, attempted = [], [], [], 0
    error, child, known = None, None, {}
    marker = dict(created=False, complete=False)
    phase, failure_phase, cleanup_error = 'start_marker', None, None
    spawn_entered, active_plan, last_cleanup = False, None, None
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        write_games_marker(out, dict(attempt=1, retry_allowed=False,
            model_seal_sha256=sha(out/'model-seal.json'), fixed_plan=queue), marker)
        for plan in queue:
            phase = 'job_configuration'
            deadline_check()
            ready(out)
            storage_ready(out, cfg, (8-len(rows))*cfg['storage']['game_write_bytes'] + 512*1024**2)
            bot = make_wrapper(out, plan, cfg)
            rival = out/'sources'/(plan['opponent']+'.py')
            item = dict(plan, bot_sha256=sha(bot), rival_sha256=sha(rival), task_checkpoint_sha256=None)
            write_new(out/'job-inputs'/(plan['game']+'.json'), item)
            attempted += 1
            active_plan = plan['game']
            tick, known = time.monotonic(), {}
            last_cleanup = None
            phase = 'spawn'
            spawn_entered = True
            child = CapturedProcess([sys.executable, '-B', '-X', 'utf8', '-m',
                'learning.task_completion_bc_reentry_v1.games', '--out', out.relative_to(ROOT).as_posix(),
                '--key', plan['game']], cwd=ROOT, env=environment,
                log=out/'job-local'/(plan['game']+'.log'), limit_bytes=cfg['storage']['process_log_bytes'])
            phase = 'monitor'
            while child.process.poll() is None:
                deadline_check()
                observe_processes(child, known)
                child.check()
                if time.monotonic()-tick >= cfg['official_limits']['per_job_seconds']:
                    raise TimeoutError('Official job deadline exceeded')
                time.sleep(.2)
            capture = child.finish()
            phase = 'cleanup'
            cleanup = reap(child, known)
            last_cleanup = cleanup
            cleanups.append(cleanup)
            child = None
            require_cleanup(cleanup)
            phase = 'job_acceptance'
            if capture['child_exit_code'] != 0:
                raise RuntimeError('Official worker failed; inspect its local bounded log')
            key = plan['game']
            files = {name: out/'job-results'/(key+'.'+name+'.json')
                     for name in ('row','audit','complete','storage')}
            row, audit, complete, storage = (read(files[name]) for name in ('row','audit','complete','storage'))
            if (capture['child_exit_code'] != 0 or not complete['passed'] or complete['game'] != key
                    or complete['row_sha256'] != sha(files['row']) or complete['certificate_sha256'] != sha(files['audit'])
                    or not storage['passed'] or storage['error_kind'] is not None
                    or storage['output']['refused'] != 0 or not row['passed']
                    or any(row[k] != v for k, v in item.items()) or row['states'] != 720
                    or row['action_turns'] != 719 or row['status'] != ['DONE','DONE'] or row['agent_errors'] != 0
                    or not audit['passed'] or audit['errors'] != 0
                    or audit['actor_actions_reproduced'] != 719 or audit['rival_actions_reproduced'] != 719
                    or audit['trace_turns_reproduced'] != 719
                    or audit['official']['official_transitions_reexecuted'] != 719
                    or audit['official']['engine_sha256'] != cfg['engine_sha256']
                    or audit['official']['final_rewards'] != row['banks']):
                raise ValueError('Official job acceptance failed')
            rows.append(row)
            certs.append(dict(game=key, certificate_sha256=sha(files['audit']),
                              replay_sha256=row['replay_gzip_sha256'], trace_sha256=row['trace_sha256'],
                              official_transitions=719, cleanup=cleanup))
            active_plan = None
            spawn_entered = False
            print(json.dumps(dict(completed_games=len(rows), attempted_games=attempted)), flush=True)
        if len(rows) != len(queue) or len(certs) != 8:
            raise ValueError('Incomplete holdout collection')
        write_new(out/'game-ledger.json', rows)
        write_new(out/'game-audit-ledger.json', certs)
        result = dict(passed=True, new_games=8, complete_games=8, live_official_transitions=8*719,
            primary_replay_transitions=8*719, model_deployed_in_games=False,
            collector='fixed teacher task choice with frozen crop model',
            purpose='offline held-out task choices, not a candidate competitive evaluation',
            all_owned_processes_reaped=all(x['owned_instances_remaining']==0 for x in cleanups),
            game_ledger_sha256=sha(out/'game-ledger.json'), audit_ledger_sha256=sha(out/'game-audit-ledger.json'),
            no_retry=True)
        write_new(out/'games-result.json', result)
        return result
    except BaseException as exc:
        error = type(exc).__name__
        failure_phase = phase
    finally:
        if child is not None:
            try:
                observe_processes(child, known)
                last_cleanup = reap(child, known)
                cleanups.append(last_cleanup)
                require_cleanup(last_cleanup)
            except BaseException as exc:
                cleanup_error = type(exc).__name__
        if (error or cleanup_error) and marker['created']:
            receipt = dict(error_kind=error, failure_phase=failure_phase, cleanup_error_kind=cleanup_error,
                attempted_games=attempted, completed_games=len(rows), retry_allowed=False,
                marker_created=marker['created'], marker_complete=marker['complete'],
                child_lifecycle=('handle_acquired' if child is not None else
                    'unknown_after_spawn_attempt' if spawn_entered and last_cleanup is None else
                    'cleanup_recorded' if last_cleanup is not None else 'not_started'),
                uncertain_game=active_plan, possible_official_steps='unknown' if active_plan else 0,
                last_cleanup=last_cleanup, cleanup_results=cleanups, partial_work_not_refunded=True)
            try:
                write_new(out/'games-failure.local.json', receipt)
            except BaseException as exc:
                raise FitFailure(error or cleanup_error, failure_phase or 'cleanup', type(exc).__name__) from None
    if error or cleanup_error:
        raise FitFailure(error or cleanup_error, failure_phase or 'cleanup')


def test_data(out=OUT):
    from .collect import test
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    verify_model_seal(out)
    storage_ready(out, cfg, 1024**3)
    result = test(out)
    verify_model_seal(out)
    return result


def evaluate(out=OUT):
    from learning.task_bc_v1.model import TaskNetwork
    from learning.task_completion_bc_v1 import numeric
    from . import training
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if (out/'evaluation-started.json').exists():
        raise ValueError('Holdout evaluation already consumed')
    source = read(out/'test-data-result.json')
    seal, fit = verify_model_seal(out)
    review = verify_independent_review(out, cfg)
    if (not source['passed'] or source['model_seal_sha256'] != sha(out/'model-seal.json')
            or source['games_result_sha256'] != sha(out/'games-result.json')
            or source['label_ledger_sha256'] != sha(out/'test-label-ledger.json')):
        raise ValueError('Data, model or independent replay review not accepted')
    marker = dict(created=False, complete=False)
    data_read_started = False
    try:
        marker_path = (out/'evaluation-started.json').resolve()
        if not marker_path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
        with marker_path.open('x', encoding='utf-8', newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1, retry_allowed=False, model_sha256=seal['checkpoint_sha256'],
                data_result_sha256=sha(out/'test-data-result.json'),
                independent_review_sha256=sha(out/'independent-result-review.json')), stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        marker['complete'] = True
        path = out/'private/test.npz'
        data_read_started = True
        if sha(path) != source['array_sha256']:
            raise ValueError('Test numeric data changed')
        limits = cfg['data_limits']
        arrays = numeric.read(path, limits['max_groups_per_split'], limits['max_candidates_per_split'])
        groups = numeric.unpack(arrays, limits['max_groups_per_split'], limits['max_candidates_per_split'])
        if set(g['seed'] for g in groups) != set(cfg['test_seeds']):
            raise ValueError('Holdout seed role differs')
        model = TaskNetwork(out/'private/model-final.npz', seal['checkpoint_sha256'])
        metrics = training.evaluate(groups, model, fit['baseline'], set(cfg['train_seeds']), holdout=True)
        gate = numeric.exploratory_gate(metrics, cfg['test_seeds'])
        result = dict(passed=True, exploratory_gate=gate, metrics=metrics,
            independent_result_review_sha256=sha(out/'independent-result-review.json'),
            model_sha256=seal['checkpoint_sha256'], test_array_sha256=source['array_sha256'],
            population='all test episode origins; completed and unknown also reported separately',
            candidate_scope='frozen controller opportunities, not all official legal actions',
            interpretation='Two new seeds support an exploratory offline signal only; no statistical or game-strength claim',
            no_candidate_model_official_matches=True, no_promotion=True, retry_allowed=False)
        write_new(out/'evaluation-result.json', result)
        return result
    except BaseException as exc:
        if marker['created']:
            try:
                write_new(out/'evaluation-failure.local.json', dict(error_kind=type(exc).__name__, retry_allowed=False,
                    marker_created=marker['created'], marker_complete=marker['complete'],
                    test_data_read_started=data_read_started,
                    test_data_read_completion='unknown' if data_read_started else 'not_started'))
            except BaseException as receipt_exc:
                raise FitFailure(type(exc).__name__, 'evaluation', type(receipt_exc).__name__) from None
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('gpu-fit','games','test-data','evaluate'))
    args = parser.parse_args()
    function = {'gpu-fit':gpu_fit, 'games':games,
                'test-data':test_data, 'evaluate':evaluate}[args.command]
    try:
        result = function()
        print(json.dumps(dict(passed=result['passed'], command=args.command), ensure_ascii=False), flush=True)
    except BaseException as exc:
        print(json.dumps(dict(passed=False, command=args.command, error_kind=getattr(exc, 'primary_error_kind', None) or type(exc).__name__, retry_allowed=False)), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
