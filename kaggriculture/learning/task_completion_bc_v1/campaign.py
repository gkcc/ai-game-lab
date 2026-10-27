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
        'verification_sha256': sha(out/'independent-replay-verification.json')}
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


def train_data(out=OUT):
    from .collect import train
    cfg = ready(out)
    verify_commit(out)
    storage_ready(out, cfg, 512*1024**2)
    return train(out)


def gpu_fit(out=OUT):
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if not read(out/'train-data-result.json')['passed'] or (out/'fit-started.json').exists():
        raise ValueError('Training not ready or attempt already consumed')
    storage_ready(out, cfg, 32*1024**2)
    write_new(out/'fit-started.json', dict(attempt=1, retry_allowed=False,
        freeze_sha256=sha(out/'freeze.json'), train_data_sha256=sha(out/'train-data-result.json'),
        test_data_allowed=False))
    launch = read(out/'gpu-launch.local.json')
    command = ['wsl.exe', '-d', launch['distro'], '--cd', launch['workdir'], '--', launch['python'],
               '-B', '-m', 'learning.task_completion_bc_v1.gpu_entry', 'guard',
               '--out', out.relative_to(ROOT).as_posix()]
    child, known, error, cleanup, capture = None, {}, None, None, None
    tick = time.monotonic()
    try:
        child = CapturedProcess(command, cwd=ROOT, log=out/'gpu-process.local.log',
                                limit_bytes=cfg['storage']['process_log_bytes'])
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
        guard = read(out/'gpu-guard-result.json')
        if not guard['passed']:
            raise RuntimeError('GPU guard did not accept fit')
    except BaseException as exc:
        error = type(exc).__name__
        if child is not None and child.process.poll() is None:
            try:
                if not (out/'gpu-cancel.local.json').exists():
                    write_new(out/'gpu-cancel.local.json', dict(cancel=True, reason='owned transport stopping'))
                child.process.wait(timeout=20)
            except BaseException:
                pass
    finally:
        cleanup_error = None
        if child is not None:
            try:
                observe_processes(child, known)
                cleanup = reap(child, known)
            except BaseException as exc:
                cleanup_error = type(exc).__name__
        guard = read(out/'gpu-guard-result.json') if (out/'gpu-guard-result.json').exists() else None
        flags = ('owned_process_group_signalled','direct_child_reaped','no_live_owned_session_members_before_reap')
        linux_cleanup = bool(guard and guard.get('supervision') and all(guard['supervision'].get(k) is True for k in flags))
        passed = (error is None and cleanup_error is None and cleanup is not None
                  and cleanup['owned_instances_remaining'] == 0 and linux_cleanup
                  and time.monotonic()-tick <= cfg['training']['process_seconds'])
        result = dict(passed=passed, error_kind=error, cleanup_error_kind=cleanup_error,
                      windows_cleanup=cleanup, linux_session_cleanup_proven=linux_cleanup,
                      capture=capture, elapsed_seconds=time.monotonic()-tick, retry_allowed=False)
        write_new(out/'fit-transport-result.json', result)
    if not result['passed']:
        raise RuntimeError('GPU fit or owned cleanup failed; no retry')
    fit = read(out/'fit-result.json')
    summary = fit['training_summary']
    model = out/'private/model-final.npz'
    parity = summary['inference_parity']
    if (not fit['passed'] or fit['updates'] != 128 or summary['synchronized_updates'] != 128
            or not summary['gpu_verified'] or not summary['weights_changed']
            or summary['nonzero_gradient_updates'] <= 0 or parity['groups'] != 16
            or parity['candidate_rows'] < 16 or parity['synchronized'] is not True
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


def games(out=OUT):
    from .games import make_wrapper
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    seal, _ = verify_model_seal(out)
    if (out/'games-started.json').exists():
        raise ValueError('Test collection attempt already consumed')
    queue = plans(cfg)
    write_new(out/'games-started.json', dict(attempt=1, retry_allowed=False,
        model_seal_sha256=sha(out/'model-seal.json'), fixed_plan=queue))
    rows, certs, cleanups, attempted = [], [], [], 0
    error, child, known = None, None, {}
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        for plan in queue:
            deadline_check()
            ready(out)
            storage_ready(out, cfg, (8-len(rows))*cfg['storage']['game_write_bytes'] + 512*1024**2)
            bot = make_wrapper(out, plan, cfg)
            rival = out/'sources'/(plan['opponent']+'.py')
            item = dict(plan, bot_sha256=sha(bot), rival_sha256=sha(rival), task_checkpoint_sha256=None)
            write_new(out/'job-inputs'/(plan['game']+'.json'), item)
            attempted += 1
            tick, known = time.monotonic(), {}
            child = CapturedProcess([sys.executable, '-B', '-X', 'utf8', '-m',
                'learning.task_completion_bc_v1.games', '--out', out.relative_to(ROOT).as_posix(),
                '--key', plan['game']], cwd=ROOT, env=environment,
                log=out/'job-local'/(plan['game']+'.log'), limit_bytes=cfg['storage']['process_log_bytes'])
            while child.process.poll() is None:
                deadline_check()
                observe_processes(child, known)
                child.check()
                if time.monotonic()-tick >= cfg['official_limits']['per_job_seconds']:
                    raise TimeoutError('Official job deadline exceeded')
                time.sleep(.2)
            capture = child.finish()
            cleanup = reap(child, known)
            child = None
            require_cleanup(cleanup)
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
            cleanups.append(cleanup)
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
        raise
    finally:
        cleanup_error = None
        if child is not None:
            try:
                observe_processes(child, known)
                reap(child, known)
            except BaseException as exc:
                cleanup_error = type(exc).__name__
        if error or cleanup_error:
            write_new(out/'games-failure.local.json', dict(error_kind=error, cleanup_error_kind=cleanup_error,
                attempted_games=attempted, completed_games=len(rows), retry_allowed=False,
                partial_work_not_refunded=True))


def test_data(out=OUT):
    from .collect import test
    cfg = ready(out)
    verify_commit(out)
    storage_ready(out, cfg, 1024**3)
    return test(out)


def evaluate(out=OUT):
    from learning.task_bc_v1.model import TaskNetwork
    from . import numeric, training
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
    write_new(out/'evaluation-started.json', dict(attempt=1, retry_allowed=False,
        model_sha256=seal['checkpoint_sha256'], data_result_sha256=sha(out/'test-data-result.json'),
        independent_review_sha256=sha(out/'independent-result-review.json')))
    try:
        path = out/'private/test.npz'
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
        write_new(out/'evaluation-failure.local.json', dict(error_kind=type(exc).__name__, retry_allowed=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train-data','gpu-fit','games','test-data','evaluate'))
    args = parser.parse_args()
    function = {'train-data':train_data, 'gpu-fit':gpu_fit, 'games':games,
                'test-data':test_data, 'evaluate':evaluate}[args.command]
    try:
        result = function()
        print(json.dumps(dict(passed=result['passed'], command=args.command), ensure_ascii=False), flush=True)
    except BaseException as exc:
        print(json.dumps(dict(passed=False, command=args.command, error_kind=type(exc).__name__, retry_allowed=False)), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
