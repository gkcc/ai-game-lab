"""One supervised, read-only GPU/CPU inference diagnostic."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .experiment import OUT, PREVIOUS, ROOT, deadline_check, read, ready, sha, verify_commit, write_new


class ReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind, cleanup_error_kind):
        super().__init__('Could not persist the diagnostic receipt')
        self.primary_error_kind = primary_error_kind
        self.receipt_error_kind = receipt_error_kind
        self.cleanup_error_kind = cleanup_error_kind


class StartFailure(RuntimeError):
    def __init__(self, primary_error_kind, failure_phase, marker_created):
        super().__init__('Diagnostic execution or cleanup failed; no retry')
        self.primary_error_kind = primary_error_kind
        self.failure_phase = failure_phase
        self.marker_created = marker_created


def write_started_marker(out, value, state):
    path = (Path(out)/'run-started.json').resolve()
    if not path.is_relative_to(OUT):
        raise ValueError('Start marker outside owned diagnostic')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        state['created'] = True
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    state['complete'] = True


def cleanup_complete(cleanup):
    return (isinstance(cleanup, dict) and cleanup.get('owned_instances_remaining') == 0
            and cleanup.get('reader_joined') is True and cleanup.get('launched_process_reaped') is True)


def worker(out=OUT):
    out = Path(out).resolve()
    phase, progress = 'configuration', dict(stage='not_started', started_groups=0, completed_groups=0, candidate_rows=0)
    try:
        cfg = ready(out)
        started = read(out/'run-started.json')
        if started['freeze_sha256'] != sha(out/'freeze.json'):
            raise ValueError('Start marker does not match freeze')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            os.environ[name] = '1'
        from learning.task_completion_bc_v1 import numeric
        from learning.task_bc_v1.model import TaskNetwork
        from learning.task_completion_parity_v1.compare import compare
        prior_cfg = read(PREVIOUS/'registration.json')
        limits = prior_cfg['data_limits']
        phase = 'cache_decode'
        arrays = numeric.read(PREVIOUS/'private/train.npz', limits['max_groups_per_split'], limits['max_candidates_per_split'])
        if sha(PREVIOUS/'private/train.npz') != cfg['training_cache_sha256']:
            raise ValueError('Training cache changed during decode')
        groups = numeric.unpack(arrays, limits['max_groups_per_split'], limits['max_candidates_per_split'])
        if set(g['seed'] for g in groups) != set(prior_cfg['train_seeds']):
            raise ValueError('Unexpected training seed roles')
        phase = 'model_reload'
        model = TaskNetwork(PREVIOUS/'private/model-final.npz', cfg['checkpoint_sha256'])
        if sha(PREVIOUS/'private/model-final.npz') != cfg['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during load')
        launch = read(PREVIOUS/'gpu-launch.local.json')
        phase = 'comparison'
        progress_path = out/'progress.local.jsonl'
        with progress_path.open('x',encoding='utf-8',newline='\n') as stream:
            def record(value):
                nonlocal progress
                progress = value
                stream.write(json.dumps(value,sort_keys=True)+'\n')
                stream.flush()
                deadline_check()
            result = compare(groups, model, launch['expected_device'], launch['expected_device_kind'], progress=record)
        if (result['groups'] != cfg['max_groups'] or result['completed_gpu_forwards'] != cfg['max_groups']
                or result['candidate_rows'] > cfg['max_candidate_rows'] or not result['gpu_device_validated']):
            raise ValueError('Comparison execution incomplete')
        if sha(PREVIOUS/'private/model-final.npz') != cfg['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during comparison')
        deadline_check()
        write_new(out/'worker-result.json',dict(completed=True, **result,
            checkpoint_sha256=cfg['checkpoint_sha256'], training_cache_sha256=cfg['training_cache_sha256'],
            selection=cfg['selection'], gradients=0, training_updates=0,
            holdout_reads=0, official_transitions=0, new_games=0,
            interpretation='Present checkpoint inference comparison only; not proof of prior ValueError cause or game strength'))
        return result
    except BaseException as exc:
        write_new(out/'worker-failure.json',dict(completed=False, error_kind=type(exc).__name__,
            phase=phase, progress=progress, retry_allowed=False,
            interpretation='Counters are last persisted callback; an interrupted forward may have begun without completing'))
        raise


def guard(out=OUT):
    if os.name != 'posix':
        raise RuntimeError('Registered guard requires Linux')
    from learning.task_dagger_once_v1.gpu_guard import supervise
    out = Path(out).resolve()
    cfg = ready(out)
    if (out/'guard-started.json').exists() or (out/'cancel.local.json').exists():
        raise ValueError('Diagnostic already started or cancelled')
    write_new(out/'guard-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),
        work_seconds=cfg['work_seconds'], process_seconds=cfg['process_seconds'], retry_allowed=False))
    tick, supervision, error = time.monotonic(), None, None
    try:
        supervision = supervise([sys.executable,'-B','-m','learning.task_completion_parity_v2.run',
            'worker','--out',str(out)], cfg['work_seconds'], lambda:(out/'cancel.local.json').exists())
        if not supervision['completed'] or not read(out/'worker-result.json')['completed']:
            raise RuntimeError('Diagnostic worker did not complete')
        deadline_check()
    except BaseException as exc:
        error = type(exc).__name__
    elapsed = time.monotonic()-tick
    result = dict(passed=error is None and supervision is not None and supervision['completed']
                         and elapsed <= cfg['process_seconds'],
                  error_kind=error, supervision=supervision, elapsed_seconds=elapsed, retry_allowed=False)
    write_new(out/'guard-result.json',result)
    if not result['passed']:
        raise RuntimeError('Diagnostic session did not complete; no retry')
    return result


def start(out=OUT):
    from learning.task_anchor_once_v1.processes import CapturedProcess
    from learning.task_action_observation_v1.campaign import observe_processes
    from learning.task_launcher_contract_repair_v1.observer import reap
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if (out/'run-started.json').exists():
        raise ValueError('Diagnostic attempt already consumed')
    tick = time.monotonic()
    child, known, error, cleanup, capture = None, {}, None, None, None
    phase, failure_phase = 'start_marker', None
    marker, spawn_entered = {'created':False,'complete':False}, False
    cancel_error = guard_read_error = cleanup_error = None
    try:
        write_started_marker(out,dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),
            retry_allowed=False, data_role='previously consumed training cache only'),marker)
        phase = 'configuration'
        launch = read(PREVIOUS/'gpu-launch.local.json')
        command = ['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],
            '-B','-m','learning.task_completion_parity_v2.run','guard','--out',out.relative_to(ROOT).as_posix()]
        phase = 'transport_start'
        spawn_entered = True
        child = CapturedProcess(command,cwd=ROOT,log=out/'transport.local.log',limit_bytes=cfg['log_bytes'])
        phase = 'transport_monitor'
        while child.process.poll() is None:
            deadline_check()
            observe_processes(child,known)
            child.check()
            if time.monotonic()-tick >= cfg['process_seconds']-cfg['cleanup_seconds']:
                raise TimeoutError('Diagnostic transport deadline reached')
            time.sleep(.2)
        capture = child.finish()
        phase = 'guard_result'
        if capture['child_exit_code'] != 0 or not read(out/'guard-result.json')['passed']:
            raise RuntimeError('Diagnostic transport failed')
    except BaseException as exc:
        error = type(exc).__name__
        failure_phase = phase
        if child is not None:
            try:
                if child.process.poll() is None:
                    if not (out/'cancel.local.json').exists():
                        write_new(out/'cancel.local.json',dict(cancel=True,reason='owned transport stopping'))
                    child.process.wait(timeout=cfg['cleanup_seconds'])
            except BaseException as exc:
                cancel_error = type(exc).__name__
    finally:
        if child is not None:
            try:
                observe_processes(child,known)
                cleanup = reap(child,known)
            except BaseException as exc:
                cleanup_error = type(exc).__name__
        try:
            guard = read(out/'guard-result.json') if child is not None and (out/'guard-result.json').exists() else None
        except BaseException as exc:
            guard, guard_read_error = None, type(exc).__name__
        supervision = guard.get('supervision') if isinstance(guard,dict) else None
        flags = ('direct_child_reaped','owned_process_group_signalled','no_live_owned_session_members_before_reap')
        linux_clean = (all(supervision.get(k) is True for k in flags) if isinstance(supervision,dict) else None)
        result = dict(passed=error is None and cleanup_error is None and cancel_error is None
                           and guard_read_error is None and cleanup_complete(cleanup) and linux_clean is True
                           and time.monotonic()-tick <= cfg['process_seconds'],
                      error_kind=error,failure_phase=failure_phase,cleanup_error_kind=cleanup_error,
                      cancel_error_kind=cancel_error,guard_read_error_kind=guard_read_error,
                      marker_created=marker['created'],marker_complete=marker['complete'],
                      child_handle_acquired=child is not None,
                      child_lifecycle=('handle_acquired' if child is not None else
                                       'unknown_after_spawn_attempt' if spawn_entered else 'not_started'),
                      windows_cleanup=cleanup,
                      linux_session_cleanup_proven=linux_clean,capture=capture,
                      elapsed_seconds=time.monotonic()-tick,retry_allowed=False)
        if marker['created']:
            try:
                write_new(out/'run-result.json',result)
            except BaseException as exc:
                raise ReceiptFailure(error,type(exc).__name__,cleanup_error) from None
    if not result['passed']:
        raise StartFailure(error,failure_phase,marker['created'])
    return read(out/'worker-result.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('start','guard','worker'))
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args()
    try:
        result = {'start':start,'guard':guard,'worker':worker}[args.command](args.out)
        print(json.dumps(dict(completed=True, command=args.command)),flush=True)
    except BaseException as exc:
        print(json.dumps(dict(completed=False,error_kind=type(exc).__name__,command=args.command,
            primary_error_kind=getattr(exc,'primary_error_kind',None),
            receipt_error_kind=getattr(exc,'receipt_error_kind',None),
            cleanup_error_kind=getattr(exc,'cleanup_error_kind',None),
            failure_phase=getattr(exc,'failure_phase',None),
            marker_created=getattr(exc,'marker_created',None))),flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
