"""One-shot data collection and separate WSL GPU transport."""
import json
from pathlib import Path
import sys
import time
from learning.task_training_job_supervision_v1.job import WindowsAPI, supervise
from learning.task_anchor_once_v1.processes import CapturedProcess
from learning.task_action_observation_v1.campaign import observe_processes
from learning.task_launcher_contract_repair_v1.observer import reap
from .experiment import OUT, ROOT, ready, verify_commit, deadline_check, read, sha, write_new


class CampaignFailure(RuntimeError):
    def __init__(self,primary,receipt=None):
        super().__init__('Campaign attempt failed; no retry')
        self.primary_error_kind,self.receipt_error_kind = primary,receipt


def marker(out,name,value,state):
    path = (Path(out)/name).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        state['created'] = True
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
    state['complete'] = True


def collect_data(out=OUT):
    out = Path(out).resolve(); cfg = ready(out); verify_commit(out)
    if any((out/n).exists() for n in ('data-started.json','data-worker-started.json','train-data-result.json')):
        raise ValueError('Collection attempt consumed')
    state=dict(created=False,complete=False); result=None; error=None
    try:
        marker(out,'data-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),state)
        result = supervise(WindowsAPI(),[sys.executable,'-B','-m','learning.task_teacher_choice_bc_v1.data_worker'],
                           ROOT,cfg['collection_supervision'])
        if not result['passed'] or not result['launcher_reaped'] or not result['job_members_zero']:
            raise RuntimeError('Collection Job cleanup or execution failed')
        worker = read(out/'data-worker-result.json')
        if (not worker['passed'] or worker['collection_attempt'] != 1 or worker['freeze_sha256'] != sha(out/'freeze.json')
            or worker['source_manifest_sha256'] != cfg['source_manifest_sha256']
            or worker['sampling_algorithm'] != cfg['sampling_algorithm']
            or worker['raw_reads']['files_opened'] != 8 or worker['raw_reads']['files_completed'] != 8
            or worker['raw_reads']['raw_decode_passes_per_file'] != 1
            or sha(out/'private/train.npz') != worker['array_sha256']):
            raise ValueError('Collection worker result changed')
        deadline_check()
        write_new(out/'data-supervision-result.json',dict(passed=True,supervision=result,retry_allowed=False))
        sealed = dict(worker,job_supervision_sha256=sha(out/'data-supervision-result.json'),sealed=True)
        write_new(out/'train-data-result.json',sealed)
        return sealed
    except BaseException as exc:
        error=type(exc).__name__
        if state['created']:
            try: write_new(out/'data-failure.json',dict(error_kind=error,marker=state,supervision=result,retry_allowed=False))
            except BaseException as receipt: raise CampaignFailure(error,type(receipt).__name__) from None
        raise CampaignFailure(error) from None


def fit(out=OUT):
    out = Path(out).resolve(); cfg = ready(out); verify_commit(out)
    if (out/'fit-started.json').exists(): raise ValueError('Fit attempt consumed')
    source = read(out/'train-data-result.json'); job = read(out/'data-supervision-result.json')
    if (not source['passed'] or not source['sealed'] or not job['passed']
        or source['job_supervision_sha256'] != sha(out/'data-supervision-result.json')
        or source['freeze_sha256'] != sha(out/'freeze.json') or source['collection_attempt'] != 1
        or source['source_manifest_sha256'] != cfg['source_manifest_sha256']
        or source['sampling_algorithm'] != cfg['sampling_algorithm'] or sha(out/'private/train.npz') != source['array_sha256']):
        raise ValueError('Sealed collection required')
    state=dict(created=False,complete=False); child=None; known={}; cleanup=None; error=None; cancel_error=None; cleanup_error=None; guard_error=None
    started=time.monotonic(); phase='marker'; spawn_entered=False
    try:
        marker(out,'fit-started.json',dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),
            train_data_sha256=sha(out/'train-data-result.json'),array_sha256=source['array_sha256'],retry_allowed=False),state)
        phase='configuration'
        launch=read(out/'gpu-launch.local.json')
        if sha(out/'gpu-launch.local.json') != cfg['gpu_launch_config_sha256']: raise ValueError('Launch identity changed')
        command=['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],'-B','-m',
                 'learning.task_teacher_choice_bc_v1.gpu_entry','guard','--out',out.relative_to(ROOT).as_posix()]
        phase='spawn'; spawn_entered=True
        child=CapturedProcess(command,cwd=ROOT,log=out/'gpu-process.local.log',limit_bytes=cfg['storage']['process_log_bytes'])
        phase='monitor'
        while child.process.poll() is None:
            deadline_check(); observe_processes(child,known); child.check()
            if time.monotonic()-started >= cfg['training']['process_seconds']-20: raise TimeoutError('Transport deadline')
            time.sleep(.2)
        capture=child.finish()
        if capture['child_exit_code'] != 0: raise RuntimeError('GPU transport failed')
    except BaseException as exc:
        error=type(exc).__name__
        if child is not None:
            try:
                if child.process.poll() is None:
                    if not (out/'gpu-cancel.local.json').exists(): write_new(out/'gpu-cancel.local.json',dict(cancel=True))
                    child.process.wait(timeout=20)
            except BaseException as exc2: cancel_error=type(exc2).__name__
    finally:
        if child is not None:
            try: observe_processes(child,known); cleanup=reap(child,known)
            except BaseException as exc: cleanup_error=type(exc).__name__
        guard=None
        try:
            if child is not None and (out/'gpu-guard-result.json').exists(): guard=read(out/'gpu-guard-result.json')
        except BaseException as exc: guard_error=type(exc).__name__
        flags=('owned_process_group_signalled','direct_child_reaped','no_live_owned_session_members_before_reap')
        supervision=guard.get('supervision') if isinstance(guard,dict) else None
        linux=isinstance(supervision,dict) and all(supervision.get(k) is True for k in flags)
        windows=isinstance(cleanup,dict) and cleanup.get('owned_instances_remaining') == 0 and cleanup.get('reader_joined') is True and cleanup.get('launched_process_reaped') is True
        passed=(error is None and cancel_error is None and cleanup_error is None and guard_error is None and windows and linux
                and guard.get('passed') is True and time.monotonic()-started <= cfg['training']['process_seconds'])
        receipt=dict(passed=passed,error_kind=error,failure_phase=phase if error else None,cancel_error_kind=cancel_error,
            cleanup_error_kind=cleanup_error,guard_read_error_kind=guard_error,marker=state,windows_cleanup=cleanup,
            linux_cleanup_proven=linux,child_lifecycle='handle_acquired' if child else 'unknown_after_spawn_attempt' if spawn_entered else 'not_started',
            windows_job_does_not_guarantee_wsl=True,gpu_memory_hard_cap_claimed=False,retry_allowed=False)
        if state['created']:
            try: write_new(out/'fit-transport-result.json',receipt)
            except BaseException as exc: raise CampaignFailure(error,type(exc).__name__) from None
    if not receipt['passed']: raise CampaignFailure(error or 'CleanupFailure')
    try:
        fitted=read(out/'fit-result.json'); summary=fitted['training_summary']; parity=summary['inference_parity']
        if (not fitted['passed'] or fitted['updates'] != 128 or summary['synchronized_updates'] != 128
            or not summary['gpu_verified'] or not summary['weights_changed'] or summary['nonzero_gradient_updates'] <= 0
            or parity['groups'] != 16 or parity['synchronized'] is not True
            or parity['requested_matmul_precision'] != 'highest' or parity['hardware_precision_measured'] is not False
            or sha(out/'private/model-final.npz') != fitted['checkpoint_sha256']
            or fitted['train_data_sha256'] != sha(out/'train-data-result.json')):
            raise ValueError('Fit identity or acceptance failed')
        sealed=dict(passed=True,checkpoint_sha256=fitted['checkpoint_sha256'],freeze_sha256=sha(out/'freeze.json'),
            fit_result_sha256=sha(out/'fit-result.json'),fit_transport_sha256=sha(out/'fit-transport-result.json'),
            no_test_results_read=True,competitive_gain_proven=False,retry_allowed=False)
        write_new(out/'model-seal.json',sealed)
        return sealed
    except BaseException as exc:
        try: write_new(out/'fit-seal-failure.json',dict(error_kind=type(exc).__name__,retry_allowed=False))
        except BaseException as receipt_exc: raise CampaignFailure(type(exc).__name__,type(receipt_exc).__name__) from None
        raise CampaignFailure(type(exc).__name__) from None
