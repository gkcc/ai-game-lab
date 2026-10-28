"""One WSL GPU fit with separate host and Linux session supervision."""
import hashlib
import copy
import json
import math
from pathlib import Path
import secrets
import sys
import time
import psutil
from learning.task_anchor_once_v1.processes import CapturedProcess
from learning.task_action_observation_v1.campaign import observe_processes
from learning.task_launcher_contract_repair_v1.observer import reap
from .experiment import OUT,ROOT,deadline_check,prepare_private_checkpoint,read,ready,sha,verify_commit,write_new
from .gpu_entry import ARCHIVE_READ_SCOPE


class HostMemoryFloor(RuntimeError): pass


class ReceiptFailure(RuntimeError):
    def __init__(self,primary,receipt):
        super().__init__('Primary failure and failure-receipt error')
        self.primary_error_kind,self.receipt_error_kind=primary,receipt


def memory_ready(cfg,available=lambda:psutil.virtual_memory().available,sleep=time.sleep):
    limits=cfg['host_limits']
    for index in range(limits['preflight_samples']):
        deadline_check(); value=available()
        if type(value) is not int or value<limits['preflight_min_available_bytes']: return False
        if index+1<limits['preflight_samples']: sleep(limits['preflight_interval_seconds'])
    return True


def marker(out,value,state):
    path=(Path(out)/'fit-started.json').resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside experiment')
    with path.open('x',encoding='utf-8',newline='\n') as f:
        state['created']=True; json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
    state['complete']=True


def verify_identities(out,identities):
    for name,digest in identities.items():
        if sha(Path(out)/name)!=digest: raise ValueError('Fit identity changed')


def validate_fit(fit,cfg,out,freeze_sha,identities):
    if not isinstance(fit,dict) or fit.get('passed') is not True: raise ValueError('Missing fit result')
    parity=fit.get('inference_parity',{})
    if (fit.get('freeze_sha256')!=freeze_sha or fit.get('archive_sha256')!=cfg['archive_sha256']
        or fit.get('schema_sha256')!=cfg['schema_sha256'] or fit.get('archive_snapshot_reads')!=dict(old=1,new=1)
        or fit.get('archive_read_scope')!=ARCHIVE_READ_SCOPE
        or fit.get('groups')!=sum(cfg['expected_groups'].values())
        or fit.get('updates')!=128 or type(fit.get('nonzero_gradient_updates')) is not int
        or not 0<fit['nonzero_gradient_updates']<=128 or fit.get('weights_changed') is not True
        or fit.get('gpu_verified') is not True or fit.get('retry_allowed') is not False
        or parity.get('groups')!=16 or type(parity.get('candidate_rows')) is not int or parity['candidate_rows']<16
        or parity.get('synchronized') is not True
        or parity.get('requested_matmul_precision')!='highest' or parity.get('hardware_precision_measured') is not False
        or type(parity.get('max_abs_error')) not in (int,float) or not math.isfinite(parity['max_abs_error'])
        or fit.get('checkpoint_sha256')!=sha(Path(out)/'private/model-final.npz')):
        raise ValueError('Fit semantic result invalid')
    expected_identities=dict(freeze=identities['freeze.json'],registration=identities['registration.json'],
        pre_execution_commit=identities['pre-execution-commit.json'],marker=identities['fit-started.json'],
        guard_marker=sha(Path(out)/'gpu-guard-started.json'),launch=identities['gpu-launch.local.json'],
        schema=cfg['schema_sha256'],new_schema=cfg['schema_sha256'],seed_inventory=cfg['seed_inventory_sha256'])
    if fit.get('identities')!=expected_identities: raise ValueError('Fit identity evidence differs')


def fit():
    out=OUT.resolve(); cfg=ready(out); verify_commit(out)
    original_cfg=copy.deepcopy(cfg)
    names=('fit-started.json','fit-result.json','fit-transport-result.json','fit-worker-failure.json',
           'gpu-guard-started.json','gpu-guard-result.json','gpu-cancel.local.json','fit-progress.local.jsonl',
           'gpu-process.local.log','model-seal.json','private/model-final.npz')
    if any((out/name).exists() for name in names): raise ValueError('Fit attempt already consumed')
    if not memory_ready(cfg):
        print('memory preflight not ready; no fit attempt or cache read',flush=True)
        return dict(passed=False,preflight_not_ready=True,attempt_started=False)
    refreshed=ready(out); verify_commit(out)
    if refreshed!=original_cfg or cfg!=original_cfg: raise ValueError('Frozen configuration changed during preflight')
    prepare_private_checkpoint(out,create=True)
    state=dict(created=False,complete=False); identities={}; child=None; known={}; capture=None; cleanup=None
    error=None; stop_reason=None; cancel_error=None; observation_error=None; cleanup_error=None; guard_error=None; spawn_entered=False
    tick=time.monotonic(); host_samples=0; phase='marker'
    try:
        identities={name:sha(out/name) for name in ('freeze.json','registration.json','pre-execution-commit.json','gpu-launch.local.json')}
        freeze_sha=identities['freeze.json']
        token=secrets.token_hex(16)
        token_sha=hashlib.sha256(token.encode('ascii')).hexdigest()
        marker(out,dict(attempt=1,freeze_sha256=freeze_sha,archive_sha256=cfg['archive_sha256'],
                        host_token_sha256=token_sha,retry_allowed=False),state)
        identities['fit-started.json']=sha(out/'fit-started.json')
        launch=read(out/'gpu-launch.local.json')
        if identities['gpu-launch.local.json']!=cfg['gpu_launch_config_sha256']: raise ValueError('GPU launch identity changed')
        command=['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],'-B','-m',
                 'learning.task_teacher_merge_bc_v1.gpu_entry','guard','--out',out.relative_to(ROOT).as_posix(),'--host-token',token]
        phase='spawn'; spawn_entered=True
        child=CapturedProcess(command,cwd=ROOT,log=out/'gpu-process.local.log',limit_bytes=cfg['process_log_bytes'])
        phase='monitor'
        while child.process.poll() is None:
            deadline_check(); observe_processes(child,known); child.check()
            available=psutil.virtual_memory().available; host_samples+=1
            if type(available) is not int or available<cfg['host_limits']['min_available_bytes']:
                stop_reason='available_memory_limit'; raise HostMemoryFloor('Host available memory below registered floor')
            if time.monotonic()-tick>=cfg['training']['process_seconds']-cfg['training']['cancellation_wait_seconds']-cfg['training']['cleanup_seconds']:
                stop_reason='wall_limit'; raise TimeoutError('Transport work allowance exhausted')
            time.sleep(cfg['host_limits']['poll_seconds'])
        capture=child.finish()
        if capture['child_exit_code']!=0: raise RuntimeError('WSL transport failed')
    except BaseException as exc:
        error=type(exc).__name__
        if spawn_entered:
            try:
                if not (out/'gpu-cancel.local.json').exists(): write_new(out/'gpu-cancel.local.json',dict(cancel=True))
                if child is not None and child.process.poll() is None:
                    child.process.wait(timeout=cfg['training']['cancellation_wait_seconds'])
            except BaseException as exc2: cancel_error=type(exc2).__name__
    finally:
        if child is not None:
            try: observe_processes(child,known)
            except BaseException as exc: observation_error=type(exc).__name__
            try: cleanup=reap(child,known)
            except BaseException as exc: cleanup_error=type(exc).__name__
        guard=None
        try:
            if (out/'gpu-guard-result.json').exists(): guard=read(out/'gpu-guard-result.json')
        except BaseException as exc: guard_error=type(exc).__name__
        flags=('owned_process_group_signalled','direct_child_reaped','no_live_owned_session_members_before_reap')
        supervision=guard.get('supervision') if isinstance(guard,dict) else None
        linux=isinstance(supervision,dict) and all(supervision.get(k) is True for k in flags)
        windows=isinstance(cleanup,dict) and cleanup.get('owned_instances_remaining')==0 and cleanup.get('reader_joined') is True and cleanup.get('launched_process_reaped') is True
        passed=(error is None and cancel_error is None and observation_error is None and cleanup_error is None and guard_error is None and windows and linux and host_samples>0
                and guard.get('passed') is True and time.monotonic()-tick<=cfg['training']['process_seconds'])
        receipt=dict(passed=passed,error_kind=error,stop_reason=stop_reason,failure_phase=phase if error else None,
            cancel_error_kind=cancel_error,observation_error_kind=observation_error,cleanup_error_kind=cleanup_error,guard_read_error_kind=guard_error,marker=state,
            host_memory_samples=host_samples,windows_cleanup=cleanup,linux_cleanup_proven=linux,
            child_lifecycle='handle_acquired' if child else 'unknown_after_spawn_attempt' if spawn_entered else 'not_started',
            windows_job_controls_wsl=False,gpu_memory_hard_cap_claimed=False,retry_allowed=False)
        if state['created']:
            try: write_new(out/'fit-transport-result.json',receipt)
            except BaseException as exc: raise ReceiptFailure(error,type(exc).__name__) from None
    if not receipt['passed']: raise RuntimeError('Fit or cleanup failed; no retry')
    try:
        verify_identities(out,identities); ready(out); deadline_check()
        fitted=read(out/'fit-result.json'); validate_fit(fitted,cfg,out,freeze_sha,identities)
        guard=read(out/'gpu-guard-result.json')
        guard_marker=read(out/'gpu-guard-started.json')
        if (guard.get('freeze_sha256')!=freeze_sha or guard.get('guard_marker_sha256')!=sha(out/'gpu-guard-started.json')
            or guard.get('marker')!=dict(created=True,complete=True) or guard.get('retry_allowed') is not False
            or guard_marker!=dict(attempt=1,freeze_sha256=freeze_sha,fit_marker_sha256=identities['fit-started.json'],
                                 host_token_sha256=token_sha,retry_allowed=False)):
            raise ValueError('Linux guard identity differs')
        fit_sha=sha(out/'fit-result.json'); guard_sha=sha(out/'gpu-guard-result.json')
        verify_identities(out,identities)
        sealed=dict(passed=True,checkpoint_sha256=fitted['checkpoint_sha256'],freeze_sha256=freeze_sha,
            fit_result_sha256=fit_sha,fit_transport_sha256=sha(out/'fit-transport-result.json'),
            linux_guard_sha256=guard_sha,identities=identities,no_test_results_read=True,
            competitive_gain_proven=False,automatic_promotion=False,retry_allowed=False)
        write_new(out/'model-seal.json',sealed)
        return sealed
    except BaseException as exc:
        try: write_new(out/'fit-seal-failure.json',dict(error_kind=type(exc).__name__,retry_allowed=False))
        except BaseException as receipt: raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


if __name__=='__main__':
    outcome=fit()
    if outcome.get('preflight_not_ready'): raise SystemExit(75)
