"""Owned Windows Job entry for one frozen CPU-only diagnostic."""
import json
import os
from pathlib import Path
import secrets
import sys
from learning.task_job_sampling_race_v1.sampling import supervise
from .experiment import OUT,ROOT,THREAD_LIMITS,ready,verify_commit,deadline_check,sha,read,write_new
from .membership import ENV_KEY,NamedWindowsAPI,token_hash


class ReceiptFailure(RuntimeError):
    def __init__(self,primary,receipt):
        super().__init__('Diagnostic receipt failed'); self.primary_error_kind=primary; self.receipt_error_kind=receipt


def progress_snapshot(out):
    path=out/'progress.local.jsonl'
    try:
        if not path.exists(): return dict(forwards_started='unknown',forwards_completed='unknown',accounting='not_created_or_unknown')
        if path.stat().st_size>100000: return dict(forwards_started='unknown',forwards_completed='unknown',accounting='invalid')
        lines=path.read_text(encoding='utf-8').splitlines()
        value=json.loads(lines[-1])
        started,completed=value['forwards_started'],value['forwards_completed']
        if type(started) is not int or type(completed) is not int or not 0<=completed<=started<=400 or started-completed>1:
            raise ValueError
        trace_reads=value.get('trace_reads','unknown')
        if trace_reads not in ('not_started','started_count_unknown',2): raise ValueError
        return dict(forwards_started=started,forwards_completed=completed,trace_reads=trace_reads,accounting='last_flushed_callback')
    except (ValueError,KeyError,IndexError,TypeError,OSError):
        return dict(forwards_started='unknown',forwards_completed='unknown',accounting='unreadable')


def launch_named(name,module,limits):
    settings={ENV_KEY:name}
    if module=='learning.task_teacher_dev_deviation_v1.worker': settings.update(THREAD_LIMITS)
    previous={key:os.environ.get(key) for key in settings}
    try:
        os.environ.update(settings)
        return supervise(NamedWindowsAPI(name),[sys.executable,'-B','-m',module],ROOT,limits)
    finally:
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value


def validate_probe(out,identities,job,probe):
    if (not isinstance(job,dict) or job.get('passed') is not True or job.get('assigned') is not True
        or job.get('resumed') is not True or job.get('launcher_reaped') is not True or job.get('job_members_zero') is not True
        or job.get('error_kind') is not None or job.get('cleanup_error_kind') is not None or job.get('exit_code')!=0
        or not isinstance(probe,dict) or probe.get('passed') is not True or probe.get('venv_verified') is not True
        or probe.get('job_membership_verified') is not True or probe.get('training_inputs_read') is not False
        or probe.get('freeze_sha256')!=identities['freeze_sha256']
        or probe.get('marker_sha256')!=identities['probe_marker_sha256']
        or sha(out/'probe-started.json')!=identities['probe_marker_sha256']):
        raise ValueError('Synthetic membership probe not accepted')


def run(out=OUT):
    out=Path(out).resolve(); cfg=ready(out); verify_commit(out)
    if any((out/n).exists() for n in ('started.json','result.json','probe-started.json','probe-result.json','probe-supervision-result.json',
        'worker-started.json','worker-result.json','worker-failure.json','progress.local.jsonl')):
        raise ValueError('Diagnostic attempt consumed')
    state=dict(created=False,complete=False); identities={}; job=None; probe_job=None; error=None; worker_sha=None
    name='Local\\TeacherDeviation-'+secrets.token_hex(16)
    try:
        path=(out/'started.json').resolve()
        if not path.is_relative_to(OUT): raise ValueError('Marker outside output')
        with path.open('x',encoding='utf-8',newline='\n') as f:
            state['created']=True
            json.dump(dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),job_token_sha256=token_hash(name),retry_allowed=False),f); f.write('\n')
        state['complete']=True
        identities=dict(marker_sha256=sha(path),freeze_sha256=sha(out/'freeze.json'),registration_sha256=sha(out/'registration.json'),
            pre_execution_commit_sha256=sha(out/'pre-execution-commit.json'),commit=read(out/'pre-execution-commit.json')['commit'])
        probe_name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        probe_path=out/'probe-started.json'
        write_new(probe_path,dict(attempt=1,freeze_sha256=identities['freeze_sha256'],
            job_token_sha256=token_hash(probe_name),retry_allowed=False))
        identities['probe_marker_sha256']=sha(probe_path)
        probe_job=launch_named(probe_name,'learning.task_teacher_dev_deviation_v1.member_probe',cfg['probe_job_limits'])
        probe=read(out/'probe-result.json')
        validate_probe(out,identities,probe_job,probe)
        ready(out); verify_commit(out); deadline_check()
        if (sha(path)!=identities['marker_sha256'] or sha(out/'freeze.json')!=identities['freeze_sha256']
            or sha(out/'registration.json')!=identities['registration_sha256']
            or sha(out/'pre-execution-commit.json')!=identities['pre_execution_commit_sha256']
            or read(out/'pre-execution-commit.json')['commit']!=identities['commit']): raise ValueError('Preflight identity changed')
        write_new(out/'probe-supervision-result.json',dict(passed=True,job=probe_job,
            probe_result_sha256=sha(out/'probe-result.json'),probe_marker_sha256=identities['probe_marker_sha256'],
            freeze_sha256=identities['freeze_sha256'],pre_execution_commit_sha256=identities['pre_execution_commit_sha256'],
            commit=identities['commit'],retry_allowed=False))
        job=launch_named(name,'learning.task_teacher_dev_deviation_v1.worker',cfg['job_limits'])
        if not job['passed'] or not job['launcher_reaped'] or not job['job_members_zero']: raise RuntimeError('Job cleanup or worker failed')
        worker=read(out/'worker-result.json')
        if (not worker['passed'] or worker['freeze_sha256']!=identities['freeze_sha256']
            or worker['checkpoint_sha256']!=cfg['checkpoint_sha256']
            or worker['old_checkpoint_sha256']!=cfg['old_checkpoint_sha256']
            or worker['aggregate']['groups']!=200 or worker['aggregate']['cpu_forwards']!=400
            or worker['aggregate']['candidate_rows_total']>120000
            or worker['raw_reads']['files_opened']!=2 or worker['raw_reads']['files_completed']!=2
            or worker['raw_reads']['raw_decode_passes_per_file']!=1): raise ValueError('Worker result changed')
        worker_sha=sha(out/'worker-result.json')
        ready(out); verify_commit(out); deadline_check()
        if (sha(path)!=identities['marker_sha256'] or sha(out/'freeze.json')!=identities['freeze_sha256']
            or sha(out/'registration.json')!=identities['registration_sha256']
            or sha(out/'pre-execution-commit.json')!=identities['pre_execution_commit_sha256']
            or read(out/'pre-execution-commit.json')['commit']!=identities['commit']
            or sha(out/'worker-result.json')!=worker_sha): raise ValueError('Execution identity changed')
    except BaseException as exc: error=type(exc).__name__
    if state['created']:
        receipt=dict(passed=error is None,error_kind=error,marker=state,identities=identities,job=job,probe_job=probe_job,
            progress=progress_snapshot(out),worker_result_sha256=worker_sha,retry_allowed=False,development_only=True,
            interpretation='Development trajectories on training seeds; no independent holdout, causation, or hard memory cap')
        try: write_new(out/'result.json',receipt)
        except BaseException as exc: raise ReceiptFailure(error,type(exc).__name__) from None
    if error: raise RuntimeError('Diagnostic failed: '+error)
    return read(out/'worker-result.json')


if __name__=='__main__': run()
