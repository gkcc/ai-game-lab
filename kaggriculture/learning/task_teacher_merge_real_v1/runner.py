"""One owned Job probe and one old-cache compatibility check; no fit."""
import json
import os
from pathlib import Path
import secrets
import sys
import time
import psutil
from learning.task_job_sampling_race_v1.sampling import supervise
from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY,NamedWindowsAPI,token_hash
from learning.task_probe_failure_reporting_v1.report import classify,safe_job
from .experiment import OUT,ROOT,deadline_check,read,ready,sha,verify_commit,write_new


class ReceiptFailure(RuntimeError):
    def __init__(self,primary,receipt):
        super().__init__('Primary failure and failure-receipt error')
        self.primary_error_kind,self.receipt_error_kind=primary,receipt


def memory_ready(cfg,available=lambda:psutil.virtual_memory().available,sleep=time.sleep):
    """No archive read or attempt marker until three sampled headroom checks pass."""
    for index in range(cfg['preflight_samples']):
        deadline_check()
        value=available()
        if type(value) is not int or value<cfg['preflight_min_available_bytes']: return False
        if index+1<cfg['preflight_samples']: sleep(cfg['preflight_interval_seconds'])
    return True


def job_ok(job):
    x=safe_job(job)
    return (x is not None and x['passed'] is True and x['assigned'] is True and x['resumed'] is True
        and x['launcher_reaped'] is True and x['job_members_zero'] is True and x['exit_code']==0
        and x['error_kind'] is None and x['cleanup_error_kind'] is None and x['stop_reason'] is None)


def launch(name,module,limits):
    previous=os.environ.get(ENV_KEY)
    try:
        os.environ[ENV_KEY]=name
        return supervise(NamedWindowsAPI(name),[sys.executable,'-B','-m',module],ROOT,limits)
    finally:
        if previous is None: os.environ.pop(ENV_KEY,None)
        else: os.environ[ENV_KEY]=previous


def marker(out,name,value,state):
    path=(Path(out)/name).resolve()
    if not path.is_relative_to(OUT.resolve()): raise ValueError('Marker outside stage')
    with path.open('x',encoding='utf-8',newline='\n') as f:
        state['created']=True
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
    state['complete']=True


def verify_identities(out,identities):
    for name,digest in identities.items():
        if sha(Path(out)/name)!=digest: raise ValueError('Execution identity changed')


def validate_worker(worker,cfg,freeze_sha):
    fields={'passed','aggregate','freeze_sha256','seed_roles_sha256','cache_sha256','schema_sha256',
            'archive_snapshot_reads','no_training','retry_allowed'}
    aggregate_fields={'passed','old_groups','old_candidate_rows','synthetic_groups','total_groups','alignment_passed',
                      'old_input_shape_valid','synthetic_input_shapes','source_identity','interpretation'}
    if not isinstance(worker,dict) or set(worker)!=fields or not isinstance(worker.get('aggregate'),dict):
        raise ValueError('Invalid worker receipt')
    a=worker['aggregate']
    if (set(a)!=aggregate_fields or worker['passed'] is not True or worker['freeze_sha256']!=freeze_sha
        or worker['seed_roles_sha256']!=cfg['seed_roles_sha256'] or worker['cache_sha256']!=cfg['cache_sha256']
        or worker['schema_sha256']!=cfg['schema_sha256'] or type(worker['archive_snapshot_reads']) is not int
        or worker['archive_snapshot_reads']!=1
        or worker['no_training'] is not True or worker['retry_allowed'] is not False
        or a['passed'] is not True or a['old_groups']!=cfg['old_groups']
        or a['old_candidate_rows']!=cfg['old_candidate_rows'] or a['synthetic_groups']!=2
        or a['total_groups']!=cfg['old_groups']+2 or a['alignment_passed'] is not True
        or a['old_input_shape_valid'] is not True or a['synthetic_input_shapes']!=[[1,373],[2,373]]
        or a['source_identity']!='Caller must verify source SHA and common trusted schema before invocation'
        or a['interpretation']!='Synthetic interface check only; no GPU training or game outcome evaluated'):
        raise ValueError('Worker semantic result failed')
    return a


def run():
    out=OUT.resolve(); cfg=ready(out); verify_commit(out)
    names=('started.json','result.json','probe-started.json','probe-result.json',
           'worker-started.json','worker-result.json','worker-failure.json')
    if any((out/name).exists() for name in names): raise ValueError('Stage already attempted')
    if not memory_ready(cfg):
        print('memory preflight not ready; no attempt or cache read',flush=True)
        return dict(passed=False,preflight_not_ready=True,attempt_started=False)
    state=dict(created=False,complete=False); probe_job=None; verdict=None; worker_job=None; worker_entered=False; identities={}
    try:
        freeze_sha=sha(out/'freeze.json')
        identities={name:sha(out/name) for name in ('freeze.json','registration.json','pre-execution-commit.json')}
        if identities['freeze.json']!=freeze_sha: raise ValueError('Freeze changed before start')
        marker(out,'started.json',dict(attempt=1,freeze_sha256=freeze_sha,retry_allowed=False),state)
        identities['started.json']=sha(out/'started.json')
        name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        write_new(out/'probe-started.json',dict(attempt=1,freeze_sha256=freeze_sha,
            job_token_sha256=token_hash(name),retry_allowed=False))
        probe_sha=sha(out/'probe-started.json')
        identities['probe-started.json']=probe_sha
        probe_job=launch(name,'learning.task_teacher_merge_real_v1.probe',cfg['probe_job_supervision'])
        verdict=classify(probe_job,lambda:read(out/'probe-result.json'),freeze_sha,probe_sha)
        if not verdict['worker_allowed']: raise RuntimeError('Probe rejected; cache worker not launched')
        identities['probe-result.json']=sha(out/'probe-result.json')
        verify_identities(out,identities)
        deadline_check(); ready(out)
        name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        write_new(out/'worker-started.json',dict(attempt=1,freeze_sha256=freeze_sha,job_token_sha256=token_hash(name),
            seed_roles_sha256=cfg['seed_roles_sha256'],schema_sha256=cfg['schema_sha256'],
            cache_sha256=cfg['cache_sha256'],retry_allowed=False))
        identities['worker-started.json']=sha(out/'worker-started.json')
        worker_entered=True
        worker_job=launch(name,'learning.task_teacher_merge_real_v1.worker',cfg['worker_job_supervision'])
        if not job_ok(worker_job): raise RuntimeError('Worker Job failed or cleanup unconfirmed')
        verify_identities(out,identities)
        identities['worker-result.json']=sha(out/'worker-result.json')
        worker=read(out/'worker-result.json'); aggregate=validate_worker(worker,cfg,freeze_sha)
        deadline_check(); ready(out)
        verify_identities(out,identities)
        result=dict(passed=True,freeze_sha256=freeze_sha,probe=verdict,worker_job=safe_job(worker_job),
            worker_result_sha256=identities['worker-result.json'],identities=identities,aggregate=aggregate,archive_snapshot_reads=1,
            no_training=True,no_model_reads=True,no_official_games=True,retry_allowed=False)
        write_new(out/'result.json',result)
        return result
    except BaseException as exc:
        if state['created']:
            try:
                write_new(out/'result.json',dict(passed=False,error_kind=type(exc).__name__,marker=state,probe=verdict,
                    probe_job=safe_job(probe_job) if probe_job is not None else None,
                    worker_job=safe_job(worker_job) if worker_job is not None else None,worker_launch_entered=worker_entered,
                    identities=identities,
                    archive_reads='unknown' if worker_entered else 'not_started_by_runner',
                    no_training=True,no_official_games=True,retry_allowed=False))
            except BaseException as receipt: raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise


if __name__=='__main__':
    outcome=run()
    if outcome.get('preflight_not_ready'): raise SystemExit(75)
