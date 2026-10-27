"""One bounded Windows worker for an official offline reexecution study."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import psutil
from learning.task_launcher_evidence_repair_v1.capture import OwnedCapture
from learning.task_launcher_contract_repair_v1.observer import reap
from learning.task_action_observation_v1.campaign import observe_processes
from .experiment import ROOT,OUT,read,sha,write_new,ready,verify_commit,deadline_check,remaining_seconds


def worker():
    cfg=ready(OUT);started=read(OUT/'run-started.json')
    if started['freeze_sha256']!=sha(OUT/'freeze.json'):raise ValueError('Worker start does not match freeze')
    # Import the pinned environment before the guarded artifact context. No
    # model or agent is instantiated by the pure outcome pipeline.
    from decision.full_audit import reexecute  # noqa: F401
    from learning.task_anchor_once_v1.storage import owned_writes
    from .pipeline import run
    initial=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    allowance=cfg['analysis_limits']['output_bytes']-initial-cfg['analysis_limits']['log_bytes']-2*1024**2
    if allowance<=0:raise ValueError('No bounded output allowance remains')
    budget=owned_writes(OUT,allowance,allowed_paths=['game-ledger.json','dataset-manifest.json',
        'analysis-result.json','failure.local.json'],allowed_prefixes=['private/games','game-results'])
    error=None
    try:
        with budget:run(OUT)
    except BaseException as exc:
        error=type(exc).__name__
        raise
    finally:
        with owned_writes(OUT,1024**2,allowed_paths=['worker-storage.json']):
            write_new(OUT/'worker-storage.json',dict(passed=error is None,error_kind=error,
                output=budget.snapshot(),initial_owned_bytes=initial,worker_log_reserved_bytes=cfg['analysis_limits']['log_bytes']))


def sampled_memory(known):
    total=psutil.Process().memory_info().rss
    live=0
    for pid,identity in list(known.items()):
        try:
            p=psutil.Process(pid)
            if p.create_time()!=identity['created']:continue
            total+=p.memory_info().rss;live+=1
        except psutil.NoSuchProcess:continue
    return total,live


def run():
    cfg=ready(OUT);verify_commit(OUT);deadline_check()
    if (OUT/'run-started.json').exists():raise FileExistsError('This frozen study attempt is already consumed')
    limits=cfg['analysis_limits'];remaining=remaining_seconds()
    if remaining<=limits['cleanup_seconds']+1:raise TimeoutError('Insufficient task time including cleanup')
    parent=psutil.Process();available=parent.cpu_affinity()
    if not available:raise RuntimeError('No processor available for this worker')
    parent.cpu_affinity([available[0]])
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    write_new(OUT/'run-started.json',dict(freeze_sha256=sha(OUT/'freeze.json'),
        registration_sha256=sha(OUT/'registration.json'),attempt_consumed=True,retry_allowed=False,
        official_reexecution_limit=cfg['main_official_transitions']))
    captured=None;known={};started=time.monotonic();error=None;cleanup_error=None;capture=None;cleanup=None
    peak=0;samples=0;max_owned=0
    try:
        captured=OwnedCapture(log=OUT/'worker.local.log',limit_bytes=limits['log_bytes'])
        captured.start([sys.executable,'-B','-X','utf8','-m',__package__+'.run','worker'],cwd=ROOT)
        observe_processes(captured,known)
        while captured.process.poll() is None:
            deadline_check();observe_processes(captured,known);captured.check()
            rss,live=sampled_memory(known);peak=max(peak,rss);max_owned=max(max_owned,live);samples+=1
            if rss>limits['sampled_memory_bytes']:raise MemoryError('Sampled owned memory exceeded the limit')
            if time.monotonic()-started>limits['work_seconds']:raise TimeoutError('Worker wall-time budget exceeded')
            time.sleep(.2)
        capture=captured.finish()
        if captured.process.returncode!=0:raise RuntimeError('Registered worker failed')
        result=read(OUT/'analysis-result.json')
        if not result['complete'] or result['official_transitions']!=cfg['main_official_transitions']:
            raise ValueError('Worker result incomplete')
        storage=read(OUT/'worker-storage.json')
        if not storage['passed'] or storage['output']['refused']!=0:raise RuntimeError('Worker output budget failed')
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        if captured is not None:
            try:
                if captured.process is None:captured.stop()
                else:cleanup=reap(captured,known)
            except BaseException as exc:cleanup_error=type(exc).__name__
        elapsed=time.monotonic()-started
        clean=cleanup is not None and cleanup.get('owned_instances_remaining')==0 and cleanup.get('reader_joined') is True
        result=dict(passed=error is None and cleanup_error is None and clean and remaining_seconds()>0,
            primary_error_kind=error,cleanup_error_kind=cleanup_error,cleanup=cleanup,capture=capture,
            elapsed_seconds=elapsed,sampled_peak_memory_bytes=peak,sampled_process_instances=max_owned,
            memory_samples=samples,memory_limit_kind='sampled; not a hard operating-system quota',
            within_total_deadline=remaining_seconds()>0,retry_allowed=False)
        write_new(OUT/'run-result.json',result)
    if not result['passed']:raise RuntimeError('Study worker or cleanup failed; retained evidence must not be retried')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','worker'));args=parser.parse_args()
    if args.command=='worker':worker()
    else:print(json.dumps(run()))
