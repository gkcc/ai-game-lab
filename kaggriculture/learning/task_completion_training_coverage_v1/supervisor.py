"""Single owned-child supervisor with sampled memory and wall-clock limits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from .experiment import OUT, ROOT, SUPERVISION, ready, verify_commit, sha, write_new


class SupervisorFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind=None):
        super().__init__('Supervised attempt failed')
        self.primary_error_kind,self.receipt_error_kind = primary_error_kind,receipt_error_kind


def run(out=OUT):
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if cfg['supervision'] != SUPERVISION: raise ValueError('Supervision limits changed')
    if (out/'supervisor-started.json').exists() or (out/'analysis-started.json').exists():
        raise ValueError('Attempt already consumed')
    marker = dict(created=False,complete=False)
    child, spawn_entered, child_reaped = None,False,False
    reason,error,cleanup_error = None,None,None
    peak = 0
    start = time.monotonic()
    try:
        marker_path = (out/'supervisor-started.json').resolve()
        if not marker_path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
        with marker_path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json')),stream,indent=2)
            stream.write('\n')
        marker['complete'] = True
        command = [sys.executable,'-B','-c',
            'from learning.task_completion_training_coverage_v1.analysis import run; run()']
        opts = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else {}
        spawn_entered = True
        child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,**opts)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            rss = process.memory_info().rss
            peak = max(peak,rss)
            if rss > SUPERVISION['max_rss_bytes']:
                reason = 'rss_limit'; break
            if psutil.virtual_memory().available < SUPERVISION['min_available_bytes']:
                reason = 'available_memory_limit'; break
            if time.monotonic()-start >= SUPERVISION['wall_seconds']:
                reason = 'wall_limit'; break
            time.sleep(SUPERVISION['poll_seconds'])
        if reason is None:
            child.wait(timeout=SUPERVISION['termination_wait_seconds'])
            child_reaped = True
            if child.returncode != 0: reason = 'child_failed'
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        if child is not None:
            try:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=SUPERVISION['termination_wait_seconds'])
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=SUPERVISION['termination_wait_seconds'])
                else:
                    child.wait(timeout=SUPERVISION['termination_wait_seconds'])
                child_reaped = True
            except BaseException as exc:
                cleanup_error = type(exc).__name__
        result = dict(passed=error is None and cleanup_error is None and reason is None and child is not None and child_reaped and child.returncode == 0,
            marker_created=marker['created'],marker_complete=marker['complete'],error_kind=error,
            cleanup_error_kind=cleanup_error,stop_reason=reason,peak_sampled_rss_bytes=peak,
            child_reaped=child_reaped,
            elapsed_seconds=time.monotonic()-start,exit_code=child.returncode if child is not None else None,
            child_lifecycle='handle_acquired' if child is not None else 'unknown_after_spawn_attempt' if spawn_entered else 'not_started',
            memory_monitoring='sampled, not a hard memory cap',retry_allowed=False,partial_work_not_refunded=True)
        if marker['created']:
            try: write_new(out/'supervisor-result.json',result)
            except BaseException as exc: raise SupervisorFailure(error or reason or cleanup_error,type(exc).__name__) from None
    if not result['passed']: raise SupervisorFailure(error or reason or cleanup_error)
    return result


if __name__ == '__main__':
    try:
        run()
        print(json.dumps(dict(passed=True)))
    except BaseException as exc:
        print(json.dumps(dict(passed=False,error_kind=getattr(exc,'primary_error_kind',None) or type(exc).__name__)))
        raise SystemExit(1) from None
