"""Linux session owner: bounded training, cancellation, whole-session cleanup.

The direct child is observed with WNOWAIT until its process group is terminated;
its PID cannot be reused before cleanup. Windows never assumes killing wsl.exe
also kills Linux training. The training module has its own inner fit deadline.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def live_session_members(session):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[3]) == session and fields[0] not in ('Z','X'):
            members.append(int(entry.name))
    return members


def supervise(command, seconds, cancelled):
    if os.name != 'posix' or not hasattr(os, 'WNOWAIT'):
        raise RuntimeError('Linux wait-without-reaping support is required')
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    reason = None
    try:
        while True:
            status = os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            if status is not None:
                reason = 'exited'; break
            if cancelled():
                reason = 'cancelled'; break
            if time.monotonic()-started >= seconds:
                reason = 'deadline'; break
            time.sleep(.05)
    finally:
        # Retaining the unreaped direct child pins this owned process-group ID.
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(.2)
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic()+9
        try:
            while live_session_members(process.pid):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Owned training session still has live members')
                time.sleep(.05)
        finally:
            process.wait(timeout=5)
    return dict(reason=reason,child_exit_code=process.returncode,
                elapsed_seconds=time.monotonic()-started,
                owned_process_group_signalled=True,direct_child_reaped=True,
                no_live_owned_session_members_before_reap=True,
                completed=reason=='exited' and process.returncode==0)


def run(out):
    from .experiment import read, write_new, training_ready
    started = time.monotonic()
    out = Path(out).resolve()
    cfg = training_ready(out)
    limits = cfg['gpu_guard']
    write_new(out/'gpu-guard-started.json',dict(work_seconds=limits['work_seconds'],
                                               total_seconds=limits['process_seconds']))
    remaining = min(limits['work_seconds'], limits['process_seconds']-
                    limits['cleanup_seconds']-(time.monotonic()-started))
    assert remaining > 0
    result = supervise([sys.executable,'-B','-m','learning.task_dagger_once_v1.training',
                        '--out',str(out)], remaining,
                       lambda:(out/'training-cancel.local.json').exists())
    result['child_supervision_seconds'] = result['elapsed_seconds']
    result['elapsed_seconds'] = time.monotonic()-started
    result['within_total_deadline'] = result['elapsed_seconds'] <= limits['process_seconds']
    result['completed'] = result['completed'] and result['within_total_deadline']
    write_new(out/'gpu-guard-result.json',result)
    if not result['completed']:
        raise RuntimeError('Owned GPU fit did not complete within its registered process boundary')


def self_test():
    """Three real Linux process paths; no model, data, GPU or temporary files."""
    cases = {}
    cases['normal'] = supervise([sys.executable,'-B','-c','pass'],5,lambda:False)
    child = ('import subprocess,sys,time; '
             'subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"]); '
             'time.sleep(30)')
    cases['deadline'] = supervise([sys.executable,'-B','-c',child],.5,lambda:False)
    tick = time.monotonic()
    cases['cancel'] = supervise([sys.executable,'-B','-c',child],5,
                               lambda:time.monotonic()-tick >= .5)
    assert cases['normal']['completed']
    assert cases['deadline']['reason'] == 'deadline' and not cases['deadline']['completed']
    assert cases['cancel']['reason'] == 'cancelled' and not cases['cancel']['completed']
    assert all(c['direct_child_reaped'] for c in cases.values())
    print(json.dumps(dict(passed=True,cases=cases,models=0,new_games=0,gpu_operations=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run(args.out)
