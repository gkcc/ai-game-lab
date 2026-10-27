"""Existing bounded Windows transport to the accepted POSIX analysis path."""
import argparse
import json
from pathlib import Path
import time

from learning.task_anchor_once_v1.processes import CapturedProcess
from learning.task_launcher_contract_repair_v1.observer import reap
from .campaign import observe_processes
from .experiment import ROOT, OUT, read, sha, write_new, ready, deadline_check


def run(fixture=False):
    cfg=ready(OUT);deadline_check()
    folder=OUT/'entry-fixture' if fixture else OUT
    guard_name='fixture-guard-result.json' if fixture else 'guard-result.json'
    name='fixture' if fixture else 'analysis'
    if not fixture:
        if not read(OUT/'campaign-result.json')['passed']:raise ValueError('Official campaign has not passed')
    limit=cfg['transport_limits'][name+'_seconds']
    write_new(folder/'transport-started.json',dict(fixture=fixture,attempt_consumed=True,
        freeze_sha256=sha(OUT/'freeze.json'),seconds=limit,retry_allowed=False))
    launch=read(OUT/'wsl-launch.local.json')
    command=['wsl.exe','-d',launch['distro'],'--cd',launch['workdir'],'--',launch['python'],
        '-B','-m','learning.task_action_observation_v1.analysis_entry','guard',
        '--out',OUT.relative_to(ROOT).as_posix()]
    if fixture:command.append('--fixture')
    started=time.monotonic();child=None;known={};error=None;capture=None;cleanup=None;linux=None
    try:
        child=CapturedProcess(command,cwd=ROOT,log=folder/'transport.local.log',
            limit_bytes=cfg['storage']['process_log_bytes'])
        while child.process.poll() is None:
            observe_processes(child,known);child.check();deadline_check()
            if time.monotonic()-started>limit-cfg['transport_limits']['cleanup_reserve_seconds']:
                raise TimeoutError('Analysis transport work allowance exhausted')
            time.sleep(.1)
        capture=child.finish()
        if child.process.returncode!=0:raise RuntimeError('Registered analysis transport returned failure')
        linux=read(folder/guard_name)
        if not linux['completed']:raise RuntimeError('Registered Linux analysis did not complete')
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        if error is not None and not (OUT/'cancel.local.json').exists():
            write_new(OUT/'cancel.local.json',dict(cancel=True,reason='owned transport stopping'))
        cleanup_error=None
        if child is not None:
            try:
                if child.process.poll() is None:
                    child.process.wait(timeout=cfg['transport_limits']['cleanup_reserve_seconds'])
                observe_processes(child,known)
                cleanup=reap(child,known)
                if (folder/guard_name).exists():linux=read(folder/guard_name)
            except BaseException as exc:
                cleanup_error=type(exc).__name__
        elapsed=time.monotonic()-started
        linux_clean=linux is not None and all(linux.get(k) is True for k in (
            'owned_process_group_signalled','direct_child_reaped','no_live_owned_session_members_before_reap'))
        clean=cleanup is not None and cleanup.get('owned_instances_remaining')==0 and cleanup.get('reader_joined') is True
        passed=error is None and cleanup_error is None and linux_clean and clean and elapsed<=limit
        result=dict(passed=passed,fixture=fixture,primary_error_kind=error,cleanup_error_kind=cleanup_error,
            linux_cleanup_proven=linux_clean,windows_cleanup_proven=clean,cleanup=cleanup,capture=capture,
            elapsed_seconds=elapsed,limit_seconds=limit,within_transport_deadline=elapsed<=limit,
            retry_allowed=False,scientific_activity='none' if fixture else 'new registered game analysis only')
        write_new(folder/'transport-result.json',result)
    if not result['passed']:raise RuntimeError('Registered analysis transport or cleanup failed; no retry')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['fixture','analysis'])
    print(json.dumps(run(parser.parse_args().command=='fixture')))
