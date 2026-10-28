"""One synthetic exit-timing scenario; an external frozen gate must call it."""
import psutil
from .sampling import supervise

LIMITS=dict(max_rss_bytes=512*1024**2,max_commit_bytes=1024**3,min_available_bytes=2*1024**3,
            wall_seconds=30,poll_seconds=.25,cleanup_seconds=10)
INJECTION_WAIT_MS=5000


def command(executable):
    if not isinstance(executable,str) or not executable: raise ValueError('Invalid executable')
    return [executable,'-B','-c','import time; time.sleep(2)']


class ArrangeExit:
    """On the second member snapshot wait, bounded, for our own launcher."""
    def __init__(self,api):
        self.api=api; self.process=None; self.snapshots=0; self.injections=0

    def __getattr__(self,name): return getattr(self.api,name)

    def create_suspended(self,command,cwd):
        process,thread=self.api.create_suspended(command,cwd)
        self.process=process
        return process,thread

    def pids(self,job):
        self.snapshots+=1
        if self.snapshots==2:
            self.injections+=1
            if self.process is None or self.api.wait(self.process,INJECTION_WAIT_MS) is not True:
                raise TimeoutError('Synthetic natural exit did not occur within bound')
        return self.api.pids(job)


def accepted(result,injections):
    return (injections==1 and result.get('passed') is True and result.get('assigned') is True
        and result.get('resumed') is True and result.get('launcher_reaped') is True
        and result.get('job_members_zero') is True and result.get('exit_code')==0
        and result.get('error_kind') is None and result.get('cleanup_error_kind') is None
        and result.get('stop_reason') is None and result.get('confirmed_empty_exits')==1
        and result.get('rss_complete_samples',0)>=1 and result.get('rss_discarded_samples',0)>=1)


def run_owned(api,executable,cwd,*,preflight,process_factory=psutil.Process):
    """Call once only after preregistration; preflight must use frozen host policy."""
    if not callable(preflight) or preflight() is not True: raise RuntimeError('Preflight failed')
    arranged=ArrangeExit(api)
    result=supervise(arranged,command(executable),cwd,dict(LIMITS),process_factory=process_factory)
    return dict(passed=accepted(result,arranged.injections),supervision=result,
        artificial_exit_timing_arrangements=arranged.injections,retry_allowed=False,
        interpretation='Artificially arranged exit timing; does not prove a natural race or a hard memory cap')
