"""Synthetic workload definition; caller must supply future frozen execution gate.

No work occurs at import. `run_owned` does not itself implement registration or
authorization; a separately reviewed gate must invoke it exactly once.
"""
import psutil
from .sampling import supervise

LIMITS = dict(max_rss_bytes=512*1024**2,max_commit_bytes=1024**3,min_available_bytes=2*1024**3,
              wall_seconds=30,poll_seconds=0.25,cleanup_seconds=10)


def command(executable):
    if not isinstance(executable,str) or not executable: raise ValueError('Invalid executable')
    stable='import time; data=bytearray(24*1024*1024); time.sleep(8)'
    brief='import time; time.sleep(0.05)'
    launcher=('import subprocess,sys\n'
        'stable=subprocess.Popen([sys.executable,"-B","-c",'+repr(stable)+'],creationflags=subprocess.CREATE_NO_WINDOW)\n'
        'failed=False\n'
        'try:\n'
        ' for _ in range(16):\n'
        '  p=subprocess.Popen([sys.executable,"-B","-c",'+repr(brief)+'],creationflags=subprocess.CREATE_NO_WINDOW)\n'
        '  if p.wait()!=0: failed=True; break\n'
        'finally:\n'
        ' stable_code=stable.wait()\n'
        'raise SystemExit(1 if failed or stable_code!=0 else 0)')
    return [executable,'-B','-c',launcher]


class InjectOnce:
    """One artificial disappearance at the first observed member, then real sampling."""
    def __init__(self,factory=psutil.Process):
        self.factory=factory
        self.injections=0

    def __call__(self,pid):
        if self.injections == 0:
            self.injections=1
            raise psutil.NoSuchProcess(pid)
        return self.factory(pid)


def accepted(result,injections):
    return bool(injections==1 and result.get('passed') is True and result.get('assigned') is True
        and result.get('resumed') is True and result.get('launcher_reaped') is True
        and result.get('job_members_zero') is True and result.get('exit_code') == 0
        and result.get('error_kind') is None and result.get('cleanup_error_kind') is None
        and result.get('rss_complete_samples',0) >= 1 and result.get('rss_sampling_retries',0) >= 1
        and result.get('rss_discarded_samples',0) >= 1)


def run_owned(api,executable,cwd,*,process_factory=psutil.Process):
    """Call exactly once only after an external preregistration gate has passed.

    `api` must be the caller-created WindowsAPI; no unowned process is managed.
    """
    injected=InjectOnce(process_factory)
    result=supervise(api,command(executable),cwd,dict(LIMITS),process_factory=injected)
    return dict(passed=accepted(result,injected.injections),supervision=result,
        artificial_disappearances_injected=injected.injections,
        interpretation='Artificial injection demonstrates retry only; it does not establish the cause of an earlier failure or a natural race',
        retry_allowed=False)
