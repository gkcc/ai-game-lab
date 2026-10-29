"""Windows, source-bound outer launcher for one registered panel attempt.

Execute this file directly with the project interpreter: ``python -B -I``.
Only standard-library imports occur here.  The caller supplies the public Git
repository, local registered stage, preregistration commit and standalone
artifact-helper path.  No seed, command line or private path is printed.

The coordinator and its descendants are created in a dedicated Windows Job.
The scratch ownership marker remains incomplete if process creation, Job
membership, exit, handle cleanup or scratch cleanup cannot be confirmed.  A
failed or interrupted launch is not resumed.  These checks do not prevent
same-user mutation races, custom Python loaders, or kernel/driver defects.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import types


OUTER_SUPERVISION = {
    'hard_deadline': 'registered_deadline_utc',
    'scope': 'owned_outer_job_including_descendants',
    'termination': 'terminate_owned_job_and_wait_for_empty',
    'cleanup_unknown': 'retain_scratch_marker',
}
ENTRY = 'learning.panel_eval_v2.runner'
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_STAGE = re.compile(r'[a-z0-9][a-z0-9-]{0,95}\Z')
_SMALL = 1024 * 1024
_PUBLIC = ('root-decision.json', 'registration.json', 'freeze.json', 'pre-review.json', 'safety-registration.json')


def _absolute(value):
    try:
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid absolute path')
        if os.name == 'nt':
            for part in path.parts:
                if part != path.anchor and (':' in part or part.rstrip(' .') != part or Path(part).is_reserved()):
                    raise ValueError('Invalid path component')
        return Path(os.path.abspath(path))
    except (TypeError, OSError):
        raise ValueError('Invalid absolute path') from None


def _no_links(path):
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Path contains a link or reparse point')


def _read(path, maximum=_SMALL):
    _no_links(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError('Missing or oversized input')
    with path.open('rb') as stream:
        data = stream.read(maximum+1)
    if len(data) > maximum:
        raise ValueError('Oversized input')
    return data


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('Duplicate metadata field')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Invalid metadata value')
    try:
        return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError('Invalid metadata') from None


def _git(repo,*args):
    try:
        return subprocess.check_output(['git','-C',str(repo),*args], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        raise ValueError('Repository verification failed') from None


def _deadline(value):
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError('Missing timezone')
        result = result.astimezone(timezone.utc)
        if result <= datetime.now(timezone.utc):
            raise ValueError('Expired')
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Invalid registered deadline') from None
    return result


def _verify_helper_commitment(cfg, decision, review):
    if not all(isinstance(value,dict) for value in (cfg,decision,review)):
        raise ValueError('Invalid helper commitment evidence')
    commitment=cfg.get('scratch_helper_commitment')
    if (not isinstance(commitment,str) or not re.fullmatch(r'[0-9a-f]{64}',commitment)
        or decision.get('scratch_helper_commitment') != commitment
        or review.get('scratch_helper_commitment') != commitment
        or review.get('passed') is not True or review.get('review_basis') != 'independent_subagent'
        or not isinstance(review.get('reviewer'),str) or not review['reviewer'].startswith('/root/')
        or not review['reviewer'][6:] or review.get('open_findings') != []
        or not isinstance(review.get('checks'),list) or not review['checks']
        or any(not isinstance(item,str) or not item.strip() for item in review['checks'])):
        raise ValueError('Helper commitment evidence changed')


def _verify_sources(repo, stage, prereg, project, launch_path):
    """Bind this script, the bootstrap and stage config to clean Git bytes."""
    repo, stage, project, launch_path = map(_absolute, (repo, stage, project, launch_path))
    for path in (repo, stage, project, launch_path):
        _no_links(path)
    if (not repo.is_dir() or not project.is_dir() or not stage.is_dir()
        or stage.parent != project/'artifacts' or not _STAGE.fullmatch(stage.name)
        or launch_path != project/'learning'/'panel_eval_v2'/'launch.py'
        or not isinstance(prereg, str) or not _COMMIT.fullmatch(prereg)):
        raise ValueError('Invalid repository or panel identity')
    root = Path(_git(repo,'rev-parse','--show-toplevel').decode('utf-8').strip())
    if _absolute(root) != repo or _git(repo,'rev-parse','HEAD').decode('ascii').strip() != prereg:
        raise ValueError('Unexpected public repository HEAD')
    if _git(repo,'status','--porcelain').strip():
        raise ValueError('Public repository is not clean')
    parents = _git(repo,'rev-list','--parents','-n','1',prereg).decode('ascii').split()
    if len(parents) != 2 or parents[0] != prereg or not _COMMIT.fullmatch(parents[1]):
        raise ValueError('Registration requires one frozen parent')
    prefix = 'review/evidence/'+stage.name+'/'
    expected = {prefix+name for name in _PUBLIC}
    changed = {p for p in _git(repo,'diff','--name-only','-z',parents[1],prereg,'--').decode('utf-8').split('\0') if p}
    if changed != expected:
        raise ValueError('Registration changed unexpected files')
    identities = {}
    for name in ('launch.py','bootstrap.py'):
        rel = 'kaggriculture/learning/panel_eval_v2/'+name
        local = _read(project/'learning'/'panel_eval_v2'/name)
        public = _read(repo/rel)
        now, before = _git(repo,'show',prereg+':'+rel), _git(repo,'show',parents[1]+':'+rel)
        if local != public or local != now or now != before:
            raise ValueError('Launcher source differs from frozen Git bytes')
        identities[name] = _sha(local)
    files = {}
    for name in _PUBLIC:
        local = _read(stage/name)
        if local != _read(repo/prefix/name) or local != _git(repo,'show',prereg+':'+prefix+name):
            raise ValueError('Registration evidence changed')
        files[name] = local
    cfg, decision, freeze, review = (_json(files[name]) for name in
        ('registration.json','root-decision.json','freeze.json','pre-review.json'))
    execution = _json(_read(stage/'pre-execution-commit.json'))
    if (not isinstance(cfg,dict) or not isinstance(decision,dict) or not isinstance(freeze,dict) or not isinstance(execution,dict)
        or cfg.get('baseline_head') != parents[1] or freeze.get('parent_commit') != parents[1]
        or cfg.get('outer_supervision') != OUTER_SUPERVISION
        or execution != {'commit':prereg,'registration_sha256':_sha(files['registration.json']),
                         'freeze_sha256':_sha(files['freeze.json'])}):
        raise ValueError('Registration supervision or execution binding changed')
    _verify_helper_commitment(cfg,decision,review)
    _deadline(cfg.get('deadline_utc'))
    if (_git(repo,'rev-parse','HEAD').decode('ascii').strip() != prereg
        or _git(repo,'status','--porcelain').strip()):
        raise ValueError('Repository changed during verification')
    return cfg, decision, identities


def _verify_interpreter(project, cfg):
    executable = project/'.venv'/'Scripts'/'python.exe'
    venv = project/'.venv'
    _no_links(executable)
    if (os.name != 'nt' or not os.path.samefile(sys.executable, executable)
        or not os.path.samefile(sys.prefix, venv) or os.path.samefile(sys.prefix,sys.base_prefix)
        or sys.flags.optimize != 0 or 'PYTHONOPTIMIZE' in os.environ):
        raise ValueError('Unexpected project interpreter')
    raw = _read(executable, 64*_SMALL)
    digest = _sha(raw)
    if not isinstance(cfg.get('environment'),dict) or cfg['environment'].get('interpreter_sha256') != digest:
        raise ValueError('Registered interpreter identity changed')
    return executable, digest


def _load_helper(path,data):
    module = types.ModuleType('panel_launcher_owned_scratch')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    if not all(callable(getattr(module, name, None)) for name in ('scratch_directory','protect_children','process_identity','read_marker','default_root','sweep_stale')):
        raise ValueError('Standalone scratch helper is incompatible')
    # This one run must never sweep other chats' old scratch directories.  All
    # ownership, child and deletion checks for its own directory stay intact.
    module.sweep_stale = lambda *args,**kwargs: {'scanned':0,'removed':[],'bytes':0}
    return module


def _helper(path, stage, cfg, decision):
    """Verify the preregistered salted helper identity before executing it."""
    path = _absolute(path)
    data = _read(path, 4*_SMALL)
    private = _read(stage/'private-seeds.local.json',16384)
    if (not isinstance(cfg,dict) or not isinstance(decision,dict)
        or _sha(private) != cfg.get('seed_document_sha256')
        or decision.get('seed_document_sha256') != cfg.get('seed_document_sha256')):
        raise ValueError('Private helper commitment source changed')
    document = _json(private)
    if (not isinstance(document,dict) or set(document) != {'version','seeds','salt'}
        or type(document['version']) is not int or document['version'] != 1
        or not isinstance(document['seeds'],list) or len(document['seeds']) != 4
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in document['seeds'])
        or len(set(document['seeds'])) != 4 or not isinstance(document['salt'],str)
        or not re.fullmatch(r'[0-9a-f]{64}',document['salt'])):
        raise ValueError('Invalid private helper commitment source')
    commitment=hashlib.sha256(b'panel-scratch-helper-v1\x00'+bytes.fromhex(document['salt'])+
                              hashlib.sha256(data).digest()).hexdigest()
    if (cfg.get('scratch_helper_commitment') != commitment
        or decision.get('scratch_helper_commitment') != commitment):
        raise ValueError('Scratch helper identity differs from preregistration')
    return _load_helper(path,data),_sha(data)


def _record(directory, name, value):
    if name not in ('launcher-started.json','launcher-result.json'):
        raise ValueError('Invalid launcher receipt')
    _no_links(directory)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if len(payload) > 16384:
        raise ValueError('Launcher receipt exceeds budget')
    with (directory/name).open('xb') as stream:
        stream.write(payload)


class _BasicLimit(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
        ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),
        ('ActiveProcessLimit',wintypes.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name,ctypes.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
        'ReadTransferCount','WriteTransferCount','OtherTransferCount')]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [('BasicLimitInformation',_BasicLimit),('IoInfo',_IoCounters),('ProcessMemoryLimit',ctypes.c_size_t),
        ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name,ctypes.c_longlong) for name in ('TotalUserTime','TotalKernelTime','ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime')] + [
        ('TotalPageFaultCount',wintypes.DWORD),('TotalProcesses',wintypes.DWORD),('ActiveProcesses',wintypes.DWORD),('TotalTerminatedProcesses',wintypes.DWORD)]


class WindowsJob:
    """Small outer Job boundary; only handles created by this object are used."""
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Windows required')
        self.k = ctypes.WinDLL('kernel32',use_last_error=True)
        k = self.k
        k.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]; k.CreateJobObjectW.restype=wintypes.HANDLE
        k.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]; k.SetInformationJobObject.restype=wintypes.BOOL
        k.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]; k.AssignProcessToJobObject.restype=wintypes.BOOL
        k.ResumeThread.argtypes=[wintypes.HANDLE]; k.ResumeThread.restype=wintypes.DWORD
        k.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT]; k.TerminateJobObject.restype=wintypes.BOOL
        k.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]; k.TerminateProcess.restype=wintypes.BOOL
        k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]; k.WaitForSingleObject.restype=wintypes.DWORD
        k.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]; k.QueryInformationJobObject.restype=wintypes.BOOL
        k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]; k.GetExitCodeProcess.restype=wintypes.BOOL
        k.CloseHandle.argtypes=[wintypes.HANDLE]; k.CloseHandle.restype=wintypes.BOOL

    def _require(self,value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def create_job(self): return self._require(self.k.CreateJobObjectW(None,None))
    def configure(self,job):
        value=_ExtendedLimit(); value.BasicLimitInformation.LimitFlags=0x2000  # KILL_ON_JOB_CLOSE, no breakaway
        self._require(self.k.SetInformationJobObject(job,9,ctypes.byref(value),ctypes.sizeof(value)))

    def create_suspended(self,command,cwd):
        import _winapi
        import msvcrt
        fd=os.open(os.devnull,os.O_RDWR)
        try:
            handle=msvcrt.get_osfhandle(fd)
            os.set_handle_inheritable(handle,True)
            startup=subprocess.STARTUPINFO()
            startup.dwFlags |= _winapi.STARTF_USESTDHANDLES
            startup.hStdInput=startup.hStdOutput=startup.hStdError=handle
            startup.lpAttributeList={'handle_list':[handle]}
            process,thread,pid,_ = _winapi.CreateProcess(None,subprocess.list2cmdline(command),None,None,True,
                0x00000004 | 0x08000000,None,str(cwd),startup)  # SUSPENDED | NO_WINDOW
            return process,thread,pid
        finally:
            os.close(fd)

    def assign(self,job,process): self._require(self.k.AssignProcessToJobObject(job,process))
    def resume(self,thread):
        count=self.k.ResumeThread(thread)
        if count != 1:
            raise RuntimeError('Unexpected suspended thread state')
    def terminate_job(self,job): self._require(self.k.TerminateJobObject(job,1))
    def terminate_process(self,process): self._require(self.k.TerminateProcess(process,1))
    def wait(self,process,ms):
        result=self.k.WaitForSingleObject(process,ms)
        if result == 0: return True
        if result == 258: return False
        raise ctypes.WinError(ctypes.get_last_error())
    def active(self,job):
        data=_Accounting()
        self._require(self.k.QueryInformationJobObject(job,1,ctypes.byref(data),ctypes.sizeof(data),None))
        return data.ActiveProcesses
    def exit_code(self,process):
        code=wintypes.DWORD(); self._require(self.k.GetExitCodeProcess(process,ctypes.byref(code))); return code.value
    def close(self,handle): self._require(self.k.CloseHandle(handle))


class _Deadline:
    """Use both UTC and monotonic time so a wall-clock rollback cannot extend a run."""
    def __init__(self, deadline, utc_now=None, monotonic=None):
        self.utc_now=utc_now or (lambda: datetime.now(timezone.utc))
        self.monotonic=monotonic or time.monotonic
        self.deadline=deadline
        now=self.utc_now(); current=self.monotonic()
        if (not isinstance(now,datetime) or now.tzinfo is None or now.utcoffset() is None
            or type(current) not in (int,float) or not math.isfinite(current)):
            raise ValueError('Unreliable deadline clock')
        remaining=(deadline-now).total_seconds()
        if not math.isfinite(remaining):
            raise ValueError('Unreliable deadline clock')
        self.last=current
        self.monotonic_deadline=current+remaining

    def remaining(self):
        now=self.utc_now(); current=self.monotonic()
        if (not isinstance(now,datetime) or now.tzinfo is None or now.utcoffset() is None
            or type(current) not in (int,float) or not math.isfinite(current) or current < self.last):
            raise ValueError('Unreliable deadline clock')
        self.last=current
        value=min((self.deadline-now).total_seconds(),self.monotonic_deadline-current)
        if not math.isfinite(value):
            raise ValueError('Unreliable deadline clock')
        return value


def _supervise(api, command, cwd, deadline, helper, scratch, scratch_root, *, _clock=None):
    """Run one suspended-and-assigned process; preserve uncertain ownership."""
    job = process = thread = None
    spawned = assigned = resumed = reaped = empty = False
    attempted = False
    error = reason = None
    code = None
    close_failed = False
    clock_failed = False
    clock = _clock if _clock is not None else _Deadline(deadline)
    def remaining():
        nonlocal clock_failed
        try:
            return clock.remaining()
        except BaseException:
            clock_failed=True
            raise
    helper.protect_children(scratch, [], root=scratch_root, complete=False)
    try:
        job=api.create_job(); api.configure(job)
        attempted=True
        process,thread,pid=api.create_suspended(command,cwd); spawned=True
        state,identity=helper.process_identity(pid)
        if state != 'active' or identity is None:
            raise ValueError('Child identity unavailable')
        helper.protect_children(scratch,[{'pid':pid,'process_identity':identity}],root=scratch_root,complete=False)
        api.assign(job,process); assigned=True
        if remaining() <= 0:
            reason='registered_hard_deadline'
        else:
            api.resume(thread); resumed=True
        while True:
            if reason:
                break
            left=max(0,remaining())
            reaped=api.wait(process,min(1000,int(left*1000)))
            empty=api.active(job)==0
            if remaining() <= 0:
                reason='registered_hard_deadline'
                break
            if reaped and empty:
                code=api.exit_code(process)
                if remaining() <= 0:
                    reason='registered_hard_deadline'
                break
            # A terminated leader's process handle remains signaled while its
            # owned descendants may still run; waiting on that handle again
            # would spin. Poll the Job with a deadline-bounded pause instead.
            if reaped:
                delay=min(.1,max(0,remaining()))
                if delay:
                    time.sleep(delay)
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        if process is not None and (error is not None or reason is not None or not (reaped and empty)):
            try:
                if assigned:
                    api.terminate_job(job)
                else:
                    api.terminate_process(process)
            except BaseException:
                error=error or 'OwnedTerminationFailed'
            try:
                reaped=api.wait(process,10000)
                end=time.monotonic()+10
                while True:
                    empty=job is not None and api.active(job)==0
                    if empty or time.monotonic() >= end:
                        break
                    time.sleep(.05)
                if reaped:
                    code=api.exit_code(process)
            except BaseException:
                reaped=empty=False
                error=error or 'OwnedCleanupUnconfirmed'
        for handle in (thread,process,job):
            if handle is not None:
                try:
                    api.close(handle)
                except BaseException:
                    close_failed=True
        known_empty = not close_failed and not clock_failed and ((not attempted) or (spawned and reaped and empty))
        if known_empty:
            try:
                helper.protect_children(scratch,[],root=scratch_root,complete=True)
            except BaseException:
                known_empty=False
                error=error or 'OwnershipCleanupUnconfirmed'
    return dict(passed=bool(error is None and reason is None and assigned and resumed and reaped and empty and code == 0 and known_empty),
        error_kind=error, stop_reason=reason, child_reaped=reaped, job_members_zero=empty,
        handles_closed=not close_failed, cleanup_known=known_empty, exit_code=code, retry_allowed=False)


def launch(repo, stage, prereg, helper_path):
    _startup()
    project = _absolute(Path(__file__).parents[2])
    launch_path = _absolute(__file__)
    cfg, decision, source = _verify_sources(repo,stage,prereg,project,launch_path)
    executable, interpreter_sha = _verify_interpreter(project,cfg)
    stage = _absolute(stage)
    helper, helper_sha = _helper(helper_path,stage,cfg,decision)
    if (stage/'results').exists() or (stage/'games-started.json').exists():
        raise ValueError('Coordinator attempt already consumed')
    receipts=stage/'replays'
    _no_links(receipts)
    receipts.mkdir(exist_ok=False)
    _record(receipts,'launcher-started.json',dict(schema='panel-outer-launch-v1',prereg_commit=prereg,
        launcher_sha256=source['launch.py'],bootstrap_sha256=source['bootstrap.py'],
        interpreter_sha256=interpreter_sha,standalone_helper_sha256=helper_sha,
        process_flags=['-B','-X pycache_prefix=<fresh direct child>','-I'],entry=ENTRY,
        native_stdio='NUL via explicit inherited handle list',temp_environment_changed=False,
        job='Windows outer Job, no breakaway, kill on close',retry_allowed=False))
    scratch_root = helper.default_root()
    scratch = None
    scratch_run_id = None
    result = None
    retained = None  # Unknown until the helper returns a directory or closes it.
    try:
        with helper.scratch_directory('panel-outer-launch') as scratch:
            marker=helper.read_marker(scratch,root=scratch_root)
            scratch_run_id=marker['run_id']
            prefix=scratch/'coordinator-cache'
            if prefix.exists():
                raise ValueError('Fresh cache prefix is occupied')
            _verify_sources(repo,stage,prereg,project,launch_path)
            _verify_interpreter(project,cfg)
            command=[str(executable),'-B','-X','pycache_prefix='+str(prefix),'-I',
                     str(project/'learning'/'panel_eval_v2'/'bootstrap.py'),ENTRY,
                     '--out',str(stage),'--prereg',prereg,'--scratch',str(scratch)]
            result=_supervise(WindowsJob(),command,project,_deadline(cfg['deadline_utc']),helper,scratch,scratch_root)
            if prefix.exists():
                result.update(passed=False,cache_prefix_absent=False)
            else:
                result['cache_prefix_absent']=True
        retained=scratch.exists()
    except BaseException as exc:
        if result is None:
            result = dict(passed=False,error_kind=type(exc).__name__,cleanup_known=False,retry_allowed=False)
        else:
            result.update(passed=False,outer_error_kind=type(exc).__name__,cleanup_known=False)
        if scratch is not None:
            retained=True if scratch.exists() else None
    if result is None:
        result = dict(passed=False,error_kind='MissingSupervisorResult',cleanup_known=False,retry_allowed=False)
    result['scratch_retained']=retained
    result['scratch_run_id_if_retained']=scratch_run_id if retained is True else None
    # This receipt is private local recovery evidence. It is never a public
    # certificate or a reason to adopt an unmarked directory.
    result['scratch_path_if_retained']=str(scratch) if retained is True else None
    result['private_receipt_never_publish']=True
    if retained is not False:
        result['passed']=False
    if result.get('passed'):
        try:
            complete=_json(_read(stage/'results'/'coordinator-complete.json',16384))
            if not isinstance(complete,dict) or complete.get('passed') is not True or complete.get('economic_evaluation_completed') is not True:
                raise ValueError('Coordinator completion receipt missing')
        except BaseException:
            result.update(passed=False,error_kind='CoordinatorCompletionUnconfirmed')
    _record(receipts,'launcher-result.json',result)
    return result['passed']


def _startup():
    argv=getattr(sys,'orig_argv',())
    if (not isinstance(argv,list) or len(argv)<4 or argv[1:3] != ['-B','-I']
        or sys.flags.dont_write_bytecode != 1 or sys.flags.isolated != 1):
        raise ValueError('Launcher requires direct isolated startup')


def main():
    _startup()
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--prereg',required=True)
    parser.add_argument('--helper',required=True)
    args=parser.parse_args()
    return launch(args.repo,args.stage,args.prereg,args.helper)


if __name__ == '__main__':
    try:
        from contextlib import redirect_stdout, redirect_stderr
        with open(os.devnull, 'w', encoding='utf-8') as sink, redirect_stdout(sink), redirect_stderr(sink):
            ok=main()
    except BaseException:
        ok=False
    raise SystemExit(0 if ok else 1)
