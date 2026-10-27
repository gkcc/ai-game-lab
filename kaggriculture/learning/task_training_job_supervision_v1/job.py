"""Windows Job supervision primitives; no process is created at import time."""
import ctypes
from ctypes import wintypes
import os
import subprocess
import time
import psutil

CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
KILL_ON_JOB_CLOSE = 0x00002000


class BasicLimit(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
        ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),
        ('ActiveProcessLimit',wintypes.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(n,ctypes.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
        'ReadTransferCount','WriteTransferCount','OtherTransferCount')]


class ExtendedLimit(ctypes.Structure):
    _fields_ = [('BasicLimitInformation',BasicLimit),('IoInfo',IoCounters),('ProcessMemoryLimit',ctypes.c_size_t),
        ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]


class StartupInfo(ctypes.Structure):
    _fields_ = [('cb',wintypes.DWORD),('lpReserved',wintypes.LPWSTR),('lpDesktop',wintypes.LPWSTR),
        ('lpTitle',wintypes.LPWSTR),('dwX',wintypes.DWORD),('dwY',wintypes.DWORD),('dwXSize',wintypes.DWORD),
        ('dwYSize',wintypes.DWORD),('dwXCountChars',wintypes.DWORD),('dwYCountChars',wintypes.DWORD),
        ('dwFillAttribute',wintypes.DWORD),('dwFlags',wintypes.DWORD),('wShowWindow',wintypes.WORD),
        ('cbReserved2',wintypes.WORD),('lpReserved2',ctypes.c_void_p),('hStdInput',wintypes.HANDLE),
        ('hStdOutput',wintypes.HANDLE),('hStdError',wintypes.HANDLE)]


class ProcessInfo(ctypes.Structure):
    _fields_ = [('hProcess',wintypes.HANDLE),('hThread',wintypes.HANDLE),('dwProcessId',wintypes.DWORD),('dwThreadId',wintypes.DWORD)]


class Accounting(ctypes.Structure):
    _fields_ = [(n,ctypes.c_longlong) for n in ('TotalUserTime','TotalKernelTime','ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime')] + [
        ('TotalPageFaultCount',wintypes.DWORD),('TotalProcesses',wintypes.DWORD),('ActiveProcesses',wintypes.DWORD),('TotalTerminatedProcesses',wintypes.DWORD)]


class MemoryUsage(ctypes.Structure):
    _fields_ = [('JobMemory',ctypes.c_ulonglong),('PeakJobMemoryUsed',ctypes.c_ulonglong)]


class WindowsAPI:
    def __init__(self):
        if os.name != 'nt': raise OSError('Windows required')
        self.k = ctypes.WinDLL('kernel32',use_last_error=True)
        k = self.k
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p,wintypes.LPCWSTR]; k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]; k.SetInformationJobObject.restype = wintypes.BOOL
        k.CreateProcessW.argtypes = [wintypes.LPCWSTR,wintypes.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,wintypes.BOOL,wintypes.DWORD,ctypes.c_void_p,wintypes.LPCWSTR,ctypes.POINTER(StartupInfo),ctypes.POINTER(ProcessInfo)]; k.CreateProcessW.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,wintypes.HANDLE]; k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.ResumeThread.argtypes = [wintypes.HANDLE]; k.ResumeThread.restype = wintypes.DWORD
        k.TerminateJobObject.argtypes = [wintypes.HANDLE,wintypes.UINT]; k.TerminateJobObject.restype = wintypes.BOOL
        k.TerminateProcess.argtypes = [wintypes.HANDLE,wintypes.UINT]; k.TerminateProcess.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]; k.WaitForSingleObject.restype = wintypes.DWORD
        k.QueryInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]; k.QueryInformationJobObject.restype = wintypes.BOOL
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]; k.GetExitCodeProcess.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]; k.CloseHandle.restype = wintypes.BOOL

    def _require(self, value):
        if not value: raise ctypes.WinError(ctypes.get_last_error())
        return value

    def create_job(self): return self._require(self.k.CreateJobObjectW(None,None))
    def configure(self,job):
        info = ExtendedLimit(); info.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
        self._require(self.k.SetInformationJobObject(job,9,ctypes.byref(info),ctypes.sizeof(info)))
    def create_suspended(self,command,cwd):
        startup = StartupInfo(); startup.cb = ctypes.sizeof(startup)
        result = ProcessInfo(); cmd = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        self._require(self.k.CreateProcessW(None,cmd,None,None,False,CREATE_SUSPENDED|CREATE_NO_WINDOW,None,str(cwd),ctypes.byref(startup),ctypes.byref(result)))
        return result.hProcess,result.hThread
    def assign(self,job,process): self._require(self.k.AssignProcessToJobObject(job,process))
    def resume(self,thread):
        previous = self.k.ResumeThread(thread)
        if previous == 0xffffffff: raise ctypes.WinError(ctypes.get_last_error())
        if previous != 1: raise RuntimeError('Unexpected prior thread suspend count')
    def terminate_job(self,job): self._require(self.k.TerminateJobObject(job,1))
    def terminate_process(self,process): self._require(self.k.TerminateProcess(process,1))
    def wait(self,process,ms):
        value = self.k.WaitForSingleObject(process,ms)
        if value == 0: return True
        if value == 258: return False
        raise ctypes.WinError(ctypes.get_last_error())
    def exit_code(self,process):
        code = wintypes.DWORD(); self._require(self.k.GetExitCodeProcess(process,ctypes.byref(code))); return code.value
    def active(self,job):
        data = Accounting(); self._require(self.k.QueryInformationJobObject(job,1,ctypes.byref(data),ctypes.sizeof(data),None)); return data.ActiveProcesses
    def committed(self,job):
        data = MemoryUsage(); self._require(self.k.QueryInformationJobObject(job,28,ctypes.byref(data),ctypes.sizeof(data),None)); return data.JobMemory
    def pids(self,job):
        # Fixed bound; fail closed if a larger job appears rather than sampling a partial tree.
        class ProcessList(ctypes.Structure):
            _fields_ = [('assigned',wintypes.DWORD),('listed',wintypes.DWORD),('ids',ctypes.c_size_t*256)]
        data = ProcessList(); self._require(self.k.QueryInformationJobObject(job,3,ctypes.byref(data),ctypes.sizeof(data),None))
        if data.assigned != data.listed: raise RuntimeError('Incomplete job membership sample')
        return list(data.ids[:data.listed])
    def sample_rss(self,job):
        # One membership snapshot; PIDs are used only for observation, never termination.
        members = self.pids(job)
        return len(members),sum(psutil.Process(pid).memory_info().rss for pid in members)
    def available(self): return psutil.virtual_memory().available
    def close(self,handle): self._require(self.k.CloseHandle(handle))


def supervise(api, command, cwd, limits, clock=time.monotonic, sleep=time.sleep):
    """Own only returned handles; report uncertainty instead of claiming isolation."""
    required = ('max_rss_bytes','max_commit_bytes','min_available_bytes','wall_seconds','poll_seconds','cleanup_seconds')
    if set(limits) != set(required) or any(type(limits[k]) not in (int,float) or limits[k] <= 0 for k in required):
        raise ValueError('Invalid supervision limits')
    if not isinstance(command,list) or not command or any(not isinstance(x,str) or not x for x in command): raise ValueError('Invalid command')
    job = process = thread = None
    assigned = resumed = False
    error = cleanup_error = reason = None
    termination_error = wait_error = membership_error = close_error = None
    peak_rss = peak_commit = max_members = 0
    threshold_sample_members = None
    start = clock()
    reaped = members_zero = False
    exit_code = None
    try:
        job = api.create_job(); api.configure(job)
        process,thread = api.create_suspended(command,cwd)
        api.assign(job,process); assigned = True
        api.resume(thread); resumed = True
        while True:
            if api.active(job) == 0:
                members_zero = True
                reaped = api.wait(process,0)
                if reaped: exit_code = api.exit_code(process)
                if clock()-start >= limits['wall_seconds']: reason = 'wall_limit'
                break
            member_count,rss = api.sample_rss(job)
            max_members = max(max_members,member_count)
            commit = api.committed(job)
            peak_rss,peak_commit = max(peak_rss,rss),max(peak_commit,commit)
            if rss > limits['max_rss_bytes']:
                reason = 'rss_limit'; threshold_sample_members = member_count; break
            if commit > limits['max_commit_bytes']: reason = 'commit_limit'; break
            if api.available() < limits['min_available_bytes']: reason = 'available_memory_limit'; break
            if clock()-start >= limits['wall_seconds']: reason = 'wall_limit'; break
            sleep(limits['poll_seconds'])
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        if process is not None and (error or reason or not members_zero):
            try:
                if assigned: api.terminate_job(job)
                else: api.terminate_process(process)
            except BaseException as exc:
                termination_error = type(exc).__name__
        try:
            if process is not None and not reaped:
                reaped = api.wait(process,int(limits['cleanup_seconds']*1000))
                if reaped: exit_code = api.exit_code(process)
        except BaseException as exc:
            wait_error = type(exc).__name__
            reaped = False
        try:
            if assigned:
                end = clock()+limits['cleanup_seconds']
                while True:
                    members_zero = api.active(job) == 0
                    if members_zero or clock() >= end: break
                    sleep(min(limits['poll_seconds'],max(0,end-clock())))
        except BaseException as exc:
            membership_error = type(exc).__name__
            members_zero = False
        for handle in (thread,process,job):
            if handle is not None:
                try: api.close(handle)
                except BaseException as exc: close_error = close_error or type(exc).__name__
        cleanup_error = termination_error or wait_error or membership_error or close_error
    return dict(passed=error is None and cleanup_error is None and reason is None and assigned and resumed and reaped and members_zero and exit_code == 0,
        assigned=assigned,resumed=resumed,launcher_reaped=reaped,job_members_zero=members_zero,
        error_kind=error,cleanup_error_kind=cleanup_error,stop_reason=reason,exit_code=exit_code,
        termination_error_kind=termination_error,wait_error_kind=wait_error,
        membership_error_kind=membership_error,close_error_kind=close_error,
        peak_sampled_rss_bytes=peak_rss,peak_sampled_job_commit_bytes=peak_commit,
        max_sampled_members=max_members,rss_threshold_sample_members=threshold_sample_members,
        measurement_limit='Same-snapshot member count and RSS are sampled; membership can race and values are not hard real-time limits',retry_allowed=False)
