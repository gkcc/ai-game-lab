"""Windows-only, single-worker process ownership for the capacity stage.

This is containment for trusted local workers using CreateProcess, not a sandbox
against code deliberately delegating work to WMI, services or another process.
Native identities and command lines are confined to optional LOCAL details.

Windows 10 / Server 2016 or newer is required. A JOB_LIST attribute assigns the
new process atomically, including if this controller dies during CreateProcess.
The primary thread remains suspended until membership/limits are checked. No
breakaway flags or fallback to an uncontained Popen are permitted.

Microsoft API contracts checked for this implementation:
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information
https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-terminatejobobject
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from contextlib import contextmanager
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


GIB = 1024 ** 3
TREE_MEMORY_LIMIT = 4 * GIB
MIN_AVAILABLE_MEMORY = 4 * GIB
CLOSE_SECONDS = 10.0
POLL_SECONDS = 0.05
FORCED_EXIT_CODE = 0xE0000001
_dispatch_lock = threading.Lock()
_dispatch_blocked = False


class TreeControlError(RuntimeError):
    """Containment or its evidence failed; do not dispatch another worker."""


class ResourceLimitError(TreeControlError):
    pass


class _BasicLimits(C.Structure):
    _fields_ = [("PerProcessUserTimeLimit", C.c_longlong),
                ("PerJobUserTimeLimit", C.c_longlong), ("LimitFlags", W.DWORD),
                ("MinimumWorkingSetSize", C.c_size_t), ("MaximumWorkingSetSize", C.c_size_t),
                ("ActiveProcessLimit", W.DWORD), ("Affinity", C.c_size_t),
                ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]


class _IoCounters(C.Structure):
    _fields_ = [(name, C.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimits(C.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", C.c_size_t), ("JobMemoryLimit", C.c_size_t),
                ("PeakProcessMemoryUsed", C.c_size_t), ("PeakJobMemoryUsed", C.c_size_t)]


class _Accounting(C.Structure):
    _fields_ = [(name, C.c_longlong) for name in (
        "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")]
    _fields_ += [(name, W.DWORD) for name in (
        "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]


class _MemoryStatus(C.Structure):
    _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD)]
    _fields_ += [(name, C.c_ulonglong) for name in (
        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


class _ProcessMemory(C.Structure):
    _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)]
    _fields_ += [(name, C.c_size_t) for name in (
        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
        "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class _StartupInfo(C.Structure):
    _fields_ = [("cb", W.DWORD), ("lpReserved", W.LPWSTR), ("lpDesktop", W.LPWSTR),
                ("lpTitle", W.LPWSTR)]
    _fields_ += [(name, W.DWORD) for name in (
        "dwX", "dwY", "dwXSize", "dwYSize", "dwXCountChars", "dwYCountChars",
        "dwFillAttribute", "dwFlags")]
    _fields_ += [("wShowWindow", W.WORD), ("cbReserved2", W.WORD),
                ("lpReserved2", C.POINTER(W.BYTE)), ("hStdInput", W.HANDLE),
                ("hStdOutput", W.HANDLE), ("hStdError", W.HANDLE)]


class _StartupInfoEx(C.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", C.c_void_p)]


class _ProcessInfo(C.Structure):
    _fields_ = [("hProcess", W.HANDLE), ("hThread", W.HANDLE),
                ("dwProcessId", W.DWORD), ("dwThreadId", W.DWORD)]


class _Win32:
    def __init__(self):
        if os.name != "nt" or C.sizeof(C.c_void_p) != 8:
            raise TreeControlError("64-bit Windows is required; no uncontained fallback")
        self.k = C.WinDLL("kernel32", use_last_error=True)
        prototypes = {
            "CreateJobObjectW": ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            "SetInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            "QueryInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.POINTER(W.DWORD)], W.BOOL),
            "TerminateJobObject": ([W.HANDLE, W.UINT], W.BOOL),
            "IsProcessInJob": ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
            "SetHandleInformation": ([W.HANDLE, W.DWORD, W.DWORD], W.BOOL),
            "GetHandleInformation": ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
            "CloseHandle": ([W.HANDLE], W.BOOL),
            "GetCurrentProcess": ([], W.HANDLE),
            "DuplicateHandle": ([W.HANDLE, W.HANDLE, W.HANDLE, C.POINTER(W.HANDLE), W.DWORD, W.BOOL, W.DWORD], W.BOOL),
            "InitializeProcThreadAttributeList": ([C.c_void_p, W.DWORD, W.DWORD, C.POINTER(C.c_size_t)], W.BOOL),
            "UpdateProcThreadAttribute": ([C.c_void_p, W.DWORD, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p], W.BOOL),
            "DeleteProcThreadAttributeList": ([C.c_void_p], None),
            "CreateProcessW": ([W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p, W.BOOL,
                W.DWORD, C.c_void_p, W.LPCWSTR, C.POINTER(_StartupInfoEx), C.POINTER(_ProcessInfo)], W.BOOL),
            "ResumeThread": ([W.HANDLE], W.DWORD),
            "WaitForSingleObject": ([W.HANDLE, W.DWORD], W.DWORD),
            "GetExitCodeProcess": ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
            "GetProcessTimes": ([W.HANDLE, C.POINTER(W.FILETIME), C.POINTER(W.FILETIME),
                C.POINTER(W.FILETIME), C.POINTER(W.FILETIME)], W.BOOL),
            "OpenProcess": ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            "TerminateProcess": ([W.HANDLE, W.UINT], W.BOOL),
            "GlobalMemoryStatusEx": ([C.POINTER(_MemoryStatus)], W.BOOL),
            "K32GetProcessMemoryInfo": ([W.HANDLE, C.POINTER(_ProcessMemory), W.DWORD], W.BOOL),
        }
        for name, (args, result) in prototypes.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result

    def check(self, result, operation):
        if not result:
            error = C.get_last_error()
            raise TreeControlError(f"{operation}: {C.WinError(error)}")
        return result

    def close(self, handle):
        if handle:
            self.check(self.k.CloseHandle(handle), "CloseHandle")

    def signaled(self, handle):
        result = self.k.WaitForSingleObject(handle, 0)
        if result == 0xFFFFFFFF:
            self.check(False, "WaitForSingleObject")
        if result not in (0, 258):
            raise TreeControlError(f"Unexpected process wait result: {result}")
        return result == 0

    def exit_code(self, handle):
        code = W.DWORD()
        self.check(self.k.GetExitCodeProcess(handle, C.byref(code)), "GetExitCodeProcess")
        return int(code.value)

    def identity(self, handle, pid):
        times = [W.FILETIME() for _ in range(4)]
        self.check(self.k.GetProcessTimes(handle, *(C.byref(item) for item in times)), "GetProcessTimes")
        return {"pid": int(pid), "creation_filetime": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}

    def in_job(self, handle, job):
        contained = W.BOOL()
        self.check(self.k.IsProcessInJob(handle, job, C.byref(contained)), "IsProcessInJob")
        return bool(contained.value)

    def query(self, job, kind, structure):
        value = structure()
        self.check(self.k.QueryInformationJobObject(job, kind, C.byref(value), C.sizeof(value), None),
                   "QueryInformationJobObject")
        return value


def available_memory_bytes():
    """Read-only physical memory availability; never change host configuration."""
    api = _Win32()
    value = _MemoryStatus()
    value.dwLength = C.sizeof(value)
    api.check(api.k.GlobalMemoryStatusEx(C.byref(value)), "GlobalMemoryStatusEx")
    return int(value.ullAvailPhys)


def current_identity():
    api = _Win32()
    return api.identity(api.k.GetCurrentProcess(), os.getpid())


@contextmanager
def _defer_console_interrupts():
    """Do not let a second Ctrl+C interrupt bounded ownership cleanup."""
    restored = []
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                restored.append((sig, signal.getsignal(sig)))
                signal.signal(sig, signal.SIG_IGN)
    try:
        yield
    finally:
        for sig, handler in restored:
            signal.signal(sig, handler)


class OwnedTree:
    """One owned job; exposed only so the real regression can observe/kill its controller.

    Always call close() in finally, including when start() raises. ``close``
    terminates the owned job only, waits at most ten seconds, then closes every
    retained process/thread/job handle. A failed proof latches dispatch closed.
    """
    def __init__(self, lifecycle, *, memory_limit_bytes=TREE_MEMORY_LIMIT):
        if not isinstance(memory_limit_bytes, int) or not 0 < memory_limit_bytes <= TREE_MEMORY_LIMIT:
            raise ValueError("Tree commit limit must be in (0, 4 GiB]")
        self.api = _Win32()
        self.lifecycle = lifecycle
        self.memory_limit = memory_limit_bytes
        self.job = self.process = self.thread = None
        self.processes = {}  # (PID, creation time) -> retained handle + receipt
        self.started = time.monotonic()
        self.closed = False
        lifecycle.update(workers_closed=False, owned_child_closed=False,
            job_handle_closed=False, all_process_handles_closed=False,
            active_processes_final=None, creation_mode="atomic_job_list_then_resume",
            kill_on_job_close=True, breakaway_allowed=False, job_handle_inherited=False,
            memory_limit_bytes=memory_limit_bytes, minimum_available_memory_bytes=MIN_AVAILABLE_MEMORY,
            peak_job_commit_bytes=0, sampled_peak_tree_working_set_bytes=0,
            sampled_peak_tree_private_commit_bytes=0, memory_sample_interval_seconds=POLL_SECONDS,
            working_set_is_sampled_lower_bound=True, processes=[], termination_requested=False)

    def _remember(self, handle, pid):
        identity = self.api.identity(handle, pid)
        key = (identity["pid"], identity["creation_filetime"])
        if key in self.processes:
            return False
        receipt = dict(identity, exit_observed=False, exit_code=None)
        self.processes[key] = (handle, receipt)
        self.lifecycle["processes"].append(receipt)
        return True

    def start(self, command, *, cwd, env, stream):
        if self.job or self.closed:
            raise TreeControlError("OwnedTree is single use")
        if isinstance(command, (str, bytes)) or not command:
            raise ValueError("An argv sequence is required")
        argv = [os.fsdecode(arg) for arg in command]
        executable = Path(argv[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("Use an existing absolute executable path")
        if any("\0" in arg for arg in argv):
            raise ValueError("NUL in command")
        available = available_memory_bytes()
        self.lifecycle["host_available_memory_before_bytes"] = available
        if available < MIN_AVAILABLE_MEMORY:
            raise ResourceLimitError("Host available physical memory is below 4 GiB")
        environment = dict(os.environ if env is None else env)
        environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for key, value in environment.items():
            if not isinstance(key, str) or not isinstance(value, str) or not key or "=" in key or "\0" in key + value:
                raise ValueError("Invalid environment entry")
        env_block = C.create_unicode_buffer("\0".join(f"{k}={v}" for k, v in
            sorted(environment.items(), key=lambda item: item[0].upper())) + "\0\0")
        command_line = C.create_unicode_buffer(subprocess.list2cmdline(argv))
        self.lifecycle.update(command=argv, cwd=str(Path(cwd).resolve()) if cwd is not None else None,
                              controller_already_in_job=self.api.in_job(self.api.k.GetCurrentProcess(), None))
        self.job = self.api.check(self.api.k.CreateJobObjectW(None, None), "CreateJobObjectW")
        self.api.check(self.api.k.SetHandleInformation(self.job, 1, 0), "SetHandleInformation(job)")
        flags = W.DWORD()
        self.api.check(self.api.k.GetHandleInformation(self.job, C.byref(flags)), "GetHandleInformation(job)")
        if flags.value & 1:
            raise TreeControlError("Job handle unexpectedly inheritable")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        limits.JobMemoryLimit = self.memory_limit
        self.api.check(self.api.k.SetInformationJobObject(self.job, 9, C.byref(limits), C.sizeof(limits)),
                       "SetInformationJobObject(limits)")
        confirmed = self.api.query(self.job, 9, _ExtendedLimits)
        if confirmed.BasicLimitInformation.LimitFlags != limits.BasicLimitInformation.LimitFlags or confirmed.JobMemoryLimit != self.memory_limit:
            raise TreeControlError("Job containment limits differ from requested limits")
        import msvcrt
        duplicates = []
        attribute_list = None
        attributes_initialized = False
        null_in = null_out = None
        info = _ProcessInfo()
        try:
            null_in = open(os.devnull, "rb")
            if stream is None:
                null_out = open(os.devnull, "wb")
                stream = null_out
            stream.flush()
            current = self.api.k.GetCurrentProcess()
            for source in (null_in, stream):
                target = W.HANDLE()
                self.api.check(self.api.k.DuplicateHandle(current, msvcrt.get_osfhandle(source.fileno()),
                    current, C.byref(target), 0, True, 2), "DuplicateHandle(stdio)")
                duplicates.append(target.value)
            handles = (W.HANDLE * 2)(*duplicates)
            jobs = (W.HANDLE * 1)(self.job)
            size = C.c_size_t()
            self.api.k.InitializeProcThreadAttributeList(None, 2, 0, C.byref(size))
            if not size.value:
                self.api.check(False, "InitializeProcThreadAttributeList(size)")
            attribute_buffer = C.create_string_buffer(size.value)
            attribute_list = C.cast(attribute_buffer, C.c_void_p)
            self.api.check(self.api.k.InitializeProcThreadAttributeList(attribute_list, 2, 0, C.byref(size)),
                           "InitializeProcThreadAttributeList")
            attributes_initialized = True
            for attribute, values in ((0x20002, handles), (0x2000D, jobs)):
                self.api.check(self.api.k.UpdateProcThreadAttribute(attribute_list, 0, attribute,
                    C.cast(values, C.c_void_p), C.sizeof(values), None, None), "UpdateProcThreadAttribute")
            startup = _StartupInfoEx()
            startup.StartupInfo.cb = C.sizeof(startup)
            startup.StartupInfo.dwFlags = 0x100 | 1  # STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW
            startup.StartupInfo.wShowWindow = 0
            startup.StartupInfo.hStdInput = duplicates[0]
            startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = duplicates[1]
            startup.lpAttributeList = attribute_list
            creation = 0x4 | 0x400 | 0x80000 | 0x08000000
            self.api.check(self.api.k.CreateProcessW(str(executable), command_line, None, None, True,
                creation, C.cast(env_block, C.c_void_p), os.fspath(cwd) if cwd is not None else None,
                C.byref(startup), C.byref(info)), "CreateProcessW(atomic job assignment)")
            self.process, self.thread = info.hProcess, info.hThread
            self.lifecycle["launcher_identity"] = self.api.identity(self.process, info.dwProcessId)
            self._remember(self.process, info.dwProcessId)
            if not self.api.in_job(self.process, self.job):
                raise TreeControlError("Suspended worker is not in the owned job")
            self.lifecycle["membership_verified_before_resume"] = True
            self.sample()
            previous = self.api.k.ResumeThread(self.thread)
            if previous != 1:
                raise TreeControlError(f"ResumeThread expected suspended count 1, got {previous}")
            self.lifecycle["work_resumed"] = True
        finally:
            # PROCESS_INFORMATION is filled by the OS before Python regains control.
            # Adopt handles even if a Python signal interrupted the assignment above.
            self.process = self.process or info.hProcess
            self.thread = self.thread or info.hThread
            if self.thread:
                self.api.close(self.thread)
                self.thread = None
                self.lifecycle["primary_thread_handle_closed"] = True
            if attributes_initialized:
                self.api.k.DeleteProcThreadAttributeList(attribute_list)
            for handle in duplicates:
                self.api.close(handle)
            if null_in:
                null_in.close()
            if null_out:
                null_out.close()
        return self

    def _job_pids(self):
        capacity = 32
        while capacity <= 65536:
            buffer = C.create_string_buffer(8 + C.sizeof(C.c_size_t) * capacity)
            if self.api.k.QueryInformationJobObject(self.job, 3, buffer, C.sizeof(buffer), None):
                count = W.DWORD.from_buffer(buffer, 4).value
                return list((C.c_size_t * count).from_buffer(buffer, 8))
            if C.get_last_error() != 234:  # ERROR_MORE_DATA
                self.api.check(False, "QueryInformationJobObject(process IDs)")
            capacity = max(capacity * 2, W.DWORD.from_buffer(buffer, 0).value)
        raise TreeControlError("Owned job process enumeration exceeds bounded capacity")

    def sample(self):
        if not self.job:
            raise TreeControlError("No job to sample")
        accounting = self.api.query(self.job, 1, _Accounting)
        extended = self.api.query(self.job, 9, _ExtendedLimits)
        self.lifecycle["active_processes_last"] = int(accounting.ActiveProcesses)
        self.lifecycle["total_processes"] = int(accounting.TotalProcesses)
        self.lifecycle["peak_job_commit_bytes"] = max(self.lifecycle["peak_job_commit_bytes"], int(extended.PeakJobMemoryUsed))
        known_live = {pid for (pid, _), (handle, _) in self.processes.items() if not self.api.signaled(handle)}
        for pid in self._job_pids():
            if pid in known_live:
                continue
            handle = self.api.k.OpenProcess(0x100000 | 0x1000 | 0x10, False, pid)
            if not handle:
                if C.get_last_error() == 87:  # Process exited between enumeration and OpenProcess.
                    continue
                self.api.check(False, "OpenProcess(owned job member)")
            retained = False
            try:
                # This check is against the opened object, so a reused PID is never adopted.
                if self.api.in_job(handle, self.job):
                    retained = self._remember(handle, pid)
            finally:
                if not retained:
                    self.api.close(handle)
        working_set = private_commit = 0
        for handle, receipt in self.processes.values():
            if self.api.signaled(handle):
                receipt.update(exit_observed=True, exit_code=self.api.exit_code(handle))
                continue
            memory = _ProcessMemory()
            memory.cb = C.sizeof(memory)
            if not self.api.k.K32GetProcessMemoryInfo(handle, C.byref(memory), C.sizeof(memory)):
                if self.api.signaled(handle):
                    continue
                self.api.check(False, "K32GetProcessMemoryInfo")
            working_set += int(memory.WorkingSetSize)
            private_commit += int(memory.PrivateUsage)
        for key, value in (("sampled_peak_tree_working_set_bytes", working_set),
                           ("sampled_peak_tree_private_commit_bytes", private_commit)):
            self.lifecycle[key] = max(self.lifecycle[key], value)
        if extended.PeakJobMemoryUsed > self.memory_limit:
            raise ResourceLimitError("Owned job commit exceeded its registered limit")
        return int(accounting.ActiveProcesses)

    def poll(self):
        if not self.process:
            raise TreeControlError("Worker has not been created")
        return self.api.exit_code(self.process) if self.api.signaled(self.process) else None

    def terminate_identity(self, identity, *, code=FORCED_EXIT_CODE):
        """Regression-only targeted interruption of a retained owned identity."""
        key = (identity["pid"], identity["creation_filetime"])
        if key not in self.processes:
            raise TreeControlError("Target identity was not observed in this owned tree")
        saved_handle, _ = self.processes[key]
        if self.api.signaled(saved_handle):
            raise TreeControlError("Target exited before the controlled interruption")
        handle = self.api.k.OpenProcess(0x1 | 0x1000, False, identity["pid"])
        if not handle:
            self.api.check(False, "OpenProcess(terminate owned identity)")
        try:
            if self.api.identity(handle, identity["pid"]) != identity or not self.api.in_job(handle, self.job):
                raise TreeControlError("Target identity changed")
            self.api.check(self.api.k.TerminateProcess(handle, code), "TerminateProcess(owned identity)")
        finally:
            self.api.close(handle)

    def close(self):
        global _dispatch_blocked
        if self.closed:
            return
        cleanup_start = time.monotonic()
        errors = []
        with _defer_console_interrupts():
            try:
                if self.job:
                    # Do not rely on closing the last handle for routine cleanup: retain
                    # the query handle until ActiveProcesses == 0 has actually been read.
                    before = self.api.query(self.job, 1, _Accounting).ActiveProcesses
                    self.lifecycle["active_processes_before_close"] = int(before)
                    if before:
                        self.lifecycle["termination_requested"] = True
                        self.api.check(self.api.k.TerminateJobObject(self.job, FORCED_EXIT_CODE), "TerminateJobObject")
                    deadline = cleanup_start + CLOSE_SECONDS
                    while True:
                        active = self.sample()
                        self.lifecycle["active_processes_final"] = active
                        root_closed = self.process is None or self.api.signaled(self.process)
                        observed_closed = all(self.api.signaled(handle) for handle, _ in self.processes.values())
                        if active == 0 and root_closed and observed_closed:
                            self.lifecycle.update(workers_closed=True, owned_child_closed=True,
                                closure_proof="job_active_processes_zero_and_retained_identity_handles_signaled")
                            if self.process:
                                self.lifecycle["direct_returncode"] = self.api.exit_code(self.process)
                            break
                        if time.monotonic() >= deadline:
                            raise TreeControlError("Tree closure proof timed out after 10 seconds")
                        time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
                elif self.process is None:
                    self.lifecycle.update(workers_closed=True, owned_child_closed=True,
                                          closure_proof="no_process_created")
                else:
                    raise TreeControlError("Process exists without its owned job")
            except BaseException as error:
                errors.append(error)
                self.lifecycle["workers_closed"] = False
            finally:
                handles = {handle for handle, _ in self.processes.values()}
                if self.process:
                    handles.add(self.process)
                if self.thread:
                    handles.add(self.thread)
                for handle in handles:
                    try:
                        self.api.close(handle)
                    except BaseException as error:
                        errors.append(error)
                self.lifecycle["all_process_handles_closed"] = not errors
                self.process = self.thread = None
                # A failed query/termination still falls back to kill-on-last-close,
                # but must NEVER be reported as proven closure.
                try:
                    self.api.close(self.job)
                    self.lifecycle["job_handle_closed"] = True
                except BaseException as error:
                    errors.append(error)
                self.job = None
                self.closed = True
                self.lifecycle.update(elapsed_seconds=time.monotonic() - self.started,
                                      close_seconds=time.monotonic() - cleanup_start)
                if errors:
                    _dispatch_blocked = True
                    self.lifecycle["workers_closed"] = False
                    self.lifecycle["close_errors"] = [f"{type(e).__name__}: {e}" for e in errors]
        if errors:
            raise TreeControlError("Owned tree cleanup/proof failed; further dispatch blocked") from errors[0]


def public_lifecycle(details):
    """Stable aggregate contract. Native identities/commands stay in local details."""
    return dict(
        tree_closed=bool(details.get("workers_closed")),
        active_processes_after_close=details.get("active_processes_final"),
        elapsed_seconds=details.get("elapsed_seconds"),
        peak_job_memory_bytes=details.get("peak_job_commit_bytes", 0),
        peak_job_memory_kind="job_commit_peak_from_Windows_PeakJobMemoryUsed",
        direct_returncode=details.get("direct_returncode", details.get("returncode")),
        close_seconds=details.get("close_seconds"),
        sampled_peak_tree_working_set_bytes=details.get("sampled_peak_tree_working_set_bytes", 0),
        sampled_peak_tree_private_commit_bytes=details.get("sampled_peak_tree_private_commit_bytes", 0),
        working_set_is_sampled_lower_bound=True,
        memory_limit_bytes=details.get("memory_limit_bytes", TREE_MEMORY_LIMIT),
        memory_sample_interval_seconds=POLL_SECONDS,
        total_processes=details.get("total_processes", 0),
        containment_established=bool(details.get("membership_verified_before_resume")),
        creation_mode="atomic_job_list_then_resume",
        termination_requested=bool(details.get("termination_requested")),
        job_handle_closed=bool(details.get("job_handle_closed")),
        all_process_handles_closed=bool(details.get("all_process_handles_closed")),
        stop_reason=details.get("stop_reason"),
        exception_type=details.get("exception_type"),
        close_error_types=[text.split(":", 1)[0] for text in details.get("close_errors", [])],
    )


def run_owned(command, *, cwd, env, stream, timeout, lifecycle,
              memory_limit_bytes=TREE_MEMORY_LIMIT, local_lifecycle=None):
    """Run once, return the root exit code, and always close the complete job.

    Timeout raises subprocess.TimeoutExpired; Python exceptions/KeyboardInterrupt
    are re-raised after bounded tree cleanup. lifecycle contains public aggregate
    fields only and is updated on every path. Optional local_lifecycle receives
    private identity evidence for the fixture; never publish that dictionary.
    Only successful ActiveProcesses/handle proof sets tree_closed=true.
    """
    details = {} if local_lifecycle is None else local_lifecycle
    lifecycle.update(public_lifecycle(details))
    tree = None
    locked = False
    started = time.monotonic()
    try:
        if not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("A finite positive timeout is required")
        if _dispatch_blocked:
            raise TreeControlError("A previous ownership failure blocks further dispatch")
        locked = _dispatch_lock.acquire(blocking=False)
        if not locked:
            raise TreeControlError("Only one worker may be dispatched at a time")
        tree = OwnedTree(details, memory_limit_bytes=memory_limit_bytes)
        tree.start(command, cwd=cwd, env=env, stream=stream)
        deadline = started + timeout
        while True:
            tree.sample()
            code = tree.poll()
            if code is not None:
                details.update(returncode=code, stop_reason="root_exit")
                return code
            if time.monotonic() >= deadline:
                details["stop_reason"] = "timeout"
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    except BaseException as error:
        details.setdefault("stop_reason", "controller_exception")
        details["exception_type"] = type(error).__name__
        raise
    finally:
        try:
            if tree is not None:
                tree.close()
        except BaseException as error:
            details["exception_type"] = type(error).__name__
            details["stop_reason"] = "cleanup_failure"
            raise
        finally:
            details["elapsed_seconds"] = time.monotonic() - started
            lifecycle.update(public_lifecycle(details))
            if locked:
                _dispatch_lock.release()
