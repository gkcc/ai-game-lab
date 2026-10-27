"""Private named-Job membership check to prevent accidental direct workers.

This is not a security boundary against a malicious process running as the same
user, which can inspect environment variables or alter local files.
"""
import ctypes
from ctypes import wintypes
import hashlib
import os
from learning.task_training_job_supervision_v1.job import WindowsAPI

ENV_KEY='TASK_TEACHER_DEVIATION_JOB'


def token_hash(name): return hashlib.sha256(name.encode('ascii')).hexdigest()


class NamedWindowsAPI(WindowsAPI):
    def __init__(self,name):
        super().__init__(); self.name=name
    def create_job(self):
        ctypes.set_last_error(0)
        handle=self._require(self.k.CreateJobObjectW(None,self.name))
        if ctypes.get_last_error()==183:
            self.close(handle)
            raise RuntimeError('Named Job collision')
        return handle


class QueryAPI:
    def __init__(self):
        if os.name!='nt': raise OSError('Windows required')
        self.k=ctypes.WinDLL('kernel32',use_last_error=True)
        self.k.OpenJobObjectW.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.LPCWSTR]
        self.k.OpenJobObjectW.restype=wintypes.HANDLE
        self.k.GetCurrentProcess.argtypes=[]; self.k.GetCurrentProcess.restype=wintypes.HANDLE
        self.k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
        self.k.IsProcessInJob.restype=wintypes.BOOL
        self.k.CloseHandle.argtypes=[wintypes.HANDLE]; self.k.CloseHandle.restype=wintypes.BOOL
    def open(self,name):
        handle=self.k.OpenJobObjectW(4,False,name)
        if not handle: raise ctypes.WinError(ctypes.get_last_error())
        return handle
    def contains_current(self,handle):
        result=wintypes.BOOL()
        if not self.k.IsProcessInJob(self.k.GetCurrentProcess(),handle,ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(result.value)
    def close(self,handle):
        if not self.k.CloseHandle(handle): raise ctypes.WinError(ctypes.get_last_error())


def verify_worker(name,marker,freeze_sha,query_factory=QueryAPI):
    if (not isinstance(name,str) or not name.startswith('Local\\TeacherDeviation-')
        or len(name)!=len('Local\\TeacherDeviation-')+32
        or any(c not in '0123456789abcdef' for c in name[-32:])
        or marker.get('attempt')!=1 or marker.get('retry_allowed') is not False
        or marker.get('freeze_sha256')!=freeze_sha or marker.get('job_token_sha256')!=token_hash(name)):
        raise ValueError('Worker marker or Job identity invalid')
    query=query_factory(); handle=query.open(name)
    try:
        if not query.contains_current(handle): raise ValueError('Worker not in registered Job')
    finally:
        query.close(handle)
