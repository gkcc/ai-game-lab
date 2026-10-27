"""Preallocated process handle and bounded capture, including partial startup."""
from pathlib import Path
import subprocess
import threading


class _NotStarted:
    ident=None
    def is_alive(self):return False
    def join(self,timeout=None):pass


class OutputLimitExceeded(RuntimeError):pass


class OwnedCapture:
    def __init__(self,*,log,limit_bytes):
        self.process=None;self.thread=_NotStarted();self.error=None;self.io_error=None
        self.seen=self.written=0;self.limit_bytes=limit_bytes
        self._stream=Path(log).open('xb',buffering=0)

    def start(self,command,*,cwd):
        # The caller already owns self before either process or thread creation.
        self.process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            bufsize=0,creationflags=subprocess.DETACHED_PROCESS)
        thread=threading.Thread(target=self._drain,daemon=False)
        self.thread=thread
        thread.start()

    def _drain(self):
        try:
            while block:=self.process.stdout.read(65536):
                self.seen+=len(block)
                if self.error is not None:continue
                remaining=self.limit_bytes-self.written
                if len(block)>remaining:
                    self.error=OutputLimitExceeded('Output storage cap exceeded');block=block[:remaining]
                if block:
                    written=self._stream.write(block);self.written+=written
                    if written!=len(block):raise OSError('Short capture write')
        except BaseException as exc:self.io_error=exc;self.error=exc
        finally:
            self._stream.close();self.process.stdout.close()

    def check(self):
        if self.error is not None:raise self.error

    def finish(self):
        self.process.wait(timeout=5);self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('Reader did not join')
        self.check()
        return dict(observed_bytes=self.seen,stored_bytes=self.written,reader_joined=True,exit_code=self.process.returncode)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:self.process.kill()
            self.process.wait(timeout=5)
        if self.thread.ident is not None:self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('Reader did not join during cleanup')
        if not self._stream.closed:self._stream.close()
        if self.process is not None and not self.process.stdout.closed:self.process.stdout.close()
