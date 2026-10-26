"""Bounded stdout/stderr capture for an owned child, including native output."""
import os
from pathlib import Path
import subprocess
import sys
import threading


class CapturedProcess:
    def __init__(self, command, *, cwd, log, limit_bytes, env=None):
        if type(limit_bytes) is not int or limit_bytes<=0:raise ValueError('Positive log byte limit required')
        self.limit_bytes,self.written,self.seen=limit_bytes,0,0
        self.error=None
        path=Path(log);path.parent.mkdir(parents=True,exist_ok=True)
        stream=path.open('xb',buffering=0)
        opts=dict(creationflags=subprocess.CREATE_NO_WINDOW) if sys.platform=='win32' else dict(start_new_session=True)
        try:
            self.process=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,bufsize=0,**opts)
        except BaseException:
            stream.close();raise
        self.thread=threading.Thread(target=self._drain,args=(stream,),daemon=False)
        try:
            self.thread.start()
        except BaseException:
            self.process.kill();self.process.wait(timeout=10)
            self.process.stdout.close();stream.close()
            raise

    def _drain(self,stream):
        try:
            while block:=self.process.stdout.read(65536):
                self.seen+=len(block)
                if self.error is not None:continue
                available=self.limit_bytes-self.written
                if len(block)>available:
                    self.error=RuntimeError('Owned process output exceeded its registered log cap')
                    block=block[:available]
                if block:
                    try:
                        written=stream.write(block)
                        if type(written) is not int or not 0<=written<=len(block):
                            raise OSError('Invalid owned output write result')
                        self.written+=written
                        if written!=len(block):raise OSError('Short owned output write')
                    except BaseException as exc:
                        self.error=exc
                # After a refusal keep draining without storing, so the owner
                # can cancel a Linux supervisor even if its child was verbose.
        except BaseException as exc:
            self.error=self.error or exc
        finally:
            stream.close();self.process.stdout.close()

    def check(self):
        if self.error is not None:raise self.error

    def finish(self):
        self.process.wait(timeout=10)
        self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('Owned output reader is still alive')
        self.check()
        return dict(limit_bytes=self.limit_bytes,written_bytes=self.written,
                    observed_bytes=self.seen,reader_joined=True,child_exit_code=self.process.returncode)

    def stop(self):
        if self.process.poll() is None:self.process.kill()
        self.process.wait(timeout=10)
        self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('Owned output reader did not finish during cancellation')

