from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch, mock_open
from contextlib import ExitStack
from . import supervisor as s


class SupervisorTest(unittest.TestCase):
    def test_owned_process_outcomes(self):
        for case in ('success','rss','available','monitor_error','spawn_error','partial','receipt','kill_timeout'):
            with self.subTest(case=case), ExitStack() as stack:
                records = {}
                class Child:
                    pid = 99
                    returncode = None
                    def poll(self): return self.returncode
                    def wait(self,timeout=None):
                        if case == 'kill_timeout':
                            raise subprocess.TimeoutExpired('owned',timeout)
                        if self.returncode is None: self.returncode = 0
                        return self.returncode
                    def terminate(self):
                        if case != 'kill_timeout': self.returncode = -1
                    def kill(self): self.returncode = -9
                child = Child()
                if case == 'success': child.returncode = 0
                for name,value in (('ready',{'supervision':s.SUPERVISION}),('verify_commit',None),('sha','h')):
                    stack.enter_context(patch.object(s,name,return_value=value))
                stack.enter_context(patch.object(Path,'exists',return_value=False))
                stack.enter_context(patch.object(Path,'open',mock_open()))
                def write(path,value):
                    if case == 'receipt': raise PermissionError()
                    records[Path(path).name] = value
                stack.enter_context(patch.object(s,'write_new',side_effect=write))
                stack.enter_context(patch.object(s.subprocess,'Popen',return_value=child,side_effect=OSError() if case in ('spawn_error','receipt') else None))
                rss = s.SUPERVISION['max_rss_bytes']+1 if case in ('rss','kill_timeout') else 1
                process = types.SimpleNamespace(memory_info=lambda:types.SimpleNamespace(rss=rss))
                stack.enter_context(patch.object(s.psutil,'Process',return_value=process,side_effect=OSError() if case == 'monitor_error' else None))
                available = 0 if case == 'available' else s.SUPERVISION['min_available_bytes']
                stack.enter_context(patch.object(s.psutil,'virtual_memory',return_value=types.SimpleNamespace(available=available)))
                if case == 'partial': stack.enter_context(patch.object(s.json,'dump',side_effect=OSError()))
                if case == 'success': self.assertTrue(s.run()['passed'])
                else:
                    with self.assertRaises(s.SupervisorFailure) as caught: s.run()
                    if case == 'receipt': self.assertEqual(caught.exception.receipt_error_kind,'PermissionError')
                if case != 'receipt':
                    result = records['supervisor-result.json']
                    if case == 'rss': self.assertEqual(result['stop_reason'],'rss_limit')
                    if case == 'available': self.assertEqual(result['stop_reason'],'available_memory_limit')
                    if case == 'partial': self.assertFalse(result['marker_complete'])
                    if case == 'spawn_error': self.assertEqual(result['child_lifecycle'],'unknown_after_spawn_attempt')
                    if case == 'kill_timeout':
                        self.assertEqual(result['cleanup_error_kind'],'TimeoutExpired')
                        self.assertFalse(result['child_reaped'])
                        self.assertFalse(result['retry_allowed'])
                    if case in ('rss','available','monitor_error'): self.assertEqual(child.returncode,-1)


if __name__ == '__main__': unittest.main()
