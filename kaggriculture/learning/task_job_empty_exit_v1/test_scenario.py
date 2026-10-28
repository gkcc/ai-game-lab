import unittest
from unittest.mock import patch
from . import scenario
from .test_sampling import FakeAPI, process


class ScenarioAPI(FakeAPI):
    def __init__(self,timeout=False):
        super().__init__(); self.exited=False; self.timeout=timeout
    def active(self,job): return 0 if self.exited or 'terminate' in self.events else 1
    def available(self): return 4*1024**3
    def wait(self,handle,timeout):
        self.events.append('wait')
        if timeout==scenario.INJECTION_WAIT_MS:
            if self.timeout: return False
            self.exited=True
        return True
    def pids(self,job): return [] if self.exited else [1]


class ScenarioTests(unittest.TestCase):
    def run_fake(self,api):
        from . import sampling
        original=sampling.supervised_job
        def controlled(wrapped,command,cwd,limits,**kwargs):
            return original(wrapped,command,cwd,limits,clock=lambda:0,sleep=lambda _:None)
        with patch.object(sampling,'supervised_job',side_effect=controlled):
            return scenario.run_owned(api,'python','unused',preflight=lambda:True,process_factory=process)

    def test_success_is_artificial_and_bounded(self):
        api=ScenarioAPI(); result=self.run_fake(api)
        self.assertTrue(result['passed']); self.assertEqual(result['artificial_exit_timing_arrangements'],1)
        self.assertEqual(result['supervision']['rss_complete_samples'],1)
        self.assertEqual(result['supervision']['confirmed_empty_exits'],1)
        self.assertIn('close_job',api.events)

    def test_timeout_fails_and_cleans_up(self):
        api=ScenarioAPI(timeout=True); result=self.run_fake(api)
        self.assertFalse(result['passed']); self.assertEqual(result['supervision']['error_kind'],'TimeoutError')
        self.assertIn('terminate',api.events); self.assertIn('close_job',api.events)

    def test_preflight_and_exceptions_fail_closed(self):
        api=ScenarioAPI()
        with self.assertRaises(RuntimeError): scenario.run_owned(api,'python','unused',preflight=lambda:False)
        self.assertEqual(api.events,[])
        api=ScenarioAPI(); api.pids=lambda _:(_ for _ in ()).throw(PermissionError())
        result=self.run_fake(api)
        self.assertFalse(result['passed']); self.assertEqual(result['supervision']['error_kind'],'PermissionError')
        self.assertIn('close_job',api.events)


if __name__=='__main__': unittest.main()
