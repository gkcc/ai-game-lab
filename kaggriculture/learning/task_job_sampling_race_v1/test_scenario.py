import sys
import types
import unittest
from unittest.mock import patch
import psutil
from . import scenario


class ScenarioTest(unittest.TestCase):
    def test_injection_exactly_once(self):
        calls=[]
        wrapped=scenario.InjectOnce(lambda pid:calls.append(pid) or 'process')
        with self.assertRaises(psutil.NoSuchProcess): wrapped(12)
        self.assertEqual(wrapped(13),'process'); self.assertEqual(wrapped(14),'process')
        self.assertEqual(wrapped.injections,1); self.assertEqual(calls,[13,14])

    def test_fake_launcher_waits_and_hides_every_child(self):
        fake=types.ModuleType('subprocess'); fake.CREATE_NO_WINDOW=8; calls=[]; waits=[]
        def popen(args,**kwargs):
            calls.append(kwargs)
            index=len(calls)
            return types.SimpleNamespace(wait=lambda:waits.append(index) or 0)
        fake.Popen=popen
        with patch.dict(sys.modules,{'subprocess':fake}),self.assertRaises(SystemExit) as exited:
            exec(scenario.command('synthetic')[-1],{})
        self.assertEqual(exited.exception.code,0)
        self.assertEqual(len(calls),17); self.assertEqual(len(waits),17)
        self.assertTrue(all(x==dict(creationflags=8) for x in calls))

    def test_acceptance_and_single_supervise(self):
        result=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
            exit_code=0,error_kind=None,cleanup_error_kind=None,rss_complete_samples=1,rss_sampling_retries=1,rss_discarded_samples=1)
        self.assertTrue(scenario.accepted(result,1))
        for change in (dict(job_members_zero=False),dict(rss_sampling_retries=0),dict(exit_code=1),dict(rss_complete_samples=0)):
            self.assertFalse(scenario.accepted({**result,**change},1))
        def fake_supervise(api,command,cwd,limits,process_factory):
            with self.assertRaises(psutil.NoSuchProcess): process_factory(1)
            process_factory(2)
            return result
        with patch.object(scenario,'supervise',side_effect=fake_supervise) as call:
            outcome=scenario.run_owned(None,'synthetic','unused',process_factory=lambda _:None)
        self.assertTrue(outcome['passed']); self.assertEqual(call.call_count,1)


if __name__ == '__main__': unittest.main()
