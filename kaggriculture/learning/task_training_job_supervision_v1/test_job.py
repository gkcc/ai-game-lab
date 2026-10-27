import unittest
from .job import supervise, WindowsAPI
import types
from .entry import run, ReceiptFailure

LIMITS = dict(max_rss_bytes=100,max_commit_bytes=200,min_available_bytes=50,wall_seconds=10,poll_seconds=1,cleanup_seconds=2)


class Fake:
    def __init__(self, fail=None, rss=1, commit=1, available=100, stuck=False):
        self.fail,self.rss_value,self.commit,self.avail,self.stuck = fail,rss,commit,available,stuck
        self.events=[]; self.active_count=1
    def step(self,name):
        self.events.append(name)
        if self.fail == name: raise OSError(name)
    def create_job(self): self.step('create_job'); return 'job'
    def configure(self,j): self.step('configure')
    def create_suspended(self,c,cwd): self.step('create_suspended'); return 'process','thread'
    def assign(self,j,p): self.step('assign')
    def resume(self,t): self.step('resume')
    def active(self,j):
        self.step('active')
        if self.stuck: return 1
        if 'terminate_job' in self.events: return 0
        return self.active_count
    def sample_rss(self,j): self.step('sample_rss'); return 2,self.rss_value
    def committed(self,j): self.step('committed'); return self.commit
    def available(self): return self.avail
    def terminate_job(self,j): self.step('terminate_job')
    def terminate_process(self,p): self.step('terminate_process')
    def wait(self,p,ms): self.step('wait'); return not self.stuck
    def exit_code(self,p): return 0
    def close(self,h): self.events.append('close_'+h)


class JobTest(unittest.TestCase):
    def test_suspend_assign_before_resume_and_fail_closed(self):
        for failure in ('assign','resume'):
            api = Fake(fail=failure)
            result = supervise(api,['python'],'unused',LIMITS,clock=lambda:0,sleep=lambda _:None)
            self.assertFalse(result['passed'])
            self.assertLess(api.events.index('create_suspended'),api.events.index('assign'))
            if failure == 'assign':
                self.assertNotIn('resume',api.events)
                self.assertIn('terminate_process',api.events)
            else: self.assertIn('terminate_job',api.events)
            self.assertIn('close_job',api.events)

    def test_limits_and_unconfirmed_cleanup(self):
        for kwargs,reason in ((dict(rss=101),'rss_limit'),(dict(commit=201),'commit_limit'),(dict(available=49),'available_memory_limit')):
            api = Fake(**kwargs)
            result = supervise(api,['python'],'unused',LIMITS,clock=lambda:0,sleep=lambda _:None)
            self.assertEqual(result['stop_reason'],reason)
            if reason == 'rss_limit': self.assertEqual(result['rss_threshold_sample_members'],2)
            self.assertTrue(result['job_members_zero'])
            self.assertTrue(result['launcher_reaped'])
        ticks = iter(range(100))
        result = supervise(Fake(rss=101,stuck=True),['python'],'unused',LIMITS,clock=lambda:next(ticks),sleep=lambda _:None)
        self.assertFalse(result['job_members_zero'])
        self.assertFalse(result['launcher_reaped'])
        self.assertFalse(result['retry_allowed'])

    def test_success_and_partial_receipt(self):
        api = Fake(); api.active_count=0
        result = supervise(api,['python'],'unused',LIMITS,clock=lambda:0,sleep=lambda _:None)
        self.assertTrue(result['passed'])
        receipts=[]
        def marker(state): state['created']=True; raise OSError('partial')
        with self.assertRaises(RuntimeError): run(api,['python'],'unused',LIMITS,marker,receipts.append)
        self.assertFalse(receipts[0]['marker_complete'])
        with self.assertRaises(ReceiptFailure):
            run(api,['python'],'unused',LIMITS,marker,lambda _:(_ for _ in ()).throw(PermissionError()))

    def test_exit_after_wall_limit_is_not_success(self):
        api = Fake(); api.active_count = 0
        ticks = iter([0,11,11,11,11,11])
        result = supervise(api,['python'],'unused',LIMITS,clock=lambda:next(ticks),sleep=lambda _:None)
        self.assertFalse(result['passed'])
        self.assertEqual(result['stop_reason'],'wall_limit')

    def test_termination_failure_still_waits_and_checks_members(self):
        api = Fake(fail='terminate_job',rss=101,stuck=True)
        ticks = iter(range(100))
        result = supervise(api,['python'],'unused',LIMITS,clock=lambda:next(ticks),sleep=lambda _:None)
        self.assertEqual(result['termination_error_kind'],'OSError')
        self.assertIn('wait',api.events)
        self.assertGreaterEqual(api.events.count('active'),2)
        self.assertFalse(result['launcher_reaped'])
        self.assertFalse(result['job_members_zero'])
        self.assertIn('close_job',api.events)

    def test_resume_requires_single_suspend_count(self):
        api = WindowsAPI.__new__(WindowsAPI)
        for value in (0,2):
            api.k = types.SimpleNamespace(ResumeThread=lambda _:value)
            with self.assertRaises(RuntimeError): api.resume('thread')
        api.k = types.SimpleNamespace(ResumeThread=lambda _:1)
        api.resume('thread')


if __name__ == '__main__': unittest.main()
