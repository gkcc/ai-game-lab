import types
import unittest
from unittest.mock import patch
import psutil
from . import sampling


class Source:
    def __init__(self,snapshots): self.snapshots=iter(snapshots); self.calls=0
    def pids(self,job): self.calls+=1; return next(self.snapshots)


class SamplingTest(unittest.TestCase):
    def test_discard_partial_then_fresh_success(self):
        source=Source([[1,2],[3,4]])
        def process(pid):
            if pid==2: raise psutil.NoSuchProcess(pid)
            return types.SimpleNamespace(memory_info=lambda:types.SimpleNamespace(rss={1:999,3:10,4:20}[pid]))
        sampler=sampling.RetryingSampler(source,process)
        self.assertEqual(sampler.sample_rss('owned'),(2,30))
        self.assertEqual((sampler.attempts,sampler.retries,sampler.incomplete_samples,sampler.complete_samples),(2,1,1,1))
        self.assertEqual(source.calls,2)

    def test_exhaustion_and_other_errors(self):
        source=Source([[1],[2],[3]])
        sampler=sampling.RetryingSampler(source,lambda pid:(_ for _ in ()).throw(psutil.NoSuchProcess(pid)))
        with self.assertRaises(sampling.SamplingExhausted): sampler.sample_rss('owned')
        self.assertEqual((sampler.attempts,sampler.retries,sampler.incomplete_samples,sampler.complete_samples),(3,2,3,0))
        sampler=sampling.RetryingSampler(Source([[1]]),lambda pid:(_ for _ in ()).throw(PermissionError()))
        with self.assertRaises(PermissionError): sampler.sample_rss('owned')
        self.assertEqual((sampler.attempts,sampler.retries,sampler.incomplete_samples),(1,0,1))
        with self.assertRaises(sampling.EmptySnapshot):
            sampling.RetryingSampler(Source([[]])).sample_rss('owned')
        for members in ([1,1],[0],[True]):
            with self.subTest(members=members),self.assertRaises(ValueError):
                sampling.RetryingSampler(Source([members])).sample_rss('owned')

    def test_real_lifecycle_with_fake_api(self):
        limits=dict(max_rss_bytes=1000,max_commit_bytes=1000,min_available_bytes=1,wall_seconds=50,poll_seconds=1,cleanup_seconds=3)
        class FakeJob:
            def __init__(self,snapshots,fail=None):
                self.source=Source(snapshots); self.events=[]; self.fail=fail; self.calls=0
            def record(self,name):
                self.events.append(name)
                if self.fail==name: raise OSError(name)
            def create_job(self): self.record('create_job'); return 'job'
            def configure(self,job): self.record('configure')
            def create_suspended(self,command,cwd): self.record('create_suspended'); return 'process','thread'
            def assign(self,job,process): self.record('assign')
            def resume(self,thread): self.record('resume')
            def active(self,job):
                self.record('active'); self.calls+=1
                return 0 if 'terminate_job' in self.events or self.calls>1 else 1
            def pids(self,job): return self.source.pids(job)
            def committed(self,job): return 1
            def available(self): return 100
            def terminate_job(self,job): self.record('terminate_job')
            def terminate_process(self,process): self.record('terminate_process')
            def wait(self,process,ms): self.record('wait'); return True
            def exit_code(self,process): return 0
            def close(self,handle): self.record('close_'+handle)
        def process(pid):
            if pid in (2,5,6,7): raise psutil.NoSuchProcess(pid)
            return types.SimpleNamespace(memory_info=lambda:types.SimpleNamespace(rss=10))
        api=FakeJob([[1,2],[3,4]])
        result=sampling.supervise(api,['fake'],'unused',limits,process_factory=process,clock=lambda:0,sleep=lambda _:None)
        self.assertTrue(result['passed']); self.assertEqual(result['rss_sampling_retries'],1)
        self.assertEqual(result['rss_complete_samples'],1); self.assertEqual(result['peak_sampled_rss_bytes'],20)
        for failure in (None,'terminate_job','wait'):
            api=FakeJob([[5],[6],[7]],fail=failure)
            result=sampling.supervise(api,['fake'],'unused',limits,process_factory=process,clock=lambda:0,sleep=lambda _:None)
            self.assertFalse(result['passed']); self.assertEqual(result['rss_complete_samples'],0)
            self.assertEqual(result['rss_sampling_retries'],2)
            self.assertIn('terminate_job',api.events); self.assertIn('wait',api.events)
            self.assertGreaterEqual(api.events.count('active'),2)
            self.assertIn('close_job',api.events); self.assertIn('close_process',api.events); self.assertIn('close_thread',api.events)
            if failure is None: self.assertTrue(result['launcher_reaped'] and result['job_members_zero'])
            else: self.assertIsNotNone(result['cleanup_error_kind'])
        api=FakeJob([[1]])
        result=sampling.supervise(api,['fake'],'unused',limits,process_factory=lambda _:(_ for _ in ()).throw(PermissionError()),clock=lambda:0,sleep=lambda _:None)
        self.assertEqual(result['error_kind'],'PermissionError'); self.assertEqual(result['rss_sampling_retries'],0)
        self.assertIn('terminate_job',api.events)
        api=FakeJob([[]])
        result=sampling.supervise(api,['fake'],'unused',limits,process_factory=process,clock=lambda:0,sleep=lambda _:None)
        self.assertEqual(result['error_kind'],'EmptySnapshot'); self.assertFalse(result['passed'])
        self.assertEqual(result['rss_complete_samples'],0)


if __name__ == '__main__': unittest.main()
