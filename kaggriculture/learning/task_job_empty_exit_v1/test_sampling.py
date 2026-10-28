import types
import unittest
import psutil
from .sampling import supervise

LIMITS=dict(max_rss_bytes=100,max_commit_bytes=100,min_available_bytes=2,wall_seconds=10,poll_seconds=1,cleanup_seconds=2)


class FakeAPI:
    def __init__(self,mode='success'):
        self.mode=mode; self.samples=0; self.active_calls=0; self.events=[]
    def create_job(self): return 'job'
    def configure(self,job): pass
    def create_suspended(self,command,cwd): return 'process','thread'
    def assign(self,job,process): pass
    def resume(self,thread): pass
    def active(self,job):
        self.active_calls+=1
        if 'terminate' in self.events: return 0
        if self.mode=='immediate': return 0
        if self.mode=='reappears' and self.active_calls>=4: return 1
        if self.mode=='active': return 1
        return 1 if self.active_calls<=2 else 0
    def pids(self,job):
        self.samples+=1
        return [] if self.samples>1 or self.mode=='no_sample' else [1]
    def committed(self,job): return 1
    def available(self): return 1 if self.mode=='resource' else 10
    def wait(self,process,timeout):
        self.events.append('wait')
        if self.mode=='wait_error': raise OSError('synthetic')
        return self.mode!='not_reaped'
    def exit_code(self,process):
        if self.mode=='exit_error': raise OSError('synthetic')
        return 1 if self.mode=='nonzero' else 0
    def terminate_job(self,job): self.events.append('terminate')
    def terminate_process(self,process): self.events.append('terminate')
    def close(self,handle): self.events.append('close_'+handle)


def process(_): return types.SimpleNamespace(memory_info=lambda:types.SimpleNamespace(rss=10))


class SamplingTests(unittest.TestCase):
    def run_fake(self,mode):
        api=FakeAPI(mode)
        result=supervise(api,['fake'],'unused',LIMITS,process_factory=process,clock=lambda:0,sleep=lambda _:None)
        return api,result

    def test_verified_empty_exit_uses_only_previous_complete_sample(self):
        api,result=self.run_fake('success')
        self.assertTrue(result['passed']); self.assertEqual(result['rss_complete_samples'],1)
        self.assertEqual(result['rss_discarded_samples'],1); self.assertEqual(result['confirmed_empty_exits'],1)
        self.assertEqual(result['peak_sampled_rss_bytes'],10)
        self.assertTrue(result['launcher_reaped'] and result['job_members_zero'])
        self.assertIn('close_job',api.events)

    def test_unconfirmed_exits_fail_and_cleanup(self):
        for mode in ('active','not_reaped','nonzero','wait_error','exit_error','no_sample'):
            with self.subTest(mode=mode):
                api,result=self.run_fake(mode)
                self.assertFalse(result['passed']); self.assertEqual(result['confirmed_empty_exits'],0)
                self.assertIn('terminate',api.events); self.assertIn('close_job',api.events)

    def test_resource_limit_keeps_original_failure_semantics(self):
        api,result=self.run_fake('resource')
        self.assertFalse(result['passed']); self.assertEqual(result['stop_reason'],'available_memory_limit')
        self.assertEqual(result['confirmed_empty_exits'],0); self.assertIn('terminate',api.events)

    def test_immediate_exit_without_sample_is_rejected_after_cleanup(self):
        api,result=self.run_fake('immediate')
        self.assertFalse(result['passed']); self.assertEqual(result['acceptance_error_kind'],'NoCompleteSample')
        self.assertIsNone(result['error_kind']); self.assertIsNone(result['cleanup_error_kind'])
        self.assertEqual(result['rss_complete_samples'],0); self.assertIn('close_job',api.events)

    def test_members_reappear_after_exit_confirmation(self):
        api,result=self.run_fake('reappears')
        self.assertFalse(result['passed']); self.assertEqual(result['confirmed_empty_exits'],1)
        self.assertIn('terminate',api.events); self.assertIn('close_job',api.events)

    def test_disappeared_member_then_empty_is_counted_as_discarded(self):
        api=FakeAPI()
        snapshots=iter([[1],[2],[]])
        api.pids=lambda _:next(snapshots)
        def factory(pid):
            if pid==2: raise psutil.NoSuchProcess(pid)
            return process(pid)
        result=supervise(api,['fake'],'unused',LIMITS,process_factory=factory,clock=lambda:0,sleep=lambda _:None)
        self.assertTrue(result['passed']); self.assertEqual(result['rss_complete_samples'],1)
        self.assertEqual(result['rss_discarded_samples'],2); self.assertEqual(result['rss_sampling_retries'],1)
        self.assertEqual(result['confirmed_empty_exits'],1)


if __name__=='__main__': unittest.main()
