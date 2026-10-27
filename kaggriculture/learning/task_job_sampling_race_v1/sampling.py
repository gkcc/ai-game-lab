"""Bounded retry for disappearing Job members; no work occurs at import."""
import psutil
from learning.task_training_job_supervision_v1.job import supervise as supervised_job

MAX_ATTEMPTS = 3


class SamplingExhausted(RuntimeError):
    """All attempts lost a member before a complete sample was obtained."""


class EmptySnapshot(RuntimeError):
    """The legacy lifecycle cannot safely classify a concurrent natural exit."""


class RetryingSampler:
    """Wrap owned-Job API; membership and partial RSS remain private and local."""
    def __init__(self,api,process_factory=psutil.Process):
        self.api,self.process_factory = api,process_factory
        self.attempts = self.retries = self.complete_samples = self.incomplete_samples = 0

    def __getattr__(self,name): return getattr(self.api,name)

    def sample_rss(self,job):
        for attempt in range(MAX_ATTEMPTS):
            self.attempts += 1
            try:
                members = self.api.pids(job)
                if isinstance(members,(list,tuple)) and not members:
                    raise EmptySnapshot('Empty Job membership during RSS sample')
                if (not isinstance(members,(list,tuple))
                    or any(type(pid) is not int or pid <= 0 for pid in members)
                    or len(members) != len(set(members))):
                    raise ValueError('Invalid membership snapshot')
                total = 0
                for pid in members:
                    rss = self.process_factory(pid).memory_info().rss
                    if type(rss) is not int or rss < 0: raise ValueError('Invalid RSS sample')
                    total += rss
            except psutil.NoSuchProcess as exc:
                self.incomplete_samples += 1
                if attempt + 1 == MAX_ATTEMPTS:
                    raise SamplingExhausted('Job RSS sample did not stabilize') from exc
                self.retries += 1
                continue
            except BaseException:
                self.incomplete_samples += 1
                raise
            self.complete_samples += 1
            return len(members),total
        raise AssertionError('Unreachable')


def supervise(api,command,cwd,limits,*,process_factory=psutil.Process,**kwargs):
    """Run the existing fail-closed owned-Job lifecycle with retry accounting."""
    wrapped = RetryingSampler(api,process_factory)
    result = supervised_job(wrapped,command,cwd,limits,**kwargs)
    result.update(rss_sampling_attempts=wrapped.attempts,rss_sampling_retries=wrapped.retries,
                  rss_complete_samples=wrapped.complete_samples,rss_discarded_samples=wrapped.incomplete_samples,
                  sampling_limit='Complete per-snapshot read only; membership can change or PID can be reused during sampling; no hard real-time memory cap')
    return result
