"""Owned-Job sampling adapter for a verified natural exit after an empty snapshot."""
import psutil
from learning.task_job_sampling_race_v1.sampling import RetryingSampler, EmptySnapshot
from learning.task_training_job_supervision_v1.job import supervise as supervised_job


class ExitAwareSampler(RetryingSampler):
    def __init__(self, api, process_factory=psutil.Process):
        super().__init__(api,process_factory)
        self.process=None
        self.last_complete=None
        self.confirmed_empty_exits=0
        self._verified_exit=False

    def active(self,job):
        count=self.api.active(job)
        if self._verified_exit and (type(count) is not int or count!=0):
            raise RuntimeError('Job membership changed after exit confirmation')
        return count

    def create_suspended(self, command, cwd):
        process,thread=self.api.create_suspended(command,cwd)
        self.process=process
        return process,thread

    def sample_rss(self, job):
        try:
            sample=super().sample_rss(job)
        except EmptySnapshot:
            if self.last_complete is None or self.process is None: raise
            count=self.api.active(job)
            if type(count) is not int or count!=0 or self.api.wait(self.process,0) is not True: raise
            code=self.api.exit_code(self.process)
            if type(code) is not int or code!=0:
                raise
            self.confirmed_empty_exits+=1
            self._verified_exit=True
            # Legacy supervisor needs a tuple to reach its own next active() check.
            # This is the previous complete measurement, not a new RSS sample.
            return self.last_complete
        self.last_complete=sample
        return sample


def supervise(api,command,cwd,limits,*,process_factory=psutil.Process,**kwargs):
    wrapped=ExitAwareSampler(api,process_factory)
    result=supervised_job(wrapped,command,cwd,limits,**kwargs)
    result['acceptance_error_kind'] = None
    if result.get('passed') is True and wrapped.complete_samples == 0:
        result['passed'] = False
        result['acceptance_error_kind'] = 'NoCompleteSample'
    result.update(rss_sampling_attempts=wrapped.attempts,rss_sampling_retries=wrapped.retries,
        rss_complete_samples=wrapped.complete_samples,rss_discarded_samples=wrapped.incomplete_samples,
        confirmed_empty_exits=wrapped.confirmed_empty_exits,
        sampling_limit='Empty snapshot is not a sample; previous complete value is passed to legacy loop after verified zero members and zero exit; sampling is not a hard memory cap')
    return result
