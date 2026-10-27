import unittest
from pathlib import Path
from unittest.mock import patch,mock_open
from . import runner


class RunnerTest(unittest.TestCase):
    def test_worker_thread_environment_restored_on_success_and_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail),patch.dict(runner.os.environ,{'OPENBLAS_NUM_THREADS':'8','OMP_NUM_THREADS':'4'},clear=True), \
                 patch.object(runner,'NamedWindowsAPI',return_value=None):
                def supervise(*args):
                    self.assertEqual({k:runner.os.environ.get(k) for k in runner.THREAD_LIMITS},runner.THREAD_LIMITS)
                    if fail: raise OSError('synthetic')
                    return {'passed':True}
                with patch.object(runner,'supervise',side_effect=supervise):
                    if fail:
                        with self.assertRaises(OSError): runner.launch_named('synthetic','learning.task_teacher_deviation_v1.worker',{})
                    else: runner.launch_named('synthetic','learning.task_teacher_deviation_v1.worker',{})
                self.assertEqual(dict(runner.os.environ),{'OPENBLAS_NUM_THREADS':'8','OMP_NUM_THREADS':'4'})

    def test_probe_identity_and_cleanup_required(self):
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,error_kind=None,cleanup_error_kind=None,exit_code=0)
        probe=dict(passed=True,venv_verified=True,job_membership_verified=True,training_inputs_read=False,freeze_sha256='freeze',marker_sha256='marker')
        identities=dict(freeze_sha256='freeze',probe_marker_sha256='marker')
        with patch.object(runner,'sha',return_value='marker'):
            runner.validate_probe(Path('unused'),identities,job,probe)
            for bad in ({**probe,'marker_sha256':'wrong'}, {**probe,'job_membership_verified':False}):
                with self.assertRaises(ValueError): runner.validate_probe(Path('unused'),identities,job,bad)
            with self.assertRaises(ValueError): runner.validate_probe(Path('unused'),identities,{**job,'job_members_zero':False},probe)

    def test_probe_failure_blocks_worker_and_success_binds_identity(self):
        for good in (False,True):
            with self.subTest(good=good):
                records={}
                cfg=dict(job_limits={},probe_job_limits={},cache_sha256='cache',checkpoint_sha256='new',old_checkpoint_sha256='old')
                job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                         error_kind=None,cleanup_error_kind=None,exit_code=0)
                def digest(path): return Path(path).name+'-hash'
                def read(path):
                    return {'pre-execution-commit.json':dict(commit='commit'),
                        'probe-result.json':dict(passed=good,venv_verified=True,job_membership_verified=True,
                            training_inputs_read=False,freeze_sha256='freeze.json-hash',marker_sha256='probe-started.json-hash'),
                        'worker-result.json':dict(passed=True,freeze_sha256='freeze.json-hash',cache_sha256='cache',
                            checkpoint_sha256='new',old_checkpoint_sha256='old',
                            aggregate=dict(groups=200,cpu_forwards=400,candidate_rows_total=1000))}[Path(path).name]
                with patch.object(runner,'ready',return_value=cfg),patch.object(runner,'verify_commit'),patch.object(runner,'deadline_check'), \
                     patch.object(Path,'exists',return_value=False),patch.object(Path,'open',mock_open()), \
                     patch.object(runner,'sha',side_effect=digest),patch.object(runner,'read',side_effect=read), \
                     patch.object(runner,'launch_named',return_value=job) as launch, \
                     patch.object(runner,'progress_snapshot',return_value={}), \
                     patch.object(runner,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
                    if good:
                        self.assertTrue(runner.run()['passed'])
                        self.assertEqual(launch.call_count,2)
                        self.assertEqual(records['probe-supervision-result.json']['commit'],'commit')
                    else:
                        with self.assertRaises(RuntimeError): runner.run()
                        self.assertEqual(launch.call_count,1)
                        self.assertNotIn('probe-supervision-result.json',records)

    def test_failure_receipt_blocks_retry(self):
        records={}
        cfg=dict(job_limits={},probe_job_limits={},cache_sha256='cache',checkpoint_sha256='model')
        def dump(*args,**kwargs): records['started.json']=True
        def digest(path): return Path(path).name+'-hash'
        with patch.object(runner,'ready',return_value=cfg),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',autospec=True,side_effect=lambda p:Path(p).name in records), \
             patch.object(Path,'open',mock_open()),patch.object(runner.json,'dump',side_effect=dump), \
             patch.object(runner,'sha',side_effect=digest),patch.object(runner,'read',return_value={'commit':'c'}), \
             patch.object(runner,'NamedWindowsAPI',return_value=None),patch.object(runner,'supervise',return_value={'passed':False}) as supervise, \
             patch.object(runner,'progress_snapshot',return_value={'forwards_started':'unknown','forwards_completed':'unknown'}), \
             patch.object(runner,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
            with self.assertRaises(RuntimeError): runner.run()
            records.pop('started.json')
            self.assertFalse(records['result.json']['passed'])
            with self.assertRaises(ValueError): runner.run()
            self.assertEqual(supervise.call_count,1)


if __name__=='__main__': unittest.main()
