import io
from pathlib import Path
import unittest
from unittest.mock import patch
from . import experiment
from learning.task_training_job_supervision_v1 import job


class GateTests(unittest.TestCase):
    def test_low_memory_has_no_marker_or_process(self):
        with patch.object(experiment,'ready',return_value={}),patch.object(Path,'exists',return_value=False), \
             patch.object(experiment,'memory_ready',return_value=False),patch.object(Path,'open') as opened:
            with self.assertRaises(RuntimeError): experiment.run()
            opened.assert_not_called()

    def test_existing_marker_or_result_refuses_start(self):
        for existing in ('started.json','result.json','result.local.json'):
            with self.subTest(existing=existing),patch.object(experiment,'ready',return_value={}), \
                 patch.object(Path,'exists',autospec=True,side_effect=lambda p:p.name==existing),patch.object(experiment,'memory_ready') as memory:
                with self.assertRaises(ValueError): experiment.run()
                memory.assert_not_called()

    def test_failure_receipts_block_second_attempt(self):
        seen=set(); saved={}; launches=[]
        class Marker(io.StringIO):
            def __enter__(self): seen.add('started.json'); return self
        def write(path,value): seen.add(path.name); saved[path.name]=value
        def execute(*args,**kwargs): launches.append(1); raise RuntimeError('synthetic')
        with patch.object(experiment,'ready',return_value={}),patch.object(experiment,'memory_ready',return_value=True), \
             patch.object(Path,'exists',autospec=True,side_effect=lambda p:p.name in seen),patch.object(Path,'open',return_value=Marker()), \
             patch.object(experiment,'sha',return_value='hash'),patch.object(experiment,'read',return_value={'commit':'commit'}), \
             patch.object(experiment,'write_new',side_effect=write),patch.object(job,'WindowsAPI',return_value=object()), \
             patch.object(experiment.scenario,'run_owned',side_effect=execute):
            with self.assertRaises(RuntimeError): experiment.run()
            seen.remove('started.json')
            with self.assertRaises(ValueError): experiment.run()
        self.assertEqual(len(launches),1); self.assertFalse(saved['result.json']['passed'])
        self.assertEqual(saved['result.local.json']['error_kind'],'RuntimeError')
        self.assertEqual(saved['result.json']['identities']['started.json'],'hash')
        self.assertEqual(saved['result.json']['pre_execution_commit'],'commit')
        self.assertIsNone(saved['result.json']['observed']['launcher_reaped'])

    def test_environment_rejects_inherited_pythonoptimize(self):
        with patch.object(experiment.os,'name','nt'),patch.dict(experiment.os.environ,{'PYTHONOPTIMIZE':'1'}):
            with self.assertRaises(ValueError): experiment.environment()

    def test_verify_rejects_attempt_or_retry_drift(self):
        cfg=dict(baseline_head=experiment.BASE,synthetic_only=True,limits=experiment.scenario.LIMITS,
            host_preflight=experiment.PREFLIGHT,environment={},attempts=1,retry_allowed=False)
        freeze=dict(registration_sha256='hash',root_decision_sha256='hash')
        def read(path): return cfg if Path(path).name=='registration.json' else freeze
        with patch.object(experiment,'deadline_check'),patch.object(experiment,'clean_head'), \
             patch.object(experiment,'environment',return_value={}),patch.object(experiment,'read',side_effect=read), \
             patch.object(experiment,'sha',return_value='hash'),patch.object(Path,'exists',return_value=False):
            for change in ({'attempts':2},{'attempts':True},{'retry_allowed':True}):
                with self.subTest(change=change):
                    old=cfg.copy(); cfg.update(change)
                    with self.assertRaises(ValueError): experiment.verify()
                    cfg.clear(); cfg.update(old)

    def test_success_summary_has_real_observed_counts(self):
        saved={}
        supervision=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
            exit_code=0,error_kind=None,cleanup_error_kind=None,stop_reason=None,rss_sampling_attempts=2,
            rss_sampling_retries=0,rss_complete_samples=1,rss_discarded_samples=1,confirmed_empty_exits=1)
        result=dict(passed=True,supervision=supervision,artificial_exit_timing_arrangements=1)
        with patch.object(experiment,'ready',return_value={}),patch.object(experiment,'memory_ready',return_value=True), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open',return_value=io.StringIO()), \
             patch.object(experiment,'sha',return_value='hash'),patch.object(experiment,'read',return_value={'commit':'commit'}), \
             patch.object(experiment,'write_new',side_effect=lambda p,v:saved.update({p.name:v})), \
             patch.object(job,'WindowsAPI',return_value=object()),patch.object(experiment.scenario,'run_owned',return_value=result):
            summary=experiment.run()
        self.assertTrue(summary['passed']); self.assertEqual(summary['observed']['confirmed_empty_exits'],1)
        self.assertTrue(summary['observed']['launcher_reaped']); self.assertEqual(summary['identities']['freeze.json'],'hash')
        self.assertEqual(summary['pre_execution_commit'],'commit')


if __name__=='__main__': unittest.main()
