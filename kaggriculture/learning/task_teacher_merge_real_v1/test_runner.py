import unittest
from unittest.mock import patch
from . import runner


class RunnerTest(unittest.TestCase):
    def test_preflight_refuses_any_low_sample(self):
        cfg=dict(preflight_samples=3,preflight_min_available_bytes=4,preflight_interval_seconds=10)
        with patch.object(runner,'deadline_check'):
            self.assertFalse(runner.memory_ready(cfg,available=iter([5,3,5]).__next__,sleep=lambda _:None))
            self.assertTrue(runner.memory_ready(cfg,available=iter([5,4,6]).__next__,sleep=lambda _:None))

    def test_stopped_probe_does_not_read_receipt_or_launch_worker(self):
        job=dict(passed=False,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                 exit_code=1,error_kind=None,cleanup_error_kind=None,stop_reason='available_memory_limit')
        saved=[]
        def mark(out,name,value,state): state.update(created=True,complete=True)
        with patch.object(runner,'ready',return_value={'probe_job_supervision':{}}), \
            patch.object(runner,'verify_commit'),patch.object(runner,'memory_ready',return_value=True), \
            patch.object(runner.Path,'exists',return_value=False),patch.object(runner,'sha',return_value='a'*64), \
            patch.object(runner,'marker',side_effect=mark), \
            patch.object(runner,'write_new',side_effect=lambda p,v:saved.append((p.name,v))), \
            patch.object(runner,'read',side_effect=AssertionError('stopped probe receipt must not be read')), \
            patch.object(runner,'launch',return_value=job) as launch:
            with self.assertRaisesRegex(RuntimeError,'Probe rejected'): runner.run()
        self.assertEqual(launch.call_count,1)
        result=dict(saved)['result.json']
        self.assertEqual(result['probe']['stop_reason'],'available_memory_limit')
        self.assertFalse(result['worker_launch_entered'])
        self.assertEqual(result['archive_reads'],'not_started_by_runner')

    def test_worker_receipt_forbids_unexpected_data_fields(self):
        cfg=dict(seed_roles_sha256='a',cache_sha256='b',schema_sha256='c',old_groups=2,old_candidate_rows=3)
        aggregate=dict(passed=True,old_groups=2,old_candidate_rows=3,synthetic_groups=2,total_groups=4,
            alignment_passed=True,old_input_shape_valid=True,synthetic_input_shapes=[[1,373],[2,373]],
            source_identity='Caller must verify source SHA and common trusted schema before invocation',
            interpretation='Synthetic interface check only; no GPU training or game outcome evaluated')
        worker=dict(passed=True,aggregate=aggregate,freeze_sha256='freeze',seed_roles_sha256='a',cache_sha256='b',
            schema_sha256='c',archive_snapshot_reads=1,no_training=True,retry_allowed=False)
        self.assertEqual(runner.validate_worker(worker,cfg,'freeze'),aggregate)
        for bad in ({**worker,'archive_snapshot_reads':True},
                    {**worker,'aggregate':{**aggregate,'features':[1,2]}},
                    {**worker,'aggregate':{**aggregate,'interpretation':'unexpected'}},
                    {**worker,'cache_sha256':'different'}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                runner.validate_worker(bad,cfg,'freeze')

    def test_changed_worker_marker_blocks_success(self):
        good=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                  exit_code=0,error_kind=None,cleanup_error_kind=None,stop_reason=None)
        probe=dict(passed=True,venv_verified=True,job_membership_verified=True,training_inputs_read=False,
                   freeze_sha256='a'*64,marker_sha256='a'*64,retry_allowed=False)
        phase={'worker_finished':False}; saved=[]
        def launch(name,module,limits):
            if module.endswith('.worker'): phase['worker_finished']=True
            return good
        def digest(path):
            return ('b' if path.name=='worker-started.json' and phase['worker_finished'] else 'a')*64
        def mark(out,name,value,state): state.update(created=True,complete=True)
        def read(path):
            if path.name=='probe-result.json': return probe
            raise AssertionError('Worker result must not be consumed after marker drift')
        cfg=dict(probe_job_supervision={},worker_job_supervision={},seed_roles_sha256='c'*64,
                 schema_sha256='d'*64,cache_sha256='e'*64)
        with patch.object(runner,'ready',return_value=cfg),patch.object(runner,'verify_commit'), \
            patch.object(runner,'memory_ready',return_value=True),patch.object(runner,'deadline_check'), \
            patch.object(runner.Path,'exists',return_value=False),patch.object(runner,'sha',side_effect=digest), \
            patch.object(runner,'marker',side_effect=mark),patch.object(runner,'read',side_effect=read), \
            patch.object(runner,'write_new',side_effect=lambda p,v:saved.append((p.name,v))), \
            patch.object(runner,'launch',side_effect=launch):
            with self.assertRaisesRegex(ValueError,'Execution identity changed'): runner.run()
        result=dict(saved)['result.json']
        self.assertFalse(result['passed'])
        self.assertTrue(result['worker_launch_entered'])
        self.assertEqual(result['archive_reads'],'unknown')
        self.assertFalse(result['retry_allowed'])


if __name__=='__main__': unittest.main()
