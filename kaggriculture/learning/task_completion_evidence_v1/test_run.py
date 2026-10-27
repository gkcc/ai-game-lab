"""Focused preflight and owned-worker failure regressions; no OS child or data."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from . import run as entry


class WorkerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'main_official_transitions':5752,'analysis_limits':{'work_seconds':60,
            'cleanup_seconds':20,'log_bytes':1024,'sampled_memory_bytes':100}}
        self.out=Path('synthetic-never-created')

    def test_commit_mismatch_prevents_worker_launch(self):
        with patch.object(entry,'OUT',self.out),patch.object(entry,'ready',return_value=self.cfg), \
                patch.object(entry,'verify_commit',side_effect=ValueError('commit mismatch')), \
                patch.object(entry,'OwnedCapture') as capture:
            with self.assertRaisesRegex(ValueError,'commit mismatch'):entry.run()
            capture.assert_not_called()

    def test_memory_limit_failure_still_reaps_owned_worker(self):
        process=SimpleNamespace(poll=lambda:None)
        child=SimpleNamespace(process=process,start=Mock(),check=Mock())
        cleanup={'owned_instances_remaining':0,'reader_joined':True}
        receipts={}
        def write(path,value):receipts[Path(path).name]=value
        with patch.object(entry,'OUT',self.out),patch.object(entry,'ready',return_value=self.cfg), \
                patch.object(entry,'verify_commit'),patch.object(entry,'deadline_check'), \
                patch.object(entry,'remaining_seconds',return_value=3600), \
                patch.object(entry,'sha',return_value='f'*64),patch.object(entry,'write_new',side_effect=write), \
                patch.object(entry.psutil,'Process',return_value=SimpleNamespace(cpu_affinity=lambda *args:[0])), \
                patch.object(entry,'OwnedCapture',return_value=child),patch.object(entry,'observe_processes'), \
                patch.object(entry,'sampled_memory',return_value=(101,1)), \
                patch.object(entry,'reap',return_value=cleanup) as reaper:
            with self.assertRaises(RuntimeError):entry.run()
        self.assertTrue(reaper.called)
        self.assertEqual(receipts['run-result.json']['primary_error_kind'],'MemoryError')
        self.assertEqual(receipts['run-result.json']['cleanup']['owned_instances_remaining'],0)
        self.assertTrue(receipts['run-result.json']['cleanup']['reader_joined'])
        self.assertFalse(receipts['run-result.json']['passed'])

    def test_unsampled_fast_worker_cannot_pass_resource_contract(self):
        process=SimpleNamespace(poll=lambda:0,returncode=0)
        child=SimpleNamespace(process=process,start=Mock(),check=Mock(),finish=lambda:{'reader_joined':True})
        cleanup={'owned_instances_remaining':0,'reader_joined':True}
        receipts={}
        def read(path):
            return {'complete':True,'official_transitions':5752} if Path(path).name=='analysis-result.json' else {'passed':True,'output':{'refused':0}}
        with patch.object(entry,'OUT',self.out),patch.object(entry,'ready',return_value=self.cfg), \
                patch.object(entry,'verify_commit'),patch.object(entry,'deadline_check'), \
                patch.object(entry,'remaining_seconds',return_value=3600), \
                patch.object(entry,'sha',return_value='f'*64), \
                patch.object(entry,'write_new',side_effect=lambda path,value:receipts.update({Path(path).name:value})), \
                patch.object(entry,'read',side_effect=read), \
                patch.object(entry.psutil,'Process',return_value=SimpleNamespace(cpu_affinity=lambda *args:[0])), \
                patch.object(entry,'OwnedCapture',return_value=child),patch.object(entry,'observe_processes'), \
                patch.object(entry,'reap',return_value=cleanup):
            with self.assertRaises(RuntimeError):entry.run()
        self.assertEqual(receipts['run-result.json']['memory_samples'],0)
        self.assertFalse(receipts['run-result.json']['passed'])


if __name__=='__main__':unittest.main()
