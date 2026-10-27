import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from learning.task_teacher_choice_bc_v1 import numeric
from . import analysis

exp=types.ModuleType('learning.task_teacher_merge_real_v1.experiment')
exp.ROOT=Path(__file__).resolve().parents[2]; exp.OUT=exp.ROOT/'synthetic'
for name in ('read','sha','ready','write_new','environment'): setattr(exp,name,lambda *args:None)
def load(name):
    spec=importlib.util.spec_from_file_location('learning.task_teacher_merge_real_v1._test_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'learning.task_teacher_merge_real_v1.experiment':exp}): spec.loader.exec_module(module)
    return module
probe=load('probe'); worker=load('worker')


class EntryTest(unittest.TestCase):
    def test_probe_seven_fields(self):
        saved={}
        with patch.object(probe,'ready',return_value={'environment':'env'}),patch.object(probe,'read',return_value={}), \
             patch.object(probe,'sha',side_effect=lambda p:Path(p).name),patch.object(probe,'environment',return_value='env'), \
             patch.object(probe,'verify_worker'),patch.object(probe,'write_new',side_effect=lambda p,v:saved.update({Path(p).name:v})):
            probe.run()
        self.assertEqual(set(saved['probe-result.json']),{'passed','venv_verified','job_membership_verified','training_inputs_read','freeze_sha256','marker_sha256','retry_allowed'})

    def test_worker_reads_single_snapshot_and_checks_roles(self):
        cfg=dict(environment='env',seed_roles_sha256='roles',schema_sha256='schema',cache_sha256='cache',
            schema_path='schemas/a.json',cache_path='private/train.npz',max_groups=10,max_candidates=20,
            max_archive_bytes=100,max_uncompressed_bytes=200,old_train_seeds=[1],synthetic_seed=2,
            merge_max_groups=12,merge_max_candidates=23,old_groups=3,old_candidate_rows=4)
        marker=dict(seed_roles_sha256='roles',schema_sha256='schema',cache_sha256='cache')
        def read(path): return marker if Path(path).name=='worker-started.json' else dict(holdout_seeds=[3])
        def sha(path): return {'freeze.json':'freeze','seed-roles.local.json':'roles','a.json':'schema'}[Path(path).name]
        saved={}
        with patch.object(worker,'ready',return_value=cfg),patch.object(worker,'read',side_effect=read), \
             patch.object(worker,'sha',side_effect=sha),patch.object(worker,'environment',return_value='env'), \
             patch.object(worker,'verify_worker'),patch.object(numeric,'read',return_value='arrays') as load_cache, \
             patch.object(analysis,'check',return_value=dict(passed=True,old_groups=3,old_candidate_rows=4)), \
             patch.object(worker,'write_new',side_effect=lambda p,v:saved.update({Path(p).name:v})):
            worker.run()
        self.assertEqual(load_cache.call_count,1)
        self.assertEqual(load_cache.call_args.args[-1],'cache')
        self.assertEqual(saved['worker-result.json']['archive_snapshot_reads'],1)

    def test_changed_marker_blocks_cache(self):
        cfg=dict(environment='env',seed_roles_sha256='roles',schema_sha256='schema',cache_sha256='cache')
        with patch.object(worker,'ready',return_value=cfg),patch.object(worker,'read',return_value={}), \
             patch.object(worker,'sha',return_value='freeze'),patch.object(worker,'verify_worker'), \
             patch.object(worker,'write_new'),patch.object(numeric,'read') as load_cache:
            with self.assertRaises(ValueError): worker.run()
            load_cache.assert_not_called()


if __name__=='__main__': unittest.main()
