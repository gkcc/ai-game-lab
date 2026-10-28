import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from learning.task_choice_v1.features import OPS

exp=types.ModuleType('learning.task_teacher_refresh_v2.experiment')
exp.OUT=Path(__file__).resolve().parent/'synthetic'
for name in ('ready','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *a:None)
runner=types.ModuleType('learning.task_teacher_refresh_v2.independent_runner')
runner.validate_completed=lambda *a:None
spec=importlib.util.spec_from_file_location('learning.task_teacher_refresh_v2._data_worker_test',Path(__file__).with_name('data_worker.py'))
w=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_refresh_v2.experiment':exp,'learning.task_teacher_refresh_v2.independent_runner':runner}): spec.loader.exec_module(w)


def arrays():
    ops=[OPS.index('DIG')]*32+[OPS.index('FERTILIZE')]*100+[OPS.index('WATER')]*124
    seed=np.array([1,2]*128,np.uint32)
    return dict(shared=np.zeros((256,275),np.float32),candidates=np.zeros((256,98),np.float32),
        offsets=np.arange(257,dtype=np.int64),teacher_choice_label=np.zeros(256,np.int64),
        ops=np.array(ops,np.uint8),seed=seed)


class DataTests(unittest.TestCase):
    def run_fake(self,mode):
        data=arrays(); saved={}; writes=[]
        if mode=='coverage': data['ops'][:32]=OPS.index('WATER')
        if mode=='seed': data['seed'][0]=3
        entries=[]
        for seed in (1,2):
            for opponent in ('round2','market'):
                selected={op:0 for op in OPS}
                if opponent=='round2':
                    for i in range(len(data['seed'])):
                        if int(data['seed'][i])==seed: selected[OPS[int(data['ops'][i])]]+=1
                entries.append(dict(seed=seed,opponent=opponent,retained=sum(selected.values()),
                                    chosen=selected.copy(),retained_chosen=selected.copy()))
        stats=dict(by_seed_opponent=entries,retained_groups=256,total_callbacks=300)
        if mode=='inconsistent': stats['by_seed_opponent'][0]['chosen']['DIG']+=1
        limits=dict(min_groups=256,min_groups_per_seed=32,min_selected_operations=3,min_dig=32,min_fertilize=100,
            min_dig_per_seed=1,max_groups=1000,max_candidates=1000,max_archive_bytes=256*1024**2,max_uncompressed_bytes=256*1024**2)
        if mode=='budget': limits['max_archive_bytes']=1
        cfg=dict(train_seeds=[1,2],reserved_test_seeds=[3,4,5,6],schema_sha256='hash',data_limits=limits,collection_limits={})
        marker=dict(source_manifest_sha256='hash',schema_sha256='hash',independent_replay_sha256='hash',independent_runner_result_sha256='hash')
        def read(path): return marker if Path(path).name=='data-worker-started.json' else [] if Path(path).name=='source-manifest.json' else {}
        loaded={k:v.copy() for k,v in data.items()}
        if mode=='mismatch': loaded['shared'][0,0]=1
        def write(path,value): saved[Path(path).name]=value
        with patch.object(w,'ready',return_value=cfg),patch.object(w,'read',side_effect=read),patch.object(w,'sha',return_value='hash'), \
             patch.object(w,'verify_worker'),patch.object(w,'check_independent'),patch.object(w,'deadline_check'), \
             patch.object(w,'collect',return_value=(data,stats)),patch.object(w,'write_new',side_effect=write), \
             patch.object(w.numeric,'write_new',side_effect=lambda *a:writes.append('archive')), \
             patch.object(w.numeric,'read',return_value=loaded),patch.object(Path,'mkdir'), \
             patch.object(Path,'stat',return_value=types.SimpleNamespace(st_size=1024)):
            if mode in ('seed','budget','mismatch','inconsistent'):
                with self.assertRaises(ValueError): w.run(exp.OUT)
            else: w.run(exp.OUT)
        return saved,writes

    def test_insufficient_coverage_never_writes_archive(self):
        saved,writes=self.run_fake('coverage')
        self.assertFalse(writes); self.assertEqual(saved['data-worker-result.json']['status'],'coverage_insufficient')
        self.assertFalse(saved['data-worker-result.json']['job_cleanup_verified_here'])

    def test_seed_and_budget_rejected_before_archive(self):
        for mode in ('seed','budget','inconsistent'):
            saved,writes=self.run_fake(mode); self.assertFalse(writes); self.assertIn('data-worker-failure.json',saved)

    def test_roundtrip_mismatch_and_success(self):
        saved,writes=self.run_fake('mismatch'); self.assertEqual(writes,['archive']); self.assertIn('data-worker-failure.json',saved)
        saved,writes=self.run_fake('success'); self.assertEqual(writes,['archive'])
        self.assertTrue(saved['data-worker-result.json']['passed']); self.assertFalse(saved['data-worker-result.json']['job_cleanup_verified_here'])
        self.assertEqual(saved['data-worker-result.json']['coverage']['dig_by_seed'],{'1':16,'2':16})
        self.assertEqual(saved['data-worker-result.json']['coverage']['retained_selected_operations']['DIG'],32)
        self.assertIn('sample_summary',saved['data-worker-result.json'])


if __name__=='__main__': unittest.main()
