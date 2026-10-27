import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

exp=types.ModuleType('learning.task_teacher_choice_bc_v1.experiment')
exp.OUT=exp.ROOT=Path(__file__).resolve().parent/'synthetic'
for name in ('ready','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *args:None)
spec=importlib.util.spec_from_file_location('learning.task_teacher_choice_bc_v1._data_worker_test',Path(__file__).with_name('data_worker.py'))
worker=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_choice_bc_v1.experiment':exp}): spec.loader.exec_module(worker)


class SampleTest(unittest.TestCase):
    def test_thresholds_and_aggregate(self):
        arrays=dict(shared=np.zeros((256,275),np.float32),seed=np.array([1]*128+[2]*128),
                    offsets=np.arange(257),teacher_choice_label=np.zeros(256,np.int64),ops=np.arange(256)%3)
        limits=dict(min_groups=256,min_groups_per_seed=32,min_selected_operations=3)
        summary=worker.sample_summary(arrays,[1,2],limits)
        self.assertEqual(summary['groups'],256)
        self.assertEqual(summary['groups_by_seed'],{'1':128,'2':128})
        self.assertEqual(len(summary['selected_operations']),3)
        for change in (dict(min_groups=257),dict(min_groups_per_seed=129),dict(min_selected_operations=4)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.sample_summary(arrays,[1,2],{**limits,**change})
        with self.assertRaises(ValueError): worker.sample_summary(arrays,[1,3],limits)
        with self.assertRaises(ValueError): worker.sample_summary(arrays,[1,2],{**limits,'min_groups':255})


if __name__ == '__main__': unittest.main()
