import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch, mock_open
import numpy as np
from learning.task_completion_bc_v1 import numeric

exp = types.ModuleType('learning.task_completion_training_coverage_v1.experiment')
exp.OUT = Path(__file__).resolve().parents[2]/'synthetic'
exp.CACHE = exp.OUT/'cache.npz'
for name in ('ready','verify_commit','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *a:None)
spec = importlib.util.spec_from_file_location('learning.task_completion_training_coverage_v1._analysis_test',Path(__file__).with_name('analysis.py'))
a = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_completion_training_coverage_v1.experiment':exp}): spec.loader.exec_module(a)


class CoverageTest(unittest.TestCase):
    def test_selection_metadata_only(self):
        segments = [dict(game='a',arm='teacher',seed=1,count=110,offset=0),dict(game='b',arm='candidate',seed=2,count=100,offset=110)]
        chosen = a.deterministic_select(segments)
        self.assertEqual(len(chosen),200)
        self.assertEqual(len({(x['game'],x['local_ordinal']) for x in chosen}),200)
        self.assertEqual(chosen,a.deterministic_select(segments))
        with self.assertRaises(ValueError): a.deterministic_select([{**segments[0],'offset':1}])
        with self.assertRaises(ValueError): a.deterministic_select(segments,211)

    def test_distance_group_and_invalid(self):
        names = ['unit.index','unit.x','unit.y','target.x','target.y','task.distance']+['task.op.'+op for op in a.OPS]
        schema = dict(feature_names=names + ['extra'+str(i) for i in range(98-len(names))])
        rows = np.zeros((3,98),np.float32)
        rows[:,0]=[0,1,2]; rows[:,1]=[0,2,3]; rows[:,3]=[3,3,4]; rows[:,5]=[3,1,1]
        rows[:,6]=1
        arrays = dict(candidates=rows,ops=np.zeros(3,np.uint8),offsets=np.array([0,3]),chosen=np.array([0]),status=np.array([0]))
        segments=[dict(game='a',arm='teacher',seed=1,count=1,offset=0)]
        positions=[dict(game='a',local_ordinal=0)]
        result = a.coverage(arrays,schema,segments,positions)
        self.assertEqual(result['distance_gap'],{'2':1})
        self.assertEqual(result['shortest_ties'],{'1':1})
        self.assertEqual(result['arm_operation_status'],[dict(arm='teacher',operation=a.OPS[0],status='completed_observed',count=1)])
        arrays['status'][0]=1
        self.assertEqual(a.coverage(arrays,schema,segments,positions)['arm_operation_status'],
                         [dict(arm='teacher',operation=a.OPS[0],status='unknown',count=1)])
        arrays['status'][0]=0
        rows[0,1]=2; rows[0,5]=1
        self.assertEqual(a.coverage(arrays,schema,segments,positions)['shortest_ties'],{'2':1})
        rows[0,5]=np.nan
        with self.assertRaises(ValueError): a.coverage(arrays,schema,segments,positions)
        rows[0,5]=1; arrays['chosen'][0]=3
        with self.assertRaises(ValueError): a.coverage(arrays,schema,segments,positions)

    def test_marker_failure_receipt(self):
        segments=[dict(game='a',arm='teacher',seed=1,count=200,offset=0)]
        positions=a.deterministic_select(segments)
        cfg=dict(source_segments=segments,sample_positions_sha256='h',schema_sha256='h',cache_sha256='h',
            limits=dict(max_archive_bytes=256*1024**2,max_uncompressed_bytes=512*1024**2,max_groups=16000,max_candidates=1200000,max_selected=200,seconds=300))
        saved={}
        with patch.object(a,'ready',return_value=cfg),patch.object(a,'verify_commit'),patch.object(a,'sha',return_value='h'), \
             patch.object(a,'read',return_value=positions),patch.object(Path,'exists',return_value=False), \
             patch.object(Path,'open',mock_open()),patch.object(a.json,'dump',side_effect=OSError()), \
             patch.object(a,'write_new',side_effect=lambda path,value:saved.update({Path(path).name:value})):
            with self.assertRaises(OSError): a.run()
        self.assertFalse(saved['analysis-failure.local.json']['marker']['complete'])
        self.assertFalse(saved['analysis-failure.local.json']['decode_started'])


if __name__ == '__main__': unittest.main()
