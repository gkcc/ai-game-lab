import unittest
from unittest.mock import patch
import numpy as np
from . import collect as c


class CollectTest(unittest.TestCase):
    def test_hash_sampling_and_numeric_labels(self):
        self.assertEqual(c.selected('g',3,4),c.selected('g',3,4))
        with self.assertRaises(ValueError): c.selected('g',719,0)
        g=dict(shared=np.zeros(275,np.float32),candidates=np.zeros((2,98),np.float32),
               teacher_choice_label=1,candidate_ops=(c.OPS[0],c.OPS[1]),seed=7)
        arrays=c.pack([g],1,2)
        self.assertEqual(arrays['teacher_choice_label'].tolist(),[1])
        self.assertNotIn('status',arrays)
        with self.assertRaises(ValueError): c.pack([g],1,1)
        with self.assertRaises(ValueError): c.pack([{**g,'teacher_choice_label':2}],1,2)

    def test_all_games_must_finish_and_attestations_match(self):
        sources=[dict(game=f'g{i}',side=0,seed=i,trace=dict(path=f'traces/{i}',size=1,sha256='a'*64),
                      replay_sha256='b'*64,audit_sha256='c'*64) for i in range(8)]
        limits=dict(compressed_bytes=10,decoded_bytes=10,max_line_bytes=10,max_groups=8,max_candidates=8,max_candidates_per_group=1)
        class Meter:
            def __init__(self,*args): self.finished=0
            def snapshot(self): return dict(files_opened=self.finished,files_completed=self.finished,raw_decode_passes_per_file=1)
        def turns(path,identity,meter,check,maximum):
            yield dict(step=0,records=[{}])
            meter.finished += 1
        def validator(game,labels,stream,schema):
            self.assertEqual(len(list(stream)),1)
            if game['key']=='g7' and failing[0]: raise ValueError('late invalid turn')
            return schema,[]
        item=dict(shared=np.zeros(275,np.float32),candidates=np.zeros((1,98),np.float32),
                  teacher_choice_label=0,candidate_ops=(c.OPS[0],))
        failing=[False]
        with patch.object(c,'Meter',Meter),patch.object(c,'raw_path',return_value=None),patch.object(c,'bounded_turns',side_effect=turns), \
             patch.object(c,'collect_game',side_effect=validator),patch.object(c,'selected',return_value=True),patch.object(c,'extract',return_value=item):
            arrays,receipt=c.collect(None,sources,sources,{},limits,lambda:None)
            self.assertEqual(len(arrays['teacher_choice_label']),8)
            self.assertEqual(receipt['files_completed'],8)
            failing[0]=True
            with self.assertRaises(ValueError): c.collect(None,sources,sources,{},limits,lambda:None)
            with self.assertRaises(ValueError): c.collect(None,sources,list(reversed(sources)),{},limits,lambda:None)


if __name__ == '__main__': unittest.main()
