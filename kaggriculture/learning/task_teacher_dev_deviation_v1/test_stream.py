import copy
import gzip
import hashlib
import io
import json
import types
import unittest
from learning.task_action_impact_diagnostic_v1.analysis import Meter
from learning.task_teacher_choice_bc_v1.collect import bounded_turns
from learning.task_completion_bc_v1.test_data import fixture
from .analysis import collect_game_sample,GAMES


class FakePath:
    def __init__(self,data): self.data=data
    def resolve(self): return self
    def __str__(self): return 'synthetic-trace'
    def stat(self): return types.SimpleNamespace(st_size=len(self.data))
    def open(self,*args,**kwargs): return io.BytesIO(self.data)


class StreamTest(unittest.TestCase):
    def test_complete_gzip_hash_and_failed_receipts(self):
        raw=b''.join((json.dumps(dict(step=i,records=[]))+'\n').encode() for i in range(719))
        packed=gzip.compress(raw); path=FakePath(packed)
        identity=dict(size=len(packed),sha256=hashlib.sha256(packed).hexdigest(),path='synthetic')
        meter=Meter(100000,100000)
        self.assertEqual(len(list(bounded_turns(path,identity,meter,lambda:None,1024))),719)
        self.assertEqual(meter.snapshot()['files_completed'],1)
        cases=[(path,{**identity,'sha256':'0'*64},100000,100000),
               (FakePath(packed[:-2]),dict(identity,size=len(packed)-2,sha256=hashlib.sha256(packed[:-2]).hexdigest()),100000,100000),
               (path,identity,4,100000),(path,identity,100000,10)]
        for path_,ident,compressed,decoded in cases:
            meter=Meter(compressed,decoded)
            with self.subTest(compressed=compressed,decoded=decoded),self.assertRaises((ValueError,EOFError)):
                list(bounded_turns(path_,ident,meter,lambda:None,1024))
            self.assertEqual(meter.snapshot()['files_completed'],0)

    def test_real_validators_full_719_turns_and_label_projection(self):
        _,_,turns,schema=fixture()
        template=turns[0]['records'][0]
        template['teacher_label']=template['chosen']
        def turns719():
            for i in range(719):
                record=copy.deepcopy(template); record['inputs']['observation']['step']=i
                yield dict(step=i,records=[record])
        game=dict(key=GAMES[0],side=0,seed=3911611436)
        result=collect_game_sample(game,turns719(),schema,max_candidates_per_group=2)
        self.assertEqual(len(result),100)
        self.assertTrue(all(x['teacher_choice_label']==0 for x in result))
        raw=b''.join((json.dumps(turn,separators=(',',':'))+'\n').encode('utf-8') for turn in turns719())
        packed=gzip.compress(raw); path=FakePath(packed)
        identity=dict(size=len(packed),sha256=hashlib.sha256(packed).hexdigest(),path='synthetic')
        meter=Meter(4*1024**2,16*1024**2)
        stream=bounded_turns(path,identity,meter,lambda:None,1024**2)
        streamed=collect_game_sample(game,stream,schema,max_candidates_per_group=2)
        self.assertEqual(len(streamed),100)
        self.assertTrue(all(x['teacher_choice_label']==0 for x in streamed))
        self.assertEqual(meter.snapshot()['files_completed'],1)
        def bad_turns():
            for turn in turns719():
                if turn['step']==718: turn['step']=717
                yield turn
        with self.assertRaises(ValueError): collect_game_sample(game,bad_turns(),schema,max_candidates_per_group=2)
        def bad_teacher():
            for turn in turns719():
                if turn['step']==718: turn['records'][0]['teacher_label']='missing'
                yield turn
        with self.assertRaises(ValueError): collect_game_sample(game,bad_teacher(),schema,max_candidates_per_group=2)


if __name__=='__main__': unittest.main()
