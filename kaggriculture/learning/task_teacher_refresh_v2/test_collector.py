import unittest
import copy
import gzip
import hashlib
import json
from unittest.mock import patch
import numpy as np
from learning.task_choice_v1.features import OPS
from . import collector
from learning.task_completion_bc_v1.test_data import fixture
from learning.task_teacher_dev_deviation_v1.test_stream import FakePath


def sources():
    result=[]
    for seed in (1,2):
        for opponent in ('round2','market'):
            for side in (0,1):
                game=f'dev-{seed}-{opponent}-{side}-teacher_task'
                result.append(dict(game=game,seed=seed,opponent=opponent,side=side,
                    trace=dict(path='traces/'+game+'.jsonl.gz',size=1,sha256='a'*64),replay_sha256='b'*64,audit_sha256='c'*64))
    return result


LIMITS=dict(compressed_bytes=1000,decoded_bytes=1000,max_line_bytes=100,max_groups=20,max_candidates=40,
    max_candidates_per_group=10,max_total_callbacks=100,max_total_candidate_rows=200)


class FakeMeter:
    def __init__(self,*args): self.opened=0; self.completed=0
    def snapshot(self): return dict(files_opened=self.opened,files_completed=self.completed,raw_decode_passes_per_file=1)


def fake_stream(path,identity,meter,check,max_line):
    meter.opened+=1
    try:
        for step in range(719):
            records=[] if step else [dict(source='teacher',teacher_label='x',chosen='x',op=op,inputs={'candidates':[{},{}]}) for op in ('DIG','FERTILIZE','WATER')]
            yield dict(step=step,records=records)
        meter.completed+=1
    finally: pass


def fake_extract(record,schema,**kwargs):
    op=record['op']
    return dict(shared=np.zeros(275,np.float32),candidates=np.zeros((2,98),np.float32),teacher_choice_label=0,
                candidate_ops=(op,'WATER'))


def fake_collect_game(game,labels,turns,schema):
    steps=[turn['step'] for turn in turns]
    if steps!=list(range(719)): raise ValueError('Missing turns')
    return None,[]


class CollectorTests(unittest.TestCase):
    def run_fake(self,limits=LIMITS,extract=fake_extract,stream=fake_stream,selection=False):
        items=sources()
        with patch.object(collector,'Meter',FakeMeter),patch.object(collector,'raw_path',return_value='fake'), \
             patch.object(collector,'bounded_turns',side_effect=stream),patch.object(collector,'extract',side_effect=extract), \
             patch.object(collector,'collect_game',side_effect=fake_collect_game),patch.object(collector,'selected',return_value=selection):
            return collector.collect('unused',items,items.copy(),[1,2],{},limits,lambda:None)

    def test_rare_retention_and_full_flow_counts(self):
        arrays,summary=self.run_fake()
        self.assertEqual(len(arrays['seed']),16); self.assertEqual(summary['retained_groups'],16)
        for entry in summary['by_seed_opponent']:
            self.assertEqual(entry['callbacks'],6); self.assertEqual(entry['retained'],4)
            self.assertEqual(entry['chosen']['WATER'],2); self.assertEqual(entry['retained_chosen']['WATER'],0)
            self.assertEqual(entry['candidate_rows']['WATER'],8); self.assertEqual(entry['candidate_callbacks']['WATER'],6)

    def test_other_selected_and_budget(self):
        arrays,_=self.run_fake(limits={**LIMITS,'max_groups':24,'max_candidates':48},selection=True)
        self.assertEqual(len(arrays['seed']),24)
        with self.assertRaises(ValueError): self.run_fake(limits={**LIMITS,'max_groups':15})
        with self.assertRaises(ValueError): self.run_fake(limits={**LIMITS,'max_candidates':31})

    def test_full_stream_budget_rejects_unsampled_water(self):
        def water_stream(*args):
            for turn in fake_stream(*args):
                for record in turn['records']: record['op']='WATER'
                yield turn
        with self.assertRaisesRegex(ValueError,'Full-stream budget'): self.run_fake(stream=water_stream,limits={**LIMITS,'max_total_callbacks':3})
        with self.assertRaisesRegex(ValueError,'Full-stream budget'): self.run_fake(stream=water_stream,limits={**LIMITS,'max_total_candidate_rows':7})

    def test_invalid_identity(self):
        items=sources(); altered=[dict(x) for x in items]; altered[0]['opponent']='market'
        with self.assertRaises(ValueError): collector.validate_sources(altered,altered,[1,2])
        with self.assertRaises(ValueError): collector.validate_sources(items[:-1],items[:-1],[1,2])
        with self.assertRaises(ValueError): collector.validate_sources(items,items,[1,1])

    def test_bad_label_or_extractor_validation_aborts(self):
        def bad_stream(*args):
            for turn in fake_stream(*args):
                if turn['records']: turn['records'][0]['chosen']='other'
                yield turn
        with self.assertRaises(ValueError): self.run_fake(stream=bad_stream)
        for message in ('observation','private fields','candidate identity'):
            with self.subTest(message=message),self.assertRaises(ValueError):
                self.run_fake(extract=lambda *a,**k:(_ for _ in ()).throw(ValueError(message)))

    def test_real_gzip_eight_source_pipeline_and_corruptions(self):
        class NamedFakePath(FakePath):
            def __init__(self,data,name): super().__init__(data); self.name=name
            def __str__(self): return self.name
        _,_,turns,schema=fixture()
        template=turns[0]['records'][0]
        items=sources(); paths={}
        for source in items:
            record=copy.deepcopy(template)
            record['source']='teacher'; record['teacher_label']=record['chosen']
            record['inputs']['observation']['player']=source['side']
            records=[dict(step=i,records=[record] if i==0 else []) for i in range(719)]
            data=gzip.compress(b''.join((json.dumps(turn,separators=(',',':'))+'\n').encode() for turn in records))
            source['trace'].update(size=len(data),sha256=hashlib.sha256(data).hexdigest())
            paths[source['trace']['path']]=NamedFakePath(data,source['trace']['path'])
        limits={**LIMITS,'compressed_bytes':8*1024**2,'decoded_bytes':8*1024**2,'max_line_bytes':1024**2}
        def run():
            with patch.object(collector,'raw_path',side_effect=lambda root,identity,kind:paths[identity['path']]):
                return collector.collect('unused',items,copy.deepcopy(items),[1,2],schema,limits,lambda:None)
        arrays,summary=run()
        self.assertEqual(len(arrays['seed']),8); self.assertEqual(summary['total_callbacks'],8)
        self.assertEqual(summary['total_candidate_rows'],16); self.assertEqual(summary['retained_groups'],8)
        self.assertEqual(summary['meter']['files_completed'],8)
        last=items[-1]; original=copy.deepcopy(last['trace'])
        last['trace']['sha256']='0'*64
        with self.assertRaises(ValueError): run()
        last['trace'].update(original)
        for corruption in ('label','private'):
            record=copy.deepcopy(template); record['source']='teacher'; record['teacher_label']=record['chosen']
            record['inputs']['observation']['player']=last['side']
            if corruption=='label': record['teacher_label']='bad'
            else: record['inputs']['observation']['farms'][1-last['side']]['private']={'seeds':{'X':1}}
            data=gzip.compress(b''.join((json.dumps(dict(step=i,records=[record] if i==0 else []),separators=(',',':'))+'\n').encode() for i in range(719)))
            last['trace'].update(size=len(data),sha256=hashlib.sha256(data).hexdigest())
            paths[last['trace']['path']]=NamedFakePath(data,last['trace']['path'])
            with self.subTest(corruption=corruption),self.assertRaises(ValueError): run()


if __name__=='__main__': unittest.main()
