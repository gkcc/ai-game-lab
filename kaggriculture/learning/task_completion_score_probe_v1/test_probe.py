import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy
from learning.task_bc_v1 import model as real_model
from unittest.mock import patch, mock_open
from contextlib import contextmanager, ExitStack

impact = types.ModuleType('learning.task_action_impact_diagnostic_v1.analysis')
def compare(a,b):
    if not a or not b or a[0]['inputs'] != b[0]['inputs'] or a[0]['chosen'] == b[0]['chosen']:
        return dict(choice=None)
    return dict(choice=dict(callback=0))
impact.compare_callbacks = compare
impact.Meter = object
impact.decoded_chunks = None
obs = types.ModuleType('learning.task_action_observation_v1.analysis'); obs.raw_path = lambda *a: None
exp = types.ModuleType('learning.task_completion_score_probe_v1.experiment')
exp.ROOT = Path(__file__).resolve().parents[2]
exp.OUT = exp.ROOT/'synthetic'
exp.RAW_PARENT = exp.PARENT = exp.ROOT/'synthetic-parent'
for name in ('ready','verify_commit','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *a:None)
spec = importlib.util.spec_from_file_location('learning.task_completion_score_probe_v1._probe_test',Path(__file__).with_name('probe.py'))
p = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_action_impact_diagnostic_v1.analysis':impact,
    'learning.task_action_observation_v1.analysis':obs,'learning.task_completion_score_probe_v1.experiment':exp}): spec.loader.exec_module(p)


class ProbeTest(unittest.TestCase):
    def records(self):
        inputs = dict(candidates=[dict(id='a',target=[0,0]),dict(id='b',target=[0,0])])
        return (dict(inputs=inputs,chosen='a',teacher_label='b',unit=0,op='PLANT',source='callback'),
                dict(inputs=inputs,chosen='b',unit=1,op='PLANT',source='teacher'))
    def test_matching_and_rank(self):
        a,b = self.records()
        self.assertEqual(p.matched_records([a],[b]),(a,b))
        result = p.ranking(a['inputs'],a,b,[1.,1.])
        self.assertTrue(result['selected_teacher_exact_tie'])
        self.assertEqual(result['teacher_rank'],2)
        self.assertTrue(result['same_target'])
        with self.assertRaises(ValueError): p.matched_records([a],[{**b,'inputs':{}}])
        with self.assertRaises(ValueError): p.matched_records([a],[{**b,'chosen':'a'}])
        with self.assertRaises(ValueError): p.ranking(a['inputs'],a,b,[0.,1.])
        with self.assertRaises(ValueError): p.ranking(a['inputs'],a,b,[float('nan'),1.])
        with self.assertRaises(ValueError): p.ranking({'candidates':[{'id':str(i)} for i in range(8193)]},a,b,[])

    def test_stream_line_framing(self):
        @contextmanager
        def chunks(*args): yield iter([b'{}\n{}\n{"step":2}\n' + b'{}\n'*716])
        with patch.object(p,'decoded_chunks',chunks):
            self.assertEqual(p.selected_line(None,None,None,lambda:None,100),{'step':2})
        @contextmanager
        def bad(*args): yield iter([b'{}\n'*718 + b'{}'])
        with patch.object(p,'decoded_chunks',bad), self.assertRaises(ValueError):
            p.selected_line(None,None,None,lambda:None,100)

    def test_marker_and_complete_forward(self):
        a,b = self.records()
        trace = [dict(step=2,task_checkpoint_sha256='model',checkpoint_sha256='crop',records=[a]),
                 dict(step=2,task_checkpoint_sha256=None,checkpoint_sha256='crop',records=[b])]
        cfg = dict(seed=3342200684,opponent='round2',side=0,step=2,limits=p.LIMITS,
            candidate_checkpoint='model.npz',candidate_checkpoint_sha256='model',crop_checkpoint_sha256='crop',
            raw_inputs={arm:dict(sha256='x') for arm in ('candidate','teacher_task')})
        class Meter:
            def __init__(self,*args): pass
            def snapshot(self): return dict(files_opened=2,files_completed=2,raw_decode_passes_per_file=1)
        class Model:
            calls = 0
            def __init__(self,*args): pass
            def __call__(self,inputs):
                Model.calls += 1
                self.last = dict(chosen='a',logits=[1.,0.])
                return 'a'
        for scenario in ('success','partial','load_changed','forward_changed'):
            with self.subTest(scenario=scenario), ExitStack() as stack:
                records = {}
                Model.calls = 0
                for name,value in (('ready',cfg),('verify_commit',None),('deadline_check',None)):
                    stack.enter_context(patch.object(p,name,return_value=value))
                model_hashes = iter(['changed'] if scenario == 'load_changed' else
                                    ['model','changed'] if scenario == 'forward_changed' else ['model','model'])
                def hash_file(path):
                    return next(model_hashes) if Path(path).name == 'model.npz' else 'hash'
                stack.enter_context(patch.object(p,'sha',side_effect=hash_file))
                stack.enter_context(patch.object(p,'write_new',side_effect=lambda path,value:records.update({Path(path).name:value})))
                stack.enter_context(patch.object(p,'Meter',Meter))
                stack.enter_context(patch.object(p,'TaskNetwork',Model))
                stack.enter_context(patch.object(p,'raw_path',return_value=None))
                stack.enter_context(patch.object(p,'selected_line',side_effect=trace))
                stack.enter_context(patch.object(Path,'exists',return_value=False))
                stack.enter_context(patch.object(Path,'open',mock_open()))
                if scenario == 'partial': stack.enter_context(patch.object(p.json,'dump',side_effect=OSError()))
                if scenario == 'partial':
                    with self.assertRaises(OSError): p.run()
                    self.assertFalse(records['probe-failure.local.json']['marker']['complete'])
                    self.assertEqual(records['probe-failure.local.json']['cpu_forward_count'],0)
                elif scenario == 'success':
                    self.assertEqual(p.run()['cpu_forwards'],1)
                    self.assertEqual(Model.calls,1)
                else:
                    with self.assertRaises(ValueError): p.run()
                    failure = records['probe-failure.local.json']
                    self.assertEqual(Model.calls,0 if scenario == 'load_changed' else 1)
                    self.assertEqual(failure['cpu_forward_count'],0 if scenario == 'load_changed' else 1)


if __name__ == '__main__': unittest.main()
