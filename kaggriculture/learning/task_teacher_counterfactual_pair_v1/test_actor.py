"""Run the real filename adapter and gate together with synthetic controllers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_single_intervention_v1.test_intervention import inputs
from . import actor, gate


class FakeBoundedAgent:
    instances=[]

    def __init__(self, *args, **kwargs):
        self.controller=SimpleNamespace(records=[],task_selector=self.original)
        self.budget=SimpleNamespace(counts={key:0 for key in gate.BUDGET_KEYS},
                                    turn={key:0 for key in gate.BUDGET_KEYS})
        self.calls=0
        self.last=None
        self.instances.append(self)

    def original(self, value):
        self.calls+=1
        for counters in (self.budget.counts,self.budget.turn):
            counters['task_forwards']+=1
            counters['task_rows']+=len(value['candidates'])
        return 'student'

    def __call__(self, observation, _configuration):
        step=observation['step']
        self.controller.records=[]
        self.budget.turn={key:0 for key in gate.BUDGET_KEYS}
        for counts in (self.budget.counts,self.budget.turn): counts['controller_calls']+=1
        selected=[]
        if step==3:
            for _ in range(3):
                value=inputs(step)
                choice=self.controller.task_selector(value)
                selected.append(choice)
                self.controller.records.append(dict(inputs=deepcopy(value),teacher_label='teacher',
                    chosen=choice,source='callback'))
        self.last=dict(step=step,records=self.controller.records)
        return {'choices':selected}


class ActorTests(unittest.TestCase):
    def test_real_build_runs_full_control_then_single_intervention(self):
        raw=Path(gate.__file__).read_bytes()
        config=dict(schema='teacher-pair-gate-v1',
            expected=dict(input_sha256=ordered_hash(inputs(3)),original_choice='student',teacher_choice='teacher'),
            step=3,ordinal=1,crop_checkpoint='crop',crop_sha256='a'*64,task_checkpoint='task',
            task_sha256='b'*64,limits={},gate_source_sha256=hashlib.sha256(raw).hexdigest(),
            actor_source_sha256='c'*64)
        events={}
        def read_bytes(path):
            self.assertEqual(path.name,'gate.py')
            return raw
        def read_text(path,**_kwargs):
            if path.name=='gate.local.json': return json.dumps(config)
            return json.dumps(events[path.name])
        def save(path,value):
            self.assertNotIn(path.name,events)
            events[path.name]=deepcopy(value)
        with patch.object(Path,'resolve',lambda path,strict=False:path), \
             patch.object(Path,'read_bytes',read_bytes), patch.object(Path,'read_text',read_text), \
             patch.object(actor,'_relative',lambda root,value:root/value), patch.object(actor,'_write',save), \
             patch('learning.task_dagger_once_v1.policy.BoundedAgent',FakeBoundedAgent):
            FakeBoundedAgent.instances=[]
            for mode,expected in (('control',['student']*3),
                                  ('intervene',['student','teacher','student'])):
                select=actor.build('synthetic','game',mode)
                for step in range(719):
                    result=select({'step':step},{})
                    if step==3: self.assertEqual(result['choices'],expected)
                self.assertEqual(FakeBoundedAgent.instances[-1].calls,3)
                self.assertTrue(events[mode+'.complete.local.json']['target_verified'])
            self.assertEqual(events['control.target.local.json']['budget_before'],
                             events['intervene.target.local.json']['budget_before'])


if __name__=='__main__': unittest.main()
