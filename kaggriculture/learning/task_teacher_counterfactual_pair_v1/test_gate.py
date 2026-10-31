"""Synthetic one-shot lifecycle, boundaries, and budget checks."""
from copy import deepcopy
import unittest

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_single_intervention_v1.test_intervention import inputs,BUDGET_KEYS
from .gate import OneShotSelector


class Harness:
    def __init__(self,mode,step,ordinal,charge=1,player=0):
        self.step,self.ordinal,self.charge=step,ordinal,charge
        self.player=player
        self.records=[];self.total={k:0 for k in BUDGET_KEYS};self.turn=dict(self.total);self.calls=0
        expected=dict(input_sha256=ordered_hash(self.input(step)),original_choice='student',teacher_choice='teacher')
        self.gate=OneShotSelector(self.original,lambda:len(self.records),
            lambda:dict(total=self.total,turn=self.turn),expected,mode,step=step,ordinal=ordinal)
    def original(self,value):
        self.calls+=1
        for counters in (self.total,self.turn):
            counters['task_forwards']+=self.charge
            counters['task_rows']+=self.charge*len(value['candidates'])
        return 'student'
    def input(self,step):
        value=inputs(step);value['observation']['player']=self.player;return value
    def start(self): self.records=[];self.turn={k:0 for k in BUDGET_KEYS}
    def choose(self,step,teacher='teacher'):
        value=self.input(step);chosen=self.gate(value)
        self.records.append(dict(inputs=deepcopy(value),teacher_label=teacher,chosen=chosen,source='callback'))
        return chosen
    def end(self,step): self.gate.verify_turn(dict(step=step,records=self.records))


class GateTests(unittest.TestCase):
    def test_arbitrary_targets_call_model_once_and_finish_all_turns(self):
        for step,ordinal in ((0,0),(2,5),(718,31)):
            proofs=[]
            for mode,choice in (('control','student'),('intervene','teacher')):
                h=Harness(mode,step,ordinal)
                for turn in range(719):
                    h.start()
                    if turn==step:
                        for _ in range(ordinal): h.choose(turn)
                        before=h.calls
                        self.assertEqual(h.choose(turn),choice)
                        self.assertEqual(h.calls,before+1)
                    h.end(turn)
                self.assertTrue(h.gate.verify_complete())
                proofs.append(h.gate.proof())
            self.assertEqual(proofs[0]['budget_before'],proofs[1]['budget_before'])
            self.assertEqual(proofs[0]['budget_after'],proofs[1]['budget_after'])

    def test_invalid_or_changed_target_and_misreported_teacher_fail_closed(self):
        for step,ordinal in ((-1,0),(719,0),(0,-1),(0,32),(True,0)):
            with self.assertRaises(ValueError): Harness('control',step,ordinal)
        for charge in (0,2):
            h=Harness('intervene',0,0,charge=charge);h.start()
            with self.assertRaises(ValueError): h.choose(0)
            with self.assertRaises(ValueError): h.gate.proof()
        h=Harness('intervene',0,0);h.start();h.choose(0,teacher='wrong')
        with self.assertRaises(ValueError): h.end(0)
        with self.assertRaises(ValueError): h.gate.proof()

    def test_second_seat_and_later_callback_keep_single_substitution(self):
        for step,ordinal in ((0,0),(718,30)):
            h=Harness('intervene',step,ordinal,player=1)
            for turn in range(719):
                h.start()
                if turn==step:
                    for _ in range(ordinal): h.choose(turn)
                    self.assertEqual(h.choose(turn),'teacher')
                    self.assertEqual(h.choose(turn),'student')
                h.end(turn)
            self.assertTrue(h.gate.verify_complete())


if __name__=='__main__': unittest.main()
