"""Synthetic-only guard against omitted shed ordering in the replay audit."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from . import audit


def frame():
    return [{'observation':{'private':{'shed':{'A':1,'B':2}}}},
            {'observation':{'private':{'shed':{'C':3,'D':4}}}}]


class Env:
    def __init__(self,frames): self.frames=frames;self.index=0;self.state=deepcopy(frames[0])
    def step(self,_actions):
        self.index+=1;self.state=deepcopy(self.frames[self.index]);return self.state


def fake_reexecute(_replay,*,trace,label_sink):
    engine=audit.full_audit.make('synthetic')
    for _ in range(719): engine.step([])
    return dict(passed=True,official_transitions_reexecuted=719,
        both_public_and_private_states_match_each_step=True,inventory_order_checked_each_step=True)


class AuditTests(unittest.TestCase):
    def test_checks_both_sheds_and_restores_factory(self):
        replay={'steps':[frame() for _ in range(720)]}
        for corrupted in (False,True):
            frames=deepcopy(replay['steps'])
            if corrupted: frames[47][1]['observation']['private']['shed']={'D':4,'C':3}
            factory=lambda *_a,**_k:Env(frames)
            with patch.object(audit.full_audit,'make',factory), \
                 patch.object(audit.full_audit,'reexecute',side_effect=fake_reexecute):
                if corrupted:
                    with self.assertRaises(ValueError): audit.reexecute(replay,guard=lambda:None)
                else:
                    result=audit.reexecute(replay,guard=lambda:None)
                    self.assertEqual(result['shed_order_frames_checked'],720)
                self.assertIs(audit.full_audit.make,factory)


if __name__=='__main__': unittest.main()
