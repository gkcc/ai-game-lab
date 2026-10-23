"""Admission regression: no training, head substitution, expanded budget or old-pass rewrite."""
from copy import deepcopy
import unittest
from .experiment import admission,MODEL_SHA,FAMILIES


class AdmissionTests(unittest.TestCase):
    def test_fixed_model_holdout(self):
        cfg=dict(evaluation_role='independent-holdout-fixed-model',training_runs=0,parameter_updates=0,
            game_budget=96,entry_budget=2,model_sha256=MODEL_SHA,model_families=list(FAMILIES),
            default_change_allowed=False,old_offline_failure_preserved=True,test_seeds=list(range(8)))
        old=dict(stage_execution_passed=True,exploration_gate_passed=True,old_offline_gate_passed=False,
            old_phase_games=0,checkpoint_sha256=MODEL_SHA['fixed_epoch6'],promotion=False)
        admission(cfg,old)
        changes=[(0,'evaluation_role','development'),(0,'training_runs',1),(0,'parameter_updates',1),
            (0,'game_budget',98),(0,'entry_budget',3),(0,'default_change_allowed',True),
            (0,'old_offline_failure_preserved',False),(0,'test_seeds',[1]*8),
            (0,'test_seeds',[True,*range(1,8)]),(1,'old_offline_gate_passed',True),
            (1,'old_phase_games',50),(1,'exploration_gate_passed',False),(1,'checkpoint_sha256','other_head')]
        for obj,key,value in changes:
            args=deepcopy([cfg,old]);args[obj][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):admission(*args)


if __name__=='__main__':unittest.main()
