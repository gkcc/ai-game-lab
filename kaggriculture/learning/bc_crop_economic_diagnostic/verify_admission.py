"""Regression: a diagnostic cannot rewrite the previous failure or reselect a head."""
from copy import deepcopy
import unittest
from .experiment import admission, MODEL_SHA, FAMILIES


class AdmissionTests(unittest.TestCase):
    def test_fixed_exception_only(self):
        cfg=dict(diagnostic_exception=True,promotion_allowed=False,training_runs=0,
            game_budget=48,entry_budget=2,model_sha256=MODEL_SHA,model_families=list(FAMILIES),
            all21_heads_already_inspected=True)
        report=dict(best_epoch=6,checkpoint_sha256=MODEL_SHA['fixed_epoch6'],offline_gate_passed=False)
        acceptance=dict(offline_learning_gate_passed=False,new_games=0,actual_candidate_gate_rejected=True)
        admission(cfg,report,acceptance)
        for obj,key,value in [(0,'diagnostic_exception',False),(0,'promotion_allowed',True),
            (0,'training_runs',1),(0,'game_budget',49),(0,'entry_budget',3),
            (0,'all21_heads_already_inspected',False),(1,'best_epoch',11),
            (1,'checkpoint_sha256','wrong'),(1,'offline_gate_passed',True),
            (2,'offline_learning_gate_passed',True),(2,'new_games',1),(2,'actual_candidate_gate_rejected',False)]:
            args=deepcopy([cfg,report,acceptance]);args[obj][key]=value
            with self.subTest(key=key,obj=obj),self.assertRaises(AssertionError):admission(*args)


if __name__=='__main__':unittest.main()
