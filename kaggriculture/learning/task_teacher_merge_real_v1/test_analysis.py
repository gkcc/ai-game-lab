import unittest
import numpy as np
from .analysis import check
from learning.task_teacher_merge_v1.test_merge import make


class AnalysisTest(unittest.TestCase):
    def test_pure_old_arrays_and_alignment(self):
        old=make([2,1],11,1); original={k:v.copy() for k,v in old.items()}
        result=check(old,old_seeds=[11],holdout_seeds=[33],synthetic_seed=22,max_groups=4,max_candidates=6)
        self.assertEqual((result['old_groups'],result['old_candidate_rows'],result['total_groups']),(2,3,4))
        self.assertTrue(result['old_input_shape_valid'])
        self.assertEqual(result['synthetic_input_shapes'],[[1,373],[2,373]])
        self.assertNotIn('input_shapes',result)
        for key,value in original.items(): np.testing.assert_array_equal(old[key],value)

    def test_seed_conflicts_and_budget(self):
        old=make([1],11,1)
        for change in (dict(old_seeds=[12]),dict(holdout_seeds=[22]),dict(synthetic_seed=11),dict(max_groups=2),dict(max_candidates=3)):
            kwargs=dict(old_seeds=[11],holdout_seeds=[33],synthetic_seed=22,max_groups=3,max_candidates=4)
            with self.subTest(change=change),self.assertRaises(ValueError): check(old,**{**kwargs,**change})


if __name__=='__main__': unittest.main()
