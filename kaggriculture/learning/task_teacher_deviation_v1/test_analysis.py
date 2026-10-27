import unittest
from unittest.mock import patch
import numpy as np
from . import analysis as a


def arrays(width=2):
    n=a.POPULATION; m=n*width
    return dict(shared=np.zeros((n,275),np.float32),candidates=np.zeros((m,98),np.float32),
        offsets=np.arange(n+1,dtype=np.int64)*width,teacher_choice_label=np.ones(n,np.int64),
        ops=np.zeros(m,np.uint8),seed=np.zeros(n,np.uint32))


class Fake:
    def __init__(self): self.calls=0
    def score(self,shared,candidates): self.calls+=1; return np.zeros(len(candidates))


class AnalysisTest(unittest.TestCase):
    def test_fixed_selection_tie_and_zero_groups(self):
        self.assertEqual(a.select_indices(),a.select_indices())
        self.assertEqual(len(set(a.select_indices())),200)
        model,old=Fake(),Fake(); result=a.analyze(arrays(),model,old)
        self.assertEqual(model.calls,200); self.assertEqual(old.calls,200)
        self.assertEqual(result['models']['new']['top1_matches'],0)
        self.assertEqual(result['paired']['both_wrong'],200)
        self.assertEqual(result['models']['new']['by_teacher_operation'][a.OPS[0]]['mean_teacher_rank'],2)
        self.assertIsNone(result['models']['new']['by_teacher_operation'][a.OPS[1]]['top1_rate'])

    def test_invalid_input_and_budget_before_forward(self):
        for mutation in ('label','finite','budget','offset','operation'):
            data=arrays(); model=Fake()
            if mutation=='label': data['teacher_choice_label'][0]=2
            if mutation=='finite': data['candidates'][0,0]=np.nan
            if mutation=='offset': data['offsets'][1]=0
            if mutation=='operation': data['ops'][0]=255
            with self.subTest(mutation=mutation),patch.object(a,'MAX_ROWS',799 if mutation=='budget' else 120000),self.assertRaises(ValueError): a.analyze(data,model,Fake())
            self.assertEqual(model.calls,0)

    def test_bad_model_scores_fail(self):
        class Bad:
            def score(self,shared,candidates): return [float('nan')]*len(candidates)
        with self.assertRaises(ValueError): a.analyze(arrays(),Bad(),Fake())

    def test_failed_forward_records_started_count(self):
        events=[]
        class Bad:
            def score(self,shared,candidates): raise RuntimeError('failed')
        with self.assertRaises(RuntimeError): a.analyze(arrays(),Bad(),Fake(),events.append)
        self.assertEqual(events[-1],dict(forwards_started=1,forwards_completed=0))


if __name__ == '__main__': unittest.main()
