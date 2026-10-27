import unittest
from unittest.mock import patch
import numpy as np
from . import analysis as a


def group():
    return dict(shared=np.zeros(275,np.float32),candidates=np.zeros((2,98),np.float32),
                teacher_choice_label=1,candidate_ops=(a.OPS[0],a.OPS[1]))


class AnalysisTest(unittest.TestCase):
    def test_selection_is_fixed_and_bounded(self):
        game=dict(key=a.GAMES[0],side=0,seed=3911611436)
        turns=[dict(step=i,records=[dict(chosen='x',teacher_label='x')]) for i in range(719)]
        def validate(game,labels,stream,schema):
            self.assertEqual(len(list(stream)),719); return schema,[]
        with patch.object(a,'collect_game',side_effect=validate),patch.object(a,'extract',side_effect=lambda *args,**kwargs:group()):
            self.assertEqual(len(a.collect_game_sample(game,iter(turns),{},max_candidates_per_group=2)),100)
            with self.assertRaises(ValueError): a.collect_game_sample({**game,'seed':1},iter(turns),{},max_candidates_per_group=2)
            with self.assertRaises(ValueError): a.collect_game_sample(game,iter(turns),{},max_candidates_per_group=2,max_retained_bytes=1)
            bad=[dict(step=0,records=[dict(chosen='x',teacher_label='different')])]
            with self.assertRaises(ValueError): a.collect_game_sample(game,iter(bad),{},max_candidates_per_group=2)

    def test_pair_counts_and_finite(self):
        class Model:
            def __init__(self,values): self.values=values; self.calls=0
            def score(self,shared,candidates): self.calls+=1; return self.values
        new,old=Model([0,1]),Model([1,0])
        result=a.analyze([group() for _ in range(200)],new,old)
        self.assertEqual(result['paired']['new_only'],200)
        self.assertEqual(result['cpu_forwards'],400)
        self.assertEqual(new.calls,200); self.assertEqual(old.calls,200)
        with self.assertRaises(ValueError): a.analyze([group() for _ in range(199)],new,old)
        with self.assertRaises(ValueError): a.analyze([group() for _ in range(200)],Model([float('nan'),0]),old)
        with patch.object(a,'MAX_TOTAL_ROWS',799),self.assertRaises(ValueError): a.analyze([group() for _ in range(200)],new,old)


if __name__=='__main__': unittest.main()
