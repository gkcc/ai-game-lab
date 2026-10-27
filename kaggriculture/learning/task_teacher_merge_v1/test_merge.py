import unittest
import numpy as np
from .merge import merge,to_groups
from learning.task_choice_v1.features import OPS
from learning.task_teacher_choice_bc_v1.training import as_groups,inputs,validate


def make(widths,seed,base):
    total=sum(widths); candidates=np.zeros((total,98),np.float32); candidates[:,0]=np.arange(total)+base
    return dict(shared=np.full((len(widths),275),base,np.float32),candidates=candidates,
        offsets=np.asarray([0,*np.cumsum(widths)],np.int64),teacher_choice_label=np.asarray([w-1 for w in widths],np.int64),
        ops=np.asarray([i%len(OPS) for i in range(total)],np.uint8),seed=np.full(len(widths),seed,np.uint32))


class MergeTest(unittest.TestCase):
    def test_alignment_and_no_aliasing(self):
        old,new=make([2,1],11,1),make([3,2],22,10)
        before={k:v.copy() for k,v in old.items()}; before_new={k:v.copy() for k,v in new.items()}
        result=merge(old,new,old_seeds=[11],new_seeds=[22],holdout_seeds=[33],max_groups=4,max_candidates=8)
        self.assertEqual(result['offsets'].tolist(),[0,2,3,6,8])
        self.assertEqual(result['teacher_choice_label'].tolist(),[1,0,2,1])
        groups=to_groups(result,train_seeds=[11,22],holdout_seeds=[33],max_groups=4,max_candidates=8)
        self.assertEqual([g['candidates'][:,0].tolist() for g in groups],[[1,2],[3],[10,11,12],[13,14]])
        self.assertEqual(groups[2]['candidate_ops'],tuple(OPS[i] for i in (0,1,2)))
        trainer_groups=as_groups(groups)
        validate(trainer_groups)
        self.assertEqual([inputs(g).shape for g in trainer_groups],[(2,373),(1,373),(3,373),(2,373)])
        result['candidates'][0,0]=999; groups[1]['shared'][0]=999
        for key in old: np.testing.assert_array_equal(old[key],before[key]); np.testing.assert_array_equal(new[key],before_new[key])

    def test_reject_invalid_roles_and_data(self):
        old,new=make([1],11,1),make([2],22,2)
        kwargs=dict(old_seeds=[11],new_seeds=[22],holdout_seeds=[33],max_groups=2,max_candidates=3)
        for changes in (dict(old_seeds=[12]),dict(new_seeds=[11]),dict(holdout_seeds=[22]),
                        dict(old_seeds=[]),dict(max_groups=1),dict(max_candidates=2),dict(holdout_seeds=[True])):
            with self.subTest(changes=changes),self.assertRaises(ValueError): merge(old,new,**{**kwargs,**changes})
        for key,value in (('teacher_choice_label',np.array([2],np.int64)),('offsets',np.array([0,3],np.int64)),
                          ('seed',np.array([22],np.int64))):
            with self.subTest(key=key),self.assertRaises(ValueError): merge(old,{**new,key:value},**kwargs)
        invalid=make([1],22,2); invalid['candidates'][0,0]=np.nan
        with self.assertRaises(ValueError): merge(old,invalid,**kwargs)
        empty=make([],22,2)
        with self.assertRaises(ValueError): merge(old,empty,**kwargs)

    def test_trainer_group_width_boundary(self):
        def wide(count,seed):
            return dict(shared=np.zeros((1,275),np.float32),
                candidates=np.broadcast_to(np.zeros((1,98),np.float32),(count,98)),
                offsets=np.array([0,count],np.int64),teacher_choice_label=np.array([0],np.int64),
                ops=np.zeros(count,np.uint8),seed=np.array([seed],np.uint32))
        old=make([1],11,1); good=wide(65536,22); bad=wide(65537,22)
        self.assertEqual(len(to_groups(good,train_seeds=[22],holdout_seeds=[33],max_groups=1,max_candidates=65536)[0]['candidates']),65536)
        merged=merge(old,good,old_seeds=[11],new_seeds=[22],holdout_seeds=[33],max_groups=2,max_candidates=65537)
        self.assertEqual(merged['offsets'].tolist(),[0,1,65537])
        with self.assertRaises(ValueError): to_groups(bad,train_seeds=[22],holdout_seeds=[33],max_groups=1,max_candidates=65537)
        with self.assertRaises(ValueError): merge(old,bad,old_seeds=[11],new_seeds=[22],holdout_seeds=[33],max_groups=2,max_candidates=65538)

    def test_to_groups_cannot_bypass_seed_roles(self):
        data=make([1],22,1)
        for train,holdout in (([11],[33]),([22],[22]),([22,11],[33]),([22],[])):
            with self.subTest(train=train,holdout=holdout),self.assertRaises(ValueError):
                to_groups(data,train_seeds=train,holdout_seeds=holdout,max_groups=1,max_candidates=1)


if __name__=='__main__': unittest.main()
