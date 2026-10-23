"""Small regression checks for aligned task cache and numeric ambiguity counts."""
from copy import deepcopy
import unittest
import numpy as np

from .cache import decode, NumericEquivalence
from learning.task_choice_v1.dataset import digest
from learning.task_choice_v1.features import schema_hash, SCHEMA_VERSION


class CacheChecks(unittest.TestCase):
    def fixture(self):
        shared_names=['s'+str(i) for i in range(275)]
        local_names=['x'+str(i) for i in range(98)]
        inputs=dict(schema_version=SCHEMA_VERSION,schema_sha256=schema_hash(local_names,shared_names),
            feature_names=local_names,shared_feature_names=shared_names,shared_x=[0.]*275,
            observation={'step':2,'player':0},
            candidates=[dict(id='u0:x0:y0:DIG',unit=0,target=[0,0],op='DIG',x=[0.]*98),
                        dict(id='u1:x1:y0:PLANT',unit=1,target=[1,0],op='PLANT',x=[1.]*98)])
        label={'teacher_label':'u1:x1:y0:PLANT','index':1}
        index=dict(game='test',split='train',seed=1,side=0,opponent='v1',step=2,decision=0,
                   input_sha256=digest(inputs))
        kwargs=dict(game='test',split='train',expected_seed=1,expected_side=0,expected_opponent='v1')
        return inputs,label,index,kwargs

    def test_numeric_inputs_exclude_metric_only_identifiers(self):
        inputs,label,index,kwargs=self.fixture()
        schema,shared,local,chosen,ops,identity=decode(inputs,label,index,**kwargs)
        self.assertEqual(shared.shape,(275,));self.assertEqual(local.shape,(2,98))
        self.assertEqual(chosen,1);self.assertEqual(identity.tolist(),[[0,0,0],[1,1,0]])
        changed=dict(index,seed=999);updated=dict(kwargs,expected_seed=999)
        other=decode(inputs,label,changed,**updated)
        np.testing.assert_array_equal(shared,other[1]);np.testing.assert_array_equal(local,other[2])

    def test_misaligned_hash_label_and_metadata_fail(self):
        for field in ('hash','label','seed','step'):
            inputs,label,index,kwargs=self.fixture()
            if field=='hash':index['input_sha256']='0'*64
            elif field=='label':label['index']=0
            elif field=='seed':index['seed']=2
            else:index['step']=3
            with self.subTest(field=field),self.assertRaises(AssertionError):
                decode(inputs,label,index,**kwargs)

    def test_nonfinite_after_float32_conversion_fails(self):
        inputs,label,index,kwargs=self.fixture();inputs['shared_x'][0]=1e100
        index['input_sha256']=digest(inputs)
        with np.errstate(over='ignore'),self.assertRaises(AssertionError):
            decode(inputs,label,index,**kwargs)

    def test_numeric_conflicts_and_first_tie_upper_bound(self):
        counter=NumericEquivalence()
        for label in (0,0,1):counter.add([0.],[[1.],[2.]],label)
        # Signed zero is the same numerical observation. Identical rows can
        # only select their first candidate with the deployed argmax tie rule.
        counter.add([-0.],[[3.],[3.]],1)
        counter.add([0.],[[3.],[3.]],0)
        result=counter.result()
        self.assertEqual(result['examples'],5)
        self.assertEqual(result['numeric_distinct_groups'],2)
        self.assertEqual(result['duplicate_numeric_groups_with_different_labels'],2)
        self.assertEqual(result['labels_unreachable_due_to_first_argmax_alias'],1)
        self.assertEqual(result['optimistic_exact_id_correct_upper_bound'],3)
        self.assertEqual(result['optimistic_exact_id_accuracy_upper_bound'],.6)
        self.assertEqual(result['group_size_counts'],{'2':5})

    def test_merging_splits_preserves_cross_split_conflicts(self):
        train=NumericEquivalence();validation=NumericEquivalence()
        train.add([0.],[[1.],[2.]],0);validation.add([0.],[[1.],[2.]],1)
        combined=NumericEquivalence();combined.merge(train);combined.merge(validation)
        self.assertEqual(train.result()['examples'],1)
        self.assertEqual(validation.result()['examples'],1)
        self.assertEqual(combined.result()['duplicate_numeric_groups_with_different_labels'],1)
        self.assertEqual(combined.result()['optimistic_exact_id_accuracy_upper_bound'],.5)

    def test_multicandidate_bound_excludes_easy_singletons(self):
        counter=NumericEquivalence()
        for _ in range(8):counter.add([0.],[[1.]],0)
        counter.add([0.],[[1.],[2.]],0);counter.add([0.],[[1.],[2.]],1)
        self.assertEqual(counter.result()['optimistic_exact_id_accuracy_upper_bound'],.9)
        multi=counter.result(multicandidate=True)
        self.assertEqual(multi['examples'],2)
        self.assertEqual(multi['optimistic_exact_id_accuracy_upper_bound'],.5)


if __name__=='__main__':unittest.main()
