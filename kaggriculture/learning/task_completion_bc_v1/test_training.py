import unittest
import numpy as np
from learning.task_completion_bc_v1.training import Group, inputs, probabilities, baseline_fit, baseline_probabilities, evaluate, validate, compare_logits


def group(chosen=0, seed=1, status='completed_observed'):
    return dict(shared=np.zeros(275), candidates=np.zeros((3,98)), chosen=chosen, candidate_ops=['DIG','DIG','WATER'], seed=seed, status=status)

class Scorer:
    def score(self, shared, candidates): return np.array([0., 1., 2.])

class TrainingChecks(unittest.TestCase):
    def test_mask_and_probability(self):
        p=probabilities([[0., 0., 100.]], [[True, True, False]])
        np.testing.assert_allclose(p, [[.5,.5,0.]])
        with self.assertRaises(ValueError): probabilities([[0.]], [[False]])
    def test_baseline_group_probabilities(self):
        counts=baseline_fit([group(2),group(0,status='unknown')])
        np.testing.assert_allclose(baseline_probabilities(['DIG','DIG','WATER'],counts), [1/6,1/6,2/3])
        np.testing.assert_allclose(baseline_probabilities(['DIG','DIG'],counts), [.5,.5])
    def test_features_exclude_metadata_and_label(self):
        a=Group(**group()); b=Group(**group(chosen=2,seed=999,status='unknown'))
        np.testing.assert_array_equal(inputs(a),inputs(b))
    def test_inference_parity(self):
        self.assertAlmostEqual(compare_logits([1., 0.], [1.00005, 0.]), .00005)
        with self.assertRaises(ValueError): compare_logits([1., 0.], [1.01, 0.])
        with self.assertRaises(ValueError): compare_logits([0., .00001], [.00001, 0.])
        with self.assertRaises(ValueError): compare_logits([float('nan')], [0.])

    def test_operation_accuracy_differs_from_candidate_accuracy(self):
        class SameOperation:
            def score(self, shared, candidates): return np.array([2., 1., 0.])
        g=group(chosen=1)
        result=evaluate([g],SameOperation(),baseline_fit([g]),set())
        self.assertEqual(result['by_operation']['DIG']['model_top1'], 0.)
        self.assertEqual(result['by_operation']['DIG']['model_operation_accuracy'], 1.)
        self.assertEqual(result['by_operation']['DIG']['baseline_operation_accuracy'], 1.)
        self.assertIsNone(result['by_operation']['WATER']['model_operation_accuracy'])

    def test_extreme_logits_and_invalid_metadata(self):
        class Extreme:
            def score(self, shared, candidates): return np.array([1000., -1000.])
        g=group(chosen=1); g['candidates']=np.zeros((2,98)); g['candidate_ops']=['DIG','WATER']
        result=evaluate([g],Extreme(),baseline_fit([g]),set())
        self.assertEqual(result['overall']['model_ce'], 2000.)
        for key, value in [('seed', -1), ('seed', 2**32), ('chosen', True)]:
            bad=group(); bad[key]=value
            with self.assertRaises(ValueError): validate([Group(**bad)])

    def test_validation_and_holdout(self):
        g=group(); bad=group(); bad['shared']=np.zeros(274)
        with self.assertRaises(ValueError): validate([Group(**bad)])
        bad=group(); bad['candidates'][0,0]=np.nan
        with self.assertRaises(ValueError): validate([Group(**bad)])
        counts=baseline_fit([g])
        with self.assertRaises(ValueError): evaluate([g],Scorer(),counts,{1})
        out=evaluate([g],Scorer(),counts,{1},holdout=False)
        self.assertEqual(out['overall']['count'],1)
        self.assertIsNone(out['by_status']['unknown']['model_ce'])
        self.assertEqual(out['cross_operation']['count'],1)

if __name__ == '__main__': unittest.main()
