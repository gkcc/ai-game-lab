"""Small deterministic regressions; no training, games, files, or GPU required."""
import io
import importlib.util
import unittest
import numpy as np

from .model import continuous_mask, numpy_logits
from .train import class_weights, microbatches, weighted_group_loss, group_metrics, metric_gate


class Contracts(unittest.TestCase):
    def test_whole_batch_weighted_gradient_accumulation(self):
        logits=np.asarray([[.4,-.8,7.],[1.,.1,.2],[-1.,.6,99.]],dtype=np.float64)
        mask=np.asarray([[1,1,0],[1,1,1],[1,1,0]],bool)
        labels=np.asarray([1,0,1]);weights=np.asarray([.5,3.,1.5]);denominator=weights.sum()
        full=lambda x:weighted_group_loss(x,mask,labels,weights,denominator)
        separate=lambda x:sum(weighted_group_loss(x[ids],mask[ids],labels[ids],weights[ids],denominator)
                              for ids in ([0,2],[1]))
        self.assertAlmostEqual(full(logits),separate(logits),places=14)
        for row in range(3):
            for column in range(3):
                plus,minus=logits.copy(),logits.copy()
                plus[row,column]+=1e-5;minus[row,column]-=1e-5
                a=(full(plus)-full(minus))/2e-5
                b=(separate(plus)-separate(minus))/2e-5
                self.assertAlmostEqual(a,b,places=9)
                if not mask[row,column]:self.assertEqual(a,0)
        wrong=sum(weighted_group_loss(logits[ids],mask[ids],labels[ids],weights[ids],weights[ids].sum())
                  for ids in ([0,2],[1]))/2
        self.assertGreater(abs(wrong-full(logits)),1e-3)

    def test_padding_no_loss_and_whole_groups(self):
        logits=np.asarray([[2.,999.],[999.,-999.]])
        mask=np.asarray([[1,0],[1,0]],bool)
        self.assertEqual(weighted_group_loss(logits,mask,np.zeros(2,int),np.asarray([2.,0.]),2.),0.)
        extreme=np.asarray([[-1e31,0.]])
        self.assertEqual(weighted_group_loss(extreme,np.asarray([[True,False]]),np.asarray([0]),np.asarray([1.]),1.),0.)
        sizes=np.asarray([1,3,65,256,65536])
        data={'offsets':np.r_[0,np.cumsum(sizes)]}
        batches=list(microbatches(data,np.arange(5)))
        ids=[i for group,capacity,width in batches for i in group]
        self.assertEqual(sorted(ids),list(range(5)))
        for group,capacity,width in batches:
            self.assertLessEqual(capacity*width,65536)
            self.assertTrue(all(sizes[i]<=width for i in group))
        with self.assertRaises(ValueError):
            list(microbatches({'offsets':np.asarray([0,65537])},[0]))

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None,'Uses existing JAX environment only')
    def test_actual_jax_loss_gradient_accumulation_and_extreme_padding(self):
        import jax
        import jax.numpy as jnp
        z=jnp.asarray([[.4,-.8,7.],[1.,.1,.2],[-1.,.6,99.],[-1e31,0.,0.]],jnp.float32)
        mask=jnp.asarray([[1,1,0],[1,1,1],[1,1,0],[1,0,0]],bool)
        labels=jnp.asarray([1,0,1,0]);weights=jnp.asarray([.5,3.,1.5,0.]);denominator=weights.sum()
        full=lambda x:weighted_group_loss(x,mask,labels,weights,denominator,jnp)
        actual_value,actual_gradient=jax.jit(jax.value_and_grad(full))(z)
        micro_value=0.;micro_gradient=jnp.zeros_like(z)
        for rows in ([0,2],[1,3]):
            ids=jnp.asarray(rows)
            loss=lambda x:weighted_group_loss(x[ids],mask[ids],labels[ids],weights[ids],denominator,jnp)
            value,gradient=jax.jit(jax.value_and_grad(loss))(z)
            micro_value+=value;micro_gradient+=gradient
        np.testing.assert_allclose(np.asarray(micro_value),np.asarray(actual_value),atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(np.asarray(micro_gradient),np.asarray(actual_gradient),atol=1e-7,rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(actual_gradient)[~np.asarray(mask)],0.)
        np.testing.assert_array_equal(np.asarray(actual_gradient)[3],0.)
        extreme=lambda x:weighted_group_loss(x,jnp.asarray([[True,False]]),jnp.asarray([0]),jnp.asarray([1.]),1.,jnp)
        value,gradient=jax.jit(jax.value_and_grad(extreme))(jnp.asarray([[-1e31,0.]],jnp.float32))
        self.assertEqual(float(value),0.)
        np.testing.assert_array_equal(np.asarray(gradient),np.zeros((1,2)))

    def test_weights_and_continuous_mask(self):
        selected=np.asarray([0]*19+[1]*7+[2]*3+[3]*2+[4])
        counts,weights=class_weights(selected)
        self.assertAlmostEqual(float(weights[selected].mean()),1.,places=6)
        self.assertEqual(counts.tolist(),[19,7,3,2,1])
        names=['task.op.DIG','target_tile.crop.MELON','target_tile.yield_units.present',
               'planning.final_day','target_tile.watered_today','town.BAKERY','target_tile.yield_units']
        self.assertEqual(continuous_mask(names).tolist(),[False]*5+[True,True])

    def test_numpy_checkpoint_tensor_roundtrip(self):
        rng=np.random.default_rng(409)
        p={}
        for i,(a,b) in enumerate(((373,128),(128,64),(64,1))):
            p['w'+str(i)]=(rng.standard_normal((a,b))*.01).astype(np.float32)
            p['b'+str(i)]=np.zeros(b,np.float32)
        x=rng.standard_normal((3,373))
        expected=numpy_logits(p,x)
        stream=io.BytesIO();np.savez_compressed(stream,**p);stream.seek(0)
        with np.load(stream,allow_pickle=False) as saved:
            restored={k:saved[k] for k in saved.files}
        np.testing.assert_array_equal(numpy_logits(restored,x),expected)
        bad=x.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):numpy_logits(restored,bad)

    def test_metrics_distinguish_joint_and_operation_success(self):
        ops=np.asarray([[i,(i+1)%5] for i in range(5)]).reshape(-1)
        data=dict(offsets=np.arange(0,11,2),labels=np.zeros(5,int),selected_ops=np.arange(5),
                  candidate_ops=ops,candidate_identity=np.tile([[0,1,1],[1,2,2]],(5,1)),
                  seeds=np.asarray([11,11,11,22,22]))
        initial=group_metrics(np.tile([0.,1.],5),data,np.ones(5))
        good=group_metrics(np.tile([1.,0.],5),data,np.ones(5))
        self.assertEqual(good['multicandidate_exact_top1'],1.)
        self.assertEqual(initial['cross_operation_macro_recall'],0.)
        self.assertTrue(all(metric_gate(good,initial).values()))
        self.assertEqual(good['decision_details']['unit_exact'],1.)
        self.assertEqual(good['decision_details']['tile_exact'],1.)
        self.assertEqual(good['decision_details']['operation_confusion_rows_actual_columns_predicted'],np.eye(5,dtype=int).tolist())
        self.assertEqual(set(good['by_seed']),{'11','22'})
        self.assertEqual(good['descriptive_baselines']['canonical_first']['all']['exact_top1'],1.)


if __name__=='__main__':unittest.main()
