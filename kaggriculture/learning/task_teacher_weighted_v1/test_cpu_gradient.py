"""Synthetic CPU-only regression of the actual weighted nested fit loss."""
import unittest
from unittest.mock import patch
import numpy as np
import jax
import jax.numpy as jnp
import optax
from learning.task_choice_v1.features import OPS,schema_hash
from . import training


class StopAfterGradient(Exception): pass


class GradientTest(unittest.TestCase):
    def test_actual_nested_loss_and_last_layer_gradient_on_cpu(self):
        cpu=jax.devices('cpu')[0]
        self.assertEqual(cpu.platform,'cpu')
        class IdentityAdapter:
            platform='gpu'; device_kind='synthetic-identity-only'
            def __str__(self): return 'synthetic-identity-only'
        adapter=IdentityAdapter()
        data=[]
        for index,op in enumerate(('DIG','DIG',*OPS[1:])):
            candidates=np.zeros((2,98),np.float32)
            candidates[0,0]=index+1; candidates[1,1]=index+2
            data.append(dict(shared=np.zeros(275,np.float32),candidates=candidates,
                teacher_choice_label=0 if op=='DIG' else 1,candidate_ops=('DIG',op if op!='DIG' else 'WATER'),seed=1))
        names=[f'f{i}' for i in range(98)]; shared=[f's{i}' for i in range(275)]
        schema=dict(feature_names=names,shared_feature_names=shared,schema_sha256=schema_hash(names,shared))
        real_devices,real_put,real_vg=jax.devices,jax.device_put,jax.value_and_grad
        captured={}
        def devices(platform=None): return [adapter] if platform=='gpu' else real_devices(platform)
        def put(value,device=None,*args,**kwargs): return real_put(value,cpu)
        def vg(loss):
            calculate=real_vg(loss)
            def run(params,x,valid,labels,weights):
                value,gradient=calculate(params,x,valid,labels,weights)
                captured.update(params=params,x=x,valid=valid,labels=labels,weights=weights,value=value,gradient=gradient)
                raise StopAfterGradient()
            return run
        with jax.default_device(cpu),patch.object(jax,'devices',side_effect=devices), \
             patch.object(jax,'device_put',side_effect=put),patch.object(jax,'jit',side_effect=lambda fn:fn), \
             patch.object(jax,'value_and_grad',side_effect=vg):
            with self.assertRaises(StopAfterGradient):
                training.fit(data,schema,'unused.npz',1,str(adapter),adapter.device_kind)
        params={k:np.asarray(v,dtype=np.float64) for k,v in captured['params'].items()}
        x=np.asarray(captured['x'],dtype=np.float64); valid=np.asarray(captured['valid'],bool)
        labels=np.asarray(captured['labels']); weights=np.asarray(captured['weights'],dtype=np.float64)
        self.assertTrue(all(v.device.platform=='cpu' for v in captured['params'].values()))
        self.assertEqual(np.count_nonzero(weights),6); self.assertTrue(np.all(weights[6:]==0))
        h=np.maximum(x@params['w0']+params['b0'],0)
        h=np.maximum(h@params['w1']+params['b1'],0)
        scores=(h@params['w2']+params['b2'])[...,0]
        masked=np.where(valid,scores,-np.inf); maximum=np.max(masked,axis=1,keepdims=True)
        exps=np.exp(masked-maximum); probabilities=exps/exps.sum(axis=1,keepdims=True)
        loss=np.sum((np.log(exps.sum(axis=1))+maximum[:,0]-scores[np.arange(len(labels)),labels])*weights)/weights.sum()
        ds=probabilities.copy(); ds[np.arange(len(labels)),labels]-=1
        ds*=weights[:,None]/weights.sum()
        grad_w=np.einsum('bnh,bn->h',h,ds)[:,None]
        grad_b=np.array([ds.sum()])
        self.assertAlmostEqual(float(captured['value']),float(loss),places=5)
        np.testing.assert_allclose(np.asarray(captured['gradient']['w2']),grad_w,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(np.asarray(captured['gradient']['b2']),grad_b,rtol=2e-5,atol=2e-6)


if __name__=='__main__': unittest.main()
