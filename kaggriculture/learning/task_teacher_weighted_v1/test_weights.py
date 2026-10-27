import math
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from learning.task_choice_v1.features import OPS,schema_hash
from .weights import operation_weights
from . import training


def group(op,label=1):
    return dict(shared=np.zeros(275,np.float32),candidates=np.zeros((2,98),np.float32),
                teacher_choice_label=label,candidate_ops=('DIG',op),seed=1)


class WeightsTest(unittest.TestCase):
    def test_known_counts_cap_and_invalid(self):
        rows=[group('WATER',0)]*100+[group(op) for op in OPS[1:]]
        weights=operation_weights(training.as_groups(rows))
        self.assertAlmostEqual(weights['DIG'],math.sqrt(104/(5*100)))
        self.assertEqual(weights['WATER'],2.)
        for bad in ([],training.as_groups([group('WATER')]),training.as_groups([group('WATER',2)]),
                    training.as_groups([group('INVALID')])):
            with self.assertRaises(ValueError): operation_weights(bad)

    def test_first_real_batch_uses_selected_operation_and_zero_padding(self):
        class Device:
            platform,device_kind='gpu','synthetic'
            def __str__(self): return 'synthetic'
        device=Device(); captured=[]
        jax=types.ModuleType('jax'); jax.devices=lambda _:[device]; jax.device_put=lambda x,_:x
        def jit(_):
            def step(*args): captured.append((args[-2].copy(),args[-1].copy())); raise RuntimeError('stop before gradient')
            return step
        jax.jit=jit; jax.numpy=types.ModuleType('jax.numpy')
        optax=types.ModuleType('optax'); optax.clip_by_global_norm=lambda _:None; optax.adam=lambda _:None
        optax.chain=lambda *args:types.SimpleNamespace(init=lambda _:None)
        names=[f'f{i}' for i in range(98)]; shared=[f's{i}' for i in range(275)]
        schema=dict(feature_names=names,shared_feature_names=shared,schema_sha256=schema_hash(names,shared))
        data=[group('WATER',0),group('WATER',0)]+[group(op) for op in OPS[1:]]
        with patch.dict(sys.modules,{'jax':jax,'jax.numpy':jax.numpy,'optax':optax}):
            with self.assertRaisesRegex(RuntimeError,'stop before gradient'):
                training.fit(data,schema,'unused.npz',1,'synthetic','synthetic')
        labels,weights=captured[0]
        self.assertEqual(len(weights),8)
        self.assertEqual(np.count_nonzero(weights),6)
        for label,weight in zip(labels[:6],weights[:6]):
            self.assertAlmostEqual(float(weight),math.sqrt(6/10) if label==0 else math.sqrt(6/5),places=6)
        self.assertTrue(np.all(weights[6:]==0))


if __name__=='__main__': unittest.main()
