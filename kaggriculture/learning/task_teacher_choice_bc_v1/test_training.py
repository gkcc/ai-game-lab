import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

from learning.task_choice_v1.features import schema_hash
from learning.task_teacher_choice_bc_v1 import training


def example():
    return dict(shared=np.zeros(275,np.float32),candidates=np.zeros((2,98),np.float32),
                teacher_choice_label=1,candidate_ops=('DIG','WATER'),seed=17)


class TeacherTrainingTest(unittest.TestCase):
    def test_label_is_metadata_and_not_completion(self):
        data = example()
        group = training.as_groups([data])[0]
        training.validate([group])
        before = training.inputs(group)
        group.teacher_choice_label = 0
        np.testing.assert_array_equal(before,training.inputs(group))
        self.assertFalse(hasattr(group,'status'))
        counts = training.baseline_fit([group])
        self.assertEqual(counts['DIG'],2.)
        self.assertEqual(counts['WATER'],1.)
        with self.assertRaises(TypeError):
            training.as_groups([{**data,'status':'completed_observed'}])
        for bad in (True,-1,2):
            with self.assertRaises(ValueError):
                training.validate(training.as_groups([{**data,'teacher_choice_label':bad}]))

    def test_gpu_entry_uses_teacher_group_and_reports_progress(self):
        class Device:
            platform,device_kind = 'gpu','synthetic'
            def __str__(self): return 'synthetic'
        device = Device()
        jax = types.ModuleType('jax')
        jax.devices = lambda _: [device]
        jax.device_put = lambda x, _: x
        def jit(_):
            def stop(*args): raise RuntimeError('synthetic stop before gradient')
            return stop
        jax.jit = jit
        jax.numpy = types.ModuleType('jax.numpy')
        optax = types.ModuleType('optax')
        optax.clip_by_global_norm = lambda _:None
        optax.adam = lambda _:None
        optax.chain = lambda *args:types.SimpleNamespace(init=lambda _:None)
        names = ['f'+str(i) for i in range(98)]
        shared = ['s'+str(i) for i in range(275)]
        schema = dict(feature_names=names,shared_feature_names=shared,schema_sha256=schema_hash(names,shared))
        events = []
        with patch.dict(sys.modules,{'jax':jax,'jax.numpy':jax.numpy,'optax':optax}):
            with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                training.fit([example()],schema,'unused.npz',1,'synthetic','synthetic',events.append)
        self.assertEqual(events[-1]['phase'],'update_started')
        self.assertEqual(events[-1]['updates_started'],1)
        self.assertEqual(events[-1]['updates_completed'],0)
        self.assertTrue(events[-1]['device_validated'])


if __name__ == '__main__': unittest.main()
