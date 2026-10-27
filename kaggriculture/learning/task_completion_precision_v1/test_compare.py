import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

from learning.task_completion_precision_v1.compare import compare
from learning.task_completion_parity_v1.test_compare import group, model
from learning.task_completion_parity_v1.compare import summarize


class PrecisionTest(unittest.TestCase):
    def test_both_settings_and_cpu_once(self):
        class Device:
            platform, device_kind = 'gpu', 'synthetic'
            def __str__(self): return 'synthetic'
        device = Device()
        state = {'precision': 'default', 'forwards': [], 'inputs': [], 'contexts': []}
        class Array:
            def __init__(self, value): self.value, self.device = value, device
            def __matmul__(self, other): return Array(self.value @ other.value)
            def __add__(self, other): return Array(self.value + other.value)
            def block_until_ready(self):
                state['forwards'].append(state['precision'])
                return self
            def __getitem__(self, key): return self.value[key]
        class Context:
            def __enter__(self):
                state['contexts'].append('highest')
                state['precision'] = 'highest'
            def __exit__(self, *args): state['precision'] = 'default'
        def put(x, _):
            if x.ndim == 2 and x.shape[1] == 373:
                state['inputs'].append(x.copy())
            return Array(x)
        fake = types.SimpleNamespace(devices=lambda _: [device], device_put=put,
            nn=types.SimpleNamespace(relu=lambda x: Array(np.maximum(x.value, 0))),
            config=types.SimpleNamespace(jax_default_matmul_precision=None),
            default_matmul_precision=lambda name: Context() if name == 'highest' else self.fail('wrong precision'))
        groups = [group(108) for _ in range(15)] + [group(121)]
        net = model()
        original = net.score
        calls = []
        def score(shared, candidates):
            calls.append(len(candidates))
            return original(shared, candidates)
        net.score = score
        events = []
        with patch.dict(sys.modules, {'jax': fake}):
            result = compare(groups, net, 'synthetic', 'synthetic', events.append)
        self.assertEqual(len(calls), 16)
        self.assertEqual(state['contexts'], ['highest'])
        self.assertEqual(state['forwards'], ['default'] * 16 + ['highest'] * 16)
        for a, b in zip(state['inputs'][:16], state['inputs'][16:]): np.testing.assert_array_equal(a, b)
        self.assertTrue(result['default']['parity_passed'])
        self.assertTrue(result['highest']['parity_passed'])
        self.assertEqual(result['completed_gpu_rows'], 3482)
        self.assertEqual(events[-1]['completed_gpu_groups'], 32)
        self.assertEqual(events[-1]['started_gpu_groups'], 32)
        gpu_starts = [e for e in events if e['stage'] == 'gpu_started']
        self.assertEqual([e['setting'] for e in gpu_starts], ['default']*16+['highest']*16)
        self.assertEqual([(e['started_gpu_groups'],e['completed_gpu_groups']) for e in gpu_starts],
                         [(i,i-1) for i in range(1,33)])

    def test_budget_fails_before_jax(self):
        with self.assertRaises(ValueError): compare([group() for _ in range(16)], model(), 'a', 'b')

    def test_cpu_relative_tolerance_and_ties(self):
        result = summarize([([10.00019, 0.00011], [10., 0.]), ([1., 1.], [1., 1.000001])])
        self.assertEqual(result['out_of_tolerance_candidates'], 1)
        self.assertEqual(result['argmax_different_groups'], 1)
        self.assertAlmostEqual(result['max_absolute_error'], 0.00019)

    def test_incomplete_first_highest_forward_is_counted(self):
        class Device:
            platform, device_kind = 'gpu', 'synthetic'
            def __str__(self): return 'synthetic'
        device = Device()
        state = {'completed':0}
        class Array:
            def __init__(self,value): self.value,self.device=value,device
            def __matmul__(self,other): return Array(self.value @ other.value)
            def __add__(self,other): return Array(self.value + other.value)
            def block_until_ready(self):
                if state['completed']==16: raise RuntimeError('synthetic interrupted forward')
                state['completed']+=1
                return self
            def __getitem__(self,key): return self.value[key]
        class Context:
            def __enter__(self): return None
            def __exit__(self,*args): return False
        fake = types.SimpleNamespace(devices=lambda _: [device],device_put=lambda x,_:Array(x),
            nn=types.SimpleNamespace(relu=lambda x:Array(np.maximum(x.value,0))),
            config=types.SimpleNamespace(jax_default_matmul_precision=None),
            default_matmul_precision=lambda _:Context())
        groups = [group(108) for _ in range(15)] + [group(121)]
        events = []
        with patch.dict(sys.modules,{'jax':fake}), self.assertRaises(RuntimeError):
            compare(groups,model(),'synthetic','synthetic',events.append)
        self.assertEqual(events[-1]['stage'],'gpu_started')
        self.assertEqual(events[-1]['setting'],'highest')
        self.assertEqual(events[-1]['started_gpu_groups'],17)
        self.assertEqual(events[-1]['completed_gpu_groups'],16)


if __name__ == '__main__': unittest.main()
