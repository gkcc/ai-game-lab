import unittest
from unittest.mock import patch
import sys
import types
import numpy as np

from learning.task_bc_v1.model import TaskNetwork, DIMENSIONS
from learning.task_completion_parity_v1.compare import compare, prepare, summarize, validate_device
from learning.task_choice_v1.features import OPS


def model():
    m = TaskNetwork.__new__(TaskNetwork)
    m.mean, m.scale = np.zeros(373), np.ones(373)
    m.params = {k: np.zeros(shape) for i in range(3) for k, shape in
                ((f'w{i}', (DIMENSIONS[i], DIMENSIONS[i+1])), (f'b{i}', (DIMENSIONS[i+1],)))}
    return m


def group(rows=1):
    return dict(shared=np.zeros(275, np.float32), candidates=np.zeros((rows, 98), np.float32),
                chosen=0, candidate_ops=[OPS[0]] * rows, seed=1, status='completed_observed')


class CompareTest(unittest.TestCase):
    def test_progress_without_jax(self):
        class Device:
            platform = 'gpu'
            device_kind = 'synthetic'
            def __str__(self): return 'synthetic'
        device = Device()
        class Array:
            def __init__(self, value): self.value, self.device = value, device
            def __matmul__(self, other): return Array(self.value @ other.value)
            def __add__(self, other): return Array(self.value + other.value)
            def block_until_ready(self): return self
            def __getitem__(self, key): return self.value[key]
        fake = types.SimpleNamespace(devices=lambda _: [device],
            device_put=lambda x, _: Array(x), nn=types.SimpleNamespace(relu=lambda x: Array(np.maximum(x.value, 0))))
        events = []
        groups = [group() for _ in range(16)]
        with patch.dict(sys.modules, {'jax': fake}):
            baseline = compare(groups, model(), 'synthetic', 'synthetic')
            result = compare(groups, model(), 'synthetic', 'synthetic', events.append)
        self.assertEqual(result, baseline)
        self.assertTrue(result['parity_passed'])
        self.assertEqual(events[0], dict(stage='prepared', started_groups=0, completed_groups=0, candidate_rows=0))
        self.assertEqual(events[1]['stage'], 'device_validated')
        self.assertEqual(events[2], dict(stage='forward_started', started_groups=1, completed_groups=0, candidate_rows=0))
        self.assertEqual(events[-1], dict(stage='forward_completed', started_groups=16, completed_groups=16, candidate_rows=16))
        self.assertEqual(len(events), 34)

    def test_tolerance_and_argmax(self):
        result = summarize([(np.array([10.00019, 0.00011]), np.array([10., 0.])),
                            (np.array([1., 1.]), np.array([1., 1.]))])
        self.assertEqual(result['out_of_tolerance_candidates'], 1)
        self.assertEqual(result['argmax_different_groups'], 0)
        self.assertFalse(result['parity_passed'])
        self.assertEqual(summarize([([1., 1.], [1., 1.000001])])['argmax_different_groups'], 1)

    def test_invalid_logits(self):
        for pairs in ([], [([], [])], [([np.nan], [0.])], [([0.], [np.inf])],
                      [(np.zeros(8193), np.zeros(8193))]):
            with self.assertRaises(ValueError): summarize(pairs)

    def test_group_budget_and_identity(self):
        groups = [group() for _ in range(16)]
        self.assertEqual(len(prepare(groups, model())), 16)
        groups[0]['seed'] = -1
        with self.assertRaises(ValueError): prepare(groups, model())
        with self.assertRaises(ValueError): prepare([group(8178)] + [group() for _ in range(15)], model())
        with self.assertRaises(ValueError): prepare([group(0)] + [group() for _ in range(15)], model())
        with self.assertRaises(ValueError): prepare([group() for _ in range(15)], model())
        class Device:
            platform = 'gpu'
            device_kind = 'kind'
            def __str__(self): return 'device'
        validate_device(Device(), 'device', 'kind')
        with self.assertRaises(RuntimeError): validate_device(Device(), 'other', 'kind')
        with self.assertRaises(RuntimeError): validate_device(Device(), 'device', 'other')
        Device.platform = 'cpu'
        with self.assertRaises(RuntimeError): validate_device(Device(), 'device', 'kind')


if __name__ == '__main__': unittest.main()
