"""In-memory CPU-only bridge/export checks; no real data or model files."""

from dataclasses import replace
from collections.abc import Mapping
import io
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest.mock import Mock, patch

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_checkpoint_v1 import snapshot
from learning.task_pair_checkpoint_v1.test_snapshot import fixture, make_archive, digest
from learning.task_pair_host_export_v1 import bridge
from learning.task_pair_orchestration_v1.core import PairResult


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = jax.devices('cpu')[0]
        cls.params, cls.mean, cls.scale, cls.metadata, cls.schema_sha = fixture()
        raw = make_archive(cls.params, cls.mean, cls.scale, cls.metadata)
        cls.checkpoint = snapshot.load_checkpoint(raw, expected_sha256=digest(raw),
                                                  expected_schema_sha256=cls.schema_sha)

    def pair(self):
        control = {name: value.copy() for name, value in self.params.items()}
        auxiliary = {name: value.copy() for name, value in self.params.items()}
        control['w0'][0, 0] += np.float32(.01)
        auxiliary['w0'][0, 0] -= np.float32(.01)
        first = MappingProxyType({k: jax.device_put(v, self.cpu) for k, v in control.items()})
        second = MappingProxyType({k: jax.device_put(v, self.cpu) for k, v in auxiliary.items()})
        return PairResult(first, second, 2, 2, True, True), control, auxiliary

    def fake_leaf(self, name, *, device=None):
        leaf = Mock(spec=jax.Array)
        leaf.shape = bridge._SHAPES[name]
        leaf.dtype = np.dtype('float32')
        leaf.devices = Mock(return_value={self.cpu if device is None else device})
        leaf.block_until_ready = Mock()
        return leaf

    def test_real_cpu_arrays_flow_into_existing_exporter(self):
        pair, control, auxiliary = self.pair()
        first, second = bridge.copy_pair_to_host(pair, device=self.cpu)
        self.assertIs(type(first), dict)
        self.assertIs(type(second), dict)
        for name in snapshot.PARAMS:
            np.testing.assert_array_equal(first[name], control[name])
            np.testing.assert_array_equal(second[name], auxiliary[name])
            self.assertEqual(first[name].dtype, np.dtype('float32'))
            self.assertFalse(np.shares_memory(first[name], second[name]))
        provenance = {'training_source': 'synthetic_only', 'initialization': 'synthetic_fixture',
                      'registration_commit': 'a' * 40, 'completed_updates': 2}
        encoded_a, encoded_b = snapshot.export_pair(self.checkpoint, first, second, provenance=provenance)
        self.assertNotEqual(encoded_a, encoded_b)
        for encoded, expected, arm in ((encoded_a, control, 'control'),
                                        (encoded_b, auxiliary, 'auxiliary')):
            loaded = snapshot.load_checkpoint(encoded, expected_sha256=digest(encoded),
                                               expected_schema_sha256=self.schema_sha)
            for name in snapshot.PARAMS:
                np.testing.assert_array_equal(loaded.params[name], expected[name])
            with np.load(io.BytesIO(encoded), allow_pickle=False) as stored:
                metadata = json.loads(str(stored['metadata']))
            self.assertEqual(metadata['pair_training']['arm'], arm)
            self.assertNotIn('SYNTHETIC_PRIVATE_MARKER', json.dumps(metadata))
        first['w0'][0, 0] = np.float32(99)
        np.testing.assert_array_equal(second['w0'], auxiliary['w0'])

    def test_missing_key_dtype_shape_and_budget_fail_before_transfer(self):
        pair, _, _ = self.pair()
        missing = dict(pair.control_params)
        del missing['b2']
        wrong_dtype = dict(pair.control_params)
        wrong_dtype['b2'] = jax.device_put(np.zeros(1, np.int32), self.cpu)
        wrong_shape = dict(pair.control_params)
        wrong_shape['b2'] = jax.device_put(np.zeros(2, np.float32), self.cpu)
        for source in (missing, wrong_dtype, wrong_shape):
            with self.subTest(kind=len(source)), patch.object(jax, 'device_get',
                    side_effect=AssertionError('Must not transfer')) as transfer:
                with self.assertRaises(ValueError):
                    bridge.copy_pair_to_host(replace(pair, control_params=MappingProxyType(source)), device=self.cpu)
                transfer.assert_not_called()
        with patch.object(bridge, 'MAX_HOST_BYTES', 1), patch.object(jax, 'device_get',
                side_effect=AssertionError('Must not transfer')) as transfer:
            with self.assertRaises(ValueError):
                bridge.copy_pair_to_host(pair, device=self.cpu)
            transfer.assert_not_called()

    def test_huge_claimed_shape_rejected_without_large_allocation_or_transfer(self):
        pair, _, _ = self.pair()
        fake = self.fake_leaf('w0')
        fake.shape = (10**9, 10**9)
        source = dict(pair.control_params)
        source['w0'] = fake
        with patch.object(jax, 'device_get', side_effect=AssertionError('Must not transfer')) as transfer:
            with self.assertRaises(ValueError):
                bridge.copy_pair_to_host(replace(pair, control_params=MappingProxyType(source)), device=self.cpu)
            transfer.assert_not_called()
        fake.block_until_ready.assert_not_called()

    def test_oversized_mapping_rejected_before_iteration_copy_or_transfer(self):
        class OversizedMapping(Mapping):
            def __len__(self):
                return 7

            def __iter__(self):
                raise AssertionError('Must not iterate an oversized mapping')

            def __getitem__(self, key):
                raise AssertionError('Must not read an oversized mapping')

        pair, _, _ = self.pair()
        invalid = replace(pair, control_params=MappingProxyType(OversizedMapping()))
        with patch.object(jax, 'device_get', side_effect=AssertionError('Must not transfer')) as transfer:
            with self.assertRaises(ValueError):
                bridge.copy_pair_to_host(invalid, device=self.cpu)
            transfer.assert_not_called()

    def test_all_leaves_are_synchronized_before_any_host_transfer(self):
        events = []
        arms = []
        for arm in range(2):
            values = {}
            for name in bridge._PARAMS:
                leaf = self.fake_leaf(name)
                leaf.block_until_ready.side_effect = lambda: events.append('synchronize')
                leaf.synthetic_arm = arm
                values[name] = leaf
            arms.append(MappingProxyType(values))
        pair = PairResult(arms[0], arms[1], 2, 2, True, True)

        def transfer(value):
            events.append('transfer')
            return np.full(value.shape, value.synthetic_arm, dtype=np.float32)

        with patch.object(jax, 'device_get', side_effect=transfer):
            first, second = bridge.copy_pair_to_host(pair, device=self.cpu)
        self.assertEqual(events[:12], ['synchronize'] * 12)
        self.assertEqual(events[12:], ['transfer'] * 12)
        self.assertFalse(np.array_equal(first['b2'], second['b2']))

    def test_synchronization_failure_prevents_all_transfers(self):
        pair, _, _ = self.pair()
        source = dict(pair.experiment_params)
        fake = self.fake_leaf('b2')
        fake.block_until_ready.side_effect = RuntimeError('synthetic sync failure')
        source['b2'] = fake
        with patch.object(jax, 'device_get', side_effect=AssertionError('Must not transfer')) as transfer:
            with self.assertRaises(RuntimeError):
                bridge.copy_pair_to_host(replace(pair, experiment_params=MappingProxyType(source)), device=self.cpu)
            transfer.assert_not_called()

    def test_invalid_metadata_nonfinite_and_identical_arms_fail(self):
        pair, _, _ = self.pair()
        for changes in ({'completed_updates': True}, {'completed_updates': 0},
                        {'auxiliary_nonzero_updates': 0}, {'auxiliary_nonzero_updates': 3},
                        {'arms_differ': False}, {'both_arms_changed': False},
                        {'control_params': dict(pair.control_params)}):
            with self.subTest(field=tuple(changes)), self.assertRaises(ValueError):
                bridge.copy_pair_to_host(replace(pair, **changes), device=self.cpu)
        with self.assertRaises(ValueError):
            bridge.copy_pair_to_host(replace(pair, experiment_params=pair.control_params), device=self.cpu)
        invalid = dict(pair.control_params)
        invalid['b2'] = jax.device_put(np.asarray([np.inf], dtype=np.float32), self.cpu)
        with self.assertRaises(ValueError):
            bridge.copy_pair_to_host(replace(pair, control_params=MappingProxyType(invalid)), device=self.cpu)

    def test_real_different_device_is_rejected_before_transfer(self):
        devices = jax.devices('cpu')
        if len(devices) < 2:
            self.skipTest('Second virtual CPU device unavailable')
        pair, _, _ = self.pair()
        with patch.object(jax, 'device_get', side_effect=AssertionError('Must not transfer')) as transfer:
            with self.assertRaises(ValueError):
                bridge.copy_pair_to_host(pair, device=devices[1])
            transfer.assert_not_called()

    def test_bad_host_return_cannot_reach_exporter(self):
        pair, _, _ = self.pair()
        with patch.object(jax, 'device_get', return_value=np.zeros(2, np.float32)):
            with self.assertRaises(ValueError):
                bridge.copy_pair_to_host(pair, device=self.cpu)


if __name__ == '__main__':
    unittest.main()
