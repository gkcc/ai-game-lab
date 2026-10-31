"""Synthetic worker regressions: no private data, real GPU, fitting or game."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_bc_v1.model import DIMENSIONS
from learning.task_choice_v1.features import OPS
from learning.task_pair_checkpoint_v1.snapshot import PARAMS, SHAPES
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_teacher_transfer_dataset_v1.test_dataset import make_groups
from learning.task_water_residual_v1 import gpu_worker as worker
from learning.task_water_residual_v1.test_policy import metadata_and_inputs


def archive(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class SyntheticNetwork:
    def __init__(self, sha, metadata, arrays, scores=None):
        self.sha, self.metadata = sha, deepcopy(metadata)
        self.params = {k: arrays[k].astype(np.float64) for k in PARAMS}
        self.mean, self.scale = arrays['mean'].copy(), arrays['scale'].copy()
        self.scores = scores

    def score(self, shared, candidates):
        if self.scores is not None:
            return np.asarray(self.scores, dtype=np.float64).copy()
        return -np.asarray(candidates, dtype=np.float64)[:, 0]


class SyntheticBackend:
    def __init__(self):
        self.called = 0
        self.corrupt = False

    def validate(self, device, kind):
        return device is self and kind == 'synthetic'

    def train(self, groups, *, mean, scale, device, guard, rng_seed):
        self.called += 1
        assert device is self and {g['role'] for g in groups} == {0, 1}
        weights = np.zeros(373, dtype=np.float32)
        weights[275] = 0.01
        return SimpleNamespace(weights=weights, updates_completed=64,
                               nonzero_gradient_seen=True, weights_changed=True)

    def residual(self, normalized, weights, *, device, guard):
        assert device is self
        result = np.sum((normalized - normalized[:1]) * weights, axis=1, dtype=np.float32)
        if self.corrupt and len(result) > 1:
            result[-1] += np.float32(1)
        return result


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, _ = metadata_and_inputs()
        arrays = {name: np.zeros(SHAPES[name], dtype=np.float32) for name in PARAMS}
        arrays['mean'] = np.zeros(373, dtype=np.float64)
        arrays['scale'] = np.ones(373, dtype=np.float64)
        cls.arrays = arrays
        cls.baseline = archive(dict(arrays, metadata=np.asarray(json.dumps(cls.metadata))))
        train = make_groups([('A01', 29), ('A02', 30)], one_operation=True)
        dev = make_groups([('A07', 31), ('A08', 31)], start=200, one_operation=True)
        cls.train = archive(transfer._split_arrays(train, lambda: None))
        cls.dev = archive(transfer._split_arrays(dev, lambda: None))
        cls.binding = dict(schema='water-residual-private-binding-v1',
            train_sha256=digest(cls.train), dev_sha256=digest(cls.dev),
            baseline_sha256=digest(cls.baseline), schema_sha256=cls.metadata['schema_sha256'],
            device_kind='synthetic', rng_seed=9)
        cls.registration = dict(schema='water-residual-gpu-prereg-v1', train_source_groups=59,
            dev_source_groups=62, training_operations=['WATER'], route='baseline_global_water_then_water_only',
            training_roles_per_batch=8, updates=64, learning_rate=0.0001, kl_coefficient=0.1,
            clip_norm=1.0, rng_seed=9, float32_training_baseline_argmax_must_match=True,
            cpu_gpu_atol=0.0001, cpu_gpu_rtol=0.00001, cpu_gpu_first_argmax_must_match=True)
        cls.commit = 'a' * 40

    def setup_execution(self):
        backend = SyntheticBackend()
        network = SyntheticNetwork(self.binding['baseline_sha256'], self.metadata, self.arrays)
        kwargs = dict(train_raw=self.train, dev_raw=self.dev, baseline_raw=self.baseline,
            registration=deepcopy(self.registration), binding=deepcopy(self.binding),
            registration_commit=self.commit, guard=lambda: None, device=backend,
            network_loader=lambda _raw, _sha: network, backend=backend)
        return kwargs, network, backend

    def test_valid_complete_in_memory_pipeline_and_private_summary(self):
        kwargs, _network, backend = self.setup_execution()
        weights, result = worker.execute(**kwargs)
        self.assertEqual(backend.called, 1)
        self.assertEqual((result['training']['input_groups'], result['training']['selected_groups']), (59, 59))
        self.assertEqual(result['training']['selected_by_role'], {'A01': 29, 'A02': 30})
        self.assertEqual(result['development']['groups'], 62)
        self.assertEqual(result['parity']['groups'], 62)
        self.assertTrue(result['parity']['allclose'])
        self.assertEqual(result['parity']['adjusted_argmax_mismatches'], 0)
        self.assertEqual(result['parity']['operation_flips'], 0)
        self.assertFalse(result['independent_competitive_evaluation'])
        self.assertEqual(weights.dtype, np.float32)
        self.assertFalse(weights.flags.writeable)
        serialized = json.dumps(result)
        for forbidden in ('synthetic', 'device_kind', 'rng_seed', 'features', 'logits'):
            self.assertNotIn(forbidden, serialized)

    def test_bound_identity_and_device_reject_before_training(self):
        for field in ('train_sha256', 'dev_sha256', 'baseline_sha256', 'schema_sha256', 'device_kind'):
            kwargs, _network, backend = self.setup_execution()
            kwargs['binding'][field] = 'b' * 64 if field != 'device_kind' else 'other'
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.execute(**kwargs)
            self.assertEqual(backend.called, 0)

    def test_model_snapshot_mismatch_and_float32_argmax_stop_before_training(self):
        kwargs, network, backend = self.setup_execution()
        network.params['b0'][0] = 1
        with self.assertRaises(ValueError): worker.execute(**kwargs)
        self.assertEqual(backend.called, 0)
        kwargs, network, backend = self.setup_execution()
        network.scores = [1.0, 1.0 + 1e-8, 0.5]
        with self.assertRaises(ValueError): worker.execute(**kwargs)
        self.assertEqual(backend.called, 0)

    def test_full_roster_cast_change_stops_even_when_water_subranking_is_stable(self):
        from learning.task_teacher_transfer_dataset_v1 import dataset
        from learning.task_water_residual_v1 import assessment, snapshot_data
        groups = make_groups([('A01', 29), ('A02', 30)], one_operation=True)
        changed = []
        for group in groups:
            features = group.features.copy()
            features[0, dataset._OP_START + OPS.index('WATER')] = 0
            features[0, dataset._OP_START + OPS.index('DIG')] = 1
            changed.append(replace(group, features=features, target_index=1))
        raw = archive(transfer._split_arrays(changed, lambda: None))
        kwargs, network, backend = self.setup_execution()
        kwargs['train_raw'], kwargs['binding']['train_sha256'] = raw, digest(raw)
        network.scores = [1.0, 1.0 + 1e-8, 0.5]
        decoded = snapshot_data.read_split(raw, digest(raw), 'train', guard=lambda: None)
        selected, report = assessment.prepare_training(decoded, network, self.metadata, guard=lambda: None)
        self.assertEqual(len(selected), 59)
        self.assertEqual(report['float32_baseline_argmax_differences'], 0)
        self.assertEqual(worker._global_cast_differences(decoded, network, self.metadata, lambda: None), (59, 59))
        with self.assertRaises(ValueError): worker.execute(**kwargs)
        self.assertEqual(backend.called, 0)

    def test_gpu_disagreement_aborts_after_training_without_a_result(self):
        kwargs, _network, backend = self.setup_execution()
        backend.corrupt = True
        with self.assertRaises(ValueError): worker.execute(**kwargs)
        self.assertEqual(backend.called, 1)

    def test_variable_candidate_parity_uses_one_fixed_gpu_shape(self):
        device = object()
        shapes = []
        class Array:
            def __init__(self, value): self.value = value
            def __getitem__(self, index): return Array(self.value[index])
            def __sub__(self, other): return Array(self.value - other.value)
            def __mul__(self, other): return Array(self.value * other.value)
            def devices(self): return {device}
            def block_until_ready(self): return self
        class Jax:
            @staticmethod
            def default_device(_device): return nullcontext()
            @staticmethod
            def default_matmul_precision(_precision): return nullcontext()
            @staticmethod
            def device_put(value, _device):
                shapes.append(value.shape)
                return Array(value)
            @staticmethod
            def device_get(value): return value.value
        Jax.Array = Array
        class Jnp:
            @staticmethod
            def sum(value, axis): return Array(np.sum(value.value, axis=axis, dtype=np.float32))
        backend = object.__new__(worker.JaxBackend)
        backend.jax, backend.jnp = Jax, Jnp
        weights = np.full(373, 0.001, dtype=np.float32)
        for count in (1, 7, 256):
            features = np.broadcast_to(np.arange(count, dtype=np.float32)[:, None], (count, 373)).copy()
            actual = backend.residual(features, weights, device=device, guard=lambda: None)
            wanted = np.sum((features - features[:1]) * weights, axis=1, dtype=np.float32)
            np.testing.assert_array_equal(actual, wanted)
        self.assertEqual(shapes[::2], [(256, 373)] * 3)
        self.assertEqual(shapes[1::2], [(373,)] * 3)
        padded, count = worker._pad_normalized(np.ones((2, 373), dtype=np.float32))
        self.assertEqual(count, 2)
        self.assertTrue(np.all(padded[2:] == 1))
        for bad in (np.zeros((0, 373), dtype=np.float32), np.zeros((257, 373), dtype=np.float32),
                    np.zeros((1, 373), dtype=np.float64)):
            with self.assertRaises(ValueError): worker._pad_normalized(bad)

    def test_disk_and_exclusive_output_contract(self):
        kwargs, _network, _backend = self.setup_execution()
        weights, result = worker.execute(**kwargs)
        stage = Path('synthetic-stage')
        files = {stage / 'baseline.npz': self.baseline}
        def read(path, limit, guard):
            guard(); return files[path]
        def write(path, raw):
            if path in files: raise FileExistsError()
            files[path] = raw
        calls = []
        def disk(_stage): calls.append(True)
        saved = worker.save_outputs(stage, weights, result, self.baseline, guard=lambda: None,
                                    disk_guard=disk, read_file=read, write_once=write)
        self.assertEqual(len(calls), 2)
        self.assertTrue(saved['baseline_unchanged'])
        self.assertEqual(saved['residual_sha256'], digest(files[stage / 'residual.local.npz']))
        self.assertEqual(json.loads(files[stage / 'gpu-worker-result.local.json']), saved)
        with self.assertRaises(FileExistsError):
            worker.save_outputs(stage, weights, result, self.baseline, guard=lambda: None,
                                disk_guard=disk, read_file=read, write_once=write)
        fresh = {stage / 'baseline.npz': self.baseline}
        files = fresh
        def no_disk(_stage): raise OSError('floor')
        with self.assertRaises(OSError):
            worker.save_outputs(stage, weights, result, self.baseline, guard=lambda: None,
                                disk_guard=no_disk, read_file=read, write_once=write)
        self.assertNotIn(stage / 'residual.local.npz', files)

    def test_second_disk_check_and_baseline_mutation_fail_closed(self):
        kwargs, _network, _backend = self.setup_execution()
        weights, result = worker.execute(**kwargs)
        stage = Path('synthetic-stage')
        for fail_disk in (True, False):
            files = {stage / 'baseline.npz': self.baseline if fail_disk else b'changed'}
            calls = []
            def read(path, limit, guard): return files[path]
            def write(path, raw): files[path] = raw
            def disk(_stage):
                calls.append(True)
                if fail_disk and len(calls) == 2: raise OSError('floor')
            with self.subTest(fail_disk=fail_disk), self.assertRaises((ValueError, OSError)):
                worker.save_outputs(stage, weights, result, self.baseline, guard=lambda: None,
                                    disk_guard=disk, read_file=read, write_once=write)
            self.assertNotIn(stage / 'gpu-worker-result.local.json', files)

    def test_single_owner_marker_and_live_parent(self):
        stage, nonce = Path('synthetic-stage'), 'b' * 32
        host = dict(schema='water-residual-host-attempt-v1', commit=self.commit, retry_allowed=False)
        supervisor = dict(schema='water-residual-supervisor-attempt-v1', registration_commit=self.commit,
                          nonce=nonce, supervisor_pid=102, supervisor_starttime=444, retry_allowed=False)
        files = {stage / 'host-attempt.local.json': json.dumps(host).encode(),
                 stage / 'gpu-supervisor-attempt.local.json': json.dumps(supervisor).encode()}
        alive = [True]
        def read(path, limit, guard): guard(); return files[path]
        def write(path, raw):
            if path in files: raise FileExistsError()
            files[path] = raw
        def stat(pid): return SimpleNamespace(pid=pid, starttime=444, state=b'S' if alive[0] else b'Z')
        args = dict(guard=lambda: None, read_file=read, write_once=write, exists=lambda p: p in files,
                    pid=lambda: 101, parent=lambda: 102, session=lambda _: 101, parent_stat=stat)
        check = worker._claim(stage, self.commit, nonce, **args)
        self.assertEqual(json.loads(files[stage / 'gpu-worker-attempt.local.json'])['nonce'], nonce)
        with self.assertRaises(ValueError): worker._claim(stage, self.commit, nonce, **args)
        alive[0] = False
        with self.assertRaises(ValueError): check()

    def test_forged_owner_and_private_argument_error_fail_closed(self):
        stage, nonce = Path('synthetic-stage'), 'b' * 32
        host = dict(schema='water-residual-host-attempt-v1', commit=self.commit, retry_allowed=False)
        supervisor = dict(schema='water-residual-supervisor-attempt-v1', registration_commit=self.commit,
                          nonce=nonce, supervisor_pid=102, supervisor_starttime=444, retry_allowed=False)
        files = {stage / 'host-attempt.local.json': json.dumps(host).encode(),
                 stage / 'gpu-supervisor-attempt.local.json': json.dumps(supervisor).encode()}
        args = dict(guard=lambda: None, read_file=lambda p, _n, _g: files[p],
                    write_once=lambda p, v: files.__setitem__(p, v), exists=lambda p: p in files,
                    pid=lambda: 101, parent=lambda: 102, session=lambda _: 101,
                    parent_stat=lambda p: SimpleNamespace(pid=p, starttime=444, state=b'S'))
        with self.assertRaises(ValueError): worker._claim(stage, self.commit, 'c' * 32, **args)
        self.assertNotIn(stage / 'gpu-worker-attempt.local.json', files)
        with self.assertRaisesRegex(ValueError, '^Invalid private worker arguments$'):
            worker.main(['--stage', 'PRIVATE_MARKER', '--unexpected', 'PRIVATE_MARKER'])

    def test_registration_and_binding_must_remain_identical(self):
        calls = []
        def guard(): calls.append('guard')
        def verify(commit, stage):
            calls.append('verify')
            return deepcopy(self.registration), deepcopy(self.binding)
        worker._verify_unchanged(verify, self.commit, Path('synthetic-stage'),
                                 self.registration, self.binding, guard)
        self.assertEqual(calls, ['guard', 'verify', 'guard'])
        for registration, binding in ((dict(self.registration, updates=63), self.binding),
                                      (self.registration, dict(self.binding, rng_seed=10))):
            with self.assertRaises(ValueError):
                worker._verify_unchanged(lambda *_: (registration, binding), self.commit,
                                         Path('synthetic-stage'), self.registration, self.binding, guard)


if __name__ == '__main__':
    unittest.main()
