"""In-memory synthetic checkpoint tests; no real model, GPU or filesystem."""

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_bc_v1 import model
from learning.task_pair_checkpoint_v1 import snapshot


def fixture():
    rng = np.random.default_rng(9)  # Synthetic values only.
    shared = ['planning.final_day'] + ['shared.' + str(i) for i in range(274)]
    local = ['task.op.' + op for op in model.OPS] + ['candidate.' + str(i) for i in range(93)]
    schema_sha = model.schema_hash(local, shared)
    metadata = {'schema_version': model.SCHEMA_VERSION, 'schema_sha256': schema_sha,
                'shared_feature_names': shared, 'feature_names': local, 'ops': list(model.OPS),
                'dimensions': list(model.DIMENSIONS), 'model_seed': 123456,
                'local_path': 'SYNTHETIC_PRIVATE_MARKER'}
    params = {name: (rng.standard_normal(snapshot.SHAPES[name]) * .01).astype(np.float32)
              for name in snapshot.PARAMS}
    mean = rng.standard_normal(373).astype(np.float64)
    scale = (1.0 + rng.random(373)).astype(np.float64)
    noncontinuous = ~model.continuous_mask(shared + local)
    mean[noncontinuous], scale[noncontinuous] = 0.0, 1.0
    mean[1] = -0.0  # Check signed-zero preservation for a continuous feature.
    return params, mean, scale, metadata, schema_sha


def make_archive(params, mean, scale, metadata):
    stream = io.BytesIO()
    np.savez_compressed(stream, **params, mean=mean, scale=scale,
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    return stream.getvalue()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_members(raw, *, replace=None, duplicate=None, extra=None):
    with zipfile.ZipFile(io.BytesIO(raw), 'r') as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]
    stream = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        warnings.simplefilter('ignore', UserWarning)
        for name, payload in members:
            out.writestr(name, replace.get(name, payload) if replace else payload)
            if name == duplicate:
                out.writestr(name, payload)
        if extra:
            out.writestr(extra, b'synthetic')
    return stream.getvalue()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.params, self.mean, self.scale, self.metadata, self.schema_sha = fixture()
        self.raw = make_archive(self.params, self.mean, self.scale, self.metadata)

    def load(self, raw=None, *, sha=None, schema_sha=None):
        raw = self.raw if raw is None else raw
        return snapshot.load_checkpoint(raw, expected_sha256=digest(raw) if sha is None else sha,
                                        expected_schema_sha256=self.schema_sha if schema_sha is None else schema_sha)

    def test_valid_snapshot_preserves_weights_and_normalization_values(self):
        loaded = self.load()
        self.assertEqual(repr(loaded), '<PrivateTaskCheckpoint>')
        for name in snapshot.PARAMS:
            np.testing.assert_array_equal(loaded.params[name], self.params[name])
            self.assertEqual(loaded.params[name].dtype, np.dtype('float32'))
            self.assertFalse(loaded.params[name].flags.writeable)
        np.testing.assert_array_equal(loaded.mean, self.mean)
        np.testing.assert_array_equal(loaded.scale, self.scale)
        self.assertEqual(loaded.mean.dtype, self.mean.dtype)
        self.assertEqual(loaded.mean.tobytes(), self.mean.tobytes())
        self.assertTrue(np.signbit(loaded.mean[1]))
        self.assertNotIn('model_seed', loaded.schema_json)
        self.assertNotIn('SYNTHETIC_PRIVATE_MARKER', loaded.schema_json)

    def test_sha_and_size_reject_before_zip_decode(self):
        with patch.object(snapshot.zipfile, 'ZipFile', side_effect=AssertionError('unexpected decode')):
            with self.assertRaises(ValueError):
                self.load(sha='0' * 64)
            with self.assertRaises(ValueError):
                self.load(raw=b'x' * (snapshot.MAX_ARCHIVE_BYTES + 1))

    def test_truncated_duplicate_extra_and_total_budget_rejected(self):
        corrupted = [self.raw[:-12], replace_members(self.raw, duplicate='w0.npy'),
                     replace_members(self.raw, extra='extra.npy')]
        for raw in corrupted:
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                self.load(raw)
        with patch.object(snapshot, 'MAX_UNCOMPRESSED_BYTES', 1000), \
             patch.object(np, 'frombuffer', side_effect=AssertionError('array allocation reached')):
            with self.assertRaises(ValueError):
                self.load()

    def test_huge_declared_npy_shape_rejected_before_array_decode(self):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            'descr': '<f4', 'fortran_order': False, 'shape': (1_000_000_000,),
        })
        self.assertLess(len(header.getvalue()), 4096)
        raw = replace_members(self.raw, replace={'w0.npy': header.getvalue()})
        with patch.object(np, 'frombuffer', side_effect=AssertionError('array allocation reached')) as decode:
            with self.assertRaises(ValueError):
                self.load(raw)
        decode.assert_not_called()

        # The invalid member is last: even earlier valid members must not be
        # materialized before all member headers have been checked.
        raw = replace_members(self.raw, replace={'metadata.npy': header.getvalue()})
        with patch.object(np, 'frombuffer', side_effect=AssertionError('array allocation reached')) as decode:
            with self.assertRaises(ValueError):
                self.load(raw)
        decode.assert_not_called()

    def test_wrong_weight_dtype_or_shape_is_rejected(self):
        bad = io.BytesIO()
        np.save(bad, self.params['w0'].astype(np.float64), allow_pickle=False)
        with self.assertRaises(ValueError):
            self.load(replace_members(self.raw, replace={'w0.npy': bad.getvalue()}))
        bad = io.BytesIO()
        np.save(bad, self.params['w0'][:-1], allow_pickle=False)
        with self.assertRaises(ValueError):
            self.load(replace_members(self.raw, replace={'w0.npy': bad.getvalue()}))

    def test_nonfinite_and_noncontinuous_normalization_are_rejected(self):
        for field, index, value in (('mean', 1, np.nan), ('scale', 1, np.inf),
                                    ('scale', 1, .5), ('mean', 0, .1), ('scale', 0, 2.0)):
            mean, scale = self.mean.copy(), self.scale.copy()
            (mean if field == 'mean' else scale)[index] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load(make_archive(self.params, mean, scale, self.metadata))
        bad = {key: value.copy() for key, value in self.params.items()}
        bad['w0'][0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.load(make_archive(bad, self.mean, self.scale, self.metadata))

    def test_schema_binding_rejects_metadata_and_feature_drift(self):
        with self.assertRaises(ValueError):
            self.load(schema_sha='0' * 64)
        for field, value in (('ops', list(reversed(model.OPS))),
                             ('dimensions', [373, 128, 64, True]),
                             ('schema_version', 'different')):
            metadata = dict(self.metadata, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(make_archive(self.params, self.mean, self.scale, metadata))
        metadata = dict(self.metadata)
        metadata['feature_names'] = list(metadata['feature_names'])
        metadata['feature_names'][6] = 'teacher_label'
        metadata['schema_sha256'] = model.schema_hash(metadata['feature_names'], metadata['shared_feature_names'])
        with self.assertRaises(ValueError):
            self.load(make_archive(self.params, self.mean, self.scale, metadata))

    def test_duplicate_metadata_keys_are_rejected(self):
        bad = io.BytesIO()
        np.save(bad, np.asarray('{"schema_version":"a","schema_version":"b"}'), allow_pickle=False)
        with self.assertRaises(ValueError):
            self.load(replace_members(self.raw, replace={'metadata.npy': bad.getvalue()}))

    def test_two_exports_are_distinct_private_snapshots_with_sanitized_provenance(self):
        loaded = self.load()
        control = {name: value.copy() for name, value in self.params.items()}
        aux = {name: value.copy() for name, value in self.params.items()}
        control['w0'][0, 0] += np.float32(.01)
        aux['w0'][0, 0] -= np.float32(.01)
        provenance = {'training_source': 'synthetic_only', 'initialization': 'synthetic_fixture',
                      'registration_commit': 'a' * 40, 'completed_updates': 2}
        first, second = snapshot.export_pair(loaded, control, aux, provenance=provenance)
        self.assertIs(type(first), bytes)
        self.assertIs(type(second), bytes)
        self.assertNotEqual(first, second)
        control['w0'][0, 0] = 99  # The already exported bytes cannot alias input arrays.
        for raw, arm, coefficient in ((first, 'control', 0.0), (second, 'auxiliary', .25)):
            reloaded = self.load(raw)
            self.assertEqual(reloaded.mean.tobytes(), self.mean.tobytes())
            self.assertEqual(reloaded.scale.tobytes(), self.scale.tobytes())
            with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata']))
            for key in snapshot.SCHEMA_KEYS:
                self.assertEqual(metadata[key], self.metadata[key])
            self.assertEqual(metadata['pair_training']['arm'], arm)
            self.assertEqual(metadata['pair_training']['training_source'], 'synthetic_only')
            self.assertEqual(metadata['pair_training']['completed_updates'], 2)
            self.assertEqual(metadata['pair_training']['worker_coefficient'], coefficient)
            self.assertNotIn('model_seed', metadata)
            self.assertNotIn('SYNTHETIC_PRIVATE_MARKER', json.dumps(metadata))
        x = np.arange(3 * 373, dtype=np.float64).reshape(3, 373) / 100
        loaded_first = self.load(first)
        normalized = (x - loaded_first.mean) / loaded_first.scale
        expected = np.maximum(normalized @ loaded_first.params['w0'] + loaded_first.params['b0'], 0)
        expected = np.maximum(expected @ loaded_first.params['w1'] + loaded_first.params['b1'], 0)
        expected = (expected @ loaded_first.params['w2'] + loaded_first.params['b2'])[:, 0]
        np.testing.assert_allclose(model.numpy_logits(loaded_first.params, normalized), expected, rtol=1e-12, atol=1e-12)

    def test_export_rejects_unchanged_arms_unprovenance_and_wrong_parameters(self):
        loaded = self.load()
        original = {name: value.copy() for name, value in self.params.items()}
        altered = {name: value.copy() for name, value in self.params.items()}
        altered['b2'][0] += np.float32(.01)
        provenance = {'training_source': 'synthetic_only', 'initialization': 'synthetic_fixture',
                      'registration_commit': 'a' * 40, 'completed_updates': 2}
        for control, aux in ((original, altered), (altered, original), (altered, altered)):
            with self.assertRaises(ValueError):
                snapshot.export_pair(loaded, control, aux, provenance=provenance)
        malformed = dict(provenance, training_source='synthetic path with personal data')
        with self.assertRaises(ValueError):
            snapshot.export_pair(loaded, original, altered, provenance=malformed)
        malformed = dict(provenance, training_source='original_teacher_train',
                         initialization='frozen_early_task_head', completed_updates=63)
        with self.assertRaises(ValueError):
            snapshot.export_pair(loaded, original, altered, provenance=malformed)
        bad = dict(altered)
        bad['w1'] = altered['w1'].astype(np.float64)
        with self.assertRaises(ValueError):
            snapshot.export_pair(loaded, bad, altered, provenance=provenance)

    def test_export_retains_independent_schema_binding(self):
        loaded = self.load()
        schema = json.loads(loaded.schema_json)
        schema['feature_names'][6] = 'teacher_label'
        schema['schema_sha256'] = model.schema_hash(schema['feature_names'], schema['shared_feature_names'])
        changed = replace(loaded, schema_json=json.dumps(schema))
        control = {name: value.copy() for name, value in self.params.items()}
        auxiliary = {name: value.copy() for name, value in self.params.items()}
        control['w0'][0, 0] += np.float32(.01)
        auxiliary['w0'][0, 0] -= np.float32(.01)
        provenance = {'training_source': 'synthetic_only', 'initialization': 'synthetic_fixture',
                      'registration_commit': 'a' * 40, 'completed_updates': 2}
        with self.assertRaises(ValueError):
            snapshot.export_pair(changed, control, auxiliary, provenance=provenance)


if __name__ == '__main__':
    unittest.main()
