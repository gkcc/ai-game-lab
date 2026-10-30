"""Synthetic CPU 2x8 end-to-end pipeline; no files, real model, GPU or games."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch


os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork
from learning.task_pair_checkpoint_v1 import snapshot
from learning.task_pair_eval_adapter_v1.test_adapter import sample_view, source_masks, SCHEMA
from learning.task_pair_memory_pipeline_v1 import pipeline
from learning.task_pair_numpy_scorer_v1 import scorer
from learning.task_pair_orchestration_v1.test_core import synthetic_data, synthetic_parameters


def synthetic_checkpoint():
    toy = synthetic_parameters()
    params = {name: np.zeros(snapshot.SHAPES[name], np.float32) for name in snapshot.PARAMS}
    params['w0'][:, :4] = toy['w0']
    params['b0'][:4] = toy['b0']
    params['w1'][:4, :2] = toy['w1']
    params['b1'][:2] = toy['b1']
    params['w2'][:2] = toy['w2']
    params['b2'][:] = toy['b2']
    mean, scale = np.zeros(373, np.float64), np.ones(373, np.float64)
    metadata = dict(SCHEMA, dimensions=list(DIMENSIONS))
    stream = io.BytesIO()
    np.savez_compressed(stream, **params, mean=mean, scale=scale,
                        metadata=np.asarray(json.dumps(metadata)))
    raw = stream.getvalue()
    item = snapshot.load_checkpoint(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                                    expected_schema_sha256=SCHEMA['schema_sha256'])
    return item, params


def synthetic_sources():
    out = {}
    for source, clusters, conditional in (('teacher_dev', (11, 22), False),
                                           ('conditional_old', (31, 32), True),
                                           ('conditional_new', (33, 34), True)):
        view = sample_view(conditional=conditional, clusters=clusters, sizes=(20, 20))
        masks = source_masks(40, list(range(5)) + list(range(20, 25)))
        out[source] = pipeline.DevSource(view, masks, clusters)
    return out


@unittest.skipUnless(importlib.util.find_spec('jax') is not None
                     and importlib.util.find_spec('optax') is not None, 'JAX and Optax are required')
class MemoryPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        cls.cpu = jax.devices('cpu')[0]
        cls.checkpoint, cls.original_params = synthetic_checkpoint()
        cls.original_mean = cls.checkpoint.mean.copy()
        cls.weights = np.ones(5, np.float32)
        cls.check_calls = []
        def mutate_caller_objects():
            if not cls.check_calls:
                cls.checkpoint.mean.setflags(write=True)
                cls.checkpoint.mean[0] = 99
                cls.checkpoint.params['w2'].setflags(write=True)
                cls.checkpoint.params['w2'][0, 0] += 5
                cls.weights[0] = 9
            cls.check_calls.append(True)
        with jax.default_device(cls.cpu):
            cls.exports = pipeline.train_and_export(
                synthetic_data(), cls.checkpoint, cls.weights,
                expected_schema_sha256=SCHEMA['schema_sha256'], device=cls.cpu, rng_seed=23,
                updates=2, batch_size=8, training_source='synthetic_only',
                registration_commit='a' * 40, check=mutate_caller_objects)

    def test_complete_production_shape_synthetic_train_export_and_memory_reload(self):
        self.assertTrue(self.check_calls)
        self.assertEqual(repr(self.exports), '<PrivateInMemoryTrainingExports>')
        self.assertEqual(self.exports.completed_updates, 2)
        self.assertGreaterEqual(self.exports.auxiliary_nonzero_updates, 1)
        self.assertNotEqual(self.exports.control_bytes, self.exports.auxiliary_bytes)
        np.testing.assert_array_equal(self.exports.baseline.mean, self.original_mean)
        for key in snapshot.PARAMS:
            self.assertEqual(self.exports.baseline.params[key].shape, snapshot.SHAPES[key])
            np.testing.assert_array_equal(self.exports.baseline.params[key], self.original_params[key])
        models = pipeline.models_from_memory(self.exports, expected_schema_sha256=SCHEMA['schema_sha256'])
        self.assertEqual(set(models), {'baseline', 'control', 'auxiliary'})
        for model in models.values():
            self.assertIs(type(model), TaskNetwork)
            self.assertIsNone(model.sha)  # Not an actual file constructor/reload.
        self.assertTrue(any(not np.array_equal(models['baseline'].params[k], models['control'].params[k])
                            for k in snapshot.PARAMS))
        self.assertTrue(any(not np.array_equal(models['control'].params[k], models['auxiliary'].params[k])
                            for k in snapshot.PARAMS))

    def test_each_source_and_model_matches_direct_synthetic_logits_and_choices(self):
        models = pipeline.models_from_memory(self.exports, expected_schema_sha256=SCHEMA['schema_sha256'])
        for model_name, model in models.items():
            for source_name, source in synthetic_sources().items():
                with self.subTest(model=model_name, source=source_name):
                    blocks = list(scorer.score_blocks(source.view, model,
                        expected_schema_sha256=SCHEMA['schema_sha256'], check=lambda: None, max_groups=7))
                    actual = np.concatenate([part.logits for part in blocks])
                    choices = np.concatenate([part.choices for part in blocks])
                    direct = []
                    for group in range(len(source.view.shared)):
                        lo, hi = map(int, source.view.offsets[group:group + 2])
                        direct.append(TaskNetwork.score(model, source.view.shared[group],
                                                        source.view.candidates[lo:hi]))
                    np.testing.assert_array_equal(actual, np.concatenate(direct))
                    np.testing.assert_array_equal(choices, np.asarray([np.argmax(row) for row in direct]))

    def test_evaluation_uses_frozen_models_masks_and_keeps_sources_separate(self):
        models = pipeline.models_from_memory(self.exports, expected_schema_sha256=SCHEMA['schema_sha256'])
        sources = synthetic_sources()
        reference = pipeline.evaluate_models(models, sources, expected_schema_sha256=SCHEMA['schema_sha256'],
                                              check=lambda: None)
        self.assertEqual(repr(reference), '<PrivateDevelopmentEvaluation>')
        calls = []
        def mutate_caller_objects():
            if not calls:
                models['baseline'].params['w2'].setflags(write=True)
                models['baseline'].params['w2'][0, 0] += 100
                for source in sources.values():
                    source.masks['seen_in_train'][:] = True
            calls.append(True)
        actual = pipeline.evaluate_models(models, sources, expected_schema_sha256=SCHEMA['schema_sha256'],
                                          check=mutate_caller_objects)
        self.assertTrue(calls)
        self.assertEqual(actual.metrics, reference.metrics)
        self.assertTrue(all(source.masks['seen_in_train'].all() for source in sources.values()))
        for model_name, by_source in actual.metrics.items():
            self.assertEqual(set(by_source), {'teacher_dev', 'conditional_old', 'conditional_new'})
            for source_name, result in by_source.items():
                self.assertEqual(result['source'], source_name)
                for subset in ('all', 'unseen_train'):
                    self.assertEqual(result[subset]['full_top1']['support'], 'supported')

    def test_invalid_contracts_fail_before_training_or_scoring(self):
        valid, _ = synthetic_checkpoint()
        kwargs = {'expected_schema_sha256': SCHEMA['schema_sha256'], 'device': self.cpu, 'rng_seed': 23,
                  'updates': 2, 'batch_size': 8, 'training_source': 'synthetic_only',
                  'registration_commit': 'a' * 40, 'check': lambda: None}
        bad_cases = ({'updates': 3}, {'batch_size': 9}, {'training_source': 'original_teacher_train'},
                     {'registration_commit': 'invalid'}, {'expected_schema_sha256': 'f' * 64})
        for bad in bad_cases:
            with self.subTest(keys=tuple(bad)), patch.object(pipeline, 'train_pair') as train, \
                 self.assertRaises(ValueError):
                pipeline.train_and_export(synthetic_data(), valid, np.ones(5, np.float32), **(kwargs | bad))
            train.assert_not_called()
        models = pipeline.models_from_memory(self.exports, expected_schema_sha256=SCHEMA['schema_sha256'])
        sources = synthetic_sources()
        missing = dict(sources)
        del missing['conditional_new']
        with patch.object(scorer, 'score_blocks') as score, self.assertRaises(ValueError):
            pipeline.evaluate_models(models, missing, expected_schema_sha256=SCHEMA['schema_sha256'],
                                     check=lambda: None)
        score.assert_not_called()

    def test_interruption_does_not_return_partial_evaluation(self):
        models = pipeline.models_from_memory(self.exports, expected_schema_sha256=SCHEMA['schema_sha256'])
        count = []
        def cancel():
            count.append(True)
            if len(count) == 4:
                raise TimeoutError('synthetic cancellation')
        with self.assertRaises(TimeoutError):
            pipeline.evaluate_models(models, synthetic_sources(),
                                     expected_schema_sha256=SCHEMA['schema_sha256'], check=cancel)

    def test_production_auxiliary_gate_precedes_transfer_and_export(self):
        # Only the control-flow gate is injected; no GPU backend or real data is used.
        checkpoint, _ = synthetic_checkpoint()
        device = SimpleNamespace(platform='gpu')
        kwargs = {'expected_schema_sha256': SCHEMA['schema_sha256'], 'device': device,
                  'rng_seed': 23, 'updates': 64, 'batch_size': 256,
                  'training_source': 'original_teacher_train', 'registration_commit': 'a' * 40,
                  'check': lambda: None}
        for observed in (1, 31):
            result = SimpleNamespace(completed_updates=64, auxiliary_nonzero_updates=observed)
            with self.subTest(nonzero=observed), patch.object(pipeline, 'train_pair', return_value=result), \
                 patch.object(pipeline, 'copy_pair_to_host') as transfer, \
                 patch.object(snapshot, 'export_pair') as export, self.assertRaises(ValueError):
                pipeline.train_and_export(synthetic_data(), checkpoint, np.ones(5, np.float32), **kwargs)
            transfer.assert_not_called()
            export.assert_not_called()
        accepted = SimpleNamespace(completed_updates=64, auxiliary_nonzero_updates=32)
        with patch.object(pipeline, 'train_pair', return_value=accepted), \
             patch.object(pipeline, 'copy_pair_to_host', return_value=({}, {})) as transfer, \
             patch.object(snapshot, 'export_pair', return_value=(b'a', b'b')) as export:
            result = pipeline.train_and_export(synthetic_data(), checkpoint, np.ones(5, np.float32), **kwargs)
        self.assertEqual(result.completed_updates, 64)
        self.assertEqual(result.auxiliary_nonzero_updates, 32)
        transfer.assert_called_once()
        export.assert_called_once()


if __name__ == '__main__':
    unittest.main()
