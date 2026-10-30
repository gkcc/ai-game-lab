"""In-memory NumPy scorer checks; no real data, model, GPU or games."""

from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from learning.task_bc_v1.model import DIMENSIONS, TaskNetwork
from learning.task_pair_eval_adapter_v1 import adapter
from learning.task_pair_eval_adapter_v1.test_adapter import sample_view, source_masks, SCHEMA
from learning.task_pair_numpy_scorer_v1 import scorer


def synthetic_network(*, favor_high_unit=False):
    model = TaskNetwork.__new__(TaskNetwork)
    model.metadata = dict(SCHEMA, dimensions=list(DIMENSIONS))
    model.mean = np.zeros(373, np.float64)
    model.scale = np.ones(373, np.float64)
    model.params = {}
    for layer in range(3):
        model.params['w' + str(layer)] = np.zeros((DIMENSIONS[layer], DIMENSIONS[layer + 1]), np.float64)
        model.params['b' + str(layer)] = np.zeros(DIMENSIONS[layer + 1], np.float64)
    if favor_high_unit:
        index = 275 + SCHEMA['feature_names'].index('unit.index')
        model.params['w0'][index, 0] = 1
        model.params['w1'][0, 0] = 1
        model.params['w2'][0, 0] = 1
    return model


def blocks(view, model, **options):
    return scorer.score_blocks(view, model, expected_schema_sha256=SCHEMA['schema_sha256'],
                               check=lambda: None, **options)


class ScorerTests(unittest.TestCase):
    def test_end_to_end_ties_and_separate_all_unseen_metrics(self):
        view, model = sample_view(), synthetic_network()
        seen = range(10, 30)
        parts = list(blocks(view, model, max_groups=7, max_candidates=21))
        self.assertEqual(parts[0].start, 0)
        self.assertEqual(parts[-1].stop, len(view.shared))
        self.assertTrue(all(type(part) is adapter.LogitBlock for part in parts))
        self.assertTrue(all(not part.logits.flags.writeable and not part.choices.flags.writeable
                            for part in parts))
        self.assertTrue(all(np.all(part.logits == 0) and np.all(part.choices == 0) for part in parts))
        result = adapter.evaluate_source(view, parts, source_masks(40, seen), source='teacher_dev',
                                         expected_clusters=(11, 22),
                                         expected_schema_sha256=SCHEMA['schema_sha256'])
        self.assertEqual(result['all']['full_top1']['value'], 1.0)
        self.assertEqual(result['unseen_train']['full_top1']['value'], 1.0)
        self.assertEqual(result['all']['cross_operation_recall']['WATER'],
                         {'support': 'unknown', 'value': None})

    def test_original_candidate_order_and_block_size_do_not_change_scores(self):
        view, model = sample_view(), synthetic_network(favor_high_unit=True)
        results = [list(blocks(view, model, max_groups=size)) for size in (1, 7, 256)]
        flattened = [np.concatenate([part.logits for part in group]) for group in results]
        choices = [np.concatenate([part.choices for part in group]) for group in results]
        for other in flattened[1:]:
            np.testing.assert_array_equal(flattened[0], other)
        for other in choices[1:]:
            np.testing.assert_array_equal(choices[0], other)
        np.testing.assert_array_equal(flattened[0][:3], [0, 1, 2])
        self.assertTrue(np.all(choices[0] == 2))
        metrics = adapter.evaluate_source(view, results[1], source_masks(40), source='teacher_dev',
                                          expected_clusters=(11, 22),
                                          expected_schema_sha256=SCHEMA['schema_sha256'])
        self.assertEqual(metrics['all']['full_top1']['value'], 0.0)
        self.assertEqual(metrics['all']['plant_same_target_unit']['value'], 0.0)

    def test_conditional_old_and_new_flow_separately_through_scorer(self):
        for source, clusters, model, expected in (
                ('conditional_old', (11, 22), synthetic_network(), 1.0),
                ('conditional_new', (33, 44), synthetic_network(favor_high_unit=True), 0.0)):
            view = sample_view(conditional=True, clusters=clusters, sizes=(10, 10))
            result = adapter.evaluate_source(view, blocks(view, model, max_groups=4), source_masks(20),
                                             source=source, expected_clusters=clusters,
                                             expected_schema_sha256=SCHEMA['schema_sha256'])
            self.assertEqual(result['source'], source)
            self.assertEqual(result['all']['full_top1']['value'], expected)

    def test_only_shared_and_candidates_reach_actual_scorer(self):
        view, model = sample_view(sizes=(10, 10)), synthetic_network()
        original = TaskNetwork.score
        with patch.object(TaskNetwork, 'score', autospec=True, side_effect=original) as score:
            list(blocks(view, model, max_groups=5))
        self.assertEqual(score.call_count, 20)
        frozen_model = score.call_args_list[0].args[0]
        self.assertIsNot(frozen_model, model)
        self.assertFalse(np.shares_memory(frozen_model.mean, model.mean))
        self.assertFalse(np.shares_memory(frozen_model.scale, model.scale))
        for key in model.params:
            self.assertFalse(np.shares_memory(frozen_model.params[key], model.params[key]))
            self.assertFalse(frozen_model.params[key].flags.writeable)
        for call in score.call_args_list:
            self.assertEqual(len(call.args), 3)
            self.assertEqual(call.kwargs, {})
            instance, shared, candidates = call.args
            self.assertIs(instance, frozen_model)
            self.assertEqual(shared.shape, (275,))
            self.assertEqual(candidates.shape, (3, 98))
            self.assertFalse(np.shares_memory(shared, view.labels))
            self.assertFalse(np.shares_memory(candidates, view.private_clusters))
        altered = replace(view, labels=1 - view.labels,
                          private_clusters=np.full_like(view.private_clusters, 77))
        first = np.concatenate([part.logits for part in blocks(view, model)])
        second = np.concatenate([part.logits for part in blocks(altered, model)])
        np.testing.assert_array_equal(first, second)

    def test_schema_mismatch_and_mid_block_mutation_fail_before_output(self):
        view, model = sample_view(sizes=(10, 10)), synthetic_network()
        for name, value in (('schema_sha256', 'f' * 64), ('schema_version', True),
                            ('dimensions', [373, 128, 64, True]), ('ops', []),
                            ('feature_names', list(reversed(SCHEMA['feature_names'])))):
            old = model.metadata[name]
            model.metadata[name] = value
            try:
                with self.subTest(field=name), self.assertRaises(ValueError):
                    list(blocks(view, model))
            finally:
                model.metadata[name] = old
        with self.assertRaises(ValueError):
            list(scorer.score_blocks(view, model, expected_schema_sha256='f' * 64, check=lambda: None))
        def mutate():
            model.metadata['schema_sha256'] = 'f' * 64
        with patch.object(TaskNetwork, 'score') as score, self.assertRaises(ValueError):
            list(scorer.score_blocks(view, model, expected_schema_sha256=SCHEMA['schema_sha256'], check=mutate))
        score.assert_not_called()

    def test_bad_score_shape_dtype_nonfinite_and_small_block_limit_fail(self):
        view, model = sample_view(sizes=(10, 10)), synthetic_network()
        invalid = (np.array([0, 1]), np.array([0, 1, 2], np.int32),
                   np.array([0., np.nan, 2.]), np.array([0., np.inf, 2.]),
                   np.array([0., -np.inf, 2.]))
        for result in invalid:
            with self.subTest(dtype=str(result.dtype)), patch.object(TaskNetwork, 'score', return_value=result), \
                 self.assertRaises(ValueError):
                list(blocks(view, model))
        with patch.object(TaskNetwork, 'score') as score, self.assertRaises(ValueError):
            list(blocks(view, model, max_candidates=2))
        score.assert_not_called()

    def test_float32_first_maximum_and_cancellation_fail_closed(self):
        view, model = sample_view(sizes=(10, 10)), synthetic_network()
        with patch.object(TaskNetwork, 'score', return_value=np.array([0., 1., 1.], np.float32)):
            part = next(blocks(view, model))
        self.assertEqual(part.logits.dtype, np.dtype('float64'))
        self.assertTrue(np.all(part.choices == 1))
        count = []
        def cancel():
            count.append(True)
            if len(count) == 4:
                raise TimeoutError('synthetic cancellation')
        iterator = scorer.score_blocks(view, model, expected_schema_sha256=SCHEMA['schema_sha256'],
                                       check=cancel, max_groups=5)
        with self.assertRaises(TimeoutError):
            adapter.evaluate_source(view, iterator, source_masks(20), source='teacher_dev',
                                    expected_clusters=(11, 22),
                                    expected_schema_sha256=SCHEMA['schema_sha256'])

    def test_callback_mutations_cannot_mix_weights_or_normalization(self):
        view = sample_view(sizes=(10, 10))
        for subject, trigger in (('weights', 1), ('weights', 3),
                                 ('normalization', 1), ('normalization', 3)):
            with self.subTest(subject=subject, trigger=trigger):
                model = synthetic_network(favor_high_unit=True)
                before = list(blocks(view, model, max_groups=5))
                checks = []
                def mutate():
                    checks.append(True)
                    if len(checks) == trigger:
                        if subject == 'weights':
                            model.params['w2'][0, 0] = -10
                        else:
                            model.mean[275] = 200
                            model.scale[275] = 2
                after = list(scorer.score_blocks(view, model,
                    expected_schema_sha256=SCHEMA['schema_sha256'], check=mutate, max_groups=5))
                self.assertGreaterEqual(len(checks), trigger)
                if subject == 'weights':
                    self.assertEqual(model.params['w2'][0, 0], -10)
                else:
                    self.assertEqual(model.mean[275], 200)
                    self.assertEqual(model.scale[275], 2)
                np.testing.assert_array_equal(np.concatenate([p.logits for p in before]),
                                              np.concatenate([p.logits for p in after]))
                np.testing.assert_array_equal(np.concatenate([p.choices for p in before]),
                                              np.concatenate([p.choices for p in after]))

    def test_float32_source_matches_existing_scorer_on_same_groups(self):
        view, model = sample_view(sizes=(10, 10)), synthetic_network(favor_high_unit=True)
        model.params = {key: value.astype(np.float32) for key, value in model.params.items()}
        model.mean = model.mean.astype(np.float32)
        model.scale = model.scale.astype(np.float32)
        direct = []
        for group in range(len(view.shared)):
            lo, hi = int(view.offsets[group]), int(view.offsets[group + 1])
            direct.append(TaskNetwork.score(model, view.shared[group], view.candidates[lo:hi]))
        observed = np.concatenate([part.logits for part in blocks(view, model, max_groups=7)])
        np.testing.assert_array_equal(observed, np.concatenate(direct))

    def test_c_and_fortran_model_layouts_match_scorer_in_same_cpu_process(self):
        view = sample_view(sizes=(10, 10))
        for layout in ('C', 'F'):
            with self.subTest(layout=layout):
                model = synthetic_network()
                rng = np.random.default_rng(813)  # Toy weights and features only.
                for name, value in model.params.items():
                    model.params[name] = np.array(rng.standard_normal(value.shape) * 0.1,
                                                  dtype=np.float64, order=layout)
                frozen = scorer._snapshot_model(view, model, SCHEMA['schema_sha256'])
                for name in ('w0', 'w1'):
                    if layout == 'F':
                        self.assertTrue(frozen.params[name].flags.f_contiguous)
                    else:
                        self.assertTrue(frozen.params[name].flags.c_contiguous)
                for width in (1, 2, 3, 8, 16, 32, 64, 128, 256):
                    shared = rng.standard_normal(275).astype(np.float32)
                    candidates = rng.standard_normal((width, 98)).astype(np.float32)
                    expected = TaskNetwork.score(model, shared, candidates)
                    actual = TaskNetwork.score(frozen, shared, candidates)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(int(np.argmax(actual)), int(np.argmax(expected)))
                direct = []
                for group in range(len(view.shared)):
                    lo, hi = int(view.offsets[group]), int(view.offsets[group + 1])
                    direct.append(TaskNetwork.score(model, view.shared[group], view.candidates[lo:hi]))
                generated = list(blocks(view, model, max_groups=7))
                np.testing.assert_array_equal(np.concatenate([part.logits for part in generated]),
                                              np.concatenate(direct))
                np.testing.assert_array_equal(np.concatenate([part.choices for part in generated]),
                                              np.asarray([np.argmax(item) for item in direct]))

    def test_snapshot_validation_precedes_callbacks_and_oversized_copies(self):
        view = sample_view(sizes=(10, 10))
        bad_models = []
        wrong_dtype = synthetic_network()
        wrong_dtype.params['b2'] = np.zeros(1, np.int32)
        bad_models.append(wrong_dtype)
        nonfinite = synthetic_network()
        nonfinite.params['w0'][0, 0] = np.nan
        bad_models.append(nonfinite)
        invalid_mean = synthetic_network()
        invalid_mean.mean[0] = np.inf
        bad_models.append(invalid_mean)
        invalid_scale = synthetic_network()
        invalid_scale.scale[0] = 0
        bad_models.append(invalid_scale)
        wrong_binary = synthetic_network()
        binary = 275 + SCHEMA['feature_names'].index('task.op.PLANT')
        wrong_binary.mean[binary] = 1
        bad_models.append(wrong_binary)
        for model in bad_models:
            called = []
            with self.subTest(case=len(bad_models)), self.assertRaises(ValueError):
                list(scorer.score_blocks(view, model, expected_schema_sha256=SCHEMA['schema_sha256'],
                                         check=lambda: called.append(True)))
            self.assertEqual(called, [])
        oversized = synthetic_network()
        oversized.params['w0'] = np.broadcast_to(np.zeros(1, np.float64), (100000, 100000))
        called = []
        with patch.object(scorer.np, 'array', side_effect=AssertionError('unexpected copy')) as copy, \
             self.assertRaises(ValueError):
            list(scorer.score_blocks(view, oversized, expected_schema_sha256=SCHEMA['schema_sha256'],
                                     check=lambda: called.append(True)))
        copy.assert_not_called()
        self.assertEqual(called, [])
        valid = synthetic_network()
        with patch.object(scorer, 'MAX_MODEL_SNAPSHOT_BYTES', 1), \
             patch.object(scorer.np, 'array', side_effect=AssertionError('unexpected copy')) as copy, \
             self.assertRaises(ValueError):
            list(scorer.score_blocks(view, valid, expected_schema_sha256=SCHEMA['schema_sha256'],
                                     check=lambda: called.append(True)))
        copy.assert_not_called()
        self.assertEqual(called, [])


if __name__ == '__main__':
    unittest.main()
