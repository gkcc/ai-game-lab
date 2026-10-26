"""Small in-memory regressions; no real cache, checkpoint, JAX or GPU reads."""
from copy import deepcopy
import unittest

import numpy as np

from learning.task_bc_v1.model import DIMENSIONS
from learning.task_bc_v1.train import microbatches, padded, weighted_group_loss
from .training import (TRAINING, Consumption, compact_microbatches,
    anchored_group_loss, conditional_metrics, mixture_indices, numeric_block,
    numerical_check_groups, offline_gate, same_array_bits, score_cpu,
    validate_config, validate_device, validate_split_seeds, warm_start_arrays,
    weighted_anchor_kl)


def data(lengths=(2, 3, 1), labels=(1, 0, 0)):
    offsets = np.asarray([0, *np.cumsum(lengths)], dtype=np.int64)
    n, m = len(lengths), int(offsets[-1])
    shared = np.zeros((n, 275), np.float32)
    shared[:, 0] = np.arange(n)+1
    local = np.zeros((m, 98), np.float32)
    local[:, 0] = np.arange(m)/10
    return dict(shared=shared, candidates=local, offsets=offsets,
        labels=np.asarray(labels, dtype=np.int64),
        selected_ops=np.arange(n, dtype=np.int8) % 5)


def arrays():
    result = dict(mean=np.zeros(373, np.float64), scale=np.ones(373, np.float64))
    for i, (a, b) in enumerate(zip(DIMENSIONS[:-1], DIMENSIONS[1:])):
        result['w'+str(i)] = np.zeros((a, b), np.float32)
        result['b'+str(i)] = np.zeros(b, np.float32)
    return result


class FixedTrainingTests(unittest.TestCase):
    def test_private_device_requirement_rejects_empty_mismatch_or_non_gpu(self):
        validate_device('gpu', 'Synthetic Device Alpha', 'Device Alpha')
        for platform, kind, expected in (
                ('gpu', 'Synthetic Device Alpha', ''),
                ('gpu', 'Synthetic Device Alpha', '  '),
                ('gpu', 'Synthetic Device Alpha', None),
                ('gpu', 'Synthetic Device Alpha', 'Device Beta'),
                ('cpu', 'Synthetic Device Alpha', 'Device Alpha')):
            with self.subTest(platform=platform, expected=expected), self.assertRaises(ValueError):
                validate_device(platform, kind, expected)

    def test_training_contract_rejects_change_and_ambiguous_types(self):
        cfg = dict(training=deepcopy(TRAINING), training_seed=2026102707)
        validate_config(cfg)
        for key, value in (('updates', 513), ('updates', 512.0),
                           ('learning_rate', .0003), ('effective_batch_groups', True),
                           ('anchor_coefficient', .5), ('anchor_temperature', 2.)):
            altered = deepcopy(cfg)
            altered['training'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(altered)
        for seed in (True, 7, 2026102707.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_config(dict(training=TRAINING, training_seed=seed))

    def test_warm_start_preserves_tensors_and_normalizer_bits(self):
        source = arrays()
        source['mean'][0] = -0.
        source['w0'][0, 0] = np.float32(.25)
        before = {k: value.copy() for k, value in source.items()}
        params, normalizer = warm_start_arrays(source)
        for k, value in params.items():
            self.assertTrue(same_array_bits(source[k], value))
        for k, value in normalizer.items():
            self.assertTrue(same_array_bits(source[k], value))
        params['w0'][0, 0] = 9.
        normalizer['mean'][0] = 1.
        for k in source:
            self.assertTrue(same_array_bits(source[k], before[k]))
        self.assertFalse(same_array_bits(np.asarray([0.]), np.asarray([-0.])))

    def test_warm_start_rejects_lossy_values_and_invalid_scale(self):
        source = arrays()
        source['w0'] = source['w0'].astype(np.float64)
        source['w0'][0, 0] = .1
        with self.assertRaises(ValueError):
            warm_start_arrays(source)
        source = arrays()
        source['scale'][1] = .9
        with self.assertRaises(ValueError):
            warm_start_arrays(source)

    def test_sampling_is_fixed_balanced_and_train_pool_bounded(self):
        first = mixture_indices(np.random.default_rng(17), 3, 7)
        repeat = mixture_indices(np.random.default_rng(17), 3, 7)
        self.assertEqual(tuple(map(len, first)), (128, 128))
        for value, same, size in zip(first, repeat, (3, 7)):
            np.testing.assert_array_equal(value, same)
            self.assertTrue(np.all((value >= 0) & (value < size)))
            self.assertLess(len(np.unique(value)), len(value))
        with self.assertRaises(ValueError):
            mixture_indices(np.random.default_rng(1), 0, 7)

    def test_seed_roles_keep_known_screens_separate_from_fresh_panel(self):
        source = lambda values: dict(seeds=np.asarray(values, dtype=np.uint32))
        cfg = dict(training_source_seeds=[11, 12, 13, 14], known_screen_seeds=[21, 22],
            validation_seeds=[31, 32], test_seeds=list(range(41, 49)))
        pools = [source([1, 2]), source([3]), source([11, 12, 13, 14, 14]), source([21, 22, 22])]
        roles = validate_split_seeds(*pools, cfg)
        self.assertEqual(roles['old_training_source_seeds'], [1, 2])
        self.assertEqual(roles['known_screen_seeds'], [21, 22])
        self.assertFalse(roles['known_screens_are_independent'])
        for wrong in ([11, 12, 13, 21], [11, 12, 13, 31], [11, 12, 13, 41], [11]):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                validate_split_seeds(pools[0], pools[1], source(wrong), pools[3], cfg)
        for wrong in ([2], [11], [21], [31], [41]):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                validate_split_seeds(pools[0], source(wrong), pools[2], pools[3], cfg)
        altered = deepcopy(cfg)
        altered['known_screen_seeds'] = list(cfg['validation_seeds'])
        with self.assertRaises(ValueError):
            validate_split_seeds(*pools, altered)
        altered = deepcopy(cfg)
        altered['validation_seeds'] = [21, 32]
        with self.assertRaises(ValueError):
            validate_split_seeds(*pools, altered)
        altered = deepcopy(cfg)
        altered['test_seeds'][0] = True
        with self.assertRaises(ValueError):
            validate_split_seeds(*pools, altered)

    def test_anchor_at_original_has_zero_value_and_zero_candidate_gradient(self):
        original = np.asarray([[.4, -.2, 9.], [-.1, .8, .3]])
        mask = np.asarray([[True, True, False], [True, True, True]])
        weights = np.asarray([.5, 2.])
        loss = lambda candidate: weighted_anchor_kl(candidate, original, mask, weights, 2.5)
        self.assertEqual(loss(original), 0.)
        epsilon = 1e-5
        for row, col in np.ndindex(original.shape):
            plus, minus = original.copy(), original.copy()
            plus[row, col] += epsilon
            minus[row, col] -= epsilon
            self.assertAlmostEqual((loss(plus)-loss(minus))/(2*epsilon), 0., places=9)

    def test_anchor_direction_nonnegative_and_padding_stays_finite(self):
        prior = np.asarray([[.75, .25], [.2, .8]])
        current = np.asarray([[.5, .5], [.7, .3]])
        weights = np.asarray([.5, 2.])
        expected = np.sum(weights*np.sum(prior*np.log(prior/current), axis=1))/weights.sum()
        actual = weighted_anchor_kl(np.log(current), np.log(prior),
            np.ones((2, 2), bool), weights, weights.sum())
        self.assertGreaterEqual(actual, 0.)
        self.assertAlmostEqual(actual, expected, places=14)
        padded_prior = np.asarray([[*np.log(prior[0]), np.nan],
            [*np.log(prior[1]), np.inf], [0., np.nan, -np.inf], [np.nan]*3])
        padded_current = np.asarray([[*np.log(current[0]), -np.inf],
            [*np.log(current[1]), np.nan], [0., np.inf, np.nan], [np.nan]*3])
        mask = np.asarray([[True, True, False], [True, True, False],
                           [True, False, False], [False, False, False]])
        with np.errstate(all='raise'):
            padded_loss = weighted_anchor_kl(padded_current, padded_prior, mask,
                np.asarray([.5, 2., 0., 0.]), weights.sum())
        self.assertAlmostEqual(padded_loss, expected, places=14)

    def test_ce_and_anchor_use_distinct_full_batch_denominators(self):
        old_scores = np.asarray([[.3, -.7], [-.5, .2]])
        prior = np.asarray([[-.8, .4], [.9, -.4]])
        old_weights, new_weights = np.asarray([2., 4.]), np.asarray([3.])
        labels = np.asarray([0, 1])
        mask = np.ones((2, 2), bool)
        new_scores = np.asarray([[1., -1.]])
        old_divisor, all_divisor = 6., 9.
        ce = lambda scores, target: np.log(np.exp(scores-scores.max()).sum())-(scores[target]-scores.max())
        old_ce = sum(w*ce(z, y) for w, z, y in zip(old_weights, old_scores, labels))
        new_ce = 3.*ce(new_scores[0], 1)
        probabilities = np.exp(prior)/np.exp(prior).sum(axis=1, keepdims=True)
        candidate = np.exp(old_scores)/np.exp(old_scores).sum(axis=1, keepdims=True)
        weighted_kl = sum(old_weights*np.sum(probabilities*np.log(probabilities/candidate), axis=1))
        expected = (old_ce+new_ce)/all_divisor+weighted_kl/old_divisor
        actual = anchored_group_loss(old_scores, prior, mask, labels, old_weights,
            all_divisor, old_divisor)
        actual += weighted_group_loss(new_scores, np.ones((1, 2), bool),
            np.asarray([1]), new_weights, all_divisor)
        self.assertAlmostEqual(actual, expected, places=14)
        split = sum(anchored_group_loss(old_scores[i:i+1], prior[i:i+1], mask[i:i+1],
            labels[i:i+1], old_weights[i:i+1], all_divisor, old_divisor) for i in range(2))
        self.assertAlmostEqual(split, actual-new_ce/all_divisor, places=14)
        self.assertGreater(abs(actual-((old_ce+new_ce+weighted_kl)/all_divisor)), .01)

    def test_compact_padding_preserves_weighted_loss_and_gradient(self):
        source = data()
        ids = np.asarray([0, 1, 2, 1, 0])
        weights = np.asarray([.5, 2., 4., 1., 1.], np.float32)
        denominator = weights[source['selected_ops'][ids]].sum(dtype=np.float64)
        mean, scale = np.zeros(373), np.ones(373)
        def loss(generator, value):
            total = 0.
            for group_ids, capacity, width in generator(source, ids):
                x, mask, labels, w = padded(source, group_ids, capacity, width, mean, scale, weights)
                # Distinct per-candidate scores ensure a meaningful derivative.
                logits = value*x[:, :, 275]
                total += weighted_group_loss(logits, mask, labels, w, denominator)
            return total
        self.assertAlmostEqual(loss(microbatches, 1.3), loss(compact_microbatches, 1.3), places=7)
        epsilon = .01
        baseline = (loss(microbatches, 1.3+epsilon)-loss(microbatches, 1.3-epsilon))/(2*epsilon)
        compact = (loss(compact_microbatches, 1.3+epsilon)-loss(compact_microbatches, 1.3-epsilon))/(2*epsilon)
        self.assertAlmostEqual(baseline, compact, places=7)
        seen = [i for group, _, _ in compact_microbatches(source, ids) for i in group]
        self.assertCountEqual(seen, ids.tolist())
        self.assertLess(sum(c*w for _, c, w in compact_microbatches(source, ids)),
                        sum(c*w for _, c, w in microbatches(source, ids)))

    def test_labels_are_targets_not_feature_inputs(self):
        first = data()
        second = deepcopy(first)
        second['labels'][0] = 0
        second['selected_ops'][0] = 4
        second['teacher_priority'] = [1e99]*3
        mean, scale, weights = np.zeros(373), np.ones(373), np.ones(5)
        left = padded(first, [0, 1], 2, 4, mean, scale, weights)
        right = padded(second, [0, 1], 2, 4, mean, scale, weights)
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertNotEqual(left[2][0], right[2][0])
        np.testing.assert_array_equal(numeric_block(first, 0, 5, mean, scale),
                                      numeric_block(second, 0, 5, mean, scale))

    def test_consumption_rejects_before_exceeding_shared_offline_cap(self):
        limits = dict(TRAINING, training_real_row_limit=4, training_padded_row_limit=8,
                      offline_row_limit=10, independent_offline_row_reserve=2,
                      synthetic_row_reserve=1)
        consumed = Consumption(limits)
        consumed.training(3, 4)
        snapshot = dict(consumed.counts)
        with self.assertRaises(RuntimeError):
            consumed.training(2, 4)
        self.assertEqual(consumed.counts, snapshot)
        consumed.offline('cpu', 3)
        consumed.offline('gpu', 4)
        with self.assertRaises(RuntimeError):
            consumed.offline('cpu', 1)
        self.assertEqual(consumed.counts['cpu_offline_candidate_rows'], 3)
        self.assertEqual(consumed.counts['gpu_offline_candidate_rows'], 4)

    def test_anchor_forward_has_separate_pre_call_real_and_padding_limits(self):
        for real_cap, padding_cap in ((3, 8), (8, 4)):
            consumed = Consumption(dict(TRAINING, training_real_row_limit=9,
                training_padded_row_limit=12, anchor_real_row_limit=real_cap,
                anchor_padded_row_limit=padding_cap))
            consumed.training(3, 4)
            consumed.anchor(3, 4)
            self.assertEqual(consumed.counts['gradient_evaluations'], 1)
            self.assertEqual(consumed.counts['anchor_forward_evaluations'], 1)
            snapshot = dict(consumed.counts)
            with self.assertRaises(RuntimeError):
                consumed.anchor(1, 1)
            self.assertEqual(consumed.counts, snapshot)
            consumed.training(3, 4)
            self.assertEqual(consumed.counts['training_real_candidate_rows'], 6)
            self.assertEqual(consumed.counts['anchor_real_candidate_rows'], 3)

    def test_numerical_review_groups_are_fixed_complete_and_reserved(self):
        source = lambda n, width: dict(labels=np.zeros(n, np.int64),
            offsets=np.arange(n+1, dtype=np.int64)*width)
        splits = dict(old_train=source(20, 2), old_validation=source(32, 3),
            dagger_train=source(48, 4), known_dev=source(16, 5))
        frozen = numerical_check_groups(splits)
        for name, n in (('old_train', 20), ('old_validation', 32), ('dagger_train', 48), ('known_dev', 16)):
            self.assertEqual(frozen['splits'][name]['group_indices'],
                [int(np.floor((i+.5)*n/16)) for i in range(16)])
        self.assertEqual(frozen['candidate_row_presentations'], 4*16*(2+3+4+5))
        self.assertFalse(frozen['scored_by_fit'])
        self.assertEqual(frozen['reserved_rows'], 100000)
        with self.assertRaises(ValueError):
            numerical_check_groups(dict(old_train=source(16, 391)))
        with self.assertRaises(ValueError):
            numerical_check_groups(dict(old_train=source(15, 2)))
        with self.assertRaises(ValueError):
            numerical_check_groups({str(i): source(16, 390) for i in range(5)})

    def test_streamed_cpu_scores_include_all_real_candidates(self):
        source = data()
        initial = arrays()
        params, normalizer = warm_start_arrays(initial)
        consumed = Consumption()
        scores = score_cpu(params, source, normalizer['mean'], normalizer['scale'], consumed, lambda: None)
        np.testing.assert_array_equal(scores, np.zeros(6))
        self.assertEqual(consumed.counts['cpu_offline_candidate_rows'], 6)
        metrics = conditional_metrics(scores, source, np.ones(5))
        self.assertAlmostEqual(metrics['weighted_ce'], (np.log(2)+np.log(3))/3)
        self.assertEqual(metrics['exact_top1'], 2/3)

    def test_offline_gate_requires_both_data_screens_and_export(self):
        metric = lambda ce: dict(weighted_ce=ce)
        args = [metric(1.), metric(.94), metric(2.), metric(2.04), dict(passed=True)]
        self.assertTrue(offline_gate(*args)['passed'])
        for index, replacement in ((1, metric(.951)), (3, metric(2.041)),
                                   (4, dict(passed=False)), (0, metric(0.)),
                                   (1, metric(float('nan')))):
            changed = list(args)
            changed[index] = replacement
            with self.subTest(index=index):
                self.assertFalse(offline_gate(*changed)['passed'])


if __name__ == '__main__':
    unittest.main()
