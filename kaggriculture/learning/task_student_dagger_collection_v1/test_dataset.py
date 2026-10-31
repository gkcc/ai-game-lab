"""Only synthetic examples: qualification, split safety and frozen gates."""
from dataclasses import replace
import unittest

import numpy as np

from learning.task_choice_v1.features import SCHEMA_VERSION, schema_hash
from learning.task_equiv_bc_v1.train import _batch
from learning.task_student_dagger_collection_v1 import dataset


SHARED_NAMES = ['shared.' + str(i) for i in range(275)]
CANDIDATE_NAMES = ['candidate.' + str(i) for i in range(98)]
MEAN = np.zeros(373, dtype=np.float64)
SCALE = np.ones(373, dtype=np.float64)


def comparison(status='same', *, physical=False):
    request_equal = status != 'different'
    checks = dict(request_equal=request_equal, reservations_equal=True,
                  stable_crop_equal=True, final_context_equal=True,
                  next_turn_control_equal=True)
    if status == 'unknown':
        checks['final_context_equal'] = False
    return dict(status=status, checks=checks,
                official=dict(factual_reproduced=True, request_changed=not request_equal,
                              state_changed=physical, scope='one_official_transition',
                              complete_game=False, future_return_evaluated=False))


def inputs(value=1.0, count=3):
    shared = [0.0] * 275
    shared[0] = value
    candidates = []
    for index in range(count):
        x = [0.0] * 98
        x[0] = float(index)
        candidates.append(dict(id='id' + str(index), x=x))
    return dict(schema_version=SCHEMA_VERSION,
                schema_sha256=schema_hash(CANDIDATE_NAMES, SHARED_NAMES),
                shared_feature_names=SHARED_NAMES.copy(), feature_names=CANDIDATE_NAMES.copy(),
                shared_x=shared, candidates=candidates,
                observation={'private': 'not-retained'})


def example(*, role='T01', opponent='market', side=0, window=0, step=None,
            ordinal=0, value=1.0, task_inputs=None, teacher_index=0,
            student_index=1, third_index=2, teacher_summary=None, third_summary=None):
    if step is None:
        step = dataset.WINDOWS[window][0]
    if teacher_summary is None:
        teacher_summary = comparison('same')
    if third_summary is None and third_index is not None:
        third_summary = comparison('different', physical=True)
    return dataset.make_example(role=role, opponent=opponent, side=side, window=window,
        step=step, ordinal=ordinal, inputs=task_inputs if task_inputs is not None else inputs(value),
        teacher_index=teacher_index, student_index=student_index, third_index=third_index,
        teacher_summary=teacher_summary, third_summary=third_summary)


def passing_examples():
    result = []
    for role_index, role in enumerate(dataset.ROLES):
        for i in range(4):
            result.append(example(role=role, opponent=('market' if i % 2 else 'round2'),
                                  side=i // 2, window=i % 2, ordinal=i,
                                  value=100 * role_index + i + 10))
    return result


class DatasetTests(unittest.TestCase):
    def test_qualification_keeps_only_strictly_verified(self):
        raw = inputs()
        item = example(task_inputs=raw)
        self.assertEqual(item.features.shape, (3, 373))
        self.assertEqual(item.features.dtype, np.dtype('float32'))
        self.assertFalse(item.features.flags.writeable)
        self.assertNotIn('not-retained', repr(item))
        raw['shared_x'][0] = 99
        self.assertEqual(float(item.features[0, 0]), 1.0)
        with self.assertRaises(ValueError):
            item.features[0, 0] = 4
        self.assertIsNone(example(teacher_summary=comparison('unknown')))
        self.assertIsNone(example(teacher_summary=comparison('different', physical=True)))
        self.assertIsNone(example(third_summary=comparison('unknown')))
        self.assertIsNone(example(third_summary=comparison('different', physical=False)))
        self.assertIsNone(example(task_inputs=inputs(count=2), third_index=None))
        bad = comparison('same')
        bad['official']['state_changed'] = True
        with self.assertRaises(ValueError):
            example(teacher_summary=bad)
        with self.assertRaises(ValueError):
            example(third_index=1)

    def test_raw_float32_collision_removes_dev_and_conflict_removes_all(self):
        train = example(role='T01', value=1.0)
        dev = example(role='V01', value=1.0 + 2 ** -25)
        self.assertNotEqual(train._float64_bytes, dev._float64_bytes)
        self.assertEqual(train.features.tobytes(), dev.features.tobytes())
        t, d, summary = dataset.build_dataset([dev, train], mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d)), (1, 0))
        self.assertEqual(summary['cross_split_dev_removed'], 1)
        self.assertEqual(summary['deployment_precision_collision_groups'], 1)
        conflict = example(role='V01', value=1.0 + 2 ** -25,
                           teacher_index=1, student_index=0, third_index=2)
        t, d, summary = dataset.build_dataset([train, conflict], mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d)), (0, 0))
        self.assertEqual((summary['conflict_groups'], summary['conflict_examples_removed']), (1, 2))

    def test_train_network_collision_closure_matches_actual_batch(self):
        mu = MEAN.copy()
        mu[0] = 100_000_000
        first = example(role='T01', value=1.0)
        second = example(role='T02', value=1.0 + 2 ** -25)
        third = example(role='V01', value=2.0, teacher_index=1, student_index=0)
        self.assertNotEqual(first.features.tobytes(), third.features.tobytes())
        groups = []
        for item in (first, third):
            positive = np.zeros(3, bool)
            positive[item.teacher_index] = True
            groups.append(dict(features=item.features, positive_mask=positive,
                               verified_mask=positive.copy(), teacher_index=item.teacher_index, weight=1.0))
        actual, *_ = _batch(groups, [0, 1], mu, SCALE)
        self.assertTrue(np.array_equal(actual[0], actual[1]))
        t, d, summary = dataset.build_dataset([third, first, second], mean=mu, scale=SCALE)
        self.assertEqual((len(t), len(d)), (0, 0))
        self.assertEqual(summary['conflict_examples_removed'], 3)
        self.assertEqual(summary['normalization_collision_groups'], 1)
        same_label = example(role='V01', value=2.0)
        t, d, summary = dataset.build_dataset([first, same_label], mean=mu, scale=SCALE)
        self.assertEqual((len(t), len(d), summary['cross_split_dev_removed']), (1, 0, 1))

    def test_fixed_order_zero_and_candidate_order(self):
        late = example(role='T02', value=-0.0)
        early = example(role='T01', value=0.0)
        t, d, summary = dataset.build_dataset([late, early], mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d), t[0].role), (1, 0, 'T01'))
        self.assertEqual(summary['same_split_train_removed'], 1)
        t2, _, _ = dataset.build_dataset([early, late], mean=MEAN, scale=SCALE)
        self.assertEqual(t2[0].role, t[0].role)
        altered = inputs(value=0.0)
        altered['candidates'][0], altered['candidates'][1] = altered['candidates'][1], altered['candidates'][0]
        other = example(role='V01', task_inputs=altered)
        t, d, _ = dataset.build_dataset([early, other], mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d)), (1, 1))

    def test_all_six_role_gates_after_global_dedup(self):
        source = passing_examples()
        t, d, summary = dataset.build_dataset(source, mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d), summary['status']), (16, 8, 'passed'))
        self.assertEqual(summary['train_by_role'], dict.fromkeys(dataset.TRAIN_ROLES, 4))
        self.assertEqual(summary['dev_by_role'], dict.fromkeys(dataset.DEV_ROLES, 4))
        self.assertEqual(sum(summary['train_by_opponent'].values()), 16)
        self.assertEqual(sum(summary['dev_by_window'].values()), 8)
        old_dev = source[16]
        source[16] = example(role=old_dev.role, opponent=old_dev.opponent, side=old_dev.side,
                             window=old_dev.window, ordinal=old_dev.ordinal,
                             value=float(source[0].features[0, 0]))
        t, d, summary = dataset.build_dataset(source, mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d), summary['status']), (16, 7, 'failed'))
        self.assertEqual(summary['cross_split_dev_removed'], 1)

    def test_coverage_gates_enforce_roles_opponents_and_time(self):
        source = passing_examples()
        # Keep at least 16/8 groups while leaving one training role short.
        spare = example(role='T01', opponent='market', side=1, window=2, value=999)
        source = source[:12] + source[13:] + [spare]
        _, _, summary = dataset.build_dataset(source, mean=MEAN, scale=SCALE)
        self.assertEqual(summary['train_groups'], 16)
        self.assertEqual(summary['train_by_role']['T04'], 3)
        self.assertEqual(summary['status'], 'failed')
        source = passing_examples()
        source = [example(role=e.role, opponent=('market' if e.role in dataset.DEV_ROLES else e.opponent),
                          side=e.side, window=e.window, step=e.step, ordinal=e.ordinal,
                          value=float(e.features[0, 0])) for e in source]
        _, _, summary = dataset.build_dataset(source, mean=MEAN, scale=SCALE)
        self.assertEqual((summary['dev_groups'], summary['dev_opponents'], summary['status']),
                         (8, 1, 'failed'))

    def test_invalid_inputs_and_multiple_per_window_fail_closed(self):
        for value in (float('nan'), float('inf'), 1e100, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                example(value=value)
        for bad_count in (1, 257):
            with self.subTest(count=bad_count), self.assertRaises(ValueError):
                example(task_inputs=inputs(count=bad_count))
        bad = inputs()
        bad['candidates'][2]['id'] = bad['candidates'][0]['id']
        with self.assertRaises(ValueError):
            example(task_inputs=bad)
        bad = inputs()
        bad['feature_names'][0] = 'tampered'
        with self.assertRaises(ValueError):
            example(task_inputs=bad)
        for role in ('F01', 'V03', '', None):
            with self.subTest(role=role), self.assertRaises(ValueError):
                example(role=role)
        with self.assertRaises(ValueError):
            example(window=1, step=0)
        with self.assertRaises(ValueError):
            dataset.build_dataset([example(step=0), example(step=1)], mean=MEAN, scale=SCALE)
        with self.assertRaises(ValueError):
            dataset.build_dataset([object()], mean=MEAN, scale=SCALE)
        valid = example()
        with self.assertRaises(ValueError):
            dataset.build_dataset([replace(valid, _validated=None)], mean=MEAN, scale=SCALE)
        with self.assertRaises(ValueError):
            dataset.build_dataset([replace(valid, teacher_index=valid.third_index)], mean=MEAN, scale=SCALE)
        with self.assertRaises(ValueError):
            dataset.build_dataset([replace(valid, teacher_index=1, student_index=0)], mean=MEAN, scale=SCALE)
        with self.assertRaises(ValueError):
            dataset.build_dataset([replace(valid, role='V01')], mean=MEAN, scale=SCALE)
        bad_scale = SCALE.copy()
        bad_scale[0] = 0
        with self.assertRaises(ValueError):
            dataset.build_dataset([], mean=MEAN, scale=bad_scale)

    def test_summary_contains_only_aggregate_and_fingerprints_remain_private(self):
        train, dev, summary = dataset.build_dataset(passing_examples() + [None], mean=MEAN, scale=SCALE)
        self.assertEqual(summary['ineligible_count'], 1)
        self.assertFalse(summary['training'])
        self.assertFalse(summary['independent_evaluation'])
        self.assertTrue(all(len(e._deployment_fingerprint) == 32
                            and len(e._training_fingerprint) == 32 for e in train + dev))
        for forbidden in ('fingerprint', 'private', 'seed', 'features'):
            self.assertNotIn(forbidden, repr(summary))


if __name__ == '__main__':
    unittest.main()
