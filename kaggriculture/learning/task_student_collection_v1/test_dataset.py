"""Pure synthetic checks for qualification, collision control and split gates."""
from dataclasses import replace
import unittest

import numpy as np

from learning.task_choice_v1.features import SCHEMA_VERSION, schema_hash
from learning.task_equiv_bc_v1.train import _batch
from learning.task_student_collection_v1 import dataset


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
                shared_feature_names=SHARED_NAMES.copy(),
                feature_names=CANDIDATE_NAMES.copy(), shared_x=shared,
                candidates=candidates, observation={'private': 'must-not-be-retained'})


def example(*, role='F01', opponent='market', side=0, window=0, step=None,
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
        length = 8 if role_index < 2 else 4
        for i in range(length):
            result.append(example(role=role, opponent=('market' if i % 2 else 'round2'),
                                  side=i % 2, window=i % 2, ordinal=i,
                                  value=100 * role_index + i + 10))
    return result


class DatasetTests(unittest.TestCase):
    def test_qualification_schema_and_private_features(self):
        raw = inputs()
        item = example(task_inputs=raw)
        self.assertEqual(item.features.shape, (3, 373))
        self.assertEqual(item.features.dtype, np.dtype('float32'))
        self.assertFalse(item.features.flags.writeable)
        self.assertNotIn('must-not-be-retained', repr(item))
        raw['shared_x'][0] = 99
        self.assertEqual(float(item.features[0, 0]), 1.0)
        with self.assertRaises(ValueError):
            item.features[0, 0] = 4
        self.assertIsNone(example(teacher_summary=comparison('unknown')))
        self.assertIsNone(example(third_summary=comparison('unknown')))
        self.assertIsNone(example(third_summary=comparison('different', physical=False)))
        two = inputs(count=2)
        self.assertIsNone(example(task_inputs=two, third_index=None))
        bad = comparison('same')
        bad['official']['state_changed'] = True
        with self.assertRaises(ValueError):
            example(teacher_summary=bad)
        with self.assertRaises(ValueError):
            example(third_index=1)

    def test_float32_collision_cross_split_and_conflicting_triple(self):
        train = example(role='F01', value=1.0)
        dev = example(role='F03', value=1.0 + 2 ** -25)
        train_out, dev_out, summary = dataset.build_dataset([dev, train], mean=MEAN, scale=SCALE)
        self.assertEqual((len(train_out), len(dev_out)), (1, 0))
        self.assertEqual(summary['cross_split_dev_removed'], 1)
        self.assertEqual(summary['deployment_precision_collision_groups'], 1)
        self.assertNotEqual(train._float64_bytes, dev._float64_bytes)
        self.assertEqual(train.features.tobytes(), dev.features.tobytes())
        conflicting = example(role='F03', value=1.0 + 2 ** -25,
                              teacher_index=1, student_index=0, third_index=2)
        t, d, summary = dataset.build_dataset([train, conflicting], mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d)), (0, 0))
        self.assertEqual((summary['conflict_groups'], summary['conflict_examples_removed']), (1, 2))

    def test_normalized_float32_collision_closure_drops_whole_conflict(self):
        mu = MEAN.copy()
        mu[0] = 100_000_000
        first = example(role='F01', value=1.0)
        second = example(role='F02', value=1.0 + 2 ** -25)
        third = example(role='F03', value=2.0, teacher_index=1, student_index=0)
        self.assertNotEqual(first.features.tobytes(), third.features.tobytes())
        self.assertEqual(np.float32(1.0 - mu[0]), np.float32(2.0 - mu[0]))
        t, d, summary = dataset.build_dataset([third, second, first], mean=mu, scale=SCALE)
        self.assertEqual((len(t), len(d)), (0, 0))
        self.assertEqual(summary['conflict_examples_removed'], 3)
        self.assertEqual(summary['normalization_collision_groups'], 1)

    def test_normalized_collision_matches_actual_training_batch(self):
        mu = MEAN.copy()
        mu[0] = 100_000_000
        train = example(role='F01', value=1.0)
        dev = example(role='F03', value=2.0)
        groups = []
        for item in (train, dev):
            mask = np.array([True, False, False])
            groups.append(dict(features=item.features, positive_mask=mask,
                               verified_mask=mask.copy(), teacher_index=0, weight=1.0))
        actual, *_ = _batch(groups, [0, 1], mu, SCALE)
        self.assertTrue(np.array_equal(actual[0], actual[1]))
        t, d, summary = dataset.build_dataset([train, dev], mean=mu, scale=SCALE)
        self.assertEqual((len(t), len(d), summary['cross_split_dev_removed']), (1, 0, 1))

    def test_order_zero_and_ties_use_coordinates_not_input_order(self):
        one = example(role='F02', ordinal=2, value=-0.0)
        two = example(role='F01', ordinal=3, value=0.0)
        t, d, summary = dataset.build_dataset([one, two], mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d)), (1, 0))
        self.assertEqual(t[0].role, 'F01')
        self.assertEqual(summary['same_split_train_removed'], 1)
        reversed_t, _, _ = dataset.build_dataset([two, one], mean=MEAN, scale=SCALE)
        self.assertEqual(reversed_t[0].role, t[0].role)
        changed = inputs(value=0.0)
        changed['candidates'][0], changed['candidates'][1] = changed['candidates'][1], changed['candidates'][0]
        other = example(role='F03', task_inputs=changed)
        t, d, _ = dataset.build_dataset([one, other], mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d)), (1, 1))

    def test_gates_after_deduplication_and_role_and_window_coverage(self):
        source = passing_examples()
        t, d, summary = dataset.build_dataset(source, mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d), summary['status']), (16, 8, 'passed'))
        self.assertEqual(summary['train_by_role'], {'F01': 8, 'F02': 8})
        self.assertEqual(summary['dev_by_role'], {'F03': 4, 'F04': 4})
        # Replace one development group by an actual raw training input.
        src = source[0]
        leaked = example(role='F03', opponent=source[16].opponent, side=source[16].side,
                         window=source[16].window, ordinal=source[16].ordinal,
                         value=float(src.features[0, 0]))
        source[16] = leaked
        t, d, summary = dataset.build_dataset(source, mean=MEAN, scale=SCALE)
        self.assertEqual((len(t), len(d), summary['status']), (16, 7, 'failed'))
        self.assertEqual(summary['cross_split_dev_removed'], 1)
        # Keep enough samples but collapse development coverage to one opponent.
        source = passing_examples()
        source = [example(role=e.role, opponent=('market' if e.role in dataset.DEV_ROLES else e.opponent),
                          side=e.side, window=e.window, step=e.step, ordinal=e.ordinal,
                          value=float(e.features[0, 0])) for e in source]
        _, _, summary = dataset.build_dataset(source, mean=MEAN, scale=SCALE)
        self.assertEqual(summary['dev_groups'], 8)
        self.assertEqual(summary['dev_opponents'], 1)
        self.assertEqual(summary['status'], 'failed')

    def test_invalid_inputs_do_not_enter_dataset(self):
        for value in (float('nan'), float('inf'), 1e100, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                example(value=value)
        for bad_count in (1, 257):
            with self.subTest(count=bad_count), self.assertRaises(ValueError):
                example(task_inputs=inputs(count=bad_count))
        inp = inputs()
        inp['candidates'][2]['id'] = inp['candidates'][0]['id']
        with self.assertRaises(ValueError):
            example(task_inputs=inp)
        inp = inputs()
        inp['feature_names'][0] = 'tampered'
        with self.assertRaises(ValueError):
            example(task_inputs=inp)
        with self.assertRaises(ValueError):
            example(window=1, step=0)
        with self.assertRaises(ValueError):
            dataset.build_dataset([example(), example()], mean=MEAN, scale=SCALE)
        with self.assertRaises(ValueError):
            dataset.build_dataset([object()], mean=MEAN, scale=SCALE)
        valid = example()
        with self.assertRaises(ValueError):
            dataset.build_dataset([replace(valid, _validated=None)], mean=MEAN, scale=SCALE)
        with self.assertRaises(ValueError):
            dataset.build_dataset([replace(valid, teacher_index=valid.third_index)],
                                  mean=MEAN, scale=SCALE)
        altered = np.zeros(valid.features.shape, dtype=np.float32)
        altered.flags.writeable = False
        with self.assertRaises(ValueError):
            dataset.build_dataset([replace(valid, features=altered)], mean=MEAN, scale=SCALE)
        bad_scale = SCALE.copy()
        bad_scale[0] = 0
        with self.assertRaises(ValueError):
            dataset.build_dataset([], mean=MEAN, scale=bad_scale)
        with self.assertRaises(ValueError):
            dataset.build_dataset([], mean=[True] * 373, scale=SCALE)

    def test_training_network_fingerprints_and_summary_privacy(self):
        source = passing_examples()
        train, dev, summary = dataset.build_dataset(source + [None], mean=MEAN, scale=SCALE)
        self.assertEqual(summary['ineligible_count'], 1)
        self.assertFalse(summary['training'])
        self.assertFalse(summary['independent_evaluation'])
        self.assertTrue(all(len(e._deployment_fingerprint) == 32 and
                            len(e._training_fingerprint) == 32 for e in train + dev))
        encoded = repr(summary)
        for forbidden in ('fingerprint', 'private', 'seed', 'features'):
            self.assertNotIn(forbidden, encoded)


if __name__ == '__main__':
    unittest.main()
