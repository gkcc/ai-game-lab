"""Synthetic regressions for mixed, seed-split conditional examples."""
from dataclasses import replace
import unittest

import numpy as np

from learning.task_student_collection_v1.dataset import DIM, SHARED_DIM
from learning.task_student_dagger_collection_v1.sampling import WINDOWS
from . import merge as subject


def group(role, *, number=1, opponent='market', side=0, window=0,
          labels=(0, 1, 2), features=None):
    if features is None:
        features = np.full((3, DIM), number, dtype=np.float32)
    return subject.Group(role, opponent, side, window, WINDOWS[window][0], 0,
                         features, *labels)


def assorted(role_counts, *, start=1, opponent=None, side=None, windows=6):
    groups = []
    value = start
    for role, count in role_counts:
        for index in range(count):
            window = index % windows
            block = index // windows
            # Up to four distinct seats/opponents per role/window.
            match_opponent = opponent if opponent is not None else ('market', 'round2')[(window + block) % 2]
            match_side = side if side is not None else ((window // 2 + block // 2) % 2)
            groups.append(group(role, number=value, opponent=match_opponent,
                                side=match_side, window=window))
            value += 1
    return groups


def good_groups():
    train = assorted([(f'A{i:02}', 6 if i <= 2 else 5) for i in range(1, 7)])
    dev = assorted([(f'A{i:02}', 4) for i in range(7, 11)], start=100)
    return train + dev


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.zeros(DIM, dtype=np.float64)
        self.scale = np.ones(DIM, dtype=np.float64)

    def merge(self, groups):
        return subject.merge(groups, mean=self.mean, scale=self.scale)

    def test_gate_pass_and_immutable_results(self):
        groups = good_groups()
        train, dev, result = self.merge(groups)
        self.assertEqual((len(train), len(dev), result['status']), (32, 16, 'passed'))
        self.assertTrue(all(all(side.values()) for side in result['gate_checks'].values()))
        self.assertEqual(result['train_distribution']['roles']['F01'], 0)
        self.assertEqual(sum(result['dev_distribution']['windows'].values()), 16)
        previous = float(train[0].features[0, 0])
        groups[0].features[:] = -999
        self.assertEqual(float(train[0].features[0, 0]), previous)
        self.assertFalse(train[0].features.flags.writeable)
        with self.assertRaises(ValueError):
            train[0].features[0, 0] = 9
        self.assertNotIn('array', repr(train[0]))

    def test_raw_cross_split_and_same_split_duplicates(self):
        games = [group('F03'), group('F02'), group('F04'), group('F01')]
        train, dev, summary = self.merge(games)
        self.assertEqual([item.role for item in train], ['F01'])
        self.assertEqual(dev, ())
        self.assertEqual(summary['same_split_train_removed'], 1)
        self.assertEqual(summary['cross_split_components'], 1)
        self.assertEqual(summary['cross_split_dev_removed'], 2)
        _, dev, summary = self.merge([group('F04'), group('F03')])
        self.assertEqual([item.role for item in dev], ['F03'])
        self.assertEqual(summary['same_split_dev_removed'], 1)

    def test_normalized_cross_split_duplicate(self):
        self.mean[0] = 2**30
        first = np.ones((3, DIM), dtype=np.float32)
        second = first.copy()
        second[:, 0] = 2
        self.assertFalse(np.array_equal(first, second))
        train, dev, summary = self.merge([group('F01', features=first),
                                          group('F03', features=second)])
        self.assertEqual((len(train), len(dev), summary['cross_split_dev_removed']), (1, 0, 1))

    def test_transitive_collision_with_conflicting_labels_discards_all(self):
        self.mean[0] = 2**30
        first = np.ones((3, DIM), dtype=np.float32)
        third = first.copy()
        third[:, 0] = 2
        games = [group('F01', features=first), group('F02', features=first.copy()),
                 group('F03', features=third, labels=(0, 2, 1))]
        train, dev, summary = self.merge(games)
        self.assertEqual((len(train), len(dev)), (0, 0))
        self.assertEqual((summary['conflict_components'], summary['conflict_groups_removed']), (1, 3))
        self.assertEqual(summary['cross_split_dev_removed'], 0)

    def test_signed_zero_is_same_raw_input(self):
        first = np.ones((3, DIM), dtype=np.float32)
        second = first.copy()
        first[:, 0], second[:, 0] = 0.0, -0.0
        self.assertNotEqual(first.tobytes(), second.tobytes())
        _, _, summary = self.merge([group('F01', features=first), group('F03', features=second)])
        self.assertEqual(summary['cross_split_dev_removed'], 1)

    def test_invalid_coordinates_and_duplicate_source_window(self):
        item = group('F01')
        invalid = [replace(item, role='F05'), replace(item, opponent='unknown'),
                   replace(item, side=True), replace(item, side=2),
                   replace(item, window=6), replace(item, step=120), replace(item, ordinal=-1)]
        for bad in invalid:
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                self.merge([bad])
        with self.assertRaises(ValueError):
            self.merge([item, replace(item, ordinal=1)])

    def test_invalid_features_labels_and_normalization(self):
        item = group('F01')
        invalid = [replace(item, features=np.zeros((3, DIM), dtype=np.float64)),
                   replace(item, features=np.zeros((2, DIM), dtype=np.float32)),
                   replace(item, features=np.zeros((257, DIM), dtype=np.float32)),
                   replace(item, features=np.zeros((3, DIM - 1), dtype=np.float32)),
                   replace(item, teacher_index=True), replace(item, student_index=0),
                   replace(item, third_index=3)]
        nonfinite = np.zeros((3, DIM), dtype=np.float32)
        nonfinite[0, 0] = np.nan
        invalid.append(replace(item, features=nonfinite))
        for bad in invalid:
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                self.merge([bad])
        with self.assertRaises(ValueError):
            subject.merge([item], mean=self.mean, scale=np.zeros(DIM))
        with self.assertRaises(ValueError):
            self.merge([item] * (subject.MAX_GROUPS + 1))
        with self.assertRaises(ValueError):
            self.merge(assorted([('F01', 10)]))

    def test_shared_features_match_across_candidates(self):
        item = group('F01')
        inconsistent = item.features.copy()
        inconsistent[1, SHARED_DIM - 1] += 1
        with self.assertRaises(ValueError):
            self.merge([replace(item, features=inconsistent)])
        differing_candidate = item.features.copy()
        differing_candidate[1, SHARED_DIM] += 1
        train, _, _ = self.merge([replace(item, features=differing_candidate)])
        self.assertEqual(len(train), 1)

    def test_each_split_minimum_count(self):
        games = good_groups()
        _, _, empty = self.merge([])
        self.assertFalse(empty['gate_checks']['train']['max_role_share'])
        self.assertFalse(empty['gate_checks']['dev']['max_role_share'])
        _, _, summary = self.merge(games[1:])
        self.assertFalse(summary['gate_checks']['train']['minimum_groups'])
        _, _, summary = self.merge(games[:-1])
        self.assertFalse(summary['gate_checks']['dev']['minimum_groups'])

    def test_each_split_minimum_covered_roles(self):
        dev = good_groups()[32:]
        small_train = assorted([(f'A{i:02}', 8) for i in range(1, 5)])
        _, _, summary = self.merge(small_train + dev)
        self.assertFalse(summary['gate_checks']['train']['minimum_roles'])
        train = good_groups()[:32]
        small_dev = assorted([('A07', 6), ('A08', 5), ('A09', 5)], start=100)
        _, _, summary = self.merge(train + small_dev)
        self.assertFalse(summary['gate_checks']['dev']['minimum_roles'])

    def test_each_split_maximum_role_share(self):
        concentrated = assorted([('A01', 12)] + [(f'A{i:02}', 4) for i in range(2, 7)])
        _, _, summary = self.merge(concentrated + good_groups()[32:])
        self.assertFalse(summary['gate_checks']['train']['max_role_share'])
        concentrated_dev = assorted([('A07', 7), ('A08', 3), ('A09', 3), ('A10', 3)], start=100)
        _, _, summary = self.merge(good_groups()[:32] + concentrated_dev)
        self.assertFalse(summary['gate_checks']['dev']['max_role_share'])
        train_boundary = assorted([('A01', 11), ('A02', 5), ('A03', 5),
                                   ('A04', 4), ('A05', 4), ('A06', 4)])
        dev_boundary = assorted([('A07', 6), ('A08', 4), ('A09', 4), ('A10', 4)], start=100)
        _, _, summary = self.merge(train_boundary + dev_boundary)
        self.assertTrue(summary['gate_checks']['train']['max_role_share'])
        self.assertTrue(summary['gate_checks']['dev']['max_role_share'])

    def test_each_split_requires_both_opponents_and_sides(self):
        counts_train = [(f'A{i:02}', 6 if i <= 2 else 5) for i in range(1, 7)]
        counts_dev = [(f'A{i:02}', 4) for i in range(7, 11)]
        for restricted in ('opponent', 'side'):
            value = 'market' if restricted == 'opponent' else 0
            key = 'both_opponents' if restricted == 'opponent' else 'both_sides'
            train = assorted(counts_train, **{restricted: value})
            _, _, summary = self.merge(train + good_groups()[32:])
            self.assertFalse(summary['gate_checks']['train'][key])
            dev = assorted(counts_dev, start=100, **{restricted: value})
            _, _, summary = self.merge(good_groups()[:32] + dev)
            self.assertFalse(summary['gate_checks']['dev'][key])

    def test_opponent_and_side_minimum_are_two_groups(self):
        for restricted in ('opponent', 'side'):
            default = 'market' if restricted == 'opponent' else 0
            other = 'round2' if restricted == 'opponent' else 1
            key = 'both_opponents' if restricted == 'opponent' else 'both_sides'
            for split, begin, end in (('train', 0, 32), ('dev', 32, 48)):
                original = good_groups()
                for i in range(begin, end):
                    original[i] = replace(original[i], **{restricted: default})
                original[begin] = replace(original[begin], **{restricted: other})
                _, _, result = self.merge(original)
                self.assertFalse(result['gate_checks'][split][key])
                original[begin + 1] = replace(original[begin + 1], **{restricted: other})
                _, _, result = self.merge(original)
                self.assertTrue(result['gate_checks'][split][key])

    def test_each_split_requires_three_windows(self):
        counts_train = [(f'A{i:02}', 6 if i <= 2 else 5) for i in range(1, 7)]
        counts_dev = [(f'A{i:02}', 4) for i in range(7, 11)]
        train = assorted(counts_train, windows=2)
        _, _, summary = self.merge(train + good_groups()[32:])
        self.assertFalse(summary['gate_checks']['train']['three_windows'])
        dev = assorted(counts_dev, start=100, windows=2)
        _, _, summary = self.merge(good_groups()[:32] + dev)
        self.assertFalse(summary['gate_checks']['dev']['three_windows'])


if __name__ == '__main__':
    unittest.main()
