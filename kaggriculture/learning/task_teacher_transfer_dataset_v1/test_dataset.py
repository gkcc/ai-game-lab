"""Synthetic regressions for private transfer data and the new coverage gate."""
from dataclasses import replace
import json
import unittest

import numpy as np

from . import dataset as d


def matrix(value, *, op='WATER', size=3):
    x = np.zeros((size, d.DIM), dtype=np.float32)
    x[:, :d.SHARED_DIM] = value
    x[:, d.SHARED_DIM] = np.arange(size, dtype=np.float32)
    x[:, d._OP_START + d.OPS.index(op)] = 1
    return x


def group(role='A01', *, opponent='market', side=0, step=48, teacher_ordinal=0,
          terminal_ordinal=0, op='WATER', value=1, features=None, target=0):
    if features is None:
        features = matrix(value, op=op)
    return d.Group(role, opponent, side, step, teacher_ordinal, terminal_ordinal,
                   op, features, target)


def coordinates(*, opponents=d.OPPONENTS, sides=(0, 1), steps=d.STEPS):
    result = []
    # Deterministically interleave hours and sides before adding another
    # combination; this generates balanced synthetic inputs, not real data.
    for block in range(4):
        for i, step in enumerate(d.STEPS):
            opponent = d.OPPONENTS[(i + block) % 2]
            side = (i // 2 + block // 2) % 2
            candidate = (opponent, side, step)
            if opponent in opponents and side in sides and step in steps and candidate not in result:
                result.append(candidate)
    return result


def make_groups(role_counts, *, start=1, opponents=d.OPPONENTS, sides=(0, 1), steps=d.STEPS,
                one_operation=False):
    coords = coordinates(opponents=opponents, sides=sides, steps=steps)
    result, value = [], start
    for role, count in role_counts:
        for index in range(count):
            opponent, side, step = coords[index % len(coords)]
            ordinal = index // len(coords)
            if ordinal >= 4:
                raise ValueError('Synthetic source exceeds per-turn fixture budget')
            op = 'WATER' if one_operation or value % 2 else 'DIG'
            result.append(group(role, opponent=opponent, side=side, step=step,
                                teacher_ordinal=ordinal, terminal_ordinal=ordinal,
                                op=op, value=value))
            value += 1
    return result


def good_groups():
    return (make_groups([('A01', 16), ('A02', 16)])
            + make_groups([('A07', 8), ('A08', 8)], start=100))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.zeros(d.DIM, dtype=np.float64)
        self.scale = np.ones(d.DIM, dtype=np.float64)

    def merge(self, groups, guard=lambda: None):
        return d.merge(groups, mean=self.mean, scale=self.scale, guard=guard)

    def test_fixed_public_rank_is_bounded_and_independent_of_labels(self):
        result = d.rank_ordinals('A01', 'market', 0, 48, 32)
        self.assertEqual(result, d.rank_ordinals('A01', 'market', 0, 48, 32))
        self.assertEqual(len(result), 4)
        self.assertEqual(len(set(result)), 4)
        self.assertTrue(all(0 <= index < 32 for index in result))
        self.assertEqual(d.rank_ordinals('A01', 'market', 0, 48, 0), ())
        self.assertEqual(set(d.rank_ordinals('A01', 'market', 0, 48, 3)), {0, 1, 2})
        for kwargs in (dict(role='A03'), dict(side=True), dict(step=59), dict(total=33), dict(total=-1)):
            args = dict(role='A01', opponent='market', side=0, step=48, total=4)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                d.rank_ordinals(**args)

    def test_good_data_gate_and_private_immutable_copy(self):
        source = good_groups()
        train, dev, summary = self.merge(source)
        self.assertEqual((len(train), len(dev), summary['status']), (32, 16, 'passed'))
        self.assertTrue(all(all(group.values()) for group in summary['gate_checks'].values()))
        self.assertEqual(summary['train_distribution']['roles'], {'A01': 16, 'A02': 16})
        self.assertEqual(summary['dev_distribution']['roles'], {'A07': 8, 'A08': 8})
        self.assertNotIn('features', repr(train[0]))
        self.assertNotIn('target_index', json.dumps(summary))
        prior = train[0].features.copy()
        source_group = next(item for item in source if item.role == train[0].role
                            and item.opponent == train[0].opponent and item.side == train[0].side
                            and item.step == train[0].step and item.teacher_ordinal == train[0].teacher_ordinal)
        source_group.features[:] = -5
        np.testing.assert_array_equal(train[0].features, prior)
        self.assertFalse(train[0].features.flags.writeable)

    def test_actual_synthetic_extractor_groups_enter_the_merge(self):
        from learning.task_teacher_support_probe_v1.test_alignment import make_stream
        from .extract import extract_turn
        observation, configuration, records, schema = make_stream()
        selected = set(d.rank_ordinals('A01', 'market', 0, 48, len(records)))
        extracted, counts = extract_turn(observation, configuration, records, records,
                                         schema=schema, guard=lambda: None)
        groups = [d.Group('A01', 'market', 0, 48, item.teacher_ordinal, item.terminal_ordinal,
                          item.operation, item.features, item.label)
                  for item in extracted if item.teacher_ordinal in selected]
        train, dev, summary = self.merge(groups)
        self.assertEqual(counts['supported'], 2)
        self.assertEqual(summary['source_groups'], 2)
        self.assertEqual((len(train), len(dev), summary['singleton_groups_removed']), (1, 0, 1))

    def test_train_wins_raw_duplicate_and_representative_is_order_independent(self):
        x = matrix(1)
        groups = [group('A07', features=x.copy()), group('A02', features=x.copy()),
                  group('A01', features=x.copy())]
        train, dev, summary = self.merge(groups)
        train2, dev2, _ = self.merge(list(reversed(groups)))
        self.assertEqual((len(train), len(dev)), (1, 0))
        self.assertEqual((train[0].role, train[0].teacher_ordinal),
                         (train2[0].role, train2[0].teacher_ordinal))
        self.assertEqual(dev2, ())
        self.assertEqual((summary['same_split_train_removed'], summary['cross_split_dev_removed']), (1, 1))

    def test_transitive_normalized_collision_discards_conflicting_labels(self):
        self.mean[0] = 2**30
        x = matrix(1)
        y = x.copy()
        y[:, 0] = 2
        groups = [group('A01', features=x), group('A02', features=x.copy()),
                  group('A07', features=y, target=1)]
        train, dev, summary = self.merge(groups)
        self.assertEqual((len(train), len(dev)), (0, 0))
        self.assertEqual((summary['conflict_components'], summary['conflict_groups_removed']), (1, 3))

    def test_signed_zero_and_clean_candidate_row_overlap(self):
        left = matrix(1)
        right = left.copy()
        left[:, 0], right[:, 0] = 0.0, -0.0
        _, dev, summary = self.merge([group('A01', features=left), group('A07', features=right)])
        self.assertEqual(len(dev), 0)
        self.assertEqual(summary['cross_split_dev_removed'], 1)
        right = left.copy()
        right[1, d.SHARED_DIM] += 10
        train, dev, summary = self.merge([group('A01', features=left), group('A07', features=right)])
        self.assertEqual((len(train), len(dev)), (1, 1))
        self.assertEqual((summary['dev_groups_with_raw_row_overlap'],
                          summary['dev_groups_with_normalized_row_overlap'],
                          summary['dev_groups_with_any_row_overlap'],
                          summary['dev_groups_without_row_overlap']), (1, 1, 1, 0))

    def test_normalized_only_candidate_row_overlap(self):
        self.mean[0] = 2**30
        left, right = matrix(1), matrix(1)
        right[:, 0] = 2
        right[1, d.SHARED_DIM] += 10
        _, dev, summary = self.merge([group('A01', features=left), group('A07', features=right)])
        self.assertEqual(len(dev), 1)
        self.assertEqual((summary['dev_groups_with_raw_row_overlap'],
                          summary['dev_groups_with_normalized_row_overlap'],
                          summary['dev_groups_with_any_row_overlap']), (0, 1, 1))

    def test_single_candidate_has_no_distinguishing_training_signal(self):
        groups = good_groups()
        groups[0] = replace(groups[0], features=groups[0].features[:1].copy())
        train, dev, summary = self.merge(groups)
        self.assertEqual((len(train), len(dev), summary['singleton_groups_removed']), (31, 16, 1))
        self.assertFalse(summary['gate_checks']['train']['minimum_groups'])
        self.assertEqual(summary['status'], 'failed')

    def test_normalized_identical_candidates_have_no_choice_signal(self):
        self.mean[d.SHARED_DIM] = 2**30
        raw = matrix(1, size=2)
        self.assertFalse(np.array_equal(raw[0], raw[1]))
        train, dev, summary = self.merge([group(features=raw)])
        self.assertEqual((len(train), len(dev)), (0, 0))
        self.assertEqual(summary['indistinguishable_groups_removed'], 1)
        self.assertEqual(summary['singleton_groups_removed'], 0)

    def test_partial_target_tie_is_retained_as_diagnostic(self):
        self.mean[d.SHARED_DIM] = 2**30
        left, right = matrix(1), matrix(2)
        left[:, d.SHARED_DIM] = (0, 1, 256)
        right[:, d.SHARED_DIM] = (0, 1, 256)
        train, dev, summary = self.merge([group('A01', features=left), group('A07', features=right)])
        self.assertEqual((len(train), len(dev)), (1, 1))
        self.assertEqual(summary['indistinguishable_groups_removed'], 0)
        self.assertEqual(summary['train_groups_with_target_normalized_tie'], 1)
        self.assertEqual(summary['dev_groups_with_target_normalized_tie'], 1)

    def test_singleton_and_conflicting_tied_components_are_not_double_counted(self):
        self.mean[d.SHARED_DIM] = 2**30
        raw = matrix(1, size=2)
        data = [group('A01', features=raw), group('A02', features=raw.copy(), target=1),
                group('A07', features=matrix(2, size=1))]
        train, dev, summary = self.merge(data)
        self.assertEqual((len(train), len(dev)), (0, 0))
        self.assertEqual((summary['singleton_groups_removed'], summary['conflict_groups_removed'],
                          summary['indistinguishable_groups_removed']), (1, 2, 0))

    def test_invalid_features_coordinates_labels_and_guards_fail_closed(self):
        item = group()
        inconsistent = item.features.copy()
        inconsistent[1, 0] += 1
        invalid_op = item.features.copy()
        invalid_op[0, d._OP_START:d._OP_START + len(d.OPS)] = 0
        invalids = [replace(item, role='A03'), replace(item, opponent='other'),
                    replace(item, side=True), replace(item, step=59),
                    replace(item, teacher_ordinal=32), replace(item, terminal_ordinal=-1),
                    replace(item, target_index=True), replace(item, target_index=3),
                    replace(item, op='PLANT'), replace(item, features=inconsistent),
                    replace(item, features=invalid_op),
                    replace(item, features=np.zeros((3, d.DIM), dtype=np.float64))]
        nonfinite = item.features.copy()
        nonfinite[0, -1] = np.nan
        invalids.append(replace(item, features=nonfinite))
        for bad in invalids:
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                self.merge([bad])
        with self.assertRaises(ValueError):
            self.merge([item] * (d.MAX_GROUPS + 1))
        with self.assertRaises(ValueError):
            d.merge([item], mean=self.mean, scale=np.zeros(d.DIM), guard=lambda: None)
        def stop(): raise MemoryError('synthetic guard')
        with self.assertRaises(MemoryError): self.merge([item], guard=stop)

    def test_callback_uniqueness_and_chronology(self):
        first = group(teacher_ordinal=0, terminal_ordinal=0)
        valid = group(teacher_ordinal=1, terminal_ordinal=1, value=2)
        self.merge([valid, first])
        for second in (replace(valid, teacher_ordinal=0, terminal_ordinal=0),
                       replace(valid, terminal_ordinal=0), replace(valid, terminal_ordinal=2)):
            with self.subTest(second=repr(second)), self.assertRaises(ValueError):
                self.merge([first, second])
        excessive = [group(teacher_ordinal=i, terminal_ordinal=i, value=i) for i in range(5)]
        with self.assertRaises(ValueError): self.merge(excessive)

    def test_each_split_group_count_and_seed_role_minima(self):
        groups = good_groups()
        for split, source in (('train', groups[1:]), ('dev', groups[:-1])):
            _, _, result = self.merge(source)
            self.assertFalse(result['gate_checks'][split]['minimum_groups'])
        train = make_groups([('A01', 31), ('A02', 1)])
        _, _, result = self.merge(train + groups[32:])
        self.assertFalse(result['gate_checks']['train']['per_role'])
        dev = make_groups([('A07', 15), ('A08', 1)], start=100)
        _, _, result = self.merge(groups[:32] + dev)
        self.assertFalse(result['gate_checks']['dev']['per_role'])

    def test_role_and_total_thresholds_include_exact_boundary(self):
        train = make_groups([('A01', 24), ('A02', 8)])
        dev = make_groups([('A07', 12), ('A08', 4)], start=100)
        _, _, result = self.merge(train + dev)
        self.assertTrue(result['gate_checks']['train']['minimum_groups'])
        self.assertTrue(result['gate_checks']['train']['per_role'])
        self.assertTrue(result['gate_checks']['dev']['minimum_groups'])
        self.assertTrue(result['gate_checks']['dev']['per_role'])
        below_train = make_groups([('A01', 25), ('A02', 7)])
        _, _, result = self.merge(below_train + dev)
        self.assertFalse(result['gate_checks']['train']['per_role'])
        below_dev = make_groups([('A07', 13), ('A08', 3)], start=100)
        _, _, result = self.merge(train + below_dev)
        self.assertFalse(result['gate_checks']['dev']['per_role'])

    def test_each_split_teacher_seat_hour_and_operation_gates(self):
        groups = good_groups()
        counts = {'train': [('A01', 16), ('A02', 16)], 'dev': [('A07', 8), ('A08', 8)]}
        for split, start, other in (('train', 1, groups[32:]), ('dev', 100, groups[:32])):
            for option, kwargs, flag in (
                    ('opponent', dict(opponents=('market',)), 'per_opponent'),
                    ('side', dict(sides=(0,)), 'per_side'),
                    ('hour', dict(steps=d.STEPS[:3]), 'four_hours'),
                    ('operation', dict(one_operation=True), 'two_ops')):
                with self.subTest(split=split, restriction=option):
                    restricted = make_groups(counts[split], start=start, **kwargs)
                    _, _, result = self.merge(restricted + other)
                    self.assertFalse(result['gate_checks'][split][flag])

    def test_each_teacher_requires_two_hours_in_each_split(self):
        groups = good_groups()
        for split, start, other, role_counts in (
                ('train', 1, groups[32:], [('A01', 8), ('A02', 8)]),
                ('dev', 100, groups[:32], [('A07', 4), ('A08', 4)])):
            # One teacher has only one hour; the other supplies all hours.
            restricted = make_groups(role_counts, start=start, opponents=('market',), steps=(48,))
            diversified = make_groups(role_counts, start=start + 1000, opponents=('round2',))
            _, _, result = self.merge(restricted + diversified + other)
            self.assertFalse(result['gate_checks'][split]['per_opponent_two_hours'])
            self.assertTrue(result['gate_checks'][split]['four_hours'])


if __name__ == '__main__':
    unittest.main()
