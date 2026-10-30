"""Synthetic group tests; no archives, model calls or game execution."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.task_choice_v1.features import OPS
from learning.task_worker_geometry_diagnostic_v1.training_summary import summarize_training_geometry


SHARED_NAMES = ['step', 'config.boardSize'] + ['shared.' + str(i) for i in range(273)]
FEATURE_NAMES = ['unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y',
                 'task.distance'] + ['task.op.' + op for op in OPS]
FEATURE_NAMES += ['unused.' + str(i) for i in range(98 - len(FEATURE_NAMES))]


def group(step, rows, label):
    """A row is (unit, unit_x, unit_y, target_x, target_y, operation)."""
    shared = np.zeros(275, dtype=np.float32)
    shared[SHARED_NAMES.index('step')] = step
    shared[SHARED_NAMES.index('config.boardSize')] = 5
    candidates = np.zeros((len(rows), 98), dtype=np.float32)
    operations = []
    for index, (unit, ux, uy, x, y, op) in enumerate(rows):
        values = {'unit.index': unit, 'unit.x': ux, 'unit.y': uy,
                  'target.x': x, 'target.y': y, 'task.distance': abs(ux - x) + abs(uy - y)}
        values.update({'task.op.' + name: int(op == name) for name in OPS})
        for name, value in values.items():
            candidates[index, FEATURE_NAMES.index(name)] = value
        operations.append(OPS.index(op))
    return shared, candidates, label, operations


def fixture():
    plant, water = 'PLANT', 'WATER'
    groups = [
        # The teacher has both a farther and an equally distant rival.
        group(2, [(0, 0, 0, 1, 0, plant), (1, 0, 1, 1, 0, plant),
                  (2, 1, 1, 1, 0, plant)], 0),
        group(2, [(0, 0, 0, 1, 1, plant), (1, 2, 2, 1, 1, plant)], 0),
        group(5, [(0, 0, 0, 1, 0, plant), (1, 0, 1, 1, 0, plant)], 0),
        group(2, [(0, 0, 0, 1, 0, water)], 0),
        # Same worker, different targets is not worker competition.
        group(2, [(0, 0, 0, 1, 0, plant), (0, 0, 0, 2, 0, plant)], 0),
    ]
    offsets = [0]
    for _, rows, _, _ in groups:
        offsets.append(offsets[-1] + len(rows))
    return dict(shared=np.stack([item[0] for item in groups]),
                candidates=np.concatenate([item[1] for item in groups]),
                offsets=np.asarray(offsets, dtype=np.int64),
                teacher_choice_label=np.asarray([item[2] for item in groups], dtype=np.int64),
                ops=np.asarray([op for item in groups for op in item[3]], dtype=np.uint8),
                shared_feature_names=list(SHARED_NAMES), feature_names=list(FEATURE_NAMES))


class TrainingGeometryTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def run_summary(self):
        return summarize_training_geometry(**self.data)

    def test_counts_groups_once_and_competition_types_can_overlap(self):
        before = {k: v.copy() for k, v in self.data.items() if isinstance(v, np.ndarray)}
        self.assertEqual(self.run_summary(), {
            'all_steps': {'groups': 5, 'teacher_plant_groups': 4,
                          'same_target_unit_competition_groups': 3,
                          'teacher_has_farther_alternative_groups': 2,
                          'equal_distance_competition_groups': 2},
            'encoded_step_2': {'groups': 4, 'teacher_plant_groups': 3,
                       'same_target_unit_competition_groups': 2,
                       'teacher_has_farther_alternative_groups': 1,
                       'equal_distance_competition_groups': 2},
            'competition_categories_may_overlap': True,
        })
        for key, expected in before.items():
            np.testing.assert_array_equal(self.data[key], expected)

    def test_missing_or_duplicate_feature_names_are_rejected(self):
        for field, value in (('feature_names', 'task.distance'),
                             ('feature_names', 'task.op.PLANT'),
                             ('shared_feature_names', 'step'),
                             ('shared_feature_names', 'config.boardSize')):
            with self.subTest(field=field, value=value):
                self.data = fixture()
                index = self.data[field].index(value)
                self.data[field][index] = 'not.the.feature'
                with self.assertRaises(ValueError):
                    self.run_summary()
        self.data = fixture()
        self.data['feature_names'][1] = self.data['feature_names'][0]
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_competition_requires_same_target_and_plant_operation(self):
        shared, candidates, label, ops = group(
            2, [(0, 0, 0, 1, 0, 'PLANT'), (1, 0, 1, 1, 0, 'WATER'),
                (2, 1, 1, 2, 0, 'PLANT')], 0)
        self.data = dict(shared=shared[None, :], candidates=candidates,
                         offsets=np.asarray([0, 3], dtype=np.int64),
                         teacher_choice_label=np.asarray([label], dtype=np.int64),
                         ops=np.asarray(ops, dtype=np.uint8),
                         shared_feature_names=list(SHARED_NAMES), feature_names=list(FEATURE_NAMES))
        result = self.run_summary()
        self.assertEqual(result['encoded_step_2']['teacher_plant_groups'], 1)
        self.assertEqual(result['encoded_step_2']['same_target_unit_competition_groups'], 0)
        self.assertEqual(result['encoded_step_2']['teacher_has_farther_alternative_groups'], 0)
        self.assertEqual(result['encoded_step_2']['equal_distance_competition_groups'], 0)

    def test_shapes_offsets_types_and_labels_fail_closed(self):
        def check(field, value):
            self.data = fixture()
            self.data[field] = value
            with self.assertRaises(ValueError):
                self.run_summary()

        check('shared', np.zeros((5, 274), dtype=np.float32))
        check('shared', np.ones((5, 275), dtype=bool))
        check('candidates', np.zeros((10, 98), dtype=np.float64))
        check('offsets', np.asarray([0, 3, 3, 7, 8, 10], dtype=np.int64))
        check('offsets', np.asarray([0, 3, 5, 7, 8, 9], dtype=np.int64))
        check('offsets', np.asarray([0, 3, 5, 7, 8, 10], dtype=np.int32))
        check('teacher_choice_label', np.asarray([3, 0, 0, 0, 0], dtype=np.int64))
        check('teacher_choice_label', np.asarray([-1, 0, 0, 0, 0], dtype=np.int64))
        check('teacher_choice_label', np.zeros(5, dtype=bool))
        check('ops', np.full(10, len(OPS), dtype=np.uint8))

    def test_nonfinite_and_noninteger_features_fail_closed(self):
        checks = [('shared', 0, SHARED_NAMES.index('step'), 2.5),
                  ('shared', 0, SHARED_NAMES.index('step'), 719),
                  ('shared', 0, 5, np.nan),
                  ('candidates', 0, FEATURE_NAMES.index('unit.index'), 0.5),
                  ('candidates', 0, FEATURE_NAMES.index('unit.x'), -1),
                  ('candidates', 0, FEATURE_NAMES.index('target.y'), 1.5),
                  ('candidates', 0, 30, np.inf)]
        for field, row, column, value in checks:
            with self.subTest(field=field, column=column, value=value):
                self.data = fixture()
                self.data[field][row, column] = value
                with self.assertRaises(ValueError):
                    self.run_summary()

    def test_wrong_distance_and_operation_encoding_fail_closed(self):
        for name, value in (('task.distance', 100), ('task.op.PLANT', 0),
                            ('task.op.WATER', 1)):
            with self.subTest(name=name):
                self.data = fixture()
                self.data['candidates'][0, FEATURE_NAMES.index(name)] = value
                with self.assertRaises(ValueError):
                    self.run_summary()

    def test_board_size_and_per_group_coordinate_bounds_fail_closed(self):
        board = SHARED_NAMES.index('config.boardSize')
        for value in (0, -1, 2.5, np.nan, np.inf):
            with self.subTest(board_size=value):
                self.data = fixture()
                self.data['shared'][0, board] = value
                with self.assertRaises(ValueError):
                    self.run_summary()

        # A valid first group with size 2 does not make size 2 valid for the
        # second group, which has a unit at x=2; each group's own size matters.
        self.data = fixture()
        self.data['shared'][0, board] = 2
        self.data['shared'][1, board] = 3
        self.run_summary()
        self.data['shared'][1, board] = 2
        with self.assertRaises(ValueError):
            self.run_summary()

        for name, distance in (('unit.x', 4), ('target.x', 5)):
            with self.subTest(outside=name):
                self.data = fixture()
                self.data['candidates'][0, FEATURE_NAMES.index(name)] = 5
                self.data['candidates'][0, FEATURE_NAMES.index('task.distance')] = distance
                with self.assertRaises(ValueError):
                    self.run_summary()

    def test_teacher_farther_and_wrong_tie_break_fail_closed(self):
        for chosen in (1, 2):
            with self.subTest(chosen=chosen):
                self.data = fixture()
                self.data['teacher_choice_label'][0] = chosen
                with self.assertRaises(ValueError):
                    self.run_summary()

    def test_candidate_identity_and_unit_position_fail_closed(self):
        # The last two rows describe one unit offered two distinct targets.
        self.data['candidates'][9, FEATURE_NAMES.index('target.x')] = 1
        self.data['candidates'][9, FEATURE_NAMES.index('task.distance')] = 1
        with self.assertRaises(ValueError):
            self.run_summary()

        self.data = fixture()
        self.data['candidates'][9, FEATURE_NAMES.index('unit.x')] = 1
        self.data['candidates'][9, FEATURE_NAMES.index('task.distance')] = 1
        with self.assertRaises(ValueError):
            self.run_summary()


if __name__ == '__main__':
    unittest.main()
