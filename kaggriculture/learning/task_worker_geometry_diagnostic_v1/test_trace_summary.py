"""Synthetic teacher/student geometry regression tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('trace_geometry', HERE / 'trace_summary.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)

NAMES = ['unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y', 'task.distance'] + [
    'task.op.' + op for op in geometry.OPERATIONS]
SCHEMA = {'feature_names': NAMES, 'shared_feature_names': ['step'], 'schema_sha256': 'synthetic'}


def fixture(student_position):
    target = [1, 0]
    positions = [[0, 0], list(student_position)]
    candidates = []
    for unit, (x, y) in enumerate(positions):
        candidates.append({'id': str(unit), 'unit': unit, 'target': target, 'op': 'PLANT',
            'x': [unit, x, y, *target, abs(x - target[0]) + abs(y - target[1])]
                 + [float(op == 'PLANT') for op in geometry.OPERATIONS]})
    inputs = {'feature_names': NAMES, 'shared_feature_names': ['step'], 'schema_sha256': 'synthetic',
        'shared_x': [2], 'planned': {'available_units': [0, 1]}, 'candidates': candidates,
        'configuration': {'boardSize': 5},
        'observation': {'player': 0, 'step': 2, 'farms': [{'farmer': positions[0], 'hands': [positions[1]]}, {}]}}
    return {'inputs': inputs, 'chosen': '1', 'teacher_label': '0', 'unit': 1,
            'target': target, 'op': 'PLANT', 'unit_request': ['WEST'], 'source': 'callback'}


class TraceGeometryTest(unittest.TestCase):
    def test_strict_distance_and_equal_distance_tie(self):
        self.assertEqual(geometry.classify_record(fixture([3, 0]), 0, SCHEMA), 'student_farther')
        self.assertEqual(geometry.classify_record(fixture([2, 0]), 0, SCHEMA),
                         'same_distance_teacher_lower_unit')

    def test_rejects_geometry_or_teacher_order_mismatch(self):
        cases = []
        wrong_distance = fixture([3, 0]); wrong_distance['inputs']['candidates'][1]['x'][5] = 99; cases.append(wrong_distance)
        unavailable = fixture([3, 0]); unavailable['inputs']['planned']['available_units'] = [0]; cases.append(unavailable)
        teacher_farther = fixture([1, 0]); cases.append(teacher_farther)
        wrong_target = fixture([3, 0]); wrong_target['inputs']['candidates'][1]['target'] = [2, 0]; cases.append(wrong_target)
        duplicate = fixture([3, 0]); duplicate['inputs']['candidates'].append(dict(duplicate['inputs']['candidates'][0], id='duplicate')); cases.append(duplicate)
        wrong_onehot = fixture([3, 0]); wrong_onehot['inputs']['candidates'][1]['x'][-1] = 1; cases.append(wrong_onehot)
        reversed_tie = fixture([2, 0]); reversed_tie.update(chosen='0', teacher_label='1', unit=0); cases.append(reversed_tie)
        bad_unit = fixture([3, 0]); bad_unit['inputs']['candidates'][1]['unit'] = True; cases.append(bad_unit)
        outside = fixture([3, 0]); outside['inputs']['candidates'][1]['target'] = [5, 0]; cases.append(outside)
        absent_label = fixture([3, 0]); absent_label['teacher_label'] = 'missing'; cases.append(absent_label)
        for record in cases:
            with self.subTest(case=len(cases)), self.assertRaises(ValueError):
                geometry.classify_record(record, 0, SCHEMA)

    def test_rejects_schema_or_turn_mismatch(self):
        record = fixture([3, 0]); record['inputs']['observation']['step'] = 3
        with self.assertRaises(ValueError):
            geometry.classify_record(record, 0, SCHEMA)
        record = fixture([3, 0]); record['inputs']['schema_sha256'] = 'different'
        with self.assertRaises(ValueError):
            geometry.classify_record(record, 0, SCHEMA)
        record = fixture([3, 0]); record['source'] = 'teacher'
        with self.assertRaises(ValueError):
            geometry.classify_record(record, 0, SCHEMA)
        record = fixture([3, 0]); record['inputs']['observation']['player'] = 1
        with self.assertRaises(ValueError):
            geometry.classify_record(record, 0, SCHEMA)
        record = fixture([3, 0]); record['inputs']['observation']['step'] = 2.0
        with self.assertRaises(ValueError):
            geometry.classify_record(record, 0, SCHEMA)
        record = fixture([3, 0]); record['unit'] = True
        with self.assertRaises(ValueError):
            geometry.classify_record(record, 0, SCHEMA)


if __name__ == '__main__':
    unittest.main()
