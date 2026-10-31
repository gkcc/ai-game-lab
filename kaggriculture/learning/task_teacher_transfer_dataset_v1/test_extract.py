"""Synthetic-only coverage for conditional transfer extraction."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from learning.task_teacher_support_probe_v1 import alignment
from learning.task_teacher_support_probe_v1.test_alignment import make_stream
from . import extract


def transfer(obs, cfg, teacher, terminal, schema, guard=lambda: None):
    return extract.extract_turn(obs, cfg, teacher, terminal, schema=schema, guard=guard)


class TransferTests(unittest.TestCase):
    def test_terminal_features_and_terminal_candidate_order_are_used(self):
        obs, cfg, teacher, schema = make_stream(order=(1, 0))
        _obs, _cfg, terminal, _schema = make_stream(order=(0, 1))
        inputs = terminal[0]['inputs']
        selected = next(i for i, candidate in enumerate(inputs['candidates'])
                        if candidate['id'] == teacher[0]['chosen'])
        self.assertEqual(selected, 1)
        self.assertNotEqual(terminal[0]['chosen'], teacher[0]['chosen'])
        shared_index = inputs['shared_feature_names'].index('planning.tasks_remaining')
        inputs['shared_x'][shared_index] += 7
        inputs['candidates'][selected]['x'][-1] += 3
        groups, counts = transfer(obs, cfg, teacher, terminal, schema)
        self.assertEqual((counts['supported'], counts['unknown']), (1, 1))
        self.assertEqual((counts['supported_target_features_equal'], counts['supported_target_features_differ']), (0, 1))
        self.assertEqual(len(groups), counts['supported'])
        group = groups[0]
        self.assertEqual((group.label, group.operation, group.teacher_ordinal, group.terminal_ordinal),
                         (selected, teacher[0]['op'], 0, 0))
        self.assertEqual(group.candidate_count, len(inputs['candidates']))
        self.assertEqual(group.features.shape, (len(inputs['candidates']), 373))
        self.assertEqual(group.features.dtype, np.dtype(np.float32))
        for i, candidate in enumerate(inputs['candidates']):
            np.testing.assert_array_equal(group.features[i, :275], np.asarray(inputs['shared_x'], dtype=np.float32))
            np.testing.assert_array_equal(group.features[i, 275:], np.asarray(candidate['x'], dtype=np.float32))
        self.assertNotEqual(float(group.features[selected, shared_index]),
                            float(teacher[0]['inputs']['shared_x'][shared_index]))
        self.assertNotEqual(float(group.features[selected, -1]),
                            float(teacher[0]['inputs']['candidates'][selected]['x'][-1]))

    def test_absent_target_and_later_diverged_prefix_do_not_make_groups(self):
        obs, cfg, teacher, schema = make_stream(order=(1, 0))
        _obs, _cfg, terminal, _schema = make_stream(order=(0, 1))
        terminal[0]['inputs']['candidates'] = [candidate for candidate in terminal[0]['inputs']['candidates']
                                                if candidate['id'] != teacher[0]['chosen']]
        groups, counts = transfer(obs, cfg, teacher, terminal, schema)
        self.assertEqual((counts['supported'], counts['unsupported'], counts['unknown']), (0, 1, 1))
        self.assertEqual(groups, ())

    def test_earlier_plant_crop_uncertainty_blocks_only_later_groups(self):
        obs, cfg, teacher, schema = make_stream(plant=True, plant_request=['NORTH'])
        groups, counts = transfer(obs, cfg, teacher, deepcopy(teacher), schema)
        self.assertEqual((counts['supported'], counts['unknown']), (1, 1))
        self.assertEqual([(group.operation, group.teacher_ordinal) for group in groups], [('PLANT', 0)])
        obs, cfg, teacher, schema = make_stream(plant=True, plant_request=['PLANT', 'MELON'])
        terminal = deepcopy(teacher)
        terminal[0]['unit_request'] = ['PLANT', 'WHEAT']
        terminal[1]['inputs']['planned']['unit_actions'][1] = ['PLANT', 'WHEAT']
        groups, counts = transfer(obs, cfg, teacher, terminal, schema)
        self.assertEqual((counts['supported'], counts['unknown']), (1, 1))
        self.assertEqual([group.teacher_ordinal for group in groups], [0])

    def test_different_past_reservations_cannot_realign(self):
        obs, cfg, teacher, schema = make_stream()
        terminal = deepcopy(teacher)
        teacher[0]['inputs']['planned']['projected_shed'] = {'WHEAT': 1, 'CARROT': 1}
        terminal[0]['inputs']['planned']['projected_shed'] = {'CARROT': 1, 'WHEAT': 1}
        groups, counts = transfer(obs, cfg, teacher, terminal, schema)
        self.assertEqual((counts['supported'], counts['unknown']), (0, 2))
        self.assertEqual(groups, ())

    def test_empty_stream_and_missing_terminal_have_no_groups(self):
        obs, cfg, records, schema = make_stream()
        groups, counts = transfer(obs, cfg, [], [], schema)
        self.assertEqual(groups, ())
        self.assertEqual((counts['teacher_empty_turn'], counts['terminal_empty_turn']), (1, 1))
        groups, counts = transfer(obs, cfg, records, [], schema)
        self.assertEqual(groups, ())
        self.assertEqual(counts['unknown'], 2)

    def test_groups_are_private_read_only_copies_with_anonymous_repr(self):
        obs, cfg, teacher, schema = make_stream()
        terminal = deepcopy(teacher)
        before = deepcopy((obs, cfg, teacher, terminal, schema))
        groups, counts = transfer(obs, cfg, teacher, terminal, schema)
        self.assertEqual((obs, cfg, teacher, terminal, schema), before)
        self.assertEqual(len(groups), counts['supported'])
        snapshot = groups[0].features.copy()
        self.assertFalse(groups[0].features.flags.writeable)
        with self.assertRaises(ValueError):
            groups[0].features[0, 0] = 123
        with self.assertRaises(ValueError):
            groups[0].features.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            groups[0].label = 123
        terminal[0]['inputs']['shared_x'][0] += 123
        terminal[0]['inputs']['candidates'][0]['x'][0] += 123
        teacher[0]['inputs']['shared_x'][0] += 123
        np.testing.assert_array_equal(groups[0].features, snapshot)
        self.assertEqual(repr(groups[0]), '<PrivateTransferGroup>')
        self.assertNotIn(teacher[0]['chosen'], repr(groups))
        self.assertNotIn('private', repr(groups).lower().replace('privatetransfergroup', ''))
        self.assertNotIn('array', repr(groups).lower())

    def test_bad_features_identity_resource_and_limits_fail_closed(self):
        obs, cfg, records, schema = make_stream()
        variants = []
        bad = deepcopy(records); bad[0]['inputs']['shared_x'][0] = float('nan'); variants.append(bad)
        bad = deepcopy(records); bad[0]['inputs']['shared_x'][0] = '0.0'; variants.append(bad)
        bad = deepcopy(records); bad[0]['inputs']['candidates'][0]['x'][-1] = True; variants.append(bad)
        bad = deepcopy(records); bad[0]['inputs']['candidates'][0]['x'].pop(); variants.append(bad)
        bad = deepcopy(records); bad[0]['chosen'] = 'untrusted'; variants.append(bad)
        bad = deepcopy(records); bad[0]['inputs']['planned']['seeds']['WHEAT'] = -1; variants.append(bad)
        bad = deepcopy(records); bad[0]['inputs']['candidates'].extend(
            [deepcopy(bad[0]['inputs']['candidates'][0]) for _ in range(255)]); variants.append(bad)
        variants.append([deepcopy(records[0])] * 33)
        for bad in variants:
            for corrupt_teacher in (False, True):
                teacher, terminal = (bad, records) if corrupt_teacher else (records, bad)
                with (self.subTest(variant=len(bad), corrupt_teacher=corrupt_teacher),
                      self.assertRaises((alignment.AlignmentError, extract.TransferError))):
                    transfer(obs, cfg, teacher, terminal, schema)

    def test_guard_exception_propagates_without_partial_result(self):
        obs, cfg, records, schema = make_stream()
        signal = MemoryError('synthetic stop')
        def guard():
            raise signal
        with self.assertRaises(MemoryError) as caught:
            transfer(obs, cfg, records, records, schema, guard=guard)
        self.assertIs(caught.exception, signal)
        with self.assertRaises(extract.TransferError):
            extract.extract_turn(obs, cfg, records, records, schema=schema, guard=None)


if __name__ == '__main__':
    unittest.main()
