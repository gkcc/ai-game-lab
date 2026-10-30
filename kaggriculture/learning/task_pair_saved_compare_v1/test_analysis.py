"""Synthetic replay and trace invariants; no game, model or private files."""
from copy import deepcopy
import json
import unittest

from learning.task_pair_dev_metrics_v1.consumer import OFFICIAL_TURNS
from learning.task_pair_dev_metrics_v1.test_consumer import candidate, record, turn
from learning.task_pair_dev_metrics_v1.test_replay_pair import replay
from .analysis import compare as compare_saved


def trace():
    return [turn(step, [record(step)]) for step in range(OFFICIAL_TURNS)]


def compare(ref, new, left, right, check=lambda: True):
    return compare_saved(ref, new, iter(left), iter(right), seed=1, side=0, check=check)


class SavedComparisonTests(unittest.TestCase):
    def test_same_recorded_inputs_different_choice_same_complete_request(self):
        ref, new, left, right = replay(), replay(), trace(), trace()
        choices = [candidate(0, [1, 0], 'PLANT'), candidate(1, [1, 0], 'WATER')]
        left[2] = turn(2, [record(2, candidates=choices)])
        right[2] = turn(2, [record(2, unit=1, op='WATER', candidates=deepcopy(choices))])
        new['private_placeholder'] = 'SECRET-PRIVATE'
        result = compare(ref, new, left, right)
        self.assertEqual((result['stop_reason'], result['first_stop_step'],
                          result['matched_selection_records']), ('choice_difference', 2, 3))
        choice = result['first_different_recorded_choice']
        self.assertEqual((choice['baseline_op'], choice['control_op']), ('PLANT', 'WATER'))
        self.assertFalse(choice['same_unit'])
        self.assertTrue(choice['same_target'])
        self.assertTrue(choice['complete_actor_request_equal'])
        self.assertTrue(choice['complete_opponent_request_equal'])
        self.assertTrue(choice['both_model_callbacks'])
        self.assertEqual(result['complete_request_prefix']['status'], 'no_difference')
        self.assertNotIn('SECRET-PRIVATE', json.dumps(result))
        self.assertNotIn('NORTH', json.dumps(result))

    def test_next_frame_and_simultaneous_request_difference(self):
        ref, new, left, right = replay(), replay(), trace(), trace()
        new['steps'][4][0]['action']['farmer'] = ['EAST']
        new['steps'][4][1]['action']['farmer'] = ['WEST']
        result = compare(ref, new, left, right)
        self.assertEqual((result['stop_reason'], result['first_stop_step']), ('joint_action_difference', 3))
        self.assertEqual((result['complete_request_prefix']['status'], result['complete_request_prefix']['reason']),
                         ('unknown', 'simultaneous_request_difference'))
        self.assertIsNone(result['first_different_recorded_choice'])

    def test_all_trace_records_and_eof_validated_after_branch(self):
        ref, new, left, right = replay(), replay(), trace(), trace()
        new['steps'][1][0]['action']['farmer'] = ['EAST']
        reached = [False, False]
        def stream(rows, i):
            yield from rows
            reached[i] = True
        result = compare_saved(ref, new, stream(left, 0), stream(right, 1), seed=1, side=0, check=lambda: True)
        self.assertEqual(result['first_stop_step'], 0)
        self.assertEqual(reached, [True, True])
        bad_sequence, bad_fallback = deepcopy(right), deepcopy(right)
        bad_sequence[-1]['step'] = 0
        bad_fallback[-1]['fallbacks'] = 1
        for invalid in (right[:-1], right + [turn(OFFICIAL_TURNS)], bad_sequence, bad_fallback):
            with self.assertRaises(ValueError):
                compare(ref, new, left, invalid)
        invalid = deepcopy(right)
        invalid[-1]['records'][0]['chosen'] = 'missing-candidate'
        with self.assertRaises(ValueError):
            compare(ref, new, left, invalid)
        def broken_eof():
            yield from right
            raise ValueError('Synthetic digest failure')
        with self.assertRaisesRegex(ValueError, 'Synthetic digest failure'):
            compare_saved(ref, new, iter(left), broken_eof(), seed=1, side=0, check=lambda: True)

    def test_guard_and_replay_identity_fail_closed(self):
        ref, new, left, right = replay(), replay(), trace(), trace()
        new['info']['seed'] += 1
        with self.assertRaises(ValueError):
            compare(ref, new, left, right)
        new = replay()
        steps = [0]
        def guard():
            steps[0] += 1
            return steps[0] < 10
        with self.assertRaises(ValueError):
            compare(ref, new, left, right, guard)
        self.assertEqual(steps[0], 10)


if __name__ == '__main__':
    unittest.main()
