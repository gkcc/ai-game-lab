"""Synthetic full-length replay pairing; no disk, engine or model access."""
from copy import deepcopy
import unittest

from .consumer import OFFICIAL_TURNS
from .panel import paired
from .replay_pair import compare_replays
from .test_consumer import actions, candidate, frame, game, record, turn


def replay(seed=1):
    frames = []
    for step in range(OFFICIAL_TURNS + 1):
        states = frame(step)
        for side, state in enumerate(states):
            state['action'] = actions()[side]
            state['status'] = 'DONE' if step == OFFICIAL_TURNS else 'ACTIVE'
        frames.append(states)
    return dict(info=dict(seed=seed), configuration=dict(boardSize=10), steps=frames)


def trace():
    return [turn(step, [record(step)]) for step in range(OFFICIAL_TURNS)]


def compare(ref, new, left, right, **kwargs):
    options = dict(reference='reference', candidate='candidate', seed=1,
                   seed_alias='S-synthetic', opponent='opponent', side=0)
    options.update(kwargs)
    return compare_replays(ref, new, left, right, **options)


class ReplayPairTests(unittest.TestCase):
    def test_joint_actions_are_from_next_frame(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        new['steps'][1][0]['action']['farmer'] = ['EAST']
        row = compare(ref, new, a, b)
        self.assertEqual((row['reference'], row['candidate'], row['seed_alias']),
                         ('reference', 'candidate', 'S-synthetic'))
        self.assertEqual((row['result']['matched_calls'], row['result']['stop_reason']),
                         (1, 'joint_action_difference'))
        self.assertTrue(row['result']['comparison_complete'])

    def test_full_trace_compares_through_end_when_no_divergence(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        row = compare(ref, new, a, b)
        self.assertEqual(row['result']['matched_calls'], OFFICIAL_TURNS)
        self.assertTrue(row['result']['comparison_complete'])
        self.assertFalse(row['result']['stopped'])

    def test_choice_difference_stops_before_subsequent_matching_turns(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        choices = [candidate(0, [1, 0], 'PLANT'), candidate(1, [1, 0], 'PLANT')]
        a[2] = turn(2, [record(2, candidates=choices)])
        b[2] = turn(2, [record(2, unit=1, candidates=deepcopy(choices))])
        row = compare(ref, new, a, b)
        self.assertEqual((row['result']['stop_reason'], row['result']['matched_calls']), ('choice_difference', 3))
        self.assertEqual(row['result']['matched_selected_different_unit'], 1)

    def test_all_trace_rows_are_checked_even_after_first_action_split(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        new['steps'][1][0]['action']['farmer'] = ['EAST']
        variants = [b[:-1], b + [turn(OFFICIAL_TURNS)], deepcopy(b), deepcopy(b)]
        variants[2][-1]['step'] = OFFICIAL_TURNS - 2
        variants[3][-1]['fallbacks'] = 1
        for invalid in variants:
            with self.subTest(kind=len(invalid)):
                with self.assertRaises(ValueError): compare(ref, new, a, invalid)

    def test_streams_are_consumed_to_eof_after_first_split(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        new['steps'][1][0]['action']['farmer'] = ['EAST']
        observations = [dict(rows=0, eof=False), dict(rows=0, eof=False)]
        def stream(rows, tracker):
            for row in rows:
                tracker['rows'] += 1
                yield row
            tracker['eof'] = True
        output = compare(ref, new, stream(a, observations[0]), stream(b, observations[1]))
        self.assertEqual(output['result']['stop_reason'], 'joint_action_difference')
        self.assertEqual(observations, [dict(rows=OFFICIAL_TURNS, eof=True), dict(rows=OFFICIAL_TURNS, eof=True)])

    def test_stream_eof_integrity_error_and_extra_line_fail(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        new['steps'][1][0]['action']['farmer'] = ['EAST']
        def corrupt_at_eof():
            yield from b
            raise ValueError('synthetic source authentication failure')
        with self.assertRaisesRegex(ValueError, 'synthetic source authentication failure'):
            compare(ref, new, iter(a), corrupt_at_eof())
        with self.assertRaises(ValueError):
            compare(ref, new, iter(a), iter(b + [turn(OFFICIAL_TURNS)]))
        with self.assertRaises(ValueError):
            compare(ref, new, iter(a), iter(b[:-1]))
        def broken_after_split():
            for i, row in enumerate(b):
                if i == 7:
                    raise RuntimeError('synthetic stream failure')
                yield row
        with self.assertRaisesRegex(RuntimeError, 'synthetic stream failure'):
            compare(ref, new, iter(a), broken_after_split())

    def test_seed_config_frames_and_done_must_match_contract(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        variants = [deepcopy(new) for _ in range(4)]
        variants[0]['info']['seed'] = 2
        variants[1]['configuration']['boardSize'] = 11
        variants[2]['steps'][-1][1]['status'] = 'ACTIVE'
        variants[3]['steps'][-1][0]['observation']['step'] = 718
        for invalid in variants:
            with self.assertRaises(ValueError): compare(ref, invalid, a, b)

    def test_result_is_accepted_by_paired_with_explicit_roles(self):
        ref, new, a, b = replay(), replay(), trace(), trace()
        row = compare(ref, new, a, b)
        rows = [game(1, side, arm) for side in (0, 1) for arm in ('reference', 'candidate')]
        result = paired(rows, seeds=[1], opponents=['opponent'], reference='reference', candidate='candidate',
                        comparisons=[row])
        self.assertIsNone(result['opportunity_support']['same_input_pairs']['total'])
        self.assertEqual(result['opportunity_support']['same_input_pairs']['support'], 'unknown')


if __name__ == '__main__':
    unittest.main()
