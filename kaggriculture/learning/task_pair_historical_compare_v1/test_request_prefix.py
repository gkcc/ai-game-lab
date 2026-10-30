"""In-memory replay-prefix regression checks; no official game or file IO."""
from copy import deepcopy
import json
import unittest

from .request_prefix import compare


SYNTHETIC_SEED = 4000000000


def replay():
    frames = []
    for step in range(720):
        frame = []
        for player in (0, 1):
            observation = dict(step=step, player=player, remainingOverageTime=60.,
                               private=dict(inventories=[dict(A=1, B=2)], shed=dict(A=2, B=1)))
            frame.append(dict(observation=observation, status='DONE' if step == 719 else 'ACTIVE',
                              action=dict(farmer=['PASS'], hands=[], market=[])))
        frames.append(frame)
    return dict(info=dict(seed=SYNTHETIC_SEED), configuration=dict(episodeSteps=720), steps=frames)


class RequestPrefixTests(unittest.TestCase):
    def setUp(self):
        self.reference = replay()
        self.candidate = deepcopy(self.reference)

    def test_same_complete_replay_is_none_and_does_not_leak_inputs(self):
        self.candidate['private_value'] = 'SECRET-OUTSIDE-FRAMES'
        result = compare(self.reference, self.candidate, side=0)
        self.assertEqual(result, dict(scope='same_complete_history', status='no_difference',
                                      reason='all_requests_equal', step=None))
        output = json.dumps(result)
        self.assertNotIn(str(SYNTHETIC_SEED), output)
        self.assertNotIn('SECRET-OUTSIDE-FRAMES', output)

    def test_actor_first_request_difference_uses_next_frame_and_both_seats(self):
        for side in (0, 1):
            with self.subTest(side=side):
                candidate = deepcopy(self.candidate)
                candidate['steps'][38][side]['action']['farmer'] = ['NORTH']
                result = compare(self.reference, candidate, side=side)
                self.assertEqual((result['status'], result['reason'], result['step']),
                                 ('observed', 'actor_request_difference', 37))
                self.assertNotIn('NORTH', json.dumps(result))

    def test_ordered_market_orders_are_part_of_the_complete_request(self):
        a = [['BUY_SEED', 'A', 1], ['SELL', 'B', 2]]
        self.reference['steps'][1][0]['action']['market'] = deepcopy(a)
        self.candidate['steps'][1][0]['action']['market'] = list(reversed(a))
        result = compare(self.reference, self.candidate, side=0)
        self.assertEqual((result['status'], result['step']), ('observed', 0))

    def test_opponent_first_and_simultaneous_differences_are_unknown(self):
        self.candidate['steps'][5][1]['action']['farmer'] = ['WEST']
        self.candidate['steps'][12][0]['action']['farmer'] = ['EAST']
        result = compare(self.reference, self.candidate, side=0)
        self.assertEqual((result['status'], result['reason'], result['step']),
                         ('unknown', 'opponent_request_difference', None))
        self.candidate['steps'][5][0]['action']['farmer'] = ['EAST']
        result = compare(self.reference, self.candidate, side=0)
        self.assertEqual((result['status'], result['reason'], result['step']),
                         ('unknown', 'simultaneous_request_difference', None))

    def test_state_or_inventory_order_difference_stops_before_request(self):
        self.candidate['steps'][0][0]['observation']['private']['inventories'][0] = dict(B=2, A=1)
        self.candidate['steps'][1][0]['action']['farmer'] = ['NORTH']
        result = compare(self.reference, self.candidate, side=0)
        self.assertEqual((result['status'], result['reason'], result['step']),
                         ('unknown', 'frame_difference', None))

    def test_timing_noise_is_ignored_but_terminal_state_difference_is_unknown(self):
        self.candidate['steps'][0][0]['observation']['remainingOverageTime'] = 2.
        self.assertEqual(compare(self.reference, self.candidate, side=0)['status'], 'no_difference')
        self.candidate['steps'][-1][0]['observation']['private']['shed']['A'] = 3
        result = compare(self.reference, self.candidate, side=0)
        self.assertEqual((result['status'], result['reason'], result['step']),
                         ('unknown', 'terminal_frame_difference', None))

    def test_later_state_validity_remains_the_outer_acceptance_responsibility(self):
        self.candidate['steps'][1][0]['action']['farmer'] = ['NORTH']
        self.candidate['steps'][600][0]['observation']['private']['shed']['A'] = float('nan')
        result = compare(self.reference, self.candidate, side=0)
        self.assertEqual((result['status'], result['reason'], result['step']),
                         ('observed', 'actor_request_difference', 0))

    def test_invalid_or_incomplete_pair_is_rejected_without_exposing_values(self):
        variants = []
        bad = deepcopy(self.candidate); bad['info']['seed'] += 1; variants.append(bad)
        bad = deepcopy(self.candidate); bad['configuration']['episodeSteps'] = 721; variants.append(bad)
        bad = deepcopy(self.candidate); bad['steps'].pop(); variants.append(bad)
        bad = deepcopy(self.candidate); bad['steps'][-1][1]['status'] = 'ACTIVE'; variants.append(bad)
        bad = deepcopy(self.candidate); bad['steps'][10][0]['observation']['step'] = 9; variants.append(bad)
        bad = deepcopy(self.candidate); bad['steps'][1][0]['action'] = {'farmer': ['PASS']}; variants.append(bad)
        bad = deepcopy(self.candidate)
        bad['steps'][1][0]['action']['farmer'] = ['NORTH']
        bad['steps'][600][1]['action']['market'] = [float('nan')]
        variants.append(bad)
        for variant in variants:
            with self.subTest(variant=len(variants)), self.assertRaises(ValueError) as error:
                compare(self.reference, variant, side=0)
            self.assertNotIn(str(SYNTHETIC_SEED), str(error.exception))
        with self.assertRaises(ValueError):
            compare(self.reference, self.candidate, side=True)


if __name__ == '__main__':
    unittest.main()
