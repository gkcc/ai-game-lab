"""Pure request changes; no disk fixtures, network forwards or game steps."""
from copy import deepcopy
import unittest

from learning.task_harvest_coverage_v1.test_coverage import turn
from learning.task_harvest_window_v1.test_selection import DAYS
from . import adapter as a
from .selection import ordered_hash
from .test_selection import select, unexpressed


class AdapterTests(unittest.TestCase):
    def test_farmer_replacement_preserves_market_order_types_and_trace_intent(self):
        row = unexpressed()
        target = select(row)['target']
        # Orders are opaque here, including over-limit/junk atoms. No repair is allowed.
        market = [['SELL', 'WHEAT', '2'], ['BUY_SEED', 'CARROT', True],
                  ['SELL', 'WHEAT', 2.0], None, {'quantity': 0, 'op': 'fixture'}]
        request = {'market': market, 'hands': [], 'farmer': ['WATER']}
        before_request, before_trace, before_target = deepcopy(request), deepcopy(row), deepcopy(target)
        result = a.replace_harvest(request, target=target, trace=row, first_yield_days=DAYS)
        self.assertEqual(result['original'], before_request)
        self.assertEqual(list(result['submitted']), list(request))
        self.assertEqual(ordered_hash(result['submitted']['market']), ordered_hash(market))
        self.assertEqual(result['submitted']['farmer'], ['HARVEST'])
        self.assertEqual(request, before_request)
        self.assertEqual(row, before_trace)
        self.assertEqual(target, before_target)
        change = result['change']
        self.assertEqual(change['changed_units'], 1)
        self.assertEqual(change['original_request_sha256'], ordered_hash(request))
        self.assertEqual(change['submitted_request_sha256'], ordered_hash(result['submitted']))
        self.assertNotEqual(change['original_request_sha256'], change['submitted_request_sha256'])
        self.assertEqual(row['records'][0]['unit_request'], ['WATER'])

    def test_hand_replacement_leaves_farmer_and_other_hands_exactly_unchanged(self):
        water = (1, (2, 1), 'WATER')
        row = turn(300, [[water]], [water], [['DROP'], ['WATER'], ['PICKUP', 'WHEAT', 2]],
                   positions=((1, 1), (2, 1), (0, 0)), available=[1], side=1)
        target = select(row, side=1)['target']
        request = dict(farmer=['DROP'], hands=[['WATER'], ['PICKUP', 'WHEAT', 2]],
                       market=[['HIRE']])
        result = a.replace_harvest(request, target=target, trace=row, first_yield_days=DAYS)
        self.assertEqual(target['unit'], 1)
        self.assertEqual(result['submitted'], dict(farmer=['DROP'],
                         hands=[['HARVEST'], ['PICKUP', 'WHEAT', 2]], market=[['HIRE']]))
        self.assertEqual(result['change']['side'], 1)

    def test_result_snapshots_and_receipt_do_not_alias_source_or_each_other(self):
        row = unexpressed()
        target = select(row)['target']
        request = dict(farmer=['WATER'], hands=[], market=[['SELL', 'WHEAT', 2]])
        result = a.replace_harvest(request, target=target, trace=row, first_yield_days=DAYS)
        result['submitted']['market'][0][2] = 9
        result['submitted']['farmer'][0] = 'DIG'
        result['change']['original_unit_request'][0] = 'PASS'
        result['change']['target'][0] = 0
        self.assertEqual(result['original'], request)
        self.assertEqual(result['change']['submitted_unit_request'], ['HARVEST'])
        self.assertEqual(target['original_unit_request'], ['WATER'])
        self.assertEqual(target['target'], [1, 1])
        result['original']['market'][0][2] = 8
        self.assertEqual(request['market'][0][2], 2)

    def test_request_mismatch_shape_and_changed_target_are_rejected(self):
        row = unexpressed()
        target = select(row)['target']
        good = dict(farmer=['WATER'], hands=[], market=[])
        for request in (dict(good, farmer=['DIG']), dict(good, hands=[['PASS']]),
                        dict(good, market=()), {'farmer': ['WATER'], 'hands': []},
                        dict(good, market=[float('nan')])):
            with self.subTest(request=request), self.assertRaises((ValueError, TypeError)):
                a.replace_harvest(request, target=target, trace=row, first_yield_days=DAYS)
        changed = deepcopy(row)
        changed['records'][0]['inputs']['observation']['farms'][0]['tiles'][1][1]['yield_units'] = 0
        with self.assertRaises(ValueError):
            a.replace_harvest(good, target=target, trace=changed, first_yield_days=DAYS)


if __name__ == '__main__':
    unittest.main()
