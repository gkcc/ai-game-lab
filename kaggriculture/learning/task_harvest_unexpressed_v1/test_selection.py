"""In-memory metadata fixtures; no source files, model or official execution."""
from copy import deepcopy
import json
import unittest

from learning.task_harvest_coverage_v1.test_coverage import turn, trace
from learning.task_harvest_window_v1.test_selection import DAYS
from . import selection as s


def unexpressed(step=300, *, side=0):
    water = (0, (1, 1), 'WATER')
    return turn(step, [[water]], [water], [['WATER']], positions=((1, 1),), side=side)


def select(*entries, side=0):
    return s.select_target(iter(trace(entries)), side=side, first_yield_days=DAYS)


class SelectionTests(unittest.TestCase):
    def test_first_point_inclusive_window_and_complete_tail(self):
        result = select(unexpressed(239), unexpressed(240), unexpressed(648), unexpressed(649))
        self.assertEqual((result['target']['step'], result['target']['unit']), (240, 0))
        self.assertEqual(result['counts']['qualifying_turns'], 2)
        self.assertEqual(result['counts']['trace_lines'], 719)
        self.assertEqual(select(unexpressed(648))['target']['step'], 648)
        self.assertIsNone(select(unexpressed(239), unexpressed(649))['target'])
        broken = trace([unexpressed(240)])
        broken[-1]['step'] = 717
        for rows in (broken, trace()[:-1], trace() + [dict(step=719, records=[])]):
            with self.assertRaises(ValueError):
                s.select_target(iter(rows), side=0, first_yield_days=DAYS)

    def test_later_own_expression_blocks_point_even_if_not_harvested(self):
        remote_water = (1, (0, 0), 'WATER')
        water, harvest = (0, (1, 1), 'WATER'), (0, (1, 1), 'HARVEST')
        row = turn(300, [[remote_water, water], [water, harvest]],
                   [remote_water, water], [['WATER'], ['NORTH']])
        # The other unit is not on a harvestable crop, so it cannot mask this case.
        for item in row['records']:
            item['inputs']['observation']['farms'][0]['tiles'][1][2] = None
        result = select(row)
        self.assertIsNone(result['target'])
        self.assertEqual(result['counts']['qualifying_unit_sites'], 0)

    def test_other_unit_harvest_on_same_site_blocks_unexpressed_unit(self):
        water, harvest = (2, (0, 0), 'WATER'), (1, (1, 1), 'HARVEST')
        row = turn(300, [[water, harvest], [harvest]], [water, harvest],
                   [['PASS'], ['HARVEST'], ['NORTH']],
                   positions=((1, 1), (1, 1), (2, 1)))
        for item in row['records']:
            item['inputs']['observation']['farms'][0]['tiles'][1][2] = None
        self.assertIsNone(select(row)['target'])

    def test_remote_harvest_move_does_not_count_as_site_harvest(self):
        harvest = (1, (1, 1), 'HARVEST')
        row = turn(300, [[harvest]], [harvest], [['PASS'], ['WEST']])
        target = select(row)['target']
        self.assertEqual((target['unit'], target['target'], target['original_unit_request']),
                         (0, [1, 1], ['PASS']))

    def test_initial_preassignment_is_excluded_and_unit_order_is_fixed(self):
        water0, water1 = (0, (1, 1), 'WATER'), (1, (2, 1), 'WATER')
        both = turn(300, [[water1, water0], [water0]], [water1, water0],
                    [['WATER'], ['WATER']])
        self.assertEqual(select(both)['target']['unit'], 0)
        assigned = turn(300, [[water1]], [water1], [['DROP'], ['WATER']], available=[1])
        self.assertEqual(select(assigned)['target']['unit'], 1)

    def test_official_maturity_and_positive_yield_without_capacity_mask(self):
        for age, amount, expected in ((1, 3, False), (2, 0, False), (2, 3, True)):
            row = unexpressed()
            inputs = row['records'][0]['inputs']
            inputs['observation']['farms'][0]['tiles'][1][1].update(
                planted_day=inputs['observation']['day'] - age, yield_units=amount)
            inputs['observation']['private']['inventories'][0] = {'WHEAT': 100000}
            inputs['observation']['private']['shed'] = {'WHEAT': 1000000}
            inputs['inventory_item_order'][0] = ['WHEAT']
            inputs['shed_item_order'] = ['WHEAT']
            with self.subTest(age=age, amount=amount):
                result = select(row)
                self.assertEqual(result['target'] is not None, expected)
                if expected:
                    self.assertGreater(result['eligibility']['shed_units'],
                                       result['eligibility']['shed_capacity'])

    def test_seat_input_order_request_binding_and_owned_copies(self):
        row = unexpressed(side=1)
        before = deepcopy(row)
        target = select(row, side=1)['target']
        self.assertEqual(set(target), s.TARGET_KEYS)
        self.assertEqual(target['side'], 1)
        self.assertEqual(target['input_sha256'], s.ordered_hash(row['records'][0]['inputs']))
        self.assertEqual(target['unit_requests_sha256'],
                         s.ordered_hash(row['final_reservations']['unit_actions']))
        self.assertEqual(s.validate_target_trace(target, row, first_yield_days=DAYS)['yield_units'], 3)
        reloaded = json.loads(json.dumps(target, sort_keys=True))
        self.assertEqual(s.validate_target_trace(reloaded, row, first_yield_days=DAYS)['yield_units'], 3)
        row['records'][0]['inputs']['observation']['private']['shed'] = {'CARROT': 1, 'WHEAT': 2}
        row['records'][0]['inputs']['shed_item_order'] = ['CARROT', 'WHEAT']
        ordered = select(row, side=1)['target']
        row['records'][0]['inputs']['observation']['private']['shed'] = {'WHEAT': 2, 'CARROT': 1}
        row['records'][0]['inputs']['shed_item_order'] = ['WHEAT', 'CARROT']
        self.assertNotEqual(select(row, side=1)['target']['input_sha256'], ordered['input_sha256'])
        target['original_unit_request'][0] = 'DIG'
        self.assertEqual(row['final_reservations']['unit_actions'], before['final_reservations']['unit_actions'])

    def test_no_callbacks_are_missing_and_labels_do_not_select(self):
        empty = select()
        self.assertIsNone(empty['target'])
        self.assertEqual(empty['reason'], 'no_task_callbacks_in_window')
        self.assertEqual(empty['counts']['unknown_no_callback_window_turns'], 409)
        row = unexpressed()
        expected = select(row)
        row['records'][0].update(teacher_label='unused', teacher_score=-1e50, future_cash=1e99)
        row['future_rewards'] = [1e99, -1e99]
        self.assertEqual(select(row), expected)
        def failed():
            yield from trace([row])[:500]
            raise OSError('synthetic reader failure')
        with self.assertRaises(OSError):
            s.select_target(failed(), side=0, first_yield_days=DAYS)

    def test_target_and_full_turn_revalidation_reject_changed_evidence(self):
        row = unexpressed()
        target = select(row)['target']
        for key, value in (('side', True), ('unit', True), ('ordinal', 1), ('step', 649),
                           ('target', [True, 1]), ('input_sha256', 'not-a-digest')):
            changed = deepcopy(target)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                s.validate_target(changed)
        changed = deepcopy(row)
        changed['records'][0]['unit_request'] = ['DIG']
        changed['final_reservations']['unit_actions'][0] = ['DIG']
        with self.assertRaisesRegex(ValueError, 'binding'):
            s.validate_target_trace(target, changed, first_yield_days=DAYS)
        changed = deepcopy(row)
        changed['records'][0]['inputs']['observation']['farms'][1]['private'] = {'hidden': 1}
        with self.assertRaises(ValueError):
            s.validate_target_trace(target, changed, first_yield_days=DAYS)


if __name__ == '__main__':
    unittest.main()
