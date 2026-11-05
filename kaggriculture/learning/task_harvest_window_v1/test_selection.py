"""Synthetic metadata regressions; no source files, models or engine execution."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

from learning.task_choice_v1.features import (
    CANDIDATE_SCOPE, SCHEMA_VERSION, candidate_id, schema_hash,
)
from . import selection as s


DAYS = {'WHEAT': 2, 'CARROT': 2, 'TOMATO': 8, 'STRAWBERRY': 10, 'MELON': 10}


def candidate(unit, target, op):
    return dict(id=candidate_id(unit, target, op), unit=unit,
                target=list(target), op=op, x=[0.0])


def record(step, *, side=0, original_unit=0, harvest_unit=0, harvest_target=(1, 1),
           original_op='WATER', age=4, amount=3):
    day, hour = divmod(step, 24)
    farms = [dict(money=100, tiles=[[None for _ in range(3)] for _ in range(3)],
                  farmer=[1, 1], hands=[[2, 1]], unlocked_quadrants=['NW'], hires_today=1)
             for _ in range(2)]
    for x, y in ((0, 1), (1, 1), (2, 1)):
        farms[side]['tiles'][y][x] = dict(kind='PLANT', crop='WHEAT',
            planted_day=day-age, watered_today=False, consecutive_unwatered=0,
            yield_units=amount, max_lifespan_step=1000, fertilized_until_day=-1)
    original_target = (0, 1) if original_unit == 0 else (2, 0)
    first = candidate(original_unit, original_target, original_op)
    alternative = candidate(harvest_unit, harvest_target, 'HARVEST')
    cfg = dict(episodeSteps=720, turnsPerDay=24, boardSize=3,
               shedCapacity=100, marketParams={})
    private = dict(inventories=[{}, {}], seeds={'WHEAT': 0}, shed={})
    obs = dict(player=side, step=step, day=day, hour=hour, farms=farms, private=private,
               market={'inventory': {'WHEAT': 10000}, 'prices': {'WHEAT': 25}},
               town={'unlocked_shops': []})
    names, shared = ['fixture.local'], ['fixture.shared']
    inputs = dict(
        schema_version=SCHEMA_VERSION, schema_sha256=schema_hash(names, shared),
        feature_names=names, shared_feature_names=shared, shared_x=[0.0],
        candidate_scope=CANDIDATE_SCOPE, observation=obs, configuration=cfg,
        planned=dict(seeds={'WHEAT': 0}, needed_seeds={}, supply={'WHEAT': 0},
                     projected_shed={}, unit_actions=[['PASS'], ['PASS']],
                     available_units=[0, 1], interpretation='fixture reservations'),
        candidates=[first, alternative], inventory_item_order=[[], []], shed_item_order=[],
    )
    return dict(inputs=inputs, chosen=first['id'], unit=first['unit'],
                target=first['target'], op=first['op'], unit_request=['NORTH'],
                teacher_label=alternative['id'])


def rows(entries=None):
    entries = {} if entries is None else entries
    return [dict(step=i, records=entries.get(i, [])) for i in range(719)]


def choose(entries=None, **kwargs):
    return s.select_target(iter(rows(entries)), side=kwargs.pop('side', 0),
                           first_yield_days=DAYS, **kwargs)


class SelectionTests(unittest.TestCase):
    def test_literal_extraction_checks_identity_and_never_executes_code(self):
        raw = ("CROPS = " + repr({k: {'first_yield_day': v} for k, v in DAYS.items()})
               + "\nraise AssertionError('must never execute')\n").encode()
        with self.assertRaises(ValueError):
            s.first_yield_days_from_source(raw)
        # Synthetic trusted-source identity; no installed rule file is read.
        with patch.object(s, 'ENGINE_SHA', hashlib.sha256(raw).hexdigest()):
            self.assertEqual(s.first_yield_days_from_source(raw), DAYS)
        unsafe = b"CROPS = dict(WHEAT={'first_yield_day': 2})\n"
        with patch.object(s, 'ENGINE_SHA', hashlib.sha256(unsafe).hexdigest()):
            with self.assertRaises(ValueError):
                s.first_yield_days_from_source(unsafe)

    def test_first_step_and_ordinal_inclusive_window_and_full_eof(self):
        before = record(239)
        skipped = record(240, original_unit=1, harvest_unit=0)
        hit = record(240)
        hit['inputs']['planned']['available_units'] = [0]
        result = choose({239: [before], 240: [skipped, hit], 648: [record(648)],
                         649: [record(649)]})
        self.assertEqual((result['target']['step'], result['target']['ordinal']), (240, 1))
        self.assertEqual(result['counts']['trace_lines'], 719)
        self.assertEqual(result['counts']['callbacks_checked'], 5)
        self.assertEqual(result['counts']['window_callbacks'], 3)
        self.assertEqual(result['counts']['qualifying_callbacks'], 2)
        edge = choose({239: [before], 648: [record(648)], 649: [record(649)]})
        self.assertEqual(edge['target']['step'], 648)
        outside = choose({239: [before], 649: [record(649)]})
        self.assertIsNone(outside['target'])
        self.assertEqual(outside['reason'], 'no_task_callbacks_in_window')

    def test_same_unit_direct_position_and_official_maturity_are_required(self):
        for bad in (record(300, harvest_unit=1, harvest_target=(2, 1)),
                    record(300, harvest_target=(2, 1)), record(300, age=1),
                    record(300, amount=0)):
            with self.subTest(bad=bad['inputs']['candidates'][1]['id']):
                result = choose({300: [bad]})
                self.assertIsNone(result['target'])
                self.assertEqual(result['counts']['qualifying_candidates'], 0)
        good = choose({300: [record(300, age=2)]})
        self.assertEqual(good['eligibility']['age'], good['eligibility']['first_yield_day'])

    def test_no_backpack_or_shed_capacity_mask(self):
        value = record(300)
        private = value['inputs']['observation']['private']
        private['inventories'][0] = {'WHEAT': 100000}
        private['shed'] = {'WHEAT': 1000000}
        value['inputs']['inventory_item_order'][0] = ['WHEAT']
        value['inputs']['shed_item_order'] = ['WHEAT']
        result = choose({300: [value]})
        self.assertIsNotNone(result['target'])
        self.assertEqual(result['eligibility']['carried_units'], 100000)
        self.assertGreater(result['eligibility']['shed_units'], result['eligibility']['shed_capacity'])

    def test_original_harvest_at_other_target_is_allowed_and_described(self):
        value = record(300, original_op='HARVEST')
        result = choose({300: [value]})
        self.assertNotEqual(result['target']['original_choice'], result['target']['harvest_choice'])
        self.assertEqual(result['eligibility']['original_op'], 'HARVEST')
        self.assertEqual(result['eligibility']['harvest_op'], 'HARVEST')

    def test_labels_scores_and_future_outcomes_do_not_change_selection(self):
        value = record(300)
        expected = choose({300: [value]})
        changed = deepcopy(value)
        changed.update(teacher_label='not-consulted', teacher_priority=1e10,
                       score=-1e12, future_cash=1e30, future_market={'WHEAT': -100})
        stream = rows({300: [changed]})
        for row in stream:
            row.update(future_rewards=[1e20, -1e20], future_market={'WHEAT': 0})
        self.assertEqual(s.select_target(iter(stream), side=0, first_yield_days=DAYS), expected)

    def test_candidate_order_does_not_choose_a_different_target(self):
        value = record(300)
        expected = choose({300: [value]})['target']
        value['inputs']['candidates'].reverse()
        actual = choose({300: [value]})['target']
        for name in ('side', 'step', 'ordinal', 'original_choice', 'harvest_choice'):
            self.assertEqual(actual[name], expected[name])
        self.assertNotEqual(actual['input_sha256'], expected['input_sha256'])

    def test_tail_corruption_missing_extra_and_generator_errors_are_not_missing_points(self):
        original = rows({300: [record(300)]})
        bad_tail = deepcopy(original)
        bad_tail[-1]['step'] = 717
        for stream in (original[:-1], original + [dict(step=719, records=[])], bad_tail):
            with self.assertRaises(ValueError):
                s.select_target(iter(stream), side=0, first_yield_days=DAYS)

        def broken():
            yield from original[:500]
            raise OSError('fixture broken input stream')
        with self.assertRaises(OSError):
            s.select_target(broken(), side=0, first_yield_days=DAYS)

    def test_malformed_callback_after_first_hit_is_still_rejected(self):
        for alter in (
            lambda r: r.update(chosen='missing'),
            lambda r: r['inputs'].update(schema_sha256='0'*64),
            lambda r: r['inputs']['shared_x'].__setitem__(0, float('nan')),
            lambda r: r['inputs']['candidates'].append(deepcopy(r['inputs']['candidates'][0])),
            lambda r: r['inputs']['observation']['farms'][1].update(private={'hidden': 1}),
        ):
            tail = record(718)
            alter(tail)
            with self.assertRaises(ValueError):
                choose({300: [record(300)], 718: [tail]})

    def test_inventory_order_and_seat_are_checked_without_retaining_whole_input(self):
        value = record(300, side=1)
        result = choose({300: [value]}, side=1)
        self.assertEqual(set(result['target']), {
            'side', 'step', 'ordinal', 'input_sha256', 'original_choice', 'harvest_choice'})
        self.assertNotIn('observation', result['eligibility'])
        value['inputs']['inventory_item_order'][0] = ['WHEAT']
        with self.assertRaises(ValueError):
            choose({300: [value]}, side=1)
        with self.assertRaises(ValueError):
            choose({300: [record(300, side=1)]})

    def test_guard_failure_is_not_swallowed(self):
        def guard():
            raise MemoryError('fixture memory gate')
        with self.assertRaises(MemoryError):
            choose({300: [record(300)]}, guard=guard)

    def test_same_turn_bool_integer_alias_does_not_bypass_shape_validation(self):
        for mutate in (
            lambda r: r['inputs']['observation']['farms'][0]['farmer'].__setitem__(0, True),
            lambda r: r['inputs']['configuration'].update(turnsPerDay=24.0),
            lambda r: r['inputs']['observation']['private']['seeds'].update(WHEAT=False),
            lambda r: r['inputs']['observation']['farms'][0]['tiles'][1][1].update(yield_units=True),
        ):
            first = record(300, original_unit=1, harvest_unit=0, amount=1)
            second = record(300, amount=1)
            mutate(second)
            with self.assertRaises(ValueError):
                choose({300: [first, second]})

    def test_empty_complete_trace_has_an_explicit_missing_reason(self):
        result = choose()
        self.assertIsNone(result['target'])
        self.assertIsNone(result['eligibility'])
        self.assertEqual(result['counts']['trace_lines'], 719)
        self.assertEqual(result['reason'], 'no_task_callbacks_in_window')


if __name__ == '__main__':
    unittest.main()
