"""Synthetic traces only: no private sources, checkpoint, or official execution."""
from copy import deepcopy
import unittest

from learning.task_harvest_window_v1.test_selection import DAYS, candidate, record
from . import coverage as c


def turn(step, offers, choices, final, *, positions=((1, 1), (2, 1)), available=None, side=0):
    """Compose real-schema metadata with explicit dispatch and final requests."""
    template = record(step, side=side)['inputs']
    own = template['observation']['farms'][side]
    own['farmer'], own['hands'] = list(positions[0]), [list(p) for p in positions[1:]]
    template['observation']['private']['inventories'] = [{} for _ in positions]
    template['inventory_item_order'] = [[] for _ in positions]
    remaining = set(range(len(positions))) if available is None else set(available)
    records = []
    for offered, choice in zip(offers, choices):
        inputs = deepcopy(template)
        inputs['candidates'] = [candidate(*spec) for spec in offered]
        inputs['planned']['available_units'] = sorted(remaining)
        inputs['planned']['unit_actions'] = [
            ['PASS'] if unit in remaining else deepcopy(action) for unit, action in enumerate(final)]
        selected = candidate(*choice)
        unit = selected['unit']
        records.append(dict(inputs=inputs, chosen=selected['id'], unit=unit,
                            target=selected['target'], op=selected['op'],
                            unit_request=deepcopy(final[unit]), teacher_label='unused'))
        remaining.remove(unit)
    return dict(step=step, records=records, final_reservations={'unit_actions': deepcopy(final)})


def trace(entries=()):
    chosen = {row['step']: row for row in entries}
    return [chosen.get(step, dict(step=step, records=[],
                final_reservations={'unit_actions': [['PASS']]})) for step in range(719)]


def analyze(*entries, side=0):
    return c.analyze(iter(trace(entries)), side=side, first_yield_days=DAYS)


def later(step=300, side=0):
    water, harvest = (0, (0, 0), 'WATER'), (1, (2, 1), 'HARVEST')
    return turn(step, [[water, harvest], [harvest]], [water, harvest],
                [['NORTH'], ['HARVEST']], side=side)


class CoverageTests(unittest.TestCase):
    def test_later_dispatch_covers_once_despite_repeated_candidate_appearances(self):
        result = analyze(later())
        n = result['counts']
        self.assertEqual(n['candidates'], dict(total=1, selected_at_first_seen=0,
            selected_later_same_unit=1, covered_by_other_unit=0, no_final_harvest_request=0))
        self.assertEqual(n['sites'], dict(total=1, request_at_first_seen=0,
                                         request_later=1, no_final_harvest_request=0))
        self.assertEqual(n['facts']['direct_candidate_occurrences'], 2)
        self.assertEqual(n['callbacks']['only_other_unit_direct_mature_harvest'], 1)
        self.assertEqual(n['callbacks']['chosen_is_direct_mature_harvest'], 1)
        self.assertEqual(n['callbacks']['has_same_unit_direct_alternative'], 0)
        self.assertIsNone(result['original_selection']['target'])
        self.assertEqual(result['examples']['candidates.selected_later_same_unit']['ordinal'], 0)

    def test_other_colocated_unit_covers_same_site_without_double_counting_sites(self):
        a, b = (0, (1, 1), 'HARVEST'), (1, (1, 1), 'HARVEST')
        result = analyze(turn(300, [[a, b]], [b], [['PASS'], ['HARVEST']],
                              positions=((1, 1), (1, 1))))
        n = result['counts']
        self.assertEqual(n['candidates']['total'], 2)
        self.assertEqual(n['candidates']['selected_at_first_seen'], 1)
        self.assertEqual(n['candidates']['covered_by_other_unit'], 1)
        self.assertEqual(n['sites']['total'], 1)
        self.assertEqual(n['sites']['request_at_first_seen'], 1)
        self.assertEqual(n['expression']['expressed'], 2)
        self.assertEqual(n['expression']['only_other_unit_candidate'], 0)

    def test_harvest_task_with_move_is_parallel_fact_not_harvest_request(self):
        direct, travel = (0, (1, 1), 'HARVEST'), (1, (1, 1), 'HARVEST')
        result = analyze(turn(300, [[direct, travel]], [travel], [['PASS'], ['WEST']]))
        n = result['counts']
        self.assertEqual(n['candidates']['no_final_harvest_request'], 1)
        self.assertEqual(n['sites']['no_final_harvest_request'], 1)
        self.assertEqual(n['facts']['harvest_task_move_assignments'], 1)
        self.assertEqual(n['facts']['sites_with_harvest_task_move'], 1)
        self.assertEqual(n['facts']['sites_with_other_task'], 0)

    def test_other_task_claim_and_old_same_unit_alternative_are_not_hidden(self):
        harvest, water = (0, (1, 1), 'HARVEST'), (0, (1, 1), 'WATER')
        result = analyze(turn(300, [[harvest, water]], [water], [['WATER'], ['PASS']]))
        self.assertEqual(result['counts']['facts']['sites_with_other_task'], 1)
        self.assertEqual(result['counts']['sites']['no_final_harvest_request'], 1)
        self.assertEqual(result['counts']['callbacks']['has_same_unit_direct_alternative'], 1)
        self.assertEqual(result['original_selection']['target']['step'], 300)

    def test_expression_preassigned_available_unexpressed_and_remote_other_unit(self):
        harvest = (1, (1, 1), 'HARVEST')
        result = analyze(turn(300, [[harvest]], [harvest], [['NORTH'], ['WEST']], available=[1]))
        self.assertEqual(result['counts']['expression'], dict(total=2, expressed=0,
            preassigned=1, available_unexpressed=1, only_other_unit_candidate=1))
        self.assertEqual(result['counts']['sites']['total'], 0)
        self.assertEqual(result['counts']['callbacks']['has_mature_positive_harvest'], 1)
        self.assertEqual(result['counts']['callbacks']['has_direct_mature_harvest'], 0)

    def test_empty_callbacks_are_unknown_even_with_final_requests(self):
        result = analyze()
        self.assertEqual(result['counts']['trace'], dict(rows=719, callbacks=0,
            window_turns=409, observed_window_turns=0, unknown_no_callback_window_turns=409))
        self.assertEqual(result['counts']['expression']['total'], 0)
        self.assertIsNone(result['examples']['trace.unknown_no_callback_window_turns']['input_sha256'])
        self.assertEqual(result['original_selection']['reason'], 'no_task_callbacks_in_window')

    def test_window_is_inclusive_but_rows_outside_window_still_validate(self):
        result = analyze(later(239), later(240), later(648), later(649))
        self.assertEqual(result['counts']['trace']['callbacks'], 8)
        self.assertEqual(result['counts']['callbacks']['total'], 4)
        self.assertEqual(result['counts']['sites']['total'], 2)
        tail = later(718)
        tail['records'][0]['unit_request'] = ['SOUTH']
        with self.assertRaisesRegex(ValueError, 'unit_request'):
            analyze(later(), tail)

    def test_maturity_filters_and_inventory_are_not_economic_masks(self):
        harvest = (0, (1, 1), 'HARVEST')
        for age, amount, expected in ((1, 3, 0), (4, 0, 0), (2, 3, 1)):
            row = turn(300, [[harvest]], [harvest], [['HARVEST'], ['PASS']])
            inputs = row['records'][0]['inputs']
            tile = inputs['observation']['farms'][0]['tiles'][1][1]
            tile.update(planted_day=12-age, yield_units=amount)
            inputs['observation']['private']['inventories'][0] = {'WHEAT': 100000}
            inputs['inventory_item_order'][0] = ['WHEAT']
            inputs['observation']['private']['shed'] = {'WHEAT': 1000000}
            inputs['shed_item_order'] = ['WHEAT']
            with self.subTest(age=age, amount=amount):
                result = analyze(row)
                self.assertEqual(result['counts']['callbacks']['has_harvest'], 1)
                self.assertEqual(result['counts']['candidates']['total'], expected)

    def test_bad_tail_short_extra_or_failed_stream_never_returns_partial_counts(self):
        original = trace([later()])
        broken = deepcopy(original)
        broken[-1]['step'] = 717
        for rows in (original[:-1], original + [dict(step=719, records=[])], broken):
            with self.assertRaises(ValueError):
                c.analyze(iter(rows), side=0, first_yield_days=DAYS)

        def failed_stream():
            yield from original[:500]
            raise OSError('synthetic stream failure')
        with self.assertRaises(OSError):
            c.analyze(failed_stream(), side=0, first_yield_days=DAYS)

    def test_record_and_final_request_identity_and_required_final_fields(self):
        for mutate in (
            lambda r: r['records'][1].update(unit_request=['PASS']),
            lambda r: r.pop('final_reservations'),
            lambda r: r['final_reservations'].update(unit_actions=[['NORTH']]),
            lambda r: r['records'][1]['inputs']['planned']['unit_actions'].__setitem__(0, ['SOUTH']),
        ):
            row = later()
            mutate(row)
            with self.assertRaises(ValueError):
                analyze(row)

    def test_duplicate_assignment_and_claimed_candidate_do_not_become_extra_opportunities(self):
        duplicate_unit = later()
        second = deepcopy(duplicate_unit['records'][0])
        duplicate_unit['records'][1] = second
        with self.assertRaises(ValueError):
            analyze(duplicate_unit)

        claimed_tile = later()
        # Unit 0 claimed (0,0) at the first callback; unit 1 cannot reconsider it.
        claimed_tile['records'][1]['inputs']['candidates'].append(candidate(1, (0, 0), 'WATER'))
        with self.assertRaisesRegex(ValueError, 'claimed tile'):
            analyze(claimed_tile)

    def test_plant_pass_is_valid_but_unassigned_unit_cannot_change_request(self):
        plant = (0, (1, 1), 'PLANT')
        row = turn(300, [[plant]], [plant], [['PASS'], ['PASS']])
        result = analyze(row)
        self.assertEqual(result['counts']['callbacks']['total'], 1)
        row['final_reservations']['unit_actions'][1] = ['NORTH']
        with self.assertRaisesRegex(ValueError, 'Unassigned available'):
            analyze(row)

    def test_later_other_unit_coverage_is_not_a_missing_harvest(self):
        water = (2, (0, 0), 'WATER')
        a, b = (0, (1, 1), 'HARVEST'), (1, (1, 1), 'HARVEST')
        row = turn(300, [[water, a, b], [a, b]], [water, b],
                   [['PASS'], ['HARVEST'], ['NORTH']], positions=((1, 1), (1, 1), (2, 1)))
        result = analyze(row)
        self.assertEqual(result['counts']['candidates']['selected_later_same_unit'], 1)
        self.assertEqual(result['counts']['candidates']['covered_by_other_unit'], 1)
        self.assertEqual(result['counts']['sites']['request_later'], 1)
        self.assertEqual(result['counts']['sites']['no_final_harvest_request'], 0)

    def test_no_teacher_or_future_fields_and_seat_one_uses_same_contract(self):
        row = later(side=1)
        baseline = analyze(row, side=1)
        for value in row['records']:
            value.update(teacher_label='different', teacher_score=1e99, future_cash=-1e99,
                         future_market={'WHEAT': -100})
        row['future_rewards'] = [1e99, -1e99]
        self.assertEqual(analyze(row, side=1), baseline)
        for example in baseline['examples'].values():
            self.assertNotIn('observation', example)
            self.assertNotIn('inputs', example)
            self.assertEqual(example['side'], 1)

    def test_guard_failure_and_fixed_empty_count_contract(self):
        def fail():
            raise MemoryError('synthetic resource stop')
        with self.assertRaises(MemoryError):
            c.analyze(iter(trace([later()])), side=0, first_yield_days=DAYS, guard=fail)
        first, second = c.empty_counts(), c.empty_counts()
        first['trace']['rows'] = 1
        self.assertEqual(second['trace']['rows'], 0)
        self.assertEqual({key: list(value) for key, value in second.items()}, c.specification()['fields'])


if __name__ == '__main__':
    unittest.main()
