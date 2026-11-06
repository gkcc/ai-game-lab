"""Synthetic metadata fixtures, not a replacement game or historical outcomes."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
import sys
import unittest

from . import analysis


PRIVATE = 'PRIVATE_MARKER_NOT_FOR_PUBLIC_OUTPUT'


def plant(crop='MELON', yield_units=3, day=0):
    return dict(kind='PLANT', crop=crop, planted_day=day, yield_units=yield_units,
        watered_today=False, consecutive_unwatered=0, max_lifespan_step=500,
        fertilized_until_day=-1)


def request(op='WATER', market=None):
    return dict(farmer=[op], hands=[], market=[] if market is None else market)


def make_frame(step, cash, tile, *, position=(0, 0), inventory=0, rival_inventory=0, stock=10000):
    farms = [dict(farmer=list(position), hands=[], money=cash[0], tiles=[[deepcopy(tile), None]]),
             dict(farmer=[1, 0], hands=[], money=cash[1], tiles=[[None, None]])]
    market = dict(inventory={item: stock for item in analysis.PRODUCTS},
                  prices={item: 10 for item in analysis.PRODUCTS})
    result = []
    for side, quantity in enumerate((inventory, rival_inventory)):
        result.append(dict(status='ACTIVE', reward=0, action=request(), observation=dict(
            step=step if side == 0 else 0, player=side, farms=deepcopy(farms), market=deepcopy(market),
            private=dict(inventories=[{'MELON': quantity, PRIVATE: 1}], shed={'MELON': 0},
                         secret_machine_config=PRIVATE))))
    return result


def atomic(start, intervene):
    before = [[['MELON', 0], [PRIVATE, 1]]]
    after = [[['MELON', 3 if intervene else 0], [PRIVATE, 1]]]
    return dict(phase='unit', step=start, player=0, unit=0, position=[0, 0],
        requested=['HARVEST'] if intervene else ['WATER'], effective=['HARVEST'] if intervene else ['WATER'],
        status='executed', before={'inventories': before}, after={'inventories': after},
        tile_before=plant(), tile_after=None if intervene else plant(yield_units=4))


def fixture(start=2):
    target = dict(side=0, step=start, ordinal=0, unit=0, target=[0, 0], input_sha256=PRIVATE)
    source = dict(role='F01', opponent='market', side=0, purpose='exploration',
                  seed_alias=PRIVATE, private_path=PRIVATE)
    case = dict(schema='task-harvest-unexpressed-case-v1', passed=True, source=source,
        target=target, row=dict(status='paired', technical_passed=True), core={},
        events={'lines': 2*(start+50)}, emitted_events=2*(start+50))
    events = []
    for arm in ('control', 'intervene'):
        intervene = arm == 'intervene'
        initial = make_frame(start, [100, 200], plant())
        events.append(dict(kind='arm_start', arm=arm, step=start, frame=deepcopy(initial)))
        for step in range(start):
            events.append(dict(kind='turn', arm=arm, scope='prefix', step=step, unused_private=PRIVATE))
        cash, position = [100, 200], (0, 0)
        target_execution = atomic(start, intervene)
        for offset in range(48):
            if offset == 2:
                cash[0] += 15 if intervene else 10
                if intervene:
                    position = (1, 0)
            if offset == 4:
                cash[1] += 25 if intervene else 5
            if offset == 5 and intervene:
                cash[0] += 5
            if offset < 3:
                tile = None if intervene else plant(yield_units=4)
            elif offset < 6:
                tile = None
            else:
                tile = plant('WHEAT', yield_units=2 if offset >= 9 else 1, day=1)
            own = request('HARVEST' if offset == 3 else 'WATER')
            rival = request('PASS')
            if intervene and offset == 0:
                own = request('HARVEST')
            if intervene and offset == 1:
                own = request('NORTH')
            if intervene and offset == 4:
                rival = request('DROP')
            if offset == 5:
                own['market'] = [['SELL', 'MELON', 20 if intervene else 10]]
            frame = make_frame(start+offset+1, cash, tile, position=position,
                inventory=0 if intervene else (4 if offset >= 3 else 0),
                rival_inventory=1 if intervene and offset >= 4 else 0,
                stock=10000+offset+(3 if intervene and offset > 3 else 0))
            frame[0]['action'], frame[1]['action'] = deepcopy(own), deepcopy(rival)
            events.append(dict(kind='turn', arm=arm, scope='suffix', step=start+offset,
                request=own, original_request=request('WATER') if intervene and offset == 0 else deepcopy(own),
                rival_request=rival, patch={'unit': 0} if intervene and offset == 0 else None,
                target_execution=deepcopy(target_execution) if offset == 0 else None,
                frame=frame, trace={'not_read': PRIVATE}))
        branch = dict(arm=arm, target_execution=target_execution,
            endpoints=dict(start={'cash': [100, 200]}, end={'cash': list(cash)}))
        case['core']['control' if arm == 'control' else 'intervention'] = deepcopy(branch)
        case['row']['baseline' if arm == 'control' else 'intervention'] = dict(
            start=dict(own=100, opponent=200), end=dict(own=cash[0], opponent=cash[1]))
        events.append(dict(kind='arm_end', arm=arm, frame=deepcopy(frame), result=deepcopy(branch)))
    return events, case


class AnalysisTests(unittest.TestCase):
    def test_cash_sums_endpoint_delta_top_three_and_explicit_frame_step_semantics(self):
        events, case = fixture()
        consumed = []
        def once():
            for i, event in enumerate(events):
                consumed.append(i)
                yield event
        result = analysis.analyze_case(once(), case)
        value = result['public']
        self.assertEqual(len(consumed), len(events))
        self.assertEqual(value['baseline_window_cash'], dict(own=10, opponent=5))
        self.assertEqual(value['intervention_window_cash'], dict(own=20, opponent=25))
        self.assertEqual(value['delta'], dict(own=10, opponent=20, margin=-10))
        self.assertEqual(value['combined_cash_delta'], 30)
        self.assertEqual([p['offset'] for p in value['top_margin_increments']], [4, 2, 5])
        for arm, expected in (('control', 10), ('intervene', 20)):
            self.assertEqual(sum(t['arms'][arm]['cash_increment']['own']
                                 for t in value['cash_and_difference_timeline']), expected)
        self.assertEqual(value['counts']['suffix_transitions'], 96)
        self.assertEqual(value['counts']['prefix_skipped'], 4)

    def test_atomic_harvest_quantity_is_not_requested_or_whole_turn_inventory_quantity(self):
        events, case = fixture()
        result = analysis.analyze_case(iter(events), case)
        atom = result['public']['arms']['intervene']['atomic_target_step']
        self.assertEqual((atom['original_op'], atom['submitted_op'], atom['effective_op']),
                         ('WATER', 'HARVEST', 'HARVEST'))
        self.assertEqual(atom['actual_harvest_units'], 3)
        self.assertEqual(result['private']['arms']['intervene']['turns'][0]['observed_unit_changes']['own'], [])
        self.assertEqual(result['public']['arms']['control']['atomic_target_step']['actual_harvest_units'], 0)
        self.assertTrue(result['public']['limitations']['actual_market_fills_unknown'])

    def test_followup_requests_positions_inventory_and_both_players_stay_separate(self):
        events, case = fixture()
        public = analysis.analyze_case(events, case)['public']
        first = public['first_divergence']
        self.assertEqual(first['followup_request'], 1)
        self.assertEqual(first['own_position'], 2)
        self.assertEqual(first['opponent_request'], 4)
        self.assertEqual(first['opponent_inventory'], 4)
        self.assertEqual((first['own_cash'], first['opponent_cash']), (2, 4))
        self.assertEqual(public['first_followup_request']['own']['operation_pairs'], {'WATER->NORTH': 1})
        self.assertEqual(public['changed_request_operation_pairs']['target_step']['own'], {'WATER->HARVEST': 1})
        self.assertEqual(public['changed_request_operation_pairs']['followup']['opponent'], {'PASS->DROP': 1})
        self.assertEqual(sum(public['arms']['control']['all_unit_request_operation_counts']['own'].values()), 48)
        self.assertEqual(public['arms']['control']['first_observed_milestones']['harvest_requested_at_target'], 3)
        self.assertEqual(public['arms']['control']['own_target_crop_end_aggregate'],
                         dict(crop='MELON', carried=4, shed=0))

    def test_typed_requests_keep_bool_int_float_array_order_and_negative_zero(self):
        for a, b, expected in ((True, 1, False), (1, 1.0, False), (-0.0, 0.0, True),
                                ([1, 2], [2, 1], False), ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}, True)):
            self.assertIs(analysis.same_request(a, b), expected)
        with self.assertRaises(ValueError):
            analysis.same_request(float('nan'), float('nan'))
        events, case = fixture()
        target = next(e for e in events if e.get('arm') == 'intervene' and e.get('scope') == 'suffix' and e['step'] == 9)
        target['request']['market'] = [['SELL', 'MELON', True]]
        target['original_request'] = deepcopy(target['request'])
        control = next(e for e in events if e.get('arm') == 'control' and e.get('scope') == 'suffix' and e['step'] == 9)
        control['request']['market'] = [['SELL', 'MELON', 1]]
        control['original_request'] = deepcopy(control['request'])
        self.assertTrue(analysis.analyze_case(events, case)['private']['paired_turns'][7]['requests']['own']['market_changed'])

    def test_rejects_phase_duplicates_missing_tail_and_state_request_step_misalignment(self):
        events, case = fixture()
        corruptions = []
        bad = deepcopy(events); bad[1]['scope'] = 'suffix'; corruptions.append(bad)
        bad = deepcopy(events); bad.insert(3, deepcopy(bad[3])); corruptions.append(bad)
        corruptions.append(deepcopy(events[:-1]))
        bad = deepcopy(events); bad[3]['frame'][0]['observation']['step'] += 1; corruptions.append(bad)
        bad = deepcopy(events); bad[3]['step'] += 1; corruptions.append(bad)
        for bad in corruptions:
            with self.assertRaises(ValueError):
                analysis.analyze_case(bad, case)

    def test_endpoint_and_atomic_record_mismatch_refuse_report(self):
        events, case = fixture()
        altered = deepcopy(case)
        altered['row']['intervention']['end']['own'] += 1
        with self.assertRaises(ValueError):
            analysis.analyze_case(events, altered)
        bad = deepcopy(events)
        next(e for e in bad if e.get('arm') == 'intervene' and e.get('scope') == 'suffix')['target_execution']['after']['inventories'][0][0][1] = 99
        with self.assertRaises(ValueError):
            analysis.analyze_case(bad, case)

    def test_public_projection_drops_private_paths_hashes_units_coordinates_and_extra_fields(self):
        events, case = fixture()
        result = analysis.analyze_case(events, case)
        self.assertIn(PRIVATE, json.dumps(result['private']))
        report = result['private']
        report['counts']['secret'] = PRIVATE
        report['delta']['secret'] = PRIVATE
        report['source']['secret'] = PRIVATE
        report['arms']['control']['target_start']['secret'] = PRIVATE
        report['top_margin_increments'][0]['arms']['control']['market_before']['secret'] = PRIVATE
        public = analysis.public_projection(report)
        encoded = json.dumps(public)
        self.assertNotIn(PRIVATE, encoded)
        forbidden = {'unit', 'target', 'input_sha256', 'positions', 'inventories', 'private_path', 'seed_alias'}
        def visit(value):
            if isinstance(value, dict):
                self.assertFalse(set(value)&forbidden)
                for child in value.values(): visit(child)
            elif isinstance(value, list):
                for child in value: visit(child)
        visit(public)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('kaggle_environments', sys.modules)

    def test_synthetic_event_sink_schema_through_production_gzip_stream_to_analyzer(self):
        from .runner import EventStream
        from .contract import LIMITS
        events, case = fixture()
        encoded = b''.join((json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n').encode()
                           for event in events)
        compressed = gzip.compress(encoded, mtime=0)
        expected = dict(bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest())
        with io.BytesIO(compressed) as raw:
            stream = EventStream(raw, expected, LIMITS, {'decoded_bytes': 0}, json.loads, lambda: None)
            result = analysis.analyze_case(stream, case)
            receipt = stream.receipt()
            self.assertEqual(result['public']['delta'], dict(own=10, opponent=20, margin=-10))
            self.assertEqual((receipt['physical_passes'], receipt['lines'], receipt['decoded_bytes']),
                             (1, len(events), len(encoded)))
            self.assertTrue(receipt['crc_eof_verified'] and receipt['compressed_eof_verified'])
            with self.assertRaises(ValueError):
                list(stream)


if __name__ == '__main__':
    unittest.main()
