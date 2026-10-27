"""Synthetic regressions for identified alignment, semantics and stream risks."""
from copy import deepcopy
import gzip
import hashlib
import io
import unittest
from types import SimpleNamespace

from .analysis import (Meter, compare_callbacks, physical_digest, request_digest,
                       read_json, trace_lines, finish_pair, aggregate, validate_trace,
                       descriptor, ordered_hash)


class MemoryPath:
    """No filesystem artifacts, including failed/over-budget input fixtures."""
    def __init__(self, raw):
        self.raw, self.read_bytes = raw, 0

    def resolve(self):
        return 'synthetic-memory-input'

    def stat(self):
        return SimpleNamespace(st_size=len(self.raw))

    def open(self, mode, buffering=0):
        assert mode == 'rb'
        parent = self

        class Stream(io.BytesIO):
            def read(self, n=-1):
                block = super().read(n)
                parent.read_bytes += len(block)
                return block
        return Stream(self.raw)

    def identity(self):
        return dict(path='synthetic.gz', size=len(self.raw), sha256=hashlib.sha256(self.raw).hexdigest())


def record(chosen, inputs=None):
    return dict(chosen=chosen, inputs={'planned': [], 'candidates': ['a', 'b']} if inputs is None else inputs,
                unit=0, target=[1, 2], op='WATER', unit_request=['NORTH'])


class AnalysisTests(unittest.TestCase):
    def test_request_semantics_are_typed_but_mapping_order_independent(self):
        self.assertEqual(request_digest({'a': [1], 'b': -0.}), request_digest({'b': 0., 'a': [1]}))
        self.assertNotEqual(request_digest([True]), request_digest([1]))
        self.assertNotEqual(request_digest([1]), request_digest([1.]))
        self.assertNotEqual(request_digest([1, 2]), request_digest([2, 1]))
        with self.assertRaises(ValueError):
            request_digest(float('nan'))

    def test_alignment_stops_on_first_choice_even_if_later_context_matches(self):
        result = compare_callbacks([record('a'), record('a')], [record('b'), record('b')])
        self.assertEqual(result['matched_context_calls'], 1)
        self.assertEqual(result['choice']['callback'], 0)
        self.assertIsNone(result['context_stop'])

    def test_alignment_stops_on_input_and_does_not_pair_later_callbacks(self):
        result = compare_callbacks([record('a', {'x': 1}), record('a')],
                                   [record('a', {'x': 2}), record('b')])
        self.assertEqual(result['matched_context_calls'], 0)
        self.assertIsNone(result['choice'])
        self.assertEqual(result['context_stop'], 'input_difference')
        self.assertEqual(compare_callbacks([], [record('a')])['context_stop'], 'callback_count_difference')

    def test_physical_semantics_exclude_actions_and_keep_both_inventory_orders(self):
        frame = [dict(action=['PASS'], observation=dict(step=0, remainingOverageTime=1,
                 private=dict(inventories=[{'WHEAT': 1, 'CARROT': 2}]))),
                 dict(action=['PASS'], observation=dict(private=dict(inventories=[{'WHEAT': 3, 'CARROT': 4}])))]
        changed = deepcopy(frame)
        changed[0]['action'] = ['NORTH']
        changed[0]['observation']['remainingOverageTime'] = 99
        self.assertEqual(physical_digest(frame), physical_digest(changed))
        for side in (0, 1):
            changed = deepcopy(frame)
            inv = changed[side]['observation']['private']['inventories'][0]
            changed[side]['observation']['private']['inventories'][0] = dict(reversed(list(inv.items())))
            self.assertNotEqual(physical_digest(frame), physical_digest(changed))

    def test_one_pass_stream_hash_and_concatenated_gzip_trace(self):
        path = MemoryPath(gzip.compress(b'{"step":0}\n') + gzip.compress(b'{"step":1}\n'))
        meter = Meter()
        self.assertEqual(list(trace_lines(path, path.identity(), meter)), [{'step': 0}, {'step': 1}])
        self.assertEqual(path.read_bytes, len(path.raw))
        self.assertEqual(meter.compressed, path.read_bytes)
        self.assertEqual(meter.decoded, len(b'{"step":0}\n{"step":1}\n'))
        self.assertEqual(len(meter.receipts), 1)
        with self.assertRaisesRegex(ValueError, 'Second raw read'):
            list(trace_lines(path, path.identity(), meter))
        self.assertEqual(path.read_bytes, len(path.raw))

    def test_input_limits_are_applied_before_actual_reads(self):
        path = MemoryPath(gzip.compress(b'{"payload":"abcdef"}'))
        meter = Meter(compressed_limit=5)
        with self.assertRaises(ValueError):
            read_json(path, path.identity(), meter)
        self.assertLessEqual(path.read_bytes, 5)
        path = MemoryPath(gzip.compress(b'{"payload":"abcdef"}'))
        meter = Meter(decoded_limit=4)
        with self.assertRaisesRegex(ValueError, 'Decoded budget'):
            read_json(path, path.identity(), meter)
        self.assertEqual(meter.decoded, 4)
        self.assertEqual(len(meter.receipts), 0)

    def test_exact_decoded_cap_completes_without_buffered_overread(self):
        path = MemoryPath(gzip.compress(b'{}'))
        meter = Meter(decoded_limit=2)
        self.assertEqual(read_json(path, path.identity(), meter), {})
        self.assertEqual(meter.decoded, 2)
        self.assertEqual(len(meter.receipts), 1)
        path = MemoryPath(gzip.compress(b'{} '))
        meter = Meter(decoded_limit=2)
        with self.assertRaisesRegex(ValueError, 'Decoded budget'):
            read_json(path, path.identity(), meter)
        self.assertEqual(meter.decoded, 2)
        self.assertEqual(len(meter.receipts), 0)

    def test_mismatched_identity_is_not_a_completed_input(self):
        path = MemoryPath(gzip.compress(b'{}'))
        identity, meter = path.identity(), Meter()
        identity['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Raw identity'):
            read_json(path, identity, meter)
        self.assertEqual(len(meter.receipts), 0)

    def test_horizon_clipping_unknown_and_completion_screen(self):
        replay = dict(banks=[[10, 5]]*720, physical=['same']*720,
                      components=[{'farmer': 'a', 'hands': 'b', 'market': 'c'}]*719)
        pair = dict(seed=1, opponent='market', side=0, reference='teacher_task',
            first_request_difference_step=718, first_choice_difference={'step': 718},
            first_request_turn_choice_linkage={'step': 718},
            common_prefix_active=True, unalignable_common_turns=0)
        result = finish_pair(pair, replay, replay)
        self.assertEqual(result['horizons']['1']['state_index'], 719)
        self.assertEqual(result['horizons']['24']['state_index'], 719)
        self.assertEqual(result['partition'], 'request_difference_no_recorded_observation_effect')
        self.assertFalse(result['duration_mechanism_supported'])
        self.assertEqual(aggregate([result])['status'], 'evidence_insufficient')
        pair = dict(seed=1, opponent='market', side=0, reference='teacher_task',
            first_request_difference_step=None, first_choice_difference=None,
            first_request_turn_choice_linkage=None,
            common_prefix_active=True, unalignable_common_turns=1)
        self.assertEqual(finish_pair(pair, replay, replay)['partition'], 'unknown')

    def test_unlinked_earlier_choice_does_not_explain_first_request(self):
        replay = dict(banks=[[10, 5]]*720, physical=['same']*720,
                      components=[{'farmer': 'a', 'hands': 'b', 'market': 'c'}]*719)
        pair = dict(seed=1, opponent='market', side=0, reference='teacher_task',
            first_request_difference_step=10, first_choice_difference={'step': 0},
            first_request_turn_choice_linkage=None, common_prefix_active=True, unalignable_common_turns=0)
        self.assertEqual(finish_pair(pair, replay, replay)['partition'], 'unknown')

    def test_unsampled_transient_effect_does_not_expand_registered_window(self):
        replay = dict(banks=[[10, 5]]*720, physical=['same']*720,
                      components=[{'farmer': 'a', 'hands': 'b', 'market': 'c'}]*719)
        other = deepcopy(replay)
        other['physical'][12] = 'transient at plus two only'
        pair = dict(seed=1, opponent='market', side=0, reference='teacher_task',
            first_request_difference_step=10, first_choice_difference={'step': 10},
            first_request_turn_choice_linkage={'step': 10}, common_prefix_active=True, unalignable_common_turns=0)
        result = finish_pair(pair, replay, other)
        self.assertEqual(result['partition'], 'request_difference_no_recorded_observation_effect')
        self.assertIsNone(result['first_sampled_observation_difference_state'])

    def test_missing_candidate_features_and_emitted_request_mismatch_close(self):
        observation = {'private': {'inventories': [{}], 'shed': {}}}
        inputs = dict(shared_x=[], observation=observation, configuration={},
            inventory_item_order=[[]], shed_item_order=[],
            planned={k: [] for k in ('seeds', 'needed_seeds', 'supply', 'projected_shed', 'unit_actions', 'available_units')},
            candidates=[dict(id='a', unit=0, target=[1, 2], op='WATER', x=[])])
        r = record('a', inputs); r['teacher_label'] = 'a'
        trace = dict(step=0, records=[r], load_errors=0, inference_errors=0, fallbacks=0)
        replay = dict(actor_observation=[ordered_hash(observation)], configuration=[ordered_hash({})],
            units=[[request_digest(['NORTH'])]], days=[0], positions=[[[0, 0]]])
        validate_trace(trace, replay, 0, descriptor())
        for owner, key in ((inputs, 'shared_x'), (inputs['candidates'][0], 'x')):
            value = owner.pop(key)
            with self.assertRaises(KeyError):
                validate_trace(trace, replay, 0, descriptor())
            owner[key] = value
        r['unit_request'] = ['SOUTH']
        with self.assertRaisesRegex(ValueError, 'emitted unit request'):
            validate_trace(trace, replay, 0, descriptor())


if __name__ == '__main__':
    unittest.main()
