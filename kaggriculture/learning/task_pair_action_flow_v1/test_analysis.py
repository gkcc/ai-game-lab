"""Synthetic common-history tests; no private data or official matches."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_action_impact_diagnostic_v1 import analysis as impact
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_pair_dev_metrics_v1.test_consumer import turn
from learning.task_prefix_diagnostic_v1.test_pairing import action, compact, set_actor
from .analysis import TURNS, compare_pair


CONFIGURATION = {'boardSize': 10}
CANDIDATES = [
    dict(id='a', unit=0, target=[0, 1], op='WATER', x=[1.]),
    dict(id='b', unit=1, target=[1, 1], op='WATER', x=[2.]),
    dict(id='c', unit=0, target=[2, 1], op='WATER', x=[3.]),
]


def observation(step):
    return dict(player=0, step=step, private=dict(inventories=[{}, {}], shed={}),
                marker='PRIVATE-OBSERVATION')


def compact_fixture():
    result = compact()
    result['actor_observation'] = [ordered_hash(observation(i)) for i in range(TURNS)]
    result['configuration'] = [ordered_hash(CONFIGURATION) for _ in range(TURNS)]
    result['units'] = [[impact.request_digest(['NORTH']), impact.request_digest(['EAST'])]
                       for _ in range(TURNS)]
    result['days'] = [1] * TURNS
    result['positions'] = [[[0, 0], [1, 0]] for _ in range(TURNS)]
    return result


def records(step, ids, *, changed_context=False):
    planned = dict(seeds={}, needed_seeds={}, supply={}, projected_shed={},
                   unit_actions=[], available_units=[])
    inputs = dict(shared_x=[1.], candidates=deepcopy(CANDIDATES), planned=planned,
                  observation=observation(step), configuration=deepcopy(CONFIGURATION),
                  inventory_item_order=[[], []], shed_item_order=[])
    if changed_context:
        inputs['extra_context'] = 'different'
    result = []
    for name in ids:
        candidate = next(c for c in CANDIDATES if c['id'] == name)
        unit = candidate['unit']
        result.append(dict(source='callback', chosen=name, teacher_label='a', unit=unit,
                           target=list(candidate['target']), op=candidate['op'],
                           unit_request=['NORTH'] if unit == 0 else ['EAST'],
                           inputs=deepcopy(inputs)))
    return result


def lines(overrides=None):
    values = overrides or {}
    return (turn(step, values.get(step, [])) for step in range(TURNS))


def compare(ref, new, reference=None, candidate=None, check=lambda: True, side=0):
    return compare_pair(ref, new, lines(reference), lines(candidate), side=side, check=check)


class FlowAnalysisTests(unittest.TestCase):
    def test_different_choice_with_same_request_keeps_comparing_later_turns(self):
        ref, new = compact_fixture(), compact_fixture()
        set_actor(new, 3, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        new['banks'][4] = [1005, 997]
        new['physical'][4] = 'different-physical'
        reference = {0: records(0, ['a', 'b']), 1: records(1, ['a', 'b']), 3: records(3, ['a'])}
        candidate = {0: records(0, ['b', 'a']), 1: records(1, ['b', 'c']), 3: records(3, ['b'])}
        result = compare(ref, new, reference, candidate)
        counts = result['counts']
        self.assertEqual((result['first_branch_kind'], result['first_branch_step']), ('actor', 3))
        self.assertIs(result['actor_request_changed_at_first_branch'], True)
        self.assertEqual(result['actor_only_branch_attribution'], 'observed')
        self.assertEqual(result['actor_request_components_differ'], dict(farmer=True, hands=False, market=False))
        self.assertEqual(counts['eligible_turns'], 4)
        self.assertEqual(counts['choice_difference_turns'], 3)
        self.assertEqual(counts['choice_same_request_same_task_multiset_turns'], 1)
        self.assertEqual(counts['choice_same_request_different_task_multiset_turns'], 1)
        self.assertEqual(counts['choice_different_request_different_task_multiset_turns'], 1)
        self.assertEqual(counts['matched_context_selection_records'], 3)
        self.assertEqual(result['first_same_input_choice_difference']['step'], 0)
        window = result['post_branch_windows']['1']
        self.assertFalse(window['semantic_physical_state_equal'])
        self.assertEqual(window['cash_delta'], dict(own_bank=5, opponent_bank=-3, margin=8))
        self.assertNotIn('PRIVATE-OBSERVATION', json.dumps(result))

    def test_input_and_record_count_mismatches_are_separately_unknown(self):
        ref, new = compact_fixture(), compact_fixture()
        set_actor(new, 3, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        reference = {1: records(1, ['a']), 2: records(2, ['a'])}
        candidate = {1: records(1, ['a'], changed_context=True), 2: records(2, ['a', 'b'])}
        result = compare(ref, new, reference, candidate)
        counts = result['counts']
        self.assertEqual((counts['input_difference_turns'], counts['selection_record_count_difference_turns']), (1, 1))
        self.assertEqual(counts['choice_difference_turns'], 0)
        self.assertIsNone(result['first_same_input_choice_difference'])

    def test_state_or_opponent_branch_has_no_actor_attribution(self):
        for mode in ('state', 'opponent', 'simultaneous'):
            with self.subTest(mode=mode):
                ref, new = compact_fixture(), compact_fixture()
                if mode == 'state':
                    new['full'][2] = 'different-state'
                else:
                    new['requests'][2][1] = impact.request_digest(dict(farmer=['STAY'], hands=[], market=[]))
                    if mode == 'simultaneous':
                        set_actor(new, 2, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
                ref_turns = {0: records(0, ['a']), 2: records(2, ['a']), 3: records(3, ['a'])}
                new_turns = {0: records(0, ['b']), 2: records(2, ['b']), 3: records(3, ['b'])}
                result = compare(ref, new, ref_turns, new_turns)
                self.assertEqual(result['first_branch_kind'], mode)
                self.assertEqual(result['actor_only_branch_attribution'], 'unknown')
                self.assertIs(result['actor_request_changed_at_first_branch'],
                              True if mode == 'simultaneous' else False if mode == 'opponent' else None)
                self.assertEqual(result['counts']['choice_difference_turns'], 1 if mode == 'state' else 2)
                if mode == 'state':
                    self.assertIsNone(result['actor_request_components_differ'])
                elif mode == 'opponent':
                    self.assertEqual(result['actor_request_components_differ'],
                                     dict(farmer=False, hands=False, market=False))

    def test_sequential_plans_do_not_create_interchangeable_labels(self):
        ref, new = compact_fixture(), compact_fixture()
        equal_requests = dict(farmer=['NORTH'], hands=[['NORTH']], market=[])
        for compact in (ref, new):
            set_actor(compact, 0, equal_requests)
            compact['units'][0] = [impact.request_digest(['NORTH'])] * 2
        set_actor(new, 1, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        ref_records = records(0, ['a', 'b'])
        new_records = records(0, ['b', 'a'])
        for record in ref_records + new_records:
            record['unit_request'] = ['NORTH']
        # After selecting a vs. b, only the remaining unit's candidate is
        # available and resource reservations have a different recorded state.
        ref_records[1]['inputs']['candidates'] = [deepcopy(CANDIDATES[1])]
        ref_records[1]['teacher_label'] = 'b'
        ref_records[1]['inputs']['planned']['available_units'] = [1]
        ref_records[1]['inputs']['planned']['seeds'] = {'wheat': 1}
        new_records[1]['inputs']['candidates'] = [deepcopy(CANDIDATES[0])]
        new_records[1]['inputs']['planned']['available_units'] = [0]
        new_records[1]['inputs']['planned']['seeds'] = {'wheat': 0}
        result = compare(ref, new, {0: ref_records}, {0: new_records})
        self.assertEqual(result['counts']['choice_same_request_same_task_multiset_turns'], 1)
        self.assertEqual(result['counts']['matched_context_selection_records'], 1)
        first = result['first_same_input_choice_difference']
        self.assertFalse(first['same_unit'])
        self.assertTrue(first['selected_request_values_equal'])
        self.assertTrue(first['complete_actor_request_equal'])
        self.assertTrue(first['task_multiset_equal'])
        self.assertEqual(first['label_interchangeability'], 'not_inferred')
        self.assertNotIn('selected_unit_request_equal', first)

    def test_no_branch_and_terminal_state_have_no_first_request_value(self):
        ref, new = compact_fixture(), compact_fixture()
        result = compare(ref, new)
        self.assertIsNone(result['actor_request_changed_at_first_branch'])
        self.assertEqual(result['actor_only_branch_attribution'], 'no_branch')
        new['full'][TURNS] = 'different-terminal-state'
        result = compare(ref, new)
        self.assertEqual(result['first_branch_kind'], 'terminal_state')
        self.assertIsNone(result['actor_request_changed_at_first_branch'])
        self.assertEqual(result['actor_only_branch_attribution'], 'unknown')

    def test_trace_must_complete_and_resource_guard_stops(self):
        ref, new = compact_fixture(), compact_fixture()
        with self.assertRaises(ValueError):
            compare_pair(ref, new, iter([turn(0)]), lines(), side=0, check=lambda: True)
        def extra():
            yield from lines()
            yield turn(TURNS)
        with self.assertRaises(ValueError):
            compare_pair(ref, new, lines(), extra(), side=0, check=lambda: True)
        with self.assertRaises(ValueError):
            compare(ref, new, check=lambda: False)
        with self.assertRaises(ValueError):
            compare(ref, new, side=True)

    def test_different_actor_and_rule_teacher_are_not_model_comparisons(self):
        ref, new = compact_fixture(), compact_fixture()
        ref_turns, new_turns = {0: records(0, ['a'])}, {0: records(0, ['a'])}
        new_turns[0][0]['source'] = 'teacher'
        with self.assertRaises(ValueError):
            compare(ref, new, ref_turns, new_turns)
        new_turns = {0: records(0, ['a'])}
        new_turns[0][0]['inputs']['observation']['player'] = 1
        with self.assertRaises(ValueError):
            compare(ref, new, ref_turns, new_turns)

    def test_real_compact_uses_next_frame_for_the_current_request(self):
        def replay():
            farms = [dict(money=1000, farmer=[0, 0], hands=[[1, 0]]),
                     dict(money=1000, farmer=[9, 9], hands=[])]
            frames = []
            for step in range(TURNS + 1):
                frame = []
                for side in (0, 1):
                    obs = dict(player=side, step=step, day=1, hour=1, farms=deepcopy(farms),
                               market={}, town={}, private={})
                    frame.append(dict(observation=obs, action=action(),
                                      status='DONE' if step == TURNS else 'ACTIVE', reward=1000))
                frames.append(frame)
            return dict(steps=frames, configuration=deepcopy(CONFIGURATION))

        reference, candidate = replay(), replay()
        candidate['steps'][5][0]['action']['farmer'] = ['SOUTH']
        expected = dict(banks=[1000, 1000], bank=1000, opponent_bank=1000, margin=0,
                        net_profit=0, opponent_net_profit=0, rewards=[1000, 1000],
                        status=['DONE', 'DONE'], score=.5)
        game = dict(side=0, expected=expected)
        with patch.object(impact, 'observe', side_effect=lambda obs, cfg: SimpleNamespace(
                observation=obs, configuration=cfg)):
            ref_compact = impact.compact_replay(reference, game)
            new_compact = impact.compact_replay(candidate, game)
        result = compare_pair(ref_compact, new_compact, lines(), lines(), side=0, check=lambda: True)
        self.assertEqual((result['first_branch_kind'], result['first_branch_step']), ('actor', 4))


if __name__ == '__main__':
    unittest.main()
