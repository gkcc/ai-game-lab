"""Synthetic regressions for historical-prefix comparison semantics."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_action_impact_diagnostic_v1.analysis import request_digest
from learning.task_prefix_diagnostic_v1.pairing import PairComparison, TURNS


def action(market=None):
    return dict(farmer=['NORTH'], hands=[['EAST']], market=market or [])


def compact():
    actor = action()
    request = request_digest(actor)
    opponent = request_digest(dict(farmer=['WEST'], hands=[], market=[]))
    components = {k: request_digest(actor[k]) for k in ('farmer', 'hands', 'market')}
    return dict(full=['same-state']*(TURNS+1), physical=['same-physical']*(TURNS+1),
        banks=[[1000, 1000] for _ in range(TURNS+1)],
        requests=[[request, opponent] for _ in range(TURNS)],
        components=[dict(components) for _ in range(TURNS)],
        actor_observation=['actor-view']*TURNS, configuration=['configuration']*TURNS)


def set_actor(value, step, updated, side=0):
    value['requests'][step][side] = request_digest(updated)
    value['components'][step] = {k: request_digest(updated[k]) for k in ('farmer', 'hands', 'market')}


def record(chosen, unit, request, *, target=None, inputs=None, teacher='A', op='WATER', source='callback'):
    return dict(chosen=chosen, unit=unit, unit_request=request, target=target if target is not None else unit,
        inputs={'context': 'same'} if inputs is None else inputs, teacher_label=teacher, op=op, source=source)


def consume(pair, turns):
    for step in range(TURNS):
        left, right = turns.get(step, ([], []))
        pair.observe_turn(step, left, right)
    return pair.finish()


class PairComparisonTest(unittest.TestCase):
    def test_branch_types_and_no_divergence(self):
        a = compact()
        for kind in ('actor', 'opponent', 'simultaneous', 'state', 'terminal_state', 'none'):
            b = deepcopy(a)
            if kind in ('actor', 'simultaneous'):
                set_actor(b, 3, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
            if kind in ('opponent', 'simultaneous'):
                b['requests'][3][1] = request_digest(dict(farmer=['STAY'], hands=[], market=[]))
            if kind == 'state':
                b['full'][3] = 'different-state'
            if kind == 'terminal_state':
                b['full'][TURNS] = 'different-terminal'
            result = consume(PairComparison(a, b, 0, 'round2', 'unweighted'), {})
            self.assertEqual(kind, result['first_branch_kind'])
            self.assertEqual(TURNS if kind == 'terminal_state' else None if kind == 'none' else 3,
                             result['first_branch_step'])
            self.assertEqual(3 if kind == 'state' else 4 if kind in ('actor', 'opponent', 'simultaneous') else TURNS,
                             result['eligible_turns'])
            self.assertEqual('unknown', result['branch_attribution'])
            if kind not in ('actor', 'simultaneous'):
                self.assertIsNone(result['first_actor_request_components_differ'])
            if kind != 'actor':
                self.assertIsNone(result['post_branch_windows'])

    def test_choice_difference_can_preserve_complete_action_and_swap_task_order(self):
        a, b = compact(), compact()
        set_actor(b, 1, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        first = record('A', 0, ['NORTH'])
        second = record('B', 1, ['EAST'])
        pair = PairComparison(a, b, 0, 'market', 'unweighted')
        result = consume(pair, {0: ([first, second], [second, first]),
                                2: ([first], [second])})
        self.assertEqual(1, result['choice_difference_turns'])
        self.assertEqual(1, result['choice_difference_same_actor_request_turns'])
        self.assertEqual(1, result['choice_difference_same_request_same_task_multiset_turns'])
        self.assertFalse(result['first_choice_difference']['selected_unit_request_equal'])
        self.assertTrue(result['first_choice_difference']['complete_actor_request_equal'])
        self.assertEqual(1, result['matched_context_selection_records'])
        self.assertEqual(1, result['teacher_label_consistent_selection_records'])
        self.assertEqual(1, result['candidate_teacher_agreement_selection_records'])
        self.assertEqual(0, result['reference_teacher_agreement_selection_records'])
        self.assertEqual(2, result['eligible_turns'])
        self.assertFalse(result['choice_difference_at_branch'])

    def test_zero_input_length_and_teacher_label_categories_reconcile(self):
        a, b = compact(), compact()
        set_actor(b, 4, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        same = record('A', 0, ['NORTH'])
        other_input = record('A', 0, ['NORTH'], inputs={'context': 'different'})
        different_teacher = record('A', 0, ['NORTH'], teacher='B')
        changed_choice = record('B', 0, ['SOUTH'])
        pair = PairComparison(a, b, 0, 'teacher', 'teacher_task')
        result = consume(pair, {1: ([same], [other_input]),
                                2: ([same], [same, record('B', 1, ['EAST'])]),
                                3: ([same], [different_teacher]),
                                4: ([same], [changed_choice])})
        self.assertEqual(1, result['zero_both_selection_record_turns'])
        self.assertEqual(1, result['input_difference_turns'])
        self.assertEqual(1, result['selection_record_count_difference_turns'])
        self.assertEqual(1, result['equal_recorded_choices_turns'])
        self.assertEqual(1, result['choice_difference_turns'])
        self.assertEqual(1, result['choice_difference_different_actor_request_turns'])
        self.assertEqual(3, result['matched_context_selection_records'])
        self.assertEqual(1, result['teacher_label_disagreement_selection_records'])
        self.assertEqual(2, result['teacher_label_consistent_selection_records'])
        self.assertTrue(result['choice_difference_at_branch'])

    def test_callback_and_rule_teacher_sources_are_selection_records(self):
        a, b = compact(), compact()
        set_actor(b, 0, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        student = record('A', 0, ['NORTH'], teacher='B', source='callback')
        teacher = record('B', 0, ['SOUTH'], teacher='B', source='teacher')
        result = consume(PairComparison(a, b, 0, 'teacher', 'teacher_task'),
                         {0: ([student], [teacher])})
        self.assertEqual(1, result['matched_context_selection_records'])
        self.assertEqual(1, result['teacher_label_consistent_selection_records'])
        self.assertEqual(0, result['candidate_teacher_agreement_selection_records'])
        self.assertEqual(1, result['reference_teacher_agreement_selection_records'])
        self.assertEqual(1, result['choice_difference_turns'])
        self.assertEqual(0, result['first_choice_difference']['selection_record_index'])
        self.assertNotIn('callback', result['first_choice_difference'])
        self.assertFalse(any('callback' in key or key.endswith('_calls') for key in result))

    def test_market_order_is_a_real_complete_request_difference(self):
        a, b = compact(), compact()
        set_actor(a, 0, action([['SELL', 'CARROT', 1], ['SELL', 'MELON', 2]]))
        set_actor(b, 0, action([['SELL', 'MELON', 2], ['SELL', 'CARROT', 1]]))
        result = consume(PairComparison(a, b, 0, 'round2', 'unweighted'), {})
        self.assertEqual('actor', result['first_branch_kind'])
        self.assertEqual({'farmer': False, 'hands': False, 'market': True},
                         result['first_actor_request_components_differ'])

    def test_seat_one_actor_and_opponent_are_not_swapped(self):
        a, b = compact(), compact()
        a['requests'] = [list(reversed(pair)) for pair in a['requests']]
        b['requests'] = [list(reversed(pair)) for pair in b['requests']]
        set_actor(b, 5, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]), side=1)
        b['banks'][6] = [1000, 1005]
        result = consume(PairComparison(a, b, 1, 'round2', 'unweighted'), {})
        self.assertEqual('actor', result['first_branch_kind'])
        self.assertEqual(5, result['first_branch_step'])
        self.assertTrue(result['post_branch_windows']['1']['own_bank_delta_changed'])
        self.assertFalse(result['post_branch_windows']['1']['opponent_bank_delta_changed'])

    def test_rival_or_simultaneous_branch_still_compares_only_the_branch_turn(self):
        for simultaneous in (False, True):
            a, b = compact(), compact()
            b['requests'][2][1] = request_digest(dict(farmer=['STAY'], hands=[], market=[]))
            if simultaneous:
                set_actor(b, 2, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
            first = record('A', 0, ['NORTH'])
            second = record('B', 1, ['EAST'])
            result = consume(PairComparison(a, b, 0, 'market', 'teacher_task'),
                             {2: ([first], [second]), 3: ([first], [second])})
            self.assertEqual('simultaneous' if simultaneous else 'opponent', result['first_branch_kind'])
            self.assertEqual(3, result['eligible_turns'])
            self.assertEqual(1, result['choice_difference_turns'])
            self.assertEqual('unknown', result['branch_attribution'])
            self.assertIsNone(result['post_branch_windows'])

    def test_unequal_actor_input_does_not_count_as_equal_selection_context(self):
        a, b = compact(), compact()
        b['actor_observation'][0] = 'different-view-order'
        set_actor(b, 0, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        same = record('A', 0, ['NORTH'])
        result = consume(PairComparison(a, b, 0, 'round2', 'unweighted'), {0: ([same], [same])})
        self.assertEqual(1, result['actor_observation_or_configuration_mismatch_turns'])
        self.assertEqual(0, result['matched_context_selection_records'])
        self.assertIsNone(result['choice_difference_at_branch'])

    def test_windows_expose_boolean_changes_not_cash_or_private_state(self):
        a, b = compact(), compact()
        set_actor(b, 3, dict(farmer=['SOUTH'], hands=[['EAST']], market=[]))
        b['banks'][4] = [1005, 1000]
        b['banks'][27] = [1005, 1003]
        b['physical'][4] = 'different-physical'
        result = consume(PairComparison(a, b, 0, 'round2', 'unweighted'), {})
        one = result['post_branch_windows']['1']
        self.assertFalse(one['semantic_physical_state_equal'])
        self.assertTrue(one['own_bank_delta_changed'])
        self.assertFalse(one['opponent_bank_delta_changed'])
        self.assertTrue(one['margin_delta_changed'])
        self.assertEqual({'state_index', 'semantic_physical_state_equal', 'own_bank_delta_changed',
                          'opponent_bank_delta_changed', 'margin_delta_changed'}, set(one))
        self.assertTrue(result['post_branch_windows']['24']['opponent_bank_delta_changed'])

    def test_bad_input_or_trace_order_fails_closed(self):
        a, b = compact(), compact()
        b['components'][0]['farmer'] = 'inconsistent'
        with self.assertRaises(ValueError):
            PairComparison(a, b, 0, 'round2', 'unweighted')
        pair = PairComparison(a, compact(), 0, 'round2', 'unweighted')
        with self.assertRaises(ValueError):
            pair.observe_turn(1, [], [])
        with self.assertRaises(ValueError):
            pair.finish()
        with self.assertRaises(ValueError):
            PairComparison(a, compact(), True, 'round2', 'unweighted')


if __name__ == '__main__':
    unittest.main()
