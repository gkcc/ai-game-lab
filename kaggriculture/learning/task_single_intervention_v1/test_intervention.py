"""Synthetic checks for one-shot model-call and action-submission boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_single_intervention_v1.intervention import (
    ACTION_TURNS, BUDGET_KEYS, OneShotSelector,
)


def inputs(step):
    return dict(observation=dict(step=step, player=0),
                candidates=[dict(id='student'), dict(id='teacher'), dict(id='other')],
                planned=dict(ordered=['first', 'second']))


class Fixture:
    def __init__(self, mode='intervene', expected=None, selector_result='student', charge=1,
                 mutate_target=False):
        self.records = []
        self.total = {key: 0 for key in BUDGET_KEYS}
        self.turn_budget = dict(self.total)
        self.calls = 0
        self.selector_result = selector_result
        self.charge = charge
        self.mutate_target = mutate_target
        self.expected = expected or dict(input_sha256=ordered_hash(inputs(2)),
                                         original_choice='student', teacher_choice='teacher')
        self.gate = OneShotSelector(self.original, lambda: len(self.records), self.budget,
                                    self.expected, mode)

    def budget(self):
        return dict(total=self.total, turn=self.turn_budget)

    def original(self, value):
        self.calls += 1
        if self.charge:
            for counts in (self.total, self.turn_budget):
                counts['task_forwards'] += self.charge
                counts['task_rows'] += self.charge*len(value['candidates'])
        if self.mutate_target and value['observation']['step'] == 2 and len(self.records) == 5:
            value['planned']['ordered'].append('mutated')
        return self.selector_result

    def begin(self):
        self.records = []
        self.turn_budget = {key: 0 for key in BUDGET_KEYS}
        self.total['controller_calls'] += 1
        self.turn_budget['controller_calls'] += 1

    def choose(self, value, teacher='teacher', source='callback'):
        choice = self.gate(value)
        self.records.append(dict(inputs=deepcopy(value), teacher_label=teacher, chosen=choice, source=source))
        return choice

    def verify(self, step):
        self.gate.verify_turn(dict(step=step, records=self.records))

    def before_target(self):
        for step in (0, 1):
            self.begin()
            self.choose(inputs(step))
            self.verify(step)
        self.begin()
        for _ in range(5):
            self.choose(inputs(2))


class InterventionTest(unittest.TestCase):
    def test_control_and_intervene_differ_only_at_target_and_use_one_forward(self):
        proofs = []
        for mode, expected_choice in (('control', 'student'), ('intervene', 'teacher')):
            f = Fixture(mode)
            f.before_target()
            before = f.calls
            self.assertEqual(expected_choice, f.choose(inputs(2)))
            self.assertEqual(before+1, f.calls)
            self.assertEqual('student', f.choose(inputs(2)))
            f.verify(2)
            proof = f.gate.proof()
            self.assertTrue(proof['teacher_label_verified'])
            self.assertEqual('student', proof['original_choice'])
            self.assertEqual(expected_choice, proof['returned_choice'])
            self.assertEqual(1, proof['budget_after']['total']['task_forwards']-
                                proof['budget_before']['total']['task_forwards'])
            for step in range(3, ACTION_TURNS):
                f.begin()
                f.verify(step)
            self.assertTrue(f.gate.verify_complete())
            self.assertEqual(9, f.calls)
            proofs.append(proof)
        self.assertEqual(proofs[0]['budget_before'], proofs[1]['budget_before'])
        self.assertEqual(proofs[0]['budget_after'], proofs[1]['budget_after'])

    def test_missing_target_fails_before_action_and_cannot_later_yield_proof(self):
        f = Fixture()
        f.before_target()
        with self.assertRaises(ValueError):
            f.verify(2)
        with self.assertRaises(ValueError):
            f.gate.proof()
        with self.assertRaises(ValueError):
            f.gate(inputs(2))

    def test_changed_or_reordered_target_inputs_fail_before_model_call(self):
        for mutate in ('value', 'order'):
            f = Fixture()
            f.before_target()
            value = inputs(2)
            if mutate == 'value':
                value['planned']['ordered'][0] = 'changed'
            else:
                value['candidates'].reverse()
            before = f.calls
            with self.assertRaises(ValueError):
                f.choose(value)
            self.assertEqual(before, f.calls)

    def test_target_missing_or_duplicate_candidate_is_rejected(self):
        for candidates in ([dict(id='student')], [dict(id='student'), dict(id='student')]):
            value = inputs(2)
            value['candidates'] = candidates
            expected = dict(input_sha256=ordered_hash(value), original_choice='student', teacher_choice='teacher')
            f = Fixture(expected=expected)
            f.before_target()
            with self.assertRaises(ValueError):
                f.choose(value)

    def test_target_original_prediction_must_match_recorded_prediction(self):
        f = Fixture(selector_result='other')
        f.before_target()
        before = f.calls
        with self.assertRaises(ValueError):
            f.choose(inputs(2))
        self.assertEqual(before+1, f.calls)
        with self.assertRaises(ValueError):
            f.gate.proof()

    def test_wrong_teacher_label_or_non_callback_source_fails_closed(self):
        for teacher, source in (('other', 'callback'), ('teacher', 'teacher')):
            f = Fixture()
            f.before_target()
            f.choose(inputs(2), teacher=teacher, source=source)
            with self.assertRaises(ValueError):
                f.verify(2)
            with self.assertRaises(ValueError):
                f.gate.proof()

    def test_wrong_index_or_duplicate_callback_is_rejected(self):
        f = Fixture()
        f.begin()
        f.records.append({})
        with self.assertRaises(ValueError):
            f.gate(inputs(0))
        self.assertEqual(0, f.calls)
        g = Fixture()
        g.before_target()
        g.choose(inputs(2))
        before = g.calls
        g.records.pop()
        with self.assertRaises(ValueError):
            g.gate(inputs(2))
        self.assertEqual(before, g.calls)

    def test_incomplete_turn_order_and_bad_training_fields_fail_closed(self):
        f = Fixture()
        with self.assertRaises(ValueError):
            f.gate.verify_turn(dict(step=1, records=[]))
        g = Fixture()
        g.begin()
        value = inputs(0)
        value['teacher_label'] = 'teacher'
        with self.assertRaises(ValueError):
            g.gate(value)
        self.assertEqual(0, g.calls)

    def test_missing_or_double_budget_charge_cannot_pass_target(self):
        for charge in (0, 2):
            f = Fixture(charge=charge)
            f.before_target()
            with self.assertRaises(ValueError):
                f.choose(inputs(2))
            with self.assertRaises(ValueError):
                f.gate.proof()

    def test_original_selector_cannot_mutate_the_recorded_target_input(self):
        f = Fixture(mutate_target=True)
        f.before_target()
        with self.assertRaises(ValueError):
            f.choose(inputs(2))
        with self.assertRaises(ValueError):
            f.gate.proof()

    def test_proof_and_completion_require_verified_target_and_complete_episode(self):
        f = Fixture()
        with self.assertRaises(ValueError):
            f.gate.proof()
        with self.assertRaises(ValueError):
            f.gate.verify_turn(dict(step=0, records=[]))
        g = Fixture()
        g.before_target()
        g.choose(inputs(2))
        g.verify(2)
        proof = g.gate.proof()
        proof['budget_before']['total']['task_rows'] = -1
        self.assertNotEqual(-1, g.gate.proof()['budget_before']['total']['task_rows'])
        with self.assertRaises(ValueError):
            g.gate.verify_complete()
        with self.assertRaises(ValueError):
            g.gate.proof()


if __name__ == '__main__':
    unittest.main()
