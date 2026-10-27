"""Selector comparison must not infer request changes from choice alone."""
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch
from . import analysis, campaign
from .analysis import first_request_linked, matched_choice


def record(choice, inputs=None):
    return dict(inputs=inputs or {'known':1},chosen=choice,op='WATER',unit=0,target=[1,1],unit_request=['NORTH'])


class ContextTest(unittest.TestCase):
    def test_equal_inputs_and_choice_difference_with_same_request(self):
        x = matched_choice([record('a')],[record('b')],'same','same')
        self.assertTrue(x['choice_different'])
        self.assertFalse(x['complete_request_different'])
        self.assertEqual(x['matched_context_calls'],1)

    def test_equal_inputs_choice_and_request_both_differ(self):
        x = matched_choice([record('a')],[record('b')],'left','right')
        self.assertTrue(x['choice_different'] and x['complete_request_different'])

    def test_different_inputs_do_not_become_a_matched_choice(self):
        x = matched_choice([record('a',{'known':1})],[record('b',{'known':2})],'left','right')
        self.assertFalse(x['choice_different'])
        self.assertEqual(x['matched_context_calls'],0)
        self.assertEqual(x['context_stop'],'input_difference')

    def test_later_choice_difference_does_not_claim_first_request_control(self):
        self.assertFalse(first_request_linked(dict(first_linked_change_step=9,first_request_difference_step=3)))
        self.assertFalse(first_request_linked(dict(first_linked_change_step=None,first_request_difference_step=None)))
        self.assertTrue(first_request_linked(dict(first_linked_change_step=3,first_request_difference_step=3)))

    def test_partial_analysis_marker_keeps_failure_receipt(self):
        cfg = dict(test_seeds=[7],opponents=['market'],analysis_limits=dict(seconds=1,compressed_bytes=1,decoded_bytes=1))
        rows = [dict(game=f'{s}-{a}',seed=7,opponent='market',side=s,policy=a) for s in (0,1) for a in ('candidate','teacher_task')]
        finished = dict(passed=True,game_ledger_sha256='hash',audit_ledger_sha256='hash')
        def read(path): return finished if Path(path).name=='games-result.json' else rows
        receipts = []
        with patch.object(analysis,'ready',return_value=cfg), patch.object(analysis,'verify_commit'), \
             patch.object(campaign,'verify_independent_review'), patch.object(analysis,'read',side_effect=read), \
             patch.object(analysis,'sha',return_value='hash'), patch.object(Path,'exists',return_value=False), \
             patch.object(Path,'open',mock_open()), patch.object(analysis.json,'dump',side_effect=OSError('partial')), \
             patch.object(analysis,'write_new',side_effect=lambda p,v:receipts.append(v)):
            with self.assertRaises(OSError): analysis.run()
        self.assertTrue(receipts[0]['marker']['created'])
        self.assertFalse(receipts[0]['marker']['complete'])
        self.assertIsNone(receipts[0]['raw_reads'])

    def test_time_boundary_is_checked_before_further_work(self):
        with patch.object(analysis,'deadline_check'), patch.object(analysis.time,'monotonic',side_effect=[11,12.1]):
            analysis.check_budget(10,2)
            with self.assertRaises(TimeoutError): analysis.check_budget(10,2)


if __name__=='__main__': unittest.main()
