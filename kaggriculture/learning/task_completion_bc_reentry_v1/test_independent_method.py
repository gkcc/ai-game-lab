import unittest
from unittest.mock import patch
from . import independent_method as method
from learning.task_action_impact_diagnostic_v1 import analysis
from .independent_method import IndependentEpisodes, choose_sample, effect

class IndependentMethodTests(unittest.TestCase):
    def test_selection_is_identity_only_and_fixed(self):
        ids = [('g', i, 0) for i in range(30)]
        self.assertEqual(choose_sample(ids), choose_sample(list(reversed(ids))))
        self.assertEqual(len(choose_sample(ids)), 20)
        with self.assertRaises(ValueError): choose_sample(ids[:19])
        with self.assertRaises(ValueError): choose_sample(ids + ids[:1])

    def test_episode_continuation_and_changed_task(self):
        tracker = IndependentEpisodes('g', 0)
        def selection(step, op):
            return dict(step=step, ordinal=0, unit=0, target=[1, 1], op=op, chosen=op, unit_request=[op])
        tracker.step(0, [selection(0, 'DIG')], [], 0)
        tracker.step(1, [selection(1, 'DIG')], [], 0)
        tracker.step(2, [selection(2, 'WATER')], [], 0)
        rows = tracker.finish()
        self.assertEqual([(r['first_step'], r['last_step'], r['status']) for r in rows], [(0,1,'unknown'),(2,2,'unknown')])
        self.assertFalse(effect('WATER', {}, 0))

class EntryFailureTests(unittest.TestCase):
    def test_partial_marker_records_primary_and_unstarted(self):
        saved = []
        def partial(path, state):
            state['created'] = True
            raise OSError('synthetic write failure')
        with patch.object(method, 'ready', return_value={}), patch.object(method, 'verify_commit'), \
             patch.object(method, 'verify_model_seal'), patch.object(method, 'create_marker', side_effect=partial), \
             patch.object(method, 'emit', side_effect=lambda p, v: saved.append(v)):
            with self.assertRaises(OSError): method.main()
        self.assertEqual((saved[0]['error_kind'], saved[0]['marker_complete'], saved[0]['replay_state']), ('OSError', False, 'not_started'))

    def test_meter_failure_and_receipt_failure_preserve_errors(self):
        def marker(path, state): state.update(created=True, complete=True)
        cfg = {'independent_limits': {'compressed_bytes': 1, 'decoded_bytes': 1}}
        with patch.object(method, 'ready', return_value=cfg), patch.object(method, 'verify_commit'), \
             patch.object(method, 'verify_model_seal'), patch.object(method, 'create_marker', side_effect=marker), \
             patch.object(analysis, 'Meter', side_effect=ValueError('synthetic')), \
             patch.object(method, 'emit', side_effect=PermissionError('synthetic')):
            with self.assertRaises(method.FailureReceiptError) as caught: method.main()
        self.assertEqual((caught.exception.original_error_kind, caught.exception.receipt_error_kind), ('ValueError', 'PermissionError'))

if __name__ == '__main__': unittest.main()
