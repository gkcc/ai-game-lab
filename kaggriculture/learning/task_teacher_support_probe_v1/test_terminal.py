"""Synthetic one-turn terminal-teacher checks; no game or learned model."""
from copy import deepcopy
import gc
import unittest
from unittest.mock import patch
import weakref

from learning.task_choice_v1.agent import TaskAgent
from learning.task_multi_teacher_probe_v1.test_teacher import configuration, observation
from .terminal import MAX_CALLBACKS_PER_TURN, MAX_CANDIDATES, TerminalProbeError, TerminalTeacherProbe


def sample():
    obs, cfg = observation(), configuration()
    obs['market']['prices']['FERTILIZER'] = 100
    return obs, cfg


class TerminalTests(unittest.TestCase):
    def test_synthetic_turn_preserves_teacher_request_and_records_intent(self):
        obs, cfg = sample()
        saved_obs, saved_cfg = deepcopy(obs), deepcopy(cfg)
        reference = TaskAgent(task_selector=None, crop_mode='teacher')
        expected = reference(obs, cfg)
        probe = TerminalTeacherProbe(guard=lambda: None)
        self.assertIsNone(probe.task_selector)
        self.assertEqual(probe.crop.mode, 'teacher')
        self.assertIsNone(probe.crop.network)
        self.assertEqual(probe(obs, cfg), expected)
        self.assertEqual((obs, cfg), (saved_obs, saved_cfg))
        self.assertEqual(repr(probe), '<PrivateTerminalTeacherProbe>')
        self.assertGreater(len(probe.last['records']), 0)
        self.assertGreater(len(probe.last['crop_records']), 0)
        self.assertEqual([r['teacher_label'] for r in probe.last['records']],
                         [r['teacher_label'] for r in reference.last['records']])
        for record in probe.last['records']:
            self.assertEqual(record['source'], 'teacher')
            self.assertEqual(record['teacher_label'], record['chosen'])
            self.assertEqual(record['intended_action'][0], record['op'])
        # A PLANT task is captured before the separate crop decision.
        plants = [r for r in probe.last['records'] if r['op'] == 'PLANT']
        self.assertTrue(plants)
        self.assertTrue(all(r['intended_action'] == ['PLANT'] for r in plants))
        self.assertTrue(all(r['source'] == 'teacher' for r in probe.last['crop_records']))

    def test_candidate_and_callback_limits_are_checked_before_encoder(self):
        probe = TerminalTeacherProbe(guard=lambda: None)
        probe._running = True
        try:
            with patch('learning.task_choice_v1.agent.encode_candidates',
                       side_effect=AssertionError('encoder must not run')) as encoder:
                with self.assertRaises(TerminalProbeError):
                    probe.select({'choices': [None] * (MAX_CANDIDATES + 1)})
                probe._callbacks = MAX_CALLBACKS_PER_TURN
                with self.assertRaises(TerminalProbeError):
                    probe.select({'choices': [None]})
                encoder.assert_not_called()
        finally:
            probe._running = False

    def test_boundary_choice_captures_an_independent_pre_execution_action(self):
        probe = TerminalTeacherProbe(guard=lambda: None)
        probe._running, probe._callbacks = True, MAX_CALLBACKS_PER_TURN - 1
        task = ([0, 0], ['PLANT', 'WHEAT'], 1.0)
        ctx = {'choices': [(1.0, 0, 0)] * MAX_CANDIDATES, 'tasks': [task]}
        def parent(agent, _context):
            agent.records.append({'source': 'teacher'})
            return (1.0, 0, 0)
        try:
            with patch.object(TaskAgent, 'select', autospec=True, side_effect=parent) as base:
                self.assertEqual(probe.select(ctx), (1.0, 0, 0))
                base.assert_called_once()
            task[1][1] = 'MELON'
            self.assertEqual(probe.records[0]['intended_action'], ['PLANT', 'WHEAT'])
            self.assertEqual(probe._callbacks, MAX_CALLBACKS_PER_TURN)
        finally:
            probe._running = False

    def test_clear_releases_owned_task_and_crop_references_and_does_not_reset_use(self):
        class Marker:
            pass
        probe = TerminalTeacherProbe(guard=lambda: None)
        probe(*sample())
        refs = []
        for container in (probe.last, probe.records[0], probe.crop.last, probe.crop.records[0]):
            marker = Marker()
            refs.append(weakref.ref(marker))
            container['synthetic_marker'] = marker
        del marker, container
        probe.clear()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertIsNone(probe.last)
        self.assertEqual(probe.records, [])
        self.assertIsNone(probe.crop.last)
        self.assertEqual(probe.crop.records, [])
        with self.assertRaises(TerminalProbeError):
            probe(*sample())

    def test_clear_and_reentry_during_active_turn_fail_and_clear(self):
        for operation in ('clear', 'reenter'):
            with self.subTest(operation=operation):
                holder = {}
                def guard():
                    if 'probe' in holder and holder['probe']._running:
                        if operation == 'clear':
                            holder['probe'].clear()
                        else:
                            holder['probe'](*sample())
                probe = TerminalTeacherProbe(guard=guard)
                holder['probe'] = probe
                with self.assertRaises(TerminalProbeError):
                    probe(*sample())
                self.assertIsNone(probe.last)
                self.assertEqual(probe.records, [])
                self.assertIsNone(probe.crop.last)
                self.assertEqual(probe.crop.records, [])
                with self.assertRaises(TerminalProbeError):
                    probe(*sample())

    def test_exception_after_partial_callback_discards_owned_records(self):
        probe = TerminalTeacherProbe(guard=lambda: None)
        def broken(agent, _context):
            agent.records.append({'synthetic': 'task'})
            agent.crop.records.append({'synthetic': 'crop'})
            raise OSError('synthetic resource failure')
        with patch.object(TaskAgent, 'select', autospec=True, side_effect=broken):
            with self.assertRaises(OSError):
                probe(*sample())
        self.assertIsNone(probe.last)
        self.assertEqual(probe.records, [])
        self.assertIsNone(probe.crop.last)
        self.assertEqual(probe.crop.records, [])
        with self.assertRaises(TerminalProbeError):
            probe(*sample())

    def test_empty_invalid_guard_and_direct_callback_are_refused(self):
        with self.assertRaises(TerminalProbeError):
            TerminalTeacherProbe(guard=None)
        probe = TerminalTeacherProbe(guard=lambda: None)
        with self.assertRaises(TerminalProbeError):
            probe.select({'choices': []})


if __name__ == '__main__':
    unittest.main()
