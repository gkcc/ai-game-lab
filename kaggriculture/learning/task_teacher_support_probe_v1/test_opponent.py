"""A private intent annotation cannot alter a frozen synthetic teacher."""
import unittest

from learning.task_multi_teacher_probe_v1.test_teacher import SOURCE, sha, observation, configuration, plain
from learning.task_multi_teacher_probe_v1.teacher import ProbeError
from .opponent import SupportTeacher


class OpponentTests(unittest.TestCase):
    def test_selected_task_intent_is_recorded_without_changing_request(self):
        probe = SupportTeacher(SOURCE, sha(SOURCE), 'market')
        obs, cfg = observation(), configuration()
        self.assertEqual(probe(obs, cfg), plain(SOURCE, obs, cfg))
        self.assertEqual(probe.last['records'][0]['intended_action'], ['HARVEST'])
        probe.clear()
        self.assertIsNone(probe.last)
        with self.assertRaises(ProbeError): probe(obs, cfg)

    def test_failed_call_also_consumes_instance(self):
        probe = SupportTeacher(SOURCE, sha(SOURCE), 'market')
        bad = observation()
        bad['step'] = 719
        with self.assertRaises(ProbeError): probe(bad, configuration())
        with self.assertRaises(ProbeError): probe(observation(), configuration())


if __name__ == '__main__':
    unittest.main()
