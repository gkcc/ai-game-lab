"""End-to-end synthetic check of the two rule probes and prefix alignment."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from learning.task_multi_teacher_probe_v1.test_teacher import SOURCE, sha
from .alignment import align_turn
from .opponent import SupportTeacher
from .terminal import TerminalTeacherProbe
from .test_terminal import sample


class IntegrationTests(unittest.TestCase):
    def test_both_seats_keep_mismatched_reservations_unknown(self):
        for side in (0, 1):
            with self.subTest(side=side):
                observation, configuration = sample()
                observation['player'] = side
                original = deepcopy((observation, configuration))
                guard = lambda: None
                # This pathway is strictly rule-based: even a mistaken attempt
                # to construct the crop neural network must fail the test.
                with patch('learning.bc_crop.agent.Network', side_effect=AssertionError('unexpected model')):
                    teacher = SupportTeacher(SOURCE, sha(SOURCE), 'market', guard=guard)
                    terminal = TerminalTeacherProbe(guard=guard)
                    try:
                        teacher(observation, configuration)
                        terminal(observation, configuration)
                        inputs = teacher.last['records'][0]['inputs']
                        schema = {key: inputs[key] for key in
                                  ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')}
                        counts = align_turn(observation, configuration,
                                            teacher.last['records'], terminal.last['records'],
                                            schema=schema, guard=guard)
                        self.assertEqual((counts['teacher_callbacks'], counts['terminal_callbacks']), (1, 2))
                        self.assertEqual((counts['supported'], counts['unsupported'], counts['unknown']),
                                         (0, 0, 1))
                        self.assertEqual((observation, configuration), original)
                    finally:
                        teacher.clear()
                        terminal.clear()
                    self.assertIsNone(teacher.last)
                    self.assertIsNone(terminal.last)


if __name__ == '__main__':
    unittest.main()
