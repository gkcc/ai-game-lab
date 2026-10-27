import unittest
from unittest.mock import patch
from learning.task_completion_bc_reentry_v1 import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_seed_inventory_excludes_known_and_selected(self):
        first = e.select_seeds(set())
        self.assertEqual(len(set(first)), 2)
        self.assertFalse(set(first) & e.EXCLUDED)
        second = e.select_seeds(set(first))
        self.assertFalse(set(first) & set(second))
        self.assertFalse(set(second) & e.EXCLUDED)

    def test_clean_head_rejects_changes(self):
        def fake(*args):
            if args[0] == 'rev-parse': return b'expected\n'
            return b' M changed\n'
        with patch.object(e, 'git', side_effect=fake):
            with self.assertRaises(ValueError): e.clean_head('expected')
        with patch.object(e, 'git', return_value=b'other\n'):
            with self.assertRaises(ValueError): e.clean_head('expected')

    def test_deadline_requires_timezone_and_future(self):
        with patch.object(e, 'read', return_value={'deadline_utc':'2000-01-01T00:00:00+00:00'}):
            with self.assertRaises(TimeoutError): e.deadline_check()
        with patch.object(e, 'read', return_value={'deadline_utc':'2999-01-01T00:00:00'}):
            with self.assertRaises(TimeoutError): e.deadline_check()


if __name__ == '__main__': unittest.main()
