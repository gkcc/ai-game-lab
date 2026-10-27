import unittest
from unittest.mock import patch
from . import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_seed_exclusion(self):
        first = e.select_seed(set())
        self.assertNotIn(first,e.EXCLUDED)
        second = e.select_seed({first, *e.EXCLUDED})
        self.assertNotEqual(first,second)
        self.assertNotIn(second,e.EXCLUDED)

    def test_dirty_head(self):
        def git(*args): return b'base\n' if args[0] == 'rev-parse' else b' M changed\n'
        with patch.object(e,'git',side_effect=git), self.assertRaises(ValueError): e.clean_head('base')
        with patch.object(e,'git',return_value=b'wrong\n'), self.assertRaises(ValueError): e.clean_head('base')

    def test_deadline(self):
        for value in ('2000-01-01T00:00:00+00:00','2999-01-01T00:00:00'):
            with patch.object(e,'read',return_value={'deadline_utc':value}), self.assertRaises(TimeoutError): e.deadline_check()

    def test_identity_tampering(self):
        with patch.object(e,'sha',return_value='new'), self.assertRaises(ValueError): e.check_identity('unused','old')


if __name__ == '__main__': unittest.main()
