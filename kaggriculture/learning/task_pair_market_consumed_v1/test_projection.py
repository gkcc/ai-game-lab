"""Synthetic projection checks; no private game data is opened."""
import json
import unittest

from learning.task_pair_historical_compare_v1.test_projection import SEEDS, fixture
from .projection import project


class ProjectionTests(unittest.TestCase):
    def prefixes(self, status='observed'):
        reason = 'actor_request_difference' if status == 'observed' else 'all_requests_equal'
        return {(alias, side): dict(scope='same_complete_history', status=status, reason=reason,
                                   step=17 if status == 'observed' else None)
                for alias in ('S02', 'S03') for side in (0, 1)}

    def test_new_roles_and_private_identity_are_not_published(self):
        paired, rows, receipts = fixture()
        result = project(paired, seeds=SEEDS, rows=rows, comparisons=receipts, prefixes=self.prefixes())
        self.assertEqual([item['seed_alias'] for item in result['pairs']], ['D01', 'D01', 'D02', 'D02'])
        self.assertEqual(set(result['by_seed']), {'D01', 'D02'})
        self.assertTrue(result['development_screen']['passed'])
        self.assertFalse(result['promotion'])
        rendered = json.dumps(result)
        for value in (*map(str, SEEDS.values()), 'S01', 'S04', 'PRIVATE-REQUEST'):
            self.assertNotIn(value, rendered)

    def test_no_request_difference_fails_screen_and_invalid_prefix_rejected(self):
        paired, rows, receipts = fixture()
        result = project(paired, seeds=SEEDS, rows=rows, comparisons=receipts,
                         prefixes=self.prefixes('no_difference'))
        self.assertFalse(result['development_screen']['passed'])
        prefixes = self.prefixes()
        prefixes[('S02', 0)]['step'] = True
        with self.assertRaises(ValueError):
            project(paired, seeds=SEEDS, rows=rows, comparisons=receipts, prefixes=prefixes)


if __name__ == '__main__':
    unittest.main()
