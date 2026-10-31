"""Baseline source selection reuses the authenticated old evidence chain."""
from unittest.mock import patch
import unittest

from learning.task_student_continuation_v1 import samples as old_samples
from learning.task_student_continuation_v1 import sources as old_sources
from learning.task_student_continuation_v1.test_sources import Fixture
from . import sources


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        # Populate both baseline seats using the prior in-memory receipt fixture.
        for role in old_samples.ROLES:
            for opponent in old_samples.OPPONENTS:
                for side in (0, 1):
                    coordinate = dict(role=role, opponent=opponent, side=side, arm='baseline')
                    with patch.object(old_samples, 'source_coordinates', return_value=coordinate):
                        self.fx.make_game(role, opponent)
        self.fx.refresh_campaign()
        for item in self.fx.patches():
            item.start()
            self.addCleanup(item.stop)

    def context(self):
        return old_sources.open_context(campaign_sha256=self.fx.campaign_sha, guard=lambda: None)

    def test_both_baseline_seats_and_one_game_lifetime(self):
        context = self.context()
        for side in (0, 1):
            with sources.read_game(context, role='F01', opponent='market', side=side,
                                   guard=lambda: None) as game:
                self.assertEqual((game.identity['role'], game.identity['opponent'], game.identity['side'],
                                  game.identity['arm']), ('F01', 'market', side, 'baseline'))
                self.assertEqual([None if s is None else s['step'] for s in game.selected],
                                 [17, None, 250, None, 500, None])
                self.assertEqual(len(game.replay['steps']), 720)
                with self.assertRaises(old_sources.SourceError):
                    sources.read_game(context, role='F01', opponent='market', side=1 - side,
                                      guard=lambda: None)
            self.assertIsNone(game.replay)
            self.assertIsNone(game.selected)

    def test_tampered_receipt_stops_before_replay_open(self):
        context = self.context()
        cohort, alias, _seed = context.role_map['F01']
        name = f'dev-{alias}-market-0-baseline'
        path = old_sources.fresh.ARTIFACTS / old_sources.entry.STAGES[cohort] / 'job-results' / (name + '.row.json')
        self.fx.files[str(path)] += b' '
        self.fx.opened.clear()
        with self.assertRaises(old_sources.SourceError):
            sources.read_game(context, role='F01', opponent='market', side=0, guard=lambda: None)
        self.assertFalse(any('replay.json.gz' in p for p in self.fx.opened))
        self.assertFalse(context.active)

    def test_invalid_coordinate_is_rejected(self):
        context = self.context()
        for side in (True, -1, 2):
            with self.assertRaises(old_sources.SourceError):
                sources.read_game(context, role='F01', opponent='market', side=side, guard=lambda: None)


if __name__ == '__main__':
    unittest.main()
