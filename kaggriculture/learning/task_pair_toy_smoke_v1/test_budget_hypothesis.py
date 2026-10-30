"""Synthetic arithmetic witness for the failed toy smoke's budget hypothesis.

The last completed turn exposed aggregate counters and two candidate counts.
Reusing those counts here demonstrates a possible next-call failure only. This
test does not read the trace, reconstruct the next state, run the game, or
establish which exception actually stopped the registered attempt.
"""
import unittest

from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from learning.task_request_gate_v1.policy import _Budget


class BudgetHypothesisTests(unittest.TestCase):
    def test_last_completed_counts_and_previous_candidate_sizes_can_exhaust_budget(self):
        budget = _Budget(dict(GAME_LIMITS))
        task_calls = 0
        # An artificial schedule reaches the same cumulative totals using only
        # the real budget API; it does not claim to reproduce earlier turns.
        for turn in range(580):
            budget.begin_turn()
            budget.controller()
            for _ in range(6 if turn < 341 else 5):
                budget.forward('task', 50 if task_calls < 391 else 49)
                task_calls += 1
            budget.forward('crop', 1)
            if turn < 9:
                budget.forward('crop', 1)
        self.assertEqual((budget.counts['controller_calls'], budget.counts['task_forwards'],
                          budget.counts['task_rows'], budget.counts['crop_forwards'],
                          budget.counts['crop_rows']), (580, 3241, 159200, 589, 589))
        self.assertEqual(GAME_LIMITS['neural_input_rows'] - budget.counts['task_rows']
                         - budget.counts['crop_rows'], 211)

        # These two sizes were observed in the previous completed turn, not in
        # the missing failure turn. Their reuse is strictly a counterexample.
        budget.begin_turn()
        budget.controller()
        budget.forward('task', 125)
        before_total, before_turn = dict(budget.counts), dict(budget.turn)
        with self.assertRaisesRegex(RuntimeError, '^Shared neural computation budget exhausted$'):
            budget.forward('task', 96)
        self.assertEqual(budget.counts, before_total)
        self.assertEqual(budget.turn, before_turn)


if __name__ == '__main__':
    unittest.main()
