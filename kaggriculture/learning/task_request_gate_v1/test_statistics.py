"""Observable gate, complete-panel and paired-cluster statistical regressions."""
from copy import deepcopy
import unittest

import numpy as np

from .statistics import (ARMS, BOOTSTRAP, DELTAS, HOLDOUT, OPPONENTS,
                         development_gate, early_signal, holdout_statistics, training_signal)


def collection(seeds):
    return [dict(seed=s, opponent=o, side=side, eligible=True,
                 delta_own=2. if side == 0 else -1., delta_opponent=0.,
                 delta_margin=2. if side == 0 else -1., delta_score=0.,
                 baseline_net_profit=100., x8=[0.]*8, prediction=1. if side == 0 else -1.)
            for s in seeds for o in OPPONENTS for side in (0, 1)]


def missing(row):
    row.update(eligible=False, x8=None, prediction=None, **{k: None for k in DELTAS})


def panel(seed_nets=None, increment=20., teacher_increment=10.):
    seeds = list(range(100, 108))
    nets = seed_nets if seed_nets is not None else [1000.]*8
    cfg = dict(test_seeds=seeds, opponents=list(OPPONENTS), arms=list(ARMS),
               bootstrap=dict(BOOTSTRAP), holdout=dict(HOLDOUT))
    rows = []
    for i, seed in enumerate(seeds):
        for opponent in OPPONENTS:
            for side in (0, 1):
                for arm in ARMS:
                    change = increment if arm == 'gated' else teacher_increment if arm == 'teacher_task' else 0.
                    net = nets[i]+change
                    opponent_net = 50.
                    margin = net-opponent_net
                    rows.append(dict(seed=seed, opponent=opponent, side=side, policy=arm,
                        bank=net+100., opponent_bank=opponent_net+100., initial_bank=100., opponent_initial_bank=100.,
                        net_profit=net, opponent_net_profit=opponent_net, margin=margin,
                        score=1. if margin > 0 else 0. if margin < 0 else .5,
                        passed=True, status=['DONE', 'DONE'], states=720, action_turns=719,
                        agent_errors=0, complete_request_changes=1 if arm == 'gated' else 0))
    return rows, cfg


class CollectionTests(unittest.TestCase):
    def test_single_early_check_needs_four_pairs_and_real_margin_signal(self):
        seeds = [10, 11]
        rows = collection(seeds)
        for row in rows[4:]:
            missing(row)
        self.assertTrue(early_signal(rows, seeds)['passed'])
        for row in rows[:4]:
            row.update(delta_margin=0., delta_own=0.)
        self.assertFalse(early_signal(rows, seeds)['passed'])
        rows[0].update(delta_own=1., delta_margin=1.)
        missing(rows[3])
        self.assertFalse(early_signal(rows, seeds)['passed'])

    def test_train_keeps_zero_negative_and_has_no_sign_quota(self):
        seeds = list(range(8))
        rows = collection(seeds)
        for row in rows[:4]:
            row.update(delta_margin=0., delta_own=0.)
        result = training_signal(rows, seeds)
        self.assertTrue(result['passed'])
        self.assertEqual(result['counts']['eligible'], 32)
        self.assertEqual(result['counts']['zero_margin'], 4)
        self.assertEqual(result['counts']['negative_margin'], 14)
        for row in rows:
            row.update(delta_own=abs(row['delta_own']), delta_margin=abs(row['delta_margin']))
        self.assertTrue(training_signal(rows, seeds)['passed'])

    def test_missing_targets_are_not_zero_imputed_and_population_is_exact(self):
        seeds = list(range(8))
        rows = collection(seeds)
        for row in rows[:9]:
            missing(row)
        result = training_signal(rows, seeds)
        self.assertFalse(result['passed'])
        self.assertEqual(result['counts']['missing'], 9)
        rows[0]['delta_margin'] = 0.
        with self.assertRaises(ValueError):
            training_signal(rows, seeds)
        with self.assertRaises(ValueError):
            training_signal(collection(seeds)[:-1], seeds)
        duplicate = collection(seeds)
        duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaises(ValueError):
            training_signal(duplicate, seeds)

    def test_dev_must_strictly_beat_both_constant_choices(self):
        seeds = [20, 21]
        rows = collection(seeds)
        result = development_gate(rows, seeds)
        self.assertTrue(result['passed'])
        self.assertEqual(result['equal_seed_means']['chosen_margin'], 1.)
        self.assertEqual(result['equal_seed_means']['always_teacher_margin'], .5)
        for row in rows:
            row['prediction'] = 1.
        always_teacher = development_gate(rows, seeds)
        self.assertTrue(always_teacher['checks']['beats_constant_student'])
        self.assertFalse(always_teacher['checks']['beats_constant_teacher'])
        self.assertFalse(always_teacher['passed'])
        for row in rows:
            row['prediction'] = 0.
        self.assertFalse(development_gate(rows, seeds)['passed'])

    def test_dev_own_score_and_each_seed_protections(self):
        seeds = [20, 21]
        rows = collection(seeds)
        for row in rows:
            if row['side'] == 0:
                row.update(delta_own=-10., delta_opponent=-12.)
        result = development_gate(rows, seeds)
        self.assertTrue(result['checks']['beats_constant_student'])
        self.assertFalse(result['checks']['own_relative_ge_minus_2percent'])
        rows = collection(seeds)
        rows[0]['delta_score'] = -.5
        self.assertFalse(development_gate(rows, seeds)['passed'])
        rows = collection(seeds)
        for row in rows:
            if row['seed'] == 20:
                row['prediction'] = -row['prediction']
        self.assertFalse(development_gate(rows, seeds)['checks']['each_seed_selected_margin_nonnegative'])

    def test_dev_missing_effect_is_zero_but_target_remains_unknown(self):
        seeds = [20, 21]
        rows = collection(seeds)
        missing(rows[1])
        result = development_gate(rows, seeds)
        self.assertTrue(result['passed'])
        self.assertIsNone(result['paired_detail'][1]['observed_delta']['delta_margin'])
        self.assertEqual(result['paired_detail'][1]['chosen_margin'], 0.)


class HoldoutTests(unittest.TestCase):
    def test_positive_research_retention_and_teacher_superiority(self):
        rows, cfg = panel()
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['research_positive'])
        self.assertTrue(result['teacher_retention'])
        self.assertTrue(result['teacher_superiority'])
        self.assertEqual(result['comparisons']['baseline']['overall']['delta_ci95']['margin'], [20., 20.])
        self.assertEqual(result['comparisons']['teacher_task']['overall']['delta_ci95']['margin'], [10., 10.])
        self.assertEqual(len(result['paired_detail']), 32)

    def test_student_improvement_does_not_establish_teacher_retention(self):
        rows, cfg = panel(teacher_increment=100.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['research_positive'])
        self.assertFalse(result['teacher_retention'])
        self.assertFalse(result['teacher_superiority'])
        rows, cfg = panel(teacher_increment=25.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['research_positive'])
        self.assertTrue(result['teacher_retention'])
        self.assertFalse(result['teacher_superiority'])

    def test_margin_and_real_request_effect_are_required(self):
        rows, cfg = panel(increment=0.)
        self.assertFalse(holdout_statistics(rows, cfg)['research_positive'])
        rows, cfg = panel()
        for row in rows:
            row['complete_request_changes'] = 0
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['student_margin_gate_passed'])
        self.assertFalse(result['research_positive'])

    def test_shared_whole_seed_draws_and_recomputed_ratio_denominators(self):
        nets = np.array([100., 100., 100., 100., 900., 900., 900., 900.])
        rows, cfg = panel(seed_nets=nets, increment=10., teacher_increment=2.)
        result = holdout_statistics(rows, cfg)
        draws = np.random.Generator(np.random.PCG64(BOOTSTRAP['seed'])).integers(0, 8, size=(20000, 8))
        for reference, offset, delta in (('baseline', 0., 10.), ('teacher_task', 2., 8.)):
            reference_nets = nets+offset
            expected = np.quantile(delta/reference_nets[draws].mean(axis=1), [.025, .975], method='linear')
            actual = result['comparisons'][reference]['overall']
            self.assertAlmostEqual(actual['relative_net_profit_mean'], delta/float(reference_nets.mean()))
            np.testing.assert_array_equal(actual['relative_net_profit_ci95'], expected)
            self.assertNotAlmostEqual(actual['relative_net_profit_mean'], float(np.mean(delta/reference_nets)))
            self.assertGreater(expected[1]-expected[0], 0.)

    def test_market_regression_cannot_hide_behind_other_opponent(self):
        rows, cfg = panel(increment=500.)
        for row in rows:
            if row['policy'] == 'gated' and row['opponent'] == 'market':
                row.update(net_profit=900., bank=1000., margin=850.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['student_margin_gate_passed'])
        self.assertFalse(result['student_opponent_point_guards']['market'])
        self.assertFalse(result['research_positive'])

    def test_incomplete_duplicate_or_inconsistent_third_arm_is_rejected(self):
        rows, cfg = panel()
        with self.assertRaises(ValueError):
            holdout_statistics(rows[:-1], cfg)
        duplicate = deepcopy(rows)
        duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaises(ValueError):
            holdout_statistics(duplicate, cfg)
        rows[-1]['net_profit'] += 1
        with self.assertRaises(ValueError):
            holdout_statistics(rows, cfg)

    def test_nonpositive_reference_denominator_blocks_relative_claims(self):
        rows, cfg = panel(seed_nets=[0.]*8, teacher_increment=0.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['student_margin_gate_passed'])
        self.assertFalse(result['research_positive'])
        self.assertFalse(result['teacher_retention'])
        self.assertIsNone(result['comparisons']['teacher_task']['overall']['relative_net_profit_ci95'])


if __name__ == '__main__':
    unittest.main()
