"""Stable population, missingness, reward-gate and clustered-inference checks."""
from copy import deepcopy
import unittest

import numpy as np

from .statistics import (ARMS, BOOTSTRAP, HOLDOUT, OPPONENTS,
                         development_gate, holdout_statistics, training_signal)


def collection(seeds):
    return [dict(seed=s, opponent=o, side=side, eligible=True,
                 delta_own=2.0 if side == 0 else -1.0,
                 x12=[0.0]*12, prediction=1.0 if side == 0 else -1.0)
            for s in seeds for o in OPPONENTS for side in (0, 1)]


def panel(seed_nets=None, increment=20.0):
    seeds = list(range(100, 108))
    nets = seed_nets if seed_nets is not None else [1000.0]*8
    cfg = dict(test_seeds=seeds, opponents=list(OPPONENTS), arms=list(ARMS),
               bootstrap=dict(BOOTSTRAP), holdout=dict(HOLDOUT))
    rows = []
    for i, s in enumerate(seeds):
        for o in OPPONENTS:
            for side in (0, 1):
                for arm in ARMS:
                    initial = 100.0
                    net = nets[i] + (increment if arm == 'gated' else 0.0)
                    opponent_net = 50.0
                    margin = net-opponent_net
                    rows.append(dict(seed=s, opponent=o, side=side, policy=arm,
                        bank=net+initial, opponent_bank=opponent_net+initial,
                        initial_bank=initial, opponent_initial_bank=initial,
                        net_profit=net, opponent_net_profit=opponent_net,
                        margin=margin, score=1.0 if margin > 0 else 0.0 if margin < 0 else .5,
                        passed=True, status=['DONE', 'DONE'], states=720, action_turns=719,
                        agent_errors=0, complete_request_changes=1 if arm == 'gated' else 0))
    return rows, cfg


class CollectionTests(unittest.TestCase):
    def test_training_keeps_zero_and_negative_effects(self):
        seeds = list(range(8))
        rows = collection(seeds)
        for row in rows[:4]:
            row['delta_own'] = 0.0
        result = training_signal(rows, seeds)
        self.assertTrue(result['passed'])
        self.assertEqual(result['counts']['eligible'], 32)
        self.assertEqual(result['counts']['zero'], 4)
        self.assertEqual(result['counts']['negative'], 14)

    def test_missing_cells_stay_unknown_and_can_stop_fit(self):
        seeds = list(range(8))
        rows = collection(seeds)
        for row in rows[:9]:
            row.update(eligible=False, delta_own=None, x12=None, prediction=None)
        result = training_signal(rows, seeds)
        self.assertFalse(result['passed'])
        self.assertEqual(result['counts']['missing'], 9)
        rows[0]['delta_own'] = 0.0
        with self.assertRaises(ValueError):
            training_signal(rows, seeds)

    def test_no_favorable_sign_filter_or_population_substitution(self):
        seeds = list(range(8))
        rows = collection(seeds)
        for row in rows:
            row['delta_own'] = abs(row['delta_own'])
        self.assertFalse(training_signal(rows, seeds)['passed'])
        with self.assertRaises(ValueError):
            training_signal(rows[:-1], seeds)
        rows[-1] = deepcopy(rows[0])
        with self.assertRaises(ValueError):
            training_signal(rows, seeds)

    def test_development_uses_decision_effect_and_both_seed_guards(self):
        seeds = [20, 21]
        rows = collection(seeds)
        result = development_gate(rows, seeds)
        self.assertTrue(result['passed'])
        self.assertEqual(result['positive_predictions'], 4)
        self.assertEqual(result['mean_realized_delta_own'], 1.0)
        # Overall gain remains positive, but one seed suffers: it must fail.
        for row in rows:
            if row['seed'] == 20:
                row['prediction'] = -row['prediction']
        failed = development_gate(rows, seeds)
        self.assertGreater(failed['mean_realized_delta_own'], 0)
        self.assertFalse(failed['passed'])

    def test_development_missing_effect_zero_not_missing_training_zero(self):
        seeds = [20, 21]
        rows = collection(seeds)
        rows[1].update(eligible=False, delta_own=None, x12=None, prediction=None)
        result = development_gate(rows, seeds)
        self.assertTrue(result['passed'])
        missing = result['paired_detail'][1]
        self.assertIsNone(missing['observed_delta_own'])
        self.assertEqual(missing['realized_chooser_delta_own'], 0.0)
        for row in rows:
            if row['eligible']:
                row['prediction'] = 0.0
        self.assertFalse(development_gate(rows, seeds)['passed'])


class HoldoutTests(unittest.TestCase):
    def test_positive_panel_and_zero_effect_panel(self):
        rows, cfg = panel()
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['research_positive'])
        self.assertEqual(result['overall']['relative_net_profit_ci95'], [.02, .02])
        self.assertEqual(result['complete_action_changed_games'], 32)
        rows, cfg = panel(increment=0.0)
        zero = holdout_statistics(rows, cfg)
        self.assertFalse(zero['research_positive'])
        self.assertEqual(zero['overall']['relative_net_profit_ci95'], [0.0, 0.0])

    def test_complete_request_effect_is_required(self):
        rows, cfg = panel()
        for row in rows:
            row['complete_request_changes'] = 0
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['primary_gate_passed'])
        self.assertFalse(result['research_positive'])

    def test_other_opponent_gain_cannot_hide_market_regression(self):
        rows, cfg = panel(increment=500.0)
        for row in rows:
            if row['policy'] == 'gated' and row['opponent'] == 'market':
                row['net_profit'] = 900.0
                row['bank'] = row['initial_bank']+row['net_profit']
                row['margin'] = row['bank']-row['opponent_bank']
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['primary_gate_passed'])
        self.assertFalse(result['opponent_point_guards']['market'])
        self.assertFalse(result['research_positive'])

    def test_ratio_of_means_and_whole_seed_denominator_resampling(self):
        nets = [100.0, 100.0, 100.0, 100.0, 900.0, 900.0, 900.0, 900.0]
        rows, cfg = panel(seed_nets=nets, increment=10.0)
        result = holdout_statistics(rows, cfg)
        self.assertAlmostEqual(result['overall']['relative_net_profit_mean'], .02)
        # Eight independent clusters: four identical rows per seed move together.
        draws = np.random.Generator(np.random.PCG64(BOOTSTRAP['seed'])).integers(0, 8, size=(20000, 8))
        expected = np.quantile(10.0/np.asarray(nets)[draws].mean(axis=1), [.025, .975], method='linear')
        np.testing.assert_array_equal(result['overall']['relative_net_profit_ci95'], expected)
        # This is neither mean of per-game ratios nor a fixed denominator CI.
        self.assertNotAlmostEqual(result['overall']['relative_net_profit_mean'],
                                  float(np.mean(10.0/np.asarray(nets))))
        self.assertGreater(expected[1]-expected[0], 0)

    def test_incomplete_duplicate_cash_errors_and_config_change_rejected(self):
        rows, cfg = panel()
        with self.assertRaises(ValueError):
            holdout_statistics(rows[:-1], cfg)
        duplicate = deepcopy(rows)
        duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaises(ValueError):
            holdout_statistics(duplicate, cfg)
        incorrect = deepcopy(rows)
        incorrect[0]['net_profit'] += 1
        with self.assertRaises(ValueError):
            holdout_statistics(incorrect, cfg)
        changed_cfg = deepcopy(cfg)
        changed_cfg['bootstrap']['replicates'] = 1000
        with self.assertRaises(ValueError):
            holdout_statistics(rows, changed_cfg)

    def test_nonpositive_denominator_is_undefined_and_fails(self):
        rows, cfg = panel(seed_nets=[0.0]*8)
        result = holdout_statistics(rows, cfg)
        self.assertFalse(result['research_positive'])
        self.assertFalse(result['overall']['relative_net_profit_defined'])
        self.assertIsNone(result['overall']['relative_net_profit_ci95'])


if __name__ == '__main__':
    unittest.main()
