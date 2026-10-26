"""Synthetic statistical contracts; no models, game execution or real data."""
from copy import deepcopy
import json
import unittest

import numpy as np

from .statistics import (ARMS, BOOTSTRAP, DEVELOPMENT, EARLY, HOLDOUT,
                         development_gate, early_gate, holdout_statistics)


def configuration():
    return dict(train_seeds=[101, 102, 103, 104], validation_seeds=[201, 202],
        test_seeds=list(range(301, 309)), opponents=['round2', 'market'], seats=[0, 1],
        arms=list(ARMS), bootstrap=dict(BOOTSTRAP), early_screen=dict(EARLY),
        development=dict(DEVELOPMENT), holdout=dict(HOLDOUT))


def economic(row, own, opponent=100.):
    row.update(bank=3000.+own, opponent_bank=3000.+opponent,
        initial_bank=3000., opponent_initial_bank=3000., net_profit=own,
        opponent_net_profit=opponent, margin=own-opponent,
        score=1. if own > opponent else .5 if own == opponent else 0.)


def panel(split='holdout', candidate=20., teacher=10., nets=None):
    cfg = configuration()
    seeds = cfg['validation_seeds' if split == 'dev' else 'test_seeds']
    nets = [1000.]*len(seeds) if nets is None else nets
    rows = []
    for i, seed in enumerate(seeds):
        for opponent in cfg['opponents']:
            for side in cfg['seats']:
                for arm in ARMS:
                    row = dict(game=f'{split}-{seed}-{opponent}-{side}-{arm}',
                        split=split, seed=seed, opponent=opponent, side=side, policy=arm,
                        passed=True, states=720, action_turns=719, status=['DONE', 'DONE'],
                        agent_errors=0)
                    if arm == 'candidate':
                        row['first_request_difference_step'] = 0
                    economic(row, nets[i]+(candidate if arm == 'candidate' else teacher if arm == 'teacher_task' else 0.))
                    rows.append(row)
    return rows, cfg


def early(disagreements=(64, 64, 0, 0)):
    cfg = configuration()
    rows = []
    for (opponent, side), n in zip([(o, s) for o in cfg['opponents'] for s in cfg['seats']], disagreements):
        rows.append(dict(game=f'train-101-{opponent}-{side}-baseline', split='train',
            policy='baseline', seed=101, opponent=opponent, side=side, passed=True,
            label_binding_passed=True, states=720, action_turns=719,
            status=['DONE', 'DONE'], agent_errors=0, groups=200, candidate_rows=800,
            teacher_ID_disagreements=n))
    return rows, cfg


class EarlyTests(unittest.TestCase):
    def test_quantity_and_game_coverage_are_separate_gates(self):
        for counts, expected in (((64, 64, 0, 0), True), ((64, 63, 0, 0), False),
                                 ((128, 0, 0, 0), False)):
            rows, cfg = early(counts)
            result = early_gate(rows, cfg)
            self.assertEqual(result['passed'], expected)
            self.assertEqual(result['counts']['teacher_ID_disagreements'], sum(counts))

    def test_bad_binding_or_nonregistered_cells_are_rejected(self):
        changes = [('label_binding_passed', False), ('split', 'dev'), ('seed', 102),
                   ('policy', 'candidate'), ('teacher_ID_disagreements', 201),
                   ('candidate_rows', 199), ('groups', True), ('action_turns', 718)]
        for key, value in changes:
            with self.subTest(key=key):
                rows, cfg = early()
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    early_gate(rows, cfg)
        rows, cfg = early()
        with self.assertRaises(ValueError):
            early_gate(rows[:-1], cfg)
        rows[-1] = deepcopy(rows[0])
        with self.assertRaises(ValueError):
            early_gate(rows, cfg)


class DevelopmentTests(unittest.TestCase):
    def test_fixed_model_development_needs_real_profit_improvement(self):
        rows, cfg = panel('dev')
        result = development_gate(rows, cfg)
        self.assertTrue(result['passed'])
        self.assertFalse(result['independent_holdout'])
        self.assertFalse(result['confidence_intervals_computed'])
        self.assertEqual(len(result['paired_detail']), 8)
        self.assertNotIn('delta_ci95', result['comparisons']['baseline']['overall'])
        rows, cfg = panel('dev', candidate=0., teacher=0.)
        result = development_gate(rows, cfg)
        self.assertFalse(result['checks']['own_mean_improved'])
        self.assertFalse(result['passed'])

    def test_development_margin_and_teacher_retention_guard(self):
        rows, cfg = panel('dev')
        for row in rows:
            if row['policy'] == 'candidate':
                economic(row, 1020., 130.)
        result = development_gate(rows, cfg)
        self.assertTrue(result['checks']['own_mean_improved'])
        self.assertFalse(result['checks']['overall_margin_not_lower'])
        self.assertFalse(result['passed'])
        rows, cfg = panel('dev', teacher=100.)
        result = development_gate(rows, cfg)
        self.assertFalse(result['checks']['teacher_mean_retained'])
        self.assertFalse(result['passed'])


class HoldoutTests(unittest.TestCase):
    def test_three_distinct_acceptance_claims(self):
        rows, cfg = panel()
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['research_positive'])
        self.assertTrue(result['teacher_retention'])
        self.assertTrue(result['admission'])
        self.assertEqual(result['clusters'], 8)
        self.assertEqual(result['comparisons']['baseline']['overall']['delta_ci95']['net_profit'], [20., 20.])
        self.assertEqual(result['comparisons']['teacher_task']['overall']['delta_ci95']['net_profit'], [10., 10.])
        self.assertEqual(result['complete_action_changed_games'], 32)
        self.assertFalse(result['default_replacement'])
        rows, cfg = panel(teacher=100.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['research_positive'])
        self.assertFalse(result['teacher_retention'])
        self.assertFalse(result['admission'])

    def test_effect_size_and_shared_market_margin_are_required(self):
        rows, cfg = panel(candidate=1., teacher=0.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['student_checks']['own_profit_ci_positive'])
        self.assertFalse(result['student_checks']['meaningful_own_profit_gain'])
        self.assertFalse(result['research_positive'])
        rows, cfg = panel()
        for row in rows:
            if row['policy'] == 'candidate':
                economic(row, 1020., 130.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['student_checks']['own_profit_ci_positive'])
        self.assertFalse(result['student_checks']['overall_margin_not_lower'])
        self.assertFalse(result['research_positive'])

    def test_market_regression_cannot_hide_behind_round2_gain(self):
        rows, cfg = panel(candidate=500.)
        for row in rows:
            if row['policy'] == 'candidate' and row['opponent'] == 'market':
                economic(row, 970.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['student_checks']['own_profit_ci_positive'])
        self.assertFalse(result['student_opponent_point_guards']['market'])
        self.assertFalse(result['research_positive'])

    def test_teacher_score_loss_blocks_retention_even_within_profit_tolerance(self):
        rows, cfg = panel(teacher=30.)
        for row in rows:
            if row['opponent'] == 'market' and row['policy'] in ('candidate', 'teacher_task'):
                economic(row, row['net_profit'], 1025.)
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['teacher_retention_primary_passed'])
        self.assertFalse(result['teacher_opponent_point_guards']['market'])
        self.assertFalse(result['teacher_retention'])

    def test_control_proof_requires_four_games_and_step_zero_counts(self):
        rows, cfg = panel()
        candidates = [r for r in rows if r['policy'] == 'candidate']
        for row in candidates[3:]:
            row['first_request_difference_step'] = None
        result = holdout_statistics(rows, cfg)
        self.assertTrue(result['research_positive'])
        self.assertTrue(result['teacher_retention'])
        self.assertEqual(result['complete_action_changed_games'], 3)
        self.assertFalse(result['admission'])
        candidates[3]['first_request_difference_step'] = 0
        self.assertTrue(holdout_statistics(rows, cfg)['admission'])

    def test_complete_seed_draws_and_ratio_denominators(self):
        nets = np.asarray([100., 200., 300., 400., 900., 1000., 1100., 1200.])
        effects = np.asarray([1., 5., -3., 10., 40., 80., 120., 160.])
        rows, cfg = panel(nets=nets)
        for row in rows:
            i = cfg['test_seeds'].index(row['seed'])
            op = cfg['opponents'].index(row['opponent'])
            base = nets[i]+20*op+10*row['side']
            change = effects[i]+5*op-3*row['side']
            economic(row, base+(change if row['policy'] == 'candidate' else 2. if row['policy'] == 'teacher_task' else 0.))
        result = holdout_statistics(rows, cfg)
        draws = np.random.Generator(np.random.PCG64(2026102706)).integers(0, 8, size=(20000, 8))
        for reference, offset in (('baseline', 0.), ('teacher_task', 2.)):
            sample_delta = (effects+1.-offset)[draws].mean(axis=1)
            denominator = (nets+15.+offset)[draws].mean(axis=1)
            expected_ci = np.quantile(sample_delta, [.025, .975], method='linear')
            expected_ratio_ci = np.quantile(sample_delta/denominator, [.025, .975], method='linear')
            actual = result['comparisons'][reference]['overall']
            np.testing.assert_array_equal(actual['delta_ci95']['net_profit'], expected_ci)
            np.testing.assert_array_equal(actual['relative_net_profit_ci95'], expected_ratio_ci)
            self.assertAlmostEqual(actual['relative_net_profit_mean'], float(np.mean(effects+1.-offset)/np.mean(nets+15.+offset)))
            self.assertNotAlmostEqual(actual['relative_net_profit_mean'], float(np.mean((effects+1.-offset)/(nets+15.+offset))))
            for op_index, op in enumerate(cfg['opponents']):
                subgroup = result['comparisons'][reference]['by_opponent'][op]
                op_delta = (effects+5*op_index-1.5-offset)[draws].mean(axis=1)
                np.testing.assert_array_equal(subgroup['delta_ci95']['net_profit'],
                                              np.quantile(op_delta, [.025, .975], method='linear'))

    def test_nonpositive_point_or_resampled_denominator_fails_without_nan(self):
        for nets in ([0.]*8, [-100.]+[100.]*7):
            with self.subTest(nets=nets):
                rows, cfg = panel(nets=nets, teacher=0.)
                result = holdout_statistics(rows, cfg)
                own = result['comparisons']['baseline']['overall']
                self.assertFalse(own['relative_net_profit_defined'])
                self.assertIsNone(own['relative_net_profit_ci95'])
                self.assertGreater(own['nonpositive_bootstrap_reference_count'], 0)
                self.assertFalse(result['admission'])
                json.dumps(result, allow_nan=False)

    def test_population_and_cash_errors_are_rejected(self):
        changes = [('seed', 999), ('side', True), ('policy', 'gated'), ('split', 'dev'),
                   ('states', 719), ('agent_errors', 1), ('passed', 1),
                   ('bank', float('nan')), ('net_profit', 999.), ('score', 0.),
                   ('first_request_difference_step', True), ('first_request_difference_step', 719)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                rows, cfg = panel()
                row = next(r for r in rows if r['policy'] == 'candidate')
                row[key] = value
                with self.assertRaises(ValueError):
                    holdout_statistics(rows, cfg)
        rows, cfg = panel()
        with self.assertRaises(ValueError):
            holdout_statistics(rows[:-1], cfg)
        rows[-1] = deepcopy(rows[0])
        with self.assertRaises(ValueError):
            holdout_statistics(rows, cfg)

    def test_cross_split_config_and_unequal_initial_cash_rejected(self):
        rows, cfg = panel()
        cfg['validation_seeds'][0] = cfg['train_seeds'][0]
        with self.assertRaises(ValueError):
            holdout_statistics(rows, cfg)
        rows, cfg = panel()
        row = rows[0]
        row['initial_bank'] += 1
        row['net_profit'] -= 1
        with self.assertRaises(ValueError):
            holdout_statistics(rows, cfg)


if __name__ == '__main__':
    unittest.main()
