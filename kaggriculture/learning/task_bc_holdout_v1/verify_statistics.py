"""Synthetic regressions for the frozen panel estimand, cluster and gates."""
from copy import deepcopy
import json
import unittest

from .statistics import analyze, BOOTSTRAP, GATES


def fixture(references=None, differences=None):
    refs = references or [1000]*8
    deltas = differences or [100]*8
    cfg = dict(test_seeds=list(range(101, 109)), opponents=['v1', 'market', 'round2'],
               bootstrap=dict(BOOTSTRAP), gates=dict(GATES), arms=['trained', 'teacher_task'])
    rows = []
    for i, seed in enumerate(cfg['test_seeds']):
        for opponent in cfg['opponents']:
            for side in (0, 1):
                for arm in cfg['arms']:
                    net = refs[i] + (deltas[i] if arm == 'trained' else 0)
                    bank, rival = 3000+net, 3000
                    rows.append(dict(seed=seed, opponent=opponent, side=side, policy=arm,
                        bank=bank, opponent_bank=rival, initial_bank=3000, opponent_initial_bank=3000,
                        net_profit=net, opponent_net_profit=0, margin=bank-rival,
                        score=1 if bank>rival else .5 if bank==rival else 0,
                        passed=True, status=['DONE', 'DONE'], states=720, action_turns=719, agent_errors=0))
    return rows, cfg


class StatisticsChecks(unittest.TestCase):
    def test_constant_positive_and_negative_gates(self):
        rows, cfg = fixture()
        result = analyze(rows, cfg)
        self.assertTrue(result['passed']); self.assertTrue(result['panel_improvement'])
        self.assertEqual(result['overall']['relative_net_profit_ci95'], [.1, .1])
        self.assertEqual(result['overall']['delta_ci95']['net_profit'], [100., 100.])
        self.assertEqual(result['overall']['arms']['trained']['wins'], 48)
        rows, cfg = fixture(differences=[-30]*8)
        result = analyze(rows, cfg)
        self.assertFalse(result['passed']); self.assertFalse(result['panel_improvement'])
        self.assertEqual(result['overall']['relative_net_profit_ci95'], [-.03, -.03])

    def test_whole_seed_not_individual_game_bootstrap(self):
        # Only one independent seed has +100, all of its six pairs move together.
        # Binomial(8, 1/8) central 95% counts are 0 and 3: [0, 37.5].
        rows, cfg = fixture(differences=[100, 0, 0, 0, 0, 0, 0, 0])
        result = analyze(rows, cfg)
        self.assertEqual(result['overall']['delta_ci95']['net_profit'], [0., 37.5])
        self.assertEqual(result['clusters'], 8)
        self.assertEqual(result['overall']['pairs'], 48)
        # Replication across opponents does not create extra independent draws.
        for subgroup in result['by_opponent'].values():
            self.assertEqual(subgroup['delta_ci95']['net_profit'], [0., 37.5])

    def test_ratio_resamples_reference_denominator(self):
        # Every seed has exactly +10% despite an eightfold reference variation.
        # Recomputing the denominator keeps every draw exactly +10%; fixing it
        # to the panel mean would yield a spurious interval of positive width.
        rows, cfg = fixture(references=[100*i for i in range(1, 9)],
                            differences=[10*i for i in range(1, 9)])
        result = analyze(rows, cfg)
        self.assertEqual(result['overall']['relative_net_profit_ci95'], [.1, .1])
        self.assertEqual(result['overall']['delta_means']['net_profit'], 45.)
        self.assertEqual(result['overall']['reference_net_profit_mean'], 450.)

    def test_duplicate_missing_and_invalid_outcomes_rejected(self):
        rows, cfg = fixture()
        variants = []
        duplicate = deepcopy(rows); duplicate[1] = deepcopy(duplicate[0]); variants.append(duplicate)
        variants.append(rows[:-1])
        for key, value in [('score', 0), ('net_profit', -1), ('opponent_net_profit', 1),
                           ('margin', -100), ('side', True), ('passed', False),
                           ('status', ['ACTIVE', 'DONE']), ('states', 719), ('action_turns', 718),
                           ('actions', 718), ('agent_errors', 1), ('errors', 1),
                           ('bank', float('nan'))]:
            changed = deepcopy(rows); changed[0][key] = value; variants.append(changed)
        for changed in variants:
            with self.subTest(row=changed[0]), self.assertRaises(ValueError):
                analyze(changed, cfg)

    def test_nonpositive_reference_no_nan_or_admission(self):
        for refs in ([0]*8, [-1]*8, [-1000, 10, 10, 10, 10, 10, 10, 2000]):
            rows, cfg = fixture(references=refs)
            result = analyze(rows, cfg)
            self.assertFalse(result['passed'])
            self.assertFalse(result['overall']['relative_net_profit_defined'])
            self.assertIsNone(result['overall']['relative_net_profit_ci95'])
            self.assertIsNone(result['overall']['relative_net_profit_mean'])
            json.dumps(result, allow_nan=False)

    def test_opponent_point_guard_cannot_be_hidden_by_other_gains(self):
        rows, cfg = fixture(differences=[300]*8)
        for row in rows:
            if row['opponent'] == 'market' and row['policy'] == 'trained':
                row.update(bank=3940, net_profit=940, margin=940)
        result = analyze(rows, cfg)
        self.assertTrue(result['primary_gate_passed'])
        self.assertFalse(result['opponent_point_guards']['market'])
        self.assertFalse(result['passed'])

    def test_score_guard_and_strict_primary_boundary(self):
        rows, cfg = fixture(differences=[-20]*8)
        self.assertFalse(analyze(rows, cfg)['passed'])  # -2% lower must be strictly exceeded.
        rows, cfg = fixture()
        for row in rows:
            if row['opponent'] == 'market' and row['policy'] == 'trained':
                row.update(opponent_bank=5000, opponent_net_profit=2000, margin=-900, score=0)
        result = analyze(rows, cfg)
        self.assertTrue(result['primary_gate_passed'])
        self.assertFalse(result['opponent_point_guards']['market'])
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
