"""Synthetic regressions for whole-seed holdout inference and source admission.

No real match, training, checkpoint, or artifact is created or changed.
"""
from copy import deepcopy
import json
import unittest

from learning.bc_crop_economic_diagnostic.analysis import FAMILIES, METRICS, OPPONENTS
from .statistics import summarize


def fixture(seed_deltas=None, *, baseline_profit=1000., opponent_deltas=None, side_offsets=False):
    seeds = list(range(101, 109))  # synthetic identifiers; never used to play games
    cfg = dict(test_seeds=seeds, opponents=list(OPPONENTS), seats=[0, 1], engine_sha256='c'*64,
               model_sha256={FAMILIES[0]: 'a'*64, FAMILIES[1]: 'b'*64},
               bootstrap=dict(draws=20000, seed=2026102326, unit='whole seed',
                              method='percentile', confidence=.95, quantile_method='linear'))
    seed_deltas = seed_deltas if seed_deltas is not None else [20.]*8
    opponent_deltas = opponent_deltas or dict.fromkeys(OPPONENTS, 0.)
    rows = []
    for family in FAMILIES:
        for i, seed in enumerate(seeds):
            for opponent in OPPONENTS:
                for side in (0, 1):
                    change = seed_deltas[i] + opponent_deltas[opponent] + ((-7 if side == 0 else 7) if side_offsets else 0)
                    profit = baseline_profit + (change if family == FAMILIES[1] else 0)
                    rival_profit = baseline_profit - 100
                    def economics(p):
                        return dict(initial_cash=3000., final_cash=3000.+p, net_profit=p,
                                    market_cash_delta=p, unit_cash_delta=0., cash_reconciled=True,
                                    successful_plants=10, seed_units_bought=10, sales_units=40,
                                    sales_revenue=2000., ending_seed_units=0, ending_warehouse_units=0)
                    own, rival = economics(profit), economics(rival_profit)
                    rewards = [own['final_cash'], rival['final_cash']] if side == 0 else [rival['final_cash'], own['final_cash']]
                    game = f'holdout-{seed}-{opponent}-{side}-trained'
                    cert = dict(passed=True, official_transitions_reexecuted=719,
                                both_public_and_private_states_match_each_step=True,
                                inventory_order_checked_each_step=True, final_rewards=rewards,
                                engine_sha256=cfg['engine_sha256'])
                    control = dict(game=game, seed=seed, side=side, opponent=opponent, policy='trained',
                                   checkpoint_sha256=cfg['model_sha256'][family], audit=deepcopy(cert),
                                   reproduced_actor_actions=719, reproduced_rival_actions=719,
                                   raw_decision_traces_reproduced=True,
                                   fallbacks=0, load_errors=0, inference_errors=0)
                    margin = own['final_cash']-rival['final_cash']
                    row = dict(game=game, seed=seed, side=side, opponent=opponent, policy='trained',
                               stage='holdout', model_family=family, passed=True,
                               checkpoint_sha256=cfg['model_sha256'][family], control_audit=control,
                               economic_audit=cert, official_replay_passes=2, official_transitions_reexecuted=1438,
                               economic=dict(own=own, opponent=rival), net_profit=profit,
                               opponent_net_profit=rival_profit, margin=margin,
                               match_score=1. if margin > 0 else .5 if margin == 0 else 0.)
                    for metric in METRICS[4:10]:
                        row[metric], row['opponent_'+metric] = own[metric], rival[metric]
                    rows.append(row)
    return rows, cfg


class StatisticsRegression(unittest.TestCase):
    def test_heterogeneous_seeds_resample_six_pairs_together(self):
        # Seven seed means=2 and one=82. A draw is 2+10*K with K~Binomial(8,1/8),
        # giving the registered deterministic percentile endpoints 2 and 32.
        # Opponent/seat perturbations cancel INSIDE each seed, not across draws.
        rows, cfg = fixture([2.]*7+[82.], opponent_deltas=dict(v1=-30., market=0., round2=30.), side_offsets=True)
        report = summarize(rows, cfg)
        self.assertEqual(report['whole_seed_clusters'], 8)
        self.assertEqual(report['holdout_games'], 96)
        self.assertEqual(report['paired_games'], 48)
        self.assertEqual(report['bootstrap']['pairs_per_seed'], 6)
        self.assertEqual(report['fixed_epoch6_minus_original_student']['net_profit'], 12.)
        self.assertEqual(report['net_profit_ci95'], dict(lower=2., upper=32.))
        self.assertTrue(report['learned_student_gate_passed'])
        loso = {r['excluded_seed']: r for r in report['leave_one_seed_out_descriptive']}
        self.assertEqual(loso[108]['delta']['net_profit'], 2.)
        self.assertAlmostEqual(loso[101]['delta']['net_profit'], 94/7)
        self.assertTrue(all(r['remaining_seed_clusters'] == 7 for r in loso.values()))

    def test_reproducible_order_independent_and_nonmutating(self):
        rows, cfg = fixture([2.]*7+[82.])
        before = deepcopy((rows, cfg))
        first = summarize(rows, cfg)
        second = summarize(list(reversed(rows)), cfg)
        self.assertEqual(first, second)
        self.assertEqual((rows, cfg), before)
        self.assertEqual(first['by_model'][FAMILIES[1]]['wins'], 48)
        self.assertFalse(first['default_changed'])
        self.assertFalse(first['automatic_deployment'])

    def test_strict_lower_ci_and_inclusive_one_percent_threshold(self):
        rows, cfg = fixture([0.]*7+[80.])
        report = summarize(rows, cfg)
        self.assertEqual(report['net_profit_ci95']['lower'], 0.)
        self.assertFalse(report['gate_checks']['net_profit_ci95_lower_positive'])
        self.assertFalse(report['learned_student_gate_passed'])
        rows, cfg = fixture([10.]*8)
        report = summarize(rows, cfg)
        self.assertEqual(report['practical_increment_threshold'], 10.)
        self.assertTrue(report['gate_checks']['net_profit_at_least_one_percent'])
        self.assertTrue(report['learned_student_gate_passed'])
        rows, cfg = fixture([9.5]*8)
        report = summarize(rows, cfg)
        self.assertTrue(report['gate_checks']['net_profit_ci95_lower_positive'])
        self.assertFalse(report['gate_checks']['net_profit_at_least_one_percent'])
        self.assertFalse(report['learned_student_gate_passed'])

    def test_market_regression_cannot_be_hidden_by_other_opponents(self):
        rows, cfg = fixture([0.]*8, opponent_deltas=dict(v1=30., market=-1., round2=30.))
        report = summarize(rows, cfg)
        self.assertTrue(report['gate_checks']['net_profit_ci95_lower_positive'])
        self.assertTrue(report['gate_checks']['net_profit_at_least_one_percent'])
        self.assertFalse(report['gate_checks']['market_profit_nonnegative'])
        self.assertFalse(report['gate_checks']['market_margin_nonnegative'])
        self.assertFalse(report['learned_student_gate_passed'])

    def test_missing_duplicate_and_extra_entry_fail_closed(self):
        rows, cfg = fixture()
        with self.assertRaisesRegex(ValueError, '96'):
            summarize(rows[:-1], cfg)
        duplicate = deepcopy(rows)
        duplicate[-1] = deepcopy(duplicate[-2])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize(duplicate, cfg)
        extra = deepcopy(rows[0]); extra.update(stage='entry', game='entry-test-0')
        with self.assertRaisesRegex(ValueError, '96'):
            summarize(rows+[extra], cfg)
        rows[0] = extra
        with self.assertRaisesRegex(ValueError, 'entry'):
            summarize(rows, cfg)

    def test_role_checkpoint_source_and_unreconciled_cash_rejected(self):
        mutations = (
            lambda r: r.update(stage='development'),
            lambda r: r.update(policy='teacher'),
            lambda r: r.update(model_family='teacher'),
            lambda r: r.update(checkpoint_sha256='d'*64),
            lambda r: r['control_audit'].update(seed=999),
            lambda r: r['control_audit'].update(raw_decision_traces_reproduced=False),
            lambda r: r['economic_audit'].update(passed=False),
            lambda r: r['economic']['own'].update(cash_reconciled=False),
            lambda r: r['economic']['opponent'].update(market_cash_delta=999),
            lambda r: r.update(sales_units=41),
            lambda r: r.update(net_profit=float('nan')),
        )
        for mutation in mutations:
            rows, cfg = fixture()
            mutation(rows[0])
            with self.assertRaises(ValueError):
                summarize(rows, cfg)

    def test_registered_bootstrap_required(self):
        rows, cfg = fixture()
        cfg['bootstrap']['confidence'] = .9
        with self.assertRaisesRegex(ValueError, 'bootstrap configuration'):
            summarize(rows, cfg)

    def test_nonpositive_reference_is_valid_failed_result(self):
        for baseline_profit in (0., -100.):
            with self.subTest(baseline_profit=baseline_profit):
                rows, cfg = fixture(baseline_profit=baseline_profit)
                report = summarize(rows, cfg)
                self.assertFalse(report['gate_checks']['original_student_mean_profit_positive'])
                self.assertFalse(report['gate_checks']['net_profit_at_least_one_percent'])
                self.assertFalse(report['learned_student_gate_passed'])
                self.assertIsNone(report['relative_net_profit_increment'])
                self.assertIsNone(report['practical_increment_threshold'])
                self.assertEqual(report['net_profit_ci95'], dict(lower=20., upper=20.))
                self.assertEqual(len(report['paired']), 48)
                self.assertEqual(len(report['by_seed']), 8)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StatisticsRegression))
    print(json.dumps(dict(passed=result.wasSuccessful(), checks=result.testsRun,
                          synthetic_only=True, new_games=0, training_runs=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
