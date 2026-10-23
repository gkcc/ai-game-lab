"""Focused regressions: official ingress, malformed requests and paired panels.

No full matches, filesystem artifacts, training, or historical evidence mutation.
Twelve real official single-step cases complement synthetic paired-panel checks.
"""
from copy import deepcopy
import json
import unittest

from .analysis import EconomicConsumer, FAMILIES, METRICS, OPPONENTS, summarize


def fixture():
    cfg = dict(development_seeds=[101, 102, 103, 104], opponents=list(OPPONENTS), seats=[0, 1],
               model_sha256={'original_student': 'a'*64, 'fixed_epoch6': 'b'*64})
    rows = []
    for family in FAMILIES:
        for seed in cfg['development_seeds']:
            for opponent in OPPONENTS:
                for side in cfg['seats']:
                    candidate = family == 'fixed_epoch6'
                    r = {m: 0 for m in METRICS}
                    r.update(game=f'{seed}-{opponent}-{side}', stage='development',
                             model_family=family, seed=seed, opponent=opponent, side=side,
                             policy='trained', passed=True, net_profit=110 if candidate else 100,
                             opponent_net_profit=90, margin=20 if candidate else 10, match_score=1.,
                             checkpoint_sha256=cfg['model_sha256'][family],
                             economic_audit={'passed': True},
                             control_audit={'audit': {'passed': True}, 'checkpoint_sha256': cfg['model_sha256'][family]},
                             economic={'own': {'cash_reconciled': True},
                                       'opponent': {'cash_reconciled': True}})
                    rows.append(r)
    return rows, cfg


class AnalysisRegression(unittest.TestCase):
    def test_official_ingress_tracer_to_consumer_malformed_requests(self):
        from decision.verify_mechanisms import env, obs, PASS, rules
        from decision.execution import ExecutionTrace
        from decision.semantics import canonical
        for side in (0, 1):
            for phase in ('unit', 'market'):
                for raw in (None, [], 17):
                    with self.subTest(side=side, phase=phase, raw=raw):
                        plain, traced = env(), env()
                        initial = [deepcopy(obs(traced, p)) for p in (0, 1)]
                        joint = [deepcopy(PASS), deepcopy(PASS)]
                        joint[side]['farmer' if phase == 'unit' else 'market'] = (
                            deepcopy(raw) if phase == 'unit' else [deepcopy(raw)])
                        untouched = deepcopy(joint)
                        consumer = EconomicConsumer()
                        events = []
                        def sink(event):
                            before = deepcopy(event)
                            consumer.consume(event)
                            self.assertEqual(event, before)
                            events.append(before)
                        plain.step(deepcopy(joint))
                        with ExecutionTrace(rules, sink).installed(traced):
                            traced.step(deepcopy(joint))
                        self.assertEqual(joint, untouched)
                        self.assertEqual(canonical(plain.state), canonical(traced.state))
                        relevant = [e for e in events if e['player'] == side and e['phase'] == phase]
                        self.assertEqual(len(relevant), 1)
                        self.assertEqual(relevant[0]['requested'], raw)
                        self.assertEqual(relevant[0]['status'], 'no_effect')
                        final = [deepcopy(obs(traced, p)) for p in (0, 1)]
                        economics = consumer.finish(initial, final)
                        self.assertTrue(all(p['cash_reconciled'] for p in economics))
                        self.assertTrue(all(p['net_profit'] == p['sales_units'] == p['seed_units_bought'] == 0
                                            for p in economics))

    def test_malformed_requests_in_both_event_branches(self):
        consumer = EconomicConsumer()
        malformed = [None, [], 17, 'SELL', {}, [None], [['SELL']]]
        for player in (0, 1):
            for raw in malformed:
                consumer.consume(dict(player=player, phase='unit', status='blocked',
                                      requested=raw, effective=['PASS'],
                                      before={'money': 3000}, after={'money': 3000}))
                consumer.consume(dict(player=player, phase='market', status='no_effect',
                                      requested=raw, executed_units=0, cash_delta=0))
            # Orders above the official limit have no fill fields.
            consumer.consume(dict(player=player, phase='market', status='ignored', requested=None))
            c = consumer.players[player]['counts']
            self.assertEqual(c['unit_events'], 7)
            self.assertEqual(c['market_events'], 8)
            self.assertEqual(c['unit_unclassified_requests'], 7)
            self.assertEqual(c['market_unclassified_requests'], 8)
            self.assertEqual(c['successful_plants'], 0)
            self.assertEqual(c['sales_units'], 0)

    def test_actual_effects_partial_fills_and_cash_closure(self):
        consumer = EconomicConsumer()
        consumer.consume(dict(player=0, phase='unit', status='executed',
                              requested=['PLANT', 'MELON'], effective=['PLANT', 'MELON'],
                              successful_plant=True, before={'money': 3000}, after={'money': 3000}))
        consumer.consume(dict(player=0, phase='market', status='partial',
                              requested=['SELL', 'MELON', 10], executed_units=2, cash_delta=101))
        consumer.consume(dict(player=1, phase='market', status='executed',
                              requested=['BUY_SEED', 'WHEAT', 5], executed_units=3, cash_delta=-30))
        consumer.consume(dict(player=0, phase='market', status='executed',
                              requested=['HIRE'], executed_units=1, cash_delta=-20))
        def observations(money):
            return [{'farms': [{'money': money[0]}, {'money': money[1]}],
                     'private': {'seeds': {'WHEAT': 3 if player else 0}, 'shed': {'MELON': 4}}}
                    for player in (0, 1)]
        initial, final = observations([3000, 3000]), observations([3081, 2970])
        own, rival = consumer.finish(initial, final)
        self.assertEqual(own['successful_plants'], 1)
        self.assertEqual(own['sales_units'], 2)
        self.assertEqual(own['sales_revenue'], 101)
        self.assertEqual(own['average_sale_price_by_item']['MELON'], 50.5)
        self.assertEqual(own['net_profit'], 81)
        self.assertEqual(rival['seed_units_bought'], 3)
        self.assertEqual(rival['ending_seed_units'], 3)
        self.assertEqual(own['ending_warehouse_units'], 4)
        final[0]['farms'][0]['money'] += 1
        with self.assertRaisesRegex(ValueError, 'reconcile'):
            consumer.finish(initial, final)

    def test_unclassified_nonzero_effect_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unclassified market effect'):
            EconomicConsumer().consume(dict(player=0, phase='market', status='executed',
                                            requested=None, executed_units=1, cash_delta=10))

    def test_exact_cartesian_pairs_and_seed_gate(self):
        rows, cfg = fixture()
        report = summarize(list(reversed(rows)), cfg)
        self.assertTrue(report['exploration_gate_passed'])
        self.assertEqual(report['whole_seed_clusters'], 4)
        self.assertEqual(len(report['paired']), 24)
        self.assertEqual(report['fixed_epoch6_minus_original_student']['net_profit'], 10)
        self.assertEqual(report['by_model']['fixed_epoch6']['wins'], 24)
        self.assertFalse(report['promotion'])
        self.assertFalse(report['independent_significance_claim'])
        for r in rows:
            if r['model_family'] == 'fixed_epoch6' and r['seed'] == 101:
                r['net_profit'] = 99
                r['margin'] = 9
        self.assertFalse(summarize(rows, cfg)['exploration_gate_passed'])

    def test_missing_duplicate_and_unknown_coordinate_rejected(self):
        rows, cfg = fixture()
        with self.assertRaises(ValueError):
            summarize(rows[:-1], cfg)
        duplicate = deepcopy(rows)
        duplicate[-1] = deepcopy(duplicate[-2])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize(duplicate, cfg)
        unknown = deepcopy(rows)
        unknown[0]['seed'] = 999
        with self.assertRaisesRegex(ValueError, 'unexpected panel'):
            summarize(unknown, cfg)

    def test_entry_must_be_separately_audited_not_silently_filtered(self):
        rows, cfg = fixture()
        entry = deepcopy(rows[0])
        entry.update(stage='entry', game='entry-0')
        with self.assertRaises(ValueError):
            summarize(rows + [entry], cfg)
        rows[0] = entry
        with self.assertRaisesRegex(ValueError, 'Entry/nondevelopment'):
            summarize(rows, cfg)

    def test_unverified_and_nonfinite_metrics_rejected(self):
        rows, cfg = fixture()
        rows[0]['economic_audit']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'official audits'):
            summarize(rows, cfg)
        rows, cfg = fixture()
        rows[0]['sales_revenue'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid numeric'):
            summarize(rows, cfg)

    def test_mislabeled_checkpoint_rejected(self):
        rows, cfg = fixture()
        rows[0]['checkpoint_sha256'] = cfg['model_sha256']['fixed_epoch6']
        with self.assertRaisesRegex(ValueError, 'model identity'):
            summarize(rows, cfg)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisRegression))
    print(json.dumps(dict(passed=result.wasSuccessful(), checks=result.testsRun,
                          synthetic_only=False, official_single_step_cases=12,
                          official_transitions_including_untraced_controls=24,
                          new_full_games=0, training_runs=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
