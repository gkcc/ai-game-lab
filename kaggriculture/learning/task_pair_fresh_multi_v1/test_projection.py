"""Synthetic checks for paired, clustered and redacted exploration results."""
import copy
import json
import unittest

from . import projection


def fixture():
    roles = {'C01': {'S02': 101, 'S03': 202}, 'C02': {'S02': 303, 'S03': 404}}
    cohorts, prefixes = {}, {}
    for cohort, aliases in projection.COHORTS.items():
        rows, comparisons = [], []
        for alias in aliases:
            for opponent in projection.OPPONENTS:
                for side in projection.SIDES:
                    for arm in projection.ARMS:
                        own = 100 if arm == 'baseline' else 110
                        other = 90 if arm == 'baseline' else 92
                        def economy(amount):
                            return dict(cash_reconciled=True, initial_cash=100, final_cash=100 + amount,
                                        net_profit=amount, successful_plants=2, seed_units_bought=3,
                                        sales_units=4, sales_revenue=5, ending_seed_units=1,
                                        ending_warehouse_units=2)
                        result = dict(actor=side, official_turns=719,
                                      economy=dict(own=economy(own), opponent=economy(other)),
                                      margin=own - other, score=1.0)
                        rows.append(dict(schema='task-pair-dev-game-metrics-v1',
                            game=f'dev-{alias}-{opponent}-{side}-{arm}', seed_alias=alias,
                            seed=roles[cohort][alias], opponent=opponent, side=side, arm=arm,
                            task_checkpoint_sha256=('a' if arm == 'baseline' else 'b') * 64,
                            crop_checkpoint_sha256='c' * 64, trace_sha256='d' * 64,
                            replay_sha256='e' * 64, result=result))
                    comparisons.append(dict(seed_alias=alias, seed=roles[cohort][alias], opponent=opponent,
                        side=side, reference='baseline', candidate='control', result=dict(
                            comparison_complete=True, matched_calls=5, matched_multi_unit_opportunities=2,
                            stopped=True, stop_reason='choice_difference')))
                    prefixes[(cohort, alias, opponent, side)] = dict(scope='same_complete_history',
                        status='observed', reason='actor_request_difference', step=11)
        cohorts[cohort] = dict(rows=rows, comparisons=comparisons)
    return cohorts, roles, prefixes


class ProjectionTests(unittest.TestCase):
    def test_four_seed_roles_and_opponents_produce_correct_paired_aggregates(self):
        cohorts, roles, prefixes = fixture()
        result = projection.project(cohorts, seed_roles=roles, prefixes=prefixes)
        self.assertEqual((result['games'], result['paired_coordinates'], result['seed_clusters']), (32, 16, 4))
        self.assertEqual(result['overall']['mean_delta']['own_net_profit'], 10)
        self.assertEqual(result['overall']['mean_delta']['opponent_net_profit'], 2)
        self.assertEqual(result['overall']['mean_delta']['margin'], 8)
        self.assertTrue(all(value['pairs'] == 4 for value in result['by_seed'].values()))
        self.assertTrue(all(value['pairs'] == 8 for value in result['by_opponent'].values()))
        self.assertEqual(result['overall']['outcomes']['control'], {'wins': 16, 'draws': 0, 'losses': 0})
        self.assertTrue(result['exploratory_screen']['passed'])
        self.assertFalse(result['independent_improvement'])
        self.assertFalse(result['promotion'])
        encoded = json.dumps(result)
        self.assertNotIn('task_checkpoint_sha256', encoded)
        self.assertNotIn('replay_sha256', encoded)
        self.assertNotIn('"seed":', encoded)

    def test_rejects_seed_role_duplicate_game_or_changed_opponent(self):
        for change in ('seed', 'duplicate', 'opponent'):
            cohorts, roles, prefixes = fixture()
            if change == 'seed':
                cohorts['C01']['rows'][0]['seed'] = 999
            elif change == 'duplicate':
                cohorts['C01']['rows'][1] = copy.deepcopy(cohorts['C01']['rows'][0])
            else:
                cohorts['C01']['comparisons'][0]['opponent'] = 'other'
            with self.subTest(change=change), self.assertRaises(ValueError):
                projection.project(cohorts, seed_roles=roles, prefixes=prefixes)

    def test_missing_prefix_and_model_identity_change_cannot_be_silently_aggregated(self):
        cohorts, roles, prefixes = fixture()
        prefixes.pop(next(iter(prefixes)))
        with self.assertRaises(ValueError):
            projection.project(cohorts, seed_roles=roles, prefixes=prefixes)
        cohorts, roles, prefixes = fixture()
        cohorts['C02']['rows'][0]['crop_checkpoint_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            projection.project(cohorts, seed_roles=roles, prefixes=prefixes)

    def test_heterogeneous_seed_opponent_seat_and_outcomes_are_grouped_correctly(self):
        cohorts, roles, prefixes = fixture()
        order = {'F01': 0, 'F02': 1, 'F03': 2, 'F04': 3}
        for cohort, payload in cohorts.items():
            for row in payload['rows']:
                if row['arm'] != 'control':
                    continue
                seed_index = order[projection.COHORTS[cohort][row['seed_alias']]]
                delta = -70 + 100 * seed_index + (10 if row['opponent'] == 'round2' else 0) + row['side']
                own, opponent = 100 + delta, 90
                economy = row['result']['economy']
                economy['own']['net_profit'], economy['own']['final_cash'] = own, 100 + own
                economy['opponent']['net_profit'], economy['opponent']['final_cash'] = opponent, 100 + opponent
                row['result']['margin'] = own - opponent
                row['result']['score'] = 1.0 if own > opponent else 0.5 if own == opponent else 0.0
        result = projection.project(cohorts, seed_roles=roles, prefixes=prefixes)
        self.assertEqual([result['by_seed'][role]['mean_delta']['own_net_profit']
                          for role in ('F01', 'F02', 'F03', 'F04')], [-64.5, 35.5, 135.5, 235.5])
        self.assertEqual([result['by_opponent'][op]['mean_delta']['own_net_profit']
                          for op in ('market', 'round2')], [80.5, 90.5])
        self.assertEqual([result['by_side'][str(side)]['mean_delta']['own_net_profit']
                          for side in (0, 1)], [85.0, 86.0])
        self.assertEqual(result['overall']['outcomes']['control'], {'wins': 12, 'draws': 0, 'losses': 4})
        self.assertFalse(result['exploratory_screen']['passed'])
        self.assertFalse(result['exploratory_screen']['per_seed_profit_guard']['F01'])

    def test_a_draw_is_counted_as_half_score_and_one_draw(self):
        cohorts, roles, prefixes = fixture()
        row = next(row for row in cohorts['C01']['rows'] if row['arm'] == 'control')
        economy = row['result']['economy']
        economy['own']['net_profit'], economy['own']['final_cash'] = 92, 192
        row['result']['margin'], row['result']['score'] = 0, 0.5
        result = projection.project(cohorts, seed_roles=roles, prefixes=prefixes)
        self.assertEqual(result['overall']['outcomes']['control'], {'wins': 15, 'draws': 1, 'losses': 0})
        self.assertEqual(result['overall']['mean_delta']['score'], -0.5 / 16)

    def test_nonpositive_baseline_does_not_fabricate_a_percentage_guard(self):
        cohorts, roles, prefixes = fixture()
        for row in cohorts['C01']['rows']:
            if row['seed_alias'] == 'S02' and row['arm'] == 'baseline':
                own = row['result']['economy']['own']
                own['net_profit'], own['final_cash'] = 0, own['initial_cash']
                row['result']['margin'], row['result']['score'] = -90, 0.0
        result = projection.project(cohorts, seed_roles=roles, prefixes=prefixes)
        self.assertEqual(result['exploratory_screen']['per_seed_profit_guard']['F01'], 'unknown')
        self.assertFalse(result['exploratory_screen']['passed'])
        self.assertEqual(result['exploratory_screen']['status'], 'unknown')

    def test_unknown_prefix_does_not_become_no_difference_or_a_pass(self):
        cohorts, roles, prefixes = fixture()
        for key in prefixes:
            if key[:2] == ('C01', 'S02'):
                prefixes[key] = dict(scope='same_complete_history', status='unknown',
                                     reason='frame_difference', step=None)
        result = projection.project(cohorts, seed_roles=roles, prefixes=prefixes)
        self.assertEqual(result['exploratory_screen']['per_seed_request_difference']['F01'], 'unknown')
        self.assertEqual(result['exploratory_screen']['status'], 'unknown')
        self.assertFalse(result['exploratory_screen']['passed'])
        for key in prefixes:
            if key[:2] == ('C01', 'S03'):
                prefixes[key] = dict(scope='same_complete_history', status='no_difference',
                                     reason='all_requests_equal', step=None)
        result = projection.project(cohorts, seed_roles=roles, prefixes=prefixes)
        self.assertIs(result['exploratory_screen']['per_seed_request_difference']['F02'], False)
        self.assertEqual(result['exploratory_screen']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
