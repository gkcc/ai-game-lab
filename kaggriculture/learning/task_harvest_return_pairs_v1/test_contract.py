"""Pure terminal accounting and public-projection regressions."""
from copy import deepcopy
import json
import unittest

from learning.task_harvest_return_pairs_v1 import contract


def endpoint(own, opponent):
    margin = own - opponent
    return dict(own_net_profit=own, opponent_net_profit=opponent, margin=margin,
                score=1. if margin > 0 else .5 if margin == 0 else 0.)


def paired_rows():
    return [dict(source, status='paired', technical_passed=True,
                 H=endpoint(300, 200), W=endpoint(200, 200)) for source in contract.sources()]


class ContractTests(unittest.TestCase):
    def test_terminal_delta_keeps_each_arm_cash_and_game_score(self):
        h, w = endpoint(300, 200), endpoint(200, 200)
        result = contract.terminal_delta(h, w)
        self.assertEqual(result['H'], h)
        self.assertEqual(result['W'], w)
        self.assertEqual(result['H_minus_W'], dict(own_net_profit=100, opponent_net_profit=0,
                                                 margin=100, score=.5))
        self.assertEqual((result['margin_sign'], result['own_profit_sign']), (1, 1))
        self.assertFalse(result['objective_conflict'])
        result['H']['own_net_profit'] = -1
        self.assertEqual(h['own_net_profit'], 300)

    def test_objective_conflict_is_preserved_instead_of_combining_targets(self):
        h, w = endpoint(300, 250), endpoint(200, 100)
        forward = contract.terminal_delta(h, w)
        reverse = contract.terminal_delta(w, h)
        self.assertEqual(forward['H_minus_W']['own_net_profit'], 100)
        self.assertEqual(forward['H_minus_W']['margin'], -50)
        self.assertTrue(forward['objective_conflict'])
        self.assertTrue(reverse['objective_conflict'])
        self.assertEqual((forward['margin_sign'], forward['own_profit_sign']), (-1, 1))
        self.assertEqual((reverse['margin_sign'], reverse['own_profit_sign']), (1, -1))
        tie = contract.terminal_delta(endpoint(200, 100), endpoint(100, 0))
        self.assertEqual(tie['margin_sign'], 0)
        self.assertFalse(tie['objective_conflict'])

    def test_shared_initial_cash_offset_is_legal_but_different_offsets_are_not(self):
        # margin is final cash difference; net_profit subtracts each player's
        # own initial cash. A common initial difference cancels across H/W.
        h = dict(own_net_profit=100, opponent_net_profit=60, margin=45, score=1.)
        w = dict(own_net_profit=90, opponent_net_profit=70, margin=25, score=1.)
        result = contract.terminal_delta(h, w)
        self.assertEqual(result['H_minus_W'], dict(own_net_profit=10, opponent_net_profit=-10,
                                                 margin=20, score=0.))
        with self.assertRaises(ValueError):
            contract.terminal_delta(h, dict(w, margin=26))

    def test_bad_score_nonfinite_boolean_and_extra_fields_are_rejected(self):
        for key, value in (('score', .5), ('own_net_profit', True),
                           ('own_net_profit', float('nan')), ('margin', float('inf')),
                           ('private_payload', 'must not pass')):
            with self.subTest(field=key, value_type=type(value).__name__):
                bad = dict(endpoint(300, 200), **{key: value})
                with self.assertRaises(ValueError):
                    contract.terminal_delta(bad, endpoint(200, 200))

    def test_public_projection_is_an_explicit_cash_only_whitelist(self):
        rows = paired_rows()
        for row in rows:
            row.update(seed='PRIVATE_SEED_MARKER', input_sha256='PRIVATE_INPUT_MARKER',
                       private_path='PRIVATE_PATH_MARKER', inventories={'PRIVATE_ITEM_MARKER': 99},
                       target={'unit': 'PRIVATE_UNIT_MARKER', 'coordinates': 'PRIVATE_COORD_MARKER'})
        saved = deepcopy(rows)
        result = contract.projection(rows)
        self.assertEqual(result['completed_pairs'], 4)
        self.assertTrue(result['four_pair_data_chain_passed'])
        for item in result['cases']:
            self.assertEqual(set(item), {'role', 'cohort', 'opponent', 'side', 'status', 'paired', 'returns'})
            for arm in ('H', 'W', 'H_minus_W'):
                self.assertEqual(set(item['returns'][arm]), {'own_net_profit', 'opponent_net_profit', 'margin', 'score'})
        self.assertNotIn('PRIVATE_', json.dumps(result))
        for key in ('training', 'gpu_training', 'new_seeds', 'independent_holdout', 'promotion', 'statistical_significance'):
            self.assertIs(result[key], False)
        self.assertEqual(rows, saved)

    def test_missing_case_is_reported_and_failed_or_reordered_case_is_not_certified(self):
        rows = paired_rows()
        rows[1].update(status='no_eligible_target', H={'private_future': 'PRIVATE_MISSING_MARKER'})
        result = contract.projection(rows)
        self.assertEqual(result['completed_pairs'], 3)
        self.assertFalse(result['four_pair_data_chain_passed'])
        self.assertIsNone(result['cases'][1]['returns'])
        self.assertFalse(result['cases'][1]['paired'])
        self.assertNotIn('PRIVATE_MISSING_MARKER', json.dumps(result))
        for defect in ('failed', 'typed_pass', 'order', 'short', 'typed_side'):
            with self.subTest(defect=defect):
                broken = paired_rows()
                if defect == 'failed':
                    broken[0]['technical_passed'] = False
                elif defect == 'typed_pass':
                    broken[0]['technical_passed'] = 1
                elif defect == 'order':
                    broken[0], broken[1] = broken[1], broken[0]
                elif defect == 'short':
                    broken.pop()
                else:
                    broken[0]['side'] = False
                with self.assertRaises(ValueError):
                    contract.projection(broken)


if __name__ == '__main__':
    unittest.main()
