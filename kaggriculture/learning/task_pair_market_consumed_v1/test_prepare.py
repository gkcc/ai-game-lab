"""Synthetic seed selection and reserved-role regression tests."""
import unittest

from .prepare import reserved_values, select, verify_consumed


class SeedSelectionTests(unittest.TestCase):
    def test_explicit_and_nested_reserved_roles_exclude_seed_values(self):
        value = {
            'test_seeds': [101, '102'],
            'seed_roles': {'holdout': {'S01': 103, 'S02': '104'}},
            'reserved': {'seeds': [105.0, 106]},
            'latest_seed': 107,
            'training_seeds': [108],
            'holdout': {'excluded_seeds': [109], 'seed_count': 20},
            'test_seed_number': 110,
        }
        self.assertEqual(reserved_values(value), {101, 102, 103, 104, 105, 106, 110})

    def test_by_seed_numeric_keys_do_not_parse_metric_values(self):
        value = {'holdout': {'comparisons': {'by_seed': {
            '201': {'profit': 0.75, 'some_metric': 44},
            '202': {'score': 0.5},
            'S03': '203',
        }}}}
        self.assertEqual(reserved_values(value), {201, 202, 203})
        rows = {'holdout': {'by_seed': [
            {'seed': 204, 'delta_means': {'profit': 1.25, 'count': 99}},
            {'seed': '205', 'score': 0.5},
        ]}}
        self.assertEqual(reserved_values(rows), {204, 205})
        # Ambiguous numeric maps are conservative: both key and value exclude.
        self.assertEqual(reserved_values({'test_seeds': {'206': 207}}), {206, 207})

    def test_nested_seed_role_does_not_treat_economics_as_seeds(self):
        value = {'seed_roles': {'holdout': {'seed': 301, 'cash': 5000, 'price': 1.5}}}
        self.assertEqual(reserved_values(value), {301})
        self.assertEqual(reserved_values({'holdout': {'seed_number': 302}}), {302})
        self.assertEqual(reserved_values({'holdout': {'seeds': {'market': [303], 'round2': [304]}}}),
                         {303, 304})
        self.assertEqual(reserved_values({'test_seeds': {'pool': [305, 306]}}), {305, 306})

    def test_select_excludes_old_prior_probe_and_protected_roles(self):
        train, validation = list(range(10, 18)), [30, 31]
        self.assertEqual(select(train, validation, {99}, {12, 15}), [13, 14])
        self.assertEqual(select(train, validation, {99}, {12, 15}), [13, 14])
        with self.assertRaises(ValueError):
            select(train[:3], validation, {99}, {12})
        with self.assertRaises(ValueError):
            select([10, 11, 12, 13, 14], [12, 14], set(), set())

    def test_fractional_seed_rejected_not_silently_coerced(self):
        with self.assertRaises(ValueError):
            reserved_values({'holdout_seeds': [1.25]})

    def test_selected_seed_requires_all_six_passed_reconstruction_coordinates(self):
        audit = dict(passed=True, official_transitions_reexecuted=719,
                     both_public_and_private_states_match_each_step=True,
                     inventory_order_checked_each_step=True)
        rows = [dict(seed=seed, opponent=opponent, side=side, split='train', passed=True,
                     original_teacher_actions_checked=719, reconstructed_actor_actions=719,
                     reproduced_rival_actions=719, audit=audit)
                for seed in (1, 2) for opponent in ('v1', 'market', 'round2') for side in (0, 1)]
        rows.extend(dict(seed=3) for _ in range(48))
        acceptance = dict(stage_execution_passed=True, original_demonstrations_reconstructed=60)
        ledger = dict(passed=True, games=60, failures=[], rows=rows)
        verify_consumed(acceptance, ledger, [1, 2])
        ledger['rows'][0]['passed'] = False
        with self.assertRaises(ValueError):
            verify_consumed(acceptance, ledger, [1, 2])


if __name__ == '__main__':
    unittest.main()
