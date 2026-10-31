"""Synthetic preparation checks; no random draw, filesystem writes or games."""
import copy
from decimal import Decimal
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('fresh_multi_prepare_test', Path(__file__).with_name('prepare.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def descriptor(path, digest='a'):
    return dict(path=path, sha256=digest * 64, bytes=2)


def inventory():
    result = dict(schema='task-pair-fresh-multi-inventory-v1', base_commit=prepare.BASE,
        public_json=[descriptor('review/sample.json')],
        artifact_json=[descriptor('prior/seed.json')],
        private_sources=[descriptor('private/a.json'), descriptor('private/b.json'),
                         descriptor('private/c.json'), descriptor('private/d.json')],
        authenticated_private_documents=[descriptor('prior/a.json'), descriptor('prior/b.json'),
                                         descriptor('prior/c.json'), descriptor('prior/d.json')],
        known_counts=dict(task_choice_train=8, task_choice_validation=2, market_formal_reserved=12,
                          market_development=2, protected_documents=4, protected_seed_values=16,
                          market_consumed=2, round2_consumed=2, failed_panel_roles=2),
        market_binding_sha256='b' * 64, input_bytes=12, excluded_uint32=[1, 2],
        scope='synthetic metadata', limitation='synthetic only')
    prepare.validate_inventory(result)
    return result


def synthetic_binding():
    inv = inventory()
    binding = dict(schema='task-pair-fresh-multi-private-binding-v1', base_commit=prepare.BASE,
        salt='3' * 64, generator_key='4' * 64, generator='hmac-sha256-counter-v1', draw_count=6,
        seed_roles={'F01': 0, 'F02': 3, 'F03': 4, 'F04': 5}, cohorts=copy.deepcopy(prepare.COHORTS),
        inventory_sha256='c' * 64, market_binding_sha256='b' * 64,
        public_source_sha256={name: 'd' * 64 for name in prepare.REQUIRED_SOURCES})
    return inv, binding


class InventoryTests(unittest.TestCase):
    def test_raw_boundaries_and_path_tokens(self):
        self.assertEqual(prepare.raw_numbers(b'1000 12 123 00012'), {1000, 12, 123})
        values = prepare.scan_json(b'{"value":22}', 'job-inputs/fixture-991.json')
        self.assertIn(22, values)
        self.assertIn(991, values)

    def test_scientific_text_and_long_zero_padded_path_tokens_are_excluded(self):
        values = prepare.scan_json(b'{"note":"prior seed=1e3 and seed 3.5e1"}',
                                   'old/seed-00000000000000123/replay-2e3-result.json')
        self.assertTrue({1000, 35, 123, 2000}.issubset(values))
        self.assertTrue({100, 1000}.issubset(prepare.scan_json(b'{}', 'old/seed.1e3.result.json')))
        self.assertIn(1200, prepare.scan_json(b'{}', 'old/seed.1.2e3.result.json'))

    def test_huge_exponents_in_text_do_not_break_scan_or_hide_escaped_seed(self):
        raw = b'{"hash":"a7e+9999999999999999999f","note":"seed=\\u0031e\\u0033"}'
        values = prepare.scan_json(raw, 'old/result.json')
        self.assertIn(1000, values)
        self.assertEqual(prepare._numeric('0e9999999999999999999'), 0)
        self.assertIsNone(prepare._numeric('1e9999999999999999999'))
        self.assertIsNone(prepare._numeric('1e-9999999999999999999'))

    def test_structured_scientific_numeric_strings_and_keys(self):
        obj = prepare.decode(b'{"00017":"4.5e1","1e3":3.5e1,"fraction":1.25,"flag":true}')
        values = prepare.structured_numbers(obj)
        self.assertTrue({17, 45, 1000, 35}.issubset(values))
        self.assertNotIn(1, prepare.structured_numbers({'flag': True}))  # JSON true is not integer 1.
        self.assertNotIn(Decimal('1.25'), values)

    def test_duplicate_keys_nonfinite_and_oversized_numeric_text_fail(self):
        for raw in (b'{"seed":1,"seed":2}', b'{"seed":NaN}', b''):
            with self.subTest(raw=raw), self.assertRaises((ValueError, json_error())):
                prepare.decode(raw)
        with self.assertRaises(ValueError):
            prepare.structured_numbers({'seed': '0' * 129})

    def test_replay_named_metadata_is_not_silently_skipped(self):
        self.assertFalse(prepare.include_local_metadata('games/fixture/replay.json'))
        self.assertTrue(prepare.include_local_metadata('audit/replay-fixture-result.json'))
        self.assertTrue(prepare.include_local_metadata('audit/replay-fixture-debit.json'))
        self.assertTrue(prepare.include_local_metadata('audit/job-inputs/fixture.json'))
        self.assertFalse(prepare.include_local_metadata('assets/irrelevant.json'))

    def test_inventory_requires_known_192_reservation_and_unique_descriptors(self):
        valid = inventory()
        for mutate in ('missing_192', 'duplicate_file', 'duplicate_seed', 'wrong_bytes'):
            changed = copy.deepcopy(valid)
            if mutate == 'missing_192': changed['known_counts']['market_formal_reserved'] = 11
            elif mutate == 'duplicate_file': changed['public_json'].append(changed['public_json'][0])
            elif mutate == 'duplicate_seed': changed['excluded_uint32'].append(2)
            else: changed['input_bytes'] -= 1
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                prepare.validate_inventory(changed)


def json_error():
    import json
    return json.JSONDecodeError


class BindingTests(unittest.TestCase):
    def test_fixed_order_rejects_only_collisions_and_duplicates(self):
        candidates = [1, 7, 7, 2, 9, 3, 4]
        with patch.object(prepare, 'candidate_at', side_effect=lambda _key, index: candidates[index]):
            roles, draws = prepare.select_roles(b'k' * 32, {1, 2})
        self.assertEqual(roles, {'F01': 7, 'F02': 9, 'F03': 3, 'F04': 4})
        self.assertEqual(draws, 7)

    def test_binding_reconstructs_roles_and_rejects_tampering(self):
        inv, binding = synthetic_binding()
        with patch.object(prepare, 'candidate_at', side_effect=lambda _key, index: index):
            self.assertTrue(prepare.validate_binding_documents(binding, inv, 'c' * 64))
            for kind in ('seed', 'boolean_seed', 'mapping', 'draw_count', 'inventory', 'market'):
                changed = copy.deepcopy(binding)
                if kind == 'seed': changed['seed_roles']['F01'] = 10
                elif kind == 'boolean_seed': changed['seed_roles']['F01'] = False
                elif kind == 'mapping': changed['cohorts']['C01']['S02'] = 'F04'
                elif kind == 'draw_count': changed['draw_count'] = 5
                elif kind == 'inventory': changed['inventory_sha256'] = 'e' * 64
                else: changed['market_binding_sha256'] = 'e' * 64
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    prepare.validate_binding_documents(changed, inv, 'c' * 64)


if __name__ == '__main__':
    unittest.main()
