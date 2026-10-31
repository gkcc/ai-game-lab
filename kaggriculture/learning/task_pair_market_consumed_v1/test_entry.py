"""Synthetic identity and comparison regressions; no actual game is started."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_historical_compare_v1.test_projection import fixture
from . import entry


class EntryTests(unittest.TestCase):
    def test_plans_are_eight_market_games_with_both_seats_and_arms(self):
        seeds = {'S02': 11017, 'S03': 22029}
        models = {name: {'path': 'artifacts/synthetic/' + name + '.npz', 'sha256': char * 64}
                  for name, char in zip(('baseline', 'control', 'auxiliary'), 'abc')}
        panel = Panel.from_private_contract(seeds=seeds, models=models,
            crop={'path': 'artifacts/synthetic/crop.npz', 'sha256': 'd' * 64},
            rivals={'market': 'e' * 64, 'round2': 'f' * 64}, worker_sha='1' * 64)
        plans = entry.plans_for(SimpleNamespace(panel=panel))
        self.assertEqual(len(plans), 8)
        self.assertEqual({(p['seed_alias'], p['opponent'], p['side'], p['policy']) for p in plans},
                         {(a, 'market', s, m) for a in ('S02', 'S03') for s in (0, 1)
                          for m in ('baseline', 'control')})

    def test_markers_are_bound_to_coordinate_and_reject_unknown_games(self):
        commit, digest = 'a' * 40, 'b' * 64
        value = json.loads(entry.marker('permit', commit=commit, binding_sha=digest, game=entry.GAMES[0]))
        self.assertEqual((value['game'], value['registration_commit'], value['retry_allowed']),
                         (entry.GAMES[0], commit, False))
        with self.assertRaises(ValueError):
            entry.marker('permit', commit=commit, binding_sha=digest, game='unregistered')

    def test_private_binding_is_recomputed_and_compared_in_full(self):
        doc = dict(schema='task-pair-market-consumed-binding-v1', base_commit=entry.BASE,
                   old_binding_sha256='b' * 64, salt='c' * 64, seeds={'S02': 1, 'S03': 2})
        raw = (json.dumps(doc) + '\n').encode()
        prep = SimpleNamespace(collect=lambda _old, _salt: deepcopy(doc))
        with patch.object(entry, 'read', return_value=raw):
            self.assertEqual(entry.binding_context(prep, entry.sha(raw))[1], doc)
            prep.collect = lambda _old, _salt: dict(doc, seeds={'S02': 1, 'S03': 3})
            with self.assertRaises(ValueError):
                entry.binding_context(prep, entry.sha(raw))
            with self.assertRaises(ValueError):
                entry.binding_context(prep, 'd' * 64)

    def test_comparison_receipt_cannot_change_stop_reason(self):
        _paired, _rows, receipts = fixture()
        receipt = receipts[0]
        comparison = deepcopy(receipt['comparison'])
        self.assertTrue(entry.verify_comparison_receipt(receipt, comparison))
        receipt['comparison']['result']['stop_reason'] = 'input_difference'
        with self.assertRaises(ValueError):
            entry.verify_comparison_receipt(receipt, comparison)

    def test_typed_comparison_rejects_bool_for_integer(self):
        self.assertFalse(entry.typed_equal({'side': 1}, {'side': True}))
        self.assertFalse(entry.typed_equal([1], (1,)))

    def test_memory_floor_before_import_or_identity_work(self):
        self.assertTrue(entry.enforce_start_memory(child=False, reader=lambda: entry.INITIAL_MEMORY))
        self.assertTrue(entry.enforce_start_memory(child=True, reader=lambda: entry.CHILD_MEMORY))
        with self.assertRaises(MemoryError):
            entry.enforce_start_memory(child=False, reader=lambda: entry.INITIAL_MEMORY - 1)
        with self.assertRaises(MemoryError):
            entry.enforce_start_memory(child=True, reader=lambda: entry.CHILD_MEMORY - 1)
        with self.assertRaises(ValueError):
            entry.enforce_start_memory(child=False, reader=lambda: True)

    def test_coordinator_budget_includes_preparation_time(self):
        from learning.task_pair_dev_campaign_v1.campaign import _Clock
        clock = {'now': 100}
        sampled, coordinator = entry.clocks(0, lambda: clock['now'])
        budget = _Clock(1000, coordinator)
        self.assertEqual(budget.remaining(), 900)
        clock['now'] = 999
        self.assertEqual(budget.remaining(), 1)
        clock['now'] = 1000
        with self.assertRaises(TimeoutError):
            budget.remaining()
        self.assertEqual(sampled(), 1000)

    def test_compact_result_does_not_duplicate_large_event_rows(self):
        payload = {'paired': {'summary': 'ok'}, 'rows': [{'events': 'secret' * 1000000}] * 8}
        prefixes = {(alias, side): {'status': 'observed'} for alias in ('S02', 'S03') for side in (0, 1)}
        result = entry.compact_result(payload=payload, commit='a' * 40, binding_sha='b' * 64,
                                      ledger=[{}] * 8, comparison_ledger=[{}] * 4, prefixes=prefixes)
        encoded = json.dumps(result)
        self.assertLess(len(encoded), 4096)
        self.assertNotIn('secret', encoded)


if __name__ == '__main__':
    unittest.main()
