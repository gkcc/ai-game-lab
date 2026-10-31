"""Synthetic regressions for historical-role and economic binding."""
from copy import deepcopy
import json
import unittest

from . import bind


def fixture(alias='S02', side=0, arm='baseline'):
    binding = {'seeds': {'S02': 101, 'S03': 202},
               'models': {'baseline': {'sha256': 'b' * 64}, 'control': {'sha256': 'c' * 64}},
               'crop': {'sha256': 'd' * 64}, 'opponents': {'market': {'sha256': 'e' * 64}}}
    name = f'dev-{alias}-market-{side}-{arm}'
    job = dict(game=name, seed_alias=alias, seed=binding['seeds'][alias], opponent='market',
               side=side, policy=arm, split='dev', parity_kind=None,
               task_checkpoint_sha256=binding['models'][arm]['sha256'],
               rival_sha256=binding['opponents']['market']['sha256'], bot_sha256='f' * 64)
    banks = [110, 120] if side == 0 else [120, 110]
    row = dict(job, crop_checkpoint_sha256=binding['crop']['sha256'], engine_sha256=bind.ENGINE_SHA,
               passed=True, states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
               semantic_replay_sha256='1' * 64, semantic_replay_sha256_v2='2' * 64,
               replay_gzip_sha256='3' * 64, trace_sha256='4' * 64,
               initial_banks=[100, 100], banks=banks, rewards=banks,
               initial_bank=100, opponent_initial_bank=100,
               net_profit=10, opponent_net_profit=20, bank=110, opponent_bank=120, margin=-10, score=0.0)
    official = dict(passed=True, official_transitions_reexecuted=719,
                    both_public_and_private_states_match_each_step=True,
                    inventory_order_checked_each_step=True, engine_sha256=bind.ENGINE_SHA,
                    semantic_sha256='1' * 64, semantic_sha256_v2='2' * 64, final_rewards=banks)
    audit = dict(game=name, passed=True, errors=0, actor_actions_reproduced=719,
                 rival_actions_reproduced=719, trace_turns_reproduced=719, official=official,
                 trace_sha256='4' * 64, replay_sha256='3' * 64)
    complete = dict(game=name, passed=True)
    own = dict(cash_reconciled=True, initial_cash=100, final_cash=110, net_profit=10,
               sales_units=3, sales_revenue=9, ending_warehouse_units=2)
    opponent = dict(cash_reconciled=True, initial_cash=100, final_cash=120, net_profit=20,
                    sales_units=4, sales_revenue=12, ending_warehouse_units=1)
    result = dict(actor=side, official_turns=719, economy=dict(own=own, opponent=opponent),
                  margin=-10, score=0.0)
    metrics = dict(game=name, seed_alias=alias, seed=binding['seeds'][alias], opponent='market',
                   side=side, arm=arm, task_checkpoint_sha256=binding['models'][arm]['sha256'],
                   crop_checkpoint_sha256=binding['crop']['sha256'], trace_sha256='4' * 64,
                   replay_sha256='3' * 64, result=result)
    public = {'margin': -10, 'score': 0.0}
    for perspective, data in (('own', own), ('opponent', opponent)):
        for key in ('net_profit', 'final_cash', 'sales_units', 'sales_revenue', 'ending_warehouse_units'):
            public[perspective + '_' + key] = data[key]
    pair = dict(seed_alias=bind.ALIASES[alias], side=side, opponent='market',
                reference=deepcopy(public), candidate=deepcopy(public))
    checks = dict(passed=True, assigned=True, resumed=True, launcher_reaped=True,
                  job_members_zero=True, error_kind=None, cleanup_error_kind=None, stop_reason=None,
                  acceptance_error_kind=None, exit_code=0, rss_complete_samples=1)
    supervision = dict(game=name, retry_allowed=False, supervision=checks)
    return [name, binding, pair, job, row, audit, complete, metrics, supervision]


class BindingTests(unittest.TestCase):
    def test_all_market_coordinates_accept_correct_identities(self):
        for alias in ('S02', 'S03'):
            for side in (0, 1):
                for arm in ('baseline', 'control'):
                    with self.subTest(alias=alias, side=side, arm=arm):
                        self.assertTrue(bind.prior_game_identity(*fixture(alias, side, arm)))

    def test_changed_coordinates_economics_or_failed_job_are_rejected(self):
        edits = [
            lambda x: x[3].update(seed=999),
            lambda x: x[4].update(side=1),
            lambda x: x[7].update(arm='control'),
            lambda x: x[2]['reference'].update(own_net_profit=11),
            lambda x: x[7]['result']['economy']['own'].update(initial_cash=99),
            lambda x: x[5]['official'].update(inventory_order_checked_each_step=False),
            lambda x: x[8]['supervision'].update(passed=False),
            lambda x: x[4].update(engine_sha256='9' * 64),
        ]
        for edit in edits:
            args = fixture()
            edit(args)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                bind.prior_game_identity(*args)

    def test_comparison_must_bind_both_game_replay_and_trace_identities(self):
        args = fixture()
        name, binding = args[:2]
        other = 'dev-S02-market-0-control'
        rows = {name: {'replay_gzip_sha256': '1', 'trace_sha256': '2'},
                other: {'replay_gzip_sha256': '3', 'trace_sha256': '4'}}
        data = dict(reference_game=name, candidate_game=other,
                    reference_replay_sha256='1', reference_trace_sha256='2',
                    candidate_replay_sha256='3', candidate_trace_sha256='4',
                    comparison=dict(seed_alias='S02', seed=101, side=0, opponent='market',
                                    reference='baseline', candidate='control'))
        self.assertTrue(bind.prior_comparison_identity(data, name, binding, rows))
        for key in ('candidate_game', 'candidate_replay_sha256', 'reference_trace_sha256'):
            changed = deepcopy(data)
            changed[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                bind.prior_comparison_identity(changed, name, binding, rows)

    def test_official_balances_must_map_to_economic_values_at_either_seat(self):
        for side in (0, 1):
            args = fixture(side=side)
            args[4]['banks'] = [900, 901]
            args[4]['rewards'] = [900, 901]
            args[5]['official']['final_rewards'] = [900, 901]
            with self.subTest(side=side), self.assertRaises(ValueError):
                bind.prior_game_identity(*args)
            args = fixture(side=side)
            args[4]['initial_banks'][side] = 99
            with self.subTest(initial_side=side), self.assertRaises(ValueError):
                bind.prior_game_identity(*args)

    def test_completed_run_requires_successful_clean_complete_receipt(self):
        result = dict(status='passed', games=8, official_replay_passes=8, comparisons=4,
                      training=False, independent_improvement=False, promotion=False)
        payload = json.dumps(result).encode()
        self.assertTrue(bind.completed_run(b'0', payload, 0))
        for exit_code, content, stderr_size in ((b'1', payload, 0), (b'0', payload, 1),
                                                (b'0', json.dumps(dict(result, games=True)).encode(), 0),
                                                (b'0', json.dumps(dict(result, comparisons=3)).encode(), 0)):
            with self.subTest(exit=exit_code, size=stderr_size), self.assertRaises(ValueError):
                bind.completed_run(exit_code, content, stderr_size)


if __name__ == '__main__':
    unittest.main()
