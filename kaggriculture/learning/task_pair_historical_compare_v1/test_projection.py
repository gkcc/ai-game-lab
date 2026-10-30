"""Synthetic, in-memory regression checks; no game, private files or seeds."""
from copy import deepcopy
import hashlib
import json
import unittest

from learning.task_pair_dev_metrics_v1.panel import paired
from learning.task_pair_dev_metrics_v1.consumer import CommonHistoryComparison
from learning.task_pair_dev_metrics_v1.test_consumer import actions, frame, turn
from learning.task_request_gate_v1.policy import same_request
from .projection import project


SEEDS = {'S02': 4000000000, 'S03': 4000000001}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def economy(initial, final):
    return dict(initial_cash=initial, final_cash=final, net_profit=final - initial,
                successful_plants=4, seed_units_bought=5, sales_units=6,
                sales_revenue=70, ending_seed_units=1, ending_warehouse_units=2,
                cash_reconciled=True)


def make_row(alias, side, arm):
    increment = 6 if arm == 'control' else 0
    own = economy(100, 120 + increment + 4 * side)
    other = economy(100, 125 + 2 * side)
    margin = own['final_cash'] - other['final_cash']
    def event(step, selected, request, actual, status, effect, label='observed'):
        return dict(step=step, player=side, unit=0, source='callback', selected_op=selected,
                    request_op=request, effective_op=actual, status=status,
                    same_target_atomic_effect=effect, label_support=label)
    events = [event(0, 'PLANT', 'NORTH', 'NORTH', 'executed', False),
              event(1, 'WATER', 'WATER', None, None, None, 'unknown'),
              event(2, 'HARVEST', 'PRIVATE-REQUEST', 'PRIVATE-REQUEST', 'executed', False)]
    counts = dict(model_callbacks=3, model_callback_multi_unit_opportunities=2,
                  model_callback_selected_multi_unit_opportunities=1, **{
                      'selected:PLANT': 1, 'selected:WATER': 1, 'selected:HARVEST': 1,
                      'request:NORTH': 1, 'request:WATER': 1, 'request:PRIVATE-REQUEST': 1,
                      'unit_status:executed': 2, 'unmatched_unit_label': 1})
    game = f'dev-{alias}-market-{side}-{arm}'
    return dict(schema='task-pair-dev-game-metrics-v1', game=game, seed=SEEDS[alias],
                seed_alias=alias, opponent='market', side=side, arm=arm,
                task_checkpoint_sha256=digest('task-' + arm), crop_checkpoint_sha256=digest('crop'),
                replay_sha256=digest('replay-' + game), trace_sha256=digest('trace-' + game),
                result=dict(actor=side, official_turns=719, economy=dict(own=own, opponent=other),
                            margin=margin, score=1. if margin > 0 else .5 if margin == 0 else 0.,
                            events=events, selection_counts=counts, execution_support='unknown'))


def make_receipt(alias, side, rows, reason):
    reference = next(r for r in rows if (r['seed_alias'], r['side'], r['arm']) == (alias, side, 'baseline'))
    candidate = next(r for r in rows if (r['seed_alias'], r['side'], r['arm']) == (alias, side, 'control'))
    calls = 0 if reason == 'frame_difference' else 2
    opportunities = 1 if calls else 0
    result = dict(support='observed' if calls else 'unknown', comparison_complete=True,
                  matched_calls=calls, first_choice_disagreement=int(reason == 'choice_difference'),
                  matched_call_request_differences=int(reason == 'choice_difference'),
                  matched_multi_unit_opportunities=opportunities, matched_selected_multi_unit=0,
                  matched_selected_different_unit=0, stopped=reason is not None, stop_reason=reason,
                  interpretation='synthetic private note')
    comparison = dict(reference='baseline', candidate='control', seed=SEEDS[alias], seed_alias=alias,
                      opponent='market', side=side, result=result)
    return dict(schema='task-pair-dev-comparison-v1', reference_game=reference['game'],
                candidate_game=candidate['game'], reference_replay_sha256=reference['replay_sha256'],
                candidate_replay_sha256=candidate['replay_sha256'],
                reference_trace_sha256=reference['trace_sha256'],
                candidate_trace_sha256=candidate['trace_sha256'], comparison=comparison)


def fixture():
    rows = [make_row(alias, side, arm) for alias in SEEDS for side in (0, 1)
            for arm in ('baseline', 'control')]
    reasons = ('choice_difference', 'joint_action_difference', None, 'frame_difference')
    receipts = [make_receipt(alias, side, rows, reason)
                for (alias, side), reason in zip(((a, s) for a in SEEDS for s in (0, 1)), reasons)]
    result = paired(rows, seeds=list(SEEDS.values()), opponents=['market'], reference='baseline',
                    candidate='control', comparisons=[r['comparison'] for r in receipts])
    return json.loads(json.dumps(result)), rows, receipts


class ProjectionTests(unittest.TestCase):
    def call(self, summary, rows, receipts, seeds=SEEDS):
        return project(summary, seeds=seeds, rows=rows, comparisons=receipts)

    def test_public_projection_contains_only_fixed_whitelist_and_paired_results(self):
        summary, rows, receipts = fixture()
        rows[0]['private_path'] = 'SECRET-PATH'
        rows[0]['result']['private_observation'] = 'SECRET-OBSERVATION'
        receipts[0]['private_trace'] = 'SECRET-TRACE'
        receipts[0]['comparison']['private_request'] = 'SECRET-REQUEST'
        result = self.call(summary, rows, receipts)
        self.assertEqual((result['games'], result['seed_clusters'], len(result['pairs'])), (8, 2, 4))
        self.assertEqual([p['seed_alias'] for p in result['pairs']], ['S01', 'S01', 'S04', 'S04'])
        self.assertEqual([p['side'] for p in result['pairs']], [0, 1, 0, 1])
        self.assertEqual(result['overall']['mean_delta']['own_net_profit'], 6)
        self.assertEqual(result['overall']['mean_delta']['opponent_net_profit'], 0)
        self.assertEqual(result['overall']['mean_delta']['margin'], 6)
        self.assertEqual(result['overall']['outcomes']['reference']['loss'], 4)
        self.assertEqual(result['overall']['outcomes']['candidate']['win'], 4)
        self.assertEqual((result['statistical_significance'], result['promotion']), ('unknown', False))
        metrics = result['pairs'][0]['candidate']['decision_counts']
        self.assertEqual(metrics['selected_task']['PLANT'], 1)
        self.assertEqual(metrics['submitted_unit_request']['movement'], 1)
        self.assertEqual(metrics['submitted_unit_request']['other'], 1)
        self.assertEqual(metrics['official_unit_labels']['unmatched'], 1)
        rendered = json.dumps(result, sort_keys=True)
        for secret in (str(SEEDS['S02']), str(SEEDS['S03']), 'SECRET-PATH', 'SECRET-OBSERVATION',
                       'SECRET-TRACE', 'SECRET-REQUEST', 'PRIVATE-REQUEST', digest('task-control'),
                       digest('crop'), digest('replay-' + rows[0]['game']), 'synthetic private note'):
            self.assertNotIn(secret, rendered)

    def test_common_history_never_invents_a_full_request_comparison(self):
        summary, rows, receipts = fixture()
        pairs = self.call(summary, rows, receipts)['pairs']
        choice, joint, complete, unmatched = (item['common_history'] for item in pairs)
        self.assertEqual(choice['first_task_choice_difference'], 'observed')
        self.assertEqual(choice['matching_record_unit_request_differences'], 1)
        self.assertEqual(choice['first_joint_request_difference'], 'unknown')
        self.assertEqual(choice['first_actor_request_difference'], 'unknown')
        self.assertIsNone(choice['first_difference_step'])
        self.assertEqual(joint['first_joint_request_difference'], 'unknown')
        self.assertIs(joint['ordered_joint_representation_difference'], True)
        self.assertEqual(complete['first_joint_request_difference'], 'none_in_complete_history')
        self.assertIs(complete['ordered_joint_representation_difference'], False)
        self.assertEqual(complete['first_actor_request_difference'], 'none_in_complete_history')
        self.assertEqual(unmatched['first_task_choice_difference'], 'unknown')
        self.assertEqual(unmatched['first_joint_request_difference'], 'unknown')

    def test_real_comparator_key_order_difference_is_not_a_semantic_request_difference(self):
        left, right = actions(), actions()
        right[0] = dict(market=[], hands=[], farmer=['PASS'])
        self.assertTrue(same_request(left[0], right[0]))
        original = CommonHistoryComparison(0)
        original.observe(0, frame(0), frame(0), turn(0), turn(0), left, right)
        comparison = original.result()
        self.assertEqual(comparison['stop_reason'], 'joint_action_difference')
        summary, rows, receipts = fixture()
        receipts[0]['comparison']['result'] = comparison
        summary = paired(rows, seeds=list(SEEDS.values()), opponents=['market'], reference='baseline',
                         candidate='control', comparisons=[r['comparison'] for r in receipts])
        common = self.call(summary, rows, receipts)['pairs'][0]['common_history']
        self.assertEqual(common['first_joint_request_difference'], 'unknown')
        self.assertIs(common['ordered_joint_representation_difference'], True)
        self.assertEqual(common['first_actor_request_difference'], 'unknown')

    def test_raw_and_json_decoded_paired_results_both_validate(self):
        summary, rows, receipts = fixture()
        raw = paired(rows, seeds=list(SEEDS.values()), opponents=['market'], reference='baseline',
                     candidate='control', comparisons=[r['comparison'] for r in receipts])
        self.assertEqual(self.call(summary, rows, receipts), self.call(raw, rows, receipts))

    def test_missing_duplicate_misaligned_or_bad_identity_game_is_rejected(self):
        summary, rows, receipts = fixture()
        class StringSubclass(str):
            pass
        bad_sets = [rows[:-1], rows[:-1] + [rows[0]]]
        for field, value in (('seed_alias', 'S03'), ('seed', SEEDS['S03']), ('side', True),
                             ('opponent', 'round2'), ('arm', 'auxiliary'),
                             ('opponent', StringSubclass('market')),
                             ('game', 'dev-S02-market-1-baseline'), ('trace_sha256', 'not-a-digest')):
            changed = deepcopy(rows); changed[0][field] = value; bad_sets.append(changed)
        changed = deepcopy(rows); changed[0]['crop_checkpoint_sha256'] = digest('different-crop'); bad_sets.append(changed)
        changed = deepcopy(rows); changed[0]['task_checkpoint_sha256'] = digest('different-task'); bad_sets.append(changed)
        changed = deepcopy(rows)
        for row in changed: row['task_checkpoint_sha256'] = digest('same-task')
        bad_sets.append(changed)
        for changed in bad_sets:
            with self.subTest(variant=len(changed)), self.assertRaises(ValueError):
                self.call(summary, changed, receipts)

    def test_missing_duplicate_misaligned_or_unbound_comparison_is_rejected(self):
        summary, rows, receipts = fixture()
        variants = [receipts[:-1], receipts[:-1] + [receipts[0]]]
        for field, value in (('reference_game', 'wrong-game'), ('candidate_game', 'wrong-game'),
                             ('reference_trace_sha256', digest('wrong-trace')),
                             ('candidate_replay_sha256', digest('wrong-replay'))):
            changed = deepcopy(receipts); changed[0][field] = value; variants.append(changed)
        for field, value in (('seed_alias', 'S03'), ('seed', SEEDS['S03']), ('side', 1),
                             ('opponent', 'round2'), ('reference', 'control'), ('candidate', 'baseline')):
            changed = deepcopy(receipts); changed[0]['comparison'][field] = value; variants.append(changed)
        for field, value in (('matched_calls', 4), ('matched_multi_unit_opportunities', 3),
                             ('first_choice_disagreement', 0), ('stopped', False),
                             ('stop_reason', 'PRIVATE-STRING'), ('support', 'unknown')):
            changed = deepcopy(receipts); changed[0]['comparison']['result'][field] = value; variants.append(changed)
        for changed in variants:
            with self.subTest(variant=len(changed)), self.assertRaises(ValueError):
                self.call(summary, rows, changed)

    def test_unreconciled_cash_nonfinite_and_corrupt_telemetry_are_rejected(self):
        summary, rows, receipts = fixture()
        for error in ('cash', 'score', 'nonfinite', 'missing', 'telemetry', 'teacher_source'):
            changed = deepcopy(rows); result = changed[0]['result']
            if error == 'cash': result['economy']['own']['final_cash'] += 1
            if error == 'score': result['score'] = .5
            if error == 'nonfinite': result['economy']['own']['sales_revenue'] = float('nan')
            if error == 'missing': result.pop('selection_counts')
            if error == 'telemetry': result['selection_counts']['selected:PLANT'] += 1
            if error == 'teacher_source': result['events'][0]['source'] = 'teacher'
            with self.subTest(error=error), self.assertRaises(ValueError):
                self.call(summary, changed, receipts)

    def test_forged_or_ambiguous_paired_result_is_rejected(self):
        summary, rows, receipts = fixture()
        variants = []
        changed = deepcopy(summary); changed['overall']['values']['margin'] += 1; variants.append(changed)
        changed = deepcopy(summary); changed['statistical_significance'] = 'significant'; variants.append(changed)
        changed = deepcopy(summary); changed['pairs'][0]['candidate']['own_net_profit'] += 1; variants.append(changed)
        changed = deepcopy(summary); changed['overall']['values']['margin'] = float('inf'); variants.append(changed)
        changed = deepcopy(summary); changed.pop('by_seed'); variants.append(changed)
        raw = paired(rows, seeds=list(SEEDS.values()), opponents=['market'], reference='baseline',
                     candidate='control', comparisons=[r['comparison'] for r in receipts])
        raw['by_seed'][str(SEEDS['S02'])] = raw['by_seed'][SEEDS['S02']]
        variants.append(raw)
        for changed in variants:
            with self.assertRaises(ValueError): self.call(changed, rows, receipts)

    def test_zero_matched_calls_keeps_task_selection_unknown(self):
        summary, rows, receipts = fixture()
        for row in rows:
            row['result']['events'] = []
            row['result']['selection_counts'] = dict(model_callbacks=0,
                model_callback_multi_unit_opportunities=0, model_callback_selected_multi_unit_opportunities=0)
            row['result']['execution_support'] = 'unknown'
        for receipt in receipts:
            receipt['comparison']['result'].update(support='unknown', matched_calls=0,
                first_choice_disagreement=0, matched_call_request_differences=0,
                matched_multi_unit_opportunities=0, matched_selected_multi_unit=0,
                matched_selected_different_unit=0, stopped=True, stop_reason='frame_difference')
        summary = paired(rows, seeds=list(SEEDS.values()), opponents=['market'], reference='baseline',
                         candidate='control', comparisons=[r['comparison'] for r in receipts])
        result = self.call(summary, rows, receipts)
        self.assertTrue(all(pair['common_history']['first_task_choice_difference'] == 'unknown'
                            and pair['common_history']['first_joint_request_difference'] == 'unknown'
                            for pair in result['pairs']))
        self.assertEqual(result['pairs'][0]['reference']['decision_counts']['official_unit_labels']['coverage'],
                         'unknown')

    def test_bad_bound_roles_are_rejected(self):
        summary, rows, receipts = fixture()
        for seeds in ({}, {'S02': SEEDS['S02'], 'S03': SEEDS['S02']},
                      {'S02': True, 'S03': SEEDS['S03']}, {'S01': SEEDS['S02'], 'S04': SEEDS['S03']}):
            with self.assertRaises(ValueError): self.call(summary, rows, receipts, seeds)


if __name__ == '__main__':
    unittest.main()
