"""Purely synthetic projection checks, without private files or game IO."""
from copy import deepcopy
import json
import unittest

from learning.task_pair_dev_games_v1.panel import ALIASES, OPPONENTS, POLICIES
from learning.task_pair_dev_metrics_v1.panel import paired
from learning.task_pair_dev_metrics_v1.test_consumer import complete_game, record, unit_event
from .projection import COMPARISONS, _decision_counts, project


SEEDS = {'S02': 4000000000, 'S03': 4000000001}


def economy(initial, final):
    return dict(initial_cash=initial, final_cash=final, net_profit=final-initial,
                successful_plants=4, seed_units_bought=5, sales_units=6,
                sales_revenue=70, ending_seed_units=1, ending_warehouse_units=2,
                cash_reconciled=True)


def row(alias, opponent, side, arm):
    n = POLICIES.index(arm)
    own, other = economy(100, 150 + n + side), economy(100, 120 + n)
    margin = own['final_cash'] - other['final_cash']
    def event(step, selected, request, actual, status, effect, label='observed'):
        return dict(step=step, player=side, unit=0, source='callback', selected_op=selected,
                    request_op=request, effective_op=actual, status=status,
                    same_target_atomic_effect=effect, label_support=label)
    events = [event(0, 'PLANT', 'NORTH', 'NORTH', 'executed', False),
              event(1, 'PLANT', 'PLANT', 'PLANT', 'executed', True),
              event(2, 'WATER', 'WATER', None, None, None, label='unknown'),
              event(3, 'HARVEST', 'REVEAL-ME', 'REVEAL-ME', 'executed', False),
              event(4, 'DIG', 'PASS', 'PASS', 'pass', False)]
    selection_counts = dict(model_callbacks=5, model_callback_multi_unit_opportunities=3,
        model_callback_selected_multi_unit_opportunities=1, **{
            'selected:PLANT': 2, 'selected:WATER': 1, 'selected:HARVEST': 1, 'selected:DIG': 1,
            'request:NORTH': 1, 'request:PLANT': 1, 'request:WATER': 1, 'request:REVEAL-ME': 1,
            'request:PASS': 1, 'unit_status:executed': 3, 'unit_status:pass': 1,
            'same_target_atomic_effect': 1, 'unmatched_unit_label': 1})
    return dict(schema='task-pair-dev-game-metrics-v1',
                game=f'dev-{alias}-{opponent}-{side}-{arm}', seed=SEEDS[alias],
                seed_alias=alias, opponent=opponent, side=side, arm=arm,
                result=dict(actor=side, official_turns=719, economy=dict(own=own, opponent=other),
                            margin=margin, score=1. if margin > 0 else .5 if margin == 0 else 0.,
                            events=events, selection_counts=selection_counts, execution_support='unknown'))


def comparison(reference, candidate, alias, opponent, side):
    return dict(reference=reference, candidate=candidate, seed=SEEDS[alias], seed_alias=alias,
                opponent=opponent, side=side,
                result=dict(comparison_complete=True, matched_calls=2, stopped=True,
                            stop_reason='joint_action_difference', first_choice_disagreement=0,
                            matched_call_request_differences=0, matched_multi_unit_opportunities=1,
                            matched_selected_multi_unit=0, matched_selected_different_unit=0))


def fixture():
    rows = [row(a, o, s, p) for a in ALIASES for o in OPPONENTS
            for s in (0, 1) for p in POLICIES]
    comparisons = [comparison(r, c, a, o, s) for r, c in COMPARISONS
                   for a in ALIASES for o in OPPONENTS for s in (0, 1)]
    summaries = {}
    for reference, candidate in COMPARISONS:
        summaries[reference + '_to_' + candidate] = paired(
            [item for item in rows if item['arm'] in (reference, candidate)],
            seeds=[SEEDS[a] for a in ALIASES], opponents=OPPONENTS,
            reference=reference, candidate=candidate,
            comparisons=[item for item in comparisons
                         if (item['reference'], item['candidate']) == (reference, candidate)])
    campaign = dict(schema='task-pair-dev-campaign-result-v1', passed=True, games=24,
                    official_replay_passes=24, seed_clusters=2, independent_improvement=False,
                    promotion=False, receipts=[dict(game=item['game'], private_hash='hidden') for item in rows],
                    summaries=summaries)
    # The driver persists JSON; integer map keys become strings after reading.
    return json.loads(json.dumps(campaign)), rows, comparisons


class ProjectionTests(unittest.TestCase):
    def call(self, campaign, rows, comparisons):
        return project(campaign, seeds=SEEDS, rows=rows, comparisons=comparisons)

    def test_complete_projection_is_allowlisted_and_descriptive(self):
        campaign, rows, comparisons = fixture()
        campaign['private_path'] = 'secret-path-example'
        rows[0]['private_data'] = 'secret-observation-example'
        comparisons[0]['private_trace'] = 'secret-trace-example'
        output = self.call(campaign, rows, comparisons)
        self.assertEqual((len(output['games']), len(output['paired'])), (24, 16))
        self.assertEqual(output['by_arm']['baseline']['all']['games'], 8)
        self.assertEqual(output['summaries']['baseline_to_control']['overall']['pairs'], 8)
        support = output['summaries']['baseline_to_control']['opportunity_support']
        self.assertEqual((support['reference']['support'], support['reference']['total']), ('supported', 24))
        self.assertEqual((support['same_input_pairs']['support'], support['same_input_pairs']['total']), ('unknown', 8))
        self.assertEqual((output['statistical_significance'], output['promotion']), ('unknown', False))
        decision = output['games'][0]['decision_counts']
        self.assertEqual((decision['scope'], decision['total']), ('selected_tasks_only', 5))
        self.assertEqual(decision['selected_task']['PLANT'], 2)
        self.assertEqual(decision['submitted_unit_request']['movement'], 1)
        self.assertEqual(decision['submitted_unit_request']['PLANT'], 1)
        self.assertEqual(decision['submitted_unit_request']['other'], 1)
        official = decision['official_unit_labels']
        self.assertEqual((official['coverage'], official['matched'], official['unmatched']), ('unknown', 4, 1))
        self.assertEqual((official['effective_action']['movement'], official['effective_action']['PLANT'],
                          official['effective_action']['other']), (1, 1, 1))
        self.assertEqual((official['status']['executed'], official['status']['pass']), (3, 1))
        self.assertEqual(official['same_target_atomic_effect'],
                         dict(observed_true=1, observed_false=3, unknown_with_label=0, unknown_unmatched=1))
        self.assertEqual(official['same_target_atomic_effect_by_selected_task']['PLANT'],
                         dict(observed_true=1, observed_false=1, unknown_with_label=0, unknown_unmatched=0))
        self.assertEqual(official['same_target_atomic_effect_by_selected_task']['WATER']['unknown_unmatched'], 1)
        self.assertEqual(output['metric_scope']['successful_plants'], 'all_official_units')
        rendered = json.dumps(output, sort_keys=True)
        for secret in ('4000000000', '4000000001', 'secret-path-example',
                       'secret-observation-example', 'secret-trace-example', 'private_hash', 'REVEAL-ME'):
            self.assertNotIn(secret, rendered)

    def test_real_consumer_selected_plant_move_and_missing_label_project_correctly(self):
        moving = _decision_counts(complete_game([record()], [unit_event()]), 0)
        self.assertEqual(moving['selected_task']['PLANT'], 1)
        self.assertEqual(moving['submitted_unit_request']['movement'], 1)
        self.assertEqual(moving['official_unit_labels']['effective_action']['movement'], 1)
        self.assertEqual(moving['official_unit_labels']['same_target_atomic_effect']['observed_false'], 1)
        self.assertEqual(moving['official_unit_labels']['same_target_atomic_effect_by_selected_task']
                         ['PLANT']['observed_false'], 1)
        self.assertEqual(moving['official_unit_labels']['coverage'], 'observed')
        missing = _decision_counts(complete_game([record()], []), 0)
        self.assertEqual((missing['official_unit_labels']['coverage'],
                          missing['official_unit_labels']['unmatched']), ('unknown', 1))
        self.assertEqual(missing['official_unit_labels']['same_target_atomic_effect']['unknown_unmatched'], 1)
        self.assertEqual(missing['official_unit_labels']['same_target_atomic_effect_by_selected_task']
                         ['PLANT']['unknown_unmatched'], 1)
        no_position = unit_event()
        no_position['position'] = None
        uncertain = _decision_counts(complete_game([record()], [no_position]), 0)
        self.assertEqual(uncertain['official_unit_labels']['same_target_atomic_effect_by_selected_task']
                         ['PLANT']['unknown_with_label'], 1)
        self.assertEqual(uncertain['official_unit_labels']['same_target_atomic_effect_by_selected_task']
                         ['PLANT']['observed_false'], 0)

    def test_zero_opportunities_remain_unknown(self):
        campaign, rows, comparisons = fixture()
        for item in rows:
            item['result']['selection_counts']['model_callback_multi_unit_opportunities'] = 0
            item['result']['selection_counts']['model_callback_selected_multi_unit_opportunities'] = 0
        for item in comparisons:
            item['result']['matched_multi_unit_opportunities'] = 0
        for reference, candidate in COMPARISONS:
            key = reference + '_to_' + candidate
            campaign['summaries'][key] = json.loads(json.dumps(paired(
                [x for x in rows if x['arm'] in (reference, candidate)],
                seeds=[SEEDS[a] for a in ALIASES], opponents=OPPONENTS,
                reference=reference, candidate=candidate,
                comparisons=[x for x in comparisons
                             if (x['reference'], x['candidate']) == (reference, candidate)])))
        result = self.call(campaign, rows, comparisons)
        support = result['summaries']['baseline_to_control']['opportunity_support']
        self.assertEqual((support['reference']['support'], support['reference']['total']), ('unknown', 0))
        self.assertEqual((support['same_input_pairs']['support'], support['same_input_pairs']['total']), ('unknown', 0))

    def test_duplicate_missing_or_misaligned_game_is_rejected(self):
        campaign, rows, comparisons = fixture()
        for changed in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                self.call(campaign, changed, comparisons)
        changed = deepcopy(rows); changed[0]['seed_alias'] = 'S03'
        with self.assertRaises(ValueError): self.call(campaign, changed, comparisons)
        changed = deepcopy(rows); changed[0]['game'] = 'dev-S02-market-1-baseline'
        with self.assertRaises(ValueError): self.call(campaign, changed, comparisons)

    def test_duplicate_missing_or_falsely_matched_comparison_is_rejected(self):
        campaign, rows, comparisons = fixture()
        for changed in (comparisons[:-1], comparisons[:-1] + [comparisons[0]]):
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                self.call(campaign, rows, changed)
        changed = deepcopy(comparisons); changed[0]['seed_alias'] = 'S03'
        with self.assertRaises(ValueError): self.call(campaign, rows, changed)
        changed = deepcopy(comparisons); changed[0]['result']['matched_calls'] = 99
        changed[0]['result']['matched_multi_unit_opportunities'] = 4
        with self.assertRaises(ValueError): self.call(campaign, rows, changed)

    def test_recomputed_summary_rejects_forged_value_or_support(self):
        campaign, rows, comparisons = fixture()
        changed = deepcopy(campaign)
        changed['summaries']['baseline_to_control']['overall']['values']['margin'] = 999
        with self.assertRaises(ValueError): self.call(changed, rows, comparisons)
        changed = deepcopy(campaign)
        changed['summaries']['baseline_to_control']['opportunity_support']['same_input_pairs']['support'] = 'supported'
        with self.assertRaises(ValueError): self.call(changed, rows, comparisons)
        changed = deepcopy(campaign)
        changed['summaries']['control_to_auxiliary']['pairs'][0]['reference']['own_net_profit'] += 1
        with self.assertRaises(ValueError): self.call(changed, rows, comparisons)

    def test_unreconciled_cash_score_or_nonfinite_rejected(self):
        campaign, rows, comparisons = fixture()
        for edit in ('cash', 'score', 'nonfinite', 'missing_count'):
            changed = deepcopy(rows)
            result = changed[0]['result']
            if edit == 'cash': result['economy']['own']['final_cash'] += 1
            if edit == 'score': result['score'] = .5
            if edit == 'nonfinite': result['economy']['own']['sales_revenue'] = float('nan')
            if edit == 'missing_count': result.pop('selection_counts')
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                self.call(campaign, changed, comparisons)

    def test_wrong_selection_request_execution_or_missing_is_rejected(self):
        campaign, rows, comparisons = fixture()
        for edit in ('task', 'request', 'status', 'effect', 'source', 'duplicate', 'false_unmatched'):
            changed = deepcopy(rows)
            result = changed[0]['result']
            if edit == 'task': result['selection_counts']['selected:PLANT'] += 1
            if edit == 'request': result['selection_counts']['request:NORTH'] += 1
            if edit == 'status': result['selection_counts']['unit_status:executed'] += 1
            if edit == 'effect': result['selection_counts']['same_target_atomic_effect'] += 1
            if edit == 'source': result['events'][0]['source'] = 'teacher'
            if edit == 'duplicate': result['events'][1]['step'] = 0
            if edit == 'false_unmatched': result['events'][2]['same_target_atomic_effect'] = False
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                self.call(campaign, changed, comparisons)

    def test_no_selected_tasks_means_unknown_execution_not_failure(self):
        campaign, rows, comparisons = fixture()
        for item in rows:
            result = item['result']
            result['events'] = []
            result['selection_counts'] = dict(model_callbacks=0, model_callback_multi_unit_opportunities=0,
                model_callback_selected_multi_unit_opportunities=0)
            result['execution_support'] = 'unknown'
        for item in comparisons:
            item['result'].update(matched_calls=0, matched_multi_unit_opportunities=0)
        for reference, candidate in COMPARISONS:
            key = reference + '_to_' + candidate
            campaign['summaries'][key] = json.loads(json.dumps(paired(
                [x for x in rows if x['arm'] in (reference, candidate)],
                seeds=[SEEDS[a] for a in ALIASES], opponents=OPPONENTS,
                reference=reference, candidate=candidate,
                comparisons=[x for x in comparisons
                             if (x['reference'], x['candidate']) == (reference, candidate)])))
        official = self.call(campaign, rows, comparisons)['games'][0]['decision_counts']['official_unit_labels']
        self.assertEqual((official['coverage'], official['matched'], official['unmatched']), ('unknown', 0, 0))
        self.assertEqual(official['same_target_atomic_effect']['observed_false'], 0)


if __name__ == '__main__':
    unittest.main()
