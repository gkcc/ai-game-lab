"""Synthetic in-memory checks only; no real replay, data, models or engine."""
from copy import deepcopy
from collections import Counter
from types import SimpleNamespace
import unittest

from decision.execution import ExecutionTrace
from .consumer import CommonHistoryComparison, GameConsumer, OFFICIAL_TURNS
from .panel import paired


def candidate(unit, target, op):
    return dict(id='c' + str(unit) + op, unit=unit, target=target, op=op, x=[1.])


def record(step=0, unit=0, target=None, op='PLANT', request=None, candidates=None):
    target = [1, 0] if target is None else target
    request = ['NORTH'] if request is None else request
    candidates = [candidate(unit, target, op)] if candidates is None else candidates
    choice = next(c for c in candidates if c['unit'] == unit and c['op'] == op)
    return dict(source='callback', unit=unit, target=target, op=op,
        chosen=choice['id'], teacher_label=candidates[0]['id'], unit_request=request,
        inputs=dict(observation=dict(player=0, step=step, private=dict(a=1, b=2)),
                    configuration=dict(boardSize=10), planned=dict(unit_actions=[]),
                    shared_x=[1.], candidates=candidates))


def turn(step, records=()):
    return dict(step=step, records=list(records), load_errors=0, inference_errors=0, fallbacks=0)


def unit_event(step=0, player=0, unit=0, request=None, effective=None, status='executed', position=None):
    request = ['NORTH'] if request is None else request
    effective = request if effective is None else effective
    return dict(step=step, player=player, phase='unit', unit=unit, requested=request,
                effective=effective, position=[0, 0] if position is None else position,
                status=status, before=dict(money=100), after=dict(money=100))


def market_event(player, request, status, units=0, cash=0, order=0, wanted=None):
    return dict(step=0, player=player, phase='market', requested=request,
                status=status, executed_units=units, cash_delta=cash, order=order,
                requested_units=wanted)


def observations(cash0, cash1):
    farms = [dict(money=cash0), dict(money=cash1)]
    return [dict(farms=deepcopy(farms), private=dict(seeds={}, shed={}, inventories=[])) for _ in (0, 1)]


def audit(labels, cash0=100, cash1=100):
    counts = Counter(event['phase']+':'+event['status'] for event in labels)
    return dict(passed=True, official_transitions_reexecuted=OFFICIAL_TURNS,
                both_public_and_private_states_match_each_step=True, inventory_order_checked_each_step=True,
                final_rewards=[cash0, cash1], execution=dict(label_version=2, counts=dict(counts)))


def complete_game(records=(), labels=(), cash0=100, cash1=100):
    consumer = GameConsumer(0)
    for step in range(OFFICIAL_TURNS):
        consumer.trace_turn(turn(step, records if step == 0 else ()))
    for label in labels:
        consumer.sink(label)
    return consumer.finish(observations(100, 100), observations(cash0, cash1), ['DONE', 'DONE'], audit(labels, cash0, cash1))


def frame(step):
    return [dict(observation=dict(step=step, private=dict(value=1))),
            dict(observation=dict(private=dict(value=2)))]


def actions():
    return [dict(farmer=['PASS'], hands=[], market=[]),
            dict(farmer=['PASS'], hands=[], market=[])]


class ConsumerTests(unittest.TestCase):
    def test_choice_request_actual_effects_and_cash_are_distinct(self):
        moving = record()
        planting = record(unit=1, target=[2, 0], request=['PLANT', 'CARROT'])
        missing = record(unit=2, target=[3, 0], op='WATER', request=['WATER'])
        plant = unit_event(unit=1, request=['PLANT', 'CARROT'], position=[2, 0])
        plant['successful_plant'] = True
        labels = [unit_event(), plant, market_event(0, ['SELL', 'CARROT', 5], 'partial', 2, 5, wanted=5),
                  market_event(0, None, 'ignored', order=1), market_event(1, ['HIRE'], 'executed', 1, -3, wanted=1)]
        result = complete_game([moving, planting, missing], labels, 105, 97)
        a, b, c = result['events']
        self.assertEqual((a['selected_op'], a['request_op'], a['effective_op']), ('PLANT', 'NORTH', 'NORTH'))
        self.assertFalse(a['same_target_atomic_effect'])
        self.assertTrue(b['same_target_atomic_effect'])
        self.assertEqual((c['label_support'], c['status'], c['same_target_atomic_effect']), ('unknown', None, None))
        self.assertEqual(result['execution_support'], 'unknown')
        own, rival = result['economy']['own'], result['economy']['opponent']
        self.assertEqual((own['sales_units'], own['sales_revenue'], own['net_profit']), (2, 5, 5))
        self.assertEqual(result['market_requests']['own']['known_requested_units:SELL'], 5)
        self.assertEqual(result['market_requests']['own']['unknown_requested_units_orders:invalid'], 1)
        self.assertEqual((rival['net_profit'], result['margin'], result['score']), (-3, 8, 1.))

    def test_rewrite_blocks_request_and_does_not_invent_plant_completion(self):
        chosen = record(request=['PLANT', 'CARROT'])
        blocked = unit_event(request=['PLANT', 'CARROT'], effective=['PASS'], status='blocked', position=[1, 0])
        result = complete_game([chosen], [blocked])
        row = result['events'][0]
        self.assertEqual((row['request_op'], row['effective_op'], row['status']), ('PLANT', 'PASS', 'blocked'))
        self.assertFalse(row['same_target_atomic_effect'])
        self.assertEqual(result['economy']['own']['successful_plants'], 0)

    def test_bad_or_duplicate_event_fails_closed_and_requests_must_match(self):
        consumer = GameConsumer(0)
        consumer.sink(unit_event())
        with self.assertRaises(ValueError): consumer.sink(unit_event())
        with self.assertRaises(ValueError): consumer.trace_turn(turn(0))
        with self.assertRaises(ValueError): complete_game([record()], [unit_event(request=['SOUTH'])])
        failed = GameConsumer(0); failed.sink(unit_event(status='error'))
        with self.assertRaises(ValueError): failed.trace_turn(turn(0))
        with self.assertRaises(ValueError): failed.finish([], [], [], {})
        with self.assertRaises(ValueError): complete_game([], [market_event(0, ['SELL', 'A', 3], 'partial', 1, 5)])
        order = market_event(0, ['SELL', 'A', 3], 'partial', 1, 5, wanted=3)
        with self.assertRaises(ValueError): complete_game([], [order, order], cash0=110)

    def test_actual_execution_trace_preserves_original_error_and_invalidates_result(self):
        class OfficialFailure(Exception):
            pass
        farm = dict(money=100, unlocked_quadrants=[], hires_today=0, farmer=[0, 0], hands=[], tiles=[[None]])
        private = dict(seeds={}, shed={}, inventories=[{}])
        state = [SimpleNamespace(observation=SimpleNamespace(farms=[farm, {}]),
                                 action=dict(farmer=['PASS'], hands=[])), SimpleNamespace()]
        rules = SimpleNamespace(_farmer_position=lambda f, index: [0, 0])
        consumer = GameConsumer(0)
        tracer = ExecutionTrace(rules, consumer.sink); tracer.state = state; tracer.step = 0
        def original(*args, **kwargs):
            raise OfficialFailure('synthetic')
        with self.assertRaises(OfficialFailure):
            tracer.unit(original, farm, private, 0, ['PASS'])
        with self.assertRaises(ValueError): consumer.trace_turn(turn(0))

    def test_caller_mutation_does_not_change_joined_records(self):
        consumer = GameConsumer(0)
        r, label = record(), unit_event()
        consumer.trace_turn(turn(0, [r])); consumer.sink(label)
        r['unit_request'][0] = 'SOUTH'; label['requested'][0] = 'WEST'
        for step in range(1, OFFICIAL_TURNS): consumer.trace_turn(turn(step))
        out = consumer.finish(observations(100, 100), observations(100, 100), ['DONE', 'DONE'],
                              audit([unit_event()]))
        self.assertEqual(out['events'][0]['request_op'], 'NORTH')

    def test_comparison_stops_after_choice_or_input_divergence(self):
        choices = [candidate(0, [1, 0], 'PLANT'), candidate(1, [1, 0], 'PLANT')]
        left, right = record(candidates=choices), record(unit=1, candidates=deepcopy(choices))
        comparison = CommonHistoryComparison(0)
        comparison.observe(0, frame(0), frame(0), turn(0, [left]), turn(0, [right]), actions(), actions())
        self.assertEqual((comparison.result()['support'], comparison.result()['first_choice_disagreement']), ('observed', 1))
        self.assertEqual((comparison.result()['matched_multi_unit_opportunities'],
                          comparison.result()['matched_selected_multi_unit'],
                          comparison.result()['matched_selected_different_unit']), (1, 1, 1))
        self.assertTrue(comparison.result()['comparison_complete'])
        with self.assertRaises(ValueError): comparison.observe(1, frame(1), frame(1), turn(1), turn(1), actions(), actions())
        modified = deepcopy(left); modified['inputs']['planned']['unit_actions'] = [['PASS']]
        comp = CommonHistoryComparison(0)
        comp.observe(0, frame(0), frame(0), turn(0, [left]), turn(0, [modified]), actions(), actions())
        self.assertEqual((comp.result()['support'], comp.result()['stop_reason']), ('unknown', 'input_difference'))

    def test_exact_input_bytes_detect_order_and_numeric_type(self):
        original = record()
        for changed in (dict(b=2, a=1), dict(a=1., b=2)):
            other = deepcopy(original); other['inputs']['observation']['private'] = changed
            comp = CommonHistoryComparison(0)
            comp.observe(0, frame(0), frame(0), turn(0, [original]), turn(0, [other]), actions(), actions())
            self.assertEqual((comp.result()['matched_calls'], comp.result()['stop_reason']), (0, 'input_difference'))

    def test_non_json_or_incomplete_input_stops_comparison(self):
        bad = record(); bad['inputs']['shared_x'] = (1.,)
        comp = CommonHistoryComparison(0)
        with self.assertRaises(ValueError):
            comp.observe(0, frame(0), frame(0), turn(0, [bad]), turn(0, [bad]), actions(), actions())
        self.assertEqual(comp.result()['stop_reason'], 'invalid_input')
        with self.assertRaises(ValueError):
            comp.observe(0, frame(0), frame(0), turn(0), turn(0), actions(), actions())

    def test_joint_actions_and_frames_stop_later_comparison(self):
        comp = CommonHistoryComparison(0)
        altered = actions(); altered[1]['farmer'] = ['EAST']
        comp.observe(0, frame(0), frame(0), turn(0, [record()]), turn(0, [record()]), actions(), altered)
        self.assertEqual(comp.result()['stop_reason'], 'joint_action_difference')
        with self.assertRaises(ValueError): comp.observe(1, frame(1), frame(1), turn(1), turn(1), actions(), actions())
        comp = CommonHistoryComparison(0); other = frame(0); other[1]['observation']['private']['value'] = 3
        comp.observe(0, frame(0), other, turn(0, [record()]), turn(0, [record()]), actions(), actions())
        self.assertEqual((comp.result()['support'], comp.result()['stop_reason']), ('unknown', 'frame_difference'))

    def test_frame_ignores_framework_timer_but_preserves_other_fields_and_order(self):
        left, right = frame(0), frame(0)
        left[0]['observation']['remainingOverageTime'] = 5.
        right[0]['observation']['remainingOverageTime'] = 3.
        comp = CommonHistoryComparison(0)
        comp.observe(0, left, right, turn(0, [record()]), turn(0, [record()]), actions(), actions())
        self.assertEqual((comp.result()['support'], comp.result()['stopped']), ('observed', False))
        for changed in (dict(a=1, b=3), dict(b=2, a=1)):
            left, right = frame(0), frame(0)
            left[1]['observation']['private'] = dict(a=1, b=2)
            right[1]['observation']['private'] = changed
            comp = CommonHistoryComparison(0)
            comp.observe(0, left, right, turn(0, [record()]), turn(0, [record()]), actions(), actions())
            self.assertEqual((comp.result()['support'], comp.result()['stop_reason']), ('unknown', 'frame_difference'))

    def test_incomplete_trace_or_unverified_replay_is_rejected(self):
        consumer = GameConsumer(0); consumer.trace_turn(turn(0))
        with self.assertRaises(ValueError):
            consumer.finish(observations(100, 100), observations(100, 100), ['DONE', 'DONE'],
                            dict(passed=True, official_transitions_reexecuted=1, final_rewards=[100, 100]))

    def test_label_stream_must_match_official_execution_counts(self):
        consumer = GameConsumer(0)
        for step in range(OFFICIAL_TURNS): consumer.trace_turn(turn(step))
        forged = audit([unit_event()])
        with self.assertRaises(ValueError):
            consumer.finish(observations(100, 100), observations(100, 100), ['DONE', 'DONE'], forged)
        with self.assertRaises(ValueError): consumer.trace_turn(turn(0))

    def test_any_opportunity_and_selected_task_opportunity_have_distinct_denominators(self):
        candidates = [candidate(0, [1, 0], 'PLANT'), candidate(1, [2, 0], 'WATER'),
                      candidate(2, [2, 0], 'WATER')]
        result = complete_game([record(candidates=candidates)], [unit_event()])
        counts = result['selection_counts']
        self.assertEqual((counts['model_callbacks'], counts['model_callback_multi_unit_opportunities'],
                          counts['model_callback_selected_multi_unit_opportunities']), (1, 1, 0))
        teacher = record(candidates=candidates); teacher['source'] = 'teacher'
        counts = complete_game([teacher], [unit_event()])['selection_counts']
        self.assertEqual((counts['model_callbacks'], counts['model_callback_multi_unit_opportunities']), (0, 0))


def economic(final):
    return dict(initial_cash=100, final_cash=final, net_profit=final-100,
        successful_plants=0, seed_units_bought=0, sales_units=0, sales_revenue=0,
        ending_seed_units=0, ending_warehouse_units=0, cash_reconciled=True)


def game(seed, side, arm):
    delta = 0 if arm == 'reference' else (5 if seed == 1 else 11) + 2*side
    own, rival = economic(150+delta), economic(120+(10 if arm == 'candidate' else 0))
    margin = own['final_cash']-rival['final_cash']
    return dict(seed=seed, opponent='opponent', side=side, arm=arm,
                result=dict(actor=side, official_turns=OFFICIAL_TURNS,
                            economy=dict(own=own, opponent=rival), margin=margin,
                            score=1. if margin>0 else .5 if margin==0 else 0.))


class PanelTests(unittest.TestCase):
    def test_whole_seed_paired_economics_and_unknown_significance(self):
        rows = [game(s, side, arm) for s in (1, 2) for side in (0, 1) for arm in ('reference', 'candidate')]
        result = paired(rows, seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate')
        self.assertEqual((result['overall']['support'], result['overall']['values']['own_net_profit']), ('observed', 9))
        self.assertEqual(result['overall']['values']['opponent_net_profit'], 10)
        self.assertEqual(result['overall']['values']['margin'], -1)
        self.assertEqual((result['seed_clusters'], result['statistical_significance']), (2, 'unknown'))
        self.assertFalse(result['independently_verified_improvement'])

    def test_missing_side_is_unknown_and_duplicate_fails(self):
        rows = [game(s, side, arm) for s in (1, 2) for side in (0, 1) for arm in ('reference', 'candidate')]
        incomplete = paired(rows[:-1], seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate')
        self.assertEqual((incomplete['overall']['support'], incomplete['overall']['values']), ('unknown', None))
        self.assertEqual(incomplete['by_seed'][1]['support'], 'observed')
        self.assertEqual(incomplete['by_seed'][2]['support'], 'unknown')
        self.assertEqual(sum(p['support']=='unknown' for p in incomplete['pairs']), 1)
        with self.assertRaises(ValueError): paired(rows+[rows[0]], seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate')

    def test_opportunity_support_uses_complete_seed_groups_and_separate_pair_counts(self):
        rows = [game(s, side, arm) for s in (1, 2) for side in (0, 1) for arm in ('reference', 'candidate')]
        for row in rows:
            row['result']['selection_counts'] = dict(model_callbacks=7, model_callback_multi_unit_opportunities=6)
        comps = [dict(seed=s, opponent='opponent', side=side, reference='reference', candidate='candidate',
                      result=dict(comparison_complete=True, matched_calls=5, matched_multi_unit_opportunities=5))
                 for s in (1, 2) for side in (0, 1)]
        out = paired(rows, seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate', comparisons=comps)
        sup = out['opportunity_support']
        self.assertEqual((sup['reference']['support'], sup['candidate']['support'], sup['same_input_pairs']['support']),
                         ('supported', 'supported', 'supported'))
        self.assertEqual((sup['reference']['total'], sup['same_input_pairs']['total']), (24, 20))
        for arm_key in ('reference', 'candidate'):
            wrong = deepcopy(comps); wrong[0][arm_key] = 'unrelated_arm'
            with self.assertRaises(ValueError):
                paired(rows, seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate', comparisons=wrong)
        comps[0]['result']['matched_multi_unit_opportunities'] = 0
        comps[1]['result']['matched_multi_unit_opportunities'] = 4
        out = paired(rows, seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate', comparisons=comps)
        self.assertEqual(out['opportunity_support']['same_input_pairs']['support'], 'unknown')
        self.assertEqual(out['opportunity_support']['reference']['support'], 'supported')
        for comp in comps[2:]:
            comp['result']['matched_calls'] = 10
            comp['result']['matched_multi_unit_opportunities'] = 10
        out = paired(rows, seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate', comparisons=comps)
        self.assertEqual((out['opportunity_support']['same_input_pairs']['support'],
                          out['opportunity_support']['same_input_pairs']['total']), ('unknown', 24))
        out = paired(rows[:-1], seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate', comparisons=comps[:-1])
        self.assertEqual(out['opportunity_support']['candidate']['support'], 'unknown')
        self.assertIsNone(out['opportunity_support']['same_input_pairs']['total'])
        out = paired(rows[:-1], seeds=[1, 2], opponents=['opponent'], reference='reference', candidate='candidate', comparisons=comps)
        self.assertIsNone(out['opportunity_support']['same_input_pairs']['total'])

    def test_wrong_score_or_unreconciled_economics_rejected(self):
        row = game(1, 0, 'candidate'); row['result']['score'] = .5
        with self.assertRaises(ValueError): paired([row], seeds=[1], opponents=['opponent'], reference='reference', candidate='candidate')
        row = game(1, 0, 'candidate'); row['result']['economy']['own']['cash_reconciled'] = False
        with self.assertRaises(ValueError): paired([row], seeds=[1], opponents=['opponent'], reference='reference', candidate='candidate')


if __name__ == '__main__':
    unittest.main()
