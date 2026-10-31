"""Pure synthetic same-turn opportunity alignment; no game or model."""
from copy import deepcopy
import unittest

from decision.api import CROPS, observe
from learning.task_choice_v1.features import candidate_id, encode_candidates
from learning.task_multi_teacher_probe_v1.test_teacher import observation, configuration
from . import alignment


def make_stream(order=(1, 0), *, plant=False, plant_request=None):
    view = observe(observation(), configuration())
    obs, cfg = view.observation, view.configuration
    units = [view.own['farmer'], *view.own['hands']]
    available, actions = {0, 1}, [['PASS'], ['PASS']]
    tasks = [([0, 0], ['WATER'], 1.0), ([1, 0], ['PLANT' if plant else 'HARVEST'], 2.0)]
    records = []
    for unit in order:
        chosen_task = next(i for i, (target, _action, _priority) in enumerate(tasks)
                           if target == ([0, 0] if unit == 0 else [1, 0]))
        choices = []
        for candidate_unit in sorted(available):
            index = next(i for i, (target, _action, _priority) in enumerate(tasks)
                         if target == ([0, 0] if candidate_unit == 0 else [1, 0]))
            choices.append((2.0 if candidate_unit == unit else 1.0, candidate_unit, index))
        context = dict(observation=obs, configuration=cfg, units=units, available=available,
                       actions=actions, tasks=tasks, choices=choices, seeds={crop:1 for crop in CROPS},
                       needed_seeds={}, supply={crop:0 for crop in CROPS}, projected_shed={},
                       remaining=100, today=12, last_plant_age=4, final_day=False)
        inputs, _lookup = encode_candidates(context)
        target, intended, _priority = tasks[chosen_task]
        chosen = candidate_id(unit, target, intended[0])
        request = (deepcopy(plant_request) if intended[0] == 'PLANT' and plant_request is not None
                   else ['NORTH'] if intended[0] == 'PLANT' else list(intended))
        records.append(dict(inputs=inputs, teacher_label=chosen, chosen=chosen, unit=unit,
                            target=list(target), op=intended[0], source='teacher',
                            unit_request=request, intended_action=list(intended)))
        actions[unit] = deepcopy(request)
        available.remove(unit)
        tasks.pop(chosen_task)
    schema = {key: records[0]['inputs'][key] for key in alignment.SCHEMA_KEYS}
    return obs, cfg, records, schema


def run(left, right, *, obs=None, cfg=None, schema=None, guard=lambda: None):
    if obs is None:
        obs, cfg, _unused, schema = make_stream()
    return alignment.align_turn(obs, cfg, left, right, schema=schema, guard=guard)


class AlignmentTests(unittest.TestCase):
    def test_same_complete_prefix_and_target_features_are_supported(self):
        obs, cfg, records, schema = make_stream()
        result = run(records, deepcopy(records), obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unsupported'], result['unknown']), (2, 0, 0))
        self.assertEqual((result['supported_target_features_equal'],
                          result['supported_target_features_differ']), (2, 0))
        self.assertTrue(all(type(value) is int for value in result.values()))
        self.assertNotIn(records[0]['chosen'], repr(result))

    def test_same_context_absent_target_is_unsupported_then_diverged_prefix_unknown(self):
        obs, cfg, left, schema = make_stream(order=(1, 0))
        _obs, _cfg, right, _schema = make_stream(order=(0, 1))
        right[0]['inputs']['candidates'] = [candidate for candidate in right[0]['inputs']['candidates']
                                           if candidate['id'] != left[0]['chosen']]
        result = run(left, right, obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unsupported'], result['unknown']), (0, 1, 1))

    def test_equal_current_candidates_do_not_realign_after_prior_choice_diverges(self):
        obs, cfg, left, schema = make_stream(order=(1, 0))
        _obs, _cfg, right, _schema = make_stream(order=(0, 1))
        result = run(left, right, obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unsupported'], result['unknown']), (1, 0, 1))

    def test_same_context_different_shared_features_still_reports_support(self):
        obs, cfg, left, schema = make_stream()
        right = deepcopy(left)
        names = right[0]['inputs']['shared_feature_names']
        right[0]['inputs']['shared_x'][names.index('planning.tasks_remaining')] += 1
        result = run(left, right, obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unknown']), (2, 0))
        self.assertEqual((result['supported_target_features_equal'],
                          result['supported_target_features_differ']), (1, 1))

    def test_prior_reservation_order_mismatch_cannot_realign_later(self):
        obs, cfg, left, schema = make_stream()
        right = deepcopy(left)
        left[0]['inputs']['planned']['projected_shed'] = {'WHEAT':1, 'CARROT':1}
        right[0]['inputs']['planned']['projected_shed'] = {'CARROT':1, 'WHEAT':1}
        result = run(left, right, obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unknown']), (0, 2))

    def test_prior_unit_request_order_and_inventory_order_are_not_ignored(self):
        obs, cfg, left, schema = make_stream()
        right = deepcopy(left)
        right[1]['inputs']['planned']['unit_actions'].reverse()
        with self.assertRaises(alignment.AlignmentError):
            run(left, right, obs=obs, cfg=cfg, schema=schema)
        left[0]['inputs']['inventory_item_order'][0].append('WHEAT')
        with self.assertRaises(alignment.AlignmentError):
            run(left, deepcopy(left), obs=obs, cfg=cfg, schema=schema)

    def test_frozen_controller_reservations_require_nonnegative_integers(self):
        obs, cfg, records, schema = make_stream()
        for resource in ('seeds', 'needed_seeds', 'supply', 'projected_shed'):
            for invalid in (-1, 0.5, 1.0, True):
                with self.subTest(resource=resource, invalid=invalid):
                    bad = deepcopy(records)
                    bad[0]['inputs']['planned'][resource]['WHEAT'] = invalid
                    with self.assertRaises(alignment.AlignmentError):
                        run(records, bad, obs=obs, cfg=cfg, schema=schema)

    def test_frozen_controller_available_units_have_pass_requests(self):
        obs, cfg, records, schema = make_stream()
        for record_index in (0, 1):
            with self.subTest(record_index=record_index):
                bad = deepcopy(records)
                available = bad[record_index]['inputs']['planned']['available_units']
                bad[record_index]['inputs']['planned']['unit_actions'][available[0]] = ['NORTH']
                with self.assertRaises(alignment.AlignmentError):
                    run(records, bad, obs=obs, cfg=cfg, schema=schema)

    def test_previous_moving_plant_leaves_later_prefix_unknown(self):
        obs, cfg, records, schema = make_stream(plant=True, plant_request=['NORTH'])
        result = run(records, deepcopy(records), obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unknown']), (1, 1))
        explicit = deepcopy(records)
        explicit[0]['intended_action'] = ['PLANT', 'MELON']
        result = run(explicit, deepcopy(explicit), obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unknown']), (1, 1))

    def test_previous_explicit_plant_requires_identical_crop(self):
        obs, cfg, left, schema = make_stream(plant=True, plant_request=['PLANT', 'MELON'])
        result = run(left, deepcopy(left), obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unknown']), (2, 0))
        right = deepcopy(left)
        right[0]['unit_request'] = ['PLANT', 'WHEAT']
        right[1]['inputs']['planned']['unit_actions'][1] = ['PLANT', 'WHEAT']
        result = run(left, right, obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((result['supported'], result['unknown']), (1, 1))

    def test_empty_turn_and_no_terminal_callback_are_explicit(self):
        obs, cfg, records, schema = make_stream()
        empty = run([], [], obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((empty['teacher_empty_turn'], empty['terminal_empty_turn']), (1, 1))
        missing = run(records, [], obs=obs, cfg=cfg, schema=schema)
        self.assertEqual((missing['teacher_empty_turn'], missing['terminal_empty_turn'], missing['unknown']), (0, 1, 2))

    def test_duplicate_unauthorized_and_oversized_inputs_fail(self):
        obs, cfg, records, schema = make_stream()
        duplicate = deepcopy(records); duplicate[0]['inputs']['candidates'].append(
            deepcopy(duplicate[0]['inputs']['candidates'][0]))
        oversized = deepcopy(records); oversized[0]['inputs']['candidates'].extend(
            [deepcopy(oversized[0]['inputs']['candidates'][0]) for _ in range(255)])
        extra = deepcopy(records); extra[0]['teacher_score'] = 1
        missing_intent = deepcopy(records); del missing_intent[0]['intended_action']
        for bad in (duplicate, oversized, extra, missing_intent, [deepcopy(records[0])]*33,
                    [deepcopy(records[0]), deepcopy(records[0])]):
            with self.assertRaises(alignment.AlignmentError):
                run(bad, records, obs=obs, cfg=cfg, schema=schema)
        dirty = deepcopy(obs); dirty['farms'][1]['private'] = {'seeds': {'MELON':1}}
        with self.assertRaises(alignment.AlignmentError):
            run(records, records, obs=dirty, cfg=cfg, schema=schema)

    def test_resource_guard_exception_propagates(self):
        obs, cfg, records, schema = make_stream()
        signal = MemoryError('synthetic stop')
        def guard(): raise signal
        with self.assertRaises(MemoryError) as caught:
            run(records, records, obs=obs, cfg=cfg, schema=schema, guard=guard)
        self.assertIs(caught.exception, signal)


if __name__ == '__main__':
    unittest.main()
