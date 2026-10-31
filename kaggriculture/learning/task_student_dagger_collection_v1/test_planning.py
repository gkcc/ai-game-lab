"""Synthetic checks for outcome-blind full-pool plan freezing."""
from copy import deepcopy
import json
import unittest

import numpy as np

from learning.task_student_collection_v1.test_dataset import inputs
from learning.task_student_dagger_collection_v1 import planning, sampling


MEAN = np.zeros(373, dtype=np.float64)
SCALE = np.ones(373, dtype=np.float64)
BASE_INPUT = inputs()
SCHEMA = {key: BASE_INPUT[key] for key in
          ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')}


def candidate(window=0, *, offset=0, value=1.0, teacher=0, student=1, count=3):
    step = sampling.WINDOWS[window][0] + offset
    data = inputs(value=value, count=count)
    ids = [item['id'] for item in data['candidates']]
    third = next((identifier for identifier in ids if identifier not in (ids[teacher], ids[student])), None)
    record = dict(source='callback', teacher_label=ids[teacher], chosen=ids[student], inputs=data)
    before = None if step == 0 else {'step': step - 1, 'model_counts': {'synthetic': 1}}
    return dict(step=step, ordinal=0, third_candidate=third,
                document=dict(step=step, records=[record]), previous_document=before)


def cells(mapping):
    for role, opponent, side, window in planning.CELL_ORDER:
        yield dict(role=role, opponent=opponent, side=side, window=window,
                   candidates=mapping.get((role, opponent, side, window), []))


def create(mapping, *, mean=MEAN, scale=SCALE, guard=lambda: None):
    return planning.create_plan(cells(mapping), schema=SCHEMA, mean=mean, scale=scale, guard=guard)


def find(plan, key):
    return next(entry for entry in plan['entries'] if tuple(entry[k] for k in
                ('role', 'opponent', 'side', 'window')) == key)


class PlanningTests(unittest.TestCase):
    def test_fixed_order_prioritizes_training_and_blocks_cross_split_duplicate(self):
        train = ('T01', 'market', 0, 0)
        development = ('V01', 'round2', 1, 0)
        mapping = {train: [candidate(value=7)], development: [candidate(value=7)]}
        plan = create(mapping)
        self.assertEqual(len(plan['entries']), 144)
        self.assertEqual(find(plan, train)['status'], 'selected')
        self.assertEqual(find(plan, development)['status'], 'all_duplicate')
        self.assertEqual(plan['summary']['selected_count'], 1)
        self.assertEqual(plan['summary']['selected_by_role']['T01'], 1)
        self.assertEqual(plan['summary']['selected_by_role']['V01'], 0)
        wrong = list(cells(mapping))
        wrong[0], wrong[1] = wrong[1], wrong[0]
        with self.assertRaises(ValueError):
            planning.create_plan(wrong, schema=SCHEMA, mean=MEAN, scale=SCALE, guard=lambda: None)

    def test_transitive_float32_collision_discards_entire_conflicting_component(self):
        a, b, c = ('T01', 'market', 0, 0), ('T02', 'market', 0, 0), ('V01', 'market', 0, 0)
        mu = MEAN.copy()
        mu[0] = 100_000_000
        mapping = {a: [candidate(value=1.0)], b: [candidate(value=1.0 + 2**-25)],
                   c: [candidate(value=2.0, teacher=1, student=0)]}
        plan = create(mapping, mean=mu)
        self.assertEqual([find(plan, key)['status'] for key in (a, b, c)], ['all_conflict'] * 3)
        self.assertEqual(plan['summary']['conflicting_components'], 1)
        self.assertEqual(plan['summary']['conflicting_candidates'], 3)
        self.assertEqual(plan['summary']['selected_count'], 0)

    def test_selects_first_nondeduplicated_ranked_candidate_and_freezes_roster(self):
        earlier, later = ('T01', 'market', 0, 0), ('T01', 'market', 0, 1)
        options = [candidate(1, offset=i, value=10+i) for i in range(4)]
        options.sort(key=lambda x: sampling.coordinate_rank(*later[:3], x['step'], x['ordinal']))
        # First ranked option duplicates the prior selection; the next is unique.
        options[0] = candidate(1, offset=options[0]['step']-120, value=7)
        mapping = {earlier: [candidate(value=7)], later: list(reversed(options))}
        plan = create(mapping)
        entry = find(plan, later)
        self.assertEqual(entry['status'], 'selected')
        self.assertEqual(entry['selected_index'], 1)
        self.assertEqual(entry['step'], options[1]['step'])
        restored = planning.verify_entry(entry, list(reversed(options)), schema=SCHEMA,
                                        mean=MEAN, scale=SCALE, guard=lambda: None)
        self.assertEqual((restored['step'], restored['ordinal']), (entry['step'], entry['ordinal']))
        # A changed nonselected candidate alters the ordered roster and stops.
        altered = deepcopy(options)
        altered[-1]['document']['records'][0]['inputs']['shared_x'][0] += 1
        with self.assertRaises(ValueError):
            planning.verify_entry(entry, altered, schema=SCHEMA, mean=MEAN, scale=SCALE,
                                  guard=lambda: None)
        altered = deepcopy(options)
        altered[1]['previous_document']['model_counts']['synthetic'] += 1
        with self.assertRaises(ValueError):
            planning.verify_entry(entry, altered, schema=SCHEMA, mean=MEAN, scale=SCALE,
                                  guard=lambda: None)

    def test_no_third_is_frozen_and_empty_or_duplicate_is_not_replaced(self):
        key = ('T01', 'market', 0, 0)
        other = ('T01', 'market', 1, 0)
        one = candidate(value=2, count=2)
        plan = create({key: [one], other: [deepcopy(one)]})
        first, duplicate = find(plan, key), find(plan, other)
        self.assertEqual(first['status'], 'selected')
        self.assertIsNone(first['third_index'])
        self.assertEqual(duplicate['status'], 'all_duplicate')
        self.assertIsNone(planning.verify_entry(duplicate, [deepcopy(one)], schema=SCHEMA,
                                                mean=MEAN, scale=SCALE, guard=lambda: None))
        empty = find(plan, ('V02', 'round2', 1, 5))
        self.assertEqual(empty['status'], 'empty')
        self.assertIsNone(planning.verify_entry(empty, [], schema=SCHEMA,
                                                mean=MEAN, scale=SCALE, guard=lambda: None))

    def test_nonzero_callback_ordinal_is_frozen_and_recovered(self):
        key = ('T01', 'market', 0, 0)
        item = candidate()
        item['document']['records'].insert(0, {'source': 'callback', 'chosen': 'irrelevant'})
        item['ordinal'] = 1
        entry = find(create({key: [item]}), key)
        self.assertEqual(entry['ordinal'], 1)
        result = planning.verify_entry(entry, [item], schema=SCHEMA, mean=MEAN,
                                       scale=SCALE, guard=lambda: None)
        self.assertIs(result, item)

    def test_private_plan_excludes_source_data_and_is_reproducible(self):
        key = ('T01', 'market', 0, 0)
        item = candidate()
        item['document']['records'][0]['inputs']['observation'] = {'private': 'SYNTHETIC_PRIVATE_SENTINEL'}
        item['document']['info'] = {'seed': 'SYNTHETIC_SEED_SENTINEL'}
        mapping = {key: [item]}
        first, second = create(mapping), create(mapping)
        self.assertEqual(first, second)
        encoded = json.dumps(first, sort_keys=True)
        self.assertNotIn('SYNTHETIC_PRIVATE_SENTINEL', encoded)
        self.assertNotIn('SYNTHETIC_SEED_SENTINEL', encoded)
        self.assertNotIn('third_candidate', encoded)
        self.assertNotIn('observation', encoded)

    def test_every_frozen_window_can_be_verified_on_a_second_pass(self):
        selected = ('T01', 'market', 0, 0)
        duplicate = ('V01', 'round2', 0, 0)
        mapping = {selected: [candidate(value=5)], duplicate: [candidate(value=5)]}
        plan = create(mapping)
        selected_count = 0
        for entry in plan['entries']:
            key = tuple(entry[k] for k in ('role', 'opponent', 'side', 'window'))
            item = planning.verify_entry(entry, mapping.get(key, []), schema=SCHEMA,
                                         mean=MEAN, scale=SCALE, guard=lambda: None)
            self.assertEqual(item is not None, entry['status'] == 'selected')
            selected_count += item is not None
        self.assertEqual(selected_count, plan['summary']['selected_count'])
        changed = deepcopy(mapping[duplicate])
        changed[0]['document']['records'][0]['inputs']['shared_x'][0] += 1
        with self.assertRaises(ValueError):
            planning.verify_entry(find(plan, duplicate), changed, schema=SCHEMA,
                                  mean=MEAN, scale=SCALE, guard=lambda: None)

    def test_bad_schema_numeric_values_width_and_grid_fail_closed(self):
        key = ('T01', 'market', 0, 0)
        bad_schema = dict(SCHEMA, schema_sha256='bad')
        with self.assertRaises(ValueError):
            planning.create_plan(cells({}), schema=bad_schema, mean=MEAN, scale=SCALE,
                                 guard=lambda: None)
        item = candidate()
        item['document']['records'][0]['inputs']['shared_x'][0] = float('nan')
        with self.assertRaises(ValueError):
            create({key: [item]})
        with self.assertRaises(ValueError):
            create({key: [candidate(count=257)]})
        item = candidate()
        item['third_candidate'] = 'not-the-first-third'
        with self.assertRaises(ValueError):
            create({key: [item]})
        with self.assertRaises(ValueError):
            create({key: [candidate(), candidate()]})  # repeated step
        with self.assertRaises(ValueError):
            planning.create_plan([], schema=SCHEMA, mean=MEAN, scale=SCALE, guard=lambda: None)
        all_cells = list(cells({}))
        with self.assertRaises(ValueError):
            planning.create_plan(all_cells + [all_cells[0]], schema=SCHEMA, mean=MEAN,
                                 scale=SCALE, guard=lambda: None)

    def test_resource_exception_propagates_and_no_partial_plan_is_returned(self):
        for kind in (OSError, MemoryError, TimeoutError):
            with self.subTest(kind=kind.__name__):
                signal = kind('synthetic resource stop')
                def guard():
                    raise signal
                with self.assertRaises(kind) as caught:
                    planning.create_plan(cells({}), schema=SCHEMA, mean=MEAN, scale=SCALE,
                                         guard=guard)
                self.assertIs(caught.exception, signal)


if __name__ == '__main__':
    unittest.main()
