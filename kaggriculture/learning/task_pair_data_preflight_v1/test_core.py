"""Synthetic-only ragged input, split, overlap and collision tests."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash
from learning.task_pair_data_preflight_v1 import core


SHARED_NAMES = ['step', 'config.boardSize'] + ['shared.' + str(i) for i in range(273)]
FEATURE_NAMES = ['unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y',
                 'task.distance'] + ['task.op.' + op for op in OPS]
FEATURE_NAMES += ['unused.' + str(i) for i in range(98 - len(FEATURE_NAMES))]
SCHEMA = {'schema_version': SCHEMA_VERSION, 'schema_sha256': schema_hash(FEATURE_NAMES, SHARED_NAMES),
          'shared_feature_names': SHARED_NAMES, 'feature_names': FEATURE_NAMES, 'ops': list(OPS)}


def record(marker, label, private_cluster, *, negative_zero=False):
    shared = np.zeros(275, np.float32)
    shared[0], shared[1], shared[2] = 2, 5, marker
    if negative_zero:
        shared[3] = -0.0
    # Two units compete for the same PLANT target; WATER is a distinct task.
    specs = [(0, 0, 0, 1, 0, 'PLANT'), (1, 0, 1, 1, 0, 'PLANT'),
             (2, 2, 2, 2, 1, 'WATER')]
    rows = np.zeros((len(specs), 98), np.float32)
    operations, identity = [], []
    for i, (unit, ux, uy, x, y, op) in enumerate(specs):
        fields = {'unit.index': unit, 'unit.x': ux, 'unit.y': uy, 'target.x': x, 'target.y': y,
                  'task.distance': abs(ux - x) + abs(uy - y)}
        fields.update({'task.op.' + name: int(op == name) for name in OPS})
        for key, value in fields.items():
            rows[i, FEATURE_NAMES.index(key)] = value
        operations.append(OPS.index(op))
        identity.append([unit, x, y])
    if negative_zero:
        rows[0, FEATURE_NAMES.index('unused.0')] = -0.0
    return shared, rows, label, operations, identity, private_cluster


def make_arrays(records, *, conditional=False):
    offsets = [0]
    for item in records:
        offsets.append(offsets[-1] + len(item[1]))
    base = {'shared': np.asarray([item[0] for item in records], np.float32),
            'candidates': np.concatenate([item[1] for item in records]),
            'offsets': np.asarray(offsets, np.int64)}
    if conditional:
        base.update(teacher_choice_label=np.asarray([r[2] for r in records], np.int64),
                    ops=np.asarray([o for r in records for o in r[3]], np.uint8),
                    seed=np.asarray([r[5] for r in records], np.uint32))
    else:
        base.update(labels=np.asarray([r[2] for r in records], np.int64),
                    selected_ops=np.asarray([r[3][r[2]] for r in records], np.int8),
                    candidate_ops=np.asarray([o for r in records for o in r[3]], np.int8),
                    candidate_identity=np.asarray([x for r in records for x in r[4]], np.int16),
                    seeds=np.asarray([r[5] for r in records], np.int64))
    return base


def fixture():
    train = [record(0, 0, 1), record(0, 1, 2), record(1, 0, 3), record(3, 2, 4)]
    train += [record(10 + i, 0, i) for i in range(5, 9)]
    original = [record(0, 0, 21, negative_zero=True), record(1, 0, 22)]
    first = [record(99, 0, 31), record(0, 1, 32)]
    second = [record(1, 0, 33), record(2, 2, 34)]
    arrays = [make_arrays(train), make_arrays(original), make_arrays(first, conditional=True),
              make_arrays(second, conditional=True)]
    views = [core.RaggedView.from_cache(arrays[0], SCHEMA), core.RaggedView.from_cache(arrays[1], SCHEMA),
             core.RaggedView.from_conditional(arrays[2], SCHEMA),
             core.RaggedView.from_conditional(arrays[3], SCHEMA)]
    roles = {'train': list(range(1, 9)), 'teacher_dev': [21, 22], 'conditional_dev': [31, 32, 33, 34]}
    return arrays, views, roles


def run(views, roles, check=lambda: None):
    return core.analyze([views[0]], [views[1]], [views[2], views[3]], roles, check=check)


class PreflightTests(unittest.TestCase):
    def test_sources_coverage_overlap_masks_and_no_mutation(self):
        arrays, views, roles = fixture()
        before = [{key: value.copy() for key, value in item.items()} for item in arrays]
        result = run(views, roles)
        self.assertEqual(repr(result), '<PrivatePreflightResult>')
        self.assertEqual(result.coverage['train']['aggregate']['groups'], 8)
        self.assertEqual(result.coverage['train']['aggregate']['qualifying_plant_groups'], 7)
        self.assertEqual(result.coverage['train']['aggregate']['operations']['WATER'],
                         {'groups': 1, 'clusters_with_groups': 1})
        self.assertEqual(result.coverage['teacher_dev']['aggregate']['qualifying_per_cluster_sorted'], [1, 1])
        self.assertEqual(result.coverage['conditional_dev']['aggregate']['operations']['PLANT'],
                         {'groups': 3, 'clusters_with_groups': 3})
        self.assertEqual(len(result.coverage['conditional_dev']['sources']), 2)
        self.assertEqual(result.overlap['teacher_dev'][0]['all'],
                         {'unmatched': 0, 'same_label_only': 1, 'label_conflict': 1})
        self.assertEqual(result.overlap['conditional_dev'][0]['all'],
                         {'unmatched': 1, 'same_label_only': 0, 'label_conflict': 1})
        np.testing.assert_array_equal(result.masks['teacher_dev'][0]['seen_in_train'], [True, True])
        np.testing.assert_array_equal(result.masks['teacher_dev'][0]['label_conflict'], [True, False])
        np.testing.assert_array_equal(result.masks['conditional_dev'][1]['seen_in_train'], [True, False])
        np.testing.assert_array_equal(result.masks['conditional_dev'][1]['qualifying_plant'], [True, False])
        self.assertEqual(set(result.masks['teacher_dev'][0]),
                         {'seen_in_train', 'label_conflict', 'qualifying_plant'})
        self.assertEqual({len(mask) for mask in result.masks['conditional_dev'][1].values()}, {2})
        for prior, after in zip(before, arrays):
            for key in prior:
                np.testing.assert_array_equal(prior[key], after[key])

    def test_signed_zero_is_normalized_and_label_conflict_takes_precedence(self):
        _, views, roles = fixture()
        self.assertEqual(core._fingerprint(views[0], 0), core._fingerprint(views[1], 0))
        # Independently construct the historic NumericEquivalence byte format.
        shared = views[1].shared[0].astype('<f4', copy=True)
        candidates = views[1].candidates[:3].astype('<f4', copy=True)
        shared[shared == 0] = 0.0
        candidates[candidates == 0] = 0.0
        payload = np.asarray([3], dtype='<i8').tobytes() + shared.tobytes() + candidates.tobytes()
        self.assertEqual(core._fingerprint(views[1], 0), hashlib.sha256(payload).digest())
        result = run(views, roles)
        self.assertTrue(result.masks['teacher_dev'][0]['seen_in_train'][0])
        self.assertTrue(result.masks['teacher_dev'][0]['label_conflict'][0])

    def test_digest_collision_never_creates_a_false_match(self):
        _, views, roles = fixture()
        with patch.object(core, '_fingerprint', return_value=b'synthetic-collision'):
            result = run(views, roles)
        np.testing.assert_array_equal(result.masks['conditional_dev'][0]['seen_in_train'], [False, True])
        np.testing.assert_array_equal(result.masks['conditional_dev'][0]['label_conflict'], [False, True])

    def test_candidate_order_matters_and_unordered_groups_are_rejected(self):
        arrays, views, roles = fixture()
        original = core._fingerprint(views[1], 0)
        order = [1, 0, 2]
        for key in ('candidates', 'candidate_ops', 'candidate_identity'):
            arrays[1][key][:3] = arrays[1][key][order]
        arrays[1]['labels'][0] = 1
        self.assertNotEqual(original, core._fingerprint(views[1], 0))
        with self.assertRaises(ValueError):
            run(views, roles)

    def test_empty_group_bad_label_and_wrong_dtypes_fail_closed(self):
        for case in ('empty', 'label', 'cluster_dtype', 'candidate_nan', 'op_encoding'):
            with self.subTest(case=case):
                arrays, views, roles = fixture()
                if case == 'empty':
                    arrays[1]['offsets'][1] = arrays[1]['offsets'][0]
                elif case == 'label':
                    arrays[1]['labels'][0] = 3
                elif case == 'cluster_dtype':
                    arrays[2]['seed'] = arrays[2]['seed'].astype(bool)
                    views[2] = core.RaggedView.from_conditional(arrays[2], SCHEMA)
                elif case == 'candidate_nan':
                    arrays[1]['candidates'][0, 20] = np.nan
                else:
                    arrays[1]['candidates'][0, FEATURE_NAMES.index('task.op.PLANT')] = 0
                with self.assertRaises(ValueError):
                    run(views, roles)

    def test_cluster_roles_are_disjoint_and_complete(self):
        _, views, roles = fixture()
        overlapping = deepcopy(roles)
        overlapping['teacher_dev'][0] = overlapping['train'][0]
        with self.assertRaises(ValueError):
            run(views, overlapping)
        missing = deepcopy(roles)
        missing['conditional_dev'][-1] = 35
        with self.assertRaises(ValueError):
            run(views, missing)
        bad_bool = deepcopy(roles)
        bad_bool['train'][0] = True
        with self.assertRaises(ValueError):
            run(views, bad_bool)

    def test_schema_field_or_candidate_identity_mismatch_fails(self):
        arrays, views, roles = fixture()
        bad_schema = deepcopy(SCHEMA)
        bad_schema['feature_names'][0] = 'other'
        with self.assertRaises(ValueError):
            core.RaggedView.from_cache(arrays[0], bad_schema)
        forged = list(views)
        forged[0] = replace(views[0], schema_digest='synthetic-invalid')
        with self.assertRaises(ValueError):
            run(forged, roles)
        arrays[1]['candidate_identity'][0, 0] = 1
        with self.assertRaises(ValueError):
            run(views, roles)

    def test_resource_callback_before_each_comparison_and_cancellation(self):
        _, views, roles = fixture()
        calls, comparisons = [0], []
        actual = core._equal_group
        def check():
            calls[0] += 1
        def compare(*args):
            comparisons.append(calls[0])
            return actual(*args)
        with patch.object(core, '_equal_group', side_effect=compare):
            run(views, roles, check)
        self.assertTrue(comparisons)
        self.assertTrue(all(a < b for a, b in zip(comparisons, comparisons[1:])))

        calls[0] = 0
        def stop():
            calls[0] += 1
            if calls[0] == 5:
                raise TimeoutError('synthetic stop')
        with self.assertRaises(TimeoutError):
            run(views, roles, stop)

    def test_resource_callback_precedes_each_equality_block(self):
        _, views, _ = fixture()
        checks, comparisons = [0], []
        actual = np.array_equal
        def check():
            checks[0] += 1
        def compare(*args, **kwargs):
            comparisons.append(checks[0])
            return actual(*args, **kwargs)
        with patch.object(core.np, 'array_equal', side_effect=compare):
            self.assertTrue(core._equal_group(views[0], 0, views[1], 0, check))
        self.assertEqual(comparisons, [1, 2])


if __name__ == '__main__':
    unittest.main()
