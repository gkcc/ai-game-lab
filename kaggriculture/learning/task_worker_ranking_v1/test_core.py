"""Tests use only synthetic arrays and fake scoring objects."""

from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash
from learning.task_worker_ranking_v1 import core


SHARED_NAMES = ['step', 'config.boardSize'] + ['shared.' + str(i) for i in range(273)]
FEATURE_NAMES = ['unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y',
                 'task.distance'] + ['task.op.' + op for op in OPS]
FEATURE_NAMES += ['unused.' + str(i) for i in range(98 - len(FEATURE_NAMES))]


def fixture(cluster_sizes=(5, 5, 5, 10), *, teacher_second=False, other_same_target=False):
    shared, all_rows, all_ops, offsets, labels, clusters = [], [], [], [0], [], []
    for cluster, size in enumerate(cluster_sizes):
        for _ in range(size):
            s = np.zeros(275, dtype=np.float32)
            s[0], s[1], s[2] = 2, 5, int(cluster == 3)
            if teacher_second:
                specs = [(0, 0, 1, 1, 0, 'PLANT'), (1, 0, 0, 1, 0, 'PLANT')]
            else:
                specs = [(0, 0, 0, 1, 0, 'PLANT'), (1, 0, 1, 1, 0, 'PLANT')]
            specs.append((2, 2, 2, 1 if other_same_target else 2,
                          0 if other_same_target else 1, 'WATER'))
            rows = np.zeros((len(specs), 98), dtype=np.float32)
            for i, (unit, ux, uy, x, y, op) in enumerate(specs):
                fields = {'unit.index': unit, 'unit.x': ux, 'unit.y': uy,
                          'target.x': x, 'target.y': y,
                          'task.distance': abs(ux - x) + abs(uy - y)}
                fields.update({'task.op.' + name: int(op == name) for name in OPS})
                for key, value in fields.items():
                    rows[i, FEATURE_NAMES.index(key)] = value
                all_ops.append(OPS.index(op))
            shared.append(s)
            all_rows.append(rows)
            offsets.append(offsets[-1] + len(rows))
            labels.append(int(teacher_second))
            clusters.append(cluster)
    return dict(shared=np.asarray(shared, dtype=np.float32),
                candidates=np.concatenate(all_rows).astype(np.float32),
                offsets=np.asarray(offsets, dtype=np.int64),
                teacher_choice_label=np.asarray(labels, dtype=np.int64),
                ops=np.asarray(all_ops, dtype=np.uint8),
                cluster_ids=np.asarray(clusters, dtype=np.uint32),
                shared_feature_names=list(SHARED_NAMES), feature_names=list(FEATURE_NAMES))


class FakeScorer:
    def __init__(self, rule):
        self.rule = rule
        self.calls = 0
        self.metadata = dict(schema_version=SCHEMA_VERSION,
                             schema_sha256=schema_hash(FEATURE_NAMES, SHARED_NAMES),
                             shared_feature_names=list(SHARED_NAMES),
                             feature_names=list(FEATURE_NAMES), ops=list(OPS))

    def score(self, shared, candidates):
        self.calls += 1
        if shared.flags.writeable or candidates.flags.writeable or candidates.shape != (3, 98):
            raise ValueError('Expected read-only complete group')
        return np.asarray(self.rule(shared), dtype=np.float64)


class PoisonScorer:
    @property
    def metadata(self):
        raise AssertionError('coverage failure must not inspect the scorer')


class RankingTests(unittest.TestCase):
    def test_gate_and_equal_cluster_weighting_with_full_candidates(self):
        data = fixture()
        a = FakeScorer(lambda s: [3, 0, 1] if s[2] else [1, 0, 3])
        b = FakeScorer(lambda s: [0, 3, 1] if s[2] else [3, 0, 1])
        self.assertTrue(core.assess_coverage(**data))
        before = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
        result = core.evaluate_private(**data, scorer_a=a, scorer_b=b)
        metrics = result['metrics']
        self.assertEqual(result['coverage'], 'sufficient')
        self.assertEqual((a.calls, b.calls), (25, 25))
        # Group pooling would be 10/25 and 15/25, not the correct 1/4 and 3/4.
        self.assertEqual(metrics['overall_teacher_top1'],
                         {'model_a': Fraction(1, 4), 'model_b': Fraction(3, 4)})
        self.assertEqual(metrics['conditional_teacher_top1'],
                         {'model_a': Fraction(1), 'model_b': Fraction(3, 4)})
        self.assertEqual(metrics['overall_same_target_plant'],
                         {'model_a': Fraction(1, 4), 'model_b': Fraction(1)})
        for key, expected in before.items():
            np.testing.assert_array_equal(data[key], expected)
        public = core.project_public(result)
        self.assertEqual(public, {
            'coverage': 'sufficient',
            'conditional_teacher_top1': {'model_a_minus_model_b_band': '>+20pp'},
            'global_guard': 'fail',
        })

    def test_insufficient_gate_does_not_inspect_either_scorer(self):
        for sizes in ((5, 5, 5, 4), (7, 7, 7), (5, 5, 5, 5, 5)):
            with self.subTest(sizes=sizes):
                data = fixture(sizes)
                self.assertFalse(core.assess_coverage(**data))
                result = core.evaluate_private(**data, scorer_a=PoisonScorer(), scorer_b=PoisonScorer())
                self.assertEqual(result, {'coverage': 'insufficient'})
                self.assertEqual(core.project_public(result), {'coverage': 'insufficient'})

    def test_a_source_cluster_with_too_few_eligible_groups_does_not_pass(self):
        data = fixture((5, 5, 5, 10))
        # Switch all labels in one source cluster to its WATER candidate.
        data['teacher_choice_label'][:5] = 2
        self.assertFalse(core.assess_coverage(**data))
        self.assertEqual(core.evaluate_private(**data, scorer_a=PoisonScorer(),
                                               scorer_b=PoisonScorer()), {'coverage': 'insufficient'})

    def test_exact_score_ties_use_full_candidate_order(self):
        data = fixture((5, 5, 5, 5), teacher_second=True)
        a = FakeScorer(lambda _: [1, 1, 0])
        b = FakeScorer(lambda _: [0, 1, 0])
        m = core.evaluate_private(**data, scorer_a=a, scorer_b=b)['metrics']
        self.assertEqual(m['conditional_teacher_top1']['model_a'], 0)
        self.assertEqual(m['overall_teacher_top1']['model_a'], 0)
        self.assertEqual(m['overall_same_target_plant']['model_a'], 1)
        self.assertEqual(m['overall_teacher_top1']['model_b'], 1)

    def test_same_target_with_other_operation_is_not_same_plant_task(self):
        data = fixture((5, 5, 5, 5), other_same_target=True)
        a = FakeScorer(lambda _: [0, 1, 2])
        b = FakeScorer(lambda _: [2, 1, 0])
        m = core.evaluate_private(**data, scorer_a=a, scorer_b=b)['metrics']
        self.assertEqual(m['overall_same_target_plant']['model_a'], 0)
        self.assertEqual(m['conditional_teacher_top1']['model_a'], 0)
        self.assertEqual(m['overall_teacher_top1']['model_a'], 0)

    def test_invalid_inputs_fail_before_scoring(self):
        for kind in ('label', 'feature', 'width', 'nonfinite', 'one_hot', 'cluster_dtype'):
            with self.subTest(kind=kind):
                data = fixture((5, 5, 5, 5))
                kwargs = {}
                if kind == 'label':
                    data['teacher_choice_label'][0] = 3
                elif kind == 'feature':
                    data['feature_names'][data['feature_names'].index('target.x')] = 'missing'
                elif kind == 'width':
                    kwargs['max_candidates_per_group'] = 2
                elif kind == 'nonfinite':
                    data['candidates'][0, 20] = np.nan
                elif kind == 'one_hot':
                    data['candidates'][0, FEATURE_NAMES.index('task.op.PLANT')] = 0
                else:
                    data['cluster_ids'] = data['cluster_ids'].astype(bool)
                with self.assertRaises(ValueError):
                    core.evaluate_private(**data, scorer_a=PoisonScorer(), scorer_b=PoisonScorer(), **kwargs)

    def test_bad_model_schema_and_nonfinite_scores_fail_closed(self):
        data = fixture((5, 5, 5, 5))
        a = FakeScorer(lambda _: [1, 0, 0])
        b = FakeScorer(lambda _: [1, 0, 0])
        b.metadata['feature_names'][0] = 'wrong'
        with self.assertRaises(ValueError):
            core.evaluate_private(**data, scorer_a=a, scorer_b=b)
        self.assertEqual((a.calls, b.calls), (0, 0))
        b = FakeScorer(lambda _: [float('nan'), 0, 0])
        with self.assertRaises(ValueError):
            core.evaluate_private(**data, scorer_a=a, scorer_b=b)

    def test_scores_with_large_integers_that_collapse_in_float64_are_rejected(self):
        # These unequal integers become equal when converted to float64: a
        # silent conversion would change the candidate selected by argmax.
        values = np.asarray([2**53, 2**53 + 1, 0], dtype=np.uint64)
        self.assertEqual(int(np.argmax(values)), 1)
        self.assertEqual(int(np.argmax(values.astype(np.float64))), 0)

        class RawScorer(FakeScorer):
            def score(self, shared, candidates):
                self.calls += 1
                return values

        data = fixture((5, 5, 5, 5))
        a, b = RawScorer(lambda _: None), FakeScorer(lambda _: [1, 0, 0])
        with self.assertRaises(ValueError):
            core.evaluate_private(**data, scorer_a=a, scorer_b=b)
        self.assertEqual((a.calls, b.calls), (1, 0))

    def test_only_native_float32_and_float64_score_arrays_are_accepted(self):
        class TypedScorer(FakeScorer):
            def __init__(self, dtype):
                super().__init__(lambda _: None)
                self.dtype = dtype

            def score(self, shared, candidates):
                self.calls += 1
                return np.asarray([1, 0, 0], dtype=self.dtype)

        data = fixture((5, 5, 5, 5))
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                result = core.evaluate_private(**data, scorer_a=TypedScorer(dtype),
                                               scorer_b=TypedScorer(dtype))
                self.assertEqual(result['coverage'], 'sufficient')
        rejected = [np.float16, np.int64, np.uint64, np.bool_]
        if np.dtype(np.longdouble) not in (np.dtype('float32'), np.dtype('float64')):
            rejected.append(np.longdouble)
        for dtype in rejected:
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    core.evaluate_private(**data, scorer_a=TypedScorer(dtype),
                                          scorer_b=TypedScorer(np.float64))

    def test_global_guard_checks_both_metrics_at_inclusive_boundary(self):
        base = {name: {'model_a': Fraction(1, 2), 'model_b': Fraction(1, 2)}
                for name in core.METRICS}
        for metric in ('overall_teacher_top1', 'overall_same_target_plant'):
            with self.subTest(metric=metric):
                values = {name: item.copy() for name, item in base.items()}
                values[metric]['model_a'] = Fraction(9, 20)  # -5 pp exactly
                public = core.project_public({'coverage': 'sufficient', 'metrics': values})
                self.assertEqual(public['global_guard'], 'pass')
                values[metric]['model_a'] = Fraction(449, 1000)  # below -5 pp
                public = core.project_public({'coverage': 'sufficient', 'metrics': values})
                self.assertEqual(public['global_guard'], 'fail')
                self.assertEqual(set(public), {'coverage', 'conditional_teacher_top1', 'global_guard'})
                self.assertEqual(set(public['conditional_teacher_top1']), {'model_a_minus_model_b_band'})

    def test_rate_and_paired_difference_boundaries_are_exact(self):
        # Rate boundaries remain valid private inputs, but their individual
        # bands must not appear in the public result.
        for number in (0, Fraction(1, 5), Fraction(2, 5), Fraction(3, 5), Fraction(4, 5), 1):
            item = {'model_a': Fraction(number), 'model_b': Fraction(number)}
            private = {'coverage': 'sufficient', 'metrics': {name: item.copy() for name in core.METRICS}}
            public = core.project_public(private)
            self.assertEqual(public['conditional_teacher_top1'],
                             {'model_a_minus_model_b_band': '[-5,+5]pp'})
            self.assertEqual(public['global_guard'], 'pass')
        for number in (Fraction(-1, 1000), Fraction(1001, 1000)):
            item = {'model_a': number, 'model_b': Fraction(0)}
            private = {'coverage': 'sufficient', 'metrics': {name: item.copy() for name in core.METRICS}}
            with self.assertRaises(ValueError):
                core.project_public(private)
        pairs = [(-Fraction(201, 1000), '<-20pp'), (-Fraction(1, 5), '[-20,-5)pp'),
                 (-Fraction(1, 20), '[-5,+5]pp'), (Fraction(1, 20), '[-5,+5]pp'),
                 (Fraction(1, 5), '(+5,+20]pp'), (Fraction(201, 1000), '>+20pp')]
        for number, expected in pairs:
            self.assertEqual(core._difference_band(number), expected)


if __name__ == '__main__':
    unittest.main()
