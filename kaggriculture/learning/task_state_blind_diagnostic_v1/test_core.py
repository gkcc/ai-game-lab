"""Synthetic-only tests; no private artifacts, inference or games."""
from copy import deepcopy
import io
import unittest

import numpy as np

from learning.task_disagreement_trace_v1.test_extract import fixture
from learning.task_student_collection_v1.test_sampling import TASK, CROP, choice, trace
from learning.task_state_residual_v1 import core as state
from . import diagnostic, extract, sampling


class SamplingTests(unittest.TestCase):
    def select(self, raw):
        sampler = sampling.RankedStream(io.BytesIO(raw), role='A05', opponent='market',
                                        side=0, guard=lambda: None)
        self.assertEqual(sampler.scan(task_sha256=TASK, crop_sha256=CROP)['lines'], 719)
        return sampler.samples()

    def test_steps_are_blind_to_callback_count_teacher_and_outcome(self):
        steps = (1, 3, 7, 13, 21, 44, 62, 91, 117)
        one = {step: [choice('student', 'student', ('student', 'teacher'))] for step in steps}
        many = {step: [choice('teacher', 'student', ('student', 'teacher'))] * (1 + step % 7)
                for step in steps}
        def altered(doc):
            doc['synthetic_reward'] = -doc['step']
        left, right = self.select(trace(one)), self.select(trace(many, alter=altered))
        expect = [step for _, step in sorted((sampling.step_rank('A05', 'market', 0, step), step)
                                             for step in steps)[:4]]
        self.assertEqual([row['step'] for row in left[0]], expect)
        self.assertEqual([row['step'] for row in right[0]], expect)
        for row in right[0]:
            self.assertEqual(row['ordinal'], min(range(len(many[row['step']])),
                key=lambda i: (sampling.ordinal_rank('A05','market',0,row['step'],i), i)))

    def test_all_windows_single_use_and_untrusted_trace_failure(self):
        values = {start: [choice()] for start, _ in sampling.WINDOWS}
        sampler = sampling.RankedStream(io.BytesIO(trace(values)), role='A06',
                                        opponent='round2', side=1, guard=lambda: None)
        with self.assertRaises(RuntimeError): sampler.samples()
        sampler.scan(task_sha256=TASK, crop_sha256=CROP)
        windows = sampler.samples()
        self.assertEqual([len(window) for window in windows], [1] * 6)
        self.assertTrue(all(row['role'] == 'A06' for window in windows for row in window))
        with self.assertRaises(RuntimeError): sampler.samples()
        for raw in (trace({1: [choice()] * 33}), trace({1: [choice()]}) + b'x'):
            bad = sampling.RankedStream(io.BytesIO(raw), role='A05', opponent='market',
                                        side=0, guard=lambda: None)
            with self.assertRaises(ValueError): bad.scan(task_sha256=TASK, crop_sha256=CROP)
            with self.assertRaises(RuntimeError): bad.samples()


class ExtractionTests(unittest.TestCase):
    def test_actual_role_and_single_candidate_are_preserved(self):
        args = fixture()
        args['role'] = 'A05'
        args['inputs']['candidates'] = args['inputs']['candidates'][:1]
        args['teacher_label'] = args['chosen'] = args['inputs']['candidates'][0]['id']
        result = extract.extract_example(**args)
        self.assertEqual((result.item.role, result.item.pool, result.item.target, result.student_index),
                         ('A05', 'new_dev', 0, 0))
        self.assertEqual(result.item.candidates.shape, (1, 98))
        self.assertFalse(result.item.candidates.flags.writeable)
        self.assertEqual(repr(result), '<PrivateBlindExample>')

    def test_rejects_invalid_identity_precision_and_private_input(self):
        variants = []
        args = fixture(); args['role'] = 'A09'; variants.append(args)
        args = fixture(); args['role'] = 'A06'; args['inputs']['candidates'].append(deepcopy(args['inputs']['candidates'][0])); variants.append(args)
        args = fixture(); args['role'] = 'A06'; args['inputs']['shared_x'][0] = float(np.nextafter(np.float64(args['inputs']['shared_x'][0]), np.inf)); variants.append(args)
        args = fixture(); args['role'] = 'A06'; args['inputs']['observation']['info'] = {'seed': 'secret'}; variants.append(args)
        for args in variants:
            with self.subTest(case=variants.index(args)), self.assertRaises(ValueError) as caught:
                extract.extract_example(**args)
            self.assertNotIn('secret', str(caught.exception))


class RealAssessmentTests(unittest.TestCase):
    def test_true_assessor_accepts_internal_aliases_and_preserves_duplicates(self):
        records = []
        meta = None
        for role, opponent, side, window, single in (
                ('A05','market',0,0,False), ('A06','round2',1,1,False),
                ('A05','round2',0,5,True)):
            args = fixture(); args['role'] = role
            if single:
                args['inputs']['candidates'] = args['inputs']['candidates'][:1]
            args['teacher_label'] = args['chosen'] = args['inputs']['candidates'][0]['id']
            example = extract.extract_example(**args)
            meta = args['metadata']
            records.append(diagnostic.PrivateRecord(role, opponent, side, window, example.item,
                                                      example.student_index))
        class Network:
            def score(self, shared, candidates):
                assert shared.shape == (275,) and candidates.shape[1] == 98
                return np.arange(len(candidates), 0, -1, dtype=np.float64)
        guard = lambda: None
        groups = diagnostic.score(records, Network(), guard=guard)
        report = diagnostic.assess(records, groups, mean=np.zeros(373, np.float64),
            scale=np.ones(373, np.float64), params=state.initial_params(123), metadata=meta, guard=guard)
        self.assertEqual(report['overall']['groups'], 3)
        self.assertEqual(report['overall']['teacher_pair']['both_match'], 3)
        self.assertEqual(report['overall']['changed'], 0)
        self.assertEqual(report['single_candidate_groups'], 1)
        self.assertEqual(report['duplicates']['raw']['repeated_groups'], 2)
        self.assertEqual(report['by_role']['A05']['groups'], 2)
        self.assertEqual(report['by_opponent']['round2']['groups'], 2)
        self.assertEqual(report['by_side']['1']['groups'], 1)
        self.assertEqual(report['by_window']['5']['groups'], 1)
        self.assertEqual(report['coverage_by_case']['A05']['market']['0']['groups'], 1)
        self.assertEqual(report['coverage_by_case']['A05']['market']['0']['windows']['0'], 1)
        self.assertEqual(report['coverage_by_case']['A06']['market']['0']['groups'], 0)
        self.assertEqual(sum(report['coverage_by_case']['A06']['market']['0']['windows'].values()), 0)
        self.assertFalse(report['independent_evaluation'])
        self.assertEqual(report['historical_screen'], 'failed')

    def test_nonzero_residual_four_teacher_outcomes_and_empty_population(self):
        records, meta = [], None
        for index, target in enumerate((0, 1, 0, 1)):
            args = fixture(); args['role'] = ('A05','A06')[index % 2]
            candidates = args['inputs']['candidates']
            self.assertEqual([c['x'][0] for c in candidates], [0, 1])
            args['chosen'], args['teacher_label'] = candidates[0]['id'], candidates[target]['id']
            example = extract.extract_example(**args); meta = args['metadata']
            records.append(diagnostic.PrivateRecord(args['role'], ('market','round2')[index % 2],
                index % 2, index, example.item, example.student_index))
        class Network:
            def __init__(self): self.values = iter((1.0, 0.1, 0.1, 1.0))
            def score(self, _shared, _candidates):
                return np.array([0.0, -next(self.values)], dtype=np.float64)
        params = state.Params(np.zeros((275,16),np.float32), np.zeros((98,16),np.float32),
                              np.zeros(16,np.float32), np.zeros(16,np.float32))
        params.wc[0,0], params.v[0] = 1, 0.5
        guard = lambda: None
        groups = diagnostic.score(records, Network(), guard=guard)
        common = dict(mean=np.zeros(373,np.float64), scale=np.ones(373,np.float64),
                      params=params, metadata=meta, guard=guard)
        result = diagnostic.assess(records, groups, **common)
        self.assertEqual(result['overall']['teacher_pair'],
            dict(both_match=1,improved=1,regressed=1,neither_match=1))
        self.assertEqual((result['overall']['changed'],result['overall']['op_flips']), (2,2))
        self.assertEqual((result['overall']['baseline_correct'],result['overall']['residual_correct']), (2,2))
        empty = diagnostic.assess([], [], **common)
        self.assertEqual(empty['overall']['groups'], 0)
        self.assertEqual(empty['overall']['coverage'], 'unknown')
        self.assertIsNone(empty['overall']['baseline_nll'])


if __name__ == '__main__':
    unittest.main()
