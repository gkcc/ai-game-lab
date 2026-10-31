"""Synthetic single-stream sampling tests; no private source is opened."""
import hashlib
import io
import json
import unittest
from unittest.mock import patch

from learning.task_student_continuation_v1 import samples as previous
from learning.task_student_collection_v1.test_sampling import TASK, CROP, choice, trace
from . import sampling


def fake_scanner(stream, *, task_sha256, crop_sha256, guard):
    """Small 719-line scanner substitute; semantic integration is tested below."""
    size = 0
    for step in range(719):
        guard()
        raw = stream.readline(previous.MAX_LINE + 1)
        size += len(raw)
        doc = json.loads(raw)
        if (doc['step'] != step or doc['task_checkpoint_sha256'] != task_sha256
                or doc['checkpoint_sha256'] != crop_sha256):
            raise ValueError('Synthetic scanner identity mismatch')
    guard()
    if stream.read(1):
        raise ValueError('Synthetic scanner found extra data')
    return dict(lines=719, decoded_bytes=size)


def selector(raw, *, mode='train', role='A03', opponent='market', side=0, guard=lambda: None):
    return sampling.RankedStream(io.BytesIO(raw), mode=mode, role=role, opponent=opponent,
                                 side=side, guard=guard)


def run(raw, *, mode='train', role='A03', opponent='market', side=0):
    stream = selector(raw, mode=mode, role=role, opponent=opponent, side=side)
    with patch.object(previous, 'scan', side_effect=fake_scanner):
        summary = stream.scan(task_sha256=TASK, crop_sha256=CROP)
    return summary, stream.samples()


def rank(role, opponent, side, step, ordinal):
    payload = json.dumps([role, opponent, side, step, ordinal], ensure_ascii=True,
                         separators=(',', ':')).encode('ascii')
    return hashlib.sha256(b'kaggriculture.task-disagreement-trace-v1.rank\x00' + payload).digest()


class SamplingTests(unittest.TestCase):
    def test_train_uses_only_disagreements_and_top_four_distinct_steps(self):
        steps = (1, 3, 7, 13, 44, 62, 91, 117)
        records = {step: [choice('student', 'student', ('student', 'teacher')),
                          choice(), choice('student', 'teacher', ('student', 'teacher'))]
                   for step in steps}
        expected = sorted(min((rank('A03', 'market', 0, step, ordinal), step, ordinal)
                              for ordinal in (1, 2)) for step in steps)[:4]
        raw = trace(records)
        summary, actual = run(raw)
        self.assertEqual(summary, {'lines': 719, 'decoded_bytes': len(raw)})
        self.assertEqual([(item['step'], item['ordinal']) for item in actual[0]],
                         [(step, ordinal) for _digest, step, ordinal in expected])
        self.assertEqual(len({item['step'] for item in actual[0]}), 4)
        self.assertTrue(all(item['teacher'] != item['chosen'] for item in actual[0]))
        self.assertTrue(all(set(item) == sampling.FIELDS for item in actual[0]))
        self.assertEqual(actual[1:], [[] for _ in range(5)])
        self.assertEqual(run(raw)[1], actual)

    def test_diagnostic_rank_ignores_teacher_chosen_operation_and_outcome(self):
        steps = (0, 1, 4, 10, 11, 20, 42, 119)
        baseline = {step: [choice('student', 'student', ('student', 'teacher')),
                           choice('teacher', 'teacher', ('student', 'teacher'))]
                    for step in steps}
        flipped = {step: [choice('teacher', 'student', ('student', 'teacher')),
                          choice('student', 'teacher', ('student', 'teacher'))]
                   for step in steps}
        def alter(doc):
            doc['synthetic_reward'] = 1e6 - doc['step']
            for record in doc['records']:
                record['inputs']['synthetic_operation'] = 'different'
        first = run(trace(baseline), mode='diagnostic', role='A09')[1]
        second = run(trace(flipped, alter=alter), mode='diagnostic', role='A09')[1]
        coordinates = lambda windows: [[(x['step'], x['ordinal']) for x in window] for window in windows]
        self.assertEqual(coordinates(first), coordinates(second))
        expected = sorted(min((rank('A09', 'market', 0, step, ordinal), step, ordinal)
                              for ordinal in (0, 1)) for step in steps)[:4]
        self.assertEqual(coordinates(first)[0], [(step, ordinal) for _digest, step, ordinal in expected])
        self.assertTrue(all(x['teacher'] == x['chosen'] for x in first[0]))
        self.assertTrue(all(x['teacher'] != x['chosen'] for x in second[0]))

    def test_window_endpoints_and_empty_windows(self):
        for endpoint in (0, 1):
            steps = [bounds[endpoint] for bounds in sampling.WINDOWS]
            _summary, windows = run(trace({step: [choice()] for step in steps}), role='A04', side=1)
            self.assertEqual([[item['step'] for item in window] for window in windows],
                             [[step] for step in steps])
            self.assertTrue(all(item['role'] == 'A04' and item['side'] == 1
                                for window in windows for item in window))
        self.assertEqual(run(trace())[1], [[] for _ in range(6)])

    def test_digest_collisions_break_ties_by_step_then_ordinal(self):
        records = {step: [choice(), choice()] for step in (0, 1, 2, 3, 4, 5)}
        stream = selector(trace(records))
        with patch.object(sampling, 'coordinate_rank', return_value=b'x' * 32), \
             patch.object(previous, 'scan', side_effect=fake_scanner):
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        self.assertEqual([(item['step'], item['ordinal']) for item in stream.samples()[0]],
                         [(0, 0), (1, 0), (2, 0), (3, 0)])

    def test_scan_and_samples_are_single_use_and_cannot_unlock_early(self):
        stream = selector(trace({1: [choice()]}))
        with self.assertRaises(RuntimeError): stream.samples()
        with self.assertRaises(RuntimeError): stream.readline(1)
        def during_scan(adapter, **kwargs):
            with self.assertRaises(RuntimeError): stream.samples()
            return fake_scanner(adapter, **kwargs)
        with patch.object(previous, 'scan', side_effect=during_scan):
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        self.assertEqual(len(stream.samples()[0]), 1)
        with self.assertRaises(RuntimeError): stream.samples()
        with self.assertRaises(RuntimeError): stream.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError): stream.read(1)

    def test_bad_eof_missing_lines_or_failed_scanner_never_unlock(self):
        raw = trace({1: [choice()]})
        for broken in (raw + b'x', b'\n'.join(raw.splitlines()[:-1]) + b'\n'):
            stream = selector(broken)
            with patch.object(previous, 'scan', side_effect=fake_scanner), self.assertRaises(ValueError):
                stream.scan(task_sha256=TASK, crop_sha256=CROP)
            with self.assertRaises(RuntimeError): stream.samples()
            with self.assertRaises(RuntimeError): stream.scan(task_sha256=TASK, crop_sha256=CROP)
        stream = selector(raw)
        def incomplete(_stream, **_kwargs):
            return dict(lines=719, decoded_bytes=len(raw))
        with patch.object(previous, 'scan', side_effect=incomplete), self.assertRaises(ValueError):
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError): stream.samples()

    def test_guard_and_stream_errors_fail_closed(self):
        signal = MemoryError('synthetic guard failure')
        count = [0]
        def guard():
            count[0] += 1
            if count[0] == 100: raise signal
        stream = selector(trace({1: [choice()]}), guard=guard)
        with patch.object(previous, 'scan', side_effect=fake_scanner), self.assertRaises(MemoryError) as caught:
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        self.assertIs(caught.exception, signal)
        with self.assertRaises(RuntimeError): stream.samples()
        enabled = [False]
        def materialization_guard():
            if enabled[0]: raise signal
        stream = selector(trace({1: [choice()]}), guard=materialization_guard)
        with patch.object(previous, 'scan', side_effect=fake_scanner):
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        enabled[0] = True
        with self.assertRaises(MemoryError): stream.samples()
        enabled[0] = False
        with self.assertRaises(RuntimeError): stream.samples()
        class FailAtEnd(io.BytesIO):
            def read(self, _size=-1): raise OSError('synthetic EOF failure')
        stream = sampling.RankedStream(FailAtEnd(trace()), mode='train', role='A03',
                                       opponent='market', side=0, guard=lambda: None)
        with patch.object(previous, 'scan', side_effect=fake_scanner), self.assertRaises(OSError):
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError): stream.samples()

    def test_bad_mode_role_coordinate_and_oversized_line(self):
        for mode, role, opponent, side in (('train', 'A09', 'market', 0),
                ('diagnostic', 'A03', 'market', 0), ('other', 'A03', 'market', 0),
                ('train', 'A03', 'other', 0), ('train', 'A03', 'market', True)):
            with self.subTest(mode=mode, role=role), self.assertRaises(ValueError):
                selector(b'', mode=mode, role=role, opponent=opponent, side=side)
        stream = selector(b'x' * (previous.MAX_LINE + 1) + b'\n')
        with patch.object(previous, 'scan', side_effect=fake_scanner), self.assertRaises(ValueError):
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError): stream.samples()

    def test_unchanged_scanner_checks_real_synthetic_budget_and_identity(self):
        good = selector(trace({1: [choice()]}))
        self.assertEqual(good.scan(task_sha256=TASK, crop_sha256=CROP)['lines'], 719)
        self.assertEqual(len(good.samples()[0]), 1)
        def corrupt(doc):
            if doc['step'] == 400:
                doc['turn_model_counts']['task_forwards'] += 1
            if doc['step'] >= 400:
                doc['model_counts']['task_forwards'] += 1
        for altered in (trace({1: [choice()]}, alter=corrupt),
                        trace({1: [choice()]}, alter=lambda doc: doc.update(
                            task_checkpoint_sha256='c' * 64) if doc['step'] == 718 else None)):
            bad = selector(altered)
            with self.assertRaises(ValueError): bad.scan(task_sha256=TASK, crop_sha256=CROP)
            with self.assertRaises(RuntimeError): bad.samples()

    def test_overlong_callback_turn_rejected_even_when_old_scanner_accepts(self):
        raw = trace({1: [choice() for _ in range(33)]})
        self.assertEqual(previous.scan(io.BytesIO(raw), task_sha256=TASK, crop_sha256=CROP)['lines'], 719)
        stream = selector(raw)
        with self.assertRaises(ValueError): stream.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError): stream.samples()
        with self.assertRaises(ValueError): sampling.coordinate_rank('A03', 'market', 0, 1, 32)

    def test_unselected_oversized_candidate_list_does_not_pre_filter(self):
        identifiers = ('student', 'teacher') + tuple('synthetic' + str(i) for i in range(255))
        records = {1: [choice(ids=identifiers)]}
        records.update({step: [choice()] for step in (2, 3, 4, 5)})
        stream = selector(trace(records))
        def forced(_role, _opponent, _side, step, _ordinal):
            return bytes([255 if step == 1 else 0]) * 32
        with patch.object(sampling, 'coordinate_rank', side_effect=forced):
            stream.scan(task_sha256=TASK, crop_sha256=CROP)
        self.assertEqual([item['step'] for item in stream.samples()[0]], [2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
