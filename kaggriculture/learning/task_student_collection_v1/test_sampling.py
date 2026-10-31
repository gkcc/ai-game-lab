"""Synthetic, in-memory checks for fixed trace sampling and validation."""
import gzip
import hashlib
import io
import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from learning.task_student_continuation_v1 import samples as previous
from learning.task_student_collection_v1 import sampling


TASK, CROP = 'a' * 64, 'b' * 64


def choice(teacher='teacher', chosen='student', ids=('student', 'third', 'teacher', 'later')):
    return dict(source='callback', teacher_label=teacher, chosen=chosen,
                inputs=dict(candidates=[dict(id=item) for item in ids]))


def trace(records_by_step=None, *, alter=None):
    records_by_step = records_by_step or {}
    total = {key: 0 for key in previous.COUNT_KEYS}
    lines = []
    for step in range(719):
        records = records_by_step.get(step, [])
        turn = {key: 0 for key in previous.COUNT_KEYS}
        turn['controller_calls'] = 1
        turn['task_forwards'] = len(records)
        turn['task_rows'] = sum(len(r['inputs']['candidates']) for r in records)
        total = {key: total[key] + turn[key] for key in total}
        doc = dict(schema_version='task-choice-v1', step=step,
                   task_checkpoint_sha256=TASK, checkpoint_sha256=CROP,
                   load_errors=0, inference_errors=0, fallbacks=0,
                   model_counts=dict(total), turn_model_counts=turn,
                   records=records, crop_records=[])
        if alter is not None:
            alter(doc)
        lines.append(json.dumps(doc, separators=(',', ':')).encode() + b'\n')
    return b''.join(lines)


def compressed_stream(raw):
    return gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(raw, mtime=0)), mode='rb')


def adapter(raw, *, guard=lambda: None, role='F01', opponent='market', side=0):
    return sampling.RankedStream(compressed_stream(raw), role=role, opponent=opponent,
                                 side=side, guard=guard)


def run(selector):
    summary = selector.scan(task_sha256=TASK, crop_sha256=CROP)
    return summary, selector.selected()


class RankedSamplingTests(unittest.TestCase):
    def test_fixed_domain_rank_uses_all_callbacks_not_only_first(self):
        records = {step: [choice(), choice(ids=('teacher', 'student', 'first', 'second'))]
                   for step in (1, 3, 7, 13, 62, 117)}
        expected = min((hashlib.sha256(b'kaggriculture.task-student-collection-v1.rank\x00'
                     + json.dumps(['F01', 'market', 0, step, ordinal], ensure_ascii=True,
                                  separators=(',', ':')).encode('ascii')).digest(), step, ordinal)
                       for step in records for ordinal in (0, 1))
        first = adapter(trace(records))
        with self.assertRaises(RuntimeError):
            first.selected()
        summary, selected = run(first)
        self.assertEqual((selected[0]['step'], selected[0]['ordinal']), expected[1:])
        self.assertNotEqual(expected[1:], (1, 0))
        self.assertEqual(summary['lines'], 719)
        self.assertGreater(summary['decoded_bytes'], 0)
        self.assertEqual(len(summary['selection_work']), 3)  # previous scanner accounting
        self.assertEqual(selected[1:], [None] * 5)
        second = adapter(trace(records))
        self.assertEqual(run(second)[1], selected)
        expected_third = 'third' if expected[2] == 0 else 'first'
        self.assertEqual(selected[0]['third_candidate'], expected_third)

    def test_each_six_window_start_and_end_and_previous_state(self):
        for boundary in (0, 1):
            with self.subTest(boundary=boundary):
                steps = [pair[boundary] for pair in sampling.WINDOWS]
                _summary, selected = run(adapter(trace({step: [choice()] for step in steps})))
                self.assertEqual([item['step'] for item in selected], steps)
                for item in selected:
                    if item['step'] == 0:
                        self.assertIsNone(item['previous_document'])
                    else:
                        prev = item['previous_document']
                        self.assertEqual(prev['step'], item['step'] - 1)
                        self.assertEqual(prev['model_counts']['controller_calls'], item['step'])
                        self.assertEqual(set(prev), set(sampling.PREVIOUS_FIELDS))

    def test_empty_windows_and_single_use(self):
        selector = adapter(trace())
        with self.assertRaises(RuntimeError):
            selector.readline(1)
        _summary, selected = run(selector)
        self.assertEqual(selected, [None] * 6)
        with self.assertRaises(RuntimeError):
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError):
            selector.read(1)

    def test_no_third_candidate_and_previous_task_budget(self):
        records = {0: [choice('same', 'same', ('same',))],
                   1: [choice(ids=('student', 'teacher'))]}
        _summary, selected = run(adapter(trace(records)))
        self.assertEqual((selected[0]['step'], selected[0]['ordinal']), (1, 0))
        self.assertIsNone(selected[0]['third_candidate'])
        previous_doc = selected[0]['previous_document']
        self.assertEqual(previous_doc['step'], 0)
        self.assertEqual(previous_doc['model_counts']['task_forwards'], 1)
        self.assertEqual(previous_doc['model_counts']['task_rows'], 1)
        self.assertEqual(previous_doc['task_checkpoint_sha256'], TASK)
        self.assertEqual(previous_doc['checkpoint_sha256'], CROP)

    def test_ranked_result_connects_to_student_continuation(self):
        from learning.task_dagger_once_v1.policy import LIMITS
        from learning.task_request_gate_v1.policy import _Budget
        from learning.task_student_continuation_v1.compare import compare_turn
        from learning.task_student_continuation_v1.test_compare import (
            SyntheticBounded, configuration, fake_encode, observation,
        )
        with patch('learning.task_choice_v1.agent.encode_candidates', side_effect=fake_encode):
            student, cfg = SyntheticBounded(), configuration()
            student(observation(), cfg)
            before = deepcopy(student.last)
            action = student(observation(1), cfg)
            after = deepcopy(student.last)
            # Ensure the selected disagreement is on the second turn, where
            # the comparator must restore the compact previous budget.
            for record in before['records']:
                record['teacher_label'] = record['chosen']
            docs, counts = [before, after], deepcopy(after['model_counts'])
            for step in range(2, 719):
                doc = deepcopy(after)
                doc['step'], doc['records'], doc['crop_records'] = step, [], []
                doc['turn_model_counts'] = {key: 0 for key in previous.COUNT_KEYS}
                doc['turn_model_counts']['controller_calls'] = 1
                counts['controller_calls'] += 1
                doc['model_counts'] = deepcopy(counts)
                docs.append(doc)
            raw = b''.join((json.dumps(doc, separators=(',', ':')) + '\n').encode() for doc in docs)
            _summary, selected = run(adapter(raw))
            item = selected[0]
            self.assertEqual(item['step'], 1)
            outcome = compare_turn(item['document'], observation(1), cfg, action, item['ordinal'],
                task_sha256=TASK, crop_sha256=CROP, game_limits=_Budget(LIMITS).limits,
                agent_factory=SyntheticBounded, guard=lambda: None,
                previous_document=item['previous_document'])
            self.assertEqual(outcome.status, 'same')

    def test_full_old_scan_failure_never_unlocks_selected(self):
        # A real mismatch late in the old scan must defeat an early selection.
        def corrupt(doc):
            if doc['step'] == 718:
                doc['task_checkpoint_sha256'] = 'c' * 64
        selector = adapter(trace({1: [choice()]}, alter=corrupt))
        with self.assertRaises(ValueError):
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError):
            selector.selected()
        with self.assertRaises(RuntimeError):
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        # The trailing-byte check happens after all 719 input lines.
        selector = adapter(trace({1: [choice()]}) + b'x')
        with self.assertRaises(ValueError):
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError):
            selector.selected()

    def test_oversized_line_is_rejected_without_selection(self):
        selector = adapter(b'x' * (previous.MAX_LINE + 1) + b'\n')
        with self.assertRaises(ValueError):
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError):
            selector.selected()

    def test_guard_and_stream_resource_errors_propagate_unchanged(self):
        for kind in (OSError, MemoryError, TimeoutError):
            with self.subTest(kind=kind.__name__):
                signal = kind('synthetic resource stop')
                def guard():
                    raise signal
                selector = adapter(trace(), guard=guard)
                with self.assertRaises(kind) as caught:
                    selector.scan(task_sha256=TASK, crop_sha256=CROP)
                self.assertIs(caught.exception, signal)
                with self.assertRaises(RuntimeError):
                    selector.selected()
        class FailingRead(io.BytesIO):
            def read(self, _size=-1):
                raise signal
        signal = OSError('synthetic final read stop')
        selector = sampling.RankedStream(FailingRead(trace()), role='F01', opponent='market',
                                         side=0, guard=lambda: None)
        with self.assertRaises(OSError) as caught:
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        self.assertIs(caught.exception, signal)
        with self.assertRaises(RuntimeError):
            selector.selected()
        class FailingLine(io.BytesIO):
            def readline(self, _size=-1):
                raise signal
        signal = TimeoutError('synthetic line read stop')
        selector = sampling.RankedStream(FailingLine(trace()), role='F01', opponent='market',
                                         side=0, guard=lambda: None)
        with self.assertRaises(TimeoutError) as caught:
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        self.assertIs(caught.exception, signal)

    def test_ranks_ignore_noncoordinate_fields_and_separate_public_sources(self):
        records = {3: [choice()], 9: [choice()], 119: [choice()]}
        original = run(adapter(trace(records)))[1][0]
        def alter(doc):
            doc['unrelated_synthetic_value'] = -10000 * doc['step']
        altered = run(adapter(trace(records, alter=alter)))[1][0]
        self.assertEqual((original['step'], original['ordinal']), (altered['step'], altered['ordinal']))
        values = [sampling.coordinate_rank('F01', 'market', 0, 3, 0),
                  sampling.coordinate_rank('F02', 'market', 0, 3, 0),
                  sampling.coordinate_rank('F01', 'round2', 0, 3, 0),
                  sampling.coordinate_rank('F01', 'market', 1, 3, 0)]
        self.assertEqual(len(set(values)), len(values))
        with self.assertRaises(ValueError):
            sampling.coordinate_rank('F01', 'market', True, 3, 0)


if __name__ == '__main__':
    unittest.main()
