"""Pure in-memory checks of fixed ranked DAgger trace sampling."""
from copy import deepcopy
import hashlib
import io
import json
import unittest
from unittest.mock import patch

from learning.task_student_continuation_v1 import samples as previous
from learning.task_student_collection_v1.test_sampling import (
    TASK, CROP, choice, compressed_stream, trace,
)
from learning.task_student_dagger_collection_v1 import sampling


def adapter(raw, *, guard=lambda: None, role='T01', opponent='market', side=0):
    return sampling.RankedStream(compressed_stream(raw), role=role, opponent=opponent,
                                 side=side, guard=guard)


def scan(selector):
    summary = selector.scan(task_sha256=TASK, crop_sha256=CROP)
    return summary, selector.candidates()


class DaggerSamplingTests(unittest.TestCase):
    def test_four_ranked_distinct_steps_and_best_callback_per_step(self):
        steps = (1, 3, 7, 13, 44, 62, 91, 117)
        records = {step: [choice(ids=('teacher', 'student', 'third0')),
                          choice(ids=('student', 'third1', 'teacher')),
                          choice('same', 'same', ('same',))] for step in steps}
        # Independently calculate the public domain-separated ordering.
        oracle = []
        for step in steps:
            options = []
            for ordinal in (0, 1):
                payload = json.dumps(['T01', 'market', 0, step, ordinal], ensure_ascii=True,
                                     separators=(',', ':')).encode('ascii')
                digest = hashlib.sha256(b'kaggriculture.task-student-dagger-collection-v1.rank\x00'
                                        + payload).digest()
                options.append((digest, step, ordinal))
            oracle.append(min(options))
        expected = sorted(oracle)[:4]
        selector = adapter(trace(records))
        with self.assertRaises(RuntimeError):
            selector.candidates()
        summary, actual = scan(selector)
        self.assertEqual([(item['step'], item['ordinal']) for item in actual[0]],
                         [(step, ordinal) for _, step, ordinal in expected])
        self.assertEqual(len({item['step'] for item in actual[0]}), 4)
        self.assertNotEqual(actual[0][0]['step'], min(steps))
        for item in actual[0]:
            self.assertEqual(item['third_candidate'], 'third' + str(item['ordinal']))
        self.assertEqual(actual[1:], [[] for _ in range(5)])
        self.assertEqual(summary['lines'], 719)
        self.assertEqual(len(summary['selection_work']), 3)  # previous scanner accounting
        self.assertEqual(scan(adapter(trace(records)))[1], actual)

    def test_all_six_window_boundaries_and_compact_previous(self):
        for endpoint in (0, 1):
            with self.subTest(endpoint=endpoint):
                steps = [pair[endpoint] for pair in sampling.WINDOWS]
                _summary, windows = scan(adapter(trace({step: [choice()] for step in steps})))
                self.assertEqual([[item['step'] for item in window] for window in windows],
                                 [[step] for step in steps])
                for window in windows:
                    item = window[0]
                    if item['step'] == 0:
                        self.assertIsNone(item['previous_document'])
                    else:
                        before = item['previous_document']
                        self.assertEqual(before['step'], item['step'] - 1)
                        self.assertEqual(before['model_counts']['controller_calls'], item['step'])
                        self.assertEqual(set(before), set(sampling.PREVIOUS_FIELDS))

    def test_empty_window_no_third_and_previous_budget(self):
        records = {0: [choice('same', 'same', ('same',))],
                   1: [choice(ids=('student', 'teacher'))]}
        _summary, windows = scan(adapter(trace(records)))
        self.assertEqual(len(windows[0]), 1)
        item = windows[0][0]
        self.assertEqual((item['step'], item['ordinal']), (1, 0))
        self.assertIsNone(item['third_candidate'])
        self.assertEqual(item['previous_document']['step'], 0)
        self.assertEqual(item['previous_document']['model_counts']['task_forwards'], 1)
        self.assertEqual(item['previous_document']['model_counts']['task_rows'], 1)
        self.assertEqual(windows[1:], [[] for _ in range(5)])

    def test_ranks_ignore_noncoordinate_fields_and_are_source_specific(self):
        records = {3: [choice()], 9: [choice()], 119: [choice()]}
        original = scan(adapter(trace(records)))[1]
        def alter(doc):
            doc['unrelated_synthetic_value'] = -1000 * doc['step']
        changed = scan(adapter(trace(records, alter=alter)))[1]
        self.assertEqual([(x['step'], x['ordinal']) for x in original[0]],
                         [(x['step'], x['ordinal']) for x in changed[0]])
        ranks = [sampling.coordinate_rank('T01', 'market', 0, 3, 0),
                 sampling.coordinate_rank('T02', 'market', 0, 3, 0),
                 sampling.coordinate_rank('V01', 'market', 0, 3, 0),
                 sampling.coordinate_rank('T01', 'round2', 0, 3, 0),
                 sampling.coordinate_rank('T01', 'market', 1, 3, 0)]
        self.assertEqual(len(set(ranks)), len(ranks))
        for invalid in (('F01', 'market', 0), ('T01', 'other', 0), ('T01', 'market', True)):
            with self.assertRaises(ValueError):
                sampling.coordinate_rank(*invalid, 3, 0)

    def test_old_audit_failure_and_extra_bytes_do_not_unlock_candidates(self):
        def corrupt(doc):
            if doc['step'] == 718:
                doc['task_checkpoint_sha256'] = 'c' * 64
        for raw in (trace({1: [choice()]}, alter=corrupt), trace({1: [choice()]}) + b'x'):
            selector = adapter(raw)
            with self.assertRaises(ValueError):
                selector.scan(task_sha256=TASK, crop_sha256=CROP)
            with self.assertRaises(RuntimeError):
                selector.candidates()
            with self.assertRaises(RuntimeError):
                selector.scan(task_sha256=TASK, crop_sha256=CROP)

    def test_rejects_missing_callback_counts_and_oversized_line(self):
        def corrupt(doc):
            if doc['step'] == 500:
                doc['turn_model_counts']['task_forwards'] = 1
                doc['turn_model_counts']['task_rows'] = 2
            if doc['step'] >= 500:
                # Keep cumulative continuity valid while omitting the callback.
                doc['model_counts']['task_forwards'] += 1
                doc['model_counts']['task_rows'] += 2
        for raw in (trace({1: [choice()]}, alter=corrupt), b'x' * (previous.MAX_LINE + 1) + b'\n'):
            selector = adapter(raw)
            with self.assertRaises(ValueError):
                selector.scan(task_sha256=TASK, crop_sha256=CROP)
            with self.assertRaises(RuntimeError):
                selector.candidates()

    def test_late_nonstudent_callback_rejects_previously_ranked_candidates(self):
        def contaminate(doc):
            if doc['step'] == 500:
                doc['records'][0]['source'] = 'teacher'
        selector = adapter(trace({1: [choice()], 500: [choice()]}, alter=contaminate))
        with self.assertRaises(ValueError):
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(RuntimeError):
            selector.candidates()

    def test_guard_and_stream_resource_exceptions_are_not_hidden(self):
        for kind in (OSError, MemoryError, TimeoutError):
            with self.subTest(kind=kind.__name__):
                signal = kind('synthetic stop')
                def guard():
                    raise signal
                selector = adapter(trace(), guard=guard)
                with self.assertRaises(kind) as caught:
                    selector.scan(task_sha256=TASK, crop_sha256=CROP)
                self.assertIs(caught.exception, signal)
                with self.assertRaises(RuntimeError):
                    selector.candidates()
        class FailingEnd(io.BytesIO):
            def read(self, _size=-1):
                raise signal
        signal = OSError('synthetic final read')
        selector = sampling.RankedStream(FailingEnd(trace()), role='T01', opponent='market',
                                         side=0, guard=lambda: None)
        with self.assertRaises(OSError) as caught:
            selector.scan(task_sha256=TASK, crop_sha256=CROP)
        self.assertIs(caught.exception, signal)
        with self.assertRaises(RuntimeError):
            selector.candidates()

    def test_materialization_guard_failure_cannot_leak_partial_candidates(self):
        enabled = [False]
        signal = MemoryError('synthetic materialization stop')
        def guard():
            if enabled[0]:
                raise signal
        selector = adapter(trace({1: [choice()]}), guard=guard)
        selector.scan(task_sha256=TASK, crop_sha256=CROP)
        enabled[0] = True
        with self.assertRaises(MemoryError) as caught:
            selector.candidates()
        self.assertIs(caught.exception, signal)
        enabled[0] = False
        with self.assertRaises(RuntimeError):
            selector.candidates()

    def test_result_connects_to_real_student_comparison_interface(self):
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
            _summary, windows = scan(adapter(raw))
            self.assertEqual(len(windows[0]), 1)  # multiple callbacks, same step
            item = windows[0][0]
            self.assertEqual(item['step'], 1)
            result = compare_turn(item['document'], observation(1), cfg, action, item['ordinal'],
                task_sha256=TASK, crop_sha256=CROP, game_limits=_Budget(LIMITS).limits,
                agent_factory=SyntheticBounded, guard=lambda: None,
                previous_document=item['previous_document'])
            self.assertEqual(result.status, 'same')


if __name__ == '__main__':
    unittest.main()
