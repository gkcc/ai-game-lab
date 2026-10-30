"""Small synthetic regressions for runner acceptance and publishing."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_prefix_diagnostic_v1 import runner


class FakeOutput:
    def __init__(self, fail_at=None):
        self.fail_at, self.pending = fail_at, b''

    def __truediv__(self, name):
        return FakeFile(self, name)


class FakeFile:
    def __init__(self, root, name):
        self.root, self.name = root, name

    def open(self, mode):
        if self.name != 'result.pending.json' or mode != 'xb':
            raise ValueError('Unexpected file opening')
        return FakeStream(self.root)

    def read_bytes(self):
        return self.root.pending

    def exists(self):
        return False


class FakeStream:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        return self

    def write(self, value):
        self.root.pending = value

    def flush(self):
        if self.root.fail_at == 'flush':
            raise OSError('synthetic flush failure')

    def fileno(self):
        return 1

    def __exit__(self, *_):
        if self.root.fail_at == 'close':
            raise OSError('synthetic close failure')


class RunnerTest(unittest.TestCase):
    def test_replay_seed_mismatch_fails_without_outputting_the_seed(self):
        runner.check_replay_seed({'info': {'seed': 17}}, {'seed': 17})
        for replay, row in (({'info': {'seed': 18}}, {'seed': 17}),
                            ({'info': {'seed': True}}, {'seed': 1}), ({}, {'seed': 17})):
            with self.assertRaises(ValueError):
                runner.check_replay_seed(replay, row)

    def test_sampled_deadline_checks_final_elapsed_time(self):
        runner.check_deadline(0, lambda: runner.WALL_SECONDS)
        with self.assertRaises(ValueError):
            runner.check_deadline(0, lambda: runner.WALL_SECONDS+0.01)

    def test_each_trace_must_identify_both_frozen_models(self):
        row = {'crop_checkpoint_sha256': 'crop', 'task_checkpoint_sha256': None}
        runner.check_trace_identity({'checkpoint_sha256': 'crop', 'task_checkpoint_sha256': None}, row)
        for trace in ({'checkpoint_sha256': 'crop'},
                      {'checkpoint_sha256': 'changed', 'task_checkpoint_sha256': None},
                      {'checkpoint_sha256': 'crop', 'task_checkpoint_sha256': 'unexpected'}):
            with self.assertRaises(ValueError):
                runner.check_trace_identity(trace, row)

    def test_teacher_records_are_not_neural_callbacks(self):
        empty = dict(selection_records=0, callback_records=0, teacher_records=0,
                     zero_selection_turns=0, zero_callback_turns=0)
        teacher, student = dict(empty), dict(empty)
        runner.count_sources('teacher_task', [{'source': 'teacher'}], teacher)
        runner.count_sources('merged', [{'source': 'callback'}], student)
        self.assertEqual((1, 0, 1, 0, 1), tuple(teacher.values()))
        self.assertEqual((1, 1, 0, 0, 0), tuple(student.values()))
        runner.count_sources('teacher_task', [], teacher)
        self.assertEqual(1, teacher['zero_selection_turns'])
        self.assertEqual(2, teacher['zero_callback_turns'])
        with self.assertRaises(ValueError):
            runner.count_sources('teacher_task', [{'source': 'callback'}], teacher)

    def test_failed_flush_or_close_cannot_publish_final_success(self):
        for failure in ('flush', 'close'):
            output = FakeOutput(failure)
            with patch.object(runner.os, 'fsync'), patch.object(runner.os, 'replace') as replace:
                with self.assertRaises(OSError):
                    runner.write_result_new(output, {'passed': True})
                self.assertTrue(json.loads(output.pending)['passed'])
                replace.assert_not_called()

    def test_completed_write_is_verified_then_published(self):
        output = FakeOutput()
        with patch.object(runner.os, 'fsync'), patch.object(runner.os, 'replace') as replace:
            runner.write_result_new(output, {'passed': True})
            replace.assert_called_once()
        self.assertEqual({'passed': True}, json.loads(output.pending))


if __name__ == '__main__':
    unittest.main()
