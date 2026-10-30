"""Check that new failure reporting is useful without disclosing inputs."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from . import runner


class RunnerFailureTests(unittest.TestCase):
    def test_preflight_memory_failure_has_no_success_output_or_exception_text(self):
        err = io.StringIO()
        with patch.object(runner.Guard, 'sample', side_effect=MemoryError('private secret')), \
             patch('sys.stderr', err), patch('builtins.open') as opened:
            code = runner.run('unused', 'unused', 'unused', 'unused', '0' * 40)
        self.assertEqual(code, 1)
        self.assertIn('"phase": "preflight"', err.getvalue())
        self.assertIn('"category": "memory_gate"', err.getvalue())
        self.assertNotIn('private secret', err.getvalue())
        opened.assert_not_called()

    def test_read_failure_is_classified_without_writing_result(self):
        err = io.StringIO()
        root = runner.REPO
        output = root / 'no-diagnostic-output-created.local.json'
        self.assertFalse(output.exists())
        with patch.object(runner.Guard, 'sample', return_value=True), \
             patch.object(runner, 'require_checkout'), \
             patch.object(runner, 'prepare_output'), \
             patch.object(runner, 'read_json', side_effect=OSError('private secret')), \
             patch('sys.stderr', err):
            code = runner.run(str(root / 'review'), 'unused', 'unused', str(output), '0' * 40)
        self.assertEqual(code, 1)
        self.assertIn('"phase": "provenance"', err.getvalue())
        self.assertIn('"category": "io_error"', err.getvalue())
        self.assertNotIn('private secret', err.getvalue())
        self.assertFalse(output.exists())

    def test_resource_samples_distinguish_initial_and_running_gates(self):
        guard = runner.Guard()
        with patch.object(runner.psutil, 'virtual_memory') as memory:
            memory.return_value.available = runner.INITIAL_MEMORY - 1
            with self.assertRaises(MemoryError):
                guard.sample(initial=True)
            memory.return_value.available = runner.DURING_MEMORY
            self.assertTrue(guard.sample())
            memory.return_value.available -= 1
            with self.assertRaises(MemoryError):
                guard.sample()

    def test_output_cannot_escape_back_into_old_stage(self):
        stage = runner.REPO / 'artifacts' / runner.ORIGINAL_STAGE
        good = stage.parent / runner.OUTPUT_STAGE / 'result.local.json'
        runner.output_layout(stage, good)
        disguised = stage.parent / runner.OUTPUT_STAGE / '..' / runner.ORIGINAL_STAGE / 'new.local.json'
        with self.assertRaises(ValueError):
            runner.output_layout(stage, disguised)
        with self.assertRaises(ValueError):
            runner.output_layout(stage, stage / 'new.local.json')


if __name__ == '__main__':
    unittest.main()
