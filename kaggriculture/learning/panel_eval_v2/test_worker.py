import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from . import isolation


class WorkerEntryTests(unittest.TestCase):
    def test_official_entry_checks_deadline_and_memory_before_calling_runner(self):
        from .worker import guarded_official_entry
        from learning.panel_safety_v1.memory_gate import MIN_AVAILABLE_BYTES
        events = []
        class Context:
            def deadline_check(self): events.append('deadline')
        def entry(): events.append('official_entry'); return 0
        with self.assertRaises(RuntimeError):
            guarded_official_entry(Context(), entry, lambda:MIN_AVAILABLE_BYTES-1)
        self.assertEqual(events, ['deadline'])
        self.assertEqual(guarded_official_entry(Context(), entry, lambda:MIN_AVAILABLE_BYTES), 0)
        self.assertEqual(events, ['deadline','deadline','official_entry'])

    def test_worker_binds_and_restores_actual_official_entry_hook(self):
        from . import worker
        from learning.panel_safety_v1.memory_gate import MIN_AVAILABLE_BYTES
        from learning.task_teacher_merged_eval_v1 import games
        class Context:
            OUT = Path('synthetic-stage')
            def __init__(self, *_): pass
            def refresh(self): pass
            def deadline_check(self): pass
        args = ['worker','--out','synthetic','--key','synthetic','--prereg','a'*40,
                '--scratch','synthetic','--mode','game']
        with patch.object(sys,'argv',args), patch('learning.panel_eval_v2.isolation.validate_runtime'), \
             patch('learning.panel_eval_v2.experiment.Context',Context), \
             patch('learning.panel_eval_v2.epoch.EpochContext',side_effect=lambda value:value), \
             patch('learning.panel_eval_v2.adapter.bind_context',return_value=nullcontext()), \
             patch.object(games,'job',side_effect=lambda *_:games.core.official_entry()), \
             patch.object(games.core,'official_entry',return_value=0) as entry, \
             patch('psutil.virtual_memory',return_value=SimpleNamespace(available=MIN_AVAILABLE_BYTES-1)) as memory:
            with self.assertRaises(RuntimeError): worker.main()
            entry.assert_not_called()
            self.assertIs(games.core.official_entry,entry)
            memory.return_value.available = MIN_AVAILABLE_BYTES
            worker.main()
            entry.assert_called_once_with()
            self.assertIs(games.core.official_entry,entry)

    def test_invalid_stage_fails_without_exposing_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch = Path(temp)
            project = Path(__file__).resolve().parents[2]
            command = isolation.bootstrap_command(sys.executable, scratch, scratch/'cache-worker', project,
                'learning.panel_eval_v2.worker')
            command += ['--out', str(scratch/'missing'), '--key', 'SYNTHETIC_PRIVATE_MARKER',
                        '--prereg', 'a'*40, '--scratch', str(scratch), '--mode', 'game']
            result = subprocess.run(command, cwd=project, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertEqual(result.stderr, b'')
            self.assertFalse((scratch/'cache-worker').exists())

    def test_worker_refuses_missing_cache_prefix_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(__file__).resolve().parents[2]
            command = [sys.executable, '-B', '-m', 'learning.panel_eval_v2.worker',
                       '--out', str(Path(temp)/'missing'), '--key', 'SYNTHETIC_PRIVATE_MARKER',
                       '--prereg', 'a'*40, '--scratch', temp, '--mode', 'canonical']
            result = subprocess.run(command, cwd=project, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertEqual(result.stderr, b'')

    def test_independent_worker_rejects_invalid_stage_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch = Path(temp)
            project = Path(__file__).resolve().parents[2]
            command = isolation.bootstrap_command(sys.executable, scratch, scratch/'cache-replay', project,
                'learning.panel_eval_v2.independent_worker')
            command += ['--out', str(scratch/'SYNTHETIC_PRIVATE_MARKER'), '--prereg', 'a'*40,
                        '--scratch', str(scratch)]
            result = subprocess.run(command, cwd=project, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertEqual(result.stderr, b'')
            self.assertFalse((scratch/'cache-replay').exists())


if __name__ == '__main__':
    unittest.main()
