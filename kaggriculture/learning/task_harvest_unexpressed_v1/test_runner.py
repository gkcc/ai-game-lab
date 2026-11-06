"""Isolated entry/import protocols only; no prepare, source reads or game Jobs."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from . import runner


def isolated(*arguments):
    """Wait for the owned child; -B avoids disposable cache artifacts."""
    return subprocess.run([sys.executable, '-I', '-B', *arguments],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


class RunnerTests(unittest.TestCase):
    def test_actual_isolated_help_returns_success_without_running_a_stage(self):
        result = isolated(str(Path(runner.__file__).resolve()), '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--registration-commit', result.stdout)
        self.assertIn('--case', result.stdout)
        self.assertNotIn('"status": "failed"', result.stderr)

    def test_contract_snapshot_and_metadata_imports_work_in_clean_isolated_process(self):
        # This child has no project path until the real metadata_modules() adds
        # it. Exercise the actual snapshot path which previously exposed a
        # premature package import. Audit hooks block data reads, not fake them.
        code = r'''
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

blocked = []
def audit(event, args):
    if event == 'import':
        name = args[0]
        if any(name == p or name.startswith(p + '.') for p in ('torch', 'kaggle_environments')):
            blocked.append('execution_import')
            raise AssertionError('Execution import during metadata loading')
    elif event == 'open' and isinstance(args[0], (str, bytes)):
        raw = args[0].decode() if isinstance(args[0], bytes) else args[0]
        path = Path(raw)
        if (set(part.lower() for part in path.parts) & {'private', 'artifacts'}
                or path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.gz', '.jsonl'}):
            blocked.append('private_or_model_read')
            raise AssertionError('Private or model data read during metadata loading')
sys.addaudithook(audit)

entry = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('_unexpressed_entry_test', entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
shared = module._shared()
infra = shared.infrastructure()
relative = module.PREFIX + 'contract.py'
raw = (module.REPO / relative).read_bytes()
contract = infra.snapshot(relative, hashlib.sha256(raw).hexdigest())
assert contract.specification()['selection_scope'] == 'initially_available_unexpressed_direct_crop_harvest'
assert len(contract.sources()) == 4 and contract.LIMITS['max_official_transitions'] == 384
metadata = module.metadata_modules()
module.no_execution_imports()
assert callable(metadata['selection'].select_target) and callable(metadata['sources'].load)
assert not blocked
print(json.dumps(dict(isolated=bool(sys.flags.isolated), no_bytecode=bool(sys.flags.dont_write_bytecode),
    snapshot_loaded=True, metadata_keys=sorted(metadata), blocked_data_attempts=len(blocked),
    execution_imported=any(n == p or n.startswith(p + '.')
        for p in ('torch', 'kaggle_environments') for n in sys.modules))))
'''
        result = isolated('-c', code, str(Path(runner.__file__).resolve()))
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value, dict(isolated=True, no_bytecode=True, snapshot_loaded=True,
            metadata_keys=['driver', 'fresh', 'maturity', 'selection', 'sources'],
            blocked_data_attempts=0, execution_imported=False))

    def test_cloned_coordinators_preserve_history_and_use_new_child_protocol(self):
        from learning.task_harvest_window_v1 import runner as historical
        keys = ('BASE', 'PREFIX', 'PRIVATE', 'STAGE', 'marker', 'run', 'infrastructure')
        history = {key: getattr(historical, key) for key in keys}
        one, two = runner._shared(), runner._shared()
        self.assertIsNot(one, two)
        self.assertIsNot(one.run.__globals__, historical.run.__globals__)
        # This is the live global used by the inherited run() for child argv,
        # not the function's original compile filename or a source-text test.
        self.assertEqual(Path(one.run.__globals__['__file__']).resolve(),
                         Path(runner.__file__).resolve())
        commit, digest = 'a' * 40, 'b' * 64
        permit = one.marker('permit', commit, digest, 'F01')
        self.assertEqual(permit, dict(schema='task-harvest-unexpressed-marker-v1',
            kind='permit', registration_commit=commit, binding_sha256=digest,
            retry_allowed=False, role='F01'))
        self.assertEqual(historical.marker('permit', commit, digest, 'F01')['schema'],
                         'task-harvest-window-marker-v1')
        calls = []
        def dispatch(value, *, preflight, role):
            calls.append((value, preflight, role))
            return None
        out, err = io.StringIO(), io.StringIO()
        with patch.object(one, 'run', dispatch), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(one.main(['--registration-commit', commit, '--case', 'F01']), 0)
        self.assertEqual(calls, [(commit, False, 'F01')])
        one.PRIVATE = Path('synthetic-in-memory-location')
        one.marker = lambda *args: {'synthetic': True}
        self.assertEqual(two.PRIVATE, runner.PRIVATE)
        self.assertEqual(two.marker('permit', commit, digest, 'F01'), permit)
        for key, value in history.items():
            self.assertIs(getattr(historical, key), value)


if __name__ == '__main__':
    unittest.main()
