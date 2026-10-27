"""Exercise the real fit entry with simulated process and in-memory receipts."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import uuid
from unittest.mock import mock_open, patch

# Import a private test instance; the temporary dependency substitute is restored
# immediately, and no campaign module with substituted dependencies is registered.
stub = types.ModuleType('learning.task_action_observation_v1.campaign')
stub.observe_processes = lambda *args: None
stub.owned_size = lambda *args: 0
spec = importlib.util.spec_from_file_location(
    'learning.task_completion_bc_reentry_v1._campaign_test_instance', Path(__file__).with_name('campaign.py'))
c = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'learning.task_action_observation_v1.campaign': stub}):
    spec.loader.exec_module(c)


class EntryTest(unittest.TestCase):
    def run_case(self, failure=None):
        out = c.ROOT / ('synthetic-entry-' + uuid.uuid4().hex)
        records = {}
        cfg = {'storage': {'process_log_bytes': 100}, 'training': {'process_seconds': 1200}}
        guard = {'passed': True, 'supervision': dict(owned_process_group_signalled=True,
            direct_child_reaped=True, no_live_owned_session_members_before_reap=True)}
        cleanup = dict(owned_instances_remaining=0, reader_joined=True, launched_process_reaped=True)
        def read(path):
            name = Path(path).name
            if name == 'train-data-result.json': return {'passed': True}
            if name == 'gpu-launch.local.json':
                if failure == 'config': raise ValueError('configuration')
                return dict(distro='x', workdir='x', python='x')
            if name == 'gpu-guard-result.json':
                if failure == 'guard': raise ValueError('guard')
                return guard
            raise AssertionError(name)
        def write(path, value):
            name = Path(path).name
            if name == 'fit-transport-result.json' and failure == 'receipt': raise OSError('receipt')
            if name in records: raise FileExistsError(name)
            records[name] = value
        def marker(path, value, state):
            state['created'] = True
            records['fit-started.json'] = value
            if failure == 'marker': raise OSError('partial marker')
            state['complete'] = True
        def exists(path):
            return Path(path).name == 'gpu-guard-result.json' or Path(path).name in records
        class Process:
            returncode = 0
            def poll(self):
                if failure == 'poll': raise LookupError('poll')
                return 0
        class Child:
            process = Process()
            def finish(self): return {'child_exit_code': 0}
        with ExitStack() as stack:
            for name, value in (('ready', cfg), ('verify_commit', None), ('storage_ready', None),
                                ('sha', 'hash'), ('deadline_check', None), ('observe_processes', None)):
                stack.enter_context(patch.object(c, name, return_value=value))
            stack.enter_context(patch.object(c, 'read', side_effect=read))
            stack.enter_context(patch.object(c, 'write_new', side_effect=write))
            stack.enter_context(patch.object(c, 'write_fit_marker', side_effect=marker))
            stack.enter_context(patch.object(Path, 'exists', autospec=True, side_effect=exists))
            stack.enter_context(patch.object(c, 'reap', side_effect=RuntimeError() if failure == 'cleanup' else None, return_value=cleanup))
            stack.enter_context(patch.object(c, 'CapturedProcess', side_effect=OSError() if failure == 'spawn' else None, return_value=Child()))
            stack.enter_context(patch.object(c, '_seal_fit', side_effect=ValueError() if failure == 'seal' else None, return_value={'passed': True}))
            if failure is None: self.assertTrue(c.gpu_fit(out)['passed'])
            else:
                with self.assertRaises(Exception) as caught: c.gpu_fit(out)
                if failure == 'receipt': self.assertEqual(caught.exception.receipt_error_kind, 'OSError')
            if failure != 'receipt':
                result = records['fit-transport-result.json']
                self.assertEqual(result['passed'], failure in (None, 'seal'))
                if failure == 'spawn': self.assertEqual(result['child_lifecycle'], 'unknown_after_spawn_attempt')
                if failure == 'marker': self.assertFalse(result['marker_complete'])
                if failure == 'poll':
                    self.assertEqual(result['error_kind'], 'LookupError')
                    self.assertEqual(result['cancel_error_kind'], 'LookupError')
                if failure == 'guard': self.assertEqual(result['guard_read_error_kind'], 'ValueError')
            if failure == 'seal': self.assertIn('fit-seal-failure.json', records)
            with self.assertRaises(ValueError): c.gpu_fit(out)

    def test_transport_outcomes(self):
        for failure in (None, 'marker', 'config', 'spawn', 'poll', 'cleanup', 'guard', 'receipt', 'seal'):
            with self.subTest(failure=failure): self.run_case(failure)

    def test_real_marker_partial_write(self):
        out = c.ROOT / ('synthetic-marker-' + uuid.uuid4().hex)
        state = dict(created=False, complete=False)
        with patch.object(c, 'OUT', out.resolve()), patch.object(Path, 'open', mock_open()), \
             patch.object(c.json, 'dump', side_effect=OSError('write')):
            with self.assertRaises(OSError): c.write_fit_marker(out, {}, state)
        self.assertEqual(state, dict(created=True, complete=False))


if __name__ == '__main__': unittest.main()
