"""Real start-entry regressions with simulated processes and no files/devices."""
from contextlib import ExitStack
from pathlib import Path
import unittest
import uuid
from unittest.mock import Mock, mock_open, patch

from learning.task_anchor_once_v1 import processes
from learning.task_action_observation_v1 import campaign as old_campaign
from learning.task_launcher_contract_repair_v1 import observer
from . import run


class FakeProcess:
    def __init__(self, running=False):
        self.code = None if running else 0

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.code = -15
        return self.code


class FakeChild:
    def __init__(self, running=False, error=None):
        self.process = FakeProcess(running)
        self.error = error

    def check(self):
        if self.error:
            raise self.error

    def finish(self):
        return {'child_exit_code':self.process.poll(), 'reader_joined':True}


class StartEntryTests(unittest.TestCase):
    def setUp(self):
        self.out = run.ROOT/('synthetic-entry-'+uuid.uuid4().hex)
        self.records = {}
        self.guard = {'passed':True, 'supervision':{'direct_child_reaped':True,
            'owned_process_group_signalled':True,'no_live_owned_session_members_before_reap':True}}
        self.worker = {'completed':True, 'parity_passed':False}
        self.cfg = {'log_bytes':1024,'process_seconds':60,'cleanup_seconds':5}
        self.cleanup = {'owned_instances_remaining':0,'reader_joined':True,'launched_process_reaped':True}
        self.config_error = False
        self.receipt_error = False
        self.guard_available = True
        self.marker_error = False

    def read(self, path):
        name = Path(path).name
        if name == 'gpu-launch.local.json':
            if self.config_error:
                raise ValueError('synthetic setup failure')
            return {'distro':'synthetic','workdir':'synthetic','python':'synthetic'}
        if name == 'guard-result.json':
            return self.guard
        if name == 'worker-result.json':
            return self.worker
        raise AssertionError('unexpected read')

    def write(self, path, value):
        name = Path(path).name
        if name == 'run-result.json' and self.receipt_error:
            raise PermissionError('synthetic receipt failure')
        if name in self.records:
            raise FileExistsError(name)
        self.records[name] = value

    def marker(self, out, value, state):
        state['created'] = True
        self.records['run-started.json'] = value
        if self.marker_error:
            raise OSError('synthetic partial marker')
        state['complete'] = True

    def exists(self, path):
        name = Path(path).name
        if name == 'guard-result.json':
            return self.guard_available
        return name in self.records

    def mocks(self, child=None, spawn_error=None, cleanup_error=None):
        stack = ExitStack()
        stack.enter_context(patch.object(run,'ready',return_value=self.cfg))
        stack.enter_context(patch.object(run,'verify_commit'))
        stack.enter_context(patch.object(run,'deadline_check'))
        stack.enter_context(patch.object(run,'sha',return_value='f'*64))
        stack.enter_context(patch.object(run,'read',side_effect=self.read))
        stack.enter_context(patch.object(run,'write_new',side_effect=self.write))
        stack.enter_context(patch.object(run,'write_started_marker',side_effect=self.marker))
        stack.enter_context(patch.object(Path,'exists',autospec=True,side_effect=self.exists))
        spawn = stack.enter_context(patch.object(processes,'CapturedProcess',return_value=child,side_effect=spawn_error))
        stack.enter_context(patch.object(old_campaign,'observe_processes'))
        reaper = stack.enter_context(patch.object(observer,'reap',return_value=self.cleanup,side_effect=cleanup_error))
        return stack, spawn, reaper

    def test_success_reaches_transport_and_reaps(self):
        stack, spawn, reaper = self.mocks(child=FakeChild())
        with stack:
            self.assertEqual(run.start(self.out),self.worker)
        self.assertEqual(spawn.call_count,1)
        self.assertEqual(reaper.call_count,1)
        self.assertTrue(self.records['run-result.json']['passed'])
        self.assertTrue(self.records['run-result.json']['child_handle_acquired'])

    def test_setup_failure_records_phase_without_child(self):
        self.config_error = True
        stack, spawn, reaper = self.mocks()
        with stack, self.assertRaises(RuntimeError):
            run.start(self.out)
        result = self.records['run-result.json']
        self.assertEqual(result['failure_phase'],'configuration')
        self.assertEqual(result['error_kind'],'ValueError')
        self.assertFalse(result['child_handle_acquired'])
        self.assertIsNone(result['linux_session_cleanup_proven'])
        spawn.assert_not_called()
        reaper.assert_not_called()

    def test_spawn_failure_records_attempt(self):
        stack, spawn, reaper = self.mocks(spawn_error=OSError('synthetic spawn failure'))
        with stack, self.assertRaises(RuntimeError):
            run.start(self.out)
        self.assertEqual(self.records['run-result.json']['failure_phase'],'transport_start')
        self.assertEqual(self.records['run-result.json']['error_kind'],'OSError')
        self.assertFalse(self.records['run-result.json']['child_handle_acquired'])
        self.assertEqual(self.records['run-result.json']['child_lifecycle'],'unknown_after_spawn_attempt')
        self.assertIsNone(self.records['run-result.json']['linux_session_cleanup_proven'])
        self.assertEqual(spawn.call_count,1)
        reaper.assert_not_called()

    def test_partially_written_marker_still_records_consumed_failure(self):
        self.marker_error = True
        stack, spawn, _ = self.mocks()
        with stack, self.assertRaises(RuntimeError):
            run.start(self.out)
        result = self.records['run-result.json']
        self.assertEqual(result['failure_phase'],'start_marker')
        self.assertTrue(result['marker_created'])
        self.assertFalse(result['marker_complete'])
        self.assertEqual(result['child_lifecycle'],'not_started')
        spawn.assert_not_called()

    def test_actual_marker_records_creation_before_serialization_failure(self):
        state = {'created':False,'complete':False}
        with patch.object(run,'OUT',self.out.resolve()), patch.object(Path,'open',mock_open()), \
                patch.object(run.json,'dump',side_effect=OSError('synthetic write failure')):
            with self.assertRaises(OSError):
                run.write_started_marker(self.out,{'attempt':1},state)
        self.assertTrue(state['created'])
        self.assertFalse(state['complete'])

    def test_runtime_failure_cancels_and_reaps(self):
        self.guard_available = False
        stack, _, reaper = self.mocks(child=FakeChild(running=True,error=RuntimeError('synthetic')))
        with stack, self.assertRaises(RuntimeError):
            run.start(self.out)
        result = self.records['run-result.json']
        self.assertEqual(result['failure_phase'],'transport_monitor')
        self.assertEqual(result['error_kind'],'RuntimeError')
        self.assertTrue(result['child_handle_acquired'])
        self.assertTrue(result['windows_cleanup']['launched_process_reaped'])
        self.assertIn('cancel.local.json',self.records)
        self.assertEqual(reaper.call_count,1)

    def test_cleanup_error_is_not_a_success(self):
        stack, _, _ = self.mocks(child=FakeChild(),cleanup_error=RuntimeError('synthetic cleanup'))
        with stack, self.assertRaises(RuntimeError):
            run.start(self.out)
        result = self.records['run-result.json']
        self.assertFalse(result['passed'])
        self.assertEqual(result['cleanup_error_kind'],'RuntimeError')
        self.assertIsNone(result['error_kind'])

    def test_poll_failure_does_not_hide_primary_error(self):
        self.guard_available = False
        child = FakeChild(running=True,error=RuntimeError('synthetic primary'))
        child.process.poll = Mock(side_effect=[None,OSError('synthetic poll failure')])
        stack, _, reaper = self.mocks(child=child)
        with stack, self.assertRaises(run.StartFailure):
            run.start(self.out)
        result = self.records['run-result.json']
        self.assertEqual(result['error_kind'],'RuntimeError')
        self.assertEqual(result['cancel_error_kind'],'OSError')
        self.assertEqual(result['failure_phase'],'transport_monitor')
        self.assertEqual(reaper.call_count,1)

    def test_marker_prevents_retry(self):
        self.records['run-started.json'] = {'retry_allowed':False}
        stack, spawn, _ = self.mocks()
        with stack, self.assertRaises(ValueError):
            run.start(self.out)
        spawn.assert_not_called()
        self.assertNotIn('run-result.json',self.records)

    def test_receipt_failure_preserves_primary_error(self):
        self.config_error = self.receipt_error = True
        stack, _, _ = self.mocks()
        with stack, self.assertRaises(run.ReceiptFailure) as caught:
            run.start(self.out)
        self.assertEqual(caught.exception.primary_error_kind,'ValueError')
        self.assertEqual(caught.exception.receipt_error_kind,'PermissionError')


if __name__ == '__main__':
    unittest.main()
