"""Synthetic CPU and injected-device tests; never discover a real GPU."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


# The test process requests only CPU before importing JAX; no system setting is changed.
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_gpu_smoke_v1 import run, supervisor


class DeviceSelectionTests(unittest.TestCase):
    def test_exactly_one_matching_injected_gpu_is_required(self):
        device = types.SimpleNamespace(platform='gpu', device_kind='synthetic target')
        fake = types.SimpleNamespace(devices=lambda name: [device] if name == 'gpu' else [])
        self.assertIs(run.select_gpu(fake, 'synthetic target'), device)
        bad_lists = [[], [device, device],
                     [types.SimpleNamespace(platform='cpu', device_kind='synthetic target')],
                     [types.SimpleNamespace(platform='gpu', device_kind='different')]]
        for index, values in enumerate(bad_lists):
            with self.subTest(case=index), self.assertRaises(ValueError):
                run.select_gpu(types.SimpleNamespace(devices=lambda _: values), 'synthetic target')

    def test_name_validation_and_argument_error_do_not_echo_values(self):
        for invalid in ('', ' leading', 'trailing ', 'line\nbreak', 'x' * 129):
            with self.assertRaises(ValueError):
                run.device_name(invalid)
            with self.assertRaises(ValueError):
                supervisor.device_name(invalid)
        self.assertEqual(supervisor.device_name('synthetic target'),
                         run.device_name('synthetic target'))
        marker = 'synthetic-private-marker'
        command = [sys.executable, '-B', '-I', str(Path(run.__file__)), '--unexpected', marker]
        process = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(process.returncode, 0)
        self.assertNotIn(marker, process.stdout + process.stderr)
        self.assertNotIn(str(Path(run.__file__)), process.stdout + process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result, {'passed': False, 'stage': 'arguments', 'error_type': 'ValueError'})

    def test_exception_text_and_device_name_are_not_printed(self):
        marker = 'synthetic-private-marker'
        capture = io.StringIO()
        with patch.object(run, 'required_gpu_smoke', side_effect=ValueError(marker)), \
             patch.object(run, 'bind_project_imports', return_value=(None, None)), \
             contextlib.redirect_stdout(capture):
            code = run.cli(['--expected-device-name', marker, '--receipt-token', 'a' * 32])
        self.assertEqual(code, 1)
        self.assertNotIn(marker, capture.getvalue())
        self.assertEqual(json.loads(capture.getvalue())['error_type'], 'ValueError')

    def test_import_guard_failure_is_reported_at_bootstrap(self):
        marker = 'synthetic-private-marker'
        capture = io.StringIO()
        with patch.object(run, 'bind_project_imports', side_effect=ValueError(marker)), \
             contextlib.redirect_stdout(capture):
            code = run.cli(['--expected-device-name', marker, '--receipt-token', 'a' * 32])
        self.assertEqual(code, 1)
        self.assertNotIn(marker, capture.getvalue())
        self.assertEqual(json.loads(capture.getvalue()),
                         {'passed': False, 'stage': 'bootstrap', 'error_type': 'ValueError'})

    def test_child_systemexit_zero_is_sanitized_failure(self):
        code = r'''
import builtins, os, runpy, sys
script, marker, token = sys.argv[1:]
original = builtins.__import__
def injected(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'jax':
        os.write(1, marker.encode()); os.write(2, marker.encode())
        raise SystemExit(0)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = injected
sys.argv = [script, '--expected-device-name', marker, '--receipt-token', token]
runpy.run_path(script, run_name='__main__')
'''
        marker, token = 'synthetic-private-marker', 'a' * 32
        process = subprocess.run([sys.executable, '-B', '-I', '-c', code, str(run.__file__),
                                  marker, token], capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 1)
        self.assertNotIn(marker, process.stdout + process.stderr)
        self.assertNotIn(token, process.stdout + process.stderr)
        self.assertNotIn(str(Path(run.__file__)), process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout)['error_type'], 'SystemExit')

    def test_failed_selection_cannot_reach_fit(self):
        if importlib.util.find_spec('jax') is None:
            self.skipTest('JAX is unavailable')
        with patch.object(run, 'select_gpu', side_effect=ValueError('synthetic mismatch')), \
             patch.object(run, 'synthetic_pair') as fit, self.assertRaises(ValueError):
            run.required_gpu_smoke('synthetic target')
        fit.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('jax') is not None
                     and importlib.util.find_spec('optax') is not None, 'JAX and Optax are required')
class SyntheticCpuTests(unittest.TestCase):
    def test_two_paired_updates_run_on_explicit_cpu_without_gpu_claim(self):
        import jax
        cpu = jax.devices('cpu')[0]
        with jax.default_device(cpu):
            report = run.synthetic_pair(cpu, lambda: None)
        self.assertEqual(report['paired_updates_each'], 2)
        self.assertEqual(report['aux_nonzero_updates'], 2)
        self.assertTrue(report['both_arms_changed'] and report['arms_differ'])
        self.assertTrue(report['fit_device_and_finite_checks_passed'])
        self.assertNotIn('unique_requested_gpu_matched', report)

    def test_incomplete_returned_parameter_set_is_rejected(self):
        import jax
        from learning.task_pair_training_v1 import fit_core
        cpu = jax.devices('cpu')[0]
        step = {'aux_gradient_nonzero': True, 'control_changed': True,
                'experiment_changed': True, 'arms_differ': True}
        fake = {'completed_updates': 2, 'steps': [dict(step), dict(step)],
                'aux_nonzero_updates': 2, 'device_platform': 'cpu',
                'control_changed_from_initial': True, 'experiment_changed_from_initial': True,
                'arms_differ': True, 'control_params': {}, 'experiment_params': {}}
        with jax.default_device(cpu), patch.object(fit_core, 'fit_pair', return_value=fake), \
             self.assertRaises(ValueError):
            run.synthetic_pair(cpu, lambda: None)

    def test_non_default_cpu_device_does_not_fall_back(self):
        import jax
        devices = jax.devices('cpu')
        if len(devices) < 2:
            self.skipTest('Run with two CPU devices for this placement check')
        with jax.default_device(devices[0]):
            report = run.synthetic_pair(devices[1], lambda: None)
        self.assertEqual(report['paired_updates_each'], 2)
        self.assertTrue(report['fit_device_and_finite_checks_passed'])

    def test_deadline_callback_stops_before_fit(self):
        import jax
        from learning.task_pair_training_v1 import fit_core
        cpu = jax.devices('cpu')[0]
        def cancel():
            raise TimeoutError('synthetic deadline')
        with jax.default_device(cpu), patch.object(fit_core, 'fit_pair') as fit, \
             self.assertRaises(TimeoutError):
            run.synthetic_pair(cpu, cancel)
        fit.assert_not_called()


class ImportBindingTests(unittest.TestCase):
    def test_both_entries_pin_project_packages_and_keep_dependency_search_isolated(self):
        # All shadow packages below are virtual path-entry finders, never files.
        code = r'''
import importlib.abc, importlib.machinery, os, runpy, sys, tempfile
from pathlib import Path
entry = Path(sys.argv[1]).resolve()
root = entry.parents[2]
site = str(Path(tempfile.gettempdir()) / 'synthetic-virtual-site')
class Loader(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module): module.selected = module.__spec__.origin
class SiteFinder(importlib.abc.PathEntryFinder):
    def find_spec(self, fullname, target=None):
        if fullname in ('learning', 'synthetic_dependency'):
            return importlib.machinery.ModuleSpec(fullname, Loader(), origin='synthetic-site',
                                                  is_package=fullname == 'learning')
        return None
class RootFinder(importlib.abc.PathEntryFinder):
    def __init__(self):
        machinery = importlib.machinery
        self.actual = machinery.FileFinder(str(root),
            (machinery.ExtensionFileLoader, machinery.EXTENSION_SUFFIXES),
            (machinery.SourceFileLoader, machinery.SOURCE_SUFFIXES),
            (machinery.SourcelessFileLoader, machinery.BYTECODE_SUFFIXES))
    def find_spec(self, fullname, target=None):
        if fullname == 'synthetic_dependency':
            return importlib.machinery.ModuleSpec(fullname, Loader(), origin='synthetic-project-shadow')
        return self.actual.find_spec(fullname, target)
def path_hook(path):
    if path == site: return SiteFinder()
    if path == str(root): return RootFinder()
    raise ImportError
sys.path_hooks.insert(0, path_hook)
sys.path_importer_cache.pop(site, None)
sys.path_importer_cache.pop(str(root), None)
sys.path.append(site)
assert importlib.machinery.PathFinder.find_spec('learning', sys.path).origin == 'synthetic-site'
assert importlib.machinery.PathFinder.find_spec('synthetic_dependency', [str(root), site]).origin == 'synthetic-project-shadow'
namespace = runpy.run_path(str(entry))
guard, bound = namespace['bind_project_imports']()
assert bound == root and str(root) not in sys.path
with guard.protected_search_path(bound):
    from learning.task_pair_data_preflight_v1.test_core import fixture
    from learning.task_pair_training_v1 import fit_core, packing
    from learning.task_pair_training_v1.targets import qualifying_groups
    assert str(root) not in sys.path
guard.verify_origins(bound)
import synthetic_dependency
assert synthetic_dependency.selected == 'synthetic-site'
module = sys.modules['learning.task_pair_training_v1.targets']
old = module.__spec__.origin
module.__spec__.origin = 'synthetic-site'
try:
    try: guard.verify_origins(bound)
    except ValueError: pass
    else: raise AssertionError('tampered project origin accepted')
finally:
    module.__spec__.origin = old
guard.verify_origins(bound)
print('synthetic-ok')
'''
        for entry in (run.__file__, supervisor.__file__):
            with self.subTest(entry=Path(entry).name):
                process = subprocess.run([sys.executable, '-B', '-I', '-c', code, entry],
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stdout.strip(), 'synthetic-ok')

    @unittest.skipUnless(importlib.util.find_spec('jax') is not None
                         and importlib.util.find_spec('optax') is not None, 'JAX and Optax are required')
    def test_real_synthetic_fixture_import_and_two_updates_with_bound_cpu(self):
        code = r'''
import runpy, sys
namespace = runpy.run_path(sys.argv[1])
binding = namespace['bind_project_imports']()
import jax
cpu = jax.devices('cpu')[0]
result = namespace['synthetic_pair'](cpu, lambda: None, binding)
binding[0].verify_origins(binding[1])
assert result['paired_updates_each'] == 2
assert result['aux_nonzero_updates'] == 2
print('synthetic-cpu-ok')
'''
        process = subprocess.run([sys.executable, '-B', '-I', '-c', code, run.__file__],
                                 capture_output=True, text=True, timeout=50)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), 'synthetic-cpu-ok')


class SupervisorTests(unittest.TestCase):
    def receipt(self, *, reason='exited', exit_code=0, completed=True, clean=True):
        return {'reason': reason, 'child_exit_code': exit_code, 'elapsed_seconds': 1.0,
                'owned_process_group_signalled': clean, 'direct_child_reaped': clean,
                'no_live_owned_session_members_before_reap': clean, 'completed': completed}

    @staticmethod
    def success_message(token='a' * 32):
        return {'schema': supervisor.RECEIPT_SCHEMA, 'token': token, 'passed': True,
                'unique_requested_gpu_matched': True, 'paired_updates_each': 2,
                'aux_nonzero_updates': 2, 'both_arms_changed': True, 'arms_differ': True,
                'fit_device_and_finite_checks_passed': True}

    @staticmethod
    def wire(value):
        return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()

    def test_guard_projects_success_and_rejects_inconsistent_receipts(self):
        success = supervisor.project_guard(self.receipt(), True)
        self.assertTrue(success['passed'] and success['owned_session_clean'])
        self.assertIs(success['guard_with_cleanup_within_budget'], True)
        self.assertEqual(success['paired_updates_each'], 2)
        self.assertEqual(success['aux_nonzero_updates'], 2)
        for receipt in (self.receipt(reason='deadline', exit_code=-1, completed=False),
                        self.receipt(reason='cancelled', exit_code=-1, completed=False),
                        self.receipt(exit_code=1, completed=False), self.receipt(clean=False)):
            with self.subTest(reason=receipt['reason']):
                self.assertFalse(supervisor.project_guard(receipt, True)['passed'])
        with self.assertRaises(ValueError):
            supervisor.project_guard({'completed': True}, True)
        late = self.receipt()
        late['elapsed_seconds'] = 136.0
        self.assertEqual(supervisor.project_guard(late, True)['error_type'], 'TimeoutError')
        invalid = self.receipt()
        invalid['elapsed_seconds'] = float('nan')
        with self.assertRaises(ValueError):
            supervisor.project_guard(invalid, True)
        self.assertEqual(supervisor.project_guard(self.receipt(), False)['error_type'], 'InvalidReceipt')

    def test_child_command_is_argument_vector_and_does_not_execute(self):
        marker = 'synthetic target $(not-a-command)'
        command = supervisor.child_command('synthetic-python', marker, 'a' * 32)
        self.assertEqual(command[-3], marker)
        self.assertEqual(command[-1], 'a' * 32)
        self.assertEqual(command[1:3], ['-B', '-I'])
        self.assertEqual(supervisor.WORK_SECONDS, 120)

    def test_success_receipt_requires_exact_token_types_and_one_canonical_message(self):
        token = 'a' * 32
        message = self.success_message(token)
        raw = self.wire(message)
        self.assertTrue(supervisor.valid_receipt(raw, token))
        invalid = [b'', raw + raw, b'noise' + raw, raw + b'\n', raw[:-1],
                   b'x' * (supervisor.MAX_RECEIPT_BYTES + 1), b'\xff',
                   self.wire(self.success_message('b' * 32)),
                   self.wire({**message, 'passed': 1}),
                   self.wire({**message, 'paired_updates_each': 1}),
                   self.wire({**message, 'extra': True}),
                   self.wire({key: value for key, value in message.items() if key != 'passed'}),
                   raw.replace(b'"passed":true', b'"passed":true,"passed":true')]
        for sample in invalid:
            with self.subTest(length=len(sample)):
                self.assertFalse(supervisor.valid_receipt(sample, token))

    def test_pipe_drain_never_waits_for_an_open_writer(self):
        reader, writer = os.pipe()
        try:
            os.set_blocking(reader, False)
            collector = supervisor.ReceiptPipe(reader)
            collector.drain()
            self.assertFalse(collector.eof)
            os.write(writer, b'synthetic')
            collector.drain()
            self.assertEqual(collector.data, b'synthetic')
            self.assertFalse(collector.eof)
            os.close(writer)
            writer = None
            collector.drain()
            self.assertTrue(collector.eof)
        finally:
            os.close(reader)
            if writer is not None:
                os.close(writer)

    @unittest.skipUnless(os.name == 'posix', 'Linux session supervisor is required')
    def test_real_owned_child_early_exit_zero_is_not_success(self):
        from learning.task_dagger_once_v1.gpu_guard import supervise as guard
        for code in ('raise SystemExit(0)', 'import os; os._exit(0)'):
            with self.subTest(code=code):
                result = supervisor.supervise_command(
                    [sys.executable, '-B', '-I', '-c', code], 'a' * 32, guard, lambda: False)
                self.assertFalse(result['passed'])
                self.assertTrue(result['owned_session_clean'])
                self.assertEqual(result['error_type'], 'InvalidReceipt')

    @unittest.skipUnless(os.name == 'posix', 'Linux session supervisor is required')
    def test_real_owned_child_receipt_cases_and_output_bound(self):
        from learning.task_dagger_once_v1.gpu_guard import supervise as guard
        token = 'a' * 32
        raw = self.wire(self.success_message(token)).decode()
        wrong = self.wire(self.success_message('b' * 32)).decode()
        for case, output, passed in (('valid', raw, True), ('empty', '', False),
                                     ('wrong-token', wrong, False), ('noise', 'noise' + raw, False),
                                     ('multiple', raw + raw, False), ('overflow', 'x' * 50000, False)):
            with self.subTest(case=case):
                result = supervisor.supervise_command(
                    [sys.executable, '-B', '-I', '-c',
                     'import os,sys;os.write(1,sys.argv[1].encode())', output],
                    token, guard, lambda: False)
                self.assertEqual(result['passed'], passed)  # Synthetic receipt is not GPU evidence.
                self.assertTrue(result['owned_session_clean'])
                if not passed:
                    self.assertEqual(result['error_type'], 'InvalidReceipt')

    @unittest.skipUnless(os.name == 'posix', 'Linux session supervisor is required')
    def test_owned_guard_is_called_with_fixed_budget_and_handlers_restored(self):
        from learning.task_dagger_once_v1 import gpu_guard
        prior = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        seen = []
        def fake_guard(command, seconds, cancelled):
            seen.append((command, seconds, cancelled()))
            os.write(1, self.wire(self.success_message(command[-1])))
            return self.receipt()
        with patch.object(gpu_guard, 'supervise', side_effect=fake_guard):
            guard = types.SimpleNamespace(verify_origins=lambda root: None,
                                          protected_search_path=lambda root: contextlib.nullcontext())
            result = supervisor.supervise('synthetic target', (guard, None))
        self.assertTrue(result['passed'])  # Injected receipt, not GPU evidence.
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1:], (120, False))
        self.assertEqual(seen[0][0][1:3], ['-B', '-I'])
        self.assertEqual({sig: signal.getsignal(sig) for sig in prior}, prior)

    def test_native_output_is_suppressed_and_streams_restored(self):
        code = (
            "import os,sys;sys.path.insert(0,sys.argv[1]);"
            "from learning.task_pair_gpu_smoke_v1.run import quiet_native_output;"
            "ctx=quiet_native_output();ctx.__enter__();"
            "os.write(1,b'synthetic-private-marker');os.write(2,b'synthetic-private-marker');"
            "ctx.__exit__(None,None,None);print('safe')"
        )
        process = subprocess.run([sys.executable, '-B', '-I', '-c', code, str(run.GAME_SOURCE)],
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(process.stdout.strip(), 'safe')
        self.assertNotIn('synthetic-private-marker', process.stdout + process.stderr)

    @unittest.skipUnless(os.name == 'posix', 'Linux bootstrap is required')
    def test_subprocess_dependency_import_failure_is_sanitized(self):
        process = self.bootstrap_subprocess('fail')
        self.assertEqual(process.returncode, 1)
        self.assertNotIn('synthetic-private-marker', process.stdout + process.stderr)
        self.assertNotIn(str(Path(supervisor.__file__)), process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout),
                         {'passed': False, 'stage': 'supervisor', 'error_type': 'ImportError'})

    @staticmethod
    def bootstrap_subprocess(mode):
        code = r'''
import builtins, os, runpy, sys
script, marker, mode = sys.argv[1:]
original = builtins.__import__
def injected(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'hashlib' and globals.get('__name__') == '__main__' and mode == 'fail':
        os.write(1, marker.encode()); os.write(2, marker.encode())
        raise ImportError(marker)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = injected
sys.argv = [script, '--expected-device-name', marker]
runpy.run_path(script, run_name='__main__')
'''
        return subprocess.run([sys.executable, '-B', '-I', '-c', code,
                               str(Path(supervisor.__file__)), 'synthetic-private-marker', mode],
                              capture_output=True, text=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
