"""Synthetic receipt and Linux process checks; no data, model, GPU or games."""

import json
import os
from dataclasses import replace
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_supervision_v1 import supervisor as tested


TOKEN = 'a' * 32  # Synthetic, never a real attempt token.
CONTRACT = tested.Contract(expected_updates=64, minimum_auxiliary_nonzero=4,
                           work_seconds=5, guard_with_cleanup_seconds=8,
                           term_seconds=.1, cleanup_seconds=3, gate_seconds=2,
                           expected_gate_sha256=tested.pidfd_guard.GATE_SHA256)


def receipt(**changes):
    value = {'schema': tested.RECEIPT_SCHEMA, 'token': TOKEN, 'passed': True,
             'completed_updates': 64, 'auxiliary_nonzero_updates': 4,
             'both_arms_changed': True, 'arms_differ': True}
    value.update(changes)
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')


def guard_report(**changes):
    value = {'reason': 'exited', 'child_exit_code': 0, 'elapsed_seconds': 1.0,
             'owned_session_empty_before_reap': True, 'direct_child_reaped': True,
             'used_pidfd': True, 'completed': True}
    value.update(changes)
    return value


def command(code):
    return [sys.executable, '-B', '-I', '-c', code, '--receipt-token', TOKEN]


def child_code(*, updates=64, auxiliary=4, token_override=None, exit_code=0, duplicate=False):
    # Only literal synthetic data are embedded; the actual child token is argv.
    return ('import json,sys; '
            f"v={{'schema':{tested.RECEIPT_SCHEMA!r},'token':sys.argv[2],"
            f"'passed':True,'completed_updates':{updates},'auxiliary_nonzero_updates':{auxiliary},"
            "'both_arms_changed':True,'arms_differ':True}; "
            + (f"v['token']={token_override!r}; " if token_override is not None else '')
            + "s=json.dumps(v,sort_keys=True,separators=(',',':')); "
            + ("print(s,flush=True); " if duplicate else '')
            + f"print(s,flush=True); sys.exit({exit_code})")


class ReceiptTests(unittest.TestCase):
    def test_receipt_gates_are_independent_and_registered_not_hardcoded(self):
        valid = tested.inspect_receipt(receipt(), TOKEN, CONTRACT)
        self.assertTrue(all(valid.values()))
        few_steps = tested.inspect_receipt(receipt(completed_updates=63), TOKEN, CONTRACT)
        self.assertTrue(few_steps['canonical_token_receipt'])
        self.assertFalse(few_steps['registered_update_count_met'])
        self.assertTrue(few_steps['registered_auxiliary_threshold_met'])
        few_aux = tested.inspect_receipt(receipt(auxiliary_nonzero_updates=3), TOKEN, CONTRACT)
        self.assertTrue(few_aux['registered_update_count_met'])
        self.assertFalse(few_aux['registered_auxiliary_threshold_met'])
        no_change = tested.inspect_receipt(receipt(arms_differ=False), TOKEN, CONTRACT)
        self.assertFalse(no_change['both_arms_changed_and_differ'])
        alternate = replace(CONTRACT, expected_updates=3, minimum_auxiliary_nonzero=1)
        self.assertTrue(all(tested.inspect_receipt(receipt(completed_updates=3,
                               auxiliary_nonzero_updates=1), TOKEN, alternate).values()))

    def test_invalid_format_token_duplicate_and_numbers_are_rejected(self):
        correct = receipt()
        invalid = (b'', correct + correct, correct.rstrip(b'\n'), b'x' * 4097,
                   receipt(token='b' * 32), receipt(token='\u00e9' * 32),
                   receipt(completed_updates=True), receipt(auxiliary_nonzero_updates=65),
                   b'{"schema":"duplicate",' + correct[1:],
                   correct.replace(b'"completed_updates":64', b'"completed_updates":NaN'))
        for raw in invalid:
            with self.subTest(length=len(raw)):
                self.assertFalse(tested.inspect_receipt(raw, TOKEN, CONTRACT)['canonical_token_receipt'])

    def test_guard_exit_receipt_cleanup_and_total_budget_do_not_mask_each_other(self):
        ok = tested.inspect_receipt(receipt(), TOKEN, CONTRACT)
        projected = tested.project_guard(guard_report(), ok, CONTRACT)
        self.assertTrue(projected['passed'])
        self.assertTrue(projected['owned_session_empty_before_reap'])
        self.assertTrue(projected['direct_child_reaped'])
        self.assertTrue(projected['used_pidfd'])
        cases = (
            (guard_report(child_exit_code=7, completed=False), ok, 'child_exit_zero'),
            (guard_report(reason='deadline', completed=False), ok, 'child_exit_observed'),
            (guard_report(used_pidfd=False), ok, 'owned_session_clean'),
            (guard_report(direct_child_reaped=False), ok, 'owned_session_clean'),
            (guard_report(owned_session_empty_before_reap=False), ok, 'owned_session_clean'),
            (guard_report(elapsed_seconds=8.01), ok, 'guard_with_cleanup_within_budget'),
            (guard_report(), tested.inspect_receipt(b'', TOKEN, CONTRACT), 'canonical_token_receipt'),
        )
        for report, result, key in cases:
            with self.subTest(key=key):
                projection = tested.project_guard(report, result, CONTRACT)
                self.assertFalse(projection['passed'])
                self.assertFalse(projection[key])
        for report in (guard_report(completed=False), guard_report(elapsed_seconds=float('nan')),
                       guard_report(child_exit_code=True), guard_report(reason='unknown')):
            with self.assertRaises(ValueError):
                tested.project_guard(report, ok, CONTRACT)

    def test_registered_contract_and_token_command_are_strict(self):
        for bad in (replace(CONTRACT, expected_updates=True), replace(CONTRACT, minimum_auxiliary_nonzero=0),
                    replace(CONTRACT, minimum_auxiliary_nonzero=65),
                    replace(CONTRACT, guard_with_cleanup_seconds=7.9),
                    replace(CONTRACT, work_seconds=float('inf')),
                    replace(CONTRACT, work_seconds=10**1000),
                    replace(CONTRACT, term_seconds=3), replace(CONTRACT, gate_seconds=6),
                    replace(CONTRACT, expected_gate_sha256='f' * 64)):
            with self.assertRaises(ValueError):
                tested.inspect_receipt(receipt(), TOKEN, bad)
        self.assertEqual(tested._command(command('pass'), TOKEN), command('pass'))
        for cmd in ([sys.executable, '-c', 'pass'], command('pass') + ['--receipt-token', TOKEN],
                    command('pass')[:-1] + ['b' * 32]):
            with self.assertRaises(ValueError):
                tested._command(cmd, TOKEN)


@unittest.skipUnless(os.name == 'posix' and hasattr(os, 'WNOWAIT'), 'Linux session guard required')
class LinuxSessionTests(unittest.TestCase):
    def run_child(self, code, contract=CONTRACT, cancelled=lambda: False, receipt_sink=lambda raw: True):
        return tested.supervise_command(command(code), TOKEN, contract,
                                        cancelled=cancelled, receipt_sink=receipt_sink)

    def test_real_owned_child_with_registered_receipt(self):
        saved = []
        def sink(raw):
            saved.append(raw)
            return True
        result = self.run_child(child_code(), receipt_sink=sink)
        self.assertTrue(result['passed'])
        self.assertTrue(result['child_exit_zero'] and result['canonical_token_receipt'])
        self.assertTrue(result['owned_session_clean'] and result['guard_with_cleanup_within_budget'])
        self.assertTrue(result['receipt_sink_attempted'] and result['receipt_sink_accepted'])
        self.assertEqual(saved, [receipt()])
        self.assertNotIn(TOKEN, repr(result))

    def test_zero_exit_without_receipt_and_nonzero_exit_with_receipt(self):
        saved = []
        sink = lambda raw: saved.append(raw) or True
        missing = self.run_child('pass', receipt_sink=sink)
        self.assertFalse(missing['passed'])
        self.assertTrue(missing['child_exit_zero'])
        self.assertFalse(missing['canonical_token_receipt'])
        failed = self.run_child(child_code(exit_code=7), receipt_sink=sink)
        self.assertFalse(failed['passed'])
        self.assertFalse(failed['child_exit_zero'])
        self.assertTrue(failed['canonical_token_receipt'])
        self.assertEqual(saved, [])
        self.assertFalse(missing['receipt_sink_attempted'] or failed['receipt_sink_attempted'])

    def test_real_wrong_token_duplicate_count_and_auxiliary_threshold(self):
        for code, key in ((child_code(token_override='b' * 32), 'canonical_token_receipt'),
                          (child_code(duplicate=True), 'canonical_token_receipt'),
                          (child_code(updates=63), 'registered_update_count_met'),
                          (child_code(auxiliary=3), 'registered_auxiliary_threshold_met')):
            with self.subTest(key=key):
                saved = []
                result = self.run_child(code, receipt_sink=lambda raw: saved.append(raw) or True)
                self.assertFalse(result['passed'])
                self.assertFalse(result[key])
                self.assertTrue(result['owned_session_clean'])
                self.assertFalse(result['receipt_sink_accepted'])
                self.assertEqual(saved, [])

    def test_receipt_sink_returns_exact_true_or_fails(self):
        for value in (False, None, 1):
            with self.subTest(return_type=type(value).__name__):
                received = []
                def sink(raw):
                    received.append(raw)
                    return value
                result = self.run_child(child_code(), receipt_sink=sink)
                self.assertEqual(received, [receipt()])
                self.assertTrue(result['receipt_sink_attempted'])
                self.assertFalse(result['receipt_sink_accepted'])
                self.assertFalse(result['passed'])

    def test_cancellation_observed_while_sink_runs_overrides_success(self):
        state = {'cancelled': False, 'receipts': []}
        def sink(raw):
            state['receipts'].append(raw)
            state['cancelled'] = True
            return True
        result = self.run_child(child_code(), cancelled=lambda: state['cancelled'], receipt_sink=sink)
        self.assertEqual(state['receipts'], [receipt()])
        self.assertTrue(result['receipt_sink_accepted'] and result['cancelled'])
        self.assertFalse(result['passed'])

    def test_sink_is_not_called_when_cleanup_or_total_budget_fails(self):
        for report in (guard_report(used_pidfd=False), guard_report(direct_child_reaped=False),
                       guard_report(owned_session_empty_before_reap=False),
                       guard_report(elapsed_seconds=8.01),
                       guard_report(reason='deadline', completed=False)):
            saved = []
            def fake_guard(command, seconds, poll, **kwargs):
                del command, seconds, poll, kwargs
                os.write(1, receipt())
                return report
            with self.subTest(report=report), patch.object(tested.pidfd_guard, 'supervise', side_effect=fake_guard):
                result = self.run_child(child_code(), receipt_sink=lambda raw: saved.append(raw) or True)
            self.assertEqual(saved, [])
            self.assertFalse(result['passed'] or result['receipt_sink_accepted'])

    def test_sink_output_and_exception_text_are_suppressed(self):
        code = r'''import json, os, sys
sys.path.insert(0, sys.argv[1])
from learning.task_pair_supervision_v1 import supervisor
marker = 'SYNTHETIC_PRIVATE_SINK_MARKER'
mode = sys.argv[2]
token = 'a' * 32
child = "import json,sys;v={'schema':'task-pair-training-supervisor-receipt-v1','token':sys.argv[2],'passed':True,'completed_updates':2,'auxiliary_nonzero_updates':1,'both_arms_changed':True,'arms_differ':True};print(json.dumps(v,sort_keys=True,separators=(',',':')),flush=True)"
command = [sys.executable, '-B', '-I', '-c', child, '--receipt-token', token]
contract = supervisor.Contract(2, 1, 2, 3, .1, 1, 1, supervisor.pidfd_guard.GATE_SHA256)
def sink(raw):
    assert type(raw) is bytes and token.encode() in raw
    os.write(1, marker.encode())
    os.write(2, marker.encode())
    if mode == 'raise': raise RuntimeError(marker)
    if mode == 'exit': raise SystemExit(marker)
    if mode == 'base': raise BaseException(marker)
    if mode == 'generator': raise GeneratorExit(marker)
    if mode == 'reject': return False
    return True
try:
    result = supervisor.supervise_command(command, token, contract, cancelled=lambda: False, receipt_sink=sink)
    print(json.dumps({'passed': result['passed'], 'accepted': result['receipt_sink_accepted']}))
except RuntimeError as exc:
    print(json.dumps({'error_type': type(exc).__name__, 'error_text': str(exc)}))
'''
        marker = 'SYNTHETIC_PRIVATE_SINK_MARKER'
        root = str(Path(__file__).resolve().parents[2])
        for mode in ('accept', 'reject', 'raise', 'exit', 'base', 'generator'):
            with self.subTest(mode=mode):
                process = subprocess.run([sys.executable, '-B', '-I', '-c', code, root, mode],
                                         capture_output=True, text=True, timeout=10)
                self.assertEqual(process.returncode, 0)
                self.assertNotIn(marker, process.stdout + process.stderr)
                self.assertNotIn(TOKEN, process.stdout + process.stderr)
                output = json.loads(process.stdout)
                self.assertEqual(output, {'passed': mode == 'accept', 'accepted': mode == 'accept'})

    def test_cancel_exception_after_guard_or_sink_is_suppressed(self):
        code = r'''import json, os, sys
sys.path.insert(0, sys.argv[1])
from learning.task_pair_supervision_v1 import supervisor
marker = 'SYNTHETIC_PRIVATE_POSTCANCEL_MARKER'
phase, kind = sys.argv[2:]
token = 'a' * 32
child = "import json,sys;v={'schema':'task-pair-training-supervisor-receipt-v1','token':sys.argv[2],'passed':True,'completed_updates':2,'auxiliary_nonzero_updates':1,'both_arms_changed':True,'arms_differ':True};print(json.dumps(v,sort_keys=True,separators=(',',':')),flush=True)"
command = [sys.executable, '-B', '-I', '-c', child, '--receipt-token', token]
contract = supervisor.Contract(2, 1, 2, 3, .1, 1, 1, supervisor.pidfd_guard.GATE_SHA256)
state = {'guard_done': False, 'guard_clean': False, 'sink_done': False}
real_guard = supervisor.pidfd_guard.supervise
def guard(*args, **kwargs):
    report = real_guard(*args, **kwargs)
    state['guard_done'] = True
    state['guard_clean'] = all(report[key] for key in ('used_pidfd', 'direct_child_reaped', 'owned_session_empty_before_reap'))
    return report
supervisor.pidfd_guard.supervise = guard
def cancelled():
    if (phase == 'guard' and state['guard_done']) or (phase == 'sink' and state['sink_done']):
        os.write(1, marker.encode())
        os.write(2, marker.encode())
        raise (BaseException if kind == 'base' else GeneratorExit)(marker)
    return False
def sink(raw):
    assert type(raw) is bytes and token.encode() in raw
    state['sink_done'] = True
    return True
try:
    result = supervisor.supervise_command(command, token, contract, cancelled=cancelled, receipt_sink=sink)
    print(json.dumps({'unexpected_passed': result['passed'], **state}))
except RuntimeError as exc:
    print(json.dumps({'error_type': type(exc).__name__, 'error_text': str(exc), **state}))
'''
        root = str(Path(__file__).resolve().parents[2])
        for phase in ('guard', 'sink'):
            for kind in ('base', 'generator'):
                with self.subTest(phase=phase, kind=kind):
                    process = subprocess.run([sys.executable, '-B', '-I', '-c', code, root, phase, kind],
                                             capture_output=True, text=True, timeout=10)
                    self.assertEqual(process.returncode, 0)
                    self.assertNotIn('SYNTHETIC_PRIVATE_POSTCANCEL_MARKER', process.stdout + process.stderr)
                    self.assertNotIn(TOKEN, process.stdout + process.stderr)
                    output = json.loads(process.stdout)
                    self.assertEqual(output['error_type'], 'RuntimeError')
                    self.assertIn('cleanup not confirmed', output['error_text'])
                    self.assertTrue(output['guard_done'] and output['guard_clean'])
                    self.assertEqual(output['sink_done'], phase == 'sink')

    def test_sampled_work_deadline_cleans_child_and_its_session(self):
        limits = replace(CONTRACT, work_seconds=1, gate_seconds=.5)
        result = self.run_child('import time; time.sleep(30)', limits)
        self.assertFalse(result['passed'])
        self.assertTrue(result['work_deadline_triggered'])
        self.assertTrue(result['owned_session_clean'])

    def test_started_descendant_is_cleaned_after_cancellation(self):
        ready = {'value': False}
        previous = signal.getsignal(signal.SIGUSR1)

        def on_ready(signum, frame):
            del signum, frame
            ready['value'] = True

        signal.signal(signal.SIGUSR1, on_ready)
        try:
            code = ('import os,signal,subprocess,sys,time; '
                    'subprocess.Popen([sys.executable,"-B","-I","-c","import time;time.sleep(30)"]); '
                    'os.kill(os.getppid(),signal.SIGUSR1); time.sleep(30)')
            result = self.run_child(code, CONTRACT, cancelled=lambda: ready['value'])
        finally:
            signal.signal(signal.SIGUSR1, previous)
        self.assertTrue(ready['value'])
        self.assertTrue(result['cancelled'] and result['owned_session_clean'])
        self.assertFalse(result['passed'])

    def test_systemexit_zero_in_callback_is_failure_after_child_reaped(self):
        started = []
        real_popen = tested.pidfd_guard.subprocess.Popen

        def start(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            return process

        calls = {'count': 0}

        def cancelled():
            calls['count'] += 1
            if calls['count'] <= 2:
                return False  # Enter the owned session before triggering it.
            raise SystemExit(0)

        with patch.object(tested.pidfd_guard.subprocess, 'Popen', side_effect=start):
            with self.assertRaisesRegex(RuntimeError, 'cleanup not confirmed'):
                self.run_child('import time; time.sleep(30)', cancelled=cancelled)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)

    def test_already_requested_cancellation_never_starts_child(self):
        with patch.object(tested.pidfd_guard.subprocess, 'Popen',
                          side_effect=AssertionError('Must not start')) as start:
            result = self.run_child('pass', cancelled=lambda: True)
        start.assert_not_called()
        self.assertTrue(result['cancelled'])
        self.assertFalse(result['child_started'])
        self.assertFalse(result['passed'])

    def test_prestart_callback_output_and_exception_text_are_suppressed(self):
        code = r'''import json, os, sys
sys.path.insert(0, sys.argv[1])
from learning.task_pair_supervision_v1 import supervisor
marker = 'SYNTHETIC_PRIVATE_CALLBACK_MARKER'
mode = sys.argv[2]
def cancelled():
    os.write(1, marker.encode())
    os.write(2, marker.encode())
    if mode == 'raise': raise RuntimeError(marker)
    if mode == 'exit': raise SystemExit(marker)
    if mode == 'base': raise BaseException(marker)
    if mode == 'generator': raise GeneratorExit(marker)
    return True
token = 'a' * 32
command = [sys.executable, '-B', '-I', '-c', 'pass', '--receipt-token', token]
try:
    contract = supervisor.Contract(2, 1, 2, 3, .1, 1, 1, supervisor.pidfd_guard.GATE_SHA256)
    result = supervisor.supervise_command(command, token, contract,
                                          cancelled=cancelled, receipt_sink=lambda raw: True)
    print(json.dumps({'cancelled': result['cancelled'], 'child_started': result['child_started']}))
except RuntimeError as exc:
    print(json.dumps({'error_type': type(exc).__name__, 'error_text': str(exc)}))
'''
        marker = 'SYNTHETIC_PRIVATE_CALLBACK_MARKER'
        root = str(Path(__file__).resolve().parents[2])
        for mode in ('cancel', 'raise', 'exit', 'base', 'generator'):
            with self.subTest(mode=mode):
                process = subprocess.run([sys.executable, '-B', '-I', '-c', code, root, mode],
                                         capture_output=True, text=True, timeout=10)
                self.assertEqual(process.returncode, 0)
                self.assertNotIn(marker, process.stdout)
                self.assertNotIn(marker, process.stderr)
                output = json.loads(process.stdout)
                if mode == 'cancel':
                    self.assertEqual(output, {'cancelled': True, 'child_started': False})
                else:
                    self.assertEqual(output['error_type'], 'RuntimeError')
                    self.assertIn('cleanup not confirmed', output['error_text'])

    def test_non_main_thread_is_rejected_before_child_creation(self):
        observed = []

        def call():
            try:
                self.run_child('pass', cancelled=lambda: True)
            except Exception as exc:
                observed.append((type(exc), str(exc)))

        with patch.object(tested.pidfd_guard.subprocess, 'Popen',
                          side_effect=AssertionError('Must not start')) as start:
            thread = threading.Thread(target=call)
            thread.start()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        start.assert_not_called()
        self.assertEqual(observed, [(ValueError, 'Owned-session supervision requires the main thread')])

    def test_signal_during_prestart_callback_prevents_child_creation(self):
        def cancelled():
            os.kill(os.getpid(), signal.SIGTERM)
            return False

        with patch.object(tested.pidfd_guard.subprocess, 'Popen',
                          side_effect=AssertionError('Must not start')) as start:
            result = self.run_child('pass', cancelled=cancelled)
        start.assert_not_called()
        self.assertTrue(result['cancelled'])
        self.assertFalse(result['child_started'])
        self.assertFalse(result['passed'])

    def test_cancellation_after_guard_overrides_valid_success(self):
        calls = iter((False, True))
        saved = []

        def fake_guard(command, seconds, poll, **kwargs):
            del command, seconds, poll, kwargs
            print(receipt().decode('utf-8').rstrip('\n'), flush=True)
            return guard_report()

        with patch.object(tested.pidfd_guard, 'supervise', side_effect=fake_guard):
            result = self.run_child(child_code(), cancelled=lambda: next(calls),
                                    receipt_sink=lambda raw: saved.append(raw) or True)
        self.assertTrue(result['child_exit_zero'] and result['canonical_token_receipt'])
        self.assertTrue(result['owned_session_clean'] and result['cancelled'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['receipt_sink_accepted'])
        self.assertEqual(saved, [])

    def test_cancellation_and_guard_including_cleanup_budget_are_separate(self):
        tick = time.monotonic()
        cancelled = self.run_child('import time; time.sleep(30)', CONTRACT,
                                   cancelled=lambda: time.monotonic() - tick >= .1)
        self.assertFalse(cancelled['passed'])
        self.assertTrue(cancelled['cancelled'])
        self.assertTrue(cancelled['owned_session_clean'])
        # The total budget is checked independently with a deterministic
        # synthetic report above; startup scheduling is not used as a clock.


if __name__ == '__main__':
    unittest.main()
