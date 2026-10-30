"""Mock-first and bounded owned-child pidfd tests; no GPU or real data."""

import errno
import array
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_pidfd_guard_v1 import guard


def stat_line(pid, session, pgrp, starttime, state=b'S'):
    fields = [state, b'1', str(pgrp).encode(), str(session).encode()]
    fields += [b'0'] * 15
    fields += [str(starttime).encode()]
    return str(pid).encode() + b' (synthetic ) name) ' + b' '.join(fields) + b'\n'


def anchor_probe(fd, pid):
    try:
        status = os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except OSError:
        return False
    return status is None or status.si_pid == pid


def finish_probe_after_error(probe, fd, anchored):
    try:
        probe.communicate(timeout=30)
        return True
    except subprocess.TimeoutExpired:
        if anchored is not True or fd is None:
            return False
        try:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            probe.communicate(timeout=2)
            return True
        except subprocess.TimeoutExpired:
            return False


class PidfdGuardUnitTests(unittest.TestCase):
    def test_bounded_stat_parser_extracts_identity_and_handles_parentheses(self):
        value = guard._parse_stat(stat_line(123, 100, 123, 777), 123)
        self.assertEqual((value.pid, value.session, value.process_group, value.starttime, value.state),
                         (123, 100, 123, 777, b'S'))
        kernel = guard._parse_stat(stat_line(124, 0, 0, 778), 124)
        self.assertEqual((kernel.session, kernel.process_group), (0, 0))
        for raw, pid in ((b'bad', 123), (stat_line(123, 100, 123, 777), 124),
                         (b'x' * (guard.STAT_LIMIT + 1), 123)):
            with self.assertRaises(guard.SessionGuardError) as error:
                guard._parse_stat(raw, pid)
            self.assertEqual(error.exception.code, 'scan')

    def test_pidfd_is_not_used_after_sid_or_starttime_changes(self):
        before = guard.Identity(123, 100, 123, 777, b'S')
        for after in (guard.Identity(123, 101, 123, 777, b'S'),
                      guard.Identity(123, 100, 123, 778, b'S'), None,
                      guard.Identity(123, 100, 123, 777, b'Z')):
            with self.subTest(after=after), patch.object(guard.os, 'pidfd_open', return_value=77, create=True), \
                 patch.object(guard, '_stat', return_value=after), patch.object(guard.os, 'close') as close:
                self.assertIsNone(guard._verified_pidfd(before, 100))
                close.assert_called_once_with(77)

    def test_esrch_is_disappearance_permission_and_scan_errors_are_distinct(self):
        identity = guard.Identity(123, 100, 123, 777, b'S')
        with patch.object(guard.os, 'pidfd_open', side_effect=OSError(errno.ESRCH, 'synthetic'), create=True):
            self.assertIsNone(guard._verified_pidfd(identity, 100))
        with patch.object(guard.os, 'pidfd_open', side_effect=OSError(errno.EPERM, 'synthetic'), create=True):
            with self.assertRaises(guard.SessionGuardError) as error:
                guard._verified_pidfd(identity, 100)
            self.assertEqual(error.exception.code, 'permission')
        with patch.object(guard.signal, 'pidfd_send_signal', side_effect=OSError(errno.ESRCH, 'synthetic'), create=True):
            guard._send(77, 0)
        with patch.object(guard.signal, 'pidfd_send_signal', side_effect=OSError(errno.EACCES, 'synthetic'), create=True):
            with self.assertRaises(guard.SessionGuardError) as error:
                guard._send(77, 0)
            self.assertEqual(error.exception.code, 'permission')
        with patch.object(guard.os, 'scandir', side_effect=OSError(errno.EIO, 'synthetic')):
            with self.assertRaises(guard.SessionGuardError) as error:
                guard._members(100)
            self.assertEqual(error.exception.code, 'scan')

    def test_repeated_scans_and_pidfd_signal_never_use_numeric_kill(self):
        identity = guard.Identity(123, 100, 123, 777, b'S')
        status = Mock(si_pid=100)
        with patch.object(guard, '_anchor', return_value=status) as anchor, \
             patch.object(guard, '_anchor_fd', return_value=status), \
             patch.object(guard, '_members', side_effect=[[], [identity], [], []]) as members, \
             patch.object(guard, '_verified_pidfd', return_value=77), \
             patch.object(guard, '_send') as send, patch.object(guard.os, 'close'), \
             patch.object(guard.time, 'sleep'), \
             patch.object(guard.os, 'kill', side_effect=AssertionError('Numeric kill forbidden')), \
             patch.object(guard.os, 'killpg', side_effect=AssertionError('Numeric killpg forbidden'), create=True):
            guard._cleanup(100, 98, term_seconds=1, cleanup_seconds=2)
        self.assertEqual(members.call_count, 4)
        self.assertGreaterEqual(anchor.call_count, 4)
        send.assert_called_once()

    def test_anchor_loss_and_cleanup_timeout_fail_without_signals(self):
        with patch.object(guard, '_anchor', side_effect=guard.SessionGuardError('anchor')), \
             patch.object(guard, '_anchor_fd', return_value=Mock(si_pid=100)), \
             patch.object(guard, '_send') as send:
            with self.assertRaises(guard.SessionGuardError) as error:
                guard._cleanup(100, 98, term_seconds=.1, cleanup_seconds=1)
            self.assertEqual(error.exception.code, 'anchor')
            send.assert_not_called()
        # A recycled numeric PID cannot substitute for the original pidfd.
        with patch.object(guard, '_anchor_fd', side_effect=guard.SessionGuardError('anchor')), \
             patch.object(guard, '_anchor', return_value=Mock(si_pid=100)), \
             patch.object(guard, '_send') as send:
            with self.assertRaises(guard.SessionGuardError) as error:
                guard._cleanup(100, 98, term_seconds=.1, cleanup_seconds=1)
            self.assertEqual(error.exception.code, 'anchor')
            send.assert_not_called()
        with patch.object(guard, '_anchor', return_value=None), patch.object(guard, '_anchor_fd', return_value=None), \
             patch.object(guard, '_members', return_value=[]), \
             patch.object(guard.time, 'monotonic', side_effect=[0., 2., 2.]), \
             patch.object(guard.signal, 'SIGKILL', 9, create=True), \
             patch.object(guard, '_send') as send:
            with self.assertRaises(guard.SessionGuardError) as error:
                guard._cleanup(100, 98, term_seconds=.1, cleanup_seconds=1)
            self.assertEqual(error.exception.code, 'cleanup_timeout')
            send.assert_not_called()

    def test_lost_pidfd_child_and_wait_permission_are_distinct(self):
        for exc, code in ((ChildProcessError(), 'anchor'), (PermissionError(), 'permission')):
            with self.subTest(code=code), patch.object(guard.os, 'P_PIDFD', 3, create=True), \
                 patch.object(guard.os, 'WEXITED', 4, create=True), \
                 patch.object(guard.os, 'WNOHANG', 1, create=True), \
                 patch.object(guard.os, 'WNOWAIT', 0x1000000, create=True), \
                 patch.object(guard.os, 'waitid', side_effect=exc, create=True), \
                 patch.object(guard, '_send') as send:
                with self.assertRaises(guard.SessionGuardError) as error:
                    guard._anchor_fd(77, 100)
                self.assertEqual(error.exception.code, code)
                send.assert_not_called()

    def test_unavailable_pidfd_fails_before_spawning(self):
        with patch.object(guard, '_capability', side_effect=guard.SessionGuardError('unsupported')), \
             patch.object(guard.subprocess, 'Popen', side_effect=AssertionError('Must not spawn')) as popen:
            with self.assertRaises(guard.SessionGuardError) as error:
                guard.supervise([sys.executable], 1, lambda: False, term_seconds=.1, cleanup_seconds=1,
                                gate_seconds=.5, expected_gate_sha256=guard.GATE_SHA256)
            self.assertEqual(error.exception.code, 'unsupported')
            popen.assert_not_called()

    def test_unanchored_test_probe_timeout_never_signals(self):
        probe = Mock()
        probe.communicate.side_effect = subprocess.TimeoutExpired('synthetic', 1)
        with patch.object(os, 'P_PIDFD', 3, create=True), patch.object(os, 'WEXITED', 4, create=True), \
             patch.object(os, 'WNOHANG', 1, create=True), patch.object(os, 'WNOWAIT', 0x1000000, create=True), \
             patch.object(os, 'waitid', side_effect=ChildProcessError(), create=True), \
             patch.object(signal, 'pidfd_send_signal', create=True) as send:
            anchored = anchor_probe(77, 100)
            self.assertFalse(anchored)
            self.assertFalse(finish_probe_after_error(probe, 77, anchored))
            send.assert_not_called()

    def test_pidfd_exhaustion_after_spawn_fails_without_unknown_signal(self):
        process = Mock(pid=123, returncode=None)
        process.wait.side_effect = subprocess.TimeoutExpired('synthetic', 1)
        with patch.object(guard, '_capability'), patch.object(guard, '_reserve_slot', return_value=77), \
             patch.object(guard, '_release_slot'), patch.object(guard.subprocess, 'Popen', return_value=process), \
             patch.object(guard, '_anchor', return_value=None), \
             patch.object(guard.os, 'pidfd_open', side_effect=OSError(errno.EMFILE, 'synthetic'), create=True), \
             patch.object(guard, '_send') as send:
            with self.assertRaises(guard.SessionGuardError) as error:
                guard.supervise([sys.executable], 1, lambda: False, term_seconds=.1, cleanup_seconds=1,
                                gate_seconds=.5, expected_gate_sha256=guard.GATE_SHA256)
            self.assertEqual(error.exception.code, 'pidfd')
            send.assert_not_called()
            self.assertEqual(process.wait.call_count, 2)

    def test_wrong_gate_hash_or_prestart_cancellation_never_spawns(self):
        with patch.object(guard.subprocess, 'Popen', side_effect=AssertionError('Must not spawn')) as popen:
            with self.assertRaises(guard.SessionGuardError) as error:
                guard.supervise([sys.executable], 1, lambda: False, term_seconds=.1, cleanup_seconds=1,
                                gate_seconds=.5, expected_gate_sha256='0' * 64)
            self.assertEqual(error.exception.code, 'gate')
            with patch.object(guard, '_capability'):
                with self.assertRaises(guard.SessionGuardError):
                    guard.supervise([sys.executable], 1, lambda: True, term_seconds=.1, cleanup_seconds=1,
                                    gate_seconds=.5, expected_gate_sha256=guard.GATE_SHA256)
            popen.assert_not_called()

    def test_gate_bytes_are_hash_checked_and_launched_from_the_same_snapshot(self):
        with patch.object(guard.hashlib, 'sha256', return_value=Mock(hexdigest=lambda: 'f' * 64)), \
             patch.object(guard.subprocess, 'Popen', side_effect=AssertionError('Must not spawn')) as popen:
            with self.assertRaises(guard.SessionGuardError) as error:
                guard.supervise([sys.executable], 1, lambda: False, term_seconds=.1, cleanup_seconds=1,
                                gate_seconds=.5, expected_gate_sha256=guard.GATE_SHA256)
            self.assertEqual(error.exception.code, 'gate')
            popen.assert_not_called()

        expected_source = guard._load_gate(guard.GATE_SHA256)
        process = Mock(pid=123, returncode=None)
        pipes = []
        real_pipe = os.pipe

        def capture_pipe():
            pair = real_pipe()
            pipes.append(pair)
            return pair

        with patch.object(guard, '_capability'), patch.object(guard, '_reserve_slot', return_value=77), \
             patch.object(guard, '_release_slot'), patch.object(guard.os, 'pipe', side_effect=capture_pipe), \
             patch.object(guard.subprocess, 'Popen', return_value=process) as popen, \
             patch.object(guard, '_anchor', side_effect=guard.SessionGuardError('anchor')):
            with self.assertRaises(guard.SessionGuardError):
                guard.supervise([sys.executable], 1, lambda: False, term_seconds=.1, cleanup_seconds=1,
                                gate_seconds=.5, expected_gate_sha256=guard.GATE_SHA256)
        args, options = popen.call_args
        self.assertEqual(args[0][1:5], ['-I', '-S', '-B', '-c'])
        self.assertEqual(args[0][5], expected_source)
        self.assertEqual(options['pass_fds'], (pipes[0][0],))
        self.assertNotIn(pipes[0][1], options['pass_fds'])

    def test_reservation_precedes_spawn_and_is_released_before_pidfd(self):
        events = []
        process = Mock(pid=123, returncode=None)
        with patch.object(guard, '_capability'), \
             patch.object(guard, '_reserve_slot', side_effect=lambda: events.append('reserve') or 77), \
             patch.object(guard, '_release_slot', side_effect=lambda fd: events.append('release')), \
             patch.object(guard.subprocess, 'Popen', side_effect=lambda *a, **k: events.append('spawn') or process), \
             patch.object(guard, '_anchor', return_value=None), \
             patch.object(guard.os, 'pidfd_open', side_effect=lambda *a: events.append('pidfd') or
                          (_ for _ in ()).throw(OSError(errno.EMFILE, 'synthetic')), create=True):
            with self.assertRaises(guard.SessionGuardError):
                guard.supervise([sys.executable], 1, lambda: False, term_seconds=.1, cleanup_seconds=1,
                                gate_seconds=.5, expected_gate_sha256=guard.GATE_SHA256)
        self.assertEqual(events, ['reserve', 'spawn', 'release', 'pidfd'])

    def test_exit_first_observed_at_or_after_work_deadline_is_rejected(self):
        identity = guard.Identity(123, 123, 123, 777, b'S')
        for observation, reason in ((.99, 'exited'), (1., 'deadline'), (1.01, 'deadline')):
            process = Mock(pid=123, returncode=0)
            with self.subTest(observation=observation), \
                 patch.object(guard, '_capability'), patch.object(guard, '_load_gate', return_value='pass'), \
                 patch.object(guard, '_reserve_slot', return_value=77), patch.object(guard, '_release_slot'), \
                 patch.object(guard.os, 'pipe', return_value=(11, 12)), \
                 patch.object(guard.os, 'get_inheritable', return_value=False), \
                 patch.object(guard.os, 'write', return_value=1), patch.object(guard.os, 'close'), \
                 patch.object(guard.os, 'pidfd_open', return_value=88, create=True), \
                 patch.object(guard.subprocess, 'Popen', return_value=process), \
                 patch.object(guard, '_anchor', side_effect=[None, None, Mock(si_pid=123)]), \
                 patch.object(guard, '_anchor_fd'), patch.object(guard, '_stat', return_value=identity), \
                 patch.object(guard, '_cleanup') as cleanup, \
                 patch.object(guard.time, 'monotonic', side_effect=[0., .1, observation, 1.2]):
                result = guard.supervise([sys.executable], 1, lambda: False, term_seconds=.1,
                                         cleanup_seconds=1, gate_seconds=.5,
                                         expected_gate_sha256=guard.GATE_SHA256)
            self.assertEqual(result['reason'], reason)
            self.assertEqual(result['completed'], reason == 'exited')
            self.assertTrue(result['direct_child_reaped'] and result['owned_session_empty_before_reap'])
            cleanup.assert_called_once()


@unittest.skipUnless(os.name == 'posix' and all(hasattr(os, x) for x in
                     ('pidfd_open', 'P_PIDFD', 'waitid', 'WNOWAIT'))
                     and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfd support required')
class OwnedSessionTests(unittest.TestCase):
    def test_self_exiting_direct_child_is_reaped(self):
        result = guard.supervise([sys.executable, '-B', '-I', '-c', 'pass'], 3, lambda: False,
                                 term_seconds=.1, cleanup_seconds=3, gate_seconds=2,
                                 expected_gate_sha256=guard.GATE_SHA256)
        self.assertTrue(result['completed'])
        self.assertTrue(result['owned_session_empty_before_reap'] and result['direct_child_reaped'])

    def test_early_stat_failure_closes_gate_and_reaps_direct_child(self):
        started, held = [], []
        real_popen = subprocess.Popen
        real_stat = guard._stat

        def spawn(command, **kwargs):
            process = real_popen(command, **kwargs)
            started.append(process)
            fd = os.pidfd_open(process.pid, 0)
            # A waitable pidfd pins a known direct child; no numeric signal.
            os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            held.append(fd)
            return process

        def fail_child_stat(pid):
            if pid == os.getpid():
                return real_stat(pid)
            raise guard.SessionGuardError('scan')

        try:
            with patch.object(guard.subprocess, 'Popen', side_effect=spawn), \
                 patch.object(guard, '_stat', side_effect=fail_child_stat):
                with self.assertRaises(guard.SessionGuardError) as error:
                    guard.supervise([sys.executable, '-B', '-I', '-c', 'import time;time.sleep(8)'],
                                    3, lambda: False, term_seconds=.1, cleanup_seconds=3,
                                    gate_seconds=2, expected_gate_sha256=guard.GATE_SHA256)
            self.assertEqual(error.exception.code, 'scan')
            self.assertEqual(len(started), 1)
            self.assertEqual(started[0].returncode, 73)  # Gate saw EOF; target never executed.
        finally:
            if started and started[0].returncode is None:
                if held:
                    try:
                        signal.pidfd_send_signal(held[0], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                started[0].wait(timeout=9)
            for fd in held:
                os.close(fd)

    def test_pidfd_exhaustion_closes_unreleased_gate_without_numeric_signal(self):
        started, held = [], []
        real_popen, real_pidfd = subprocess.Popen, os.pidfd_open

        def spawn(command, **kwargs):
            process = real_popen(command, **kwargs)
            started.append(process)
            fd = real_pidfd(process.pid, 0)
            os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            held.append(fd)
            return process

        try:
            with patch.object(guard, '_capability'), \
                 patch.object(guard.subprocess, 'Popen', side_effect=spawn), \
                 patch.object(guard.os, 'pidfd_open', side_effect=OSError(errno.EMFILE, 'synthetic')), \
                 patch.object(guard.os, 'kill', side_effect=AssertionError('Numeric kill forbidden')) as kill, \
                 patch.object(guard, '_send') as send:
                with self.assertRaises(guard.SessionGuardError) as error:
                    guard.supervise([sys.executable, '-B', '-I', '-c', 'pass'], 3, lambda: False,
                                    term_seconds=.1, cleanup_seconds=3, gate_seconds=2,
                                    expected_gate_sha256=guard.GATE_SHA256)
            self.assertEqual(error.exception.code, 'pidfd')
            self.assertEqual(started[0].returncode, 73)
            kill.assert_not_called()
            send.assert_not_called()
        finally:
            if started and started[0].returncode is None:
                if held:
                    try:
                        signal.pidfd_send_signal(held[0], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                started[0].wait(timeout=5)
            for fd in held:
                os.close(fd)

    def test_cancellation_before_release_closes_gate_without_target(self):
        started, held = [], []
        real_popen = subprocess.Popen
        votes = iter((False, True))

        def spawn(command, **kwargs):
            process = real_popen(command, **kwargs)
            started.append(process)
            fd = os.pidfd_open(process.pid, 0)
            os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            held.append(fd)
            return process

        try:
            with patch.object(guard.subprocess, 'Popen', side_effect=spawn):
                with self.assertRaises(guard.SessionGuardError) as error:
                    guard.supervise([sys.executable, '-B', '-I', '-c', 'pass'], 3, lambda: next(votes),
                                    term_seconds=.1, cleanup_seconds=3, gate_seconds=2,
                                    expected_gate_sha256=guard.GATE_SHA256)
            self.assertEqual(error.exception.code, 'gate')
            self.assertEqual(started[0].returncode, 73)
        finally:
            if started and started[0].returncode is None:
                if held:
                    try:
                        signal.pidfd_send_signal(held[0], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                started[0].wait(timeout=5)
            for fd in held:
                os.close(fd)

    def test_gate_timeout_and_sole_writer_exit_do_not_execute_target(self):
        source = guard._load_gate(guard.GATE_SHA256)
        for mode in ('timeout', 'writer_exit'):
            with self.subTest(mode=mode):
                read, write = os.pipe()
                gate = writer = None
                held = []
                try:
                    gate = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', source, str(read), '.3',
                                             sys.executable, '-B', '-I', '-c', 'pass'],
                                            pass_fds=(read,), start_new_session=True)
                    held.append(os.pidfd_open(gate.pid, 0))
                    os.waitid(os.P_PIDFD, held[-1], os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    os.close(read)
                    read = None
                    if mode == 'writer_exit':
                        writer = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import os;os._exit(0)'],
                                                  pass_fds=(write,))
                        held.append(os.pidfd_open(writer.pid, 0))
                        os.waitid(os.P_PIDFD, held[-1], os.WEXITED | os.WNOHANG | os.WNOWAIT)
                        os.close(write)
                        write = None
                        writer.wait(timeout=3)
                    gate.wait(timeout=3)
                    self.assertEqual(gate.returncode, 74 if mode == 'timeout' else 73)
                finally:
                    for pipe in (read, write):
                        if pipe is not None:
                            os.close(pipe)
                    for index, process in enumerate((gate, writer)):
                        if process is not None and process.returncode is None:
                            if index < len(held):
                                try:
                                    signal.pidfd_send_signal(held[index], signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            process.wait(timeout=5)
                    for fd in held:
                        os.close(fd)

    def test_different_group_same_session_with_pidfd_and_self_exit_safety(self):
        # A dedicated subprocess owns the subreaper and all test descendants;
        # this unittest runner never calls waitpid(-1).
        outer, inner = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        received = []
        probe = None
        probe_fd = None
        probe_fd_anchored = False
        timed_out = False
        result = None
        cleanup_confirmed = True
        code = r'''import runpy,signal,socket,sys
module = runpy.run_path(sys.argv[1])
sender = socket.socket(fileno=int(sys.argv[2]))
def timeout(signum, frame): raise TimeoutError('synthetic probe timeout')
signal.signal(signal.SIGALRM, timeout)
signal.alarm(25)
try:
    module['OwnedSessionTests']('test_different_group_same_session_with_pidfd_and_self_exit_safety')._cross_group_probe(sender)
    print('isolated-owned-session-ok', flush=True)
finally:
    signal.alarm(0)
    sender.close()
'''
        try:
            probe = subprocess.Popen([sys.executable, '-B', '-I', '-c', code, __file__, str(inner.fileno())],
                                     pass_fds=(inner.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            inner.close()
            probe_fd = os.pidfd_open(probe.pid, 0)
            probe_fd_anchored = anchor_probe(probe_fd, probe.pid)
            if not probe_fd_anchored:
                raise AssertionError('Synthetic probe pidfd was not anchored')
            try:
                out, _ = probe.communicate(timeout=38)
                result = (probe.returncode, out.strip())
            except subprocess.TimeoutExpired:
                timed_out = True
                # The probe is our unreaped direct child; use its pinned fd.
                if not probe_fd_anchored:
                    raise AssertionError('Synthetic probe pidfd was not anchored')
                try:
                    signal.pidfd_send_signal(probe_fd, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    probe.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        signal.pidfd_send_signal(probe_fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    probe.communicate(timeout=2)
        finally:
            inner.close()
            if probe is not None and probe.returncode is None:
                # If setup failed, do not signal an unpinned PID. The isolated
                # probe has its own alarm and EOF/8s safety exits.
                if not finish_probe_after_error(probe, probe_fd, probe_fd_anchored):
                    cleanup_confirmed = False
            outer.setblocking(False)
            while True:
                try:
                    payload, ancillary, flags, _ = outer.recvmsg(128, socket.CMSG_SPACE(array.array('i').itemsize))
                except BlockingIOError:
                    break
                if not payload:
                    break
                descriptors = array.array('i')
                for level, kind, data in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        descriptors.frombytes(data[:len(data) - (len(data) % descriptors.itemsize)])
                if flags & socket.MSG_CTRUNC or len(descriptors) != 1:
                    for fd in descriptors:
                        os.close(fd)
                    cleanup_confirmed = False
                    continue
                try:
                    label, pid, started = payload.decode('ascii').split(':')
                    received.append((label, int(pid), int(started), descriptors[0]))
                except (UnicodeError, ValueError):
                    os.close(descriptors[0])
                    cleanup_confirmed = False
            outer.close()
            deadline = time.monotonic() + 10
            for _, pid, started, fd in received:
                try:
                    poller = select.poll()
                    poller.register(fd, select.POLLIN)
                    if not poller.poll(0):
                        # Only pidfds validated and sent by the isolated probe.
                        try:
                            signal.pidfd_send_signal(fd, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    while time.monotonic() < deadline:
                        identity = guard._stat(pid)
                        if poller.poll(0) and (identity is None or identity.starttime != started):
                            break
                        time.sleep(.02)
                    else:
                        cleanup_confirmed = False
                except (OSError, guard.SessionGuardError):
                    cleanup_confirmed = False
                finally:
                    os.close(fd)
            if probe_fd is not None:
                os.close(probe_fd)
        self.assertFalse(timed_out, 'Isolated probe timed out; cleanup must be treated as unconfirmed')
        self.assertTrue(cleanup_confirmed, 'Owned test processes not confirmed exited and reaped')
        self.assertEqual({item[0] for item in received}, {'leader', 'descendant'})
        self.assertEqual(result, (0, b'isolated-owned-session-ok'))

    def _cross_group_probe(self, sender):
        # All children are synthetic and exit on safety-pipe EOF or after 8s.
        # This test process becomes a subreaper only for the test's lifetime,
        # so it can wait for the known orphan via pidfd; restore the old setting.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        original = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(original), 0, 0, 0) != 0:
            self.skipTest('Process-local subreaper unavailable')
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            self.skipTest('Process-local subreaper unavailable')
        allocated = []
        try:
            safety_r, safety_w = os.pipe()
            allocated.extend((safety_r, safety_w))
            ready_r, ready_w = os.pipe()
            allocated.extend((ready_r, ready_w))
            os.set_blocking(ready_r, False)
        except BaseException:
            for fd in allocated:
                os.close(fd)
            libc.prctl(36, original.value, 0, 0, 0)
            raise
        started, leader_fds, descendant_fds = [], [], []
        buffer = bytearray()
        ready = {'value': False}
        descendant = {'pid': None}
        fd_targets = {}
        sent_descendant = {'value': False}
        direct_reaped = descendant_reaped = no_children = False
        cleanup_error = False
        real_popen = subprocess.Popen
        real_verified = guard._verified_pidfd
        real_send = guard._send
        code = r'''import os,select,sys
safety, ready = int(sys.argv[1]), int(sys.argv[2])
child = os.fork()
if child == 0:
    os.setpgid(0, 0)
    os.write(ready, (str(os.getpid()) + '\n').encode('ascii'))
    select.select([safety], [], [], 8)
    os._exit(0)
select.select([safety], [], [], 8)
try: os.waitpid(child, 0)
except ChildProcessError: pass
os._exit(0)
'''

        def send_handle(label, identity, fd):
            payload = f'{label}:{identity.pid}:{identity.starttime}'.encode('ascii')
            sender.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                       array.array('i', [fd]))])

        def spawn(command, **kwargs):
            kwargs['pass_fds'] = tuple(kwargs.get('pass_fds', ())) + (safety_r, ready_w)
            process = real_popen(command, **kwargs)
            started.append(process)
            identity = guard._stat(process.pid)
            if (identity is None or identity.session != process.pid
                    or identity.process_group != process.pid):
                raise AssertionError('Synthetic leader identity differs')
            fd = guard._verified_pidfd(identity, process.pid)
            if fd is None:
                raise AssertionError('Synthetic leader pidfd unavailable')
            leader_fds.append(fd)
            send_handle('leader', identity, fd)
            return process

        def cancelled():
            if ready['value']:
                return True
            try:
                buffer.extend(os.read(ready_r, 64))
            except BlockingIOError:
                return False
            if b'\n' not in buffer:
                return False
            pid = int(bytes(buffer).split(b'\n', 1)[0])
            identity = guard._stat(pid)
            if (identity is None or not started or identity.session != started[0].pid
                    or identity.process_group != pid or pid == started[0].pid):
                raise AssertionError('Synthetic descendant identity differs')
            descendant['pid'] = pid
            fd = guard._verified_pidfd(identity, started[0].pid)
            if fd is None:
                raise AssertionError('Synthetic descendant pidfd unavailable')
            descendant_fds.append(fd)
            send_handle('descendant', identity, fd)
            ready['value'] = True
            return True

        def observe_verified(identity, session):
            fd = real_verified(identity, session)
            if fd is not None:
                fd_targets[fd] = identity.pid
            return fd

        def observe_send(fd, sig):
            if descendant['pid'] is not None and fd_targets.get(fd) == descendant['pid']:
                sent_descendant['value'] = True
            return real_send(fd, sig)

        try:
            command = [sys.executable, '-B', '-I', '-c', code, str(safety_r), str(ready_w)]
            with patch.object(guard.subprocess, 'Popen', side_effect=spawn), \
                 patch.object(guard, '_verified_pidfd', side_effect=observe_verified), \
                 patch.object(guard, '_send', side_effect=observe_send):
                result = guard.supervise(command, 5, cancelled, term_seconds=.2, cleanup_seconds=4,
                                         gate_seconds=2, expected_gate_sha256=guard.GATE_SHA256)
            self.assertTrue(ready['value'])
            self.assertTrue(sent_descendant['value'])
            self.assertEqual(result['reason'], 'cancelled')
            self.assertTrue(result['used_pidfd'] and result['owned_session_empty_before_reap'])
            self.assertTrue(result['direct_child_reaped'])
        finally:
            # EOF is an independent bounded self-exit even if the guard failed.
            for fd in (safety_w, safety_r, ready_w, ready_r):
                try:
                    os.close(fd)
                except OSError:
                    cleanup_error = True
            for fd in descendant_fds + leader_fds:
                # Signal only already-verified, identity-stable pidfds.
                try:
                    poller = select.poll()
                    poller.register(fd, select.POLLIN)
                    if not poller.poll(0):
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                except OSError as exc:
                    if exc.errno != errno.ESRCH:
                        cleanup_error = True
            if started:
                try:
                    # The independent safety pipe guarantees this synthetic
                    # child exits on EOF or at its 8-second deadline.
                    started[0].wait(timeout=9)
                    direct_reaped = started[0].returncode is not None
                except subprocess.TimeoutExpired:
                    direct_reaped = False
            if descendant_fds:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        status = os.waitid(os.P_PIDFD, descendant_fds[0], os.WEXITED | os.WNOHANG)
                    except ChildProcessError:
                        # The leader may have reaped this known descendant.
                        poller = select.poll()
                        poller.register(descendant_fds[0], select.POLLIN)
                        if poller.poll(0):
                            descendant_reaped = True
                            break
                        status = None
                    if status is not None:
                        descendant_reaped = True
                        break
                    time.sleep(.02)
            # The isolated test process spawned no unrelated children; as a
            # subreaper it must collect even a child whose handshake failed.
            deadline = time.monotonic() + 9
            while time.monotonic() < deadline:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    no_children = True
                    break
                if pid == 0:
                    time.sleep(.02)
            for fd in descendant_fds + leader_fds:
                try:
                    os.close(fd)
                except OSError:
                    cleanup_error = True
            if libc.prctl(36, original.value, 0, 0, 0) != 0:
                raise AssertionError('Could not restore process-local subreaper setting')
            if cleanup_error or not no_children:
                raise AssertionError('Synthetic child cleanup incomplete')
        self.assertTrue(direct_reaped)
        self.assertTrue(descendant_reaped)


if __name__ == '__main__':
    unittest.main()
