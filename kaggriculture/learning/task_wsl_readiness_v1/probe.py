"""Brief owned WSL warm-up and sampled Windows memory check.

This is a necessary condition only: it does not load JAX, test a GPU, start a
training attempt, or guarantee future memory headroom. The final-window claim
means consecutive *samples* with a bounded sampling gap, not continuous
observation between samples. No process is terminated by this module.
"""
from collections import deque
from dataclasses import dataclass, field, replace
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


GIB = 1024 ** 3
START_FLOOR = 5 * GIB
RUN_FLOOR = 4 * GIB
SLEEP_SECONDS = 12
TAIL_SECONDS = 2.0
POLL_SECONDS = 0.25
MAX_SAMPLE_GAP = 0.75
TRANSPORT_DEADLINE = 45.0
MAX_CONTRACT_BYTES = 65536


@dataclass(frozen=True, slots=True, repr=False)
class ProbeResult:
    """Sanitized outcome; an unconfirmed owned process stays private here."""

    readiness: object
    reason: str
    started: object
    transport_reaped: bool
    exit_code: object
    _process: object = field(default=None, repr=False, compare=False)

    def __repr__(self):
        return '<PrivateWslReadinessResult>'

    def public(self):
        return dict(readiness=self.readiness, reason=self.reason, started=self.started,
                    transport_reaped=self.transport_reaped, exit_code=self.exit_code,
                    sampling='bounded_samples_not_continuous', sufficient_for_gpu_training=False)

    def wait_retained(self, *, timeout=10.0):
        """Keep waiting on an owned handle; never upgrade an unknown result."""
        if self._process is None:
            return self
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 30:
            raise ValueError('Invalid bounded wait')
        try:
            code = self._process.wait(timeout=timeout)
            if type(code) is not int:
                return self
            return replace(self, transport_reaped=True, exit_code=code, _process=None)
        except BaseException:
            # Keep the same handle even on timeout, interruption or wait error.
            return self


def _clock(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid monotonic time')
    return float(value)


def _available(value):
    if type(value) is not int or value < 0:
        raise ValueError('Invalid memory sample')
    return value


def windows_available():
    """Return current Windows available physical memory in bytes, locally."""
    if os.name != 'nt':
        raise OSError('Windows memory sampler unavailable')

    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('memory_load', ctypes.c_uint32),
                    ('total_phys', ctypes.c_uint64), ('available_phys', ctypes.c_uint64),
                    ('total_page_file', ctypes.c_uint64), ('available_page_file', ctypes.c_uint64),
                    ('total_virtual', ctypes.c_uint64), ('available_virtual', ctypes.c_uint64),
                    ('available_extended_virtual', ctypes.c_uint64)]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('Windows memory sample failed')
    return _available(int(value.available_phys))


def _distribution(value):
    if (type(value) is not str or not 1 <= len(value) <= 128 or value != value.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('Invalid private WSL identity')
    return value


def _load_distribution(path):
    """Read only the established private host contract when the CLI runs."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Private WSL contract unavailable')
    with path.open('rb') as source:
        raw = source.read(MAX_CONTRACT_BYTES + 1)
    if not 0 < len(raw) <= MAX_CONTRACT_BYTES:
        raise ValueError('Private WSL contract unavailable')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Invalid private WSL contract')
            result[key] = value
        return result
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid contract')))
    # The existing consumer authenticates/uses the distribution field, but
    # does not require a specific host-contract schema revision.
    if type(value) is not dict:
        raise ValueError('Invalid private WSL contract')
    return _distribution(value.get('distribution'))


def _wait_after_error(process, reason):
    """Best-effort wait only; retain the actual owned handle on uncertainty."""
    result = ProbeResult(None, reason, True, False, None, process)
    return result.wait_retained(timeout=30.0)


def probe(wsl_executable, distribution, *, available=windows_available, clock=time.monotonic,
          sleep=time.sleep, popen=subprocess.Popen):
    """Start one owned sleep only after the memory gate, and wait for it.

    Callbacks are injectable for synthetic tests. The caller authenticates the
    executable and private distribution; inputs and memory values are never
    included in the result. An unknown result with an owned process retains its
    handle and can be polled using ``wait_retained`` without killing it.
    """
    try:
        if (type(wsl_executable) is not str or not wsl_executable or '\x00' in wsl_executable
                or not all(callable(item) for item in (available, clock, sleep, popen))):
            raise ValueError('Invalid probe input')
        _distribution(distribution)
        start = _clock(clock())
        if _available(available()) < START_FLOOR:
            return ProbeResult(False, 'initial_memory_low', False, False, None)
    except BaseException:
        return ProbeResult(None, 'initial_check_unknown', False, False, None)

    try:
        process = popen([wsl_executable, '--distribution', distribution, '--exec', '/bin/sleep',
                         str(SLEEP_SECONDS)], shell=False, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000), close_fds=True)
    except BaseException:
        # A constructor error can occur after OS process creation. No handle
        # was returned, so neither creation nor cleanup can be asserted.
        return ProbeResult(None, 'spawn_unconfirmed', None, False, None)

    try:
        spawned = _clock(clock())
        if spawned < start:
            raise ValueError('Monotonic time moved backward')
    except BaseException:
        return _wait_after_error(process, 'post_spawn_clock_unknown')
    tail, previous, below_running, gap = deque(), start, False, False
    try:
        while True:
            before_sample = _clock(clock())
            if before_sample < previous:
                raise ValueError('Monotonic time moved backward')
            sample = _available(available())
            sample_time = _clock(clock())
            if sample_time < before_sample:
                raise ValueError('Monotonic time moved backward')
            if sample_time - previous > MAX_SAMPLE_GAP:
                gap = True
            previous = sample_time
            below_running = below_running or sample < RUN_FLOOR
            tail.append((sample_time, sample >= START_FLOOR))
            while tail and tail[0][0] < sample_time - TAIL_SECONDS - MAX_SAMPLE_GAP:
                tail.popleft()
            code = process.poll()
            now = _clock(clock())
            if now < sample_time:
                raise ValueError('Monotonic time moved backward')
            if now - sample_time > MAX_SAMPLE_GAP:
                gap = True
            if code is not None:
                if type(code) is not int or process.wait(timeout=0) != code:
                    raise ValueError('Transport exit cannot be verified')
                if now - start >= TRANSPORT_DEADLINE:
                    return ProbeResult(None, 'exit_observed_after_deadline', True, True, code)
                if code != 0:
                    return ProbeResult(False, 'transport_nonzero', True, True, code)
                if now - spawned < SLEEP_SECONDS:
                    return ProbeResult(False, 'transport_too_short', True, True, code)
                if below_running:
                    return ProbeResult(False, 'running_memory_low', True, True, code)
                in_tail = [(stamp, ok) for stamp, ok in tail if stamp >= now - TAIL_SECONDS]
                if not in_tail or in_tail[0][0] > now - TAIL_SECONDS + MAX_SAMPLE_GAP:
                    return ProbeResult(None, 'incomplete_sampling', True, True, code)
                if not all(ok for _stamp, ok in in_tail):
                    return ProbeResult(False, 'tail_memory_low', True, True, code)
                if gap:
                    return ProbeResult(None, 'incomplete_sampling', True, True, code)
                return ProbeResult(True, 'sampled_ready', True, True, code)
            if now - start >= TRANSPORT_DEADLINE:
                return ProbeResult(None, 'transport_unconfirmed', True, False, None, process)
            sleep(POLL_SECONDS)
    except BaseException:
        return _wait_after_error(process, 'sampling_or_exit_unknown')


def main():
    """Read the private host contract only here; never launch training."""
    if os.name != 'nt':
        raise OSError('Windows host required')
    from learning.task_pair_wsl_launch_v1.host import _system_wsl
    contract = (Path(__file__).resolve().parents[5] / 'kaggriculture' / 'artifacts'
                / 'task-pair-wsl-contract-v1-20261030' / 'host-contract.local.json')
    distribution = _load_distribution(contract)
    result = probe(str(_system_wsl()), distribution)
    def emit(value):
        try:
            print(json.dumps(value, allow_nan=False), flush=True)
        except BaseException:
            # A broken output stream must not discard a still-owned handle.
            pass
    emit(result.public())
    # On uncertainty keep the only owned handle in this process until the
    # transport exits naturally. A later exit does not change readiness.
    while result._process is not None:
        began_wait = time.monotonic()
        try:
            result = result.wait_retained(timeout=10.0)
        except BaseException:
            pass
        if result._process is not None:
            # A persistent wait failure can return immediately. Keep the
            # handle without burning a core while waiting for natural exit.
            try:
                pause = 1.0 - (time.monotonic() - began_wait)
                if pause > 0:
                    time.sleep(pause)
            except BaseException:
                pass
    if result.readiness is None and result.transport_reaped:
        emit(dict(readiness=None, transport_reaped=True, exit_code=result.exit_code))
    return 0 if result.readiness is True else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        print(json.dumps(dict(readiness=None, reason='setup_unknown',
                              sufficient_for_gpu_training=False)), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
