"""Cooperative, bounded resource preflight before an unstarted game.

The caller supplies an availability function and a monotonic deadline.  It
must establish that no game is running, serialize calls using the same budget,
bind that budget to its panel, and check its wall-clock deadline again before
launching.  This module starts no jobs and records no sampled memory values.
A synchronous callback can block past a deadline; the checks here are not a
hard interruption mechanism.
"""
from dataclasses import dataclass
import math
import time


MIN_AVAILABLE_BYTES = 4 * 1024**3
SAMPLES = 3
SAMPLE_INTERVAL_SECONDS = 10.0
RETRY_INTERVAL_SECONDS = 30.0
SINGLE_WAIT_SECONDS = 15.0 * 60
TOTAL_WAIT_SECONDS = 45.0 * 60


@dataclass
class WaitBudget:
    """Carry waiting time; ``blocked_seconds`` is incomplete when unknown."""
    blocked_seconds: float = 0.0
    terminal: bool = False
    duration_known: bool = True


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    reason: str
    attempts: int
    samples: int
    shortages: int
    last_shortage_sample: int | None
    blocked_seconds: float | None
    total_blocked_seconds: float | None


class _Stop(Exception):
    def __init__(self, reason):
        self.reason = reason


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (ValueError, OverflowError):
        return False


def wait_for_preflight(*, budget, deadline, available, job_not_started,
                       clock=time.monotonic, sleep=time.sleep):
    """Return once all samples pass or a fixed safety condition stops the panel.

    ``deadline`` is exclusive and uses the same monotonic clock as ``clock``.
    On a shortage, blocked time starts at the beginning of that sample and
    ends after the complete passing preflight or at failure.  Initial passing
    sampling does not consume blocked time.  Every failed result makes this
    budget terminal; retrying an already-started game is outside this API.
    The counters describe calls, not memory quantities.  No callback exception
    text or sampled memory value is returned.
    """
    if not isinstance(budget, WaitBudget):
        raise TypeError('WaitBudget required')
    if (type(budget.terminal) is not bool or type(budget.duration_known) is not bool
        or not _number(budget.blocked_seconds) or budget.blocked_seconds < 0
        or (not budget.duration_known and not budget.terminal)):
        raise ValueError('Invalid wait budget')
    initial = float(budget.blocked_seconds)
    attempts = samples = shortages = 0
    last_shortage_sample = None
    blocked_since = None
    last_clock = None

    def result(passed, reason):
        if not passed:
            budget.terminal = True
        if reason == 'clock_error' or (blocked_since is not None and last_clock is None):
            # A broken clock cannot substantiate a duration.  The terminal
            # flag prevents reuse with an underestimated cumulative budget.
            budget.duration_known = False
            return PreflightResult(False, reason, attempts, samples, shortages,
                                   last_shortage_sample, None, None)
        if blocked_since is None:
            blocked = 0.0
        else:
            blocked = max(0.0, last_clock - blocked_since)
        budget.blocked_seconds = initial + blocked
        return PreflightResult(passed, reason, attempts, samples, shortages,
                               last_shortage_sample, blocked, budget.blocked_seconds)

    if budget.terminal:
        if not budget.duration_known:
            return PreflightResult(False, 'budget_terminal', 0, 0, 0, None, 0.0, None)
        return result(False, 'budget_terminal')
    if job_not_started is not True:
        return result(False, 'job_active_or_unknown')
    if not _number(deadline):
        return result(False, 'invalid_deadline')
    if initial >= TOTAL_WAIT_SECONDS:
        return result(False, 'total_wait_limit')
    deadline = float(deadline)

    def tick():
        nonlocal last_clock
        try:
            value = clock()
        except Exception:
            raise _Stop('clock_error') from None
        if not _number(value):
            raise _Stop('clock_error')
        value = float(value)
        if last_clock is not None and value < last_clock:
            raise _Stop('clock_error')
        last_clock = value
        return value

    def bounds(now):
        if now >= deadline:
            raise _Stop('deadline')
        if blocked_since is not None:
            blocked = now - blocked_since
            if blocked >= SINGLE_WAIT_SECONDS:
                raise _Stop('single_wait_limit')
            if initial + blocked >= TOTAL_WAIT_SECONDS:
                raise _Stop('total_wait_limit')

    def interval_after(start, seconds):
        target = start + seconds
        # Very large but finite clocks can lose the interval to rounding.
        # Never silently skip a required delay or spin without progress.
        if not math.isfinite(target) or target <= start:
            raise _Stop('clock_error')
        if target - start < seconds:
            target = math.nextafter(target, math.inf)
        if not math.isfinite(target) or target - start < seconds:
            raise _Stop('clock_error')
        return target

    def sleep_until(target):
        # Retry an early wake, but fail closed if a broken sleeper repeatedly
        # wakes without reaching its target or the monotonic clock stalls.
        for _ in range(128):
            now = tick()
            bounds(now)
            if now >= target:
                return
            stop = deadline
            if blocked_since is not None:
                stop = min(stop, blocked_since + SINGLE_WAIT_SECONDS,
                           blocked_since + TOTAL_WAIT_SECONDS - initial)
            try:
                sleep(min(target, stop) - now)
            except Exception:
                # A sleeper may fail after spending some time.  Re-read the
                # clock rather than recording the pre-sleep time as the end.
                tick()
                raise _Stop('sleep_error') from None
            after = tick()
            if after <= now:
                raise _Stop('clock_error')
            bounds(after)
        raise _Stop('sleep_error')

    try:
        while True:
            attempt_start = tick()
            bounds(attempt_start)
            attempts += 1
            for index in range(SAMPLES):
                sample_start = tick()
                bounds(sample_start)
                samples += 1
                try:
                    value = available()
                except Exception:
                    tick()
                    raise _Stop('sample_error') from None
                now = tick()
                bounds(now)
                if type(value) is not int or value < 0:
                    raise _Stop('invalid_sample')
                if value < MIN_AVAILABLE_BYTES:
                    shortages += 1
                    last_shortage_sample = index + 1
                    if blocked_since is None:
                        blocked_since = sample_start
                    bounds(now)
                    sleep_until(interval_after(attempt_start, RETRY_INTERVAL_SECONDS))
                    break
                if index + 1 < SAMPLES:
                    sleep_until(interval_after(now, SAMPLE_INTERVAL_SECONDS))
            else:
                return result(True, 'ready')
    except _Stop as stopped:
        return result(False, stopped.reason)
    except BaseException:
        # Do not swallow user interrupts.  The elapsed blocked time is no
        # longer reliable, so the same budget must never be reused.
        budget.terminal = True
        budget.duration_known = False
        raise
