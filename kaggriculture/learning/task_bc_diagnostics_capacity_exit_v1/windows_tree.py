"""Bounded natural completion after root exit, using the frozen Windows owner.

CreateProcess, atomic JOB_LIST containment, retained native identities, limits,
membership and resource sampling are inherited unchanged from capacity_v1.
Only completion and final cleanup differ. A zero-code launcher is not proof
that its whole Job has ended. All native identities stay in LOCAL details.
"""
from __future__ import annotations

import math
import subprocess
import time

from learning.task_bc_diagnostics_capacity_v1 import windows_tree as base

TREE_MEMORY_LIMIT = base.TREE_MEMORY_LIMIT
MIN_AVAILABLE_MEMORY = base.MIN_AVAILABLE_MEMORY
CLOSE_SECONDS = base.CLOSE_SECONDS
POLL_SECONDS = base.POLL_SECONDS
FORCED_EXIT_CODE = base.FORCED_EXIT_CODE
TreeControlError = base.TreeControlError
ResourceLimitError = base.ResourceLimitError
available_memory_bytes = base.available_memory_bytes
current_identity = base.current_identity
NATURAL_SECONDS = 2.0
NATURAL_PROOF = "job_active_processes_zero_and_retained_identity_handles_signaled"


class OwnedTree(base.OwnedTree):
    """Reuse the original owner; do not restart its ten-second close allowance."""

    def _empty(self, active):
        root_done = self.process is None or self.api.signaled(self.process)
        identities_done = all(self.api.signaled(handle) for handle, _ in self.processes.values())
        return active == 0 and root_done and identities_done

    def close(self, *, natural=False, work_deadline=None, phase_deadline=None):
        if self.closed:
            return
        cleanup_start = time.monotonic()
        deadline = cleanup_start + CLOSE_SECONDS
        if phase_deadline is not None:
            deadline = min(deadline, phase_deadline)
        errors = []
        self.lifecycle.setdefault("natural_tree_exit", False)
        self.lifecycle.setdefault("natural_wait_seconds", 0.0)
        self.lifecycle.setdefault("active_before_force", None)
        self.lifecycle["cleanup_started_monotonic"] = cleanup_start
        self.lifecycle["cleanup_deadline_monotonic"] = deadline
        with base._defer_console_interrupts():
            try:
                if self.job:
                    self.lifecycle["active_processes_before_close"] = self.sample()
                    if natural:
                        if work_deadline is None or self.poll() != 0:
                            raise TreeControlError("Natural drain requires a zero root and its original work deadline")
                        natural_start = self.lifecycle.get("root_exit_observed_monotonic", cleanup_start)
                        natural_deadline = min(natural_start + NATURAL_SECONDS, work_deadline, deadline)
                        self.lifecycle.update(natural_started_monotonic=natural_start,
                            natural_deadline_monotonic=natural_deadline,
                            natural_limit_reason=("original_work_deadline" if work_deadline <= min(natural_start + NATURAL_SECONDS, deadline)
                                else "phase_or_cleanup_deadline" if deadline < natural_start + NATURAL_SECONDS else "fixed_grace"))
                        while True:
                            # An empty observation made after the bound cannot prove
                            # completion before it. Do not renew a deadline on success.
                            if time.monotonic() >= natural_deadline:
                                break
                            active = self.sample()
                            if available_memory_bytes() < MIN_AVAILABLE_MEMORY:
                                raise ResourceLimitError("Host available physical memory fell below 4 GiB during natural drain")
                            if time.monotonic() >= natural_deadline:
                                break
                            if self._empty(active):
                                self.lifecycle.update(natural_tree_exit=True, natural_exit_proof=NATURAL_PROOF)
                                break
                            time.sleep(min(POLL_SECONDS, max(0.0, natural_deadline - time.monotonic())))
                        self.lifecycle["natural_wait_seconds"] = time.monotonic() - natural_start
                    active = self.sample()
                    if not self._empty(active):
                        self.lifecycle.update(active_before_force=active, termination_requested=True)
                        self.api.check(self.api.k.TerminateJobObject(self.job, FORCED_EXIT_CODE), "TerminateJobObject")
                    while True:
                        active = self.sample()
                        self.lifecycle["active_processes_final"] = active
                        if self._empty(active):
                            if time.monotonic() > deadline:
                                raise TreeControlError("Closure proof arrived outside the original total close allowance")
                            self.lifecycle.update(workers_closed=True, owned_child_closed=True, closure_proof=NATURAL_PROOF)
                            if self.process:
                                self.lifecycle["direct_returncode"] = self.api.exit_code(self.process)
                            break
                        if time.monotonic() >= deadline:
                            raise TreeControlError("Tree closure exceeded the original total close allowance")
                        time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
                elif self.process is None:
                    self.lifecycle.update(workers_closed=True, owned_child_closed=True, closure_proof="no_process_created")
                else:
                    raise TreeControlError("Process exists without its owned job")
            except BaseException as error:
                errors.append(error)
                self.lifecycle["workers_closed"] = False
                if natural and "natural_started_monotonic" in self.lifecycle:
                    self.lifecycle["natural_wait_seconds"] = time.monotonic() - self.lifecycle["natural_started_monotonic"]
                # A resource/query error during drain must still request owned
                # termination. Closing the last handle is only a final safeguard.
                if self.job and not self.lifecycle.get("termination_requested"):
                    try:
                        self.lifecycle["active_before_force"] = int(self.api.query(self.job, 1, base._Accounting).ActiveProcesses)
                        self.lifecycle["termination_requested"] = True
                        self.api.check(self.api.k.TerminateJobObject(self.job, FORCED_EXIT_CODE), "TerminateJobObject(error)")
                        while time.monotonic() < deadline:
                            active = self.sample()
                            self.lifecycle["active_processes_final"] = active
                            if self._empty(active):
                                self.lifecycle["error_cleanup_proof"] = NATURAL_PROOF
                                break
                            time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
                    except BaseException as cleanup_error:
                        errors.append(cleanup_error)
            finally:
                handles = {handle for handle, _ in self.processes.values()}
                if self.process:
                    handles.add(self.process)
                if self.thread:
                    handles.add(self.thread)
                handle_errors = []
                for handle in handles:
                    try:
                        self.api.close(handle)
                    except BaseException as error:
                        errors.append(error)
                        handle_errors.append(error)
                self.lifecycle["all_process_handles_closed"] = not handle_errors
                self.process = self.thread = None
                try:
                    self.api.close(self.job)
                    self.lifecycle["job_handle_closed"] = True
                except BaseException as error:
                    errors.append(error)
                self.job = None
                self.closed = True
                self.lifecycle.update(elapsed_seconds=time.monotonic() - self.started,
                    close_seconds=time.monotonic() - cleanup_start)
                if errors:
                    base._dispatch_blocked = True
                    self.lifecycle["workers_closed"] = False
                    self.lifecycle["close_errors"] = [f"{type(error).__name__}: {error}" for error in errors]
        if errors:
            raise TreeControlError("Owned tree cleanup/proof failed; further dispatch blocked") from errors[0]


def public_lifecycle(details):
    result = base.public_lifecycle(details)
    result.update(active_at_root_exit=details.get("active_at_root_exit"),
        retained_live_at_root_exit=details.get("retained_live_at_root_exit"),
        natural_wait_seconds=details.get("natural_wait_seconds", 0.0),
        natural_tree_exit=bool(details.get("natural_tree_exit")),
        natural_exit_proof=details.get("natural_exit_proof"),
        natural_limit_reason=details.get("natural_limit_reason"),
        active_before_force=details.get("active_before_force"), natural_wait_limit_seconds=NATURAL_SECONDS)
    return result


def run_owned(command, *, cwd, env, stream, timeout, lifecycle,
              memory_limit_bytes=TREE_MEMORY_LIMIT, local_lifecycle=None, phase_deadline=None):
    """Return the root code after complete closure; forced zero-code closure raises.

    ``timeout`` is the original work interval, including process startup and normal
    drain. Optional ``phase_deadline`` is absolute time.monotonic(), also limiting
    cleanup. The caller reserves ten cleanup seconds inside its total phase budget.
    The one-worker latch/lock are shared with the inherited containment primitives.
    """
    details = {} if local_lifecycle is None else local_lifecycle
    lifecycle.update(public_lifecycle(details))
    tree = None
    locked = False
    started = time.monotonic()
    try:
        if not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("A finite positive timeout is required")
        if phase_deadline is not None and (not isinstance(phase_deadline, (float, int)) or
                not math.isfinite(phase_deadline) or phase_deadline <= started):
            raise ValueError("phase_deadline must be a future finite monotonic timestamp")
        deadline = min(started + timeout, phase_deadline) if phase_deadline is not None else started + timeout
        details.update(work_started_monotonic=started, work_deadline_monotonic=deadline,
                       phase_deadline_monotonic=phase_deadline)
        if base._dispatch_blocked:
            raise TreeControlError("A previous ownership failure blocks further dispatch")
        locked = base._dispatch_lock.acquire(blocking=False)
        if not locked:
            raise TreeControlError("Only one worker may be dispatched at a time")
        tree = OwnedTree(details, memory_limit_bytes=memory_limit_bytes)
        tree.start(command, cwd=cwd, env=env, stream=stream)
        while True:
            active = tree.sample()
            code = tree.poll()
            if time.monotonic() >= deadline:
                if code is not None:
                    details["returncode"] = code
                details["stop_reason"] = "timeout"
                raise subprocess.TimeoutExpired(command, timeout)
            if code is not None:
                details.update(returncode=code, stop_reason="root_exit", active_at_root_exit=active,
                    root_exit_observed_monotonic=time.monotonic(),
                    retained_live_at_root_exit=sum(not tree.api.signaled(handle) for handle, _ in tree.processes.values()))
                tree.close(natural=(code == 0), work_deadline=deadline, phase_deadline=phase_deadline)
                if code == 0 and (not details.get("natural_tree_exit") or details.get("termination_requested")):
                    details["stop_reason"] = "natural_tree_exit_unproved"
                    raise TreeControlError("Zero-code root did not prove natural whole-job completion within its original bounds")
                return code
            time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    except BaseException as error:
        details.setdefault("stop_reason", "controller_exception")
        details["exception_type"] = type(error).__name__
        raise
    finally:
        try:
            if tree is not None:
                tree.close(phase_deadline=phase_deadline)
        except BaseException as error:
            details.update(exception_type=type(error).__name__, stop_reason="cleanup_failure")
            raise
        finally:
            details["elapsed_seconds"] = time.monotonic() - started
            lifecycle.update(public_lifecycle(details))
            if locked:
                base._dispatch_lock.release()
