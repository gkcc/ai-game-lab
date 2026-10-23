"""Seven single-attempt real Windows regressions; call only after registration.

run_regressions(out) writes detailed LOCAL receipts and returns a public-safe
whitelist summary. Existing started markers reject accidental reruns. No pytest, mock,
training, source observation, model scoring or game dependency is involved.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time

from .windows_tree import FORCED_EXIT_CODE, OwnedTree, TreeControlError, run_owned, NATURAL_SECONDS, NATURAL_PROOF


KINDS = ("normal_exit", "cooperative_deadline", "noncooperative_descendants", "controller_hard_interrupt",
    "root_zero_natural_descendants", "root_zero_stubborn_descendants", "root_zero_original_deadline")
MATRIX_SECONDS = 120.0
CASE_BODY_SECONDS = 5.0
CASE_CLOSE_SECONDS = 10.0
COOPERATIVE_EXIT = 23
SAFETY_EXIT = 99


def _write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _read_ready(path):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _require(condition, message):
    if not condition:
        raise TreeControlError(message)


def _identity(receipt):
    return {key: receipt[key] for key in ("pid", "creation_filetime")}


def _nodes(out):
    return [_read_ready(out / f"node-{depth}.json") for depth in range(3)]


def _observed(tree, identity):
    key = (identity["pid"], identity["creation_filetime"])
    return tree.processes.get(key)


def _wait_ready(tree, predicate, deadline):
    while time.monotonic() < deadline:
        tree.sample()
        value = predicate()
        if value:
            return value
        _require(tree.poll() is None, "Fixture exited before the readiness condition")
        time.sleep(0.02)
    raise TreeControlError("Fixture readiness exceeded its fixed body budget")


def _assert_sentinel(sentinel, receipt, tested_tree=None):
    sentinel.sample()
    entry = _observed(sentinel, _identity(receipt))
    _require(entry is not None and not sentinel.api.signaled(entry[0]), "Owned external sentinel is not alive")
    _require(sentinel.poll() is None, "Sentinel launcher exited unexpectedly")
    if tested_tree is not None:
        _require(not tested_tree.api.in_job(entry[0], tested_tree.job), "Sentinel is inside the tested job")


def _assert_nodes(local, out, *, expected_code=None):
    nodes = _nodes(out)
    _require(all(nodes), "Not all worker/child/grandchild identities were recorded")
    local["fixture_nodes"] = nodes
    observed = {(item["pid"], item["creation_filetime"]): item for item in local["lifecycle"]["processes"]}
    for node in nodes:
        item = observed.get((node["pid"], node["creation_filetime"]))
        _require(item is not None and item["exit_observed"], "A fixture identity has no retained-handle exit proof")
        if expected_code is not None:
            _require(item["exit_code"] == expected_code, "Fixture interpreter exit code differs")
    launcher = local.get("inner_job_before_interrupt", local["lifecycle"])["launcher_identity"]
    _require(launcher["pid"] != nodes[0]["pid"], "Real venv launcher/interpreter distinction was not observed")
    _require(not list(out.glob("*-safety-fired.json")), "Fixture safety deadline fired instead of ownership cleanup")
    local["venv_launcher_and_actual_interpreter_observed"] = True
    local["three_generation_identities_closed"] = True


def _regular_case(kind, out, python, fixture, environment, matrix_deadline):
    started = time.monotonic()
    lifecycle = {}
    public_life = {}
    local = {"kind": kind, "attempt": 1, "passed": False, "lifecycle": lifecycle}
    mode = {"normal_exit": "normal", "cooperative_deadline": "cooperative",
            "noncooperative_descendants": "ignore"}[kind]
    command = [str(python), "-B", str(fixture), "tree", "--out", str(out), "--mode", mode,
        "--deadline", repr(started + 1.2), "--safety-until", repr(started + CASE_BODY_SECONDS)]
    timeout = 2.5 if mode == "ignore" else 4.5
    try:
        with (out / "worker.log").open("x", encoding="utf-8") as stream:
            try:
                code = run_owned(command, cwd=fixture.parents[2], env=environment, stream=stream,
                                 timeout=timeout, lifecycle=public_life, local_lifecycle=lifecycle,
                                 phase_deadline=min(started + CASE_BODY_SECONDS + CASE_CLOSE_SECONDS, matrix_deadline))
            except subprocess.TimeoutExpired:
                _require(mode == "ignore", "A cooperative fixture timed out")
                local["expected_timeout_observed"] = True
            else:
                _require(mode != "ignore", "Noncooperative descendants did not reach the controller timeout")
                _require(code == (0 if mode == "normal" else COOPERATIVE_EXIT), "Unexpected fixture root return code")
        _require(lifecycle["workers_closed"] and lifecycle["active_processes_final"] == 0,
                 "Whole-tree closure is unproved")
        _require(lifecycle["close_seconds"] <= CASE_CLOSE_SECONDS, "Close exceeded its ten-second budget")
        _assert_nodes(local, out, expected_code=FORCED_EXIT_CODE if mode == "ignore" else
                      (0 if mode == "normal" else COOPERATIVE_EXIT))
        if mode == "ignore":
            _require(lifecycle["termination_requested"], "Noncooperative job was not forcibly terminated")
            _require(not list(out.glob("*-done.json")), "Noncooperative fixture completed itself")
        else:
            _require(all(_read_ready(out / f"node-{depth}-done.json") for depth in range(3)),
                     "Normal/cooperative descendants did not complete")
            _require(not lifecycle["termination_requested"], "Normal/cooperative closure needed forced cleanup")
        local["passed"] = True
    except BaseException as error:
        local.update(error_type=type(error).__name__, error=str(error))
    finally:
        local["public_lifecycle"] = public_life
        local["wall_seconds"] = time.monotonic() - started
    return local


def _interrupt_case(out, python, fixture, environment, sentinel, sentinel_receipt, matrix_deadline):
    started = time.monotonic()
    lifecycle = {}
    local = {"kind": "controller_hard_interrupt", "attempt": 1, "passed": False, "lifecycle": lifecycle}
    command = [str(python), "-B", str(fixture), "controller", "--out", str(out),
        "--deadline", repr(started + 0.6), "--safety-until", repr(started + CASE_BODY_SECONDS)]
    guard = OwnedTree(lifecycle)
    try:
        with (out / "controller.log").open("x", encoding="utf-8") as stream:
            guard.start(command, cwd=fixture.parents[2], env=environment, stream=stream)
            _assert_sentinel(sentinel, sentinel_receipt, guard)
            def ready():
                controller = _read_ready(out / "controller.json")
                established = _read_ready(out / "inner-established.json")
                nodes = _nodes(out / "workers")
                if not controller or not established or not all(nodes):
                    return None
                identities = [controller, *nodes]
                entries = [_observed(guard, _identity(item)) for item in identities]
                if not all(entry is not None and not guard.api.signaled(entry[0]) for entry in entries):
                    return None
                return controller, established, nodes
            controller, established, nodes = _wait_ready(guard, ready, started + 3.0)
            local.update(controller_identity=controller, inner_job_before_interrupt=established,
                         fixture_nodes=nodes)
            _require(established["membership_verified_before_resume"] and
                established["kill_on_job_close"] and not established["job_handle_inherited"] and
                not established["breakaway_allowed"], "Inner job did not establish required containment")
            # Retained handles identify the interpreter exactly, not merely its launcher.
            guard.terminate_identity(_identity(controller))
            interrupted = time.monotonic()
            local["controller_terminated_by_identity"] = True
            while guard.sample() != 0:
                _require(time.monotonic() < min(started + CASE_BODY_SECONDS, interrupted + CASE_CLOSE_SECONDS),
                         "Kill-on-close did not empty the outer observed tree before the fixture safety bound")
                time.sleep(0.02)
            _require(guard.poll() is not None, "Outer launcher did not exit with the interrupted interpreter")
            _require(not (out / "controller-finally.json").exists(), "Controller ran finally; hard-interruption path was not tested")
            for node in nodes:
                handle, _ = _observed(guard, _identity(node))
                _require(guard.api.signaled(handle) and guard.api.exit_code(handle) != SAFETY_EXIT,
                         "Worker did not exit through job closure")
            local.update(kill_on_close_observed_without_controller_finally=True,
                outer_job_active_processes_zero_before_guard_cleanup=True,
                interruption_to_empty_seconds=time.monotonic() - interrupted)
    except BaseException as error:
        local.update(error_type=type(error).__name__, error=str(error))
    finally:
        try:
            guard.close(phase_deadline=min(started + CASE_BODY_SECONDS + CASE_CLOSE_SECONDS, matrix_deadline))
        except BaseException as error:
            local.update(error_type=type(error).__name__, error=str(error))
        local["wall_seconds"] = time.monotonic() - started
    if "error" not in local:
        try:
            _require(lifecycle["workers_closed"] and lifecycle["active_processes_final"] == 0,
                     "Interrupted tree closure is unproved")
            _require(not lifecycle["termination_requested"], "Outer guard had to kill survivors; inner kill-on-close failed")
            _assert_nodes(local, out / "workers")
            local["passed"] = True
        except BaseException as error:
            local.update(error_type=type(error).__name__, error=str(error))
    return local


def _root_first_case(kind, out, python, fixture, environment, matrix_deadline):
    started = time.monotonic()
    lifecycle = {}
    public_life = {}
    local = dict(kind=kind, attempt=1, passed=False, lifecycle=lifecycle)
    mode = {"root_zero_natural_descendants": "natural", "root_zero_stubborn_descendants": "beyond-grace",
            "root_zero_original_deadline": "work-deadline"}[kind]
    command = [str(python), "-B", str(fixture), "root-first", "--out", str(out), "--mode", mode,
        "--safety-until", repr(started + CASE_BODY_SECONDS)]
    stop = threading.Event()
    observer_errors = []

    def release_after_retention():
        try:
            while not stop.is_set() and time.monotonic() < started + CASE_BODY_SECONDS:
                nodes = _nodes(out)
                launcher = lifecycle.get("launcher_identity")
                known = {(item["pid"], item["creation_filetime"]) for item in tuple(lifecycle.get("processes", []))}
                if launcher and all(nodes) and all((node["pid"], node["creation_filetime"]) in known for node in nodes):
                    local["retained_identities_before_release"] = True
                    _write_new(out / "owner-observed.json", dict(root_launcher_identity=launcher,
                        work_deadline_monotonic=lifecycle["work_deadline_monotonic"],
                        retained_nodes=[_identity(node) for node in nodes], observed_monotonic=time.monotonic()))
                    return
                stop.wait(0.01)
            if not stop.is_set():
                raise TreeControlError("The owner did not retain all fixture identities before release")
        except BaseException as error:
            observer_errors.append(error)

    observer = threading.Thread(target=release_after_retention, name="owned-tree-fixture-handshake", daemon=False)
    observer.start()
    caught = None
    try:
        with (out / "worker.log").open("x", encoding="utf-8") as stream:
            try:
                code = run_owned(command, cwd=fixture.parents[2], env=environment, stream=stream,
                    timeout=3.5 if mode == "work-deadline" else 4.5, lifecycle=public_life,
                    local_lifecycle=lifecycle,
                    phase_deadline=min(started + CASE_BODY_SECONDS + CASE_CLOSE_SECONDS, matrix_deadline))
            except TreeControlError as error:
                caught = error
                _require(mode != "natural", "Natural descendants were rejected")
            else:
                _require(mode == "natural" and code == 0, "Late descendants were accepted as success")
        _require(local.get("retained_identities_before_release") and not observer_errors,
                 "Identity-retention handshake did not complete")
        _require(lifecycle.get("direct_returncode") == 0, "The real root did not exit zero")
        _require(lifecycle.get("active_at_root_exit", 0) > 0 and lifecycle.get("retained_live_at_root_exit", 0) > 0,
                 "A root-before-live-descendant boundary was not observed")
        _require(lifecycle["workers_closed"] and lifecycle["active_processes_final"] == 0,
                 "Whole-tree closure is unproved")
        _require(lifecycle["all_process_handles_closed"] and lifecycle["job_handle_closed"], "Owned handles were not closed")
        _require(lifecycle["close_seconds"] <= CASE_CLOSE_SECONDS, "Drain plus cleanup exceeded one ten-second allowance")
        _assert_nodes(local, out)
        for depth in (1, 2):
            handle_receipt = _read_ready(out / f"node-{depth}-root-handle.json")
            exit_receipt = _read_ready(out / f"node-{depth}-root-exit.json")
            _require(handle_receipt and handle_receipt["root_was_alive"] and exit_receipt,
                     "A descendant did not retain then observe the original launcher identity exit")
            _require(exit_receipt["planned_finish_monotonic"] > lifecycle["root_exit_observed_monotonic"],
                     "Descendant completion was not planned after the observed root exit")
        local["root_before_descendants_observed"] = True
        by_identity = {(row["pid"], row["creation_filetime"]): row for row in lifecycle["processes"]}
        for depth, node in enumerate(_nodes(out)):
            expected = 0 if depth == 0 or mode == "natural" else FORCED_EXIT_CODE
            _require(by_identity[(node["pid"], node["creation_filetime"])]["exit_code"] == expected,
                     "Fixture identity did not take the registered natural/forced exit path")
        if mode == "natural":
            _require(lifecycle.get("natural_tree_exit") and lifecycle.get("natural_exit_proof") == NATURAL_PROOF
                and not lifecycle["termination_requested"], "Natural completion depended on cleanup termination")
            _require(all(_read_ready(out / f"node-{depth}-done.json") for depth in range(3)),
                     "A natural descendant did not reach its done receipt")
        else:
            _require(caught is not None and lifecycle["termination_requested"] and lifecycle.get("active_before_force", 0) > 0
                and not lifecycle.get("natural_tree_exit"), "The required rejected/forced result was not observed")
            _require(not any((out / f"node-{depth}-done.json").exists() for depth in (1, 2)),
                     "A deliberately late descendant exited itself")
            if mode == "work-deadline":
                _require(lifecycle["natural_limit_reason"] == "original_work_deadline" and
                    lifecycle["natural_deadline_monotonic"] == lifecycle["work_deadline_monotonic"],
                    "Natural drain renewed the original work deadline")
                _require(lifecycle["natural_wait_seconds"] < NATURAL_SECONDS and
                    lifecycle["root_exit_observed_monotonic"] < lifecycle["work_deadline_monotonic"],
                    "The original deadline did not win after an observed zero root")
                local["original_deadline_won"] = True
            else:
                _require(lifecycle["natural_limit_reason"] == "fixed_grace", "Beyond-grace fixture hit a different limit")
        local["passed"] = True
    except BaseException as error:
        local.update(error_type=type(error).__name__, error=str(error))
    finally:
        stop.set()
        observer.join(timeout=max(0.0, min(matrix_deadline, started + CASE_BODY_SECONDS + CASE_CLOSE_SECONDS) - time.monotonic()))
        if observer.is_alive() or observer_errors:
            local.update(passed=False, error_type="TreeControlError", error="Fixture handshake observer failed or did not close")
        local.update(public_lifecycle=public_life, wall_seconds=time.monotonic() - started)
    return local


def public_summary(local):
    """Whitelist only; do not publish native identities, paths, argv or raw errors."""
    cases = []
    for case in local["cases"]:
        life = case["lifecycle"]
        cases.append(dict(kind=case["kind"], attempt=case["attempt"], passed=case["passed"],
            workers_closed=bool(life.get("workers_closed")),
            active_processes_final=life.get("active_processes_final"), wall_seconds=case["wall_seconds"],
            close_seconds=life.get("close_seconds"), peak_job_commit_bytes=life.get("peak_job_commit_bytes"),
            sampled_peak_tree_working_set_bytes=life.get("sampled_peak_tree_working_set_bytes"),
            sampled_peak_tree_private_commit_bytes=life.get("sampled_peak_tree_private_commit_bytes"),
            venv_launcher_and_actual_interpreter_observed=case.get("venv_launcher_and_actual_interpreter_observed", False),
            three_generation_identities_closed=case.get("three_generation_identities_closed", False),
            sentinel_survived=case.get("sentinel_survived", False),
            kill_on_close_without_controller_finally=case.get("kill_on_close_observed_without_controller_finally", False),
            error_type=case.get("error_type"),
            active_at_root_exit=life.get("active_at_root_exit"),
            retained_live_at_root_exit=life.get("retained_live_at_root_exit"),
            natural_wait_seconds=life.get("natural_wait_seconds", 0.0),
            natural_tree_exit=bool(life.get("natural_tree_exit")),
            natural_exit_proof=life.get("natural_exit_proof"),
            natural_limit_reason=life.get("natural_limit_reason"),
            active_before_force=life.get("active_before_force"),
            termination_requested=bool(life.get("termination_requested")),
            direct_returncode=life.get("direct_returncode"),
            retained_identities_before_release=bool(case.get("retained_identities_before_release")),
            root_before_descendants_observed=bool(case.get("root_before_descendants_observed")),
            original_deadline_won=bool(case.get("original_deadline_won"))))
    return dict(schema="task-capacity-exit-windows-tree-regressions-public-v1", passed=local["passed"],
        error_type=local.get("error_type"),
        cases=cases, registered_case_count=7, executed_case_count=len(cases), attempts_per_case_max=1,
        fixture_body_seconds_max=CASE_BODY_SECONDS, tree_close_seconds_max=CASE_CLOSE_SECONDS,
        matrix_wall_seconds_max=MATRIX_SECONDS, wall_seconds=local.get("wall_seconds"),
        natural_drain_seconds_max=NATURAL_SECONDS,
        sentinel_closed=local["sentinel_lifecycle"].get("workers_closed", False),
        working_set_is_sampled_lower_bound=True, peak_commit_source="Windows PeakJobMemoryUsed",
        model_calls=0, source_observations_read=0, training_runs=0, new_games=0, official_transitions=0)


def run_regressions(out, *, python_executable=None):
    """Run the registered matrix exactly once; return a public-safe summary.

    ``out`` is a new local evidence directory, never a public export directory.
    Failure stops the remaining matrix. Callers must require receipt['passed'].
    Detailed files written here are local only; callers publish the return value.
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = Path(__file__).resolve().parents[2]
    python = Path(python_executable or runtime / ".venv" / "Scripts" / "python.exe").resolve()
    fixture = Path(__file__).with_name("tree_fixture.py").resolve()
    _require(os.name == "nt" and python.is_file(), "Real Windows venv interpreter is required")
    _write_new(out / "tree-regressions-started.json", dict(schema="task-capacity-exit-windows-tree-start-v1",
        cases=list(KINDS), attempts_per_case_max=1, fixture_body_seconds_max=CASE_BODY_SECONDS,
        close_seconds_max=CASE_CLOSE_SECONDS, python_executable=str(python)))
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    local = dict(schema="task-capacity-exit-windows-tree-regressions-local-v1", passed=False,
                 cases=[], sentinel_lifecycle={}, all_details_local_only=True)
    sentinel_dir = out / "external-owned-sentinel"
    sentinel_dir.mkdir()
    sentinel = OwnedTree(local["sentinel_lifecycle"])
    sentinel_started = time.monotonic()
    matrix_deadline = sentinel_started + MATRIX_SECONDS
    try:
        with (sentinel_dir / "sentinel.log").open("x", encoding="utf-8") as stream:
            sentinel.start([str(python), "-B", str(fixture), "sentinel", "--out", str(sentinel_dir),
                "--safety-until", repr(matrix_deadline)], cwd=runtime, env=environment, stream=stream)
            sentinel_receipt = _wait_ready(sentinel, lambda: _read_ready(sentinel_dir / "sentinel.json"),
                                           sentinel_started + 5.0)
            local["sentinel_identity"] = sentinel_receipt
            for kind in KINDS:
                _require(time.monotonic() + CASE_BODY_SECONDS + CASE_CLOSE_SECONDS < matrix_deadline,
                    "Matrix cannot start another case inside its original deadline")
                _assert_sentinel(sentinel, sentinel_receipt)
                case_out = out / kind
                case_out.mkdir()
                if kind == "controller_hard_interrupt":
                    case = _interrupt_case(case_out, python, fixture, environment, sentinel, sentinel_receipt, matrix_deadline)
                elif kind.startswith("root_zero_"):
                    case = _root_first_case(kind, case_out, python, fixture, environment, matrix_deadline)
                else:
                    case = _regular_case(kind, case_out, python, fixture, environment, matrix_deadline)
                local["cases"].append(case)
                try:
                    _assert_sentinel(sentinel, sentinel_receipt)
                    case["sentinel_survived"] = True
                except BaseException as error:
                    case.update(passed=False, sentinel_survived=False, error_type=type(error).__name__, error=str(error))
                _write_new(case_out / "receipt-local.json", case)
                if not case["passed"]:
                    break
    except BaseException as error:
        local.update(error_type=type(error).__name__, error=str(error))
    finally:
        try:
            sentinel.close(phase_deadline=matrix_deadline)
        except BaseException as error:
            local.update(error_type=type(error).__name__, error=str(error))
        local["wall_seconds"] = time.monotonic() - sentinel_started
        local["passed"] = ("error" not in local and len(local["cases"]) == 7 and time.monotonic() <= matrix_deadline and
            all(case["passed"] for case in local["cases"]) and local["sentinel_lifecycle"]["workers_closed"])
        _write_new(out / "tree-regressions-local.json", local)
    return public_summary(local)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_regressions(args.out)
    print(json.dumps(receipt, sort_keys=True))
    raise SystemExit(0 if receipt["passed"] else 1)
