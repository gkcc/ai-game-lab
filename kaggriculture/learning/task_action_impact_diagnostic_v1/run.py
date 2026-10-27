"""One-shot parent-monitored execution of the preregistered saved-data analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

from .experiment import (ROOT, OUT, OUTPUT_LIMIT, TASK_DEADLINE_UTC, artifact_helper_path, meter, output_bytes,
                         read, utc_now, verify, write_new, write_receipt)

MONITOR_INTERVAL_SECONDS = 0.25


def _scratch_helper():
    # Only the user-required ownership utility; no Flow skill, router or hooks.
    spec = importlib.util.spec_from_file_location("impact_owned_artifacts", artifact_helper_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deadline_remaining():
    return (datetime.fromisoformat(TASK_DEADLINE_UTC) - datetime.now(timezone.utc)).total_seconds()


def _worker_audit(event, args):
    if event in {"subprocess.Popen", "os.system", "os.fork", "os.posix_spawn", "os.spawn"}:
        raise RuntimeError("The analysis worker may not launch any process")
    if event == "import":
        name = args[0]
        forbidden = ("torch", "numpy", "tensorflow", "kaggle_environments", "bots", "models")
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            raise RuntimeError("Model/agent/official imports are prohibited")
        if name.startswith("learning.") and not any(
                name == prefix or name.startswith(prefix + ".") for prefix in (
                    "learning.task_action_impact_diagnostic_v1", "learning.task_bc_error_diagnostics_v1.shadow")):
            # Python may import the namespace parent of the permitted pure helper.
            if name != "learning.task_bc_error_diagnostics_v1":
                raise RuntimeError("Unregistered learning module import is prohibited")


def worker():
    started = read(OUT / "run-started.json")
    if (os.environ.get("TASK_IMPACT_ATTEMPT") != started["attempt_id"]
            or os.getppid() != started["parent_pid"]):
        raise RuntimeError("Worker must be started once by its monitored parent")
    if _deadline_remaining() <= 0:
        raise RuntimeError("The total task deadline has already elapsed")
    write_new(OUT / "worker-started.json", {"attempt_id": started["attempt_id"], "pid": os.getpid(),
              "started_utc": utc_now(), "parent_pid": os.getppid()})
    sys.addaudithook(_worker_audit)
    from . import analysis
    try:
        result = analysis.run(OUT)
        if not (OUT / "analysis-result.json").exists():
            write_new(OUT / "analysis-result.json", result)
        write_new(OUT / "worker-finished.json", {"finished_utc": utc_now(), "metadata_meter": meter(),
                  "analysis_cpu_seconds": time.process_time(), "returned": True})
    except BaseException as exc:
        if not (OUT / "worker-failure.json").exists():
            write_receipt(OUT / "worker-failure.json", {"failed_utc": utc_now(),
                          "error_type": type(exc).__name__, "error": str(exc)[:1000],
                          "metadata_meter": meter(), "analysis_cpu_seconds": time.process_time()})
        raise


def launch(commit):
    if (OUT / "run-started.json").exists():
        raise RuntimeError("The only analysis attempt has already been consumed; retry is forbidden")
    checked = verify(OUT, commit=commit)
    registration, budgets = checked["registration"], checked["registration"]["budgets"]
    if _deadline_remaining() <= 0:
        raise RuntimeError("The total task deadline has already elapsed")
    import psutil
    from learning.task_anchor_once_v1.processes import CapturedProcess
    parent = psutil.Process()
    allowed_cpus = parent.cpu_affinity()
    if not allowed_cpus:
        raise RuntimeError("Cannot establish the single-CPU affinity boundary")
    parent.cpu_affinity([allowed_cpus[0]])
    attempt_id = uuid.uuid4().hex
    started = {"schema": "task-action-impact-run-started-v1", "attempt_id": attempt_id,
               "started_utc": utc_now(), "parent_pid": os.getpid(),
               "local_preregistration_commit": commit,
               "registration_sha256": checked["registration_sha256"],
               "freeze_sha256": checked["freeze_sha256"],
               "parent_preflight_metadata_meter": meter(), "attempt_consumed": True,
               "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
               "memory_enforcement": "Externally sampled working set; not an OS hard quota",
               "cpu_affinity": [allowed_cpus[0]], "psutil_version": psutil.__version__}
    write_new(OUT / "run-started.json", started)
    environment = dict(os.environ)
    environment.update({"TASK_IMPACT_ATTEMPT": attempt_id,
                        "TASK_IMPACT_METADATA_BASE_BYTES": str(meter()["total_bytes"]),
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                        "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    command = [sys.executable, "-B", "-m", "learning.task_action_impact_diagnostic_v1.run", "--worker"]
    process = None
    error = None
    scratch_path = None
    log_summary = None
    worker_metadata = None
    peak_child = peak_total = peak_output = 0
    child_cpu = parent_cpu = 0.0
    parent_cpu_start = time.process_time()
    samples = 0
    beginning = time.monotonic()
    scratch_cleaned = False
    try:
        scratch_module = _scratch_helper()
        # A dedicated owned root avoids sweeping any unrelated temporary run.
        scratch_root = Path(tempfile.gettempdir()) / "task-action-impact-diagnostic-v1"
        with scratch_module.scratch_directory("task-action-impact-diagnostic-v1", root=scratch_root) as scratch:
            scratch_path = str(scratch)
            process = CapturedProcess(command, cwd=ROOT, log=scratch / "worker.log",
                                      limit_bytes=budgets["log_bytes_included_in_output_cap"], env=environment)
            try:
                scratch_module.protect_children(scratch, [{"pid": process.process.pid,
                    "process_identity": scratch_module.process_identity(process.process.pid)[1]}],
                    root=scratch_root, complete=False)
                child = psutil.Process(process.process.pid)
                child.cpu_affinity([allowed_cpus[0]])
                while True:
                    process.check()
                    running = process.process.poll() is None
                    if running:
                        try:
                            with child.oneshot():
                                child_rss = child.memory_info().rss
                                cpu = child.cpu_times()
                                child_cpu = cpu.user + cpu.system
                            descendants = child.children(recursive=True)
                        except psutil.NoSuchProcess:
                            # A worker may exit between poll() and the sample.
                            if process.process.poll() is None:
                                raise
                            continue
                        if descendants:
                            for descendant in descendants:
                                descendant.kill()
                            _, surviving = psutil.wait_procs(descendants, timeout=10)
                            if surviving:
                                raise RuntimeError("Unregistered descendants could not be reaped")
                            raise RuntimeError("Unregistered descendant process detected")
                        parent_rss = parent.memory_info().rss
                        peak_child = max(peak_child, child_rss)
                        peak_total = max(peak_total, parent_rss + child_rss)
                        samples += 1
                        if child_rss > budgets["max_process_working_set_bytes"]:
                            raise RuntimeError("Analysis child working-set cap exceeded")
                        if parent_rss + child_rss > budgets["max_total_memory_bytes"]:
                            raise RuntimeError("Parent plus analysis child memory cap exceeded")
                    current_output = output_bytes(OUT) + process.written
                    peak_output = max(peak_output, current_output)
                    if current_output > budgets["owned_output_bytes"]:
                        raise RuntimeError("Owned output bytes cap exceeded")
                    parent_cpu = time.process_time() - parent_cpu_start
                    if child_cpu + parent_cpu > budgets["analysis_cpu_seconds"]:
                        raise RuntimeError("Analysis CPU time cap exceeded")
                    if time.monotonic() - beginning > budgets["analysis_wall_seconds"]:
                        raise RuntimeError("Analysis wall time cap exceeded")
                    if _deadline_remaining() <= 0:
                        raise RuntimeError("Total 90-minute task deadline exceeded")
                    if not running:
                        break
                    time.sleep(MONITOR_INTERVAL_SECONDS)
                log_summary = process.finish()
                if process.process.returncode != 0:
                    raise RuntimeError("Analysis worker failed; inspect its compact failure receipt")
                if not (OUT / "worker-finished.json").exists():
                    raise RuntimeError("Successful worker exit lacks its completion receipt")
                finished = read(OUT / "worker-finished.json")
                worker_metadata = finished["metadata_meter"]
                child_cpu = max(child_cpu, finished["analysis_cpu_seconds"])
                if child_cpu + time.process_time() - parent_cpu_start > budgets["analysis_cpu_seconds"]:
                    raise RuntimeError("Completed analysis exceeded its CPU budget")
            finally:
                process.stop()
                scratch_module.protect_children(scratch, [], root=scratch_root, complete=True)
            if process.written:
                log = (scratch / "worker.log").read_bytes()
                if output_bytes(OUT) + 2 * len(log) > OUTPUT_LIMIT - 65536:
                    raise RuntimeError("Persisted log would exhaust terminal receipt reserve")
                with (OUT / "worker.log").open("xb") as stream:
                    stream.write(log)
        scratch_cleaned = True
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        # A rejected cleanup is reported; no alternate cleanup route is attempted.
        scratch_cleaned = scratch_path is not None and not Path(scratch_path).exists()
    finally:
        if process is not None:
            process.stop()
        if worker_metadata is None and (OUT / "worker-failure.json").exists():
            worker_metadata = read(OUT / "worker-failure.json")["metadata_meter"]
        elapsed = time.monotonic() - beginning
        parent_preflight = started["parent_preflight_metadata_meter"]
        parent_final = meter()
        metadata_accounting = {
            "inherited_prior_phases_bytes": parent_final["inherited_prior_phase_bytes"],
            "parent_preflight_bytes": parent_preflight["this_process_bytes"],
            "parent_postflight_bytes": parent_final["this_process_bytes"] - parent_preflight["this_process_bytes"],
            "worker_analysis_bytes": None if worker_metadata is None else worker_metadata["this_process_bytes"],
            "worker_inherited_parent_bytes_not_counted_twice": None if worker_metadata is None else worker_metadata["inherited_prior_phase_bytes"],
            "total_known_bytes": parent_final["total_bytes"] + (0 if worker_metadata is None else worker_metadata["this_process_bytes"]),
            "completeness": "worker_meter_missing" if worker_metadata is None else "complete_for_this_launch",
            "limit_bytes": parent_final["limit_bytes"],
            "separate_phases": "Preparation, freeze and independent review receipts must be included once in inherited_prior_phases_bytes or added by final task accounting; a process-local count is not the task total."}
        if metadata_accounting["total_known_bytes"] > metadata_accounting["limit_bytes"]:
            error = error or {"type": "RuntimeError", "message": "Combined metadata/source budget exceeded"}
        result = {"schema": "task-action-impact-run-summary-v1", "finished_utc": utc_now(),
                  "attempt_id": attempt_id, "status": "failed_closed" if error else "completed",
                  "attempt_consumed": True, "retry_allowed": False, "error": error,
                  "analysis_wall_seconds": elapsed, "sampled_analysis_cpu_seconds": child_cpu,
                  "parent_cpu_seconds": time.process_time() - parent_cpu_start,
                  "total_analysis_cpu_seconds": child_cpu + time.process_time() - parent_cpu_start,
                  "peak_sampled_child_working_set_bytes": peak_child,
                  "peak_sampled_parent_plus_child_memory_bytes": peak_total,
                  "peak_observed_output_including_temporary_log_bytes": peak_output,
                  "monitor_samples": samples, "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
                  "memory_enforcement": "Sampled external monitor, not an OS hard quota; between-sample overshoot is unknown",
                  "owned_child_reaped": process is None or process.process.poll() is not None,
                  "scratch_cleaned": scratch_cleaned,
                  "retained_scratch_path": None if scratch_cleaned else scratch_path,
                  "log": log_summary,
                  "parent_preflight_metadata_meter": started["parent_preflight_metadata_meter"],
                  "metadata_phase_accounting": metadata_accounting,
                  "official_transitions": 0, "new_games": 0, "neural_forwards": 0,
                  "training_updates": 0, "gpu_seconds": 0,
                  "task_deadline_utc": registration["task_deadline_utc"]}
        write_receipt(OUT / "run-summary.json", result)
    if error:
        raise RuntimeError("The single attempt failed and is closed: " + error["message"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        if not args.commit:
            parser.error("--commit must identify the exact-byte local preregistration")
        print(json.dumps(launch(args.commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
