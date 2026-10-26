"""One frozen, synthetic-only prerequisite repair; no original experiment retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent / "publication" / "ai-game-lab"
PACKAGE = "learning/task_bc_teacher_label_protocol_repair_v1"
STAGE = "task-bc-teacher-label-protocol-repair-v1-20261026"
OLD = ROOT / "artifacts/task-bc-teacher-label-equivalence-v1-20261026"
BASE = "5a98fa04c9ca58a8997ba53525f561bade6f8bd5"
BUDGET = dict(wall_seconds=60, worker_phase_seconds=40, close_seconds=10,
              memory_bytes=268435456, synthetic_raw_bytes_max=131072,
              returned_payload_bytes_max=1024, json_decodes_max=2,
              scenarios=3, output_bytes_max=2097152, receipt_bytes_max=131072,
              retries=0)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        require(stream.write(data) == len(data), "Incomplete receipt write")


def stat(path):
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=PUBLIC)


def tick(deadline):
    require(time.monotonic() < deadline, "Original execution deadline exceeded")


def state(out):
    cfg, frozen = read(out / "registration.json"), read(out / "freeze.json")
    require(cfg["budget"] == BUDGET, "Budget changed")
    require(sha(out / "registration.json") == frozen["registration_sha256"], "Registration changed")
    return cfg, frozen


def verify(out, cfg, frozen):
    require(sha(out / "planning.json") == cfg["planning_sha256"], "Planning record changed")
    require(sys.version.split()[0] == cfg["python_version"], "Python version changed")
    for name, expected in frozen["runtime_files"].items():
        require(sha(ROOT / name) == expected, "Frozen runtime source changed")
        public_expected = frozen["public_source_overrides"].get(name, expected)
        require(sha(PUBLIC / "kaggriculture" / name) == public_expected, "Frozen public source changed")
    for name, expected in cfg["previous_receipt_sha256"].items():
        require(sha(OLD / name) == expected, "Previous safe receipt changed")
    require(sha(ROOT / "main.py") == cfg["default_sha256"], "Default Bot changed")
    for name, expected in cfg["model_stat"].items():
        require(stat(ROOT / name) == expected, "Protected model stat changed")
    require(sha(out / "protection-before.json") == cfg["history_manifest_sha256"], "History manifest changed")


def prepare(out):
    require(not (out / "registration.json").exists(), "Registration already exists")
    require(git("rev-parse", "HEAD").decode().strip() == BASE, "Unexpected preparation HEAD")
    old_cfg, old_freeze = read(OLD / "registration.json"), read(OLD / "freeze.json")
    protected = read(OLD / "protection-before.json")["stat_before"]
    for name, expected in protected.items():
        require(stat(ROOT / name) == expected, "Old history stat changed")
    protected = dict(protected)
    for path in OLD.rglob("*"):
        if path.is_file():
            protected[path.relative_to(ROOT).as_posix()] = stat(path)
    old_report = ROOT / "TASK_BC_TEACHER_LABEL_EQUIVALENCE_V1.zh-CN.md"
    protected[old_report.relative_to(ROOT).as_posix()] = stat(old_report)
    write(out / "protection-before.json", dict(stat_before=protected,
          method="Inherited history hashes plus unchanged stat; no original data or model content reads."))
    prior = {name: sha(OLD / name) for name in ["registration.json", "freeze.json", "summary.json",
             "primary-receipt.json", "root-failure-adjudication.json", "publication-verification.json"]}
    cfg = dict(schema="teacher-label-synthetic-protocol-repair-registration-v1",
          created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=BASE, budget=BUDGET,
          planning_sha256=sha(out / "planning.json"), python_version=sys.version.split()[0],
          scope="Exactly three in-memory synthetic cases; no original samples, controllers, models or matches.",
          cases=["zero_drain", "positive_drain", "read_failure_boundary"],
          acceptance=["both normal paths hash all compressed bytes once and decode only one prefix JSON",
                      "zero drain is valid and positive drain is exercised using a controlled boundary",
                      "injected read failure and final completeness rejection retain partial caller-owned counts",
                      "all owned handles close, no retries, original deadline and output limits hold",
                      "old frozen bytes and history remain; this is not a teacher-label experiment result"],
          previous_receipt_sha256=prior, history_manifest_sha256=sha(out / "protection-before.json"),
          protected_history_stat_paths=len(protected), model_stat=old_cfg["model_protection"],
          default_sha256=old_cfg["default_sha256"], inherited_official_sha256=old_cfg["official_sha256"],
          prior_task_scoring=dict(groups=3921, candidates=228505), old_execution_passed=False,
          old_strict_result_review_passed=False, original_policy_admission=False,
          local_commit_not_external_timestamp=True, original_data_reads_allowed=0,
          model_reads_allowed=0, controller_calls_allowed=0, training_runs_allowed=0,
          new_games_allowed=0, official_transitions_allowed=0)
    write(out / "registration.json", cfg)
    files = dict(old_freeze["runtime_files"])
    for name in ["__init__.py", "protocol.py", "experiment.py", "README.zh-CN.md"]:
        relative = PACKAGE + "/" + name
        files[relative] = sha(ROOT / relative)
    frozen = dict(registration_sha256=sha(out / "registration.json"), runtime_files=files,
                  public_source_overrides=old_freeze["public_source_overrides"],
                  line_ending_mappings=old_freeze["line_ending_mappings"], previous_source_files=154)
    write(out / "freeze.json", frozen)
    verify(out, cfg, frozen)
    return dict(prepared=True, source_files=len(files), history_paths=len(protected), fixtures_run=0)


def history(out):
    expected = read(out / "protection-before.json")["stat_before"]
    changed = [name for name, value in expected.items() if stat(ROOT / name) != value]
    return dict(passed=not changed, checked_paths=len(expected), changed_paths=changed,
                full_history_content_rehashed=False)


def worker(out):
    cfg, frozen = state(out)
    request = read(out / "request.json")
    require(request["freeze_sha256"] == sha(out / "freeze.json"), "Worker freeze changed")
    deadline = request["work_deadline_monotonic"]
    tick(deadline)
    verify(out, cfg, frozen)
    started = time.monotonic()
    ledger, result, failure = {}, None, None
    try:
        from learning.task_bc_teacher_label_protocol_repair_v1.protocol import protocol_checks
        result = protocol_checks(deadline, ledger)
        require(result["passed"] is True and result["scenario_count"] == 3, "Protocol completion missing")
        verify(out, cfg, frozen)
        tick(deadline)
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        receipt = dict(schema="teacher-label-synthetic-protocol-primary-v1", passed=failure is None,
                       result=result, ledger=ledger, failure_type=failure,
                       elapsed_seconds=time.monotonic() - started, original_data_reads=0,
                       model_reads=0, controller_calls=0, training_runs=0, new_games=0, official_transitions=0)
        write(out / "primary.json", receipt)
        require((out / "primary.json").stat().st_size <= BUDGET["receipt_bytes_max"], "Primary receipt too large")
        tick(deadline)


def check_primary(value):
    require(value["passed"] is True and value["result"]["passed"] is True, "Primary failed")
    cases = value["ledger"]["cases"]
    require([row["name"] for row in cases] == ["zero_drain", "positive_drain", "read_failure_boundary"], "Case order changed")
    require(all(row["passed"] is True for row in cases), "Incomplete case")
    for row in cases:
        c = row["consumption"]
        require(c["raw_opens_completed"] == c["raw_closes_completed"] == 1, "Raw closure mismatch")
        require(c["gzip_opens_completed"] == c["gzip_closes_completed"] == 1, "Gzip closure mismatch")
        require(c["suffix_json_decodes"] == c["raw_reread_passes"] == 0, "Unexpected suffix or reread")
    for row in cases[:2]:
        c = row["consumption"]
        require(c["raw_bytes_returned"] == c["raw_bytes_hashed"] == c["per_source"][0]["registered_bytes"], "Full synthetic hash consumption mismatch")
        require(c["raw_hashes_completed"] == c["json_decodes_completed"] == 1, "Normal prefix/hash count mismatch")
        require(c["per_source"][0]["raw_sha256_verified"] is True, "Synthetic digest not verified")
    require(cases[0]["consumption"]["raw_drain_bytes_returned"] == 0, "Zero-drain branch missing")
    require(cases[1]["consumption"]["raw_drain_bytes_returned"] > 0, "Positive-drain branch missing")
    failed = cases[2]["consumption"]
    require(0 < failed["raw_bytes_returned"] < failed["per_source"][0]["registered_bytes"], "Partial read missing")
    require(failed["raw_read_calls_started"] > failed["raw_read_calls_completed"] and failed["unknown_accounting"]["failures"], "Failure accounting lost")
    require(failed["raw_hashes_completed"] == failed["json_decodes_completed"] == 0, "Failed read counted as complete")
    fault = cases[2]["failure"]
    require(fault["operation_error"] == dict(type="OSError", message="synthetic_second_raw_read")
            and fault["assertion_error"] == dict(type="AssertionError", message="expected_complete_sha_assertion_failure")
            and fault["unexpected_error"] is None and not fault["close_errors"], "Expected assertion boundary not retained")
    require(all(row["observed"]["raw_closed"] and row["observed"]["gzip_closed"]
                and row["observed"]["raw_reads_during_gzip_and_aux_close"] == 0 for row in cases), "Unclosed handle or close read")
    require(cases[0]["observed"]["buffer_closed"] and cases[0]["observed"]["bridge_closed"], "Auxiliary handles not closed")
    require(not any(value["ledger"]["prohibited"].values()), "Unexpected prohibited work")
    totals = {name: sum(row["consumption"][name] for row in cases) for name in
              ["raw_bytes_returned", "raw_bytes_hashed", "returned_payload_bytes", "json_decodes_completed"]}
    require(all(value["ledger"]["totals"][name] == count for name, count in totals.items()), "Aggregate counters changed")
    require(totals["raw_bytes_returned"] <= BUDGET["synthetic_raw_bytes_max"]
            and totals["returned_payload_bytes"] <= BUDGET["returned_payload_bytes_max"]
            and totals["json_decodes_completed"] == BUDGET["json_decodes_max"], "Synthetic budget mismatch")
    return dict(passed=True, cases=3, observed_totals=totals,
                independent_original_reproduction=False, scope="Parent checks the newly written synthetic receipt only.")


def run(out):
    require(not (out / "started.json").exists(), "Existing attempt forbids rerun")
    cfg, frozen = state(out)
    verify(out, cfg, frozen)
    require(not git("status", "--porcelain").strip(), "Publication workspace must be clean")
    head = git("rev-parse", "HEAD").decode().strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head], cwd=PUBLIC, check=True)
    review = read(out / "pre-review.json")
    require(review["passed"] is True and review["freeze_sha256"] == sha(out / "freeze.json"), "Static review missing")
    bindings = {"kaggriculture/" + name: frozen["public_source_overrides"].get(name, expected)
                for name, expected in frozen["runtime_files"].items()}
    prefix = "review/evidence/" + STAGE + "/"
    for name in ["registration.json", "freeze.json", "pre-review.json", "planning.json"]:
        bindings[prefix + name] = sha(out / name)
    queries = "".join(head + ":" + name + "\n" for name in bindings).encode()
    raw = subprocess.check_output(["git", "cat-file", "--batch"], input=queries, cwd=PUBLIC)
    offset = 0
    for name, expected in bindings.items():
        end = raw.index(b"\n", offset)
        header = raw[offset:end].split()
        require(len(header) == 3 and header[1] == b"blob", "Missing preregistration blob")
        size = int(header[2]); data = raw[end + 1:end + 1 + size]
        require(hashlib.sha256(data).hexdigest() == expected, "Preregistration blob changed")
        offset = end + size + 2
    require(offset == len(raw), "Unexpected Git batch output")
    write(out / "pre-execution-commit.json", dict(commit=head, git_files_sha256=bindings,
          freeze_sha256=sha(out / "freeze.json"), local_commit_not_external_timestamp=True))
    started = time.monotonic(); deadline = started + BUDGET["wall_seconds"]
    write(out / "started.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
          deadline_monotonic=deadline, freeze_sha256=sha(out / "freeze.json")))
    work = started + BUDGET["worker_phase_seconds"] - BUDGET["close_seconds"]
    write(out / "request.json", dict(work_deadline_monotonic=work, freeze_sha256=sha(out / "freeze.json")))
    life, local, failure, primary, checks, protection, code = {}, {}, None, None, None, None, None
    try:
        from learning.task_bc_diagnostics_capacity_exit_v1.windows_tree import run_owned
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (out / "worker.log").open("x", encoding="utf-8") as log:
            code = run_owned([sys.executable, "-B", "-m", PACKAGE.replace("/", ".") + ".experiment",
                              "worker", "--out", str(out)], cwd=ROOT, env=env, stream=log,
                             timeout=work - time.monotonic(), phase_deadline=started + BUDGET["worker_phase_seconds"],
                             lifecycle=life, local_lifecycle=local, memory_limit_bytes=BUDGET["memory_bytes"])
        require(code == 0 and life["tree_closed"] and life["natural_tree_exit"]
                and not life["termination_requested"] and life["active_processes_after_close"] == 0
                and life["job_handle_closed"] and life["all_process_handles_closed"]
                and life["close_seconds"] <= BUDGET["close_seconds"], "Worker execution failed")
        primary = read(out / "primary.json")
        checks = check_primary(primary)
        verify(out, cfg, frozen)
        protection = history(out)
        require(protection["passed"], "Historical stat changed")
        tick(deadline)
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        write(out / "local-lifecycle.json", local)
        write(out / "summary.json", dict(schema="teacher-label-synthetic-protocol-summary-v1",
              passed=failure is None, failure_type=failure, worker_returncode=code, lifecycle=life,
              primary_sha256=sha(out / "primary.json") if (out / "primary.json").exists() else None,
              parent_receipt_check=checks, history_check=protection, retries=0,
              elapsed_seconds=time.monotonic() - started, old_execution_passed=False,
              old_strict_result_review_passed=False, original_policy_admission=False,
              cumulative_task_scoring=cfg["prior_task_scoring"], original_data_reads=0,
              model_reads=0, controller_calls=0, training_runs=0, new_games=0, official_transitions=0))
        total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        write(out / "output-accounting.json", dict(bytes_before_this_receipt=total, limit_bytes=BUDGET["output_bytes_max"],
              scope="All current stage files including preregistration, private history manifest, logs and failures."))
    tick(deadline)
    total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    require(total + 4096 <= BUDGET["output_bytes_max"], "Output cap exceeded")
    write(out / "execution-acceptance.json", dict(passed=True, summary_sha256=sha(out / "summary.json"),
          freeze_sha256=sha(out / "freeze.json"), elapsed_seconds=time.monotonic() - started,
          artifact_bytes_before_acceptance=total, scope="Synthetic prerequisite repair only",
          requires_parent_zero_exit=True))
    tick(deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "worker"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out == ROOT / "artifacts" / STAGE, "Wrong experiment directory")
    result = globals()[args.command](out)
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
