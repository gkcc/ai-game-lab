"""Register and run the single historical operation-request diagnostic once."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from learning.task_bc_first_water_unit_requests_v1 import experiment as prior

ROOT, PUBLIC = prior.ROOT, prior.PUBLIC
PREVIOUS = ROOT / "artifacts/task-bc-first-water-unit-requests-v1-20261026"
PACKAGE = Path(__file__).resolve().parent
STAGE = "task-bc-first-operation-unit-request-v1"
PLANS = ("follow-on-operation-request-advice.public.json", "root-follow-on-operation-request-decision.json")
require, sha, stat, git, read = prior.require, prior.sha, prior.stat, prior.git, prior.read


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb", buffering=0) as stream:
        require(stream.write(raw) == len(raw), "Incomplete metadata write")


def account_outputs(out, limit):
    """Retain aggregate size even when the original worker phase failed."""
    files, known_bytes, unknown = 0, 0, []
    try:
        for path in out.rglob("*"):
            if path.is_file():
                known_bytes += path.stat().st_size
                files += 1
    except BaseException as error:
        unknown.append(dict(quantity="remaining_artifact_file_count_and_bytes", kind=type(error).__name__))
    value = dict(known=not unknown, measured_files_before_this_receipt=files,
                 known_bytes_before_this_receipt=known_bytes, receipt_bytes=0,
                 known_bytes_including_this_receipt=known_bytes, limit_bytes=limit,
                 within_limit=False, unknown_accounting=unknown,
                 scope="All current OUT files, including logs and partial failures, plus this receipt. Later successful execution-acceptance is measured again by final_gate.")
    while True:
        raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        total = known_bytes + len(raw)
        within = not unknown and total <= limit
        if value["receipt_bytes"] == len(raw) and value["known_bytes_including_this_receipt"] == total and value["within_limit"] == within:
            break
        value.update(receipt_bytes=len(raw), known_bytes_including_this_receipt=total, within_limit=within)
    write(out / "output-accounting.json", value)
    return value


def normalized_budget(scope):
    """Aliases for unchanged meters; no allowance exceeds the adopted scope."""
    source, controller, models = scope["raw_source"], scope["controller"], scope["models"]
    normalized_source = dict(
        gzip_opens_max=source["gzip_opens_max"], prefix_lines_returned_max=source["prefix_lines_returned_max"],
        prefix_lines_per_source_max=source["prefix_lines_returned_max"],
        target_json_turns_decoded_max=source["target_json_turns_decoded_max"], skipped_prefix_lines_json_decoded_max=0,
        compressed_bytes_read_per_source_max=source["compressed_bytes_read_max"],
        compressed_bytes_read_total_max=source["compressed_bytes_read_max"],
        returned_payload_bytes_per_source_max=source["returned_payload_bytes_max"],
        returned_payload_bytes_total_max=source["returned_payload_bytes_max"],
        single_line_bytes_max=source["single_line_bytes_max"], target_json_bytes_total_max=source["target_json_bytes_max"],
        target_task_records_max=source["target_task_records_max"],
        target_candidate_objects_and_x_arrays_max=source["target_candidate_objects_and_x_arrays_max"],
        target_shared_x_observation_and_configuration_objects_max_each=source["target_shared_x_observation_configuration_objects_max_each"],
        target_crop_records_and_numeric_payloads_max=source["target_crop_records_and_numeric_payloads_max"],
        reread_passes_max=0, complete_trace_hash_passes_max=0, raw_replay_reads_max=0,
        existing_compact_or_first_witness_reads_max=0)
    normalized_models = dict(models, total_model_bytes_returned_by_reads_max=models["model_returned_read_bytes_max"], training_calls_max=0)
    normalized_controller = dict(controller, callbacks_per_controller_call_max=controller["callbacks_per_call_max"],
                                 forced_choices_exact_on_completed_stage=controller["forced_choices_exact_when_complete"])
    checks = scope["one_off_in_memory_protocol_checks"]
    return dict(source=normalized_source, models=normalized_models, controller=normalized_controller,
                model_load_phase_seconds_max=scope["model_load_phase_seconds_max"],
                per_close_seconds_max=scope["per_close_seconds_max"], per_sample_pair_seconds_max=20,
                one_off_protocol_checks=dict(in_memory_cases_max=checks["cases_max"],
                    fake_callback_events_max=checks["fake_callback_events_max"], fake_candidates_max=checks["fake_candidates_max"]))


def prepare(out):
    require(not out.exists() and not git("status", "--porcelain").strip(), "New output and clean accepted checkout required")
    decision, advice = (read(PREVIOUS / name) for name in (PLANS[1], PLANS[0]))
    require(decision["status"] == advice["status"] == "planned_not_started", "Plan already consumed")
    require(sha(PREVIOUS / PLANS[0]) == decision["accepted_public_advice_sha256"], "Advice binding changed")
    require(read(PREVIOUS / "publication-verification.json")["passed"] and
            read(PREVIOUS / "root-evidence-acceptance.json")["passed"], "Prior stage not accepted")
    head = git("rev-parse", "HEAD").decode().strip()
    require(head == decision["current_published_commit"], "Published base differs")
    base = read(PREVIOUS / "freeze.json")
    for rel, digest in base["runtime_files"].items():
        require(sha(ROOT / rel) == digest, "Frozen source changed")
    old = read(PREVIOUS / "registration.json")
    scope = decision["fixed_input_scope"]
    source_stat = dict(size=scope["inherited_trace_size_bytes"], mtime_ns=scope["inherited_trace_mtime_ns"])
    require(stat(ROOT / scope["trace"]) == source_stat, "Source stat changed")
    models = {name: scope[name + "_model"] for name in ("task", "crop")}
    for spec in models.values():
        require(stat(ROOT / spec["path"]) == old["model_protection"][spec["path"]], "Model stat changed")
    history = read(PREVIOUS / "protection-before.json")["stat_before"].copy()
    for path in PREVIOUS.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            history[path.relative_to(ROOT).as_posix()] = stat(path)
    report = ROOT / "TASK_BC_FIRST_WATER_UNIT_REQUESTS_V1.zh-CN.md"
    history[report.relative_to(ROOT).as_posix()] = stat(report)
    require(all((ROOT / rel).is_file() and stat(ROOT / rel) == value for rel, value in history.items()), "Historical stat changed")
    out.mkdir(parents=True)
    for name in PLANS:
        with (out / name).open("xb") as stream:
            stream.write((PREVIOUS / name).read_bytes())
    write(out / "protection-before.json", dict(stat_before=history,
        method="Inherited hashes and before/after stat; frozen code/metadata separately hashed. No private history contents read."))
    metadata = {"artifacts/" + PREVIOUS.name + "/" + name: sha(PREVIOUS / name)
                for name in ("freeze.json", "registration.json", "summary.json", "evidence-review.json",
                             "root-evidence-acceptance.json", "publication-verification.json")}
    selection = {key: decision["fixed_selection"][key] for key in (
        "source_compact", "compact_sha256", "compact_bytes_exact", "compact_total_rows_inherited",
        "early_callback_rows_inherited", "projection_receipt", "projection_receipt_sha256")}
    cfg = dict(schema=STAGE, created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=head,
        fixed_selection=selection, fixed_input_scope={key: scope[key] for key in (
            "trace", "inherited_trace_sha256", "inherited_trace_size_bytes", "inherited_trace_mtime_ns", "task_model", "crop_model")},
        source_stat=source_stat, compact_stat=stat(ROOT / selection["source_compact"]),
        projection_stat=stat(ROOT / selection["projection_receipt"]), models=models,
        model_protection={spec["path"]: stat(ROOT / spec["path"]) for spec in models.values()},
        scope_budget=decision["budget"], budget=normalized_budget(decision["budget"]),
        source_metadata_sha256=metadata, planning_sha256={name: sha(out / name) for name in PLANS},
        protection_sha256=sha(out / "protection-before.json"), default_sha256=sha(ROOT / "main.py"),
        official_sha256=prior.official(), engine_commit=old["engine_commit"], dependencies=prior.dependencies(),
        line_ending_mappings=base["line_ending_mappings"],
        historical_consumed_seed_not_independent_holdout=True, local_commit_not_external_timestamp=True,
        control_metadata_limit_scope="Aggregate registration+freeze raw input bytes returned to three workers, one parent final binding read, and the primary selection receipt read. Preflight static binding reads precede execution; checker private receipt reads use its separate 512 KiB cap. Original projection receipt is read only by selector once.",
        output_limit_scope="All files in the new OUT, including registration copies, private logs and partial results, are counted at final executable acceptance. Later human review and publication receipts are outside that executable phase and separately identified.",
        previous_diagnostic_stage_acceptance=False, original_policy_admission=False, previous_strict_review_acceptance=False,
        retries_allowed=0)
    require(cfg["official_sha256"] == old["official_sha256"], "Official source changed")
    require(cfg["compact_stat"]["size"] == selection["compact_bytes_exact"], "Compact size changed")
    write(out / "registration.json", cfg)
    tracked = [name.decode("utf-8") for name in git("ls-files", "-z").split(b"\0") if name]
    write(out / "public-before.json", dict(head=head, tracked_files=len(tracked), clean=True))
    return dict(prepared=True, source_data_reads=0, model_reads=0, tests_executed=0)


def published_hashes(frozen):
    return dict(frozen["runtime_files"], **frozen["public_source_overrides"])


def freeze(out):
    cfg = read(out / "registration.json")
    sources = read(PREVIOUS / "freeze.json")["runtime_files"].copy()
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in (".py", ".md"):
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    overrides = {}
    for rel, digest in sources.items():
        require(sha(ROOT / rel) == digest, "Runtime source changed")
        published = PUBLIC / "kaggriculture" / rel
        other = sha(published)
        if other != digest:
            require(rel in cfg["line_ending_mappings"] and (ROOT / rel).read_bytes().replace(b"\r\n", b"\n") ==
                    published.read_bytes().replace(b"\r\n", b"\n"), "Undeclared public source mapping")
            overrides[rel] = other
    write(out / "freeze.json", dict(registration_sha256=sha(out / "registration.json"), runtime_files=sources,
        public_source_overrides=overrides, line_ending_mappings=cfg["line_ending_mappings"],
        public_hash_rule="Use runtime SHA unless the explicit public_source_overrides supplies the inherited CRLF/LF mapping."))
    size = (out / "registration.json").stat().st_size + (out / "freeze.json").stat().st_size
    require(size * 4 + cfg["scope_budget"]["outputs"]["selection_receipt_bytes_max"] <=
            cfg["scope_budget"]["source_and_metadata"]["fixed_registration_and_selection_metadata_bytes_max"],
            "Fixed control input allocation exceeds metadata budget before execution")
    return dict(frozen=True, freeze_sha256=sha(out / "freeze.json"), sources=len(sources), state_payload_bytes=size)


def load_state(out):
    values, digests, count = [], [], 0
    for name in ("registration.json", "freeze.json"):
        path = out / name
        require(0 < path.stat().st_size <= 262144, "Control metadata size")
        raw = path.read_bytes()
        count += len(raw)
        values.append(json.loads(raw))
        digests.append(hashlib.sha256(raw).hexdigest())
    require(values[1]["registration_sha256"] == digests[0], "Registration byte binding changed")
    return *values, dict(bytes=count, registration_sha256=digests[0], freeze_sha256=digests[1])


def verify_runtime(out, cfg, frozen):
    for rel, digest in frozen["runtime_files"].items():
        require(sha(ROOT / rel) == digest, "Frozen runtime source changed")
    for rel, digest in cfg["source_metadata_sha256"].items():
        require(sha(ROOT / rel) == digest, "Safe metadata changed")
    for name, digest in cfg["planning_sha256"].items():
        require(sha(out / name) == digest, "Planning changed")
    require(sha(out / "protection-before.json") == cfg["protection_sha256"], "Protection metadata changed")
    for rel, expected in cfg["model_protection"].items():
        require(stat(ROOT / rel) == expected, "Model stat changed")
    for rel, expected in ((cfg["fixed_input_scope"]["trace"], cfg["source_stat"]),
                          (cfg["fixed_selection"]["source_compact"], cfg["compact_stat"]),
                          (cfg["fixed_selection"]["projection_receipt"], cfg["projection_stat"])):
        require(stat(ROOT / rel) == expected, "Bound data stat changed")
    require(prior.official() == cfg["official_sha256"] and prior.dependencies() == cfg["dependencies"], "Official/dependency identity changed")


def terminal_passed(key, value):
    """Validate usable terminal counters; retain malformed partial work outside."""
    require(type(value) is dict and value.get("status") in ("complete", "failed"), "Terminal status missing")

    def integers(group, names):
        require(type(group) is dict and all(type(group.get(name)) is int and group[name] >= 0 for name in names),
                "Incomplete terminal counter envelope")

    if key == "selection":
        from .select import COUNTERS
        require(value.get("schema") == "first-operation-selection-status-v1" and
                type(value.get("unknown_accounting")) is list, "Selection terminal schema")
        integers(value.get("consumption"), COUNTERS)
        if value["status"] == "complete":
            require(not value["unknown_accounting"] and type(value.get("private_receipt")) is dict and
                    type(value.get("selected")) is dict, "Incomplete successful selection")
    elif key == "primary":
        require(value.get("schema") == "first-operation-unit-request-primary-v1", "Primary terminal schema")
        ledger = value.get("ledger")
        require(type(ledger) is dict and set(ledger) == {"source", "model_io", "models", "controller", "unknown_accounting"},
                "Primary ledger groups")
        integers(ledger["source"], ("gzip_opens", "prefix_lines_returned", "compressed_bytes_read", "returned_payload_bytes",
            "target_json_decode_attempts", "target_json_turns_decoded", "target_task_records", "target_candidate_objects"))
        integers(ledger["model_io"], ("checkpoint_hash_passes_started", "checkpoint_hash_passes_completed",
            "model_object_loads_started", "model_object_loads_completed", "np_load_calls_started", "np_load_calls_completed",
            "model_read_calls_started", "model_read_calls_completed", "total_model_bytes_returned_by_reads"))
        integers(ledger["models"], ("task_forward_calls", "task_forward_calls_completed", "task_scored_candidates",
            "task_scored_candidates_completed", "crop_forward_calls", "crop_forward_calls_completed", "training_calls"))
        integers(ledger["controller"], ("real_calls", "real_calls_completed", "baseline_calls", "baseline_calls_completed",
            "intervention_calls", "intervention_calls_completed", "forced_choices", "callbacks", "callbacks_completed",
            "task_candidate_encodings", "task_candidate_encodings_completed", "generated_candidate_objects"))
        unknown = ledger["unknown_accounting"]
        require(type(unknown) is dict and set(unknown) == {"planned_prefix", "failures"} and
                all(type(items) is list for items in unknown.values()), "Primary unknown envelope")
        if value["status"] == "complete":
            require(value.get("baseline_gate_passed") is True and value.get("completed_baselines") == 1 and
                    value.get("completed_interventions") == 1 and not unknown["failures"], "Primary complete flag inconsistent")
    else:
        require(value.get("schema") in ("first-operation-independent-check-v1", "first-operation-independent-check-failure-v1")
                and type(value.get("passed")) is bool and type(value.get("unknown_accounting")) is list, "Checker terminal schema")
        integers(value.get("consumption"), ("selection_receipt_open_attempts", "selection_receipt_files_read",
            "selection_receipt_bytes_read", "primary_receipt_open_attempts", "primary_receipt_files_read", "primary_receipt_bytes_read",
            "pair_receipt_open_attempts", "pair_receipt_files_read", "pair_receipt_bytes_read", "json_decode_attempts",
            "json_files_decoded", "json_bytes_submitted", "json_bytes_decoded", "pairs_verified", "selections_verified"))
        require(not value["passed"] or not value["unknown_accounting"], "Successful checker has unresolved unknown work")
    return value["status"] == "complete" and value.get("passed", True)


def worker(out, key):
    request = read(out / "requests" / (key + ".json"))
    deadline = request["deadline_monotonic"]
    cfg, frozen, state_io = load_state(out)
    inputs = dict(state=state_io, selection_reads_started=0, selection_reads_completed=0, selection_bytes=0,
                  unknown_accounting=[])
    try:
        require(state_io["freeze_sha256"] == request["freeze_sha256"] and time.monotonic() < deadline, "Worker binding or deadline")
        verify_runtime(out, cfg, frozen)
        write(out / "worker-started" / (key + ".json"), dict(request_sha256=sha(out / "requests" / (key + ".json"))))
        if key == "selection":
            from .select import run_selection
            result = run_selection(ROOT, out, cfg, deadline)
        elif key == "primary":
            from .probe import run_primary
            path = out / "selection.json"
            require(0 < path.stat().st_size <= cfg["scope_budget"]["outputs"]["selection_receipt_bytes_max"], "Selection input size")
            inputs["selection_reads_started"] += 1
            try:
                raw = path.read_bytes()
            except BaseException as error:
                inputs["unknown_accounting"].append(dict(quantity="primary_selection_read_bytes", kind=type(error).__name__))
                raise
            inputs["selection_reads_completed"] += 1
            inputs["selection_bytes"] += len(raw)
            require(hashlib.sha256(raw).hexdigest() == request["selection_receipt_sha256"], "Selection receipt binding")
            result = run_primary(ROOT, out, cfg, json.loads(raw), deadline)
        elif key == "independent-check":
            from .check import check
            result = check(out, dict(cfg, selection_receipt_sha256=request["selection_receipt_sha256"],
                                    primary_receipt_sha256=request["primary_receipt_sha256"],
                                    selection_status=request["selection_status"]), deadline)
            require(not (out / "check/failure.json").exists(), "Checker failure exists")
        else:
            raise ValueError("Unknown worker")
        require(result.get("status") == "complete" and result.get("passed", True), "Worker result failed")
        verify_runtime(out, cfg, frozen)
        require(time.monotonic() < deadline, "Worker final deadline")
        write(out / "worker-complete" / (key + ".json"), dict(passed=True,
            request_sha256=sha(out / "requests" / (key + ".json")), freeze_sha256=state_io["freeze_sha256"]))
    finally:
        write(out / "worker-inputs" / (key + ".json"), inputs)


def run(out):
    cfg, frozen, state_io = load_state(out)
    verify_runtime(out, cfg, frozen)
    review, commit = (read(out / name) for name in ("pre-review.json", "pre-execution-commit.json"))
    require(review["passed"] and review["freeze_sha256"] == state_io["freeze_sha256"] == commit["freeze_sha256"]
            and commit["commit"] == git("rev-parse", "HEAD").decode().strip(), "Exact pre-reviewed local commit required")
    expected = {"kaggriculture/" + rel: digest for rel, digest in published_hashes(frozen).items()}
    for name in ("registration.json", "freeze.json", "pre-review.json", *PLANS):
        expected["review/evidence/" + out.name + "/" + name] = sha(out / name)
    require(commit["git_files_sha256"] == expected, "Incomplete committed binding")
    for rel, digest in expected.items():
        require(hashlib.sha256(git("show", commit["commit"] + ":" + rel)).hexdigest() == digest, "Committed bytes changed")
    scope = cfg["scope_budget"]
    started = time.monotonic()
    deadline = started + scope["total_executable_seconds_max"]
    write(out / "started.json", dict(started_monotonic=started, deadline_monotonic=deadline,
        created_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=state_io["freeze_sha256"]))
    attempts, terminal, failure = [], {}, None
    control_bytes, final_state_read = 0, False
    from learning.task_bc_diagnostics_capacity_exit_v1.windows_tree import run_owned
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def dispatch(key, phase_limit, duration):
        nonlocal control_bytes
        phase = min(started + phase_limit, time.monotonic() + duration)
        work = phase - scope["per_close_seconds_max"]
        require(work > time.monotonic(), "No original phase work budget")
        request = dict(freeze_sha256=state_io["freeze_sha256"], deadline_monotonic=work)
        if key != "selection":
            request["selection_receipt_sha256"] = terminal["selection"]["result"]["private_receipt"]["sha256"]
        if key == "independent-check":
            request.update(primary_receipt_sha256=terminal["primary"]["sha256"], selection_status=terminal["selection"]["result"])
        write(out / "requests" / (key + ".json"), request)
        attempt = dict(key=key, passed=False, lifecycle={})
        attempts.append(attempt)
        local = {}
        print("first-operation:", key, flush=True)
        try:
            (out / "logs").mkdir(exist_ok=True)
            with (out / "logs" / (key + ".log")).open("x", encoding="utf-8") as log:
                code = run_owned([sys.executable, "-B", "-m", "learning.task_bc_first_operation_unit_request_v1.experiment",
                    "worker", "--out", str(out), "--key", key], cwd=ROOT, env=env, stream=log,
                    timeout=work - time.monotonic(), phase_deadline=phase, lifecycle=attempt["lifecycle"],
                    local_lifecycle=local, memory_limit_bytes=scope["process_tree_private_commit_bytes_max"])
            life = attempt["lifecycle"]
            require(code == 0 and life["tree_closed"] and life["active_processes_after_close"] == 0 and
                    life["natural_tree_exit"] and not life["termination_requested"] and life["job_handle_closed"] and
                    life["all_process_handles_closed"], "Worker lifecycle rejected")
            complete = read(out / "worker-complete" / (key + ".json"))
            require(complete["passed"] and complete["request_sha256"] == sha(out / "requests" / (key + ".json")) and
                    complete["freeze_sha256"] == state_io["freeze_sha256"], "Worker completion binding differs")
            require(time.monotonic() < phase, "Worker phase deadline after bookkeeping")
            attempt["passed"] = True
        finally:
            write(out / "local-lifecycle" / (key + ".json"), local)
            path = {"selection": "selection-status.json", "primary": "primary-receipt.json",
                    "independent-check": "check/failure.json" if (out / "check/failure.json").exists() else "check/complete.json"}[key]
            record = dict(known=False, reason="Missing/malformed terminal counters are unknown, not zero.")
            try:
                require(0 < (out / path).stat().st_size <= 131072, "Terminal receipt size")
                raw = (out / path).read_bytes()
                value = json.loads(raw)
                record["partial_receipt"] = value
                passed = terminal_passed(key, value)
                record = dict(known=True, passed=passed,
                              sha256=hashlib.sha256(raw).hexdigest(), result=value)
            except Exception as error:
                record["error_type"] = type(error).__name__
            terminal[key] = record
            try:
                inputs = read(out / "worker-inputs" / (key + ".json"))
                require(inputs["state"] == state_io and not inputs["unknown_accounting"], "Worker control input accounting")
                control_bytes += inputs["state"]["bytes"] + inputs["selection_bytes"]
                attempt["control_input_bytes"] = inputs["state"]["bytes"] + inputs["selection_bytes"]
                attempt["control_input_consumption_known"] = True
            except Exception as error:
                attempt["control_input_consumption_known"] = False
                attempt["control_input_error_type"] = type(error).__name__
            write(out / "attempts" / (key + ".json"), attempt)
        require(record["known"] and record["passed"] and attempt["control_input_consumption_known"], "Unverified terminal work")
        require(control_bytes + state_io["bytes"] <= scope["source_and_metadata"]["fixed_registration_and_selection_metadata_bytes_max"],
                "Control input byte budget")

    try:
        dispatch("selection", 15, 15)
        dispatch("primary", 80, 65)
        dispatch("independent-check", 100, 20)
        new_cfg, new_frozen, final_io = load_state(out)
        control_bytes += final_io["bytes"]
        final_state_read = True
        require(final_io == state_io and new_cfg == cfg and new_frozen == frozen, "Final control binding changed")
        verify_runtime(out, cfg, frozen)
        require(not (out / "check/failure.json").exists(), "Independent checker failure")
        require(control_bytes <= scope["source_and_metadata"]["fixed_registration_and_selection_metadata_bytes_max"], "Control metadata budget")
    except BaseException as error:
        failure = dict(kind=type(error).__name__, retry=False)
        raise
    finally:
        try:
            history = read(out / "protection-before.json")["stat_before"]
            changed = [rel for rel, value in history.items() if not (ROOT / rel).is_file() or stat(ROOT / rel) != value]
            write(out / "history-stat-check.json", dict(passed=not changed, checked_files=len(history), changed_files=changed,
                method="Inherited hashes/stat, plus separately frozen code/metadata byte verification; not a fresh whole-history hash pass."))
            if time.monotonic() >= deadline and failure is None:
                failure = dict(kind="TimeoutError", phase="finalization", retry=False)
            summary = dict(passed=failure is None and not changed, attempts=len(attempts),
                accepted_jobs=sum(row["passed"] for row in attempts), terminal_consumption=terminal,
                failure=failure, retries=0, elapsed_seconds=time.monotonic() - started, freeze_sha256=state_io["freeze_sha256"],
                control_input_bytes_known=control_bytes, final_binding_read_completed=final_state_read,
                all_control_consumption_known=all(row.get("control_input_consumption_known", False) for row in attempts),
                original_policy_admission=False, previous_diagnostic_stage_acceptance=False, previous_strict_review_acceptance=False,
                training_runs=0, new_games=0, official_transitions=0, history_stat_check_sha256=sha(out / "history-stat-check.json"))
            raw = (json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            require(len(raw) <= scope["outputs"]["public_summary_bytes_max"], "Public summary size budget")
            write(out / "summary.json", summary)
        finally:
            output_accounting = account_outputs(out, scope["outputs"]["all_new_diagnostic_artifact_bytes_max"])
        require(output_accounting["known"] and output_accounting["within_limit"], "Output aggregate unknown or over budget")
        require(not changed and failure is None, "Stopped with preserved failure or historical change")

    def final_gate():
        artifact_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        if time.monotonic() >= deadline or artifact_bytes > scope["outputs"]["all_new_diagnostic_artifact_bytes_max"]:
            write(out / "final-deadline-failure.json", dict(passed=False, elapsed_seconds=time.monotonic() - started,
                artifact_bytes=artifact_bytes, provisional_summary_not_final_acceptance=True))
            raise ValueError("Final execution budget exceeded")
        return artifact_bytes

    total = final_gate()
    write(out / "execution-acceptance.json", dict(passed=True, summary_sha256=sha(out / "summary.json"),
        elapsed_seconds=time.monotonic() - started, artifact_bytes_before_acceptance=total,
        requires_zero_process_exit_and_no_final_deadline_or_checker_failure=True))
    final_gate()
    return dict(passed=True, accepted_jobs=len(attempts), elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "freeze", "run", "worker"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--key")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    result = worker(out, args.key) if args.command == "worker" else globals()[args.command](out)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
