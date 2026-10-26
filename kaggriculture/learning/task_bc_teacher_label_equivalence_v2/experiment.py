"""Versioned integration of the repaired 3+4 protocol and one fixed case.

Preparation reads source and safe metadata plus raw/model file stat only.
Original payloads belong exclusively to source.read_case in the primary worker.
The parent consumes safe terminal receipts and never reads case.json or models.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from learning.task_bc_first_operation_unit_request_v1 import experiment as previous


ROOT, PUBLIC = previous.ROOT, previous.PUBLIC
PACKAGE = Path(__file__).resolve().parent
STAGE = "task-bc-teacher-label-equivalence-v2-20261026"
SCHEMA = "task-bc-teacher-label-equivalence-v2"
PREVIOUS = ROOT / "artifacts/task-bc-teacher-label-protocol-repair-v1-20261026"
OLD = ROOT / "artifacts/task-bc-teacher-label-equivalence-v1-20261026"
OUT = ROOT / "artifacts" / STAGE
BASE_HEAD = "1bd2ca626400a597dafb4df342c8f84689af328f"
PLANS = ("follow-on-integration-advice.json", "root-follow-on-integration-decision.json")
PLAN_SHA = "21261295d9ca6599274954a5d85b3759b687fce2cdd9f950ec91f48bdc7097b8"
ADVICE_SHA = "b7771c18ec3090b7f1a940fa5878adf7af2c443097af1fc693ece3bed935e8df"
NEW_FILES = {"__init__.py", "CONTRACT.md", "probe.py", "check.py", "experiment.py"}
WORKERS = ("primary", "independent-check")
require, read, sha, stat, git = previous.require, previous.read, previous.sha, previous.stat, previous.git
write, account_outputs = previous.write, previous.account_outputs
load_state, published_hashes = previous.load_state, previous.published_hashes
official, dependencies = previous.prior.official, previous.prior.dependencies


def _out(path):
    path = Path(path).resolve()
    require(path == OUT.resolve(), "Only the fixed new output directory is authorized")
    return path


def _tick(deadline):
    require(time.monotonic() < deadline, "Original execution deadline exhausted")


def _package_sources():
    paths = [path for path in PACKAGE.iterdir() if path.is_file() and path.suffix in (".py", ".md")]
    require({path.name for path in paths} == NEW_FILES, "Exactly the five new versioned source files must be present")
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def _copy_new(source, target):
    raw = source.read_bytes()  # Only the two pinned safe planning documents.
    expected = {PLANS[0]: ADVICE_SHA, PLANS[1]: PLAN_SHA}[source.name]
    require(target.name == source.name and hashlib.sha256(raw).hexdigest() == expected,
            "Pinned safe planning source changed")
    if target.exists():
        require(target.is_file() and not target.is_symlink() and sha(target) == expected,
                "Existing planning copy is not the exact pinned file")
        return  # Preserve the already-reviewed copy byte-for-byte; never overwrite.
    with target.open("xb", buffering=0) as stream:
        require(stream.write(raw) == len(raw), "Incomplete safe planning copy")


def _verify_public(sources, mappings):
    overrides = {}
    for relative, digest in sources.items():
        runtime, public = ROOT / relative, PUBLIC / "kaggriculture" / relative
        require(sha(runtime) == digest, "Frozen runtime source changed")
        other = sha(public)
        if other != digest:
            mapping = mappings.get(relative)
            require(type(mapping) is dict and mapping["runtime_sha256"] == digest
                    and mapping["public_sha256"] == other
                    and runtime.read_bytes().replace(b"\r\n", b"\n")
                    == public.read_bytes().replace(b"\r\n", b"\n"), "Undeclared source byte mapping")
            overrides[relative] = other
    return overrides


def prepare(out):
    out = _out(out)
    if out.exists():
        require(out.is_dir(), "Preparation output must be a directory")
        entries = list(out.iterdir())
        require(len(entries) == 2 and {path.name for path in entries} == set(PLANS)
                and all(path.is_file() and not path.is_symlink() for path in entries)
                and sha(out / PLANS[0]) == ADVICE_SHA and sha(out / PLANS[1]) == PLAN_SHA,
                "Only the two exact pinned planning copies may precede registration; other artifacts forbid preparation")
    require(not git("status", "--porcelain").strip(), "Clean accepted checkout required")
    require(git("rev-parse", "HEAD").decode().strip() == BASE_HEAD, "Accepted public base changed")
    require(sha(PREVIOUS / PLANS[1]) == PLAN_SHA and sha(PREVIOUS / PLANS[0]) == ADVICE_SHA,
            "Adopted planning bytes changed")
    decision, advice = read(PREVIOUS / PLANS[1]), read(PREVIOUS / PLANS[0])
    require(decision["status"] == "planned_not_started" and decision["one_next_stage"] == SCHEMA
            and decision["accepted"] is True and type(advice) is dict
            and decision["based_on_public_commit"] == BASE_HEAD and decision["advice_sha256"] == ADVICE_SHA,
            "Planning state or binding changed")
    require(decision["planned_out"] == OUT.relative_to(ROOT).as_posix()
            and decision["runtime_sources_to_preserve"] == 158, "New stage path or source count changed")
    accepted, publication = read(PREVIOUS / "root-adjudication.json"), read(PREVIOUS / "publication-verification.json")
    require(accepted["passed"] is True and publication["passed"] is True
            and publication["commit"] == publication["remote_head"] == BASE_HEAD and publication["clean"] is True,
            "Prior root/publication acceptance missing")
    for field, name in (("freeze_sha256", "freeze.json"), ("summary_sha256", "summary.json"),
                        ("acceptance_sha256", "execution-acceptance.json"),
                        ("review_sha256", "independent-result-review.json")):
        require(accepted[field] == sha(PREVIOUS / name), "Prior acceptance evidence changed")
    require(type(accepted["parent_exit_code"]) is int and accepted["parent_exit_code"] == 0
            and accepted["old_154_source_bytes_preserved"] is True and accepted["runtime_source_files_verified"] == 158
            and accepted["model_scoring_groups"] == 3921 and accepted["model_scoring_candidates"] == 228505
            and all(accepted[key] is False for key in ("original_equivalence_execution_passed",
                "old_strict_result_review_passed", "original_policy_admission")),
            "Prior execution or cumulative consumption changed")
    base, repair = read(PREVIOUS / "freeze.json"), read(PREVIOUS / "registration.json")
    old, old_freeze = read(OLD / "registration.json"), read(OLD / "freeze.json")
    require(len(base["runtime_files"]) == 158 and len(old_freeze["runtime_files"]) == 154
            and all(base["runtime_files"].get(name) == digest for name, digest in old_freeze["runtime_files"].items())
            and len(base["line_ending_mappings"]) == 4
            and base["registration_sha256"] == sha(PREVIOUS / "registration.json"), "Inherited source inventory changed")
    require(_verify_public(base["runtime_files"], base["line_ending_mappings"]) == base["public_source_overrides"],
            "Inherited public source bindings changed")
    for name, digest in repair["previous_receipt_sha256"].items():
        require(sha(OLD / name) == digest, "Preserved failed-stage safe receipt changed")
    basis = old["verified_source_basis_sha256"]
    for relative, digest in basis.items():
        require(sha(ROOT / relative) == digest, "Adopted source/safe-metadata basis changed")
    fixed = deepcopy(decision["fixed_case"])
    require(fixed == old["fixed_case"] and (fixed["game"], fixed["seed"], fixed["opponent"], fixed["side"],
            fixed["step"], fixed["callback_index_zero_based"], fixed["split"]) ==
            ("collect-31317742-v1-0-teacher", 31317742, "v1", 0, 15, 0, "original_train"), "Fixed original case changed")
    require(decision["budget"] == dict(old["budget"], tiny_in_memory_cases_max=7), "Original real-source/controller budget changed")
    source_stats = {}
    for spec in fixed["raw_source_files"].values():
        expected = dict(size=spec["bytes"], mtime_ns=spec["mtime_ns"])
        require(stat(ROOT / spec["path"]) == expected, "Fixed original source stat changed")
        source_stats[spec["path"]] = expected
    require(repair["model_stat"] == old["model_protection"] and repair["default_sha256"] == old["default_sha256"]
            and repair["inherited_official_sha256"] == old["official_sha256"], "Inherited protection changed")
    for relative, expected in repair["model_stat"].items():
        require(stat(ROOT / relative) == expected, "Inherited model stat changed")
    require(sha(ROOT / "main.py") == old["default_sha256"] and official() == old["official_sha256"]
            and dependencies() == old["dependencies"], "Default/official/dependency identity changed")
    require(fixed["official_engine_sha256"] == old["official_sha256"]["rules"], "Plan official engine differs")
    history = read(PREVIOUS / "protection-before.json")["stat_before"].copy()
    require(all((ROOT / relative).is_file() and stat(ROOT / relative) == expected
                for relative, expected in history.items()), "Inherited historical stat changed")
    for path in PREVIOUS.rglob("*"):
        if path.is_file():
            history[path.relative_to(ROOT).as_posix()] = stat(path)
    report = ROOT / "TASK_BC_TEACHER_LABEL_PROTOCOL_REPAIR_V1.zh-CN.md"
    history[report.relative_to(ROOT).as_posix()] = stat(report)
    require(all((ROOT / relative).is_file() and stat(ROOT / relative) == expected
                for relative, expected in history.items()), "Protected historical stat changed")
    out.mkdir(parents=True, exist_ok=True)  # The narrow unregistered-state gate above has passed.
    for name in PLANS:
        _copy_new(PREVIOUS / name, out / name)
    write(out / "protection-before.json", dict(stat_before=history,
        method="Inherited history SHA claims plus unchanged before/after stat. Only separately frozen source and safe metadata are byte-hashed; no whole-history content pass."))
    metadata_names = ("freeze.json", "registration.json", "summary.json", "primary.json", "independent-result-review.json",
                      "execution-acceptance.json", "root-adjudication.json", "publication-verification.json", "parent-exit-observation.json")
    metadata = dict(old["source_metadata_sha256"])
    metadata.update({(OLD / name).relative_to(ROOT).as_posix(): digest for name, digest in repair["previous_receipt_sha256"].items()})
    metadata.update({(PREVIOUS / name).relative_to(ROOT).as_posix(): sha(PREVIOUS / name) for name in metadata_names})
    for relative, digest in metadata.items():
        require(sha(ROOT / relative) == digest, "Inherited safe metadata changed")
    cfg = dict(schema=SCHEMA, stage_id=STAGE, created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=BASE_HEAD,
        fixed_case=fixed, budget=deepcopy(decision["budget"]), new_synthetic_limits=deepcopy(decision["new_synthetic_limits"]),
        versioned_contract=dict(primary_schema="teacher-label-equivalence-primary-v2",
            check_schema="teacher-label-equivalence-independent-check-v2", private_case_schema="teacher-label-equivalence-case-v1",
            source_certificate_schema="teacher-label-original-source-certificate-v1",
            original_reader="learning.task_bc_teacher_label_equivalence_v1.source.read_case",
            repaired_protocol="learning.task_bc_teacher_label_protocol_repair_v1.protocol.protocol_checks",
            new_protocol_cases=7, original_source_sha_reads_only_in_unique_run=True,
            expected_fault_exception_scope="Only read_failure_boundary: OSError, final AssertionError and its three null unknown quantities. Other protocol failures and all production-source failure unknowns still reject.",
            index_prefix_limit="At 256 inspected rows without the target, inconclusive_source_prefix_limit proves neither presence nor absence and never permits replacement."),
        inherited_runtime_files=base["runtime_files"], line_ending_mappings=base["line_ending_mappings"],
        source_metadata_sha256=metadata, verified_source_basis_sha256=basis,
        planning_sha256={name: sha(out / name) for name in PLANS}, protection_sha256=sha(out / "protection-before.json"),
        source_stat_protection=source_stats, model_protection=deepcopy(old["model_protection"]),
        default_sha256=old["default_sha256"], official_sha256=old["official_sha256"],
        engine_commit=old["engine_commit"], dependencies=old["dependencies"],
        control_metadata_limit_scope="Registration+freeze raw bytes returned to primary and independent-check workers plus the parent's one final binding read: three execution reads total. Parent preregistration preflight and other frozen source/safe identity verification reads are code/identity checks, not new original-data consumption.",
        output_limit_scope="All current OUT files including planning/registration, private case, logs, process identity, partial and failure receipts count toward the 4 MiB executable cap. Later independent human review and publication are separately identified phases.",
        prior_task_scoring=dict(groups=3921, candidates=228505, new_model_forwards=0),
        historical_training_case_not_independent_holdout=True, local_commit_not_external_timestamp=True,
        original_policy_admission=False, previous_diagnostic_stage_acceptance=False,
        previous_strict_review_acceptance=False, original_equivalence_execution_passed=False,
        old_strict_result_review_passed=False, retries_allowed=0)
    write(out / "registration.json", cfg)
    write(out / "public-before.json", dict(head=BASE_HEAD, clean=True,
        tracked_files=len([value for value in git("ls-files", "-z").split(b"\0") if value])))
    return dict(prepared=True, source_data_reads=0, model_reads=0, fixtures_executed=0)


def freeze(out):
    out = _out(out)
    require(not (out / "freeze.json").exists() and not (out / "started.json").exists(), "Freeze is exclusive and precedes execution")
    cfg = read(out / "registration.json")
    base = read(PREVIOUS / "freeze.json")
    require(cfg["inherited_runtime_files"] == base["runtime_files"] and len(base["runtime_files"]) == 158,
            "Inherited source inventory differs")
    sources = dict(base["runtime_files"], **_package_sources())
    require(len(sources) == 163, "Exactly 158 inherited plus five new files must be frozen")
    overrides = _verify_public(sources, cfg["line_ending_mappings"])
    frozen = dict(registration_sha256=sha(out / "registration.json"), runtime_files=sources,
        public_source_overrides=overrides, line_ending_mappings=cfg["line_ending_mappings"],
        public_hash_rule="Runtime bytes are exact except the four inherited explicit CRLF/LF public mappings.")
    verify_runtime(out, cfg, frozen)
    write(out / "freeze.json", frozen)
    state_bytes = (out / "registration.json").stat().st_size + (out / "freeze.json").stat().st_size
    require(state_bytes * 3 <= cfg["budget"]["fixed_control_metadata_bytes_max"], "Three control-state reads exceed the fixed cap")
    return dict(frozen=True, freeze_sha256=sha(out / "freeze.json"), sources=len(sources),
                state_payload_bytes=state_bytes, execution_state_read_count=3)


def verify_runtime(out, cfg, frozen):
    require(cfg["schema"] == SCHEMA and cfg["stage_id"] == STAGE and cfg["baseline_head"] == BASE_HEAD,
            "Registration identity changed")
    decision = read(out / PLANS[1])
    require(sha(out / PLANS[1]) == PLAN_SHA and sha(out / PLANS[0]) == ADVICE_SHA
            and cfg["fixed_case"] == decision["fixed_case"]
            and cfg["budget"] == decision["budget"] and cfg["new_synthetic_limits"] == decision["new_synthetic_limits"],
            "Registered plan/scope changed")
    require(frozen["registration_sha256"] == sha(out / "registration.json"), "Registration/freeze binding changed")
    require(set(frozen["runtime_files"]) == set(cfg["inherited_runtime_files"]) | set(_package_sources()),
            "Runtime source inventory changed")
    for relative, digest in frozen["runtime_files"].items():
        require(sha(ROOT / relative) == digest, "Frozen source changed")
    for bindings in (cfg["source_metadata_sha256"], cfg["verified_source_basis_sha256"]):
        for relative, digest in bindings.items():
            require(sha(ROOT / relative) == digest, "Safe source/metadata binding changed")
    for name, digest in cfg["planning_sha256"].items():
        require(sha(out / name) == digest, "Planning copy changed")
    require(sha(out / "protection-before.json") == cfg["protection_sha256"], "History protection metadata changed")
    for protection in (cfg["source_stat_protection"], cfg["model_protection"]):
        for relative, expected in protection.items():
            require(stat(ROOT / relative) == expected, "Original data/model stat changed")
    require(sha(ROOT / "main.py") == cfg["default_sha256"] and official() == cfg["official_sha256"]
            and dependencies() == cfg["dependencies"], "Default/official/dependency identity changed")


def _commit_gate(out, cfg, frozen, state_io):
    review, commit = read(out / "pre-review.json"), read(out / "pre-execution-commit.json")
    require(review["passed"] is True and review["freeze_sha256"] == state_io["freeze_sha256"]
            == commit["freeze_sha256"] and commit["commit"] == git("rev-parse", "HEAD").decode().strip()
            and not git("status", "--porcelain").strip(), "Exact clean pre-reviewed local commit required")
    git("merge-base", "--is-ancestor", BASE_HEAD, commit["commit"])
    expected = {"kaggriculture/" + relative: digest for relative, digest in published_hashes(frozen).items()}
    for name in ("registration.json", "freeze.json", "pre-review.json", *PLANS):
        expected["review/evidence/" + out.name + "/" + name] = sha(out / name)
    require(commit["git_files_sha256"] == expected, "Committed manifest is incomplete")
    require(_verify_public(frozen["runtime_files"], cfg["line_ending_mappings"]) == frozen["public_source_overrides"],
            "Public source changed after freeze")
    for relative, digest in expected.items():
        require(hashlib.sha256(git("show", commit["commit"] + ":" + relative)).hexdigest() == digest,
                "Actual committed bytes differ from manifest")
    return sha(out / "pre-execution-commit.json")


def _state_input(out, accounting):
    accounting.update(reads_started=1, bytes_returned=None)
    try:
        cfg, frozen, state_io = load_state(out)
    except BaseException as error:
        accounting["unknown_accounting"].append(dict(quantity="partial_control_state_bytes_and_decodes",
                                                     value=None, kind=type(error).__name__))
        raise
    accounting.update(reads_completed=1, bytes_returned=state_io["bytes"], state=state_io)
    return cfg, frozen, state_io


def _new_input_accounting():
    return dict(reads_started=0, reads_completed=0, bytes_returned=0, state=None, unknown_accounting=[])


def _protocol_passed(protocol):
    """Validate the new receipt only; never execute either fixture producer."""
    from learning.task_bc_teacher_label_protocol_repair_v1.experiment import check_primary
    require(type(protocol) is dict and protocol.get("schema") == "teacher-label-equivalence-protocol-v2"
            and protocol.get("passed") is True, "V2 protocol completion missing")
    first, second = protocol["source"], protocol["probe"]
    require(first["started"] is True and first["completed"] is True
            and second["started"] is True and second["completed"] is True
            and second["partial_consumption_unknown"] is False, "Protocol progress incomplete")
    require(first["result"] == dict(schema="teacher-label-protocol-repair-v1", passed=True, scenario_count=3),
            "Repaired protocol result changed")
    ledger = first["ledger"]
    require(ledger["schema"] == "teacher-label-protocol-repair-v1" and ledger["status"] == "complete"
            and ledger["passed"] is True, "Repaired protocol ledger incomplete")
    # Reuse the accepted, frozen receipt comparison. It performs no fixture,
    # source, model or filesystem work and does not alter the supplied ledger.
    checked = check_primary(dict(passed=True, result=first["result"], ledger=ledger))
    gate = first["gate_receipt"]
    require(gate["passed"] is True and gate["cases"] == checked["cases"] == 3
            and gate["observed_totals"] == checked["observed_totals"], "Producer protocol gate binding differs")
    rows = ledger["cases"]
    require([row["name"] for row in rows] == ["zero_drain", "positive_drain", "read_failure_boundary"]
            and all(row["status"] == "complete" and row["passed"] is True for row in rows), "Protocol case identity/progress")
    for row in rows[:2]:
        failure, unknown = row["failure"], row["consumption"]["unknown_accounting"]
        require(failure["operation_error"] is None and failure["assertion_error"] is None
                and failure["unexpected_error"] is None and not failure["close_errors"]
                and not unknown["failures"] and len(unknown["planned_prefix"]) == 1,
                "Normal synthetic case has unexpected failure")
        entry = unknown["planned_prefix"][0]
        require(entry["kind"] == "index" and entry["quantity"] == "gzip_internal_unreturned_payload_bytes"
                and entry["value"] is None and entry["reason"] == "intentional_prefix_without_eof_probe",
                "Normal synthetic prefix unknown changed")
    fault = rows[2]
    unknown = fault["consumption"]["unknown_accounting"]
    expected = {"raw_bytes_in_failed_read": "OSError", "payload_bytes_in_failed_gzip_call": "OSError",
                "gzip_internal_unreturned_payload_bytes": "failed_prefix_read"}
    require(not unknown["planned_prefix"] and len(unknown["failures"]) == 3
            and {entry["quantity"] for entry in unknown["failures"]} == set(expected)
            and all(entry["kind"] == "index" and entry["value"] is None
                    and entry["reason"] == expected[entry["quantity"]] for entry in unknown["failures"]),
            "Only the exact synthetic read-failure unknowns are expected")
    count = fault["consumption"]
    require(count["raw_read_calls_started"] == 2 and count["raw_read_calls_completed"] == 1
            and count["raw_bytes_returned"] == count["raw_bytes_hashed"] == 2,
            "Synthetic fault consumed an unexpected read")
    result = second["result"]
    require(result["passed"] is True and type(result["cases"]) is int and result["cases"] == 4
            and type(result["synthetic_candidates"]) is int and result["synthetic_candidates"] == 6
            and all(type(result[key]) is int and result[key] == 0 for key in
                    ("real_controller_calls", "model_calls", "official_transitions", "temporary_files")),
            "Original four probe checks incomplete")
    known = sum(row["status"] == "complete" and row["passed"] is True for row in rows) + result["cases"]
    require(type(protocol["cases"]) is int and type(protocol["cases_known_complete"]) is int
            and protocol["cases"] == protocol["cases_known_complete"] == known == 7
            and type(protocol["synthetic_candidates"]) is int and protocol["synthetic_candidates"] == 6,
            "Actual repaired three plus original four protocol counts differ")


def terminal_passed(key, value):
    require(type(value) is dict and value.get("status") in ("complete", "failed"), "Terminal status missing")

    def integers(group, names):
        require(type(group) is dict and all(type(group.get(name)) is int and group[name] >= 0 for name in names),
                "Incomplete terminal counter envelope")

    if key == "primary":
        require(value.get("schema") == "teacher-label-equivalence-primary-v2", "Primary terminal schema")
        protocol = value.get("protocol_checks")
        require(type(protocol) is dict and protocol.get("schema") == "teacher-label-equivalence-protocol-v2"
                and type(protocol.get("source")) is dict and type(protocol["source"].get("ledger")) is dict
                and type(protocol.get("probe")) is dict, "Caller-owned protocol receipt missing")
        ledger = value.get("ledger")
        require(type(ledger) is dict and type(ledger.get("unknown_accounting")) is list
                and type(ledger.get("patches_restored")) is bool, "Primary ledger envelope")
        integers(ledger.get("controller"), tuple(name + suffix for name in
            ("calls", "original", "baseline", "trials", "callbacks", "encodings", "encoded_candidates",
             "crop_callbacks", "crop_encodings", "crop_feature_values") for suffix in ("_started", "_completed")))
        integers(ledger.get("prohibited"), ("model_read_attempts", "np_load_attempts", "model_forwards",
                                           "training_runs", "new_games", "official_transitions"))
        source = ledger.get("source")
        if source is not None:
            integers(source, ("sources_started", "sources_completed", "sources_failed", "raw_opens_started",
                "raw_opens_completed", "raw_bytes_returned", "raw_bytes_hashed", "raw_hashes_completed",
                "returned_payload_bytes", "json_decodes_started", "json_decodes_completed", "json_bytes_submitted",
                "json_bytes_decoded", "raw_closes_started", "raw_closes_completed"))
            require("index_prefix" not in source and type(source.get("unknown_accounting")) is dict,
                    "Safe source ledger envelope")
        if value["status"] == "complete":
            _protocol_passed(protocol)
            require(source is not None and source.get("status") in ("completed", "completed_unavailable")
                    and not source["unknown_accounting"]["failures"] and not ledger["unknown_accounting"]
                    and ledger["patches_restored"] is True and all(number == 0 for number in ledger["prohibited"].values()),
                    "Completed primary has unresolved work")
            require(type(value.get("case")) is dict and value["case"].get("path") == "case.json"
                    and type(value["case"].get("bytes")) is int and 0 < value["case"]["bytes"] <= 1048576,
                    "Primary case manifest missing")
            require(value.get("outcome") in ("equivalent_alternative", "local_no_match", "inconclusive_trial_limit",
                "inconclusive_no_fixed_callback", "inconclusive_source_prefix_limit", "inconclusive_single_candidate"),
                "Primary complete outcome missing")
    else:
        require(value.get("schema") == "teacher-label-equivalence-independent-check-v2"
                and type(value.get("passed")) is bool and type(value.get("partial_unknown")) is list,
                "Independent checker terminal envelope")
        counts = value.get("consumption")
        integers(counts, ("opens_started", "opens_completed", "reads_started", "reads_completed", "bytes_returned",
            "bytes_hashed", "json_decodes_started", "json_decodes_completed", "json_bytes_submitted",
            "json_bytes_decoded", "closes_started", "closes_completed"))
        if value["status"] == "complete":
            require(value["passed"] is True and not value["partial_unknown"] and type(value.get("safe_result")) is dict,
                    "Completed checker has unresolved work")
            require(all(counts[name + suffix] == 2 for name in ("opens", "reads", "json_decodes", "closes")
                        for suffix in ("_started", "_completed"))
                    and counts["bytes_returned"] == counts["bytes_hashed"] == counts["json_bytes_submitted"]
                        == counts["json_bytes_decoded"] <= 2097152, "Checker exact receipt consumption")
    return value["status"] == "complete" and value.get("passed", True) is True


def worker(out, key):
    out = _out(out)
    require(key in WORKERS, "Unknown worker")
    inputs, cfg, frozen, result = _new_input_accounting(), None, None, None
    deadline = None
    try:
        request = read(out / "requests" / (key + ".json"))
        started = read(out / "started.json")
        deadline = request["deadline_monotonic"]
        _tick(deadline)
        require(request["key"] == key and request["started_sha256"] == sha(out / "started.json")
                and deadline < request["phase_deadline_monotonic"] <= started["deadline_monotonic"], "Worker request deadline binding")
        cfg, frozen, state_io = _state_input(out, inputs)
        require(state_io["freeze_sha256"] == request["freeze_sha256"] == started["freeze_sha256"], "Worker freeze binding")
        limits = cfg["budget"]
        phase_limit = limits["primary_phase_deadline_seconds"] if key == "primary" else limits["independent_phase_deadline_seconds"]
        duration = phase_limit if key == "primary" else limits["independent_phase_deadline_seconds"] - limits["primary_phase_deadline_seconds"]
        require(started["deadline_monotonic"] == started["started_monotonic"] + limits["total_executable_seconds_max"]
                and request["phase_deadline_monotonic"] == min(started["started_monotonic"] + phase_limit,
                    request["phase_started_monotonic"] + duration)
                and deadline == request["phase_deadline_monotonic"] - limits["per_close_seconds_max_included_in_original_phase"],
                "Original worker phase allocation changed")
        require(_commit_gate(out, cfg, frozen, state_io) == request["preexecution_manifest_sha256"], "Worker commit manifest changed")
        verify_runtime(out, cfg, frozen)
        _tick(deadline)
        write(out / "worker-started" / (key + ".json"), dict(request_sha256=sha(out / "requests" / (key + ".json"))))
        if key == "primary":
            from .probe import run_primary
            result = run_primary(ROOT, out, cfg, deadline)
        else:
            from .check import check
            result = check(out, dict(cfg, case_receipt_sha256=request["case_receipt_sha256"],
                                    primary_receipt_sha256=request["primary_receipt_sha256"]), deadline)
            require(not (out / "check/failure.json").exists(), "Checker failure receipt dominates")
        require(terminal_passed(key, result), "Worker returned a failed terminal")
    finally:
        original_error, post_error = sys.exc_info()[1], None
        if cfg is not None and frozen is not None:
            try:
                verify_runtime(out, cfg, frozen)
                inputs["post_runtime_verified"] = True
            except BaseException as error:
                inputs.update(post_runtime_verified=False, post_runtime_error_type=type(error).__name__)
                post_error = error
        try:
            write(out / "worker-inputs" / (key + ".json"), inputs)
        finally:
            if post_error is not None:
                if original_error is None:
                    raise post_error
                original_error.add_note("Post-worker identity verification also failed; see worker-inputs.")
    _tick(deadline)
    write(out / "worker-complete" / (key + ".json"), dict(passed=True,
        request_sha256=sha(out / "requests" / (key + ".json")), freeze_sha256=state_io["freeze_sha256"]))
    _tick(deadline)
    return dict(passed=True, worker=key)


def _terminal(out, key):
    path = out / ("primary-receipt.json" if key == "primary" else
                  "check/failure.json" if (out / "check/failure.json").exists() else "check/complete.json")
    record = dict(known=False, passed=False, reason="Missing or malformed terminal consumption is partial/unknown, never zero.")
    try:
        require(0 < path.stat().st_size <= (2097152 if key == "primary" else 65536), "Safe terminal byte limit")
        raw = path.read_bytes()
        value = json.loads(raw)
        record["partial_receipt"] = value
        passed = terminal_passed(key, value)
        if key == "independent-check" and (out / "check/failure.json").exists():
            passed = False
        record = dict(known=True, passed=passed, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), result=value)
    except BaseException as error:
        record["error_type"] = type(error).__name__
    return record


def _history(out):
    result = dict(passed=False, checked_files=0, changed_files=[], unknown_accounting=[],
        method="Inherited history hashes and unchanged stat; source/safe metadata are separately byte-verified. No whole-history content pass.")
    try:
        history = read(out / "protection-before.json")["stat_before"]
        for relative, expected in history.items():
            result["checked_files"] += 1
            if not (ROOT / relative).is_file() or stat(ROOT / relative) != expected:
                result["changed_files"].append(relative)
        result["passed"] = not result["changed_files"]
    except BaseException as error:
        result["unknown_accounting"].append(dict(quantity="remaining_history_stat_checks", value=None,
                                                  kind=type(error).__name__))
    write(out / "history-stat-check.json", result)
    return result


def run(out):
    out = _out(out)
    require(not (out / "started.json").exists(), "Existing start forbids retries")
    cfg, frozen, state_io = load_state(out)  # Static preregistration identity check, before the executable clock.
    verify_runtime(out, cfg, frozen)
    manifest_sha = _commit_gate(out, cfg, frozen, state_io)
    budget = cfg["budget"]
    require(state_io["bytes"] * 3 <= budget["fixed_control_metadata_bytes_max"], "Control read allocation changed")
    started = time.monotonic()
    deadline = started + budget["total_executable_seconds_max"]
    write(out / "started.json", dict(started_monotonic=started, deadline_monotonic=deadline,
        created_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=state_io["freeze_sha256"],
        preexecution_manifest_sha256=manifest_sha))
    attempts, terminal, failures = [], {}, []
    control = dict(bytes_known=0, completed_reads=0, unknown_accounting=[], final_binding_read_completed=False)
    final_input = _new_input_accounting()
    original_error, history, output_accounting = None, None, None
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    def record_control(inputs, name):
        amount = inputs.get("bytes_returned")
        if type(amount) is int and amount >= 0:
            control["bytes_known"] += amount
        else:
            control["unknown_accounting"].append(dict(component=name, quantity="control_state_bytes", value=None))
        completed = inputs.get("reads_completed")
        if type(completed) is int and completed >= 0:
            control["completed_reads"] += completed
        else:
            control["unknown_accounting"].append(dict(component=name, quantity="completed_control_reads", value=None))
        unknown = inputs.get("unknown_accounting")
        if type(unknown) is list and all(type(entry) is dict for entry in unknown):
            control["unknown_accounting"].extend(dict(entry, component=name) for entry in unknown)
        else:
            control["unknown_accounting"].append(dict(component=name, quantity="control_input_failure_details", value=None))

    def dispatch(key, phase_limit, duration):
        from learning.task_bc_diagnostics_capacity_exit_v1.windows_tree import run_owned
        verify_runtime(out, cfg, frozen)
        phase_start = time.monotonic()
        phase = min(started + phase_limit, phase_start + duration)
        work = phase - budget["per_close_seconds_max_included_in_original_phase"]
        _tick(work)
        request = dict(key=key, started_sha256=sha(out / "started.json"), freeze_sha256=state_io["freeze_sha256"],
            preexecution_manifest_sha256=manifest_sha, deadline_monotonic=work, phase_deadline_monotonic=phase,
            phase_started_monotonic=phase_start)
        if key == "independent-check":
            request.update(case_receipt_sha256=terminal["primary"]["result"]["case"]["sha256"],
                           primary_receipt_sha256=terminal["primary"]["sha256"])
        write(out / "requests" / (key + ".json"), request)
        attempt = dict(key=key, passed=False, lifecycle={}, control_input_consumption_known=False)
        attempts.append(attempt)
        local, bookkeeping_errors = {}, []
        try:
            try:
                (out / "logs").mkdir(exist_ok=True)
                with (out / "logs" / (key + ".log")).open("x", encoding="utf-8") as log:
                    code = run_owned([sys.executable, "-B", "-m", "learning.task_bc_teacher_label_equivalence_v2.experiment",
                        "worker", "--out", str(out), "--key", key], cwd=ROOT, env=env, stream=log,
                        timeout=work - time.monotonic(), phase_deadline=phase, lifecycle=attempt["lifecycle"],
                        local_lifecycle=local, memory_limit_bytes=budget["process_tree_private_commit_bytes_max"])
                life = attempt["lifecycle"]
                require(code == 0 and life["tree_closed"] and life["active_processes_after_close"] == 0
                        and life["natural_tree_exit"] and not life["termination_requested"]
                        and life["job_handle_closed"] and life["all_process_handles_closed"], "Worker natural lifecycle rejected")
                complete = read(out / "worker-complete" / (key + ".json"))
                require(complete["passed"] is True and complete["request_sha256"] == sha(out / "requests" / (key + ".json"))
                        and complete["freeze_sha256"] == state_io["freeze_sha256"], "Worker completion binding changed")
            finally:
                try:
                    write(out / "local-lifecycle" / (key + ".json"), local)
                except BaseException as error:
                    bookkeeping_errors.append(error)
                terminal[key] = _terminal(out, key)
                try:
                    inputs = read(out / "worker-inputs" / (key + ".json"))
                    attempt["control_inputs"] = inputs
                    record_control(inputs, key)
                    require(inputs["state"] == state_io and type(inputs["reads_started"]) is int
                            and type(inputs["reads_completed"]) is int and inputs["reads_started"] == inputs["reads_completed"] == 1
                            and inputs["bytes_returned"] == state_io["bytes"] and not inputs["unknown_accounting"]
                            and inputs["post_runtime_verified"] is True, "Worker control input accounting incomplete")
                    attempt["control_input_consumption_known"] = True
                except BaseException as error:
                    if "control_inputs" not in attempt:
                        control["unknown_accounting"].append(dict(component=key, quantity="worker_control_inputs", value=None))
                    attempt["control_input_error_type"] = type(error).__name__
                    bookkeeping_errors.append(error)
            require(not bookkeeping_errors and terminal[key]["known"] and terminal[key]["passed"]
                    and attempt["control_input_consumption_known"], "Unverified terminal work")
            if key == "independent-check":
                receipt = terminal[key]["result"]
                require(not (out / "check/failure.json").exists() and receipt["case_sha256"] == request["case_receipt_sha256"]
                        and receipt["primary_sha256"] == request["primary_receipt_sha256"], "Independent evidence binding failed")
            verify_runtime(out, cfg, frozen)
            _tick(phase)
            require(control["bytes_known"] + state_io["bytes"] <= budget["fixed_control_metadata_bytes_max"], "Control input byte cap")
            attempt["passed"] = True
        except BaseException as error:
            attempt["error_type"] = type(error).__name__
            raise
        finally:
            attempt["elapsed_seconds"] = time.monotonic() - phase_start
            write(out / "attempts" / (key + ".json"), attempt)

    try:
        dispatch("primary", budget["primary_phase_deadline_seconds"], budget["primary_phase_deadline_seconds"])
        dispatch("independent-check", budget["independent_phase_deadline_seconds"],
                 budget["independent_phase_deadline_seconds"] - budget["primary_phase_deadline_seconds"])
    except BaseException as error:
        original_error = error
        failures.append(dict(phase="workers", kind=type(error).__name__, retry=False))
    finally:
        try:
            try:
                final_cfg, final_frozen, final_io = _state_input(out, final_input)
                control["final_binding_read_completed"] = True
                require(final_io == state_io and final_cfg == cfg and final_frozen == frozen, "Final control state changed")
                verify_runtime(out, cfg, frozen)
            except BaseException as error:
                failures.append(dict(phase="final_binding", kind=type(error).__name__, retry=False))
            finally:
                record_control(final_input, "parent_final")
                try:
                    write(out / "parent-final-inputs.json", final_input)
                except BaseException as error:
                    failures.append(dict(phase="final_binding_receipt", kind=type(error).__name__, retry=False))
            try:
                history = _history(out)
                if not history["passed"]:
                    failures.append(dict(phase="history_stat", kind="ChangedOrUnknownHistory", retry=False))
            except BaseException as error:
                failures.append(dict(phase="history_stat_receipt", kind=type(error).__name__, retry=False))
            if time.monotonic() >= deadline:
                failures.append(dict(phase="finalization", kind="TimeoutError", retry=False))
            if control["unknown_accounting"] or control["bytes_known"] > budget["fixed_control_metadata_bytes_max"]:
                failures.append(dict(phase="control_inputs", kind="UnknownOrOverBudget", retry=False))
            passed = (not failures and len(attempts) == 2 and all(row["passed"] for row in attempts)
                      and control["completed_reads"] == 3 and control["final_binding_read_completed"]
                      and history is not None and history["passed"])
            primary_record = terminal.get("primary", {})
            primary_value = primary_record.get("result", primary_record.get("partial_receipt", {}))
            primary_ledger = primary_value.get("ledger", {}) if type(primary_value) is dict else {}
            prohibited = primary_ledger.get("prohibited", {}) if type(primary_ledger) is dict else {}
            measured_prohibited = {name: prohibited.get(name) if type(prohibited) is dict
                and type(prohibited.get(name)) is int and prohibited[name] >= 0 else None
                for name in ("model_read_attempts", "np_load_attempts", "model_forwards", "training_runs",
                             "new_games", "official_transitions")}
            history_sha = None
            if history is not None:
                try:
                    history_sha = sha(out / "history-stat-check.json")
                except BaseException as error:
                    failures.append(dict(phase="history_receipt_hash", kind=type(error).__name__, retry=False))
                    passed = False
            summary = dict(schema="teacher-label-equivalence-execution-summary-v2", passed=passed,
                accepted_jobs=sum(row["passed"] for row in attempts), attempts=len(attempts),
                terminal_consumption=terminal, control_input_consumption=control,
                history_stat_check_sha256=history_sha,
                failures=failures, retries=0, freeze_sha256=state_io["freeze_sha256"],
                elapsed_seconds=time.monotonic() - started, original_deadline_monotonic=deadline,
                original_policy_admission=False, previous_diagnostic_stage_acceptance=False,
                previous_strict_review_acceptance=False, original_equivalence_execution_passed=False,
                old_strict_result_review_passed=False, cumulative_task_scoring=dict(groups=3921, candidates=228505),
                prohibited_consumption=measured_prohibited,
                new_model_forwards=measured_prohibited["model_forwards"],
                training_runs=measured_prohibited["training_runs"], new_games=measured_prohibited["new_games"],
                official_transitions=measured_prohibited["official_transitions"],
                acceptance_requires_execution_certificate_original_deadline_and_parent_zero_exit=True)
            write(out / "summary.json", summary)
        finally:
            output_accounting = account_outputs(out, budget["all_executable_artifact_bytes_max"])
    if original_error is not None:
        raise original_error
    require(not failures and len(attempts) == 2 and all(row["passed"] for row in attempts)
            and control["completed_reads"] == 3 and control["final_binding_read_completed"], "Preserved incomplete execution")
    require(output_accounting["known"] and output_accounting["within_limit"], "Output aggregate unknown or over budget")

    def final_gate():
        artifact_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        if time.monotonic() >= deadline or artifact_bytes > budget["all_executable_artifact_bytes_max"]:
            write(out / "final-deadline-failure.json", dict(passed=False,
                artifact_bytes_before_failure_receipt=artifact_bytes, elapsed_seconds=time.monotonic() - started,
                summary_or_acceptance_alone_is_provisional=True))
            raise ValueError("Final original deadline or total artifact cap exceeded")
        require(not (out / "check/failure.json").exists(), "Checker failure dominates all completion receipts")
        return artifact_bytes

    total = final_gate()
    write(out / "execution-acceptance.json", dict(schema="teacher-label-equivalence-execution-acceptance-v2",
        passed=True, summary_sha256=sha(out / "summary.json"), freeze_sha256=state_io["freeze_sha256"],
        elapsed_seconds=time.monotonic() - started, artifact_bytes_before_acceptance=total,
        requires_parent_zero_exit_original_deadline_and_no_final_or_checker_failure=True))
    final_gate()
    return dict(passed=True, accepted_jobs=2, elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "run", "worker"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--key", choices=WORKERS)
    args = parser.parse_args()
    result = worker(args.out, args.key) if args.command == "worker" else globals()[args.command](args.out)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
