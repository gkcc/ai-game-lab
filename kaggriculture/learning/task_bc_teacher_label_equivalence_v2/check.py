"""V2 receipt-only check: three repaired source cases plus four probe cases.

The unchanged V1 source/action validators and bounded receipt reader are reused.
Its six-case verifier and limits are never called or mutated. No producer,
controller, original payload, model, or synthetic fixture is executed here.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

from learning.task_bc_teacher_label_equivalence_v1 import check as old


STAGE = "task-bc-teacher-label-equivalence-v2"
BASE_HEAD = "1bd2ca626400a597dafb4df342c8f84689af328f"
SCHEMA = "teacher-label-equivalence-independent-check-v2"
SYNTHETIC_LIMITS = dict(source_protocol_cases=3, probe_protocol_cases=4,
    synthetic_candidates_on_success=6, synthetic_raw_bytes_max=131072,
    synthetic_returned_payload_bytes_max=1024, synthetic_json_decodes=2)
CASE_NAMES = ("zero_drain", "positive_drain", "read_failure_boundary")
SUMMARY_COUNTERS = ("cases_started", "cases_completed", "cases_passed", "operation_errors",
    "assertion_errors", "planned_prefix_unknowns", "failure_unknowns", "cases_with_unknown_consumption")
SYNTHETIC_PROHIBITED = ("real_files_opened", "source_observations", "controller_calls",
                       "model_reads", "model_forwards", "training_runs", "temporary_files")
GATE_COUNTERS = ("raw_bytes_returned", "raw_bytes_hashed", "returned_payload_bytes", "json_decodes_completed")
SCOPE = old.SCOPE + " Synthetic fault unknowns are separate from production and checker unknowns."
_need, _same, _keys = old._need, old._same, old._keys


def _synthetic_case(row, fixture):
    """Check one frozen synthetic receipt without generating/reading its bytes."""
    _keys(row, ("name", "status", "passed", "consumption", "observed", "failure", "elapsed_seconds"),
          "synthetic_case_shape")
    name, c, observed, failure = row["name"], row["consumption"], row["observed"], row["failure"]
    _need(name in CASE_NAMES and row["status"] == "complete" and row["passed"] is True,
          "synthetic_case_incomplete")
    old._seconds(row["elapsed_seconds"], 220, "synthetic_case_seconds")
    _keys(c, (*old.SOURCE_COUNTERS, "status", "per_source", "index_prefix", "unknown_accounting"),
          "synthetic_consumption_shape")
    _need(all(old._integer(c[k]) for k in old.SOURCE_COUNTERS), "synthetic_counter_type")
    _need(type(c["per_source"]) is list and len(c["per_source"]) == 1 and c["index_prefix"] == [],
          "synthetic_source_inventory")
    item = c["per_source"][0]
    fault = name == "read_failure_boundary"
    item_fields = (*old.SOURCE_COUNTERS, "kind", "source_path", "status", "registered_bytes",
        "inherited_sha256", "raw_sha256", "raw_sha256_verified", "source_stat_before_verified",
        "source_stat_after_verified", "gzip_eof_verified", "internal_unreturned_payload_bytes", "prefix_stop_reason")
    _keys(item, (*item_fields, "partial_raw_sha256") if fault else item_fields, "synthetic_item_shape")
    _need(all(type(item[k]) is int and item[k] == c[k] for k in old.SOURCE_COUNTERS),
          "synthetic_item_global_counter_binding")
    total_bytes, complete_sha = fixture["compressed_bytes"], fixture["compressed_sha256"]
    _need(item["kind"] == "index" and item["source_path"] == "synthetic:" + name
          and type(item["registered_bytes"]) is int and item["registered_bytes"] == total_bytes
          and item["inherited_sha256"] == complete_sha
          and item["source_stat_before_verified"] is False and item["source_stat_after_verified"] is False
          and item["gzip_eof_verified"] is False and item["internal_unreturned_payload_bytes"] is None
          and item["prefix_stop_reason"] is None, "synthetic_not_original_file")
    observation_keys = ("raw_bytes_before_drain", "raw_calls_before_drain", "remaining_raw_before_drain",
        "raw_reads_during_gzip_and_aux_close", "raw_closed", "gzip_closed", "raw_calls", "drain_bytes",
        "raw_sha256_verified", "internal_unreturned_payload_bytes")
    _keys(observed, (*observation_keys, "buffer_closed", "bridge_closed") if name == "zero_drain"
          else observation_keys, "synthetic_observed_shape")
    for key in ("raw_bytes_before_drain", "raw_calls_before_drain", "remaining_raw_before_drain",
                "raw_reads_during_gzip_and_aux_close", "raw_calls", "drain_bytes"):
        _need(old._integer(observed[key]), "synthetic_observed_counter_type")
    _need(observed["raw_closed"] is True and observed["gzip_closed"] is True
          and observed["raw_reads_during_gzip_and_aux_close"] == 0
          and observed["internal_unreturned_payload_bytes"] is None
          and observed["raw_sha256_verified"] is (not fault)
          and observed["raw_calls"] == c["raw_read_calls_started"]
          and observed["drain_bytes"] == c["raw_drain_bytes_returned"]
          and observed["raw_bytes_before_drain"] + observed["remaining_raw_before_drain"] == total_bytes,
          "synthetic_real_close_and_drain_binding")
    _keys(failure, ("operation_error", "assertion_error", "unexpected_error", "close_errors"),
          "synthetic_failure_shape")
    _need(failure["unexpected_error"] is None and failure["close_errors"] == [],
          "unexpected_synthetic_failure")

    expected = dict.fromkeys(old.SOURCE_COUNTERS, 0)
    expected.update(sources_started=1, raw_opens_started=1, raw_opens_completed=1,
        raw_closes_started=1, raw_closes_completed=1, gzip_opens_started=1, gzip_opens_completed=1,
        gzip_closes_started=1, gzip_closes_completed=1, gzip_readline_calls_started=1)
    if fault:
        _need(c["status"] == item["status"] == "failed"
              and item["raw_sha256"] is None and item["raw_sha256_verified"] is False
              and item["partial_raw_sha256"] == hashlib.sha256(b"\x1f\x8b").hexdigest(),
              "synthetic_fault_partial_hash")
        _need(_same(failure["operation_error"], dict(type="OSError", message="synthetic_second_raw_read"))
              and _same(failure["assertion_error"],
                        dict(type="AssertionError", message="expected_complete_sha_assertion_failure")),
              "synthetic_expected_fault_boundaries")
        _need(observed["raw_bytes_before_drain"] == 2 and observed["raw_calls_before_drain"] == 2,
              "synthetic_fault_returned_prefix")
        expected.update(sources_failed=1, raw_read_calls_started=2, raw_read_calls_completed=1,
                        raw_bytes_returned=2, raw_bytes_hashed=2)
        unknown = c["unknown_accounting"]
        _keys(unknown, ("planned_prefix", "failures"), "synthetic_fault_unknown_shape")
        _need(unknown["planned_prefix"] == [] and type(unknown["failures"]) is list
              and len(unknown["failures"]) == 3, "synthetic_fault_unknown_count")
        quantities = ("raw_bytes_in_failed_read", "payload_bytes_in_failed_gzip_call",
                      "gzip_internal_unreturned_payload_bytes")
        _need(all(type(entry) is dict for entry in unknown["failures"])
              and {entry.get("quantity") for entry in unknown["failures"]} == set(quantities),
              "synthetic_fault_unknown_quantities")
        by_quantity = {entry["quantity"]: entry for entry in unknown["failures"]}
        for quantity in quantities:
            entry = by_quantity[quantity]
            fields = ("kind", "quantity", "value", "reason")
            _keys(entry, fields if quantity == quantities[-1] else (*fields, "requested_bytes"),
                  "synthetic_fault_unknown_entry_shape")
            _need(entry["kind"] == "index" and entry["value"] is None
                  and entry["reason"] == ("failed_prefix_read" if quantity == quantities[-1] else "OSError"),
                  "synthetic_fault_unknown_not_zero")
        raw_request = by_quantity[quantities[0]]["requested_bytes"]
        _need(old._integer(raw_request, 1) and raw_request <= min(total_bytes - 2, 65536),
              "synthetic_failed_raw_request_bound")
        payload_request = by_quantity[quantities[1]]["requested_bytes"]
        _need(type(payload_request) is int and payload_request == 1024 - 2 * len(b'{"ok":1}\n'),
              "synthetic_failed_payload_request_bound")
    else:
        _need(c["status"] == item["status"] == "completed" and item["raw_sha256"] == complete_sha
              and item["raw_sha256_verified"] is True and failure["operation_error"] is None
              and failure["assertion_error"] is None, "synthetic_normal_full_hash")
        planned = dict(kind="index", quantity="gzip_internal_unreturned_payload_bytes", value=None,
                       reason="intentional_prefix_without_eof_probe")
        _need(_same(c["unknown_accounting"], dict(planned_prefix=[planned], failures=[])),
              "synthetic_normal_prefix_unknown")
        expected.update(sources_completed=1, raw_bytes_returned=total_bytes, raw_bytes_hashed=total_bytes,
            raw_hashes_completed=1, gzip_readline_calls_completed=1,
            returned_payload_bytes=len(b'{"ok":1}\n'), prefix_lines_returned=1,
            json_decodes_started=1, json_decodes_completed=1,
            json_bytes_submitted=len(b'{"ok":1}\n'), json_bytes_decoded=len(b'{"ok":1}\n'),
            decoded_json_values=2, decoded_mapping_objects=1, decoded_scalar_values=1,
            decoded_mapping_keys=1, index_json_decodes_started=1, index_json_decodes_completed=1)
        if name == "zero_drain":
            _need(observed["raw_calls_before_drain"] == 1 and observed["remaining_raw_before_drain"] == 0
                  and observed["buffer_closed"] is True and observed["bridge_closed"] is True,
                  "synthetic_controlled_zero_drain")
            # Bounded EOF returns do not read physical bytes and can vary with
            # gzip/BufferedReader prefetch. Preserve, type-check and sum them.
            expected.update(raw_read_calls_started=1, raw_read_calls_completed=1,
                            raw_bound_eof_returns=c["raw_bound_eof_returns"])
        else:
            remaining = observed["remaining_raw_before_drain"]
            before_bytes, before_calls = observed["raw_bytes_before_drain"], observed["raw_calls_before_drain"]
            _need(remaining > 0 and 1 <= before_calls <= before_bytes <= 256 * before_calls,
                  "synthetic_controlled_positive_drain")
            # The explicitly frozen synthetic raw transport caps each drain
            # return at 256 bytes. This does not assume gzip's prefetch size.
            drain_calls = (remaining + 255) // 256
            expected.update(raw_read_calls_started=before_calls + drain_calls,
                raw_read_calls_completed=before_calls + drain_calls,
                raw_drain_read_calls_started=drain_calls, raw_drain_read_calls_completed=drain_calls,
                raw_drain_bytes_returned=remaining)
    _need(_same({key: c[key] for key in old.SOURCE_COUNTERS}, expected),
          "synthetic_independent_counter_expectations")
    return expected


def _protocol(tests, cfg):
    """Reconcile all synthetic counters, not the producer's four-field gate."""
    _keys(tests, ("schema", "passed", "cases", "synthetic_candidates", "cases_known_complete", "source", "probe"),
          "v2_protocol_shape")
    _need(tests["schema"] == "teacher-label-equivalence-protocol-v2" and tests["passed"] is True
          and type(tests["cases"]) is int and tests["cases"] == 7
          and type(tests["cases_known_complete"]) is int and tests["cases_known_complete"] == 7
          and type(tests["synthetic_candidates"]) is int and tests["synthetic_candidates"] == 6,
          "v2_protocol_actual_seven_cases")
    _need(_same(cfg["new_synthetic_limits"], SYNTHETIC_LIMITS), "registered_synthetic_limits_changed")
    source, probe = tests["source"], tests["probe"]
    _keys(source, ("started", "completed", "result", "ledger", "gate_receipt"), "source_protocol_boundary")
    _keys(probe, ("started", "completed", "result", "partial_consumption_unknown"), "probe_protocol_boundary")
    _need(source["started"] is True and source["completed"] is True
          and probe["started"] is True and probe["completed"] is True
          and probe["partial_consumption_unknown"] is False, "completed_protocol_boundary")
    _need(_same(source["result"], dict(schema="teacher-label-protocol-repair-v1", passed=True, scenario_count=3)),
          "source_protocol_original_return")
    ledger = source["ledger"]
    _keys(ledger, ("schema", "status", "passed", "cases", "totals", "counter_scope", "prohibited", "fixture",
                   "elapsed_seconds"), "source_protocol_ledger_shape")
    _need(ledger["schema"] == "teacher-label-protocol-repair-v1" and ledger["status"] == "complete"
          and ledger["passed"] is True and type(ledger["counter_scope"]) is str and bool(ledger["counter_scope"]),
          "source_protocol_ledger_completion")
    old._seconds(ledger["elapsed_seconds"], 220, "source_protocol_seconds")
    _keys(ledger["prohibited"], SYNTHETIC_PROHIBITED, "synthetic_prohibited_shape")
    _need(all(type(v) is int and v == 0 for v in ledger["prohibited"].values()), "synthetic_prohibited_work")
    fixture = ledger["fixture"]
    _keys(fixture, ("compressed_bytes", "compressed_sha256", "generated_payload_bytes"), "synthetic_fixture_identity")
    _need(old._integer(fixture["compressed_bytes"]) and 512 < fixture["compressed_bytes"] < 65536
          and old._sha(fixture["compressed_sha256"]) and type(fixture["generated_payload_bytes"]) is int
          and fixture["generated_payload_bytes"] == len(b'{"ok":1}\n') + 512 * 64 + 1,
          "synthetic_fixture_bounds")
    cases = ledger["cases"]
    _need(type(cases) is list and [row["name"] for row in cases] == list(CASE_NAMES), "synthetic_case_order")
    rows = [_synthetic_case(row, fixture) for row in cases]
    _need(sum(row["elapsed_seconds"] for row in cases) <= ledger["elapsed_seconds"],
          "synthetic_elapsed_sum")
    totals = {key: sum(row[key] for row in rows) for key in old.SOURCE_COUNTERS}
    totals.update(cases_started=len(cases), cases_completed=sum(row["status"] == "complete" for row in cases),
        cases_passed=sum(row["passed"] is True for row in cases),
        operation_errors=sum(row["failure"]["operation_error"] is not None for row in cases),
        assertion_errors=sum(row["failure"]["assertion_error"] is not None for row in cases),
        planned_prefix_unknowns=sum(len(row["consumption"]["unknown_accounting"]["planned_prefix"]) for row in cases),
        failure_unknowns=sum(len(row["consumption"]["unknown_accounting"]["failures"]) for row in cases),
        cases_with_unknown_consumption=sum(row["consumption"] is None for row in cases))
    _keys(ledger["totals"], (*old.SOURCE_COUNTERS, *SUMMARY_COUNTERS), "synthetic_totals_shape")
    _need(_same(ledger["totals"], totals), "all_synthetic_counters_recomputed")
    _need(totals["raw_bytes_returned"] == totals["raw_bytes_hashed"] == 2 * fixture["compressed_bytes"] + 2
          <= SYNTHETIC_LIMITS["synthetic_raw_bytes_max"]
          and totals["returned_payload_bytes"] <= SYNTHETIC_LIMITS["synthetic_returned_payload_bytes_max"]
          and totals["json_decodes_completed"] == SYNTHETIC_LIMITS["synthetic_json_decodes"]
          and totals["planned_prefix_unknowns"] == 2 and totals["failure_unknowns"] == 3
          and totals["cases_with_unknown_consumption"] == 0, "synthetic_aggregate_budget")
    gate = source["gate_receipt"]
    _keys(gate, ("passed", "cases", "observed_totals", "independent_original_reproduction", "scope"),
          "source_gate_receipt_shape")
    _need(gate["passed"] is True and type(gate["cases"]) is int and gate["cases"] == 3
          and gate["independent_original_reproduction"] is False and type(gate["scope"]) is str
          and _same(gate["observed_totals"], {key: totals[key] for key in GATE_COUNTERS}),
          "source_gate_crosscheck_only")
    _need(_same(probe["result"], dict(cases=4, synthetic_candidates=6, real_controller_calls=0,
        model_calls=0, official_transitions=0, temporary_files=0, passed=True)), "original_four_probe_cases")
    return dict(cases=7, source_cases=3, probe_cases=4, synthetic_candidates=6,
        synthetic_totals=totals, expected_synthetic_failure_unknowns=3, unexpected_synthetic_failures=0,
        scope="Synthetic raw_* are in-memory operations; they are not original-file reads or production failures.")


def _verify(case, primary, cfg):
    fixed, limits = cfg["fixed_case"], cfg["budget"]
    expected_limits = dict(old.LIMITS, tiny_in_memory_cases_max=7)
    expected_limits.update(crop_encodings_max=1040, crop_callbacks_per_adapter_call_max=16,
        per_crop_vector_values_max=1024, crop_feature_values_max=1064960,
        total_executable_seconds_max=300, primary_phase_deadline_seconds=220,
        independent_phase_deadline_seconds=270, final_deadline_seconds=300,
        all_executable_artifact_bytes_max=4194304, fixed_control_metadata_bytes_max=262144)
    _need(cfg["schema"] == STAGE and cfg["baseline_head"] == BASE_HEAD
          and _same({key: fixed[key] for key in old.IDENTITY}, old.IDENTITY), "v2_registration_identity")
    _need(all(type(limits[key]) is int and limits[key] == value for key, value in expected_limits.items()),
          "v2_registered_limits_changed")
    public_fixed = {key: old.IDENTITY[key] for key in ("game", "side", "step", "callback_index_zero_based")}
    _need(case["schema"] == "teacher-label-equivalence-case-v1"
          and primary["schema"] == "teacher-label-equivalence-primary-v2"
          and primary["status"] == "complete" and primary["phase"] == "complete"
          and primary["failure_type"] is None and _same(case["fixed"], public_fixed)
          and _same(primary["fixed"], public_fixed), "v2_complete_receipt_identity")
    case_keys = ["schema", "fixed", "available", "reason", "source_certificate", "training_binding",
                 "first_candidates", "trials", "outcome", "comparison"]
    _keys(case, case_keys + (["stored_action", "original_teacher", "baseline"] if case["available"] else []),
          "private_case_schema")
    old._seconds(primary["elapsed_seconds"], 220, "primary_seconds")
    protocol = _protocol(primary["protocol_checks"], cfg)
    ledger = primary["ledger"]
    _keys(ledger, ("controller", "prohibited", "unknown_accounting", "source", "patches_restored"),
          "primary_ledger_shape")
    _need(ledger["patches_restored"] is True and ledger["unknown_accounting"] == [], "primary_failure_or_patch_state")
    _keys(ledger["prohibited"], ("model_read_attempts", "np_load_attempts", "model_forwards",
                               "training_runs", "new_games", "official_transitions"), "prohibited_counter_shape")
    _need(all(type(v) is int and v == 0 for v in ledger["prohibited"].values()), "prohibited_work")
    source = old._source(case, ledger["source"], cfg)
    totals, outcome, comparison = old._calls(case, cfg)
    _need(case["outcome"] == primary["outcome"] == outcome
          and _same(case["comparison"], comparison) and _same(primary["comparison"], comparison),
          "independent_outcome_or_comparison_mismatch")
    counts = ledger["controller"]
    _keys(counts, [name + suffix for name in old.CONTROLLER_FIELDS for suffix in ("_started", "_completed")]
          + ["forced_choices"], "controller_counter_shape")
    old._paired_counts(counts, old.CONTROLLER_FIELDS)
    _need(all(counts[name + "_completed"] == totals[name] for name in old.CONTROLLER_FIELDS)
          and type(counts["forced_choices"]) is int and counts["forced_choices"] == totals["forced_choices"],
          "controller_event_ledger_binding")
    return dict(fixed=public_fixed, outcome=outcome, comparison=comparison, controller=totals,
        original_compressed_bytes=source["raw_bytes_returned"], original_payload_bytes=source["returned_payload_bytes"],
        planned_prefix_unknowns=len(source["unknown_accounting"]["planned_prefix"]),
        production_failure_unknowns=0, source_target_row_index=case["source_certificate"]["target_row_index"],
        protocol=protocol, scope=SCOPE)


def check(out: Path, cfg: dict, deadline: float) -> dict:
    """Read the two parent-bound receipts once; failure dominates completion."""
    started = time.monotonic()
    _need(type(deadline) in (int, float) and math.isfinite(deadline), "invalid_checker_deadline")
    reader = old._Reader(deadline, cfg["budget"])
    directory = Path(out) / "check"
    directory.mkdir(exist_ok=False)
    certificate = dict(schema=SCHEMA, status="failed", passed=False,
        case_sha256=cfg.get("case_receipt_sha256"), primary_sha256=cfg.get("primary_receipt_sha256"),
        consumption=reader.counts, partial_unknown=reader.unknown, safe_result=None, scope=SCOPE)
    try:
        old._write(directory / "started.json", dict(schema=SCHEMA, status="started",
            case_sha256=certificate["case_sha256"], primary_sha256=certificate["primary_sha256"]), reader)
        case, case_bytes = reader.read(Path(out) / "case.json", certificate["case_sha256"], 1048576)
        primary, _ = reader.read(Path(out) / "primary-receipt.json", certificate["primary_sha256"], 2097152)
        _keys(primary["case"], ("path", "sha256", "bytes"), "primary_case_manifest_shape")
        _need(primary["case"]["path"] == "case.json" and primary["case"]["sha256"] == certificate["case_sha256"]
              and type(primary["case"]["bytes"]) is int and primary["case"]["bytes"] == case_bytes,
              "primary_case_manifest_binding")
        safe = _verify(case, primary, cfg)
        reader.tick()
        _need(all(reader.counts[name + suffix] == 2 for name in ("opens", "reads", "json_decodes", "closes")
                  for suffix in ("_started", "_completed"))
              and reader.counts["bytes_returned"] == reader.counts["bytes_hashed"]
                  == reader.counts["json_bytes_submitted"] == reader.counts["json_bytes_decoded"]
              and reader.unknown == [], "independent_reader_accounting")
        certificate.update(status="complete", passed=True, safe_result=safe,
                           elapsed_seconds=time.monotonic() - started)
        old._write(directory / "safe-summary.json", certificate, reader)
        old._write(directory / "complete.json", certificate, reader)
        reader.tick()
        return certificate
    except BaseException as error:
        certificate.update(status="failed", passed=False, safe_result=None,
            failure_code=str(error) if isinstance(error, old.CheckFailure) else type(error).__name__,
            elapsed_seconds=time.monotonic() - started)
        try:
            old._write(directory / "failure.json", certificate, reader, enforce_deadline=False)
        finally:
            if not (directory / "safe-summary.json").exists():
                old._write(directory / "safe-summary.json", certificate, reader, enforce_deadline=False)
        raise
