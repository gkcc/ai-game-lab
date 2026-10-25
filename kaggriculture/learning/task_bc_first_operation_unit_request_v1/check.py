"""Independent single-pass check of one selected operation intervention.

Only the three newly bound receipts are read.  The reused helpers are the old
independently authored JSON/type/event comparators, not producer code.  A valid
return also requires no check/failure.json: a late deadline failure is retained.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time

from learning.task_bc_first_water_unit_requests_v1 import check as previous

CheckFailure = previous.CheckFailure
_need, _keys, _same = previous._need, previous._keys, previous._same
_integer, _sha, _seconds = previous._integer, previous._sha, previous._seconds
OPS, REQUEST_OPS = previous.OPS, previous.REQUEST_OPS
GAME = "holdout-1714414373-v1-0-trained"
MAX_BYTES = 524288
LIMIT = ("One consumed historical scene and newly bound receipts only; no second "
         "original-source/model/controller reproduction, official execution, economic "
         "effect, generalization or policy-admission evidence.")


def _configuration(cfg):
    scope = cfg["scope_budget"]
    fixed = cfg["fixed_selection"]
    _need(fixed["compact_sha256"] ==
          "e5218e62f5d09db7acbbd6feeab6bbe3b7169011957254399ea18b6fa0b339da"
          and fixed["compact_bytes_exact"] == 923156
          and fixed["compact_total_rows_inherited"] == 2053
          and fixed["early_callback_rows_inherited"] == 691,
          "fixed_selection_identity")
    _need(_sha(cfg["selection_receipt_sha256"]) and _sha(cfg["primary_receipt_sha256"]),
          "receipt_bindings")
    for key, limit in {"total_executable_seconds_max": 120,
                       "selection_phase_seconds_max_including_close": 15,
                       "primary_phase_seconds_max_including_close": 65,
                       "independent_checker_seconds_max_including_close": 20,
                       "per_controller_call_seconds_max": 10,
                       "model_load_phase_seconds_max": 20,
                       "training_runs_max": 0, "new_games_max": 0,
                       "official_transitions_max": 0, "retries_max": 0}.items():
        _need(_same(scope[key], limit), "fixed_stage_budget")
    _need(_same(scope["outputs"]["machine_checker_aggregate_receipt_bytes_max"], MAX_BYTES)
          and _same(scope["outputs"]["machine_checker_passes_per_receipt_max"], 1),
          "fixed_checker_budget")
    _keys(cfg["models"], {"task", "crop"}, "model_inventory")
    for kind in ("task", "crop"):
        _need(cfg["models"][kind] == cfg["fixed_input_scope"][kind + "_model"]
              and cfg["models"][kind]["sha256"] == previous.MODEL_SHA[kind],
              "fixed_model_identity")


def _descriptor(value):
    _keys(value, {"id", "unit", "target", "op"}, "selected_descriptor_keys")
    unit, target, op = value["unit"], value["target"], value["op"]
    _need(_integer(unit) and unit < 16 and type(target) is list and len(target) == 2
          and all(_integer(part) for part in target) and op in OPS,
          "selected_descriptor_shape")
    _need(value["id"] == f"u{unit}:x{target[0]}:y{target[1]}:{op}", "selected_descriptor_id")
    return unit, target[0], target[1], op


def _selection(value, cfg, receipt_bytes):
    _need(type(value) is dict and value["schema"] == "first-operation-selection-v1"
          and value["status"] == "complete" and value["unknown_accounting"] == []
          and value["failure_code"] is None and value["phase"] == "complete",
          "selection_completion")
    fixed, chosen = cfg["fixed_selection"], value["selected"]
    _keys(chosen, {"game", "side", "step", "target_index", "source_line_sha256", "teacher", "chosen"},
          "selection_target_keys")
    _need(chosen["game"] == GAME and _same(chosen["side"], 0)
          and _integer(chosen["step"]) and chosen["step"] < 240
          and _integer(chosen["target_index"]) and chosen["target_index"] < 16
          and _sha(chosen["source_line_sha256"]), "selection_target_identity")
    teacher, student = _descriptor(chosen["teacher"]), _descriptor(chosen["chosen"])
    _need(teacher[3] != student[3], "selected_operation_mismatch")
    _need(value["projection_receipt_sha256"] == fixed["projection_receipt_sha256"]
          and value["compact_sha256"] == fixed["compact_sha256"], "selection_input_bindings")
    _seconds(value["elapsed_seconds"], 15, "selection_time_limit")
    turns, prefix = value["projection_prefix_turns"], value["prefix"]
    _need(type(turns) is list and len(turns) == chosen["step"] + 1
          and type(prefix) is list and 1 <= len(prefix) <= 691, "selection_prefix_sizes")
    expected = []
    for step, turn in enumerate(turns):
        _need(type(turn) is list and len(turn) == 3 and _same(turn[0], step)
              and _integer(turn[1]) and _sha(turn[2]),
              "projection_prefix_turn")
        end = turn[1] if step < chosen["step"] else chosen["target_index"] + 1
        _need(end <= turn[1] and len(expected) + end <= 691, "selected_index_or_prefix_limit")
        expected.extend((step, index, turn[2]) for index in range(end))
    _need(len(expected) == len(prefix) and turns[-1][2] == chosen["source_line_sha256"],
          "selection_prefix_complete")
    for index, (row, identity) in enumerate(zip(prefix, expected)):
        _need(type(row) is list and len(row) == 5
              and all(_same(row[i], identity[i]) for i in range(3))
              and row[3] in OPS and row[4] in OPS, "selection_prefix_row")
        _need((row[3] != row[4]) == (index == len(prefix) - 1),
              "first_operation_mismatch")
    _need(prefix[-1] == [chosen["step"], chosen["target_index"], chosen["source_line_sha256"],
                         teacher[3], student[3]], "selected_prefix_identity")
    safe = {key: chosen[key] for key in ("game", "side", "step", "target_index", "source_line_sha256")}
    safe.update(teacher_op=teacher[3], chosen_op=student[3])
    status = cfg["selection_status"]
    _need(status["schema"] == "first-operation-selection-status-v1"
          and status["status"] == "complete" and status["unknown_accounting"] == []
          and status["failure_code"] is None and status["phase"] == "complete", "safe_selection_status")
    _seconds(status["elapsed_seconds"], 15, "safe_selection_time_limit")
    _keys(status["selected"], safe, "safe_selection_keys")
    _need(all(_same(status["selected"][k], v) for k, v in safe.items())
          and status["private_receipt"] == {"path": "selection.json",
              "sha256": cfg["selection_receipt_sha256"], "bytes": receipt_bytes}
          and _same(status["consumption"], value["consumption"])
          and status["elapsed_seconds"] >= value["elapsed_seconds"],
          "safe_private_selection_binding")
    expected_verified = dict(projection_stat_after_verified=True, compact_stat_after_verified=True,
        compact_full_hash_verified=True, compact_exact_registered_length_verified=True, compact_eof_probed=False)
    for receipt in (value, status):
        _keys(receipt["verification"], expected_verified, "selection_verification_keys")
        _need(all(_same(receipt["verification"][k], v) for k, v in expected_verified.items()),
              "selection_verification_status")
    _selection_consumption(value["consumption"], len(prefix), chosen, cfg)
    return chosen, safe


def _selection_consumption(counts, prefix_rows, selected, cfg):
    """The selector's explicit counter envelope is reconciled here."""
    pairs = ("projection_opens", "projection_read_calls", "projection_closes",
             "compact_opens", "compact_read_calls", "compact_closes")
    keys = {name + suffix for name in pairs for suffix in ("_started", "_completed")}
    keys |= {"projection_bytes_returned", "projection_json_decode_attempts", "projection_json_decode_completed",
        "projection_json_bytes_submitted", "projection_json_bytes_decoded", "compact_bytes_returned",
        "compact_bytes_hashed", "compact_physical_lines_hashed", "compact_prefix_lines_returned",
        "compact_json_decode_attempts", "compact_json_decode_completed", "compact_json_bytes_submitted",
        "compact_json_bytes_decoded", "compact_teacher_descriptors_decoded", "compact_chosen_descriptors_decoded",
        "compact_unit_requests_decoded", "compact_suffix_json_decodes", "raw_trace_reads", "replay_reads",
        "model_reads", "controller_calls", "old_witness_or_pair_reads"}
    _keys(counts, keys, "selection_count_keys")
    _need(all(_integer(counts[k]) for k in keys), "selection_counter_types")
    expected = dict(compact_bytes_returned=923156, compact_bytes_hashed=923156,
        compact_physical_lines_hashed=2053, projection_json_decode_attempts=1, projection_json_decode_completed=1,
        compact_prefix_lines_returned=prefix_rows, compact_json_decode_attempts=prefix_rows,
        compact_json_decode_completed=prefix_rows, compact_teacher_descriptors_decoded=prefix_rows,
        compact_chosen_descriptors_decoded=prefix_rows, compact_unit_requests_decoded=prefix_rows,
        compact_suffix_json_decodes=0, raw_trace_reads=0, replay_reads=0, model_reads=0,
        controller_calls=0, old_witness_or_pair_reads=0)
    _need(all(_same(counts.get(key), val) for key, val in expected.items()),
          "selection_compact_full_hash_consumption")
    for name in pairs:
        _need(0 < counts[name + "_started"] == counts[name + "_completed"], "selection_started_completed")
        if name != "compact_read_calls":
            _need(counts[name + "_started"] == 1, "selection_single_open_read_close")
    _need(0 < counts["projection_bytes_returned"] == cfg["projection_stat"]["size"] <= 131072
          and counts["projection_json_bytes_submitted"] == counts["projection_json_bytes_decoded"]
              == counts["projection_bytes_returned"]
          and 0 < counts["compact_json_bytes_submitted"] == counts["compact_json_bytes_decoded"] <= 923156,
          "selection_json_byte_accounting")


def _vector(value):
    _need(type(value) is list and 1 <= len(value) <= 16, "unit_vector_size")
    for request in value:
        _need(type(request) is list and request and type(request[0]) is str
              and request[0] in REQUEST_OPS, "unit_request_operation")
    previous._typed(value)
    return value


def _call(value, mode, call_index, units, target):
    _keys(value, {"status", "unit_actions", "returned_unit_actions", "trace_sha256", "events",
                  "crop_events", "counts", "active_seconds", "override_count", "call_index"}, "call_keys")
    _need(value["status"] == "complete" and _same(value["call_index"], call_index)
          and _sha(value["trace_sha256"]), "call_completion_order")
    _seconds(value["active_seconds"], 10, "call_active_seconds_limit")
    trace, returned = _vector(value["unit_actions"]), _vector(value["returned_unit_actions"])
    _need(len(trace) == len(returned) == units and _same(trace, returned), "returned_unit_vector_identity")
    events, crops = value["events"], value["crop_events"]
    _need(type(events) is list and target < len(events) <= 16
          and type(crops) is list and len(crops) <= 16, "call_event_limits")
    maps, assigned = [], set()
    for index, event in enumerate(events):
        identities = previous._event(event, index, units)
        forced = mode == "intervention" and index == target
        _need(event["selector"] == ("override" if forced else "student"), "unique_override_position")
        _need(not any(item[0] in assigned for item in identities.values()), "assigned_unit_reused")
        assigned.add(identities[event["chosen_id"]][0])
        maps.append(identities)
    forced_count = int(mode == "intervention")
    _need(_same(value["override_count"], forced_count), "unique_override_count")
    for index, event in enumerate(crops):
        _keys(event, {"index", "sequence", "logits_values", "forward_calls", "record_sha256", "status",
                      "forward_calls_completed", "logits_values_completed"}, "crop_event_keys")
        _need(event["status"] == "complete" and _same(event["index"], index)
              and _integer(event["sequence"], 1) and _sha(event["record_sha256"])
              and all(_same(event[key], val) for key, val in {"forward_calls": 1,
                  "forward_calls_completed": 1, "logits_values": 5, "logits_values_completed": 5}.items()),
              "crop_event_scoring")
    model = dict(task_forward_calls=len(events) - forced_count,
                 task_scored_candidates=sum(e["task_scored_candidates"] for e in events),
                 crop_forward_calls=len(crops), crop_logits_values=5 * len(crops))
    model.update({key + "_completed": val for key, val in list(model.items())})
    controller = dict(real_calls=1, baseline_calls=int(mode == "baseline"),
        intervention_calls=int(mode == "intervention"), callbacks=len(events), task_candidate_encodings=len(events))
    controller.update({key + "_completed": val for key, val in list(controller.items())})
    controller.update(forced_choices=forced_count,
                      generated_candidate_objects=sum(e["candidate_count"] for e in events))
    counts = dict(models=model, controller=controller)
    _keys(value["counts"], counts, "call_count_groups")
    for group, expected in counts.items():
        _keys(value["counts"][group], expected, "call_count_keys")
        _need(all(_same(value["counts"][group][key], val) for key, val in expected.items()),
              "call_count_reconciliation")
    return maps, counts


def _pair(value, selected):
    _keys(value, {"schema", "status", "sample", "historical", "baseline", "intervention", "comparison"},
          "pair_keys")
    _need(value["schema"] == "first-operation-unit-request-pair-v1" and value["status"] == "complete",
          "pair_schema")
    sample = {key: selected[key] for key in ("game", "side", "step", "target_index", "source_line_sha256")}
    _keys(value["sample"], sample, "pair_sample_keys")
    _need(all(_same(value["sample"][k], v) for k, v in sample.items()), "pair_sample_binding")
    historical, baseline, intervention = (value[key] for key in ("historical", "baseline", "intervention"))
    _keys(historical, {"unit_actions", "trace_sha256", "events", "crop_events"}, "historical_keys")
    units, target = len(_vector(historical["unit_actions"])), selected["target_index"]
    bm, bc = _call(baseline, "baseline", 0, units, target)
    im, ic = _call(intervention, "intervention", 1, units, target)
    _seconds(baseline["active_seconds"] + intervention["active_seconds"], 20, "pair_active_limit")
    _need(_same(historical["unit_actions"], baseline["unit_actions"])
          and historical["trace_sha256"] == baseline["trace_sha256"], "historical_baseline_identity")
    for key, expected_keys in (("events", {"index", "inputs_sha256", "record_sha256", "chosen_id", "teacher_id"}),
                               ("crop_events", {"index", "record_sha256"})):
        saved = historical[key]
        _need(type(saved) is list and len(saved) == len(baseline[key]), "historical_event_count")
        for index, (old, actual) in enumerate(zip(saved, baseline[key])):
            _keys(old, expected_keys, "historical_event_keys")
            _need(_same(old["index"], index) and all(_same(old[k], actual[k]) for k in old),
                  "historical_event_identity")
    # Earlier candidate IDs need not agree with the conditional teacher.  The
    # frozen selection rule excludes only an earlier *operation* disagreement.
    for index in range(target):
        event = baseline["events"][index]
        _need(bm[index][event["chosen_id"]][3] == bm[index][event["teacher_id"]][3],
              "earlier_operation_mismatch")
    for index in range(target + 1):
        left, right = baseline["events"][index], intervention["events"][index]
        for key in ("inputs_sha256", "context_sha256", "candidate_ids_sha256", "teacher_id", "legal_candidates"):
            _need(_same(left[key], right[key]), "factual_prefix_context_identity")
        if index < target:
            _need(left["chosen_id"] == right["chosen_id"] and left["record_sha256"] == right["record_sha256"],
                  "factual_prefix_choice_identity")
    old, new = baseline["events"][target], intervention["events"][target]
    _need(new["chosen_id"] == new["teacher_id"] == old["teacher_id"]
          and new["chosen_id"] != old["chosen_id"], "current_teacher_override")
    _need(bm[target][old["chosen_id"]] == _descriptor(selected["chosen"])
          and im[target][new["chosen_id"]] == _descriptor(selected["teacher"])
          and bm[target][old["chosen_id"]][3] != im[target][new["chosen_id"]][3],
          "bound_operation_choice")
    calls = (baseline, intervention)
    events = [event for call in calls for event in call["events"]]
    for key in ("callback_sequence", "encoding_sequence"):
        _need([e[key] for e in events] == list(range(1, len(events) + 1)), "fresh_callback_sequence")
    scores = [e["score_sequence"] for e in events if e["selector"] == "student"]
    crops = [e["sequence"] for call in calls for e in call["crop_events"]]
    _need(scores == list(range(1, len(scores) + 1)) and crops == list(range(1, len(crops) + 1)),
          "fresh_model_score_sequences")
    left, right = baseline["unit_actions"], intervention["unit_actions"]
    changed = [i for i in range(units) if not _same(left[i], right[i])]
    expected = dict(historical_baseline_equal=True, baseline_intervention_equal=not changed,
        changed_unit_indices=changed, changed_unit_count=len(changed),
        vector_lengths=dict(historical=units, baseline=units, intervention=units),
        first_difference=previous._difference(left, right), baseline_returned_equal=True,
        intervention_returned_equal=True)
    _keys(value["comparison"], expected, "comparison_keys")
    for key, item in expected.items():
        _need(_same(value["comparison"][key], item), "independent_comparison_identity")
    operations = {mode: dict.fromkeys(REQUEST_OPS, 0) for mode in ("baseline", "intervention")}
    for index in changed:
        operations["baseline"][left[index][0]] += 1
        operations["intervention"][right[index][0]] += 1
    safe = dict(game=GAME, step=selected["step"], target_index=target, units_per_vector=units,
        teacher_op=selected["teacher"]["op"], chosen_op=selected["chosen"]["op"],
        historical_baseline_equal=True, baseline_returned_equal=True, intervention_returned_equal=True,
        unit_vectors_equal=not changed, changed_unit_count=len(changed),
        changed_request_operation_counts=operations, baseline_callback_count=len(baseline["events"]),
        intervention_callback_count=len(intervention["events"]),
        resumed_student_callback_count=len(intervention["events"]) - target - 1, override_count=1)
    return safe, calls, (bc, ic)


def _primary(value, size):
    _keys(value, {"schema", "status", "baseline_gate_passed", "completed_baselines", "completed_interventions",
                  "pair", "protocol_checks", "ledger", "timing", "phase", "failure_code", "safe_result", "output"},
          "primary_keys")
    _need(value["schema"] == "first-operation-unit-request-primary-v1" and value["status"] == "complete"
          and value["baseline_gate_passed"] is True and _same(value["completed_baselines"], 1)
          and _same(value["completed_interventions"], 1) and value["phase"] == "completed"
          and value["failure_code"] is None, "primary_completion")
    checks = value["protocol_checks"]
    _keys(checks, {"cases", "fake_callback_events", "fake_candidates", "real_model_calls",
                   "real_controller_calls", "official_transitions", "passed"}, "protocol_keys")
    _need(checks["passed"] is True and _integer(checks["cases"], 1) and checks["cases"] <= 5
          and _integer(checks["fake_callback_events"]) and checks["fake_callback_events"] <= 20
          and _integer(checks["fake_candidates"]) and checks["fake_candidates"] <= 40
          and all(_same(checks[key], 0) for key in ("real_model_calls", "real_controller_calls", "official_transitions")),
          "protocol_limits")
    timing = value["timing"]
    _keys(timing, {"primary_started_perf", "deadline_perf", "elapsed_seconds", "model_load_seconds"}, "timing_keys")
    _seconds(timing["elapsed_seconds"], 65, "primary_time_limit")
    _seconds(timing["model_load_seconds"], 20, "model_load_time_limit")
    _need(all(type(timing[k]) in (int, float) and math.isfinite(timing[k])
              for k in ("primary_started_perf", "deadline_perf"))
          and 0 < timing["deadline_perf"] - timing["primary_started_perf"] <= 65, "primary_deadline_interval")
    binding = value["pair"]
    _keys(binding, {"path", "sha256", "bytes"}, "pair_binding_keys")
    _need(binding["path"] == "pair.json" and _sha(binding["sha256"])
          and _integer(binding["bytes"], 1) and binding["bytes"] <= 262144, "pair_binding")
    _keys(value["output"], {"pair_files_written", "pair_bytes_written", "primary_receipt_bytes"}, "primary_output_keys")
    _need(_same(value["output"]["pair_files_written"], 1)
          and _same(value["output"]["pair_bytes_written"], binding["bytes"])
          and _same(value["output"]["primary_receipt_bytes"], size), "primary_output_reconciliation")
    return binding


def _ledger(value, cells, baseline, selected, cfg):
    _keys(value, {"source", "model_io", "models", "controller", "unknown_accounting"}, "ledger_keys")
    totals = {group: {key: sum(cell[group][key] for cell in cells) for key in cells[0][group]}
              for group in ("models", "controller")}
    for group, extras in {"models": dict(training_calls=0, numeric_cache_reuse=False),
                          "controller": dict(full_teacher_policy_calls=0, official_transitions=0, new_games=0)}.items():
        expected = {**totals[group], **extras}
        _keys(value[group], expected, "ledger_execution_keys")
        _need(all(_same(value[group][key], count) for key, count in expected.items()), "ledger_event_totals")
    for key, cap in dict(task_forward_calls=31, task_scored_candidates=15872,
                         crop_forward_calls=32, crop_logits_values=160).items():
        _need(totals["models"][key] <= cap, "model_budget")
    for key, cap in dict(real_calls=2, baseline_calls=1, intervention_calls=1, forced_choices=1,
                         callbacks=32, task_candidate_encodings=32, generated_candidate_objects=16384).items():
        _need(totals["controller"][key] <= cap, "controller_budget")
    unknown = value["unknown_accounting"]
    _keys(unknown, {"planned_prefix", "failures"}, "unknown_groups")
    expected = dict(component="source", game=GAME, quantity="gzip_internal_unreturned_payload_bytes",
                    value=None, reason="intentional_selected_prefix_without_eof_probe")
    _need(unknown["failures"] == [] and type(unknown["planned_prefix"]) is list
          and len(unknown["planned_prefix"]) == 1, "unknown_completion")
    _keys(unknown["planned_prefix"][0], expected, "planned_unknown_keys")
    _need(all(_same(unknown["planned_prefix"][0][k], v) for k, v in expected.items()), "planned_unknown_identity")
    source = _source_ledger(value["source"], baseline, selected, cfg)
    model_io = _model_ledger(value["model_io"], cfg)
    return dict(**totals, source=source, model_io=model_io, intentional_prefix_unknown_count=1)


def _source_ledger(source, baseline, selected, cfg):
    zeros = ("skipped_prefix_lines_json_decoded", "reread_passes", "complete_trace_hash_passes",
             "raw_replay_reads", "existing_compact_or_first_witness_reads")
    lines, events, crops = selected["step"] + 1, baseline["events"], baseline["crop_events"]
    candidates = sum(e["candidate_count"] for e in events)
    exact = dict.fromkeys(zeros, 0)
    exact.update(gzip_opens=1, prefix_lines_returned=lines, target_json_decode_attempts=1,
        target_json_turns_decoded=1, source_opens_started=1, source_opens_completed=1,
        readline_calls_started=lines, readline_calls_completed=lines, source_closes_started=1,
        source_closes_completed=1, gzip_closes_started=1, gzip_closes_completed=1, sources_completed=1,
        target_task_records=len(events), target_candidate_objects=candidates, target_candidate_x_arrays=candidates,
        target_shared_x_arrays=len(events), target_observation_objects=len(events), target_configuration_objects=len(events),
        target_crop_records=len(crops), target_crop_x_arrays=len(crops), target_crop_logits_arrays=len(crops))
    counts = set(exact) | {"compressed_bytes_read", "returned_payload_bytes", "target_json_bytes",
                           "compressed_read_calls_started", "compressed_read_calls_completed"}
    _keys(source, counts | {"status", "per_source", "whole_trace_hash_proof", "elapsed_seconds"}, "source_keys")
    _need(source["status"] == "completed" and source["whole_trace_hash_proof"] == "inherited_only"
          and type(source["per_source"]) is list and len(source["per_source"]) == 1, "source_completion")
    _seconds(source["elapsed_seconds"], 65, "source_time_limit")
    item = source["per_source"][0]
    metadata = {"game", "source_path", "status", "gzip_eof_verified", "internal_unreturned_payload_bytes",
                "inherited_whole_trace_sha256", "whole_trace_hash_proof", "source_stat_after_verified", "target_line_sha256"}
    _keys(item, counts | metadata, "source_item_keys")
    _need(all(_integer(source[k]) and _same(source[k], item[k]) for k in counts)
          and all(_same(item[k], v) for k, v in exact.items()), "source_counts_reconcile")
    fixed = cfg["fixed_input_scope"]
    _need(item["game"] == GAME and item["source_path"] == fixed["trace"]
          and item["target_line_sha256"] == selected["source_line_sha256"]
          and item["inherited_whole_trace_sha256"] == fixed["inherited_trace_sha256"]
          and item["whole_trace_hash_proof"] == "inherited_only" and item["status"] == "completed"
          and item["gzip_eof_verified"] is False and item["internal_unreturned_payload_bytes"] is None
          and item["source_stat_after_verified"] is True, "source_identity")
    _need(0 < item["compressed_bytes_read"] <= 6825738
          and 0 < item["target_json_bytes"] <= 4194304
          and item["target_json_bytes"] <= item["returned_payload_bytes"] <= 67108864
          and 0 < item["compressed_read_calls_started"] == item["compressed_read_calls_completed"]
          and len(events) <= 16 and candidates <= 8192 and len(crops) <= 16, "source_budget")
    return {key: source[key] for key in sorted(counts)}


def _model_ledger(value, cfg):
    once = ("checkpoint_hash_passes_started", "checkpoint_hash_passes_completed", "model_object_loads_reserved",
            "model_object_loads_started", "model_object_loads_completed", "np_load_calls_started", "np_load_calls_completed")
    twice = ("model_opens_started", "model_opens_completed", "model_closes_started", "model_closes_completed")
    counts = set(once + twice) | {"model_object_loads_failed", "model_read_calls_started", "model_read_calls_completed",
        "total_model_bytes_returned_by_reads", "hash_bytes_returned_by_reads", "np_load_bytes_returned_by_reads",
        "model_seeks_started", "model_seeks_completed"}
    _keys(value, counts | {"status", "distinct_npz_files", "blocked_operations", "per_model", "patches_restored",
                          "open_handles_remaining", "elapsed_seconds"}, "model_io_keys")
    _need(value["status"] == "completed" and _same(value["distinct_npz_files"], 2)
          and _same(value["blocked_operations"], 0) and value["patches_restored"] is True
          and _same(value["open_handles_remaining"], 0), "model_io_completion")
    _seconds(value["elapsed_seconds"], 20, "model_io_time_limit")
    _keys(value["per_model"], {"task", "crop"}, "model_io_inventory")
    _need(all(_integer(value[k]) for k in counts)
          and value["total_model_bytes_returned_by_reads"] <= 4194304, "model_io_budget")
    for name, item in value["per_model"].items():
        _keys(item, counts | {"path", "expected_sha256", "actual_sha256", "sha_verified", "status",
            "file_identity_checked", "hash_opens_started", "np_load_opens_started"}, "model_item_keys")
        _need(all(_integer(item[k]) for k in counts)
              and all(_same(item[k], 1) for k in (*once, "hash_opens_started", "np_load_opens_started"))
              and all(_same(item[k], 2) for k in twice) and _same(item["model_object_loads_failed"], 0),
              "model_once_counts")
        _need(item["path"] == cfg["models"][name]["path"]
              and item["expected_sha256"] == item["actual_sha256"] == previous.MODEL_SHA[name]
              and item["sha_verified"] is True and item["file_identity_checked"] is True
              and item["status"] == "completed", "model_identity")
        _need(0 < item["model_read_calls_started"] == item["model_read_calls_completed"]
              and item["model_seeks_started"] == item["model_seeks_completed"]
              and item["hash_bytes_returned_by_reads"] == cfg["model_protection"][item["path"]]["size"]
              and item["np_load_bytes_returned_by_reads"] > 0
              and item["total_model_bytes_returned_by_reads"] ==
                  item["hash_bytes_returned_by_reads"] + item["np_load_bytes_returned_by_reads"], "model_read_accounting")
    _need(all(value[k] == sum(item[k] for item in value["per_model"].values()) for k in counts),
          "model_io_totals")
    return {key: value[key] for key in sorted(counts)}


def check(out: Path, cfg: dict, deadline: float):
    """Read selection, primary and pair once; never execute a source or policy."""
    started = time.monotonic()
    _need(type(deadline) in (int, float) and math.isfinite(deadline), "checker_deadline_type")
    cutoff, folder = min(deadline, started + 20), Path(out) / "check"
    consumption = {f"{kind}_{suffix}": 0 for kind in ("selection_receipt", "primary_receipt", "pair_receipt")
                   for suffix in ("open_attempts", "read_calls_started", "read_calls_completed", "files_read", "bytes_read")}
    consumption.update(json_decode_attempts=0, json_files_decoded=0, json_bytes_submitted=0,
        json_bytes_decoded=0, pairs_verified=0, selections_verified=0, source_reads=0,
        model_reads_or_scores=0, controller_calls=0, old_compact_or_witness_reads=0, replay_reads=0)
    unknown, owned, safe = [], False, None

    def tick():
        _need(time.monotonic() < cutoff, "checker_deadline")

    def read_once(name, kind, cap, expected_sha, expected_bytes=None):
        tick()
        path = Path(out) / name
        stat = path.stat()
        size = stat.st_size
        _need(0 < size <= cap and (expected_bytes is None or size == expected_bytes), "receipt_size")
        used = sum(consumption[k + "_bytes_read"] for k in ("selection_receipt", "primary_receipt", "pair_receipt"))
        _need(used + size <= MAX_BYTES, "aggregate_receipt_bytes")
        consumption[kind + "_open_attempts"] += 1
        with path.open("rb", buffering=0) as stream:
            consumption[kind + "_read_calls_started"] += 1
            try:
                raw = stream.read(size)
            except BaseException:
                unknown.append(kind + "_failed_read_unreturned_bytes_unknown")
                raise
            consumption[kind + "_read_calls_completed"] += 1
            consumption[kind + "_bytes_read"] += len(raw)
            _need(len(raw) == size, "receipt_short_read")
            current = os.fstat(stream.fileno())
            after = path.stat()
            _need((stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino) ==
                  (current.st_size, current.st_mtime_ns, current.st_dev, current.st_ino) ==
                  (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino), "receipt_changed")
        consumption[kind + "_files_read"] += 1
        _need(hashlib.sha256(raw).hexdigest() == expected_sha, "receipt_sha")
        consumption["json_decode_attempts"] += 1
        consumption["json_bytes_submitted"] += len(raw)
        try:
            value = previous._decode(raw)
        except BaseException:
            unknown.append("failed_json_decode_partial_objects_unknown")
            raise
        consumption["json_files_decoded"] += 1
        consumption["json_bytes_decoded"] += len(raw)
        tick()
        return value, size

    try:
        _configuration(cfg)
        folder.mkdir(exist_ok=False)
        owned = True
        previous._write_new(folder / "started.json", dict(schema="first-operation-independent-check-started-v1",
            primary_receipt_sha256=cfg["primary_receipt_sha256"],
            selection_receipt_sha256=cfg["selection_receipt_sha256"],
            aggregate_receipt_bytes_max=MAX_BYTES, passes_per_receipt_max=1))
        selection, selection_bytes = read_once("selection.json", "selection_receipt", 131072,
                                              cfg["selection_receipt_sha256"])
        selected, safe_selected = _selection(selection, cfg, selection_bytes)
        consumption["selections_verified"] += 1
        primary, primary_bytes = read_once("primary-receipt.json", "primary_receipt", 131072,
                                          cfg["primary_receipt_sha256"])
        binding = _primary(primary, primary_bytes)
        pair, _ = read_once("pair.json", "pair_receipt", 262144, binding["sha256"], binding["bytes"])
        safe, calls, cells = _pair(pair, selected)
        counts = _ledger(primary["ledger"], cells, calls[0], selected, cfg)
        reported = primary["safe_result"]
        _keys(reported, {"game", "unit_vectors_equal", "changed_unit_count"}, "primary_safe_keys")
        _need(all(_same(reported[key], safe[key]) for key in reported), "primary_safe_reconciliation")
        consumption["pairs_verified"] += 1
        tick()
        _need(not unknown and consumption["json_files_decoded"] == consumption["json_decode_attempts"] == 3
              and all(consumption[k + "_files_read"] == consumption[k + "_open_attempts"] == 1
                      for k in ("selection_receipt", "primary_receipt", "pair_receipt")), "complete_read_accounting")
        summary = dict(schema="first-operation-unit-request-safe-summary-v1", selected=safe_selected,
            result=safe, counts=counts, selection_consumption=selection["consumption"], independence_limit=LIMIT)
        summary_sha = previous._write_new(folder / "safe-summary.json", summary)
        tick()
        certificate = dict(schema="first-operation-independent-check-v1", status="complete", passed=True,
            consumption=consumption.copy(), unknown_accounting=[], primary_receipt_sha256=cfg["primary_receipt_sha256"],
            selection_receipt_sha256=cfg["selection_receipt_sha256"], pair_sha256=binding["sha256"],
            safe_summary_sha256=summary_sha, bound_primary_counts=counts, unit_vectors_equal=safe["unit_vectors_equal"],
            changed_unit_count=safe["changed_unit_count"], elapsed_seconds_before_terminal_write=time.monotonic() - started,
            independence_limit=LIMIT)
        previous._write_new(folder / "complete.json", certificate)
        tick()
        return certificate
    except BaseException as error:
        if owned:
            previous._write_new(folder / "failure.json", dict(schema="first-operation-independent-check-failure-v1",
                status="failed", passed=False, consumption=consumption.copy(), unknown_accounting=unknown.copy(),
                verified_result=safe, elapsed_seconds_before_terminal_write=time.monotonic() - started,
                failure=dict(type=type(error).__name__, code=str(error) if isinstance(error, CheckFailure) else "checker_failed"),
                independence_limit=LIMIT))
        raise
