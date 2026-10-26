"""Independent, bounded verification of the two new label-equivalence receipts.

This checker never imports the producer or opens original inputs, Replay,
policies, models, or historical evidence. It verifies the newly bound evidence,
not a second original-source execution. Only a complete requested-action match
is a positive; opportunity-generator candidates are not all official actions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


IDENTITY = dict(game="collect-31317742-v1-0-teacher", seed=31317742,
                opponent="v1", side=0, step=15, split="original_train",
                callback_index_zero_based=0)
SOURCE_IDENTITIES = {
    "index": (99098, "87bc7ef55b97f11fcb14c02cfd5378a307f42e47d51b25b5a1664b1576e89248"),
    "inputs": (6055173, "d54dc5c2438e66e9026867b613adbcffcd8472c27d8622bf7c0a2772e32e8ff0"),
    "labels": (5223, "c2db56ec26f77cdd84b5e4151ba662254dd294de2eba4b673572f49a427924c2"),
    "replay": (427472, "6ebe2a567af569bb425b7acec640da84ee727adb5666223fcf062ea30c395aac"),
}
OPS = ("DIG", "FERTILIZE", "HARVEST", "PLANT", "WATER")
SOURCE_COUNTERS = (
    "sources_started", "sources_completed", "sources_failed",
    "raw_opens_started", "raw_opens_completed", "raw_read_calls_started",
    "raw_read_calls_completed", "raw_bytes_returned", "raw_bytes_hashed",
    "raw_bound_eof_returns", "raw_drain_read_calls_started",
    "raw_drain_read_calls_completed", "raw_drain_bytes_returned",
    "raw_closes_started", "raw_closes_completed", "raw_hashes_completed",
    "gzip_opens_started", "gzip_opens_completed", "gzip_read_calls_started",
    "gzip_read_calls_completed", "gzip_readline_calls_started",
    "gzip_readline_calls_completed", "gzip_closes_started", "gzip_closes_completed",
    "returned_payload_bytes", "prefix_lines_returned", "partial_prefix_lines_returned",
    "skipped_prefix_lines", "skipped_prefix_bytes", "json_decodes_started",
    "json_decodes_completed", "json_bytes_submitted", "json_bytes_decoded",
    "decoded_json_values", "decoded_mapping_objects", "decoded_list_objects",
    "decoded_scalar_values", "decoded_mapping_keys", "decoded_replay_states",
    "decoded_replay_agent_states", "decoded_input_candidate_entries",
    "decoded_input_candidate_objects", "decoded_input_candidate_x_arrays",
    "decoded_input_shared_x_arrays", "decoded_observation_objects",
    "decoded_configuration_objects", "stat_checks_started", "stat_checks_completed",
    "raw_reread_passes", "suffix_json_decodes", "model_reads", "controller_calls",
) + tuple(kind + suffix for kind in SOURCE_IDENTITIES for suffix in
          ("_json_decodes_started", "_json_decodes_completed"))
CONTROLLER_FIELDS = ("calls", "original", "baseline", "trials", "callbacks", "encodings",
                     "encoded_candidates", "crop_callbacks", "crop_encodings", "crop_feature_values")
LIMITS = dict(teacher_controller_calls_max=66, original_teacher_calls_max=1,
    teacher_adapter_baseline_calls_max=1, alternative_calls_max=64,
    per_controller_call_seconds_max=10, callbacks_per_adapter_call_max=16,
    candidate_encodings_max=1040, candidates_per_callback_max=512,
    encoded_candidate_objects_max=200000, original_compressed_opens_max=4,
    original_compressed_bytes_total_exact_if_all_completed=6586966,
    source_returned_payload_bytes_total_max=134217728, index_prefix_json_rows_max=256,
    inputs_labels_prefix_lines_each_max=256, target_input_json_rows_max=1,
    target_label_json_rows_max=1, per_input_json_line_bytes_max=4194304,
    replay_json_decodes_max=1, replay_payload_bytes_max=67108864,
    full_replay_states_decoded_max=720, replay_agent_states_decoded_max=1440,
    raw_data_reread_passes=0, model_reads_hashes_loads_forwards=0,
    training_new_games_official_steps=0,
    new_independent_machine_receipt_read_bytes_max=2097152,
    new_private_case_receipt_bytes_max=1048576, tiny_in_memory_cases_max=6,
    synthetic_candidates_max=32, temporary_files=0,
    per_close_seconds_max_included_in_original_phase=10)
SCOPE = ("One fixed original-training context; complete requested actions only. "
         "Independent receipt check, not a second source reproduction or official execution. "
         "An equivalent alternative is existential in this context, not a reusable label set, "
         "a full official legality mask, a reward improvement, or holdout evidence.")


class CheckFailure(ValueError):
    """Only fixed non-private codes are exposed to safe failure receipts."""


def _need(condition, code):
    if not condition:
        raise CheckFailure(code)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _keys(value, expected, code):
    _need(type(value) is dict and set(value) == set(expected), code)


def _seconds(value, maximum, code):
    _need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum, code)


def _same(left, right):
    """JSON types and array order matter; object field insertion order does not."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left == right
    return left == right


def _raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_raw(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _number(value):
    result = float(value)
    _need(math.isfinite(result), "nonfinite_json_number")
    return result


def _constant(_value):
    raise CheckFailure("nonfinite_json_constant")


def _descriptor(value):
    _keys(value, ("id", "unit", "target", "op"), "candidate_descriptor_shape")
    _need(_integer(value["unit"]) and type(value["target"]) is list
          and len(value["target"]) == 2 and all(_integer(v) for v in value["target"])
          and value["op"] in OPS and type(value["id"]) is str, "candidate_descriptor_values")
    x, y = value["target"]
    _need(value["id"] == f"u{value['unit']}:x{x}:y{y}:{value['op']}", "candidate_descriptor_id")


def _vector(action):
    _keys(action, ("farmer", "hands", "market"), "whole_action_envelope")
    _need(all(type(action[k]) is list for k in action)
          and all(type(v) is list for v in action["hands"]), "whole_action_ordered_branches")
    return [action["farmer"], *action["hands"]]


def _paired_counts(counts, fields):
    for name in fields:
        start, done = counts[name + "_started"], counts[name + "_completed"]
        _need(_integer(start) and _integer(done) and start == done, "incomplete_count_pair")


def _source(case, safe_source, cfg):
    fixed, limits = cfg["fixed_case"], cfg["budget"]
    cert = case["source_certificate"]
    _keys(cert, ("schema", "fixed_identity", "available", "target_row_index", "reason", "consumption"),
          "source_certificate_shape")
    _need(cert["schema"] == "teacher-label-original-source-certificate-v1"
          and _same(cert["fixed_identity"], IDENTITY)
          and type(case["available"]) is bool and cert["available"] is case["available"]
          and cert["reason"] == case["reason"], "source_certificate_identity")
    source = cert["consumption"]
    _need(type(source) is dict and type(safe_source) is dict and "index_prefix" not in safe_source,
          "source_public_prefix_leak")
    expected_safe = {k: v for k, v in source.items() if k != "index_prefix"}
    _seconds(source["elapsed_seconds"], 220, "source_seconds")
    _seconds(safe_source["elapsed_seconds"], 220, "safe_source_seconds")
    _need(_same(expected_safe, safe_source), "source_ledger_certificate_binding")
    _need(source["status"] == ("completed" if case["available"] else "completed_unavailable")
          and source["target_row_index"] == cert["target_row_index"]
          and source["unavailable_reason"] == cert["reason"], "source_completion_binding")
    items, prefix = source["per_source"], source["index_prefix"]
    kinds = ["index", "inputs", "labels", "replay"] if case["available"] else ["index"]
    _need(type(items) is list and [v["kind"] for v in items] == kinds, "source_read_order")
    _need(type(prefix) is list and len(prefix) <= 256, "index_prefix_limit")
    previous = None
    for row in prefix:
        _need(type(row) is list and len(row) == 3 and _integer(row[0]) and row[0] < 719
              and _integer(row[1]) and _sha(row[2]), "index_prefix_row")
        position = (row[0], row[1])
        _need((previous is None and row[1] == 0)
              or (previous is not None and ((row[0] == previous[0] and row[1] == previous[1] + 1)
                  or (row[0] > previous[0] and row[1] == 0))), "index_prefix_order")
        previous = position
    target = cert["target_row_index"]
    if case["available"]:
        _need(_integer(target) and target < 256 and len(prefix) == target + 1
              and previous == (15, 0) and all(row[0] < 15 for row in prefix[:-1])
              and cert["reason"] is None, "fixed_target_prefix_binding")
    else:
        _need(target is None and all(row[:2] != [15, 0] for row in prefix), "unavailable_target_found")
        reason = cert["reason"]
        if reason == "index_target_passed":
            _need(bool(prefix) and prefix[-1][0] > 15 and all(row[0] < 15 for row in prefix[:-1]),
                  "unavailable_passed_evidence")
        elif reason == "index_target_absent":
            _need(items[0]["gzip_eof_verified"] is True and all(row[0] < 15 for row in prefix),
                  "unavailable_absent_evidence")
        elif reason == "index_prefix_limit":
            _need(len(prefix) == 256 and all(row[0] < 15 for row in prefix),
                  "unresolved_prefix_bound_evidence")
        else:
            raise CheckFailure("unavailable_reason_not_proven")
    for counter in SOURCE_COUNTERS:
        _need(_integer(source[counter]) and all(_integer(item[counter]) for item in items)
              and source[counter] == sum(item[counter] for item in items), "source_counter_sum")
    zeros = ("sources_failed", "partial_prefix_lines_returned", "raw_reread_passes",
             "suffix_json_decodes", "model_reads", "controller_calls")
    _need(all(source[k] == 0 for k in zeros), "source_forbidden_or_partial_work")
    planned = []
    for item in items:
        kind, meta = item["kind"], fixed["raw_source_files"][item["kind"]]
        byte_count, sha = SOURCE_IDENTITIES[kind]
        _need(item["status"] == "completed" and item["source_path"] == meta["path"]
              and item["registered_bytes"] == meta["bytes"] == byte_count
              and item["inherited_sha256"] == item["raw_sha256"] == meta["sha256_inherited"] == sha
              and item["raw_bytes_returned"] == item["raw_bytes_hashed"] == byte_count,
              "source_full_compressed_identity")
        _need(all(item[k] is True for k in ("raw_sha256_verified", "source_stat_before_verified",
                                            "source_stat_after_verified")), "source_stat_or_hash_unverified")
        _paired_counts(item, ("sources", "raw_opens", "raw_read_calls", "raw_drain_read_calls",
            "raw_closes", "gzip_opens", "gzip_read_calls", "gzip_readline_calls",
            "gzip_closes", "json_decodes", "stat_checks"))
        _need(all(item[k] == 1 for k in ("sources_completed", "raw_opens_completed",
            "raw_closes_completed", "gzip_opens_completed", "gzip_closes_completed", "raw_hashes_completed"))
              and item["stat_checks_completed"] == 4 and item["raw_read_calls_completed"] > 0
              and item["raw_drain_read_calls_completed"] <= item["raw_read_calls_completed"]
              and item["raw_drain_bytes_returned"] <= byte_count, "source_once_and_close_counts")
        _need(item["json_bytes_submitted"] == item["json_bytes_decoded"]
              and item["decoded_json_values"] == item["decoded_mapping_objects"]
                  + item["decoded_list_objects"] + item["decoded_scalar_values"], "decoded_inventory_accounting")
        for other in SOURCE_IDENTITIES:
            _need(item[other + "_json_decodes_started"] == item[other + "_json_decodes_completed"]
                  == (item["json_decodes_completed"] if other == kind else 0), "source_decode_kind_counts")
        if kind == "replay":
            _need(item["gzip_eof_verified"] is True and item["internal_unreturned_payload_bytes"] == 0
                  and item["gzip_read_calls_completed"] >= 1 and item["gzip_readline_calls_completed"] == 0
                  and item["prefix_lines_returned"] == item["skipped_prefix_lines"] == 0
                  and item["skipped_prefix_bytes"] == 0 and item["prefix_stop_reason"] is None
                  and item["json_decodes_completed"] == 1 and item["decoded_replay_states"] == 720
                  and item["decoded_replay_agent_states"] == 1440
                  and item["returned_payload_bytes"] == item["json_bytes_decoded"] <= 67108864,
                  "complete_replay_consumption")
        else:
            eof = kind == "index" and cert["reason"] == "index_target_absent"
            line_count = len(prefix) if kind == "index" else target + 1
            _need(item["gzip_eof_verified"] is eof
                  and item["gzip_read_calls_completed"] == 0
                  and item["prefix_lines_returned"] == line_count
                  and item["gzip_readline_calls_completed"] == line_count + int(eof)
                  and item["json_decodes_completed"] == (line_count if kind == "index" else 1)
                  and item["skipped_prefix_lines"] == (0 if kind == "index" else target)
                  and item["returned_payload_bytes"] == item["json_bytes_decoded"] + item["skipped_prefix_bytes"],
                  "aligned_prefix_consumption")
            _need((type(item["internal_unreturned_payload_bytes"]) is int
                   and item["internal_unreturned_payload_bytes"] == 0) if eof
                  else item["internal_unreturned_payload_bytes"] is None, "prefix_buffer_accounting")
            if not eof:
                planned.append(dict(kind=kind, quantity="gzip_internal_unreturned_payload_bytes", value=None,
                                    reason="intentional_prefix_without_eof_probe"))
            expected_stop = ("fixed_callback_found" if case["available"] else cert["reason"])
            _need(item["prefix_stop_reason"] == (expected_stop if kind == "index" else "aligned_target_found"),
                  "source_prefix_stop_reason")
            if kind in ("inputs", "labels"):
                _need(item["target_row_index"] == target and _sha(item["target_line_sha256"]), "aligned_target_row_hash")
            if kind == "inputs":
                _need(item["target_input_sha256"] == prefix[-1][2]
                      and item["json_bytes_decoded"] <= limits["per_input_json_line_bytes_max"], "saved_input_index_binding")
        if kind != "replay":
            _need(item["decoded_replay_states"] == item["decoded_replay_agent_states"] == 0,
                  "unexpected_replay_decode")
    _need(_same(source["unknown_accounting"], dict(planned_prefix=planned, failures=[])),
          "source_unknown_accounting")
    _need(source["raw_bytes_returned"] == (6586966 if case["available"] else 99098)
          and source["returned_payload_bytes"] <= 134217728, "source_total_budget")
    return source


def _calls(case, cfg):
    totals = dict(calls=0, original=0, baseline=0, trials=0, callbacks=0, encodings=0,
                  encoded_candidates=0, crop_callbacks=0, crop_encodings=0, crop_feature_values=0,
                  forced_choices=0)
    if not case["available"]:
        _need(case["training_binding"] is None and case["first_candidates"] == [] and case["trials"] == []
              and not any(k in case for k in ("stored_action", "original_teacher", "baseline")),
              "unavailable_controller_work")
        outcome = ("inconclusive_source_prefix_limit" if case["reason"] == "index_prefix_limit"
                   else "inconclusive_no_fixed_callback")
        return totals, outcome, dict(attempted=0, available_alternatives=0,
            exhaustive=False, first_full_match_trial_index=None, unit_only_matches_market_different=0)
    choices, binding = case["first_candidates"], case["training_binding"]
    _need(type(choices) is list and 1 <= len(choices) <= 512, "first_candidate_count")
    for choice in choices:
        _descriptor(choice)
    ids = [c["id"] for c in choices]
    _need(len(set(ids)) == len(ids), "duplicate_first_candidate")
    _keys(binding, ("index_input_sha256", "label", "saved_input_sha256", "saved_neural_projection_sha256",
                    "baseline_input_sha256", "baseline_neural_projection_sha256"), "training_binding_shape")
    _need(all(_sha(v) for k, v in binding.items() if k != "label"), "training_binding_hashes")
    label = binding["label"]
    _keys(label, ("teacher_label", "index"), "saved_single_label_shape")
    _need(_integer(label["index"]) and label["index"] < len(ids)
          and label["teacher_label"] == ids[label["index"]], "saved_single_label_binding")
    source = case["source_certificate"]["consumption"]
    _need(binding["index_input_sha256"] == binding["saved_input_sha256"] == binding["baseline_input_sha256"]
          == source["index_prefix"][-1][2] == source["per_source"][1]["target_input_sha256"]
          and binding["saved_neural_projection_sha256"] == binding["baseline_neural_projection_sha256"],
          "training_numeric_projection_binding")
    # The frozen dataset writer emits exactly canonical UTF-8 JSON plus LF.
    # This independently binds the shown positive label to its actual source
    # row, rather than accepting an unconnected label in the producer receipt.
    label_line = _raw(label) + b"\n"
    label_item = source["per_source"][2]
    _need(hashlib.sha256(label_line).hexdigest() == label_item["target_line_sha256"]
          and len(label_line) == label_item["json_bytes_decoded"], "saved_label_source_line_binding")
    input_item = source["per_source"][1]
    _need(all(input_item[k] == len(choices) for k in ("decoded_input_candidate_entries",
        "decoded_input_candidate_objects", "decoded_input_candidate_x_arrays"))
          and input_item["decoded_input_shared_x_arrays"] == 1, "saved_candidate_inventory_binding")
    original, base, trials = case["original_teacher"], case["baseline"], case["trials"]
    _need(type(trials) is list and len(trials) <= 64, "alternative_trial_cap")
    baseline_vector = _vector(case["stored_action"])
    snapshot = original["input_snapshot_sha256_before"]
    _need(_sha(snapshot), "original_snapshot_hash")
    alternatives = [value for value in ids if value != label["teacher_label"]]
    _need(len(trials) <= len(alternatives), "too_many_alternatives")
    for index, call in enumerate([original, base, *trials]):
        _need(call["completed"] is True and _integer(call["call_index"]) and call["call_index"] == index,
              "call_completion_order")
        _seconds(call["seconds"], 10, "individual_controller_seconds")
        _need(call["input_snapshot_sha256_before"] == call["input_snapshot_sha256_after"] == snapshot,
              "fresh_clone_snapshot_binding")
        vector = _vector(call["action"])
        _need(1 <= len(vector) <= 16 and _same(vector, call["unit_vector"]), "complete_unit_vector")
        totals["calls"] += 1
        totals["original" if index == 0 else "baseline" if index == 1 else "trials"] += 1
        if index < 2:
            _need(call["forced_id"] is None and _same(call["action"], case["stored_action"]),
                  "stored_original_baseline_whole_action")
        if index == 0:
            _need(not any(k in call for k in ("events", "crop_events", "first_input_sha256")),
                  "original_teacher_not_adapter")
            continue
        forced = alternatives[index - 2] if index > 1 else None
        _need(call["forced_id"] == forced and call["first_input_sha256"] == binding["baseline_input_sha256"]
              and call["first_neural_projection_sha256"] == binding["baseline_neural_projection_sha256"],
              "trial_order_and_first_context")
        events, crops = call["events"], call["crop_events"]
        _need(type(events) is list and 1 <= len(events) <= 16
              and type(crops) is list and len(crops) <= 16, "adapter_callback_limits")
        assigned_units = set()
        for callback, event in enumerate(events):
            _keys(event, ("callback_index", "encoding_sequence", "input_sha256", "candidate_sha256",
                "candidate_count", "teacher_label", "chosen", "forced", "unit", "target", "op"), "task_event_shape")
            _need(_integer(event["callback_index"]) and event["callback_index"] == callback
                  and _integer(event["encoding_sequence"], 1)
                  and event["encoding_sequence"] == totals["encodings"] + 1
                  and _sha(event["input_sha256"]) and _sha(event["candidate_sha256"])
                  and _integer(event["candidate_count"], 1) and event["candidate_count"] <= 512,
                  "fresh_task_encoding_sequence")
            _descriptor(dict(id=event["chosen"], unit=event["unit"], target=event["target"], op=event["op"]))
            _need(event["unit"] < len(vector) and event["unit"] not in assigned_units,
                  "task_unit_assignment_reused_or_out_of_range")
            assigned_units.add(event["unit"])
            _need(type(event["teacher_label"]) is str and type(event["forced"]) is bool,
                  "task_choice_types")
            first_forced = callback == 0 and index > 1
            _need(event["forced"] is first_forced
                  and event["chosen"] == (forced if first_forced else event["teacher_label"]),
                  "only_first_override_and_dynamic_teacher_suffix")
            if callback == 0:
                _need(event["candidate_count"] == len(choices) and event["candidate_sha256"] == _digest(choices)
                      and event["input_sha256"] == binding["baseline_input_sha256"]
                      and event["teacher_label"] == label["teacher_label"]
                      and event["chosen"] in ids, "first_candidate_list_and_label_binding")
            totals["callbacks"] += 1
            totals["encodings"] += 1
            totals["encoded_candidates"] += event["candidate_count"]
            totals["forced_choices"] += int(first_forced)
        for callback, crop in enumerate(crops):
            _keys(crop, ("callback_index", "encoding_sequence", "teacher", "chosen", "x_sha256",
                         "x_values", "unit", "model_present"), "crop_event_shape")
            _need(_integer(crop["callback_index"]) and crop["callback_index"] == callback
                  and _integer(crop["encoding_sequence"], 1)
                  and crop["encoding_sequence"] == totals["crop_encodings"] + 1
                  and _same(crop["teacher"], crop["chosen"]) and type(crop["teacher"]) is str
                  and _sha(crop["x_sha256"]) and _integer(crop["x_values"], 1) and crop["x_values"] <= 1024
                  and _integer(crop["unit"]) and crop["unit"] < len(vector)
                  and crop["model_present"] is False, "fresh_pure_teacher_crop_encoding")
            totals["crop_callbacks"] += 1
            totals["crop_encodings"] += 1
            totals["crop_feature_values"] += crop["x_values"]
    _need(totals["calls"] == 2 + len(trials) <= 66 and totals["encodings"] <= 1040
          and totals["encoded_candidates"] <= 200000 and totals["crop_encodings"] <= 1040
          and totals["crop_feature_values"] <= 1064960,
          "whole_controller_budget")
    match, unit_only = None, 0
    for index, trial in enumerate(trials):
        if _same(trial["action"], base["action"]):
            _need(match is None and index == len(trials) - 1, "did_not_stop_at_first_full_match")
            match = index
        elif _same(trial["unit_vector"], baseline_vector):
            _need(not _same(trial["action"]["market"], base["action"]["market"]), "unit_only_difference_not_market")
            unit_only += 1
    exhaustive = len(trials) == len(alternatives)
    if not alternatives:
        outcome = "inconclusive_single_candidate"
    elif match is not None:
        outcome = "equivalent_alternative"
    elif exhaustive:
        outcome = "local_no_match"
    else:
        _need(len(trials) == 64, "incomplete_alternative_enumeration")
        outcome = "inconclusive_trial_limit"
    comparison = dict(attempted=len(trials), available_alternatives=len(alternatives), exhaustive=exhaustive,
        first_full_match_trial_index=match, unit_only_matches_market_different=unit_only)
    return totals, outcome, comparison


def _verify(case, primary, cfg):
    fixed, limits = cfg["fixed_case"], cfg["budget"]
    _need(_same({k: fixed[k] for k in IDENTITY}, IDENTITY), "registered_fixed_case_changed")
    _need(all(type(limits[k]) is int and limits[k] == v for k, v in LIMITS.items()), "registered_limits_changed")
    crop_limits = dict(crop_encodings_max=1040, crop_callbacks_per_adapter_call_max=16,
                       per_crop_vector_values_max=1024, crop_feature_values_max=1064960)
    _need(all(type(limits[k]) is int and limits[k] == v for k, v in crop_limits.items()),
          "crop_limits_changed")
    public_fixed = {k: IDENTITY[k] for k in ("game", "side", "step", "callback_index_zero_based")}
    _need(case["schema"] == "teacher-label-equivalence-case-v1"
          and primary["schema"] == "teacher-label-equivalence-primary-v1"
          and primary["status"] == "complete" and primary["phase"] == "complete"
          and primary["failure_type"] is None and _same(case["fixed"], public_fixed)
          and _same(primary["fixed"], public_fixed), "complete_receipt_identity")
    case_keys = ["schema", "fixed", "available", "reason", "source_certificate", "training_binding",
                 "first_candidates", "trials", "outcome", "comparison"]
    _keys(case, case_keys + (["stored_action", "original_teacher", "baseline"] if case["available"] else []),
          "private_case_schema")
    _seconds(primary["elapsed_seconds"], 220, "primary_seconds")
    ledger = primary["ledger"]
    _keys(ledger, ("controller", "prohibited", "unknown_accounting", "source", "patches_restored"),
          "primary_ledger_shape")
    _need(ledger["patches_restored"] is True and ledger["unknown_accounting"] == [], "primary_failure_or_patch_state")
    _keys(ledger["prohibited"], ("model_read_attempts", "np_load_attempts", "model_forwards",
                               "training_runs", "new_games", "official_transitions"), "prohibited_counter_shape")
    _need(all(type(v) is int and v == 0 for v in ledger["prohibited"].values()), "prohibited_work")
    source = _source(case, ledger["source"], cfg)
    totals, outcome, comparison = _calls(case, cfg)
    _need(case["outcome"] == primary["outcome"] == outcome
          and _same(case["comparison"], comparison) and _same(primary["comparison"], comparison),
          "independent_outcome_or_comparison_mismatch")
    counts = ledger["controller"]
    _keys(counts, [name + suffix for name in CONTROLLER_FIELDS for suffix in ("_started", "_completed")]
          + ["forced_choices"], "controller_counter_shape")
    _paired_counts(counts, CONTROLLER_FIELDS)
    _need(all(counts[name + "_completed"] == totals[name] for name in CONTROLLER_FIELDS)
          and type(counts["forced_choices"]) is int and counts["forced_choices"] == totals["forced_choices"],
          "controller_event_ledger_binding")
    tests = primary["protocol_checks"]
    _need(tests["passed"] is True and type(tests["cases"]) is int and tests["cases"] == 6
          and _integer(tests["synthetic_candidates"]) and tests["synthetic_candidates"] <= 32,
          "protocol_total_scope")
    source_tests, probe_tests = tests["source"], tests["probe"]
    _need(source_tests["passed"] is True and source_tests["tiny_in_memory_cases"] == 2
          and type(source_tests["cases"]) is list and len(source_tests["cases"]) == 2
          and all(v["passed"] is True for v in source_tests["cases"])
          and source_tests["synthetic_candidates"] == source_tests["temporary_files"] == 0
          and probe_tests["passed"] is True and probe_tests["cases"] == 4
          and all(type(probe_tests[k]) is int and probe_tests[k] == 0 for k in
                  ("real_controller_calls", "model_calls", "official_transitions", "temporary_files"))
          and tests["synthetic_candidates"] == probe_tests["synthetic_candidates"], "protocol_partition")
    return dict(fixed=public_fixed, outcome=outcome, comparison=comparison, controller=totals,
        original_compressed_bytes=source["raw_bytes_returned"],
        original_payload_bytes=source["returned_payload_bytes"],
        planned_prefix_unknowns=len(source["unknown_accounting"]["planned_prefix"]),
        failure_unknowns=0, source_target_row_index=case["source_certificate"]["target_row_index"],
        scope=SCOPE)


class _Reader:
    def __init__(self, deadline, limits):
        self.deadline, self.limits = deadline, limits
        self.counts = dict.fromkeys(("opens_started", "opens_completed", "reads_started", "reads_completed",
            "bytes_returned", "bytes_hashed", "json_decodes_started", "json_decodes_completed",
            "json_bytes_submitted", "json_bytes_decoded", "closes_started", "closes_completed"), 0)
        self.files, self.unknown = [], []

    def tick(self):
        _need(time.monotonic() < self.deadline, "checker_deadline")

    def read(self, path, expected_sha, maximum):
        self.tick()
        _need(_sha(expected_sha) and path.name not in self.files, "receipt_hash_or_reread")
        self.files.append(path.name)
        before = path.stat()
        size = before.st_size
        _need(0 < size <= maximum and self.counts["bytes_returned"] + size
              <= min(2097152, self.limits["new_independent_machine_receipt_read_bytes_max"]),
              "receipt_read_byte_budget")
        stream = None
        try:
            self.counts["opens_started"] += 1
            try:
                stream = path.open("rb", buffering=0)
            except BaseException:
                self.unknown.append(dict(file=path.name, quantity="open_handle_acquired", value=None))
                raise
            self.counts["opens_completed"] += 1
            self.tick()
            self.counts["reads_started"] += 1
            try:
                raw = stream.read(size)
            except BaseException:
                self.unknown.append(dict(file=path.name, quantity="bytes_returned_in_failed_read", value=None))
                raise
            self.counts["reads_completed"] += 1
            self.counts["bytes_returned"] += len(raw)
            try:
                actual_sha = hashlib.sha256(raw).hexdigest()
            except BaseException:
                self.unknown.append(dict(file=path.name, quantity="bytes_hashed_in_failed_hash", value=None))
                raise
            self.counts["bytes_hashed"] += len(raw)
            self.tick()
            after = os.fstat(stream.fileno())
            _need(len(raw) == size and (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
                  == (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
                  and actual_sha == expected_sha, "receipt_identity_or_hash_changed")
        finally:
            if stream is not None:
                close_start = time.monotonic()
                self.counts["closes_started"] += 1
                try:
                    stream.close()
                except BaseException:
                    self.unknown.append(dict(file=path.name, quantity="handle_closed", value=None))
                    raise
                self.counts["closes_completed"] += 1
                _need(time.monotonic() - close_start <= 10, "receipt_close_deadline")
        self.tick()
        self.counts["json_decodes_started"] += 1
        self.counts["json_bytes_submitted"] += len(raw)
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                               parse_float=_number, parse_constant=_constant)
        except BaseException:
            self.unknown.append(dict(file=path.name, quantity="partially_decoded_json_values", value=None))
            raise
        self.counts["json_decodes_completed"] += 1
        self.counts["json_bytes_decoded"] += len(raw)
        self.tick()
        _need(type(value) is dict, "receipt_root_not_object")
        return value, len(raw)


def _write(path, value, reader, enforce_deadline=True):
    if enforce_deadline:
        reader.tick()
    raw = _raw(value) + b"\n"
    _need(len(raw) <= 65536, "checker_output_size")
    with path.open("xb", buffering=0) as stream:
        try:
            written = stream.write(raw)
        except BaseException:
            reader.unknown.append(dict(file=path.name, quantity="bytes_written_in_failed_write", value=None))
            raise
        _need(written == len(raw), "checker_short_write")
        os.fsync(stream.fileno())
    if enforce_deadline:
        reader.tick()


def check(out: Path, cfg: dict, deadline: float) -> dict:
    """Read each new receipt once; write an exclusive, independently derived certificate.

    A failure receipt dominates an already-written complete receipt. The parent
    must also enforce its original deadline, exit status, and natural closure.
    This function neither retries reads nor starts original-source work.
    """
    start = time.monotonic()
    _need(type(deadline) in (int, float) and math.isfinite(deadline), "invalid_checker_deadline")
    reader = _Reader(deadline, cfg["budget"])
    directory = Path(out) / "check"
    directory.mkdir(exist_ok=False)
    certificate = dict(schema="teacher-label-equivalence-independent-check-v1", status="failed", passed=False,
        case_sha256=cfg.get("case_receipt_sha256"), primary_sha256=cfg.get("primary_receipt_sha256"),
        consumption=reader.counts, partial_unknown=reader.unknown, safe_result=None, scope=SCOPE)
    try:
        _write(directory / "started.json", dict(schema=certificate["schema"], status="started",
               case_sha256=certificate["case_sha256"], primary_sha256=certificate["primary_sha256"]), reader)
        case, case_bytes = reader.read(Path(out) / "case.json", certificate["case_sha256"], 1048576)
        primary, _ = reader.read(Path(out) / "primary-receipt.json", certificate["primary_sha256"], 2097152)
        _keys(primary["case"], ("path", "sha256", "bytes"), "primary_case_manifest_shape")
        _need(primary["case"]["path"] == "case.json" and primary["case"]["sha256"] == certificate["case_sha256"]
              and type(primary["case"]["bytes"]) is int and primary["case"]["bytes"] == case_bytes,
              "primary_case_manifest_binding")
        safe = _verify(case, primary, cfg)
        reader.tick()
        _need(reader.counts["opens_started"] == reader.counts["opens_completed"] == 2
              and reader.counts["reads_started"] == reader.counts["reads_completed"] == 2
              and reader.counts["json_decodes_started"] == reader.counts["json_decodes_completed"] == 2
              and reader.counts["closes_started"] == reader.counts["closes_completed"] == 2
              and reader.counts["bytes_returned"] == reader.counts["bytes_hashed"]
                  == reader.counts["json_bytes_submitted"] == reader.counts["json_bytes_decoded"]
              and reader.unknown == [], "independent_reader_accounting")
        certificate.update(status="complete", passed=True, safe_result=safe,
                           elapsed_seconds=time.monotonic() - start)
        _write(directory / "safe-summary.json", certificate, reader)
        _write(directory / "complete.json", certificate, reader)
        reader.tick()  # A late terminal write cannot leave accepted evidence.
        return certificate
    except BaseException as error:
        certificate.update(status="failed", passed=False, safe_result=None,
            failure_code=str(error) if isinstance(error, CheckFailure) else type(error).__name__,
            elapsed_seconds=time.monotonic() - start)
        try:
            _write(directory / "failure.json", certificate, reader, enforce_deadline=False)
        finally:
            # Keep an earlier summary/complete for forensic accounting. Failure
            # is decisive, and callers must not mistake either for acceptance.
            if not (directory / "safe-summary.json").exists():
                _write(directory / "safe-summary.json", certificate, reader, enforce_deadline=False)
        raise
