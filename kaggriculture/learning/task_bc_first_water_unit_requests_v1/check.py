"""Independent one-pass check of the six newly produced request-pair receipts.

Only cfg and the new primary/pair receipts are inputs.  No producer comparison
code, source trace, old compact/witness, model, or policy is imported or opened.
Acceptance requires a successful return and absence of check/failure.json;
even a late output/deadline failure retains its actual consumed accounting.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


OPS = ("DIG", "FERTILIZE", "HARVEST", "PLANT", "WATER")
REQUEST_OPS = OPS + ("PASS", "EAST", "WEST", "SOUTH", "NORTH", "DROP", "PICKUP")
MAX_PRIMARY_BYTES = 64 * 1024
MAX_PAIR_BYTES = 256 * 1024
MAX_DECODED_BYTES = 1024 * 1024
SELECTION = (
    ("v1", 0, 1, "9f8d0dfd5b9c3db69aba220147a540bc9b4fbf3f0202d97b0c070e07e9c08cbc"),
    ("v1", 1, 1, "2393bd9e453684babfec6dfd7323e9200b8161911470a136615daba495284c09"),
    ("market", 0, 1, "10cdcee851af015f8cbfdb0129a40846dec1a4b093242a1b9c8f85893c021155"),
    ("market", 1, 1, "fa63efd11dc6e8937226206056f74d7a365d3d827364c3ee0b54914d542a62d7"),
    ("round2", 0, 0, "a459aa67421d5cc46a460fe28357d4116684555a680faf6ad454ecbb3db22f47"),
    ("round2", 1, 0, "c683e65dae97622e918f9fde6ffa5881f12292d5d5ee30c8c5387811cfd184b8"),
)
MODEL_SHA = {
    "task": "9253bc05e632eec2d5a813557272655ab97e023c09843c60a80781b6cc2ab143",
    "crop": "d497e8284dde1f7b836b1cf88915822f82a96be1f99e7b126d06ea263e281ffd",
}
INDEPENDENCE_LIMIT = (
    "Independent check of newly bound receipts, not a second source or policy "
    "execution. Six correlated historical samples from one seed; unit requests "
    "only, without official execution, economic effects, or generalization."
)


class CheckFailure(ValueError):
    """Only fixed non-private codes may be retained in failure output."""


def _need(condition, code):
    if not condition:
        raise CheckFailure(code)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _keys(value, expected, code):
    _need(type(value) is dict and set(value) == set(expected), code)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        _need(key not in value, "duplicate_json_key")
        value[key] = item
    return value


def _float(value):
    number = float(value)
    _need(math.isfinite(number), "nonfinite_json_number")
    return number


def _constant(_value):
    raise CheckFailure("nonfinite_json_constant")


def _decode(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_float=_float, parse_constant=_constant)


def _same(left, right):
    """Compare JSON types, object order, array order, and exact f64 bits."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (list(left) == list(right)
                and all(_same(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left.hex() == right.hex()
    return left == right


def _typed(value):
    # Independent implementation of the preregistered ordered JSON-domain hash.
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite_comparison_number")
        return ["f64", value.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is list:
        return ["array", [_typed(item) for item in value]]
    if type(value) is dict and all(type(key) is str for key in value):
        return ["object", [[key, _typed(item)] for key, item in value.items()]]
    raise CheckFailure("non_json_comparison_value")


def _ordered_hash(value):
    return hashlib.sha256(json.dumps(_typed(value), ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _difference(left, right, path=()):
    if _same(left, right):
        return None
    if type(left) is dict and type(right) is dict:
        if list(left) != list(right):
            return [*path, "<keys-or-order>"]
        for key in left:
            if not _same(left[key], right[key]):
                return _difference(left[key], right[key], (*path, key))
    if type(left) is list and type(right) is list:
        for index, (a, b) in enumerate(zip(left, right)):
            if not _same(a, b):
                return _difference(a, b, (*path, index))
        return [*path, "<length>"]
    return list(path)


def _write_new(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        _need(stream.write(data) == len(data), "output_short_write")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _seconds(value, maximum, code):
    _need(type(value) in (int, float) and math.isfinite(value)
          and 0 <= value <= maximum, code)


def _vector(value):
    _need(type(value) is list and 1 <= len(value) <= 8, "unit_vector_size")
    for request in value:
        _need(type(request) is list and bool(request) and type(request[0]) is str
              and request[0] in REQUEST_OPS, "unit_request_operation")
    _typed(value)
    return value


def _configuration(cfg):
    _need(type(cfg) is dict and type(cfg["samples"]) is list
          and len(cfg["samples"]) == 6, "fixed_six_samples")
    selected = []
    for entry, (opponent, side, target, sha) in zip(cfg["samples"], SELECTION):
        game = f"holdout-1714414373-{opponent}-{side}-trained"
        expected = {"game": game, "side": side, "opponent": opponent,
                    "step": 15, "first_callback_index_0_based": target,
                    "source_line_sha256": sha,
                    "source_path": f"artifacts/task-bc-holdout-v1-20261023/traces/{game}.jsonl.gz"}
        _need(type(entry) is dict and all(key in entry and _same(entry[key], value)
              for key, value in expected.items()), "sample_binding")
        _need(type(entry["source_stat"]) is dict
              and all(_integer(entry["source_stat"][key], 1) for key in ("size", "mtime_ns")),
              "registered_source_stat")
        selected.append({"game": game, "side": side, "step": 15,
                         "target_index": target, "source_line_sha256": sha})
    _keys(cfg["models"], {"task", "crop"}, "fixed_model_keys")
    paths = {"task": "artifacts/task-bc-holdout-v1-20261023/model/trained.npz",
             "crop": "artifacts/bc-epoch6-holdout-v1-20261023/panels/fixed_epoch6/model/trained.npz"}
    for kind in ("task", "crop"):
        _need(cfg["models"][kind]["sha256"] == MODEL_SHA[kind]
              and cfg["models"][kind]["path"] == paths[kind], "fixed_model_identity")
        protection = cfg["model_protection"][paths[kind]]
        _need(type(protection) is dict and all(_integer(protection[key], 1)
              for key in ("size", "mtime_ns")), "registered_model_stat")
    budget = cfg["budget"]
    for key, value in {"stage_seconds_max": 180, "primary_phase_seconds_max": 120,
                       "independent_checker_seconds_max": 40,
                       "per_sample_pair_seconds_max": 20,
                       "model_load_phase_seconds_max": 20,
                       "retries_max": 0, "additional_seeds_or_turns_max": 0}.items():
        _need(key in budget and _same(budget[key], value), "fixed_budget")
    for key, value in {"pair_receipt_files_max": 6, "aggregate_decoded_bytes_max": MAX_DECODED_BYTES,
                       "passes_per_receipt_max": 1, "source_reads_max": 0,
                       "model_reads_or_scores_max": 0, "controller_calls_max": 0}.items():
        _need(_same(budget["independent_checker"][key], value), "fixed_checker_budget")
    _need(_sha(cfg["primary_receipt_sha256"]), "primary_hash_binding")
    return selected


def _event(event, index, units):
    expected = {"status", "index", "inputs_sha256", "context_sha256", "record_sha256",
                "candidate_ids_sha256", "legal_candidates", "candidate_count",
                "chosen_id", "teacher_id", "selector", "task_forward_calls",
                "task_forward_calls_completed", "task_scored_candidates",
                "task_scored_candidates_completed", "encoding_calls_reserved",
                "encoding_calls_completed", "callback_sequence",
                "encoding_sequence", "score_sequence"}
    _keys(event, expected, "callback_keys")
    _need(event["status"] == "complete" and _same(event["index"], index), "callback_completion_order")
    for key in ("inputs_sha256", "context_sha256", "record_sha256", "candidate_ids_sha256"):
        _need(_sha(event[key]), "callback_hash")
    candidates = event["legal_candidates"]
    _need(type(candidates) is list and 1 <= len(candidates) <= 512, "legal_candidate_count")
    identities = {}
    for candidate in candidates:
        _need(type(candidate) is list and len(candidate) == 4
              and all(_integer(part) for part in candidate), "legal_candidate_shape")
        unit, x, y, op = candidate
        _need(unit < units and op < len(OPS), "legal_candidate_range")
        identity = f"u{unit}:x{x}:y{y}:{OPS[op]}"
        _need(identity not in identities, "duplicate_legal_candidate")
        identities[identity] = (unit, x, y, OPS[op])
    _need(_same(event["candidate_count"], len(candidates))
          and event["candidate_ids_sha256"] == _ordered_hash(list(identities)),
          "legal_candidate_digest")
    _need(type(event["chosen_id"]) is str and event["chosen_id"] in identities
          and type(event["teacher_id"]) is str and event["teacher_id"] in identities,
          "chosen_and_teacher_current_legal")
    _need(_same(event["encoding_calls_reserved"], 1)
          and _same(event["encoding_calls_completed"], 1)
          and _integer(event["callback_sequence"], 1)
          and _integer(event["encoding_sequence"], 1), "callback_enumeration_accounting")
    _need(event["selector"] in ("student", "override"), "callback_selector")
    student = event["selector"] == "student"
    _need(_same(event["task_forward_calls"], int(student))
          and _same(event["task_forward_calls_completed"], int(student))
          and _same(event["task_scored_candidates"], len(candidates) if student else 0)
          and _same(event["task_scored_candidates_completed"], len(candidates) if student else 0),
          "callback_real_scoring_accounting")
    _need(_integer(event["score_sequence"], 1) if student else event["score_sequence"] is None,
          "callback_score_sequence")
    return identities


def _call(value, mode, call_index, units, target):
    _keys(value, {"status", "unit_actions", "returned_unit_actions", "trace_sha256",
                  "events", "crop_events", "counts", "active_seconds", "override_count",
                  "call_index"}, "call_keys")
    _need(value["status"] == "complete" and _same(value["call_index"], call_index), "call_completion_order")
    _need(_sha(value["trace_sha256"]), "call_trace_digest")
    _seconds(value["active_seconds"], 10, "call_active_seconds_limit")
    trace_units, returned = _vector(value["unit_actions"]), _vector(value["returned_unit_actions"])
    _need(len(trace_units) == len(returned) == units and _same(trace_units, returned),
          "returned_trace_unit_vector_identity")
    events = value["events"]
    _need(type(events) is list and target < len(events) <= 8, "call_callback_count")
    maps, chosen_units = [], set()
    for index, event in enumerate(events):
        identities = _event(event, index, units)
        forced = mode == "intervention" and index == target
        _need(event["selector"] == ("override" if forced else "student"), "exact_override_position")
        _need(not any(candidate[0] in chosen_units for candidate in identities.values()),
              "later_enumeration_reused_assigned_unit")
        chosen_units.add(identities[event["chosen_id"]][0])
        maps.append(identities)
    forced_count = int(mode == "intervention")
    _need(_same(value["override_count"], forced_count), "unique_override_count")
    crops = value["crop_events"]
    _need(type(crops) is list and len(crops) <= 8, "crop_callback_count")
    for index, event in enumerate(crops):
        _keys(event, {"index", "sequence", "logits_values", "forward_calls", "record_sha256", "status",
                     "forward_calls_completed", "logits_values_completed"},
              "crop_callback_keys")
        _need(event["status"] == "complete" and _same(event["index"], index) and _integer(event["sequence"], 1)
              and _same(event["logits_values"], 5) and _same(event["forward_calls"], 1)
              and _same(event["logits_values_completed"], 5)
              and _same(event["forward_calls_completed"], 1)
              and _sha(event["record_sha256"]), "crop_scoring_accounting")
    counts = {
        "models": {"task_forward_calls": len(events) - forced_count,
                   "task_forward_calls_completed": len(events) - forced_count,
                   "task_scored_candidates": sum(e["task_scored_candidates"] for e in events),
                   "task_scored_candidates_completed": sum(e["task_scored_candidates"] for e in events),
                   "crop_forward_calls": len(crops), "crop_forward_calls_completed": len(crops),
                   "crop_logits_values": len(crops) * 5, "crop_logits_values_completed": len(crops) * 5},
        "controller": {"real_calls": 1, "real_calls_completed": 1,
                       "baseline_calls": int(mode == "baseline"),
                       "baseline_calls_completed": int(mode == "baseline"),
                       "intervention_calls": int(mode == "intervention"),
                       "intervention_calls_completed": int(mode == "intervention"),
                       "forced_choices": forced_count, "callbacks": len(events),
                       "callbacks_completed": len(events), "task_candidate_encodings": len(events),
                       "task_candidate_encodings_completed": len(events),
                       "generated_candidate_objects": sum(e["candidate_count"] for e in events)},
    }
    _keys(value["counts"], counts, "call_count_groups")
    for group, expected in counts.items():
        _keys(value["counts"][group], expected, "call_count_keys")
        _need(all(_same(value["counts"][group][key], count) for key, count in expected.items()),
              "call_event_count_reconciliation")
    return maps, counts


def _pair(value, selected, sample_index):
    _keys(value, {"schema", "status", "sample", "historical", "baseline", "intervention", "comparison"},
          "pair_keys")
    _need(value["schema"] == "first-water-unit-request-pair-v1" and value["status"] == "complete", "pair_schema")
    _keys(value["sample"], selected, "pair_sample_keys")
    _need(all(_same(value["sample"][key], item) for key, item in selected.items()),
          "pair_sample_binding")
    historical = value["historical"]
    _keys(historical, {"unit_actions", "trace_sha256", "events", "crop_events"}, "historical_keys")
    history_units = _vector(historical["unit_actions"])
    units, target = len(history_units), selected["target_index"]
    _need(_sha(historical["trace_sha256"]), "historical_trace_hash")
    baseline, intervention = value["baseline"], value["intervention"]
    baseline_maps, baseline_counts = _call(baseline, "baseline", sample_index, units, target)
    intervention_maps, intervention_counts = _call(intervention, "intervention", sample_index + 6, units, target)
    _seconds(baseline["active_seconds"] + intervention["active_seconds"], 20, "sample_pair_active_limit")
    _need(_same(history_units, baseline["unit_actions"]), "historical_baseline_vector_identity")
    _need(historical["trace_sha256"] == baseline["trace_sha256"], "historical_complete_trace_identity")
    previous = historical["events"]
    _need(type(previous) is list and len(previous) == len(baseline["events"]), "historical_callback_count")
    for index, (saved, actual) in enumerate(zip(previous, baseline["events"])):
        _keys(saved, {"index", "inputs_sha256", "record_sha256", "chosen_id", "teacher_id"},
              "historical_callback_keys")
        _need(_same(saved["index"], index)
              and all(_same(saved[key], actual[key]) for key in saved), "historical_callback_identity")
        _need(index >= target or saved["chosen_id"] == saved["teacher_id"], "earliest_bound_disagreement")
    saved_crops = historical["crop_events"]
    _need(type(saved_crops) is list and len(saved_crops) == len(baseline["crop_events"]),
          "historical_crop_count")
    for index, (saved, actual) in enumerate(zip(saved_crops, baseline["crop_events"])):
        _keys(saved, {"index", "record_sha256"}, "historical_crop_keys")
        _need(_same(saved["index"], index) and saved["record_sha256"] == actual["record_sha256"],
              "historical_crop_identity")
    for index in range(target + 1):
        left, right = baseline["events"][index], intervention["events"][index]
        for key in ("inputs_sha256", "context_sha256", "candidate_ids_sha256", "teacher_id"):
            _need(_same(left[key], right[key]), "forced_prefix_context_identity")
        _need(_same(left["legal_candidates"], right["legal_candidates"]), "forced_prefix_candidate_identity")
        if index < target:
            _need(left["chosen_id"] == right["chosen_id"]
                  and left["record_sha256"] == right["record_sha256"], "forced_prefix_choice_identity")
    old, new = baseline["events"][target], intervention["events"][target]
    _need(new["chosen_id"] == new["teacher_id"] == old["teacher_id"]
          and new["chosen_id"] != old["chosen_id"], "forced_current_teacher_identity")
    student_choice = baseline_maps[target][old["chosen_id"]]
    teacher_choice = intervention_maps[target][new["chosen_id"]]
    _need(student_choice[3] == teacher_choice[3] == "WATER"
          and student_choice[0] != teacher_choice[0]
          and student_choice[1:3] != teacher_choice[1:3], "bound_water_worker_target_disagreement")
    left_units, right_units = baseline["unit_actions"], intervention["unit_actions"]
    changed = [index for index in range(units) if not _same(left_units[index], right_units[index])]
    expected = {"historical_baseline_equal": True, "baseline_returned_equal": True,
                "intervention_returned_equal": True, "baseline_intervention_equal": not changed,
                "changed_unit_indices": changed, "changed_unit_count": len(changed),
                "vector_lengths": {"historical": units, "baseline": units, "intervention": units},
                "first_difference": _difference(left_units, right_units)}
    _keys(value["comparison"], expected, "comparison_keys")
    for key, item in expected.items():
        if key == "vector_lengths":
            _keys(value["comparison"][key], item, "comparison_length_keys")
            _need(all(_same(value["comparison"][key][name], count) for name, count in item.items()),
                  "comparison_vector_lengths")
        else:
            _need(_same(value["comparison"][key], item), "independent_comparison_identity")
    operations = {mode: {op: 0 for op in REQUEST_OPS} for mode in ("baseline", "intervention")}
    for index in changed:
        operations["baseline"][left_units[index][0]] += 1
        operations["intervention"][right_units[index][0]] += 1
    safe = {"game": selected["game"], "historical_baseline_equal": True,
            "baseline_returned_equal": True, "intervention_returned_equal": True,
            "baseline_intervention_equal": not changed, "changed_unit_count": len(changed),
            "changed_request_operation_counts": operations,
            "baseline_callback_count": len(baseline["events"]),
            "intervention_callback_count": len(intervention["events"]),
            "resumed_student_callback_count": len(intervention["events"]) - target - 1,
            "override_count": 1}
    return safe, (baseline, intervention), (baseline_counts, intervention_counts)


def _sequences(calls):
    ordered = sorted(calls, key=lambda call: call["call_index"])
    _need([call["call_index"] for call in ordered] == list(range(12)), "all_baselines_precede_interventions")
    events = [event for call in ordered for event in call["events"]]
    for key in ("callback_sequence", "encoding_sequence"):
        _need([event[key] for event in events] == list(range(1, len(events) + 1)),
              "fresh_global_callback_enumeration_sequence")
    scores = [event["score_sequence"] for event in events if event["selector"] == "student"]
    _need(scores == list(range(1, len(scores) + 1)), "fresh_global_student_score_sequence")
    crops = [event["sequence"] for call in ordered for event in call["crop_events"]]
    _need(crops == list(range(1, len(crops) + 1)), "fresh_global_crop_score_sequence")


def _primary(value, selected, primary_bytes):
    _keys(value, {"schema", "status", "baseline_gate_passed", "completed_baselines",
                  "completed_interventions", "pairs", "protocol_checks", "ledger", "timing",
                  "phase", "failure_code", "safe_sample_results", "output"}, "primary_keys")
    _need(value["schema"] == "first-water-unit-request-primary-v1"
          and value["status"] == "complete" and value["baseline_gate_passed"] is True
          and _same(value["completed_baselines"], 6) and _same(value["completed_interventions"], 6)
          and value["failure_code"] is None and value["phase"] == "completed", "primary_completion")
    checks = value["protocol_checks"]
    _keys(checks, {"cases", "fake_callback_events", "fake_candidates", "real_model_calls",
                   "real_controller_calls", "official_transitions", "passed"}, "protocol_check_keys")
    _need(checks["passed"] is True and _integer(checks["cases"], 1) and checks["cases"] <= 4
          and _integer(checks["fake_callback_events"]) and checks["fake_callback_events"] <= 16
          and _integer(checks["fake_candidates"]) and checks["fake_candidates"] <= 32,
          "protocol_check_completion")
    _need(all(_same(checks[key], 0) for key in ("real_model_calls", "real_controller_calls", "official_transitions")),
          "protocol_no_real_work")
    timing = value["timing"]
    _keys(timing, {"primary_started_perf", "deadline_perf", "elapsed_seconds", "model_load_seconds"},
          "primary_timing_keys")
    _seconds(timing["elapsed_seconds"], 120, "primary_time_limit")
    _seconds(timing["model_load_seconds"], 20, "model_load_time_limit")
    _need(all(type(timing[key]) in (int, float) and math.isfinite(timing[key])
              for key in ("primary_started_perf", "deadline_perf"))
          and 0 < timing["deadline_perf"] - timing["primary_started_perf"] <= 120,
          "primary_deadline_interval")
    pairs = value["pairs"]
    _need(type(pairs) is list and len(pairs) == 6, "primary_pair_count")
    total_bytes = 0
    for binding, sample in zip(pairs, selected):
        _keys(binding, {"game", "path", "sha256", "bytes", "status"}, "pair_binding_keys")
        _need(binding["game"] == sample["game"]
              and binding["path"] == f"pairs/{sample['game']}.json"
              and binding["status"] == "complete" and _sha(binding["sha256"])
              and _integer(binding["bytes"], 1) and binding["bytes"] <= MAX_PAIR_BYTES,
              "pair_binding_identity")
        total_bytes += binding["bytes"]
    _need(primary_bytes + total_bytes <= MAX_DECODED_BYTES, "aggregate_receipt_bytes_limit")
    _keys(value["output"], {"pair_files_written", "pair_bytes_written", "primary_receipt_bytes"},
          "primary_output_keys")
    _need(_same(value["output"]["pair_files_written"], 6)
          and _same(value["output"]["pair_bytes_written"], total_bytes)
          and _same(value["output"]["primary_receipt_bytes"], primary_bytes), "primary_output_reconciliation")
    return pairs


def _ledger(ledger, cells, calls, selected, cfg):
    _keys(ledger, {"source", "model_io", "models", "controller", "unknown_accounting"}, "ledger_groups")
    totals = {group: {key: sum(cell[group][key] for cell in cells) for key in cells[0][group]}
              for group in ("models", "controller")}
    extras = {"models": {"training_calls": 0, "numeric_cache_reuse": False},
              "controller": {"full_teacher_policy_calls": 0, "official_transitions": 0, "new_games": 0}}
    for group in totals:
        expected = {**totals[group], **extras[group]}
        _keys(ledger[group], expected, "ledger_execution_count_keys")
        _need(all(_same(ledger[group][key], value) for key, value in expected.items()),
              "ledger_callback_count_reconciliation")
    for key, limit in {"task_forward_calls": 90, "task_scored_candidates": 46080,
                       "crop_forward_calls": 96, "crop_logits_values": 480}.items():
        _need(totals["models"][key] <= limit, "model_forward_budget")
    for key, limit in {"real_calls": 12, "baseline_calls": 6, "intervention_calls": 6,
                       "callbacks": 96, "task_candidate_encodings": 96,
                       "generated_candidate_objects": 49152, "forced_choices": 6}.items():
        _need(totals["controller"][key] <= limit, "controller_budget")
    unknown = ledger["unknown_accounting"]
    _keys(unknown, {"planned_prefix", "failures"}, "primary_unknown_keys")
    _need(type(unknown["failures"]) is list and not unknown["failures"], "unresolved_primary_unknowns")
    planned = unknown["planned_prefix"]
    _need(type(planned) is list and len(planned) == 6, "planned_prefix_unknown_count")
    for item, sample in zip(planned, selected):
        expected = {"component": "source", "game": sample["game"],
                    "quantity": "gzip_internal_unreturned_payload_bytes", "value": None,
                    "reason": "intentional_16_line_prefix_without_eof_probe"}
        _keys(item, expected, "planned_prefix_unknown_keys")
        _need(all(_same(item[key], value) for key, value in expected.items()), "planned_prefix_unknown_identity")
    source_limits = {
        "gzip_opens": 6, "prefix_lines_returned": 96, "target_json_decode_attempts": 6,
        "target_json_turns_decoded": 6, "skipped_prefix_lines_json_decoded": 0,
        "compressed_bytes_read": 12582912, "returned_payload_bytes": 50331648,
        "target_json_bytes": 12582912, "target_task_records": 48,
        "target_candidate_objects": 24576, "target_candidate_x_arrays": 24576,
        "target_shared_x_arrays": 48, "target_observation_objects": 48,
        "target_configuration_objects": 48, "target_crop_records": 48,
        "target_crop_x_arrays": 48, "target_crop_logits_arrays": 48,
        "reread_passes": 0, "complete_trace_hash_passes": 0, "raw_replay_reads": 0,
        "existing_compact_or_first_witness_reads": 0,
    }
    other = ("source_opens_started", "source_opens_completed", "compressed_read_calls_started",
             "compressed_read_calls_completed", "readline_calls_started", "readline_calls_completed",
             "source_closes_started", "source_closes_completed", "gzip_closes_started",
             "gzip_closes_completed", "sources_completed")
    source_keys = set(source_limits) | set(other)
    source = ledger["source"]
    _keys(source, source_keys | {"status", "per_source", "whole_trace_hash_proof", "elapsed_seconds"},
          "source_ledger_keys")
    _need(source["status"] == "completed" and source["whole_trace_hash_proof"] == "inherited_only",
          "source_completion")
    _seconds(source["elapsed_seconds"], 120, "source_time_limit")
    _need(all(_integer(source[key]) for key in source_keys), "source_counter_types")
    _need(all(source[key] <= limit for key, limit in source_limits.items()), "source_budget")
    per_source = source["per_source"]
    _need(type(per_source) is list and len(per_source) == 6, "source_inventory")
    baselines = sorted((call for call in calls if call["call_index"] < 6), key=lambda call: call["call_index"])
    for item, sample, call in zip(per_source, cfg["samples"], baselines):
        metadata = {"game", "source_path", "status", "gzip_eof_verified", "internal_unreturned_payload_bytes",
                    "inherited_whole_trace_sha256", "whole_trace_hash_proof", "source_stat_after_verified",
                    "target_line_sha256"}
        _keys(item, source_keys | metadata, "per_source_keys")
        _need(all(_integer(item[key]) for key in source_keys), "per_source_counter_types")
        _need(item["game"] == sample["game"] and item["source_path"] == sample["source_path"]
              and item["target_line_sha256"] == sample["source_line_sha256"]
              and item["inherited_whole_trace_sha256"] == sample["inherited_whole_trace_sha256"]
              and item["whole_trace_hash_proof"] == "inherited_only" and item["status"] == "completed"
              and item["gzip_eof_verified"] is False and item["internal_unreturned_payload_bytes"] is None
              and item["source_stat_after_verified"] is True, "per_source_identity")
        exact = {"gzip_opens": 1, "prefix_lines_returned": 16,
                 "target_json_decode_attempts": 1, "target_json_turns_decoded": 1,
                 "source_opens_started": 1, "source_opens_completed": 1,
                 "readline_calls_started": 16, "readline_calls_completed": 16,
                 "source_closes_started": 1, "source_closes_completed": 1,
                 "gzip_closes_started": 1, "gzip_closes_completed": 1, "sources_completed": 1,
                 "target_task_records": len(call["events"]),
                 "target_candidate_objects": sum(event["candidate_count"] for event in call["events"]),
                 "target_candidate_x_arrays": sum(event["candidate_count"] for event in call["events"]),
                 "target_shared_x_arrays": len(call["events"]), "target_observation_objects": len(call["events"]),
                 "target_configuration_objects": len(call["events"]), "target_crop_records": len(call["crop_events"]),
                 "target_crop_x_arrays": len(call["crop_events"]), "target_crop_logits_arrays": len(call["crop_events"])}
        exact.update({key: 0 for key, limit in source_limits.items() if limit == 0})
        _need(all(item[key] == value for key, value in exact.items()), "source_decoded_object_reconciliation")
        _need(0 < item["compressed_bytes_read"] <= min(2097152, sample["source_stat"]["size"])
              and 0 < item["target_json_bytes"] <= 2097152
              and item["target_json_bytes"] <= item["returned_payload_bytes"] <= 8388608
              and 0 < item["compressed_read_calls_started"] == item["compressed_read_calls_completed"],
              "per_source_read_budget")
    _need(all(source[key] == sum(item[key] for item in per_source) for key in source_keys),
          "source_global_reconciliation")
    model_keys = {
        "checkpoint_hash_passes_started", "checkpoint_hash_passes_completed", "model_object_loads_reserved",
        "model_object_loads_started", "model_object_loads_completed", "model_object_loads_failed",
        "np_load_calls_started", "np_load_calls_completed", "model_opens_started", "model_opens_completed",
        "model_read_calls_started", "model_read_calls_completed", "total_model_bytes_returned_by_reads",
        "hash_bytes_returned_by_reads", "np_load_bytes_returned_by_reads", "model_seeks_started",
        "model_seeks_completed", "model_closes_started", "model_closes_completed",
    }
    model = ledger["model_io"]
    _keys(model, model_keys | {"status", "distinct_npz_files", "blocked_operations", "per_model",
                              "patches_restored", "open_handles_remaining", "elapsed_seconds"}, "model_io_keys")
    _need(model["status"] == "completed" and _same(model["distinct_npz_files"], 2)
          and _same(model["blocked_operations"], 0) and model["patches_restored"] is True
          and _same(model["open_handles_remaining"], 0), "model_io_completion")
    _seconds(model["elapsed_seconds"], 20, "model_io_time_limit")
    _need(all(_integer(model[key]) for key in model_keys)
          and model["total_model_bytes_returned_by_reads"] <= 4194304, "model_io_byte_budget")
    _keys(model["per_model"], {"task", "crop"}, "model_io_inventory")
    for name, item in model["per_model"].items():
        _keys(item, model_keys | {"path", "expected_sha256", "actual_sha256", "sha_verified", "status",
                                 "file_identity_checked", "hash_opens_started", "np_load_opens_started"}, "per_model_keys")
        _need(all(_integer(item[key]) for key in model_keys), "per_model_counter_types")
        _need(item["path"] == cfg["models"][name]["path"]
              and item["expected_sha256"] == item["actual_sha256"] == MODEL_SHA[name]
              and item["sha_verified"] is True and item["status"] == "completed"
              and item["file_identity_checked"] is True, "per_model_identity")
        once = ("checkpoint_hash_passes_started", "checkpoint_hash_passes_completed", "model_object_loads_reserved",
                "model_object_loads_started", "model_object_loads_completed", "np_load_calls_started",
                "np_load_calls_completed", "hash_opens_started", "np_load_opens_started")
        twice = ("model_opens_started", "model_opens_completed", "model_closes_started", "model_closes_completed")
        _need(all(_same(item[key], 1) for key in once) and all(item[key] == 2 for key in twice)
              and item["model_object_loads_failed"] == 0
              and 0 < item["model_read_calls_started"] == item["model_read_calls_completed"]
              and item["model_seeks_started"] == item["model_seeks_completed"], "per_model_once_and_closed")
        _need(item["hash_bytes_returned_by_reads"] == cfg["model_protection"][item["path"]]["size"]
              and item["np_load_bytes_returned_by_reads"] > 0
              and item["total_model_bytes_returned_by_reads"] ==
                  item["hash_bytes_returned_by_reads"] + item["np_load_bytes_returned_by_reads"], "model_read_byte_reconciliation")
    _need(all(model[key] == sum(item[key] for item in model["per_model"].values()) for key in model_keys),
          "model_io_global_reconciliation")
    return {"models": totals["models"], "controller": totals["controller"],
            "source": {key: source[key] for key in sorted(source_keys)},
            "model_io": {key: model[key] for key in sorted(model_keys)},
            "intentional_prefix_unknown_count": 6}


def check(out: Path, cfg: dict, deadline: float):
    """Read one primary and six pairs once, with an absolute monotonic deadline."""
    started = time.monotonic()
    _need(type(deadline) in (int, float) and math.isfinite(deadline), "deadline_type")
    cutoff = min(deadline, started + 40.0)
    folder = Path(out) / "check"
    consumption = {
        "primary_receipt_open_attempts": 0, "primary_receipt_files_read": 0,
        "primary_receipt_bytes_read": 0, "pair_receipt_open_attempts": 0,
        "pair_receipt_files_read": 0, "pair_receipt_bytes_read": 0,
        "json_decode_attempts": 0, "json_files_decoded": 0,
        "json_bytes_submitted": 0, "json_bytes_decoded": 0, "pairs_verified": 0,
        "source_reads": 0, "model_reads_or_scores": 0, "controller_calls": 0,
        "old_compact_or_witness_reads": 0, "replay_reads": 0,
    }
    unknown, owned = [], False
    safe_samples, identities = [], []

    def tick():
        _need(time.monotonic() < cutoff, "independent_checker_deadline")

    def read_once(path, kind, limit, expected_sha, expected_bytes=None):
        tick()
        size = path.stat().st_size
        _need(_integer(size, 1) and size <= limit, "receipt_file_size_limit")
        _need(consumption["json_bytes_submitted"] + size <= MAX_DECODED_BYTES,
              "aggregate_receipt_bytes_limit")
        if expected_bytes is not None:
            _need(size == expected_bytes, "receipt_bound_byte_identity")
        consumption[f"{kind}_open_attempts"] += 1
        digest, chunks, returned = hashlib.sha256(), [], 0
        with path.open("rb") as stream:
            consumption[f"{kind}_files_read"] += 1
            while returned < size:
                tick()
                try:
                    block = stream.read(min(65536, size - returned))
                except BaseException:
                    unknown.append(f"{kind}_failed_read_returned_bytes_unknown")
                    raise
                # Charge returned bytes before any malformed/changed/hash checks.
                consumption[f"{kind}_bytes_read"] += len(block)
                returned += len(block)
                digest.update(block)
                chunks.append(block)
                _need(bool(block), "receipt_short_read")
                tick()
            _need(os.fstat(stream.fileno()).st_size == size and path.stat().st_size == size,
                  "receipt_file_size_changed")
        tick()
        _need(digest.hexdigest() == expected_sha, "receipt_hash_identity")
        raw = b"".join(chunks)
        consumption["json_decode_attempts"] += 1
        consumption["json_bytes_submitted"] += len(raw)
        try:
            value = _decode(raw)
        except BaseException:
            unknown.append("failed_json_decode_partial_object_graph_unknown")
            raise
        consumption["json_files_decoded"] += 1
        consumption["json_bytes_decoded"] += len(raw)
        tick()
        return value

    try:
        tick()
        folder.mkdir()
        owned = True
        _write_new(folder / "started.json", {
            "schema": "first-water-unit-request-check-started-v1", "status": "started",
            "checker_seconds_max": 40, "primary_receipt_bytes_max": MAX_PRIMARY_BYTES,
            "pair_receipt_bytes_max": MAX_PAIR_BYTES, "aggregate_decoded_bytes_max": MAX_DECODED_BYTES,
            "passes_per_receipt_max": 1, "consumption": consumption.copy(),
            "unknown_accounting": [], "independence_limit": INDEPENDENCE_LIMIT,
            "missing_terminal_accounting": "unknown; started counters are not terminal consumption",
        })
        tick()
        selected = _configuration(cfg)
        primary = read_once(Path(out) / "primary-receipt.json", "primary_receipt",
                            MAX_PRIMARY_BYTES, cfg["primary_receipt_sha256"])
        bindings = _primary(primary, selected, consumption["primary_receipt_bytes_read"])
        calls, count_cells = [], []
        for sample_index, (sample, binding) in enumerate(zip(selected, bindings)):
            value = read_once(Path(out) / binding["path"], "pair_receipt", MAX_PAIR_BYTES,
                              binding["sha256"], binding["bytes"])
            safe, pair_calls, pair_counts = _pair(value, sample, sample_index)
            safe_samples.append(safe)
            calls.extend(pair_calls)
            count_cells.extend(pair_counts)
            identities.append({"game": sample["game"], "sha256": binding["sha256"],
                               "bytes": binding["bytes"]})
            consumption["pairs_verified"] += 1
            tick()
        _sequences(calls)
        reconciled = _ledger(primary["ledger"], count_cells, calls, selected, cfg)
        reported = primary["safe_sample_results"]
        _need(type(reported) is list and len(reported) == 6, "primary_safe_result_count")
        for item, verified in zip(reported, safe_samples):
            _keys(item, {"game", "unit_vectors_equal", "changed_unit_count"}, "primary_safe_result_keys")
            _need(item["game"] == verified["game"]
                  and _same(item["unit_vectors_equal"], verified["baseline_intervention_equal"])
                  and _same(item["changed_unit_count"], verified["changed_unit_count"]),
                  "primary_safe_result_reconciliation")
        tick()
        _need(consumption["primary_receipt_files_read"] == 1
              and consumption["pair_receipt_files_read"] == 6
              and consumption["json_decode_attempts"] == consumption["json_files_decoded"] == 7
              and consumption["pairs_verified"] == 6 and not unknown, "complete_onepass_consumption")
        different = sum(not item["baseline_intervention_equal"] for item in safe_samples)
        summary = {"schema": "first-water-unit-request-safe-summary-v1", "samples": safe_samples,
                   "equal_samples": 6 - different, "different_samples": different,
                   "all_six_unit_vectors_equal": different == 0, "counts": reconciled,
                   "independence_limit": INDEPENDENCE_LIMIT}
        summary_sha = _write_new(folder / "safe-summary.json", summary)
        tick()
        certificate = {
            "schema": "first-water-unit-request-independent-check-v1", "status": "complete", "passed": True,
            "consumption": consumption.copy(), "unknown_accounting": [],
            "primary_receipt_sha256": cfg["primary_receipt_sha256"], "pairs": identities,
            "safe_summary_sha256": summary_sha, "bound_primary_counts": reconciled,
            "equal_samples": 6 - different, "different_samples": different,
            "elapsed_seconds_before_terminal_write": time.monotonic() - started,
            "independence_limit": INDEPENDENCE_LIMIT,
        }
        tick()
        _write_new(folder / "complete.json", certificate)
        tick()
        return certificate
    except BaseException as error:
        if owned:
            _write_new(folder / "failure.json", {
                "schema": "first-water-unit-request-independent-check-failure-v1", "status": "failed",
                "passed": False, "consumption": consumption.copy(), "unknown_accounting": unknown.copy(),
                "verified_samples": safe_samples,
                "elapsed_seconds_before_terminal_write": time.monotonic() - started,
                "failure": {"type": type(error).__name__,
                            "code": str(error) if isinstance(error, CheckFailure) else "checker_failed"},
                "independence_limit": INDEPENDENCE_LIMIT,
            })
        raise
