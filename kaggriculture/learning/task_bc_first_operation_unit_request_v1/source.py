"""One selected operation-mismatch turn, using the frozen byte/object meters.

Only the selected source line is JSON-decoded. The unchanged old module owns
the low-level meter, compressed reader, and nested-object accounting; this
adapter changes the target/selection checks and the finite prefix length.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from learning.task_bc_first_water_unit_requests_v1.source import (
    CROPS, OPS, SCHEMA_VERSION, UNIT_OPS, _COUNT_LIMITS, _OTHER_COUNTS,
    _SHA, _CompressedReader, _Meter, _charge_children, _finite, _identity,
    _integer, _numeric_array, _require, ordered_hash, schema_hash,
)


GAME = "holdout-1714414373-v1-0-trained"


def _descriptor(candidate):
    # Match the original compact descriptor, including its declared key order.
    return {key: candidate[key] for key in ("id", "unit", "target", "op")}


def _validate(document, sample, cfg, meter):
    """Keep the original saved-trace checks; select the first operation change."""
    _require(isinstance(document, dict) and _finite(document), "invalid or nonfinite target JSON")
    _require(document.get("schema_version") == SCHEMA_VERSION, "target schema differs")
    _require(type(document.get("step")) is int and document["step"] == sample["step"], "target step differs")
    for key in ("load_errors", "inference_errors", "fallbacks"):
        _require(type(document.get(key)) is int and document[key] == 0, "saved failure counter: " + key)
    for key, name in (("task_checkpoint_sha256", "task"), ("checkpoint_sha256", "crop")):
        _require(document.get(key) == cfg["models"][name]["sha256"], "saved model identity differs: " + name)
    records, crops = document.get("records"), document.get("crop_records")
    _require(isinstance(records, list) and 0 < len(records) <= cfg["budget"]["controller"]["callbacks_per_controller_call_max"],
             "invalid saved callback count")
    _require(isinstance(crops, list), "missing crop records")
    reservations = document.get("final_reservations")
    _require(isinstance(reservations, dict), "missing final reservations")
    actions = reservations.get("unit_actions")
    _require(isinstance(actions, list) and actions and all(isinstance(action, list) and action
             and isinstance(action[0], str) and action[0] in UNIT_OPS for action in actions),
             "invalid historical unit vector")
    observation_hash = configuration_hash = None
    first_operation_mismatch = None
    for index, record in enumerate(records):
        meter.check()
        _require(isinstance(record, dict) and record.get("source") == "callback", "invalid saved callback")
        inputs = record.get("inputs")
        _require(isinstance(inputs, dict) and inputs.get("schema_version") == SCHEMA_VERSION,
                 "callback schema differs")
        observation, configuration = inputs.get("observation"), inputs.get("configuration")
        _require(isinstance(observation, dict) and isinstance(configuration, dict), "missing factual input")
        _require(type(observation.get("step")) is int and observation["step"] == sample["step"]
                 and type(observation.get("player")) is int and observation["player"] == sample["side"],
                 "saved observation identity differs")
        current_obs, current_cfg = ordered_hash(observation), ordered_hash(configuration)
        if index == 0:
            observation_hash, configuration_hash = current_obs, current_cfg
            farms = observation.get("farms")
            _require(isinstance(farms, list) and len(farms) == 2, "invalid observed farms")
            own = farms[sample["side"]]
            _require(isinstance(own, dict) and isinstance(own.get("hands"), list), "invalid observed units")
            unit_count = 1 + len(own["hands"])
            _require(0 < unit_count <= cfg["budget"]["controller"]["units_per_observation_max"]
                     and len(actions) == unit_count, "historical unit-vector length differs")
        _require((current_obs, current_cfg) == (observation_hash, configuration_hash),
                 "callbacks have different factual observations/configurations")
        private = observation.get("private")
        _require(isinstance(private, dict) and isinstance(private.get("inventories"), list)
                 and len(private["inventories"]) == unit_count and isinstance(private.get("shed"), dict)
                 and all(isinstance(inv, dict) for inv in private["inventories"]), "invalid observed inventories")
        _require(inputs.get("inventory_item_order") == [list(inv) for inv in private["inventories"]]
                 and inputs.get("shed_item_order") == list(private["shed"]), "saved item order differs")
        names, shared_names, shared = inputs.get("feature_names"), inputs.get("shared_feature_names"), inputs.get("shared_x")
        _require(isinstance(names, list) and len(names) == 98 and all(isinstance(name, str) for name in names)
                 and isinstance(shared_names, list) and len(shared_names) == 275
                 and all(isinstance(name, str) for name in shared_names) and _numeric_array(shared, 275),
                 "saved feature dimensions differ")
        _require(inputs.get("schema_sha256") == schema_hash(names, shared_names), "saved feature schema hash differs")
        candidates = inputs.get("candidates")
        _require(isinstance(candidates, list) and 0 < len(candidates) <= cfg["budget"]["controller"]["candidates_per_callback_max"],
                 "saved candidate count differs")
        lookup = {}
        for candidate in candidates:
            _require(isinstance(candidate, dict), "invalid saved candidate")
            unit, target, op = candidate.get("unit"), candidate.get("target"), candidate.get("op")
            _require(_integer(unit) and unit < unit_count and isinstance(target, list) and len(target) == 2
                     and all(_integer(value) for value in target) and op in OPS, "invalid candidate descriptor")
            identifier = f"u{unit}:x{target[0]}:y{target[1]}:{op}"
            _require(candidate.get("id") == identifier and identifier not in lookup, "invalid candidate identity")
            _require(_numeric_array(candidate.get("x"), 98), "invalid candidate vector")
            lookup[identifier] = candidate
        chosen, teacher = record.get("chosen"), record.get("teacher_label")
        _require(isinstance(chosen, str) and isinstance(teacher, str) and chosen in lookup and teacher in lookup,
                 "saved choice absent from current candidates")
        selected, preferred = lookup[chosen], lookup[teacher]
        _require(all(ordered_hash(record.get(key)) == ordered_hash(selected[key]) for key in ("unit", "target", "op"))
                 and ordered_hash(record.get("unit_request")) == ordered_hash(actions[selected["unit"]]),
                 "saved selected descriptor/request differs")
        if selected["op"] != preferred["op"] and first_operation_mismatch is None:
            first_operation_mismatch = index
        if index == sample["first_callback_index_0_based"]:
            _require(ordered_hash(_descriptor(selected)) == ordered_hash(_descriptor(sample["chosen"]))
                     and ordered_hash(_descriptor(preferred)) == ordered_hash(_descriptor(sample["teacher"])),
                     "selected compact descriptors differ from current candidates")
    _require(first_operation_mismatch == sample["first_callback_index_0_based"],
             "bound first operation callback differs")
    for index, crop in enumerate(crops):
        _require(isinstance(crop, dict) and crop.get("source") == "model" and crop.get("step") == sample["step"]
                 and type(crop.get("decision")) is int and crop["decision"] == index,
                 "saved crop record identity differs")
        _require(_integer(crop.get("unit")) and crop["unit"] < unit_count
                 and _numeric_array(crop.get("x")) and _numeric_array(crop.get("logits"), 5)
                 and crop.get("chosen") in CROPS and crop.get("teacher") in CROPS,
                 "invalid saved crop numeric payload")
        _require(ordered_hash(crop.get("unit_request")) == ordered_hash(actions[crop["unit"]]),
                 "saved crop request differs from unit vector")


def _read_one(root, sample, cfg, meter):
    item = {key: 0 for key in (*_COUNT_LIMITS, *_OTHER_COUNTS)}
    item.update(game=sample["game"], source_path=sample["source_path"], status="started",
                gzip_eof_verified=False, internal_unreturned_payload_bytes=None,
                inherited_whole_trace_sha256=sample["inherited_whole_trace_sha256"],
                whole_trace_hash_proof="inherited_only", source_stat_after_verified=False)
    meter.counts["per_source"].append(item)
    source = (root / sample["source_path"]).resolve()
    _require(source.is_relative_to(root), "source path escapes registered root")
    raw = decoded = None
    completed_prefix = False
    try:
        meter.check()
        before = _identity(source.stat())
        _require(all(before[key] == sample["source_stat"][key] for key in ("size", "mtime_ns")),
                 "registered source stat differs")
        meter._identities[sample["game"]] = before
        _require(meter.counts["gzip_opens"] < meter.limits["gzip_opens_max"], "gzip open budget exhausted")
        meter.charge(item, source_opens_started=1)
        raw = source.open("rb", buffering=0)
        meter.charge(item, source_opens_completed=1)
        _require(_identity(os.fstat(raw.fileno())) == before, "source changed while opening")
        meter.charge(item, gzip_opens=1)
        decoded = gzip.GzipFile(fileobj=_CompressedReader(raw, meter, item), mode="rb")
        for line_index in range(sample["step"] + 1):
            meter.check()
            _require(item["prefix_lines_returned"] < meter.limits["prefix_lines_per_source_max"]
                     and meter.counts["prefix_lines_returned"] < meter.limits["prefix_lines_returned_max"],
                     "prefix line budget exhausted")
            room = min(meter.limits["returned_payload_bytes_per_source_max"] - item["returned_payload_bytes"],
                       meter.limits["returned_payload_bytes_total_max"] - meter.counts["returned_payload_bytes"])
            _require(room > 0, "returned payload budget exhausted")
            meter.charge(item, readline_calls_started=1)
            try:
                line = decoded.readline(min(meter.limits["single_line_bytes_max"], room))
            except BaseException as exc:
                meter.unknown_failure(item, "returned_payload_bytes_in_failed_readline", type(exc).__name__)
                raise
            meter.charge(item, readline_calls_completed=1, returned_payload_bytes=len(line),
                         prefix_lines_returned=int(bool(line)))
            meter.check()
            _require(line and line.endswith(b"\n"), "short prefix or over-limit physical line")
            if line_index != sample["step"]:
                continue  # Earlier physical lines are never JSON-decoded.
            completed_prefix = True
            item["target_line_sha256"] = hashlib.sha256(line).hexdigest()
            _require(item["target_line_sha256"] == sample["source_line_sha256"], "registered target line SHA differs")
            _require(meter.counts["target_json_decode_attempts"] < meter.limits["target_json_turns_decoded_max"]
                     and len(line) <= meter.limits["target_json_bytes_total_max"] - meter.counts["target_json_bytes"],
                     "target JSON budget exhausted")
            meter.charge(item, target_json_decode_attempts=1, target_json_bytes=len(line))
            try:
                document = json.loads(line)
            except BaseException as exc:
                meter.unknown_failure(item, "partially_decoded_nested_objects", type(exc).__name__)
                raise
            meter.charge(item, target_json_turns_decoded=1)
            _charge_children(document, meter, item)
            meter.check()
            _validate(document, sample, cfg, meter)
        item["status"] = "completed"
        return {"sample": sample, "document": document}
    except BaseException as exc:
        item.update(status="failed", failure_type=type(exc).__name__, failure=type(exc).__name__)
        raise
    finally:
        original_error = sys.exc_info()[1]
        cleanup_error = None
        if decoded is not None:
            category = "planned_prefix" if completed_prefix else "failures"
            meter.unknown[category].append(dict(component="source", game=sample["game"],
                quantity="gzip_internal_unreturned_payload_bytes", value=None,
                reason="intentional_selected_prefix_without_eof_probe" if completed_prefix else "failed_before_complete_prefix"))
        for handle, label in ((decoded, "gzip"), (raw, "source")):
            if handle is None:
                continue
            tick = time.monotonic()
            try:
                if label == "source":
                    after = _identity(os.fstat(raw.fileno()))
                    _require(after == meter._identities[sample["game"]] == _identity(source.stat()),
                             "source changed during prefix read")
                    item["source_stat_after_verified"] = True
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            finally:
                meter.charge(item, **{label + "_closes_started": 1})
                try:
                    handle.close()
                    meter.charge(item, **{label + "_closes_completed": 1})
                    _require(time.monotonic() - tick <= cfg["budget"]["per_close_seconds_max"], "source close time exhausted")
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            item.update(status="failed", cleanup_failure_type=type(cleanup_error).__name__,
                        cleanup_failure=type(cleanup_error).__name__)
            if original_error is None:
                raise cleanup_error
            original_error.add_note("Source cleanup/stat verification also failed: " + type(cleanup_error).__name__)


def read_target(root: Path, cfg: dict, selection: dict, deadline: float, ledger: dict) -> dict:
    """Consume the one bound dynamic prefix once; the caller owns its receipt."""
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "invalid source deadline")
    root = Path(root).resolve()
    meter = _Meter(cfg, deadline, ledger)
    tick = time.monotonic()
    try:
        _require(type(selection) is dict and selection.get("schema") == "first-operation-selection-v1"
                 and selection.get("status") == "complete", "selection not complete")
        selected = selection.get("selected")
        _require(type(selected) is dict and selected.get("game") == GAME
                 and type(selected.get("side")) is int and selected["side"] == 0,
                 "selected game or side differs")
        _require(_integer(selected.get("step")) and selected["step"] < 240
                 and _integer(selected.get("target_index"))
                 and selected["target_index"] < cfg["budget"]["controller"]["callbacks_per_controller_call_max"],
                 "selected target exceeds scope")
        _require(selection.get("compact_sha256") == cfg["fixed_selection"]["compact_sha256"]
                 and selection.get("projection_receipt_sha256") == cfg["fixed_selection"]["projection_receipt_sha256"],
                 "selection provenance differs")
        scope = cfg["fixed_input_scope"]
        _require(scope["trace"] == f"artifacts/task-bc-holdout-v1-20261023/traces/{GAME}.jsonl.gz",
                 "fixed source binding differs")
        source_stat = cfg["source_stat"]
        _require(type(source_stat) is dict and source_stat == {
            "size": scope["inherited_trace_size_bytes"], "mtime_ns": scope["inherited_trace_mtime_ns"]},
            "registered source stat binding differs")
        _require(all(_integer(source_stat.get(key), 1) for key in ("size", "mtime_ns")),
                 "missing registered source stat")
        sample = dict(game=GAME, side=0, step=selected["step"],
                      first_callback_index_0_based=selected["target_index"],
                      source_line_sha256=selected.get("source_line_sha256"),
                      source_path=scope["trace"], source_stat=source_stat,
                      inherited_whole_trace_sha256=scope["inherited_trace_sha256"],
                      teacher=selected.get("teacher"), chosen=selected.get("chosen"))
        _require(all(isinstance(sample[key], str) and _SHA.fullmatch(sample[key])
                     for key in ("source_line_sha256", "inherited_whole_trace_sha256")),
                 "invalid inherited or line hash")
        _require(all(type(sample[key]) is dict and set(sample[key]) == {"id", "unit", "target", "op"}
                     for key in ("teacher", "chosen")), "invalid selected descriptors")
        target = _read_one(root, sample, cfg, meter)
        meter.charge(meter.counts["per_source"][-1], sources_completed=1)
        meter.check()
        meter.counts["status"] = "completed"
        return target
    except BaseException as exc:
        meter.counts.update(status="failed", failure_type=type(exc).__name__, failure=type(exc).__name__)
        raise
    finally:
        meter.counts["elapsed_seconds"] = time.monotonic() - tick
