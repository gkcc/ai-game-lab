"""Read only the six registered step-15 lines; never read a whole trace.

Counters measure bytes actually returned by the unbuffered compressed reader
and by gzip.readline.  Gzip's internal, unreturned payload is explicitly unknown.
The caller owns the terminal receipt, including when this module raises.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_choice_v1.features import OPS, SCHEMA_VERSION, schema_hash
from decision.api import CROPS, UNIT_OPS


class SourceStopped(ValueError):
    """The single registered source pass stopped; it must not be retried."""


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COUNT_LIMITS = {
    "gzip_opens": "gzip_opens_max",
    "prefix_lines_returned": "prefix_lines_returned_max",
    "target_json_decode_attempts": "target_json_turns_decoded_max",
    "target_json_turns_decoded": "target_json_turns_decoded_max",
    "skipped_prefix_lines_json_decoded": "skipped_prefix_lines_json_decoded_max",
    "compressed_bytes_read": "compressed_bytes_read_total_max",
    "returned_payload_bytes": "returned_payload_bytes_total_max",
    "target_json_bytes": "target_json_bytes_total_max",
    "target_task_records": "target_task_records_max",
    "target_candidate_objects": "target_candidate_objects_and_x_arrays_max",
    "target_candidate_x_arrays": "target_candidate_objects_and_x_arrays_max",
    "target_shared_x_arrays": "target_shared_x_observation_and_configuration_objects_max_each",
    "target_observation_objects": "target_shared_x_observation_and_configuration_objects_max_each",
    "target_configuration_objects": "target_shared_x_observation_and_configuration_objects_max_each",
    "target_crop_records": "target_crop_records_and_numeric_payloads_max",
    "target_crop_x_arrays": "target_crop_records_and_numeric_payloads_max",
    "target_crop_logits_arrays": "target_crop_records_and_numeric_payloads_max",
    "reread_passes": "reread_passes_max",
    "complete_trace_hash_passes": "complete_trace_hash_passes_max",
    "raw_replay_reads": "raw_replay_reads_max",
    "existing_compact_or_first_witness_reads": "existing_compact_or_first_witness_reads_max",
}
_OTHER_COUNTS = (
    "source_opens_started", "source_opens_completed", "compressed_read_calls_started",
    "compressed_read_calls_completed", "readline_calls_started", "readline_calls_completed",
    "source_closes_started", "source_closes_completed", "gzip_closes_started",
    "gzip_closes_completed", "sources_completed",
)


def _require(condition, message):
    if not condition:
        raise SourceStopped(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _identity(stat):
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def _finite(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(type(key) is str and _finite(item) for key, item in value.items())
    return False


def _numeric_array(value, length=None):
    return (isinstance(value, list) and (length is None or len(value) == length)
            and all(type(item) in (int, float) and math.isfinite(item) for item in value))


class _Meter:
    def __init__(self, cfg, deadline, ledger):
        _require("source" not in ledger, "source pass already registered; no retry")
        self.limits = cfg["budget"]["source"]
        _require(all(_integer(self.limits.get(key)) for key in set(_COUNT_LIMITS.values())),
                 "invalid source counter limits")
        self.deadline = deadline
        self.counts = {key: 0 for key in (*_COUNT_LIMITS, *_OTHER_COUNTS)}
        self.counts.update(status="started", per_source=[], whole_trace_hash_proof="inherited_only")
        ledger["source"] = self.counts
        self.unknown = ledger.setdefault("unknown_accounting", {"planned_prefix": [], "failures": []})
        _require(isinstance(self.unknown, dict) and all(
            isinstance(self.unknown.get(key), list) for key in ("planned_prefix", "failures")),
            "invalid unknown-accounting ledger")
        self._identities = {}  # File-system identities never enter the receipt.

    def charge(self, item=None, **amounts):
        # No checks in this method: decoded children must all be consumed first.
        for key, value in amounts.items():
            self.counts[key] += value
            if item is not None:
                item[key] = item.get(key, 0) + value

    def check(self):
        _require(time.monotonic() < self.deadline, "source deadline exhausted")
        for key, limit in _COUNT_LIMITS.items():
            _require(self.counts[key] <= self.limits[limit], "source budget exceeded: " + key)

    def unknown_failure(self, item, quantity, reason, **detail):
        self.unknown["failures"].append(dict(component="source", game=item["game"],
            quantity=quantity, value=None, reason=reason, **detail))


class _CompressedReader:
    def __init__(self, raw, meter, item):
        self.raw, self.meter, self.item = raw, meter, item

    def read(self, size=-1):
        meter, item = self.meter, self.item
        meter.check()
        room = min(meter.limits["compressed_bytes_read_per_source_max"] - item["compressed_bytes_read"],
                   meter.limits["compressed_bytes_read_total_max"] - meter.counts["compressed_bytes_read"])
        if size == 0:
            return b""
        _require(room > 0, "compressed byte budget exhausted")
        amount = min(room, 64 * 1024 if size < 0 else size)
        meter.charge(item, compressed_read_calls_started=1)
        try:
            data = self.raw.read(amount)
        except BaseException as exc:
            meter.unknown_failure(item, "compressed_bytes_in_failed_read", type(exc).__name__,
                                  requested_bytes=amount)
            raise
        meter.charge(item, compressed_read_calls_completed=1, compressed_bytes_read=len(data))
        meter.check()
        return data


def _charge_children(document, meter, item):
    """Consume all known decoded children before schema or budget validation."""
    def malformed(reason):
        meter.unknown_failure(item, "decoded_nested_objects", reason)

    if not isinstance(document, dict):
        malformed("decoded_root_not_object")
        return
    records = document.get("records")
    if not isinstance(records, list):
        malformed("decoded_records_not_list")
    else:
        for record in records:
            if not isinstance(record, dict):
                malformed("decoded_task_record_not_object")
                continue
            meter.charge(item, target_task_records=1)
            inputs = record.get("inputs")
            if not isinstance(inputs, dict):
                malformed("decoded_inputs_not_object")
                continue
            for key, counter, kind in (
                ("shared_x", "target_shared_x_arrays", list),
                ("observation", "target_observation_objects", dict),
                ("configuration", "target_configuration_objects", dict),
            ):
                if isinstance(inputs.get(key), kind):
                    meter.charge(item, **{counter: 1})
                else:
                    malformed("invalid_decoded_" + key)
            candidates = inputs.get("candidates")
            if not isinstance(candidates, list):
                malformed("decoded_candidates_not_list")
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    malformed("decoded_candidate_not_object")
                    continue
                meter.charge(item, target_candidate_objects=1)
                if isinstance(candidate.get("x"), list):
                    meter.charge(item, target_candidate_x_arrays=1)
                else:
                    malformed("decoded_candidate_x_not_array")
    crops = document.get("crop_records")
    if not isinstance(crops, list):
        malformed("decoded_crop_records_not_list")
    else:
        for record in crops:
            if not isinstance(record, dict):
                malformed("decoded_crop_record_not_object")
                continue
            meter.charge(item, target_crop_records=1)
            for key, counter in (("x", "target_crop_x_arrays"), ("logits", "target_crop_logits_arrays")):
                if isinstance(record.get(key), list):
                    meter.charge(item, **{counter: 1})
                else:
                    malformed("decoded_crop_" + key + "_not_array")


def _validate(document, sample, cfg, meter):
    _require(isinstance(document, dict) and _finite(document), "invalid or nonfinite target JSON")
    _require(document.get("schema_version") == SCHEMA_VERSION, "target schema differs")
    _require(type(document.get("step")) is int and document["step"] == 15, "target step differs")
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
    first_mismatch = None
    for index, record in enumerate(records):
        meter.check()
        _require(isinstance(record, dict) and record.get("source") == "callback", "invalid saved callback")
        inputs = record.get("inputs")
        _require(isinstance(inputs, dict) and inputs.get("schema_version") == SCHEMA_VERSION,
                 "callback schema differs")
        observation, configuration = inputs.get("observation"), inputs.get("configuration")
        _require(isinstance(observation, dict) and isinstance(configuration, dict), "missing factual input")
        _require(type(observation.get("step")) is int and observation["step"] == 15
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
        selected = lookup[chosen]
        _require(all(ordered_hash(record.get(key)) == ordered_hash(selected[key]) for key in ("unit", "target", "op"))
                 and ordered_hash(record.get("unit_request")) == ordered_hash(actions[selected["unit"]]),
                 "saved selected descriptor/request differs")
        if chosen != teacher and first_mismatch is None:
            first_mismatch = index
            preferred = lookup[teacher]
            _require(selected["op"] == preferred["op"] == "WATER"
                     and selected["unit"] != preferred["unit"] and selected["target"] != preferred["target"],
                     "bound first divergence is not WATER with distinct worker and target")
    _require(first_mismatch == sample["first_callback_index_0_based"], "bound first callback differs")
    for index, crop in enumerate(crops):
        _require(isinstance(crop, dict) and crop.get("source") == "model" and crop.get("step") == 15
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
        for line_index in range(16):
            meter.check()
            _require(item["prefix_lines_returned"] < meter.limits["prefix_lines_per_source_max"]
                     and meter.counts["prefix_lines_returned"] < meter.limits["prefix_lines_returned_max"],
                     "prefix line budget exhausted")
            room = min(meter.limits["returned_payload_bytes_per_source_max"] - item["returned_payload_bytes"],
                       meter.limits["returned_payload_bytes_total_max"] - meter.counts["returned_payload_bytes"])
            _require(room > 0, "returned payload budget exhausted")
            meter.charge(item, readline_calls_started=1)
            line = decoded.readline(min(meter.limits["single_line_bytes_max"], room))
            meter.charge(item, readline_calls_completed=1, returned_payload_bytes=len(line),
                         prefix_lines_returned=int(bool(line)))
            meter.check()
            _require(line and line.endswith(b"\n"), "short prefix or over-limit physical line")
            if line_index != 15:
                continue  # Deliberately no JSON decoder for the first 15 lines.
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
                reason="intentional_16_line_prefix_without_eof_probe" if completed_prefix else "failed_before_complete_prefix"))
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
            original_error.add_note("Source cleanup/stat verification also failed: " + str(cleanup_error))


def read_targets(root: Path, cfg: dict, deadline: float, ledger: dict) -> list[dict]:
    """Consume six source prefixes once; retain every counter on any exception."""
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "invalid source deadline")
    root = Path(root).resolve()
    samples = cfg["samples"]
    _require(isinstance(samples, list) and len(samples) == 6, "exactly six fixed samples required")
    expected = [(opponent, side) for opponent in ("v1", "market", "round2") for side in (0, 1)]
    _require([(sample["opponent"], sample["side"]) for sample in samples] == expected, "fixed sample order differs")
    for sample in samples:
        game = f"holdout-1714414373-{sample['opponent']}-{sample['side']}-trained"
        _require(sample["game"] == game and sample["step"] == 15
                 and sample["source_path"] == f"artifacts/task-bc-holdout-v1-20261023/traces/{game}.jsonl.gz"
                 and sample["first_callback_index_0_based"] == (0 if sample["opponent"] == "round2" else 1),
                 "fixed sample binding differs")
        _require(all(isinstance(sample.get(key), str) and _SHA.fullmatch(sample[key]) for key in
                     ("source_line_sha256", "inherited_whole_trace_sha256")), "invalid inherited or line hash")
        _require(isinstance(sample.get("source_stat"), dict) and all(
            _integer(sample["source_stat"].get(key), 1) for key in ("size", "mtime_ns")), "missing registered source stat")
    meter = _Meter(cfg, deadline, ledger)
    tick = time.monotonic()
    try:
        targets = []
        for sample in samples:
            targets.append(_read_one(root, sample, cfg, meter))
            meter.charge(meter.counts["per_source"][-1], sources_completed=1)
        meter.check()
        meter.counts["status"] = "completed"
        return targets
    except BaseException as exc:
        meter.counts.update(status="failed", failure_type=type(exc).__name__, failure=type(exc).__name__)
        raise
    finally:
        meter.counts["elapsed_seconds"] = time.monotonic() - tick
