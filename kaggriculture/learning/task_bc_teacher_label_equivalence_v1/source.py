"""The single preregistered training case, with one compressed pass per file.

This module has no project imports or import-time I/O. Its charge-before-check
meters and finally-owned closes follow the previous operation-request reader.
JSON inventory includes unused values. Prefix gzip read-ahead is unknown, not
zero; after closing gzip, the same raw reader hashes its compressed remainder.
The caller must retain ``ledger`` even when read_case raises. No retry exists.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time


class SourceStopped(ValueError):
    """A fixed non-private failure code; the original pass must not be retried."""


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = ("index", "inputs", "labels", "replay")
_COUNTERS = (
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
) + tuple(kind + suffix for kind in _KINDS for suffix in
          ("_json_decodes_started", "_json_decodes_completed"))


def _need(condition, code):
    if not condition:
        raise SourceStopped(code)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _identity(value):
    # Device/inode values remain in memory; receipts contain only size/mtime.
    return (value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino)


def _failure_code(error):
    return str(error) if isinstance(error, SourceStopped) else type(error).__name__


class _Meter:
    def __init__(self, cfg, deadline, ledger):
        _need("source" not in ledger, "source_already_started_no_retry")
        _need(type(deadline) in (int, float) and math.isfinite(deadline), "invalid_deadline")
        self.limits, self.deadline = cfg["budget"], deadline
        self.counts = dict.fromkeys(_COUNTERS, 0)
        self.counts.update(status="started", per_source=[], index_prefix=[],
                           unknown_accounting={"planned_prefix": [], "failures": []})
        ledger["source"] = self.counts

    def tick(self):
        _need(time.monotonic() < self.deadline, "source_deadline")

    def charge(self, item=None, **amounts):
        # All actual returned/decoded work is charged before rejecting a limit.
        for key, amount in amounts.items():
            self.counts[key] += amount
            if item is not None:
                item[key] += amount

    def check(self):
        self.tick()
        pairs = (
            ("raw_opens_started", "original_compressed_opens_max"),
            ("raw_bytes_returned", "original_compressed_bytes_total_exact_if_all_completed"),
            ("returned_payload_bytes", "source_returned_payload_bytes_total_max"),
            ("index_json_decodes_started", "index_prefix_json_rows_max"),
            ("inputs_json_decodes_started", "target_input_json_rows_max"),
            ("labels_json_decodes_started", "target_label_json_rows_max"),
            ("replay_json_decodes_started", "replay_json_decodes_max"),
            ("decoded_replay_states", "full_replay_states_decoded_max"),
            ("decoded_replay_agent_states", "replay_agent_states_decoded_max"),
            ("decoded_input_candidate_entries", "candidates_per_callback_max"),
            ("raw_reread_passes", "raw_data_reread_passes"),
        )
        for counter, limit in pairs:
            _need(self.counts[counter] <= self.limits[limit], "source_limit_" + counter)

    def start(self, kind, metadata):
        item = dict.fromkeys(_COUNTERS, 0)
        item.update(kind=kind, source_path=metadata["path"], status="started",
                    registered_bytes=metadata["bytes"],
                    inherited_sha256=metadata["sha256_inherited"],
                    raw_sha256=None, raw_sha256_verified=False,
                    source_stat_before_verified=False, source_stat_after_verified=False,
                    gzip_eof_verified=False, internal_unreturned_payload_bytes=None,
                    prefix_stop_reason=None)
        self.counts["per_source"].append(item)
        self.charge(item, sources_started=1)
        return item

    def unknown(self, item, quantity, reason, planned=False, **detail):
        category = "planned_prefix" if planned else "failures"
        self.counts["unknown_accounting"][category].append(
            dict(kind=item["kind"], quantity=quantity, value=None, reason=reason, **detail))


class _CompressedReader:
    """A non-seeking, bounded raw reader; every returned byte is hashed once."""
    def __init__(self, raw, meter, item):
        self.raw, self.meter, self.item = raw, meter, item
        self.digest = hashlib.sha256()
        self.draining = False

    def read(self, size=-1):
        meter, item = self.meter, self.item
        meter.check()
        if size == 0:
            return b""
        room = item["registered_bytes"] - item["raw_bytes_returned"]
        if room == 0:
            # The unchanged size/stat binding supplies this raw boundary. No
            # fifth open, seek, second hash pass, or extra physical EOF read.
            meter.charge(item, raw_bound_eof_returns=1)
            return b""
        _need(room > 0, "raw_registered_byte_limit")
        amount = min(room, 65536, size if size >= 0 else 65536)
        meter.charge(item, raw_read_calls_started=1)
        if self.draining:
            meter.charge(item, raw_drain_read_calls_started=1)
        try:
            data = self.raw.read(amount)
        except BaseException as error:
            meter.unknown(item, "raw_bytes_in_failed_read", type(error).__name__,
                          requested_bytes=amount)
            raise
        meter.charge(item, raw_read_calls_completed=1, raw_bytes_returned=len(data))
        if self.draining:
            meter.charge(item, raw_drain_read_calls_completed=1, raw_drain_bytes_returned=len(data))
        try:
            self.digest.update(data)
        except BaseException as error:
            meter.unknown(item, "raw_bytes_hashed_in_failed_update", type(error).__name__)
            raise
        meter.charge(item, raw_bytes_hashed=len(data))
        _need(0 < len(data) <= amount, "raw_short_or_empty_read")
        meter.check()
        return data

    def drain(self):
        # The caller closes GzipFile before reaching this method.
        _need(self.item["gzip_closes_completed"] == 1, "raw_drain_before_gzip_close")
        self.draining = True
        while self.item["raw_bytes_returned"] < self.item["registered_bytes"]:
            self.read(65536)
        self.meter.check()


def _payload(gz, meter, item, *, line=False):
    meter.check()
    room = meter.limits["source_returned_payload_bytes_total_max"] - meter.counts["returned_payload_bytes"]
    if item["kind"] == "replay":
        room = min(room, meter.limits["replay_payload_bytes_max"] - item["returned_payload_bytes"])
    _need(room > 0, "payload_limit_before_eof")
    amount = room if line else min(room, 65536)
    if line and item["kind"] == "inputs":
        amount = min(amount, meter.limits["per_input_json_line_bytes_max"])
    counter = "gzip_readline_calls_" if line else "gzip_read_calls_"
    meter.charge(item, **{counter + "started": 1})
    try:
        data = gz.readline(amount) if line else gz.read(amount)
    except BaseException as error:
        meter.unknown(item, "payload_bytes_in_failed_gzip_call", type(error).__name__,
                      requested_bytes=amount)
        raise
    meter.charge(item, returned_payload_bytes=len(data), **{counter + "completed": 1})
    if not data:
        item.update(gzip_eof_verified=True, internal_unreturned_payload_bytes=0)
    elif line:
        meter.charge(item, **{"prefix_lines_returned" if data.endswith(b"\n")
                             else "partial_prefix_lines_returned": 1})
    meter.check()
    if line and data:
        _need(data.endswith(b"\n"), "unterminated_or_over_limit_source_line")
    return data


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _constant(_value):
    raise SourceStopped("nonfinite_json_constant")


def _inventory(document, meter, item):
    """Count every decoded JSON value, including all unused replay payloads."""
    # Iterator stack bounds auxiliary space by nesting depth, not replay width.
    stack, nonfinite, visited = [iter((document,))], False, 0
    while stack:
        try:
            value = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        meter.charge(item, decoded_json_values=1)
        visited += 1
        if type(value) is dict:
            meter.charge(item, decoded_mapping_objects=1, decoded_mapping_keys=len(value))
            stack.append(iter(value.values()))
        elif type(value) is list:
            meter.charge(item, decoded_list_objects=1)
            stack.append(iter(value))
        else:
            meter.charge(item, decoded_scalar_values=1)
            nonfinite |= type(value) is float and not math.isfinite(value)
        if visited % 2048 == 0:
            meter.tick()
    meter.tick()
    _need(not nonfinite, "nonfinite_decoded_number")


def _named_inventory(document, meter, item):
    if type(document) is not dict:
        return
    if item["kind"] == "replay":
        steps = document.get("steps")
        if type(steps) is list:
            meter.charge(item, decoded_replay_states=len(steps),
                         decoded_replay_agent_states=sum(len(state) for state in steps if type(state) is list))
        if type(document.get("configuration")) is dict:
            meter.charge(item, decoded_configuration_objects=1)
    if item["kind"] == "inputs":
        candidates = document.get("candidates")
        if type(candidates) is list:
            meter.charge(item, decoded_input_candidate_entries=len(candidates),
                         decoded_input_candidate_objects=sum(type(c) is dict for c in candidates),
                         decoded_input_candidate_x_arrays=sum(type(c) is dict and type(c.get("x")) is list
                                                              for c in candidates))
        for key, counter, kind in (("shared_x", "decoded_input_shared_x_arrays", list),
                                   ("observation", "decoded_observation_objects", dict),
                                   ("configuration", "decoded_configuration_objects", dict)):
            if type(document.get(key)) is kind:
                meter.charge(item, **{counter: 1})


def _decode(raw, meter, item):
    meter.check()
    kind = item["kind"]
    meter.charge(item, json_decodes_started=1, json_bytes_submitted=len(raw),
                 **{kind + "_json_decodes_started": 1})
    try:
        document = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except BaseException as error:
        meter.unknown(item, "partially_decoded_json_values", type(error).__name__)
        raise
    meter.charge(item, json_decodes_completed=1, json_bytes_decoded=len(raw),
                 **{kind + "_json_decodes_completed": 1})
    try:
        meter.tick()  # Check immediately after JSON; a late decode is still charged.
        _named_inventory(document, meter, item)
        _inventory(document, meter, item)
    except BaseException as error:
        meter.unknown(item, "decoded_values_not_fully_inventoried", type(error).__name__)
        raise
    meter.check()
    _need(type(document) is dict, "decoded_root_not_object")
    return document


def _close(handle, label, meter, item):
    """Always attempt an owned close, even after the original deadline."""
    started, failure = time.monotonic(), None
    try:
        meter.tick()
    except BaseException as error:
        failure = error
    meter.charge(item, **{label + "_closes_started": 1})
    try:
        handle.close()
    except BaseException as error:
        meter.unknown(item, label + "_handle_closed", type(error).__name__)
        failure = failure or error
    else:
        meter.charge(item, **{label + "_closes_completed": 1})
    try:
        _need(time.monotonic() - started <= meter.limits["per_close_seconds_max_included_in_original_phase"],
              label + "_close_deadline")
        meter.tick()
    except BaseException as error:
        failure = failure or error
    if failure is not None:
        raise failure


def _stat(call, meter, item):
    meter.tick()
    meter.charge(item, stat_checks_started=1)
    result = _identity(call())
    meter.charge(item, stat_checks_completed=1)
    meter.tick()
    return result


def _source(root, cfg, meter, kind, consume):
    metadata = cfg["fixed_case"]["raw_source_files"][kind]
    item = meter.start(kind, metadata)
    raw = gz = reader = before = None
    prefix_complete, result = False, None
    try:
        meter.check()
        relative = Path(metadata["path"])
        path = (root / relative).resolve()
        _need(not relative.is_absolute() and path.is_relative_to(root), "source_path_outside_root")
        before = _stat(path.stat, meter, item)
        _need(before[:2] == (metadata["bytes"], metadata["mtime_ns"]), "registered_source_stat_changed")
        item["source_stat_before_verified"] = True
        _need(meter.counts["raw_opens_started"] < meter.limits["original_compressed_opens_max"],
              "raw_open_budget")
        meter.charge(item, raw_opens_started=1)
        try:
            raw = path.open("rb", buffering=0)
        except BaseException as error:
            meter.unknown(item, "raw_open_handle_acquired", type(error).__name__)
            raise
        meter.charge(item, raw_opens_completed=1)
        _need(_stat(lambda: os.fstat(raw.fileno()), meter, item) == before, "opened_source_identity_changed")
        reader = _CompressedReader(raw, meter, item)
        meter.charge(item, gzip_opens_started=1)
        try:
            gz = gzip.GzipFile(fileobj=reader, mode="rb")
        except BaseException as error:
            meter.unknown(item, "gzip_open_handle_acquired", type(error).__name__)
            raise
        meter.charge(item, gzip_opens_completed=1)
        result = consume(gz, item)
        prefix_complete = kind != "replay"
        meter.check()
    except BaseException as error:
        item.update(status="failed", failure_code=_failure_code(error))
        raise
    finally:
        original_error, cleanup_errors = sys.exc_info()[1], []
        if gz is not None:
            if not item["gzip_eof_verified"]:
                meter.unknown(item, "gzip_internal_unreturned_payload_bytes",
                              "intentional_prefix_without_eof_probe" if prefix_complete
                              else "failed_before_gzip_eof_or_prefix_stop", planned=prefix_complete)
            try:
                _close(gz, "gzip", meter, item)
            except BaseException as error:
                cleanup_errors.append(error)
        if reader is not None:
            if original_error is None and not cleanup_errors:
                try:
                    reader.drain()
                    _need(raw.tell() == metadata["bytes"], "raw_position_or_length_changed")
                    digest = reader.digest.hexdigest()
                    item["raw_sha256"] = digest
                    _need(item["raw_bytes_hashed"] == metadata["bytes"]
                          and digest == metadata["sha256_inherited"], "raw_full_sha256_changed")
                    item["raw_sha256_verified"] = True
                    meter.charge(item, raw_hashes_completed=1)
                    meter.check()
                except BaseException as error:
                    cleanup_errors.append(error)
            if not item["raw_sha256_verified"]:
                item["partial_raw_sha256"] = reader.digest.hexdigest()
                item["unread_registered_raw_bytes"] = metadata["bytes"] - item["raw_bytes_returned"]
        if raw is not None:
            try:
                _need(_stat(lambda: os.fstat(raw.fileno()), meter, item) == before,
                      "open_source_identity_changed")
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                _close(raw, "raw", meter, item)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                _need(_stat(path.stat, meter, item) == before, "closed_source_identity_changed")
                item["source_stat_after_verified"] = True
            except BaseException as error:
                cleanup_errors.append(error)
        if original_error is not None or cleanup_errors:
            item["status"] = "failed"
            meter.charge(item, sources_failed=1)
            if cleanup_errors:
                item["cleanup_failure_codes"] = [_failure_code(error) for error in cleanup_errors]
                if original_error is None:
                    item["failure_code"] = _failure_code(cleanup_errors[0])
                    raise cleanup_errors[0]
                original_error.add_note("Source cleanup also failed; see retained source ledger.")
    try:
        meter.check()
    except BaseException as error:
        item.update(status="failed", failure_code=_failure_code(error))
        meter.charge(item, sources_failed=1)
        raise
    item["status"] = "completed"
    meter.charge(item, sources_completed=1)
    return result


def _index(gz, meter, item, fixed):
    previous = None
    prefix = meter.counts["index_prefix"]
    for row_index in range(meter.limits["index_prefix_json_rows_max"]):
        line = _payload(gz, meter, item, line=True)
        if not line:
            item["prefix_stop_reason"] = "index_target_absent"
            return None, None, "index_target_absent"
        row = _decode(line, meter, item)
        _need(set(row) == {"game", "seed", "opponent", "side", "step", "decision", "split", "input_sha256"},
              "index_row_schema")
        _need(row["game"] == fixed["game"] and type(row["seed"]) is int and row["seed"] == fixed["seed"]
              and row["opponent"] == fixed["opponent"] and type(row["side"]) is int
              and row["side"] == fixed["side"] and row["split"] == "train", "index_original_train_identity")
        step, decision = row["step"], row["decision"]
        _need(_integer(step) and step < 719 and _integer(decision)
              and type(row["input_sha256"]) is str and _SHA.fullmatch(row["input_sha256"]), "index_row_identity")
        _need((previous is None and decision == 0)
              or (previous is not None and ((step == previous[0] and decision == previous[1] + 1)
                  or (step > previous[0] and decision == 0))), "index_prefix_order")
        previous = (step, decision)
        prefix.append([step, decision, row["input_sha256"]])
        if previous == (fixed["step"], fixed["callback_index_zero_based"]):
            item["prefix_stop_reason"] = "fixed_callback_found"
            return row_index, row, None
        if step > fixed["step"]:
            item["prefix_stop_reason"] = "index_target_passed"
            return None, None, "index_target_passed"
    item["prefix_stop_reason"] = "index_prefix_limit"
    return None, None, "index_prefix_limit"


def _target(gz, meter, item, target_row, index):
    _need(target_row < meter.limits["inputs_labels_prefix_lines_each_max"], "target_row_prefix_limit")
    for row_index in range(target_row + 1):
        line = _payload(gz, meter, item, line=True)
        _need(bool(line), "aligned_target_line_missing")
        if row_index != target_row:
            meter.charge(item, skipped_prefix_lines=1, skipped_prefix_bytes=len(line))
            continue
        document = _decode(line, meter, item)
        item["target_row_index"] = target_row
        item["target_line_sha256"] = hashlib.sha256(line).hexdigest()
        if item["kind"] == "inputs":
            item["target_input_sha256"] = hashlib.sha256(line[:-1]).hexdigest()
            _need(item["target_input_sha256"] == index["input_sha256"], "target_input_index_hash_mismatch")
            _need(type(document.get("candidates")) is list, "target_input_candidates_missing")
        else:
            _need(set(document) == {"teacher_label", "index"}
                  and type(document["teacher_label"]) is str and _integer(document["index"]), "target_label_schema")
        item["prefix_stop_reason"] = "aligned_target_found"
        return document
    raise SourceStopped("target_row_unreachable")


def _replay(gz, meter, item, fixed):
    chunks = []
    while True:
        raw = _payload(gz, meter, item)
        if not raw:
            break
        chunks.append(raw)
    document = _decode(b"".join(chunks), meter, item)
    states = document.get("steps")
    _need(type(states) is list and len(states) == fixed["raw_source_files"]["replay"]["registered_states"]
          and all(type(state) is list and len(state) == 2
                  and all(type(agent) is dict for agent in state) for state in states), "replay_state_shape")
    _need(type(document.get("configuration")) is dict, "replay_configuration_missing")
    return document


def read_case(root: Path, cfg: dict, deadline: float, ledger: dict) -> dict:
    """Read the exact step-15/decision-0 case once; leave failures in ledger.

    Returned replay preserves original JSON object order. The probe must use its
    existing legitimate observation extractor on that replay, never the sorted
    cached observation contained in training_input. certificate is private
    evidence metadata: it includes the inspected index-prefix identities.
    """
    meter, started = _Meter(cfg, deadline, ledger), time.monotonic()
    try:
        fixed = cfg["fixed_case"]
        _need((fixed["game"], fixed["seed"], fixed["opponent"], fixed["side"], fixed["step"],
               fixed["callback_index_zero_based"], fixed["split"]) ==
              ("collect-31317742-v1-0-teacher", 31317742, "v1", 0, 15, 0, "original_train"), "fixed_case_changed")
        _need(set(fixed["raw_source_files"]) == set(_KINDS), "fixed_source_inventory")
        _need(sum(value["bytes"] for value in fixed["raw_source_files"].values())
              == cfg["budget"]["original_compressed_bytes_total_exact_if_all_completed"] == 6586966,
              "fixed_raw_total_changed")
        root = Path(root).resolve()
        target_row, index, reason = _source(root, cfg, meter, "index",
            lambda gz, item: _index(gz, meter, item, fixed))
        result = dict(available=False, reason=reason, target_row_index=None)
        if target_row is not None:
            training_input = _source(root, cfg, meter, "inputs",
                lambda gz, item: _target(gz, meter, item, target_row, index))
            label = _source(root, cfg, meter, "labels",
                lambda gz, item: _target(gz, meter, item, target_row, index))
            replay = _source(root, cfg, meter, "replay",
                lambda gz, item: _replay(gz, meter, item, fixed))
            _need(meter.counts["raw_bytes_returned"] == meter.counts["raw_bytes_hashed"] == 6586966
                  and meter.counts["raw_hashes_completed"] == meter.counts["sources_completed"] == 4,
                  "all_sources_exact_consumption")
            result = dict(available=True, replay=replay, training_input=training_input,
                          label=label, index=index, target_row_index=target_row)
        meter.check()
        _need(not meter.counts["unknown_accounting"]["failures"], "source_failure_unknowns")
        meter.counts.update(status="completed" if result["available"] else "completed_unavailable",
                            target_row_index=target_row, unavailable_reason=reason,
                            elapsed_seconds=time.monotonic() - started)
        result["certificate"] = dict(schema="teacher-label-original-source-certificate-v1",
            fixed_identity={key: fixed[key] for key in ("game", "seed", "opponent", "side", "step", "split",
                                                       "callback_index_zero_based")},
            available=result["available"], target_row_index=target_row, reason=reason,
            consumption=deepcopy(meter.counts))
        elapsed = time.monotonic() - started
        meter.counts["elapsed_seconds"] = elapsed
        result["certificate"]["consumption"]["elapsed_seconds"] = elapsed
        meter.check()  # Includes certificate construction and its return boundary.
        return result
    except BaseException as error:
        meter.counts.update(status="failed", failure_code=_failure_code(error))
        raise
    finally:
        if meter.counts["status"] == "failed":
            meter.counts["elapsed_seconds"] = time.monotonic() - started


def protocol_checks(deadline) -> dict:
    """Two preregistered in-memory gzip cases; defines no model/candidate work.

    This function is intentionally never called at import. Its caller owns the
    same original executable deadline and counts these two cases in its six.
    """
    _need(time.monotonic() < deadline, "protocol_start_deadline")
    limits = dict(original_compressed_opens_max=4,
        original_compressed_bytes_total_exact_if_all_completed=6586966,
        source_returned_payload_bytes_total_max=134217728, index_prefix_json_rows_max=256,
        target_input_json_rows_max=1, target_label_json_rows_max=1, replay_json_decodes_max=1,
        full_replay_states_decoded_max=720, replay_agent_states_decoded_max=1440,
        candidates_per_callback_max=512, raw_data_reread_passes=0,
        per_close_seconds_max_included_in_original_phase=10)
    reports = []
    payload = b'{"ok":1}\n' + b"".join(hashlib.sha256(str(i).encode("ascii")).hexdigest().encode("ascii")
                                    for i in range(512)) + b"\n"
    for number, data in enumerate((payload, b"{}\n")):
        _need(time.monotonic() < deadline, "protocol_case_deadline")
        compressed = gzip.compress(data, mtime=0)
        local = {}
        meter = _Meter({"budget": limits}, deadline, local)
        item = meter.start("index", {"path": "in_memory", "bytes": len(compressed),
                                    "sha256_inherited": hashlib.sha256(compressed).hexdigest()})

        class FaultRaw(io.BytesIO):
            calls = 0

            def read(self, amount=-1):
                self.calls += 1
                if self.calls == 2:
                    raise OSError("synthetic_read_failure")
                return super().read(amount)

        raw = io.BytesIO(compressed) if number == 0 else FaultRaw(compressed)
        reader = _CompressedReader(raw, meter, item)
        gz = gzip.GzipFile(fileobj=reader, mode="rb")
        meter.charge(item, raw_opens_started=1, raw_opens_completed=1,
                     gzip_opens_started=1, gzip_opens_completed=1)
        failure = None
        try:
            line = _payload(gz, meter, item, line=True)
            document = _decode(line, meter, item)
            _need(number == 0 and document == {"ok": 1}, "protocol_prefix_value")
            meter.unknown(item, "gzip_internal_unreturned_payload_bytes",
                          "intentional_prefix_without_eof_probe", planned=True)
        except OSError as error:
            _need(number == 1, "unexpected_protocol_read_failure")
            failure = type(error).__name__
            meter.unknown(item, "gzip_internal_unreturned_payload_bytes", "synthetic_failed_gzip_read")
        finally:
            original_error = sys.exc_info()[1]
            cleanup_errors = []
            try:
                _close(gz, "gzip", meter, item)
                if number == 0 and original_error is None:
                    reader.drain()
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                try:
                    _close(raw, "raw", meter, item)
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors and original_error is None:
                raise cleanup_errors[0]
        if number == 0:
            _need(reader.digest.hexdigest() == item["inherited_sha256"]
                  and item["raw_bytes_hashed"] == item["raw_bytes_returned"] == len(compressed)
                  and item["raw_drain_bytes_returned"] > 0 and item["json_decodes_completed"] == 1
                  and item["returned_payload_bytes"] == len(b'{"ok":1}\n')
                  and not meter.counts["unknown_accounting"]["failures"], "protocol_prefix_drain_hash")
            item.update(status="completed", raw_sha256=reader.digest.hexdigest(), raw_sha256_verified=True)
            meter.charge(item, raw_hashes_completed=1, sources_completed=1)
            meter.counts["status"] = "completed"
        else:
            failure_unknowns = meter.counts["unknown_accounting"]["failures"]
            required_unknowns = {
                "raw_bytes_in_failed_read": "OSError",
                "payload_bytes_in_failed_gzip_call": "OSError",
                "gzip_internal_unreturned_payload_bytes": "synthetic_failed_gzip_read",
            }
            _need(failure == "OSError" and item["raw_bytes_returned"] > 0
                  and item["raw_read_calls_started"] > item["raw_read_calls_completed"]
                  and item["gzip_readline_calls_started"] == 1
                  and item["gzip_readline_calls_completed"] == 0
                  and all(any(entry["kind"] == "index" and entry["quantity"] == quantity
                              and entry["reason"] == reason and entry["value"] is None
                              for entry in failure_unknowns)
                          for quantity, reason in required_unknowns.items())
                  and all(entry["value"] is None for entry in failure_unknowns
                          if entry["quantity"] in required_unknowns),
                  "protocol_failure_consumption_not_zero")
            item.update(status="expected_failure", failure_code=failure,
                        partial_raw_sha256=reader.digest.hexdigest(),
                        unread_registered_raw_bytes=len(compressed) - item["raw_bytes_returned"])
            meter.charge(item, sources_failed=1)
            meter.counts["status"] = "expected_failure"
        meter.check()
        reports.append(dict(case="prefix_drain_once" if number == 0 else "failed_read_retains_consumption",
                            passed=True, consumption=deepcopy(meter.counts)))
    _need(time.monotonic() < deadline, "protocol_return_deadline")
    return dict(passed=True, tiny_in_memory_cases=2, synthetic_candidates=0, temporary_files=0, cases=reports)
