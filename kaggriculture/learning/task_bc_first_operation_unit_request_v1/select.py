"""One independent, bounded selection pass over the fixed historical compact.

No I/O is performed at import. The caller must freeze and preregister this
module before run_selection. Private certificates contain only the decoded
prefix necessary to establish the first operation mismatch, not a new dataset.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

from learning.task_bc_saved_student_decisions_v1.check import (
    CheckFailure, FLAGS, OPS, REQUEST_OPS, ROW_KEYS, _decode, _descriptor,
    _integer, _json_value, _keys, _sha,
)


GAME = "holdout-1714414373-v1-0-trained"
MAX_STEP = 239
COUNTERS = (
    "projection_opens_started", "projection_opens_completed",
    "projection_read_calls_started", "projection_read_calls_completed",
    "projection_bytes_returned", "projection_json_decode_attempts",
    "projection_json_decode_completed", "projection_json_bytes_submitted",
    "projection_json_bytes_decoded", "projection_closes_started",
    "projection_closes_completed", "compact_opens_started", "compact_opens_completed",
    "compact_read_calls_started", "compact_read_calls_completed",
    "compact_bytes_returned", "compact_bytes_hashed", "compact_physical_lines_hashed",
    "compact_prefix_lines_returned", "compact_json_decode_attempts",
    "compact_json_decode_completed", "compact_json_bytes_submitted",
    "compact_json_bytes_decoded", "compact_teacher_descriptors_decoded",
    "compact_chosen_descriptors_decoded", "compact_unit_requests_decoded",
    "compact_suffix_json_decodes", "compact_closes_started", "compact_closes_completed",
    "raw_trace_reads", "replay_reads", "model_reads", "controller_calls",
    "old_witness_or_pair_reads",
)


class SelectionStopped(ValueError):
    """A fixed, non-private failure code. No retry is permitted."""


def _need(condition, code):
    if not condition:
        raise SelectionStopped(code)


def _tick(deadline):
    _need(time.monotonic() < deadline, "selection_deadline")


def _identity(value):
    # Native file identities are checked in memory only, never serialized.
    return (value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino)


def _path(root, relative):
    _need(type(relative) is str and not Path(relative).is_absolute(), "relative_source_required")
    path = (root / relative).resolve()
    _need(path.is_relative_to(root), "source_escapes_root")
    return path


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _write(path, raw, accounting=None):
    with path.open("xb", buffering=0) as stream:
        if accounting is not None:
            accounting["started"] = True
        written = stream.write(raw)
        if accounting is not None:
            accounting["returned_bytes"] = written
    _need(written == len(raw), "output_short_write")


def validate_prefix(prefix, projection_prefix_turns, target):
    """Pure protocol check; earlier same-operation candidate IDs may differ.

    target needs step, target_index, source_line_sha256 and teacher/chosen op.
    The real selector validates complete descriptors separately. This helper is
    also suitable for the single preregistered in-memory dynamic-target case.
    """
    _need(type(target) is dict and _integer(target.get("step"))
          and target["step"] <= MAX_STEP and _integer(target.get("target_index")),
          "prefix_target_identity")
    _need(type(target.get("teacher")) is dict and type(target.get("chosen")) is dict,
          "prefix_target_descriptors")
    step, index = target["step"], target["target_index"]
    _need(type(projection_prefix_turns) is list
          and len(projection_prefix_turns) == step + 1, "prefix_turn_count")
    expected = []
    for ordinal, turn in enumerate(projection_prefix_turns):
        _need(type(turn) is list and len(turn) == 3 and _integer(turn[0])
              and turn[0] == ordinal and _integer(turn[1]) and _sha(turn[2]),
              "prefix_turn_schema_or_order")
        count = turn[1] if ordinal < step else index + 1
        _need(ordinal < step or index < turn[1], "prefix_target_index")
        _need(len(expected) + count <= 691, "prefix_callback_count_limit")
        expected.extend((ordinal, number, turn[2]) for number in range(count))
    _need(type(prefix) is list and 0 < len(prefix) == len(expected), "prefix_row_count")
    for number, (row, identity) in enumerate(zip(prefix, expected)):
        _need(type(row) is list and len(row) == 5
              and _integer(row[0]) and _integer(row[1])
              and tuple(row[:3]) == identity and _sha(row[2]), "prefix_row_identity")
        _need(type(row[3]) is str and row[3] in OPS
              and type(row[4]) is str and row[4] in OPS, "prefix_operation_schema")
        _need((row[3] != row[4]) == (number == len(prefix) - 1),
              "prefix_not_first_operation_mismatch")
    last = prefix[-1]
    _need(last[:3] == [step, index, target.get("source_line_sha256")]
          and last[3] == target.get("teacher", {}).get("op")
          and last[4] == target.get("chosen", {}).get("op"), "prefix_target_binding")


def _unknown(items, quantity, reason, **detail):
    items.append(dict(quantity=quantity, value=None, reason=reason, **detail))


def _open_bound(path, expected, kind, counts, deadline):
    _tick(deadline)
    before = _identity(path.stat())
    _need(type(expected) is dict and before[:2] == (expected["size"], expected["mtime_ns"]),
          kind + "_registered_stat")
    counts[kind + "_opens_started"] += 1
    stream = path.open("rb", buffering=0)
    counts[kind + "_opens_completed"] += 1
    # The caller immediately owns the handle, including fstat validation failure.
    return stream, before


def _read(stream, amount, kind, counts, unknown, deadline):
    _tick(deadline)
    _need(type(amount) is int and amount > 0, "invalid_read_amount")
    counts[kind + "_read_calls_started"] += 1
    try:
        raw = stream.read(amount)
    except BaseException as error:
        _unknown(unknown, kind + "_failed_read_returned_bytes", type(error).__name__,
                 requested_bytes=amount)
        raise
    counts[kind + "_read_calls_completed"] += 1
    counts[kind + "_bytes_returned"] += len(raw)
    _need(0 < len(raw) <= amount, kind + "_short_or_empty_read")
    _tick(deadline)
    return raw


def _close_bound(stream, path, before, kind, counts, unknown, verification):
    if stream is None:
        return
    try:
        _need(_identity(os.fstat(stream.fileno())) == before, kind + "_open_identity_changed")
    finally:
        counts[kind + "_closes_started"] += 1
        try:
            stream.close()
        except BaseException as error:
            _unknown(unknown, kind + "_handle_closed", type(error).__name__)
            raise
        counts[kind + "_closes_completed"] += 1
    _need(_identity(path.stat()) == before, kind + "_closed_identity_changed")
    verification[kind + "_stat_after_verified"] = True


def _projection(root, cfg, deadline, counts, unknown, verification):
    fixed, limits = cfg["fixed_selection"], cfg["scope_budget"]["source_and_metadata"]
    path = _path(root, fixed["projection_receipt"])
    stream = None
    _need(limits["bound_projection_receipt_reads_max"] == 1, "one_projection_read_required")
    _need(0 < cfg["projection_stat"]["size"] <= limits["bound_projection_receipt_bytes_max"],
          "projection_byte_limit")
    try:
        stream, before = _open_bound(path, cfg["projection_stat"], "projection", counts, deadline)
        _need(_identity(os.fstat(stream.fileno())) == before, "projection_open_identity")
        raw = _read(stream, before[0], "projection", counts, unknown, deadline)
        _need(len(raw) == before[0], "projection_incomplete_read")
        _need(hashlib.sha256(raw).hexdigest() == fixed["projection_receipt_sha256"],
              "projection_hash_binding")
        counts["projection_json_decode_attempts"] += 1
        counts["projection_json_bytes_submitted"] += len(raw)
        try:
            receipt = _decode(raw)
        except BaseException as error:
            _unknown(unknown, "projection_json_partial_objects", type(error).__name__)
            raise
        counts["projection_json_decode_completed"] += 1
        counts["projection_json_bytes_decoded"] += len(raw)
        _need(type(receipt) is dict and receipt.get("schema") == "saved-student-extraction-v1"
              and receipt.get("status") == "complete" and receipt.get("passed") is True
              and receipt.get("game") == GAME and receipt.get("failure") is None
              and receipt.get("unknown_accounting") == [], "projection_completion_or_identity")
        _need(receipt.get("source_sha256") == cfg["fixed_input_scope"]["inherited_trace_sha256"]
              and receipt.get("compact_sha256") == fixed["compact_sha256"]
              and type(receipt.get("compact_bytes")) is int
              and receipt["compact_bytes"] == fixed["compact_bytes_exact"], "projection_source_binding")
        turns = receipt.get("turns")
        _need(type(turns) is list and len(turns) == 719, "projection_turn_count")
        for step, turn in enumerate(turns):
            _keys(turn, {"step", "task_count", "crop_count", "line_sha256"}, "projection_turn_keys")
            _need(_integer(turn["step"]) and turn["step"] == step
                  and _integer(turn["task_count"]) and _integer(turn["crop_count"])
                  and _sha(turn["line_sha256"]), "projection_turn_schema_or_order")
        _need(sum(t["task_count"] for t in turns) == fixed["compact_total_rows_inherited"]
              and sum(t["task_count"] for t in turns[:240]) == fixed["early_callback_rows_inherited"],
              "projection_callback_inventory")
        _tick(deadline)
        return turns
    finally:
        if stream is not None:
            _close_bound(stream, path, before, "projection", counts, unknown, verification)


def _validate_row(row, expected):
    _keys(row, ROW_KEYS, "compact_row_keys")
    step, index, line_hash = expected
    _need(row["game"] == GAME and _integer(row["step"]) and row["step"] == step
          and _integer(row["record_index"]) and row["record_index"] == index
          and row["source_line_sha256"] == line_hash, "compact_callback_order_or_source")
    _need(_integer(row["candidate_count"], 1), "compact_candidate_count")
    teacher, chosen = _descriptor(row["teacher"]), _descriptor(row["chosen"])
    _keys(row["flags"], FLAGS, "compact_flags_keys")
    expected_flags = dict(candidate_mismatch=row["teacher"]["id"] != row["chosen"]["id"],
                          op_mismatch=teacher[3] != chosen[3], unit_mismatch=teacher[0] != chosen[0],
                          target_mismatch=teacher[1:3] != chosen[1:3])
    _need(all(type(row["flags"][key]) is bool and row["flags"][key] == value
              for key, value in expected_flags.items()), "compact_flags_recomputed")
    request = row["unit_request"]
    _need(type(request) is list and request and type(request[0]) is str
          and request[0] in REQUEST_OPS, "compact_request_schema")
    _json_value(request)
    return teacher[3] != chosen[3]


def _compact(root, cfg, turns, deadline, counts, unknown, verification, prefix):
    fixed, limits = cfg["fixed_selection"], cfg["scope_budget"]["compact"]
    size = fixed["compact_bytes_exact"]
    _need(size == cfg["compact_stat"]["size"] == limits["bytes_read_max"]
          and limits["files_opened_max"] == limits["read_passes_max"] == 1, "compact_scope")
    expected = [(t["step"], index, t["line_sha256"])
                for t in turns[:240] for index in range(t["task_count"])]
    _need(len(expected) == fixed["early_callback_rows_inherited"]
          and len(expected) <= limits["json_decoded_prefix_rows_max"], "compact_prefix_scope")
    path, stream = _path(root, fixed["source_compact"]), None
    digest, pending, selected = hashlib.sha256(), b"", None

    def read_chunk():
        room = size - counts["compact_bytes_returned"]
        _need(room > 0, "compact_premature_end")
        raw = _read(stream, min(65536, room), "compact", counts, unknown, deadline)
        digest.update(raw)
        counts["compact_bytes_hashed"] += len(raw)
        counts["compact_physical_lines_hashed"] += raw.count(b"\n")
        return raw

    try:
        stream, before = _open_bound(path, cfg["compact_stat"], "compact", counts, deadline)
        _need(_identity(os.fstat(stream.fileno())) == before, "compact_open_identity")
        for identity in expected:
            _tick(deadline)
            while b"\n" not in pending:
                _need(len(pending) < limits["single_line_bytes_max"], "compact_line_byte_limit")
                pending += read_chunk()
            line, pending = pending.split(b"\n", 1)
            line += b"\n"
            counts["compact_prefix_lines_returned"] += 1
            _need(len(line) <= limits["single_line_bytes_max"], "compact_line_byte_limit")
            _need(counts["compact_json_decode_attempts"] < limits["json_decoded_prefix_rows_max"],
                  "compact_decode_budget")
            counts["compact_json_decode_attempts"] += 1
            counts["compact_json_bytes_submitted"] += len(line)
            try:
                row = _decode(line)
            except BaseException as error:
                _unknown(unknown, "compact_json_partial_objects", type(error).__name__)
                raise
            counts["compact_json_decode_completed"] += 1
            counts["compact_json_bytes_decoded"] += len(line)
            # Charge every named child decoded before any schema/limit rejection.
            for field, kind, counter in (("teacher", dict, "compact_teacher_descriptors_decoded"),
                                        ("chosen", dict, "compact_chosen_descriptors_decoded"),
                                        ("unit_request", list, "compact_unit_requests_decoded")):
                if type(row) is dict and type(row.get(field)) is kind:
                    counts[counter] += 1
                else:
                    _unknown(unknown, "compact_decoded_child_objects", "unexpected_" + field + "_shape")
            for counter, cap in (("compact_teacher_descriptors_decoded", "decoded_teacher_descriptors_max"),
                                 ("compact_chosen_descriptors_decoded", "decoded_chosen_descriptors_max"),
                                 ("compact_unit_requests_decoded", "decoded_unit_requests_max")):
                _need(counts[counter] <= limits[cap], "compact_child_budget")
            differs = _validate_row(row, identity)
            prefix.append([row["step"], row["record_index"], row["source_line_sha256"],
                           row["teacher"]["op"], row["chosen"]["op"]])
            if differs:
                selected = dict(game=GAME, side=0, step=row["step"], target_index=row["record_index"],
                                source_line_sha256=row["source_line_sha256"],
                                teacher=deepcopy(row["teacher"]), chosen=deepcopy(row["chosen"]))
                break
        _need(selected is not None and selected["step"] <= MAX_STEP, "no_eligible_early_mismatch")
        # pending contains already charged bytes; neither it nor the remaining
        # stream is JSON-decoded. No extra EOF read is allowed after exact size.
        while counts["compact_bytes_returned"] < size:
            read_chunk()
        _need(counts["compact_bytes_hashed"] == size and stream.tell() == size,
              "compact_exact_length")
        verification["compact_exact_registered_length_verified"] = True
        _need(counts["compact_physical_lines_hashed"] == fixed["compact_total_rows_inherited"],
              "compact_physical_line_inventory")
        _need(digest.hexdigest() == fixed["compact_sha256"], "compact_full_hash_binding")
        verification["compact_full_hash_verified"] = True
        _tick(deadline)
        return selected
    finally:
        if stream is not None:
            _close_bound(stream, path, before, "compact", counts, unknown, verification)


def run_selection(root: Path, out: Path, cfg: dict, deadline: float) -> dict:
    """Run once. Failed/partial work is retained, never silently replaced."""
    root, out = Path(root).resolve(), Path(out)
    _need(not (out / "selection.json").exists()
          and not (out / "selection-status.json").exists(), "existing_selection_no_retry")
    started = time.monotonic()
    counts = dict.fromkeys(COUNTERS, 0)
    unknown, prefix, turns, selected = [], [], [], None
    verification = dict(projection_stat_after_verified=False, compact_stat_after_verified=False,
                        compact_full_hash_verified=False, compact_exact_registered_length_verified=False,
                        compact_eof_probed=False)
    status, failure, phase, private_ref = "failed", None, "projection", None
    try:
        _tick(deadline)
        turns = _projection(root, cfg, deadline, counts, unknown, verification)
        phase = "compact"
        selected = _compact(root, cfg, turns, deadline, counts, unknown, verification, prefix)
        phase = "prefix_certificate"
        prefix_turns = [[t["step"], t["task_count"], t["line_sha256"]]
                        for t in turns[:selected["step"] + 1]]
        validate_prefix(prefix, prefix_turns, selected)
        _need(not unknown, "selection_unknown_work")
        _tick(deadline)
        status, phase = "complete", "complete"
    except BaseException as error:
        code = str(error) if isinstance(error, (SelectionStopped, CheckFailure)) else ""
        failure = code if re.fullmatch(r"[a-z][a-z0-9_]*", code) else type(error).__name__
    prefix_turns = ([[t["step"], t["task_count"], t["line_sha256"]]
                     for t in turns[:selected["step"] + 1]] if selected is not None else [])
    private = dict(schema="first-operation-selection-v1", status=status, selected=selected,
                   prefix=prefix, projection_prefix_turns=prefix_turns,
                   projection_receipt_sha256=cfg["fixed_selection"]["projection_receipt_sha256"],
                   compact_sha256=cfg["fixed_selection"]["compact_sha256"], consumption=counts,
                   unknown_accounting=unknown, verification=verification,
                   elapsed_seconds=time.monotonic() - started, phase=phase, failure_code=failure)
    write_accounting = {"started": False, "returned_bytes": None}
    try:
        raw = _encoded(private)
        _need(len(raw) <= cfg["scope_budget"]["outputs"]["selection_receipt_bytes_max"],
              "selection_receipt_byte_limit")
        _write(out / "selection.json", raw, write_accounting)
        private_ref = dict(path="selection.json", sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    except BaseException as error:
        status, failure, phase = "failed", type(error).__name__, "private_receipt_write"
        if write_accounting["started"] and write_accounting["returned_bytes"] is None:
            _unknown(unknown, "private_receipt_write_returned_bytes", type(error).__name__)
    if time.monotonic() >= deadline:
        status, failure, phase = "failed", "selection_deadline", "finalization"
    safe_selected = None
    if status == "complete":
        safe_selected = {key: selected[key] for key in
                         ("game", "side", "step", "target_index", "source_line_sha256")}
        safe_selected.update(teacher_op=selected["teacher"]["op"], chosen_op=selected["chosen"]["op"])
    safe = dict(schema="first-operation-selection-status-v1", status=status, selected=safe_selected,
                private_receipt=private_ref, consumption=counts, unknown_accounting=unknown,
                verification=verification, elapsed_seconds=time.monotonic() - started,
                phase=phase, failure_code=failure)
    encoded = _encoded(safe)
    _need(len(encoded) <= cfg["scope_budget"]["outputs"]["public_summary_bytes_max"],
          "selection_status_byte_limit")
    _write(out / "selection-status.json", encoded)
    # A late terminal receipt remains provisional: the worker must fail and the
    # parent requires zero exit and its original phase deadline for acceptance.
    _tick(deadline)
    return safe
