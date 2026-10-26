"""Three synthetic gzip boundary checks; caller-owned partial accounting.

The inherited raw_* counters describe BytesIO operations, never real file opens.
No fixture runs at import. The unchanged production reader owns all byte/JSON
meters; this module only controls synthetic read boundaries and retains results.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import math
import sys
import time

from learning.task_bc_teacher_label_equivalence_v1.source import (
    _Meter, _CompressedReader, _payload, _decode, _close,
)


SCHEMA = "teacher-label-protocol-repair-v1"
NAMES = ("zero_drain", "positive_drain", "read_failure_boundary")


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _alive(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("original_protocol_deadline")


def _totals(ledger):
    totals = dict(cases_started=len(ledger["cases"]), cases_completed=0, cases_passed=0,
                  operation_errors=0, assertion_errors=0, planned_prefix_unknowns=0,
                  failure_unknowns=0, cases_with_unknown_consumption=0)
    for row in ledger["cases"]:
        totals["cases_completed"] += row["status"] == "complete"
        totals["cases_passed"] += row["passed"] is True
        totals["operation_errors"] += row["failure"]["operation_error"] is not None
        totals["assertion_errors"] += row["failure"]["assertion_error"] is not None
        consumption = row["consumption"]
        if consumption is None:
            totals["cases_with_unknown_consumption"] += 1
            continue
        for key, value in consumption.items():
            if type(value) is int:
                totals[key] = totals.get(key, 0) + value
        unknown = consumption["unknown_accounting"]
        totals["planned_prefix_unknowns"] += len(unknown["planned_prefix"])
        totals["failure_unknowns"] += len(unknown["failures"])
    ledger["totals"] = totals


class _BoundaryRaw(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name, self.calls = name, 0

    def read(self, amount=-1):
        self.calls += 1
        if self.name == "read_failure_boundary" and self.calls == 2:
            raise OSError("synthetic_second_raw_read")
        if self.name == "positive_drain":
            amount = 256 if amount < 0 else min(amount, 256)
        return super().read(amount)


class _Bridge(io.RawIOBase):
    """Fixed buffer prefetches through the meter; closing never reads or owns raw."""
    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    def readable(self):
        return True

    def readinto(self, target):
        data = self.reader.read(len(target))
        target[:len(data)] = data
        return len(data)


def _case(name, compressed, expected_sha, deadline, ledger):
    started = time.monotonic()
    row = dict(name=name, status="started", passed=False, consumption=None, observed={},
               failure=dict(operation_error=None, assertion_error=None, unexpected_error=None,
                            close_errors=[]))
    ledger["cases"].append(row)  # Visible before the first stream/meter operation.
    raw = gz = bridge = buffered = meter = item = reader = None
    operation_complete, expected_read_error = False, False
    observed, failure = row["observed"], row["failure"]
    try:
        _alive(deadline)
        limits = dict(original_compressed_opens_max=1,
            original_compressed_bytes_total_exact_if_all_completed=131072 - ledger["totals"].get("raw_bytes_returned", 0),
            source_returned_payload_bytes_total_max=1024 - ledger["totals"].get("returned_payload_bytes", 0),
            index_prefix_json_rows_max=1 if name != "read_failure_boundary" else 0,
            target_input_json_rows_max=0, target_label_json_rows_max=0, replay_json_decodes_max=0,
            full_replay_states_decoded_max=0, replay_agent_states_decoded_max=0,
            candidates_per_callback_max=0, raw_data_reread_passes=0,
            per_close_seconds_max_included_in_original_phase=5)
        holder = {}
        meter = _Meter({"budget": limits}, deadline, holder)
        row["consumption"] = meter.counts  # Live reference survives every later failure.
        item = meter.start("index", dict(path="synthetic:" + name, bytes=len(compressed),
                                        sha256_inherited=expected_sha))
        try:
            meter.charge(item, raw_opens_started=1)
            raw = _BoundaryRaw(compressed, name)
            meter.charge(item, raw_opens_completed=1)
            reader = _CompressedReader(raw, meter, item)
            transport = reader
            if name == "zero_drain":
                bridge = _Bridge(reader)
                buffered = io.BufferedReader(bridge, buffer_size=65536)
                transport = buffered
            meter.charge(item, gzip_opens_started=1)
            gz = gzip.GzipFile(fileobj=transport, mode="rb")
            meter.charge(item, gzip_opens_completed=1)
            try:
                line = _payload(gz, meter, item, line=True)
            except OSError as error:
                failure["operation_error"] = dict(type=type(error).__name__, message=str(error))
                if name != "read_failure_boundary" or raw.calls != 2 or str(error) != "synthetic_second_raw_read":
                    raise
                expected_read_error = True
            else:
                _require(name != "read_failure_boundary", "expected_read_failure_missing")
                _require(ledger["totals"].get("json_decodes_started", 0) < 2, "aggregate_json_decode_cap")
                _require(_decode(line, meter, item) == {"ok": 1}, "prefix_json_value")
                operation_complete = True
            observed.update(raw_bytes_before_drain=item["raw_bytes_returned"], raw_calls_before_drain=raw.calls,
                            remaining_raw_before_drain=len(compressed) - item["raw_bytes_returned"])
        finally:
            original_error, close_error = sys.exc_info()[1], None
            reads_before_close = item["raw_read_calls_started"]
            if gz is not None:
                meter.unknown(item, "gzip_internal_unreturned_payload_bytes",
                    "intentional_prefix_without_eof_probe" if operation_complete else "failed_prefix_read",
                    planned=operation_complete)
                try:
                    _close(gz, "gzip", meter, item)
                except BaseException as error:
                    close_error = error
                    failure["close_errors"].append(dict(handle="gzip", type=type(error).__name__))
            # Both auxiliary objects are explicitly closed; bridge.close never
            # closes/re-reads the separately owned metered raw BytesIO.
            for handle, label in ((buffered, "buffer"), (bridge, "bridge")):
                if handle is not None:
                    tick = time.monotonic()
                    try:
                        handle.close()
                        _require(time.monotonic() - tick <= 5, "auxiliary_close_limit")
                        _alive(deadline)
                    except BaseException as error:
                        close_error = close_error or error
                        failure["close_errors"].append(dict(handle=label, type=type(error).__name__))
                    observed[label + "_closed"] = handle.closed
            observed["raw_reads_during_gzip_and_aux_close"] = item["raw_read_calls_started"] - reads_before_close
            try:
                if operation_complete and original_error is None and close_error is None:
                    reader.drain()
                    _require(reader.digest.hexdigest() == expected_sha
                             and item["raw_bytes_returned"] == item["raw_bytes_hashed"] == len(compressed),
                             "complete_raw_hash_binding")
                    item.update(raw_sha256=reader.digest.hexdigest(), raw_sha256_verified=True, status="completed")
                    meter.charge(item, raw_hashes_completed=1, sources_completed=1)
                    meter.counts["status"] = "completed"
            finally:
                if raw is not None:
                    try:
                        _close(raw, "raw", meter, item)
                    except BaseException as error:
                        close_error = close_error or error
                        failure["close_errors"].append(dict(handle="raw", type=type(error).__name__))
                if not item["raw_sha256_verified"]:
                    item.update(status="failed", partial_raw_sha256=reader.digest.hexdigest() if reader else None)
                    meter.charge(item, sources_failed=1)
                    meter.counts["status"] = "failed"
            if close_error is not None:
                if original_error is None:
                    raise close_error
                original_error.add_note("Synthetic close also failed; caller ledger retains details.")
        _alive(deadline)
        _require(item["raw_closes_completed"] == item["gzip_closes_completed"] == 1
                 and raw.closed and gz.closed and observed["raw_reads_during_gzip_and_aux_close"] == 0,
                 "all_owned_handles_closed_without_reads")
        if name == "read_failure_boundary":
            _require(expected_read_error and failure["operation_error"] is not None, "injected_operation_error_missing")
            # Deliberately exercise the formerly lossy final-assertion boundary
            # on this same partial pass, after its real read error and closure.
            try:
                _require(item["raw_sha256_verified"], "expected_complete_sha_assertion_failure")
            except AssertionError as error:
                if str(error) != "expected_complete_sha_assertion_failure":
                    raise
                failure["assertion_error"] = dict(type=type(error).__name__, message=str(error))
            else:
                raise AssertionError("expected_final_assertion_did_not_fail")
            _require(item["raw_read_calls_started"] == raw.calls == 2 and item["raw_read_calls_completed"] == 1
                     and item["raw_bytes_returned"] == item["raw_bytes_hashed"] == 2
                     and item["gzip_readline_calls_started"] == 1 and item["gzip_readline_calls_completed"] == 0
                     and item["json_decodes_started"] == item["raw_hashes_completed"] == 0
                     and not item["raw_sha256_verified"] and item["raw_sha256"] is None, "partial_fault_consumption")
            unknown = meter.counts["unknown_accounting"]["failures"]
            quantities = {"raw_bytes_in_failed_read", "payload_bytes_in_failed_gzip_call",
                          "gzip_internal_unreturned_payload_bytes"}
            _require(all(any(entry["quantity"] == quantity and entry["value"] is None for entry in unknown)
                         for quantity in quantities) and all(entry["value"] is None for entry in unknown
                         if entry["quantity"] in quantities), "failure_unknowns_must_not_be_zero")
        else:
            _require(item["json_decodes_started"] == item["json_decodes_completed"] == 1
                     and item["returned_payload_bytes"] == len(b'{"ok":1}\n')
                     and item["suffix_json_decodes"] == item["raw_reread_passes"] == 0
                     and not meter.counts["unknown_accounting"]["failures"], "single_prefix_once")
            _require(any(entry["quantity"] == "gzip_internal_unreturned_payload_bytes" and entry["value"] is None
                         for entry in meter.counts["unknown_accounting"]["planned_prefix"]), "planned_unknown_retained")
            if name == "zero_drain":
                _require(observed["raw_calls_before_drain"] == 1 and observed["remaining_raw_before_drain"] == 0
                         and item["raw_drain_bytes_returned"] == 0 and buffered.closed and bridge.closed,
                         "controlled_zero_drain")
            else:
                _require(observed["remaining_raw_before_drain"] > 0 and item["raw_drain_bytes_returned"] > 0,
                         "controlled_positive_drain")
        meter.check()
        row.update(status="complete", passed=True)
    except BaseException as error:
        row["status"] = "failed"
        failure["unexpected_error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        observed.update(raw_closed=raw.closed if raw is not None else None,
                        gzip_closed=gz.closed if gz is not None else None,
                        raw_calls=raw.calls if raw is not None else None,
                        drain_bytes=item["raw_drain_bytes_returned"] if item is not None else None,
                        raw_sha256_verified=item["raw_sha256_verified"] if item is not None else False,
                        internal_unreturned_payload_bytes=item["internal_unreturned_payload_bytes"] if item else None)
        row["elapsed_seconds"] = time.monotonic() - started
        _totals(ledger)


def protocol_checks(deadline: float, ledger: dict) -> dict:
    """Execute exactly three synthetic cases; retain every started case on error."""
    _require("cases" not in ledger and "schema" not in ledger, "protocol_ledger_already_used_no_retry")
    ledger.update(schema=SCHEMA, status="started", passed=False, cases=[], totals={},
        counter_scope="raw_* counters are synthetic in-memory stream operations; no real payload file is opened.",
        prohibited=dict(real_files_opened=0, source_observations=0, controller_calls=0, model_reads=0,
                        model_forwards=0, training_runs=0, temporary_files=0))
    started = time.monotonic()
    try:
        _require(type(deadline) in (int, float) and math.isfinite(deadline), "finite_original_deadline_required")
        _alive(deadline)
        _totals(ledger)
        payload = b'{"ok":1}\n' + b"".join(hashlib.sha256(str(i).encode("ascii")).hexdigest().encode("ascii")
                                         for i in range(512)) + b"\n"
        compressed = gzip.compress(payload, mtime=0)
        _require(512 < len(compressed) < 65536, "fixed_synthetic_compressed_size")
        expected_sha = hashlib.sha256(compressed).hexdigest()
        ledger["fixture"] = dict(compressed_bytes=len(compressed), compressed_sha256=expected_sha,
                                 generated_payload_bytes=len(payload))
        for name in NAMES:
            _case(name, compressed, expected_sha, deadline, ledger)
        totals = ledger["totals"]
        _require(totals["cases_completed"] == totals["cases_passed"] == 3
                 and totals["raw_bytes_returned"] == 2 * len(compressed) + 2 <= 131072
                 and totals["returned_payload_bytes"] <= 1024 and totals["json_decodes_completed"] == 2
                 and totals["cases_with_unknown_consumption"] == 0, "aggregate_synthetic_limits")
        _alive(deadline)
        ledger.update(status="complete", passed=True)
        return dict(schema=SCHEMA, passed=True, scenario_count=3)
    except BaseException as error:
        ledger.update(status="failed", passed=False, failure_type=type(error).__name__)
        raise
    finally:
        ledger["elapsed_seconds"] = time.monotonic() - started
        _totals(ledger)
        if sys.exc_info()[1] is None and time.monotonic() >= deadline:
            ledger.update(status="failed", passed=False, failure_type="TimeoutError")
            raise TimeoutError("protocol_final_receipt_deadline")
