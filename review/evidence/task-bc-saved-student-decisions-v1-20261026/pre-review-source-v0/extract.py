"""Single-pass projection of saved student decisions; no agent/model imports.

Only ``extract_game`` is a real-source entrypoint.  Its private implementation
also accepts a small expected-turn count for the separately registered synthetic
checks.  All output is local and private until the root's public allowlist runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


OPS = ("DIG", "FERTILIZE", "HARVEST", "PLANT", "WATER")
REQUEST_OPS = OPS + ("PASS", "EAST", "WEST", "SOUTH", "NORTH", "DROP", "PICKUP")
FLAGS = ("candidate_mismatch", "op_mismatch", "unit_mismatch", "target_mismatch")
PHASES = ("early", "middle", "late")
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_GAME_BYTES = 256 * 1024 * 1024
COUNTER_KEYS = (
    "source_trace_files", "compressed_sha_passes", "gzip_json_passes",
    "compressed_bytes_hashed", "compressed_bytes_for_decode",
    "compressed_source_bytes_read", "uncompressed_bytes", "physical_lines",
    "task_callback_objects", "candidate_objects", "candidate_x_arrays",
    "shared_x_arrays", "observation_objects", "configuration_objects",
    "crop_records", "compact_rows",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ExtractionStopped(ValueError):
    """A bounded incomplete result, never an instruction to retry."""


def _require(condition, message):
    if not condition:
        raise ExtractionStopped(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path, value):
    encoded = _json_bytes(value)
    with path.open("xb") as output:
        output.write(encoded)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _empty_summary():
    def cell():
        return {"task_callbacks": 0, "task_candidates": 0,
                "flags": dict.fromkeys(FLAGS, 0)}
    result = cell()
    result["phase_teacher_op"] = {p: {op: cell() for op in OPS} for p in PHASES}
    result["op_confusion"] = {a: dict.fromkeys(OPS, 0) for a in OPS}
    result["request_op_counts"] = dict.fromkeys(REQUEST_OPS, 0)
    return result


def _add_summary(summary, row):
    phase = "early" if row["step"] <= 239 else "middle" if row["step"] <= 479 else "late"
    teacher, chosen = row["teacher"]["op"], row["chosen"]["op"]
    for cell in (summary, summary["phase_teacher_op"][phase][teacher]):
        cell["task_callbacks"] += 1
        cell["task_candidates"] += row["candidate_count"]
        for flag in FLAGS:
            cell["flags"][flag] += int(row["flags"][flag])
    summary["op_confusion"][teacher][chosen] += 1
    summary["request_op_counts"][row["unit_request"][0]] += 1


class _Ledger:
    def __init__(self, budget, game_cfg, deadline, clock):
        self.counts = dict.fromkeys(COUNTER_KEYS, 0)
        self.unknown = []
        self.deadline, self.clock = deadline, clock
        _require(set(budget) == set(COUNTER_KEYS), "budget counter keys differ")
        _require(all(_integer(v) for v in budget.values()), "invalid counter budget")
        self.limits = dict(budget)
        local = {
            "source_trace_files": 1, "compressed_sha_passes": 1, "gzip_json_passes": 1,
            "compressed_bytes_hashed": game_cfg["compressed_bytes"],
            "compressed_bytes_for_decode": game_cfg["compressed_bytes"],
            "compressed_source_bytes_read": 2 * game_cfg["compressed_bytes"],
            "uncompressed_bytes": MAX_GAME_BYTES, "physical_lines": game_cfg["turns"],
            "task_callback_objects": game_cfg["task_callbacks"],
            "candidate_objects": game_cfg["task_candidates"],
            "candidate_x_arrays": game_cfg["task_candidates"],
            "shared_x_arrays": game_cfg["task_callbacks"],
            "observation_objects": game_cfg["task_callbacks"],
            "configuration_objects": game_cfg["task_callbacks"],
            "crop_records": game_cfg["crop_records"], "compact_rows": game_cfg["task_callbacks"],
        }
        self.limits = {k: min(self.limits[k], local[k]) for k in COUNTER_KEYS}

    def charge(self, **amounts):
        # Charging deliberately has no guard: one json.loads has already decoded
        # all children, so every child must be counted before any late stop.
        for name, amount in amounts.items():
            self.counts[name] += amount

    def check(self):
        if self.clock() >= self.deadline:
            raise ExtractionStopped("deadline exhausted")
        for name in COUNTER_KEYS:
            if self.counts[name] > self.limits[name]:
                raise ExtractionStopped("counter budget exceeded: " + name)

    def remaining(self, name):
        return self.limits[name] - self.counts[name]


class _CompressedReader:
    """Unbuffered compressed source reads, charged before returning to gzip."""
    def __init__(self, source, ledger, counter, expected_bytes):
        self.source, self.ledger, self.counter = source, ledger, counter
        self.remaining = expected_bytes

    def read(self, size=-1):
        self.ledger.check()
        if self.remaining == 0:
            return b""
        room = min(self.remaining, self.ledger.remaining(self.counter),
                   self.ledger.remaining("compressed_source_bytes_read"))
        _require(room > 0, "compressed byte budget exhausted")
        # Neither gzip nor this adapter may request an unbounded file read.
        amount = min(room, 64 * 1024 if size < 0 else size)
        if amount == 0:
            return b""
        data = self.source.read(amount)
        self.remaining -= len(data)
        self.ledger.charge(**{self.counter: len(data),
                              "compressed_source_bytes_read": len(data)})
        self.ledger.check()
        _require(data, "compressed source ended before registered size")
        return data


def _charge_children(document, ledger, line_number):
    """Count parsed children, including unused arrays, before schema checks."""
    def unknown(reason):
        ledger.unknown.append({"line": line_number, "reason": reason})

    if not isinstance(document, dict):
        unknown("decoded_root_not_object")
        return
    records = document.get("records")
    if not isinstance(records, list):
        unknown("decoded_records_not_list")
    else:
        for record in records:
            if not isinstance(record, dict):
                unknown("decoded_task_record_not_object")
                continue
            ledger.charge(task_callback_objects=1)
            inputs = record.get("inputs")
            if not isinstance(inputs, dict):
                unknown("decoded_inputs_not_object")
                continue
            for field, counter, kind in (
                ("shared_x", "shared_x_arrays", list),
                ("observation", "observation_objects", dict),
                ("configuration", "configuration_objects", dict),
            ):
                if isinstance(inputs.get(field), kind):
                    ledger.charge(**{counter: 1})
            candidates = inputs.get("candidates")
            if not isinstance(candidates, list):
                unknown("decoded_candidates_not_list")
                continue
            for candidate in candidates:
                if isinstance(candidate, dict):
                    ledger.charge(candidate_objects=1)
                    if isinstance(candidate.get("x"), list):
                        ledger.charge(candidate_x_arrays=1)
                else:
                    unknown("decoded_candidate_not_object")
    crop_records = document.get("crop_records")
    if not isinstance(crop_records, list):
        unknown("decoded_crop_records_not_list")
    else:
        for record in crop_records:
            if isinstance(record, dict):
                ledger.charge(crop_records=1)
            else:
                unknown("decoded_crop_record_not_object")


def _descriptor(candidate):
    _require(isinstance(candidate, dict), "candidate is not an object")
    unit, target, op = candidate.get("unit"), candidate.get("target"), candidate.get("op")
    _require(_integer(unit), "invalid candidate unit")
    _require(isinstance(target, list) and len(target) == 2
             and all(_integer(v) for v in target), "invalid candidate target")
    _require(isinstance(op, str) and op in OPS, "invalid candidate operation")
    identifier = f"u{unit}:x{target[0]}:y{target[1]}:{op}"
    _require(candidate.get("id") == identifier, "noncanonical candidate identity")
    _require(isinstance(candidate.get("x"), list), "missing candidate x array")
    return {"id": identifier, "unit": unit, "target": list(target), "op": op}


def _valid_json(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_valid_json(v) for v in value)
    if isinstance(value, dict):
        return all(type(k) is str and _valid_json(v) for k, v in value.items())
    return False


def _project(document, cfg, step, line_hash, ledger):
    _require(isinstance(document, dict), "turn is not an object")
    _require(document.get("schema_version") == "task-choice-v1", "turn schema differs")
    _require(type(document.get("step")) is int and document["step"] == step,
             "turn step/order differs")
    for name in ("load_errors", "inference_errors", "fallbacks"):
        _require(type(document.get(name)) is int and document[name] == 0,
                 "saved error/fallback or missing counter: " + name)
    _require(document.get("task_checkpoint_sha256") == cfg["task_checkpoint_sha256"],
             "task checkpoint identity differs")
    _require(document.get("checkpoint_sha256") == cfg["crop_checkpoint_sha256"],
             "crop checkpoint identity differs")
    records, crops = document.get("records"), document.get("crop_records")
    _require(isinstance(records, list) and isinstance(crops, list), "turn record arrays missing")
    _require(all(isinstance(c, dict) for c in crops), "invalid crop record")
    reservations = document.get("final_reservations")
    _require(isinstance(reservations, dict), "final reservations missing")
    actions = reservations.get("unit_actions")
    _require(isinstance(actions, list), "final unit actions missing")
    for index, record in enumerate(records):
        ledger.check()
        _require(isinstance(record, dict) and record.get("source") == "callback",
                 "record is not a saved student callback")
        inputs = record.get("inputs")
        _require(isinstance(inputs, dict) and inputs.get("schema_version") == "task-choice-v1",
                 "callback input schema differs")
        obs = inputs.get("observation")
        _require(isinstance(obs, dict) and type(obs.get("step")) is int and obs["step"] == step
                 and type(obs.get("player")) is int and obs["player"] == cfg["side"],
                 "callback observation identity differs")
        _require(isinstance(inputs.get("configuration"), dict)
                 and isinstance(inputs.get("shared_x"), list), "callback arrays/objects missing")
        candidates = inputs.get("candidates")
        _require(isinstance(candidates, list) and candidates, "candidate list missing or empty")
        lookup = {}
        for candidate in candidates:
            selected = _descriptor(candidate)
            _require(selected["id"] not in lookup, "duplicate candidate identity")
            lookup[selected["id"]] = selected
        teacher_id, chosen_id = record.get("teacher_label"), record.get("chosen")
        _require(isinstance(teacher_id, str) and isinstance(chosen_id, str)
                 and teacher_id in lookup and chosen_id in lookup, "saved selection not in candidates")
        teacher, chosen = lookup[teacher_id], lookup[chosen_id]
        _require(type(record.get("unit")) is int and record["unit"] == chosen["unit"]
                 and isinstance(record.get("target"), list) and len(record["target"]) == 2
                 and all(_integer(v) for v in record["target"])
                 and record["target"] == chosen["target"] and record.get("op") == chosen["op"],
                 "selected descriptor differs from chosen candidate")
        request = record.get("unit_request")
        _require(isinstance(request, list) and request and type(request[0]) is str
                 and request[0] in REQUEST_OPS and _valid_json(request), "invalid saved unit request")
        _require(chosen["unit"] < len(actions) and request == actions[chosen["unit"]],
                 "unit request differs from final reservations")
        yield {
            "game": cfg["game"], "step": step, "record_index": index,
            "candidate_count": len(candidates), "source_line_sha256": line_hash,
            "teacher": teacher, "chosen": chosen, "unit_request": request,
            "flags": {"candidate_mismatch": chosen_id != teacher_id,
                      "op_mismatch": chosen["op"] != teacher["op"],
                      "unit_mismatch": chosen["unit"] != teacher["unit"],
                      "target_mismatch": chosen["target"] != teacher["target"]},
        }


def _validate_cfg(cfg, expected_turns):
    for name in ("compressed_bytes", "turns", "task_callbacks", "task_candidates", "crop_records"):
        _require(_integer(cfg.get(name)), "invalid registered count: " + name)
    _require(cfg["compressed_bytes"] > 0 and cfg["turns"] == expected_turns,
             "registered stream size/turn count differs")
    _require(type(cfg.get("game")) is str and cfg["game"], "missing game identity")
    _require(_integer(cfg.get("seed")) and type(cfg.get("side")) is int and cfg["side"] in (0, 1),
             "invalid registered game identity")
    for name in ("trace_sha256", "task_checkpoint_sha256", "crop_checkpoint_sha256"):
        _require(type(cfg.get(name)) is str and _SHA.fullmatch(cfg[name]),
                 "invalid registered hash: " + name)


def _extract_game(source_path, out_dir, game_cfg, budget, deadline, *, expected_turns,
                  clock=time.monotonic):
    """Private synthetic seam; production must call extract_game below."""
    started = clock()
    _validate_cfg(game_cfg, expected_turns)
    _require(type(deadline) in (float, int) and math.isfinite(deadline), "invalid deadline")
    ledger = _Ledger(budget, game_cfg, min(deadline, started + 60), clock)
    output_dir = Path(out_dir)
    # An existing directory (even empty) is never adopted or resumed.
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "started.json", {"schema": "saved-student-extraction-start-v1",
                                             "game": game_cfg["game"]})
    source_hash = None
    compact_hash, compact_bytes = hashlib.sha256(), 0
    first, turns, summary, failure = None, [], _empty_summary(), None
    source = Path(source_path)
    try:
        ledger.check()
        with source.open("rb", buffering=0) as raw:
            ledger.charge(source_trace_files=1, compressed_sha_passes=1)
            ledger.check()
            initial_stat = os.fstat(raw.fileno())
            identity = (initial_stat.st_size, initial_stat.st_mtime_ns, initial_stat.st_ino)
            _require(initial_stat.st_size == game_cfg["compressed_bytes"], "compressed size differs")
            reader = _CompressedReader(raw, ledger, "compressed_bytes_hashed", initial_stat.st_size)
            digest = hashlib.sha256()
            while reader.remaining:
                digest.update(reader.read(64 * 1024))
            final_stat = os.fstat(raw.fileno())
            _require((final_stat.st_size, final_stat.st_mtime_ns, final_stat.st_ino) == identity,
                     "source changed during hash pass")
            source_hash = digest.hexdigest()
        _require(source_hash == game_cfg["trace_sha256"], "compressed SHA256 differs")
        ledger.check()
        with source.open("rb", buffering=0) as raw, (output_dir / "compact.jsonl").open("xb", buffering=0) as compact:
            ledger.charge(gzip_json_passes=1)
            ledger.check()
            current = os.fstat(raw.fileno())
            _require((current.st_size, current.st_mtime_ns, current.st_ino) == identity,
                     "source changed before decode pass")
            reader = _CompressedReader(raw, ledger, "compressed_bytes_for_decode", current.st_size)
            with gzip.GzipFile(fileobj=reader, mode="rb") as decoded:
                while True:
                    ledger.check()
                    room = ledger.remaining("uncompressed_bytes")
                    _require(room > 0, "decompressed byte budget exhausted before EOF")
                    # The +1 detects an oversized line without an unbounded read;
                    # the remaining whole-stream quota always takes precedence.
                    limit = min(MAX_LINE_BYTES + 1, room)
                    if ledger.remaining("physical_lines") == 0:
                        limit = min(limit, 1)  # EOF probe, never another JSON decode.
                    try:
                        line = decoded.readline(limit)
                    except Exception:
                        ledger.unknown.append({
                            "line": ledger.counts["physical_lines"] + 1,
                            "reason": "gzip_read_failed_unreturned_decompressed_bytes_unknown"})
                        raise
                    if not line:
                        break
                    ledger.charge(uncompressed_bytes=len(line), physical_lines=1)
                    line_number = ledger.counts["physical_lines"]
                    try:
                        ledger.check()
                        _require(len(line) <= MAX_LINE_BYTES, "physical line byte limit exceeded")
                        _require(line.endswith(b"\n") or len(line) < limit,
                                 "bounded line ended before newline/EOF")
                    except Exception:
                        ledger.unknown.append({"line": line_number, "reason": "consumed_line_not_decoded"})
                        raise
                    duplicate_keys = []

                    def object_pairs(pairs):
                        result = {}
                        for key, value in pairs:
                            if key in result:
                                duplicate_keys.append(True)
                            result[key] = value
                        return result

                    try:
                        document = json.loads(line, object_pairs_hook=object_pairs)
                    except Exception:
                        ledger.unknown.append({"line": line_number, "reason": "json_decode_failed_children_unknown"})
                        raise ExtractionStopped("source JSON decode failed") from None
                    _charge_children(document, ledger, line_number)
                    if duplicate_keys:
                        ledger.unknown.append({"line": line_number, "reason": "duplicate_keys_overwritten_children_unknown"})
                    ledger.check()
                    _require(not duplicate_keys, "duplicate source JSON keys")
                    line_hash = hashlib.sha256(line).hexdigest()
                    step = len(turns)
                    for row in _project(document, game_cfg, step, line_hash, ledger):
                        encoded = _json_bytes(row)
                        written = compact.write(encoded)
                        compact_hash.update(encoded[:written])
                        compact_bytes += written
                        _require(written == len(encoded), "partial compact write")
                        ledger.charge(compact_rows=1)
                        _add_summary(summary, row)
                        if first is None and row["flags"]["candidate_mismatch"]:
                            first = row
                        ledger.check()
                    turns.append({"step": step, "task_count": len(document["records"]),
                                  "crop_count": len(document["crop_records"]), "line_sha256": line_hash})
                    ledger.check()
            final_stat = os.fstat(raw.fileno())
            _require((final_stat.st_size, final_stat.st_mtime_ns, final_stat.st_ino) == identity,
                     "source changed during decode pass")
            _require(reader.remaining == 0, "gzip did not consume the registered compressed stream")
        expected = {
            "source_trace_files": 1, "compressed_sha_passes": 1, "gzip_json_passes": 1,
            "compressed_bytes_hashed": game_cfg["compressed_bytes"],
            "compressed_bytes_for_decode": game_cfg["compressed_bytes"],
            "compressed_source_bytes_read": 2 * game_cfg["compressed_bytes"],
            "physical_lines": expected_turns, "task_callback_objects": game_cfg["task_callbacks"],
            "candidate_objects": game_cfg["task_candidates"], "candidate_x_arrays": game_cfg["task_candidates"],
            "shared_x_arrays": game_cfg["task_callbacks"], "observation_objects": game_cfg["task_callbacks"],
            "configuration_objects": game_cfg["task_callbacks"], "crop_records": game_cfg["crop_records"],
            "compact_rows": game_cfg["task_callbacks"],
        }
        _require(all(ledger.counts[k] == v for k, v in expected.items()), "completed source counts differ")
        _require(len(turns) == expected_turns and not ledger.unknown, "incomplete/unknown source accounting")
        ledger.check()
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    # Never reread compact or the source for receipts/hashes, including failure.
    if not (output_dir / "compact.jsonl").exists():
        with (output_dir / "compact.jsonl").open("xb"):
            pass
    witness = {"schema": "saved-student-first-witness-v1", "game": game_cfg["game"],
               "source_sha256": source_hash, "first_mismatch": first,
               "provenance": None if first is None else {
                   "step": first["step"], "record_index": first["record_index"],
                   "source_line_sha256": first["source_line_sha256"]}}
    witness_hash, witness_bytes = _write_json(output_dir / "first-witness.json", witness)
    if failure is None:
        try:
            ledger.check()
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
    receipt = {"schema": "saved-student-extraction-v1", "passed": failure is None,
               "status": "failed" if failure else "complete",
               "game": game_cfg["game"], "source_sha256": source_hash,
               "compact_sha256": compact_hash.hexdigest(), "compact_bytes": compact_bytes,
               "counts": ledger.counts, "unknown_accounting": ledger.unknown, "turns": turns,
               "summary": summary, "first_mismatch": first,
               "first_witness_sha256": witness_hash, "first_witness_bytes": witness_bytes,
               "failure": failure, "elapsed_seconds": max(0.0, clock() - started)}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def extract_game(source_path, out_dir, game_cfg, budget, deadline):
    """Extract one registered historical stream, with no retry or model fallback.

    ``deadline`` is an absolute ``time.monotonic()`` value. ``budget`` contains
    exactly COUNTER_KEYS with this worker's remaining global source quotas.
    ``out_dir`` must not exist. Errors after creation return a failed receipt.
    """
    _require(game_cfg.get("seed") == 1714414373 and game_cfg.get("opponent") in ("v1", "market", "round2")
             and type(game_cfg.get("side")) is int and game_cfg["side"] in (0, 1),
             "game outside the fixed historical panel")
    _require(game_cfg.get("game") == f"holdout-{game_cfg['seed']}-{game_cfg['opponent']}-{game_cfg['side']}-trained",
             "registered game name differs")
    return _extract_game(source_path, out_dir, game_cfg, budget, deadline, expected_turns=719)
