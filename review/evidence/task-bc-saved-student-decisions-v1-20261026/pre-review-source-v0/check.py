"""Independent, one-pass verifier for the newly extracted six-game compact.

Only ``cfg`` and the new receipt/compact/witness files are inputs.  This module
does not import the extractor or open any original trace, replay, or model.
``check`` returns a certificate; a failure raises after retaining consumed
counters.  A certificate is usable only when the call returns successfully and
there is no failure.json.  All output files are created exclusively.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time


OPS = ("DIG", "FERTILIZE", "HARVEST", "PLANT", "WATER")
REQUEST_OPS = OPS + ("PASS", "EAST", "WEST", "SOUTH", "NORTH", "DROP", "PICKUP")
PHASES = ("early", "middle", "late")
FLAGS = ("candidate_mismatch", "op_mismatch", "unit_mismatch", "target_mismatch")
ROW_KEYS = {"game", "step", "record_index", "candidate_count", "source_line_sha256",
            "teacher", "chosen", "unit_request", "flags"}
COUNT_KEYS = {
    "source_trace_files", "compressed_sha_passes", "gzip_json_passes",
    "compressed_bytes_hashed", "compressed_bytes_for_decode",
    "compressed_source_bytes_read", "uncompressed_bytes", "physical_lines",
    "task_callback_objects", "candidate_objects", "candidate_x_arrays",
    "shared_x_arrays", "observation_objects", "configuration_objects",
    "crop_records", "compact_rows",
}
RECEIPT_KEYS = {
    "schema", "status", "passed", "game", "source_sha256", "compact_sha256", "compact_bytes",
    "counts", "unknown_accounting", "turns", "summary", "first_mismatch",
    "first_witness_sha256", "first_witness_bytes", "failure", "elapsed_seconds",
}
# The source selection is fixed before this verifier can consume any compact.
SELECTION = (
    ("v1", 0, 6825738, 2053, 98614, 153,
     "a07ac7ccecd84d492bc1b25adfde238bc423692e4c574ff31782e2be174269a5"),
    ("v1", 1, 6822335, 2053, 98614, 153,
     "02785d557095caecfe7647c93239e7419c744837e95a7520c1bc48f808e126cc"),
    ("market", 0, 6671785, 2013, 91815, 152,
     "9490f1a8ef308d9a820f99f97d50e03f7f420f05b4bb5acdef6568315486cc26"),
    ("market", 1, 6681928, 2013, 91815, 152,
     "82fcc219f4c2804516395804601979382d8f9beba150bc7e9f0a34d2dccbb8cb"),
    ("round2", 0, 6829322, 2050, 97408, 155,
     "11e6192e08f4977378bede71232139a1a59635306b923f0934fe9596902e68c1"),
    ("round2", 1, 6829781, 2050, 97408, 155,
     "721d8b85e68e8c60f69a420a6505b4305b63f95988b35fc8bb634860f44f7843"),
)
MAX_ROWS = 12232
MAX_LINE_BYTES = 65536
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_WITNESS_BYTES = 128 * 1024
INDEPENDENCE_LIMIT = (
    "Independent verification of the new compact and bound producer receipts; "
    "not independent reproduction of original traces. One historical seed, six "
    "correlated games; conditional choices only, without official effects, "
    "economic-loss attribution, or across-seed inference."
)


class CheckFailure(ValueError):
    """A stable, non-private failure code suitable for the retained certificate."""


def _need(condition, code):
    if not condition:
        raise CheckFailure(code)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _keys(value, expected, code):
    _need(type(value) is dict and set(value) == set(expected), code)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _constant(_value):
    raise CheckFailure("nonfinite_json_constant")


def _decode(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_constant)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _same(left, right):
    # Keep JSON scalar types distinct (True is not 1); never sort array values.
    return _json_bytes(left) == _json_bytes(right)


def _write_new(path, value):
    data = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def _json_value(value, depth=0):
    _need(depth <= 32, "request_nesting_limit")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite_request_number")
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            _need(type(key) is str, "request_object_key")
            _json_value(item, depth + 1)
        return
    raise CheckFailure("non_json_request_value")


def _descriptor(value):
    _keys(value, {"id", "unit", "target", "op"}, "descriptor_keys")
    unit, target, op = value["unit"], value["target"], value["op"]
    _need(_integer(unit), "descriptor_unit")
    _need(type(target) is list and len(target) == 2
          and all(_integer(part) for part in target), "descriptor_target")
    _need(type(op) is str and op in OPS, "descriptor_op")
    _need(value["id"] == f"u{unit}:x{target[0]}:y{target[1]}:{op}",
          "descriptor_canonical_id")
    return unit, target[0], target[1], op


def _cell():
    return {"task_callbacks": 0, "task_candidates": 0,
            "flags": {flag: 0 for flag in FLAGS}}


def _summary():
    result = _cell()
    result["phase_teacher_op"] = {
        phase: {op: _cell() for op in OPS} for phase in PHASES
    }
    result["op_confusion"] = {teacher: {chosen: 0 for chosen in OPS} for teacher in OPS}
    result["request_op_counts"] = {op: 0 for op in REQUEST_OPS}
    return result


def _phase(step):
    return "early" if step < 240 else "middle" if step < 480 else "late"


def _tally(summary, row, flags):
    for cell in (summary, summary["phase_teacher_op"][_phase(row["step"])][row["teacher"]["op"]]):
        cell["task_callbacks"] += 1
        cell["task_candidates"] += row["candidate_count"]
        for name in FLAGS:
            cell["flags"][name] += int(flags[name])
    summary["op_confusion"][row["teacher"]["op"]][row["chosen"]["op"]] += 1
    summary["request_op_counts"][row["unit_request"][0]] += 1


def _safe_marginals(summary):
    phases = {phase: _cell() for phase in PHASES}
    classes = {op: _cell() for op in OPS}
    for phase in PHASES:
        for op in OPS:
            cell = summary["phase_teacher_op"][phase][op]
            for target in (phases[phase], classes[op]):
                target["task_callbacks"] += cell["task_callbacks"]
                target["task_candidates"] += cell["task_candidates"]
                for name in FLAGS:
                    target["flags"][name] += cell["flags"][name]
    return {**summary, "phase_totals": phases, "teacher_op_totals": classes}


def _configuration(cfg):
    selected = cfg["selected_games"]
    _need(type(selected) is list and len(selected) == 6, "exact_six_games")
    for entry, frozen in zip(selected, SELECTION):
        opponent, side, size, callbacks, candidates, crops, sha = frozen
        expected = {
            "game": f"holdout-1714414373-{opponent}-{side}-trained",
            "seed": 1714414373, "side": side, "opponent": opponent,
            "trace_sha256": sha, "compressed_bytes": size, "turns": 719,
            "task_callbacks": callbacks, "task_candidates": candidates,
            "crop_records": crops,
        }
        _need(type(entry) is dict, "game_configuration_object")
        _need(all(key in entry and _same(entry[key], value)
                  for key, value in expected.items()), "frozen_game_selection")
        for key in ("task_checkpoint_sha256", "crop_checkpoint_sha256"):
            _need(_sha(entry[key]), "checkpoint_identity_shape")
        _need(_integer(entry["expected_conditional_mismatches"])
              and entry["expected_conditional_mismatches"] <= callbacks,
              "historical_mismatch_count")
        _keys(entry["expected_selected_ops"], OPS, "historical_selected_ops_keys")
        _need(all(_integer(value) for value in entry["expected_selected_ops"].values())
              and sum(entry["expected_selected_ops"].values()) == callbacks,
              "historical_selected_ops_count")
    for key in ("task_checkpoint_sha256", "crop_checkpoint_sha256"):
        _need(len({entry[key] for entry in selected}) == 1, "common_checkpoint_identity")
    hashes = cfg["primary_receipts_sha256"]
    _keys(hashes, [entry["game"] for entry in selected], "receipt_binding_games")
    _need(all(_sha(value) for value in hashes.values()), "receipt_binding_hash")
    return selected, hashes


def _receipt(receipt, selected):
    _keys(receipt, RECEIPT_KEYS, "receipt_keys")
    _need(receipt["schema"] == "saved-student-extraction-v1"
          and receipt["status"] == "complete" and receipt["passed"] is True,
          "receipt_completion")
    _need(receipt["game"] == selected["game"]
          and receipt["source_sha256"] == selected["trace_sha256"], "receipt_source_identity")
    _need(receipt["unknown_accounting"] == [] and receipt["failure"] is None,
          "receipt_accounting_complete")
    _need(type(receipt["elapsed_seconds"]) in (int, float)
          and math.isfinite(receipt["elapsed_seconds"])
          and 0 <= receipt["elapsed_seconds"] <= 60, "receipt_elapsed_limit")
    for name in ("compact_sha256", "first_witness_sha256"):
        _need(_sha(receipt[name]), "receipt_output_hash")
    _need(_integer(receipt["compact_bytes"])
          and receipt["compact_bytes"] <= selected["task_callbacks"] * MAX_LINE_BYTES,
          "receipt_compact_bytes")
    _need(_integer(receipt["first_witness_bytes"], 1)
          and receipt["first_witness_bytes"] <= MAX_WITNESS_BYTES,
          "receipt_witness_bytes")
    counts = receipt["counts"]
    _keys(counts, COUNT_KEYS, "receipt_count_keys")
    _need(all(_integer(value) for value in counts.values()), "receipt_count_types")
    expected_counts = {
        "source_trace_files": 1, "compressed_sha_passes": 1, "gzip_json_passes": 1,
        "compressed_bytes_hashed": selected["compressed_bytes"],
        "compressed_bytes_for_decode": selected["compressed_bytes"],
        "compressed_source_bytes_read": selected["compressed_bytes"] * 2,
        "physical_lines": selected["turns"], "task_callback_objects": selected["task_callbacks"],
        "candidate_objects": selected["task_candidates"],
        "candidate_x_arrays": selected["task_candidates"],
        "shared_x_arrays": selected["task_callbacks"],
        "observation_objects": selected["task_callbacks"],
        "configuration_objects": selected["task_callbacks"],
        "crop_records": selected["crop_records"], "compact_rows": selected["task_callbacks"],
    }
    _need(all(counts[key] == value for key, value in expected_counts.items()),
          "receipt_inventory_counts")
    _need(0 < counts["uncompressed_bytes"] <= 256 * 1024 * 1024,
          "receipt_uncompressed_limit")
    turns = receipt["turns"]
    _need(type(turns) is list and len(turns) == 719, "receipt_turn_count")
    for step, turn in enumerate(turns):
        _keys(turn, {"step", "task_count", "crop_count", "line_sha256"}, "turn_keys")
        _need(_integer(turn["step"]) and turn["step"] == step, "turn_order")
        _need(_integer(turn["task_count"]) and _integer(turn["crop_count"]), "turn_counts")
        _need(_sha(turn["line_sha256"]), "turn_source_hash")
    _need(sum(turn["task_count"] for turn in turns) == selected["task_callbacks"]
          and sum(turn["crop_count"] for turn in turns) == selected["crop_records"],
          "turn_inventory_totals")
    return counts, turns


def check(out, cfg, deadline):
    """Check the new six-game compact once within an absolute monotonic deadline."""
    started = time.monotonic()
    _need(type(deadline) in (int, float) and math.isfinite(deadline), "deadline_type")
    cutoff = min(deadline, started + 120.0)
    folder = Path(out) / "check"
    counters = {
        "receipt_files_read": 0, "receipt_bytes_read": 0,
        "compact_files_opened": 0, "compact_rows_reserved": 0,
        "compact_rows_read": 0, "compact_rows_decoded": 0,
        "compact_bytes_read": 0, "witness_files_read": 0, "witness_bytes_read": 0,
        "games_completed": 0, "turns_verified": 0, "zero_task_turns_verified": 0,
    }
    owned = False

    def tick():
        _need(time.monotonic() < cutoff, "compact_review_deadline")

    def small_file(path, kind, limit, expected_sha, expected_bytes=None):
        tick()
        size = path.stat().st_size
        _need(0 < size <= limit, f"{kind}_file_size_limit")
        if expected_bytes is not None:
            _need(size == expected_bytes, f"{kind}_byte_identity")
        counters[f"{kind}_files_read"] += 1
        with path.open("rb") as stream:
            raw = stream.read(size)
            counters[f"{kind}_bytes_read"] += len(raw)
            _need(len(raw) == size and path.stat().st_size == size,
                  f"{kind}_file_changed")
        tick()
        _need(hashlib.sha256(raw).hexdigest() == expected_sha, f"{kind}_hash_identity")
        value = _decode(raw)
        tick()
        return value

    try:
        tick()
        folder.mkdir()
        owned = True
        _write_new(folder / "started.json", {
            "schema": "saved-student-compact-check-started-v1", "status": "started",
            "review_seconds_max": 120, "compact_rows_max": MAX_ROWS,
            "independent_compact_passes_max": 1, "counters": counters.copy(),
            "independence_limit": INDEPENDENCE_LIMIT,
        })
        selected_games, bindings = _configuration(cfg)
        aggregate, games, identities = _summary(), [], []
        primary_totals = {key: 0 for key in COUNT_KEYS}
        for selected in selected_games:
            tick()
            game = selected["game"]
            game_dir = Path(out) / "extracted" / game
            receipt = small_file(game_dir / "receipt.json", "receipt",
                                 MAX_RECEIPT_BYTES, bindings[game])
            source_counts, turns = _receipt(receipt, selected)
            for key in COUNT_KEYS:
                primary_totals[key] += source_counts[key]
            _need(primary_totals["uncompressed_bytes"] <= 1024 * 1024 * 1024,
                  "total_uncompressed_limit")
            summary, first, digest = _summary(), None, hashlib.sha256()
            compact = game_dir / "compact.jsonl"
            size = compact.stat().st_size
            _need(size == receipt["compact_bytes"], "compact_byte_identity")
            bytes_read = 0
            counters["compact_files_opened"] += 1
            with compact.open("rb") as stream:
                for turn in turns:
                    tick()
                    for index in range(turn["task_count"]):
                        tick()
                        _need(counters["compact_rows_reserved"] < MAX_ROWS, "compact_row_limit")
                        counters["compact_rows_reserved"] += 1
                        raw = stream.readline(MAX_LINE_BYTES + 1)
                        counters["compact_bytes_read"] += len(raw)
                        bytes_read += len(raw)
                        counters["compact_rows_read"] += int(bool(raw))
                        _need(raw and len(raw) <= MAX_LINE_BYTES and raw.endswith(b"\n"),
                              "compact_line_size_or_termination")
                        digest.update(raw)
                        row = _decode(raw)
                        counters["compact_rows_decoded"] += 1
                        _keys(row, ROW_KEYS, "compact_row_keys")
                        _need(row["game"] == game and _integer(row["step"])
                              and row["step"] == turn["step"]
                              and _integer(row["record_index"])
                              and row["record_index"] == index, "compact_original_order")
                        _need(row["source_line_sha256"] == turn["line_sha256"],
                              "compact_source_line_identity")
                        _need(_integer(row["candidate_count"], 1)
                              and row["candidate_count"] <= selected["task_candidates"],
                              "compact_candidate_count")
                        teacher, chosen = _descriptor(row["teacher"]), _descriptor(row["chosen"])
                        flags = {"candidate_mismatch": teacher != chosen,
                                 "op_mismatch": teacher[3] != chosen[3],
                                 "unit_mismatch": teacher[0] != chosen[0],
                                 "target_mismatch": teacher[1:3] != chosen[1:3]}
                        _keys(row["flags"], FLAGS, "compact_flags_keys")
                        _need(all(type(row["flags"][key]) is bool
                                  and row["flags"][key] == flags[key] for key in FLAGS),
                              "compact_recomputed_flags")
                        _need(row["candidate_count"] > 1 or not flags["candidate_mismatch"],
                              "single_candidate_disagreement")
                        request = row["unit_request"]
                        _need(type(request) is list and request and type(request[0]) is str
                              and request[0] in REQUEST_OPS, "compact_request_op")
                        _json_value(request)
                        _tally(summary, row, flags)
                        _tally(aggregate, row, flags)
                        _need(summary["task_candidates"] <= selected["task_candidates"],
                              "compact_candidate_total_limit")
                        if first is None and flags["candidate_mismatch"]:
                            first = row
                        tick()
                    counters["turns_verified"] += 1
                    counters["zero_task_turns_verified"] += int(turn["task_count"] == 0)
                # File size proves no trailing rows without reading a second pass or sentinel row.
                _need(stream.tell() == size and compact.stat().st_size == size,
                      "compact_extra_or_changed_bytes")
            tick()
            _need(bytes_read == receipt["compact_bytes"]
                  and digest.hexdigest() == receipt["compact_sha256"], "compact_hash_identity")
            _need(summary["task_callbacks"] == selected["task_callbacks"]
                  and summary["task_candidates"] == selected["task_candidates"],
                  "compact_inventory_totals")
            _need(_same(summary, receipt["summary"]), "independent_summary_mismatch")
            _need(_same(first, receipt["first_mismatch"]), "receipt_earliest_mismatch")
            _need(summary["flags"]["candidate_mismatch"] == selected["expected_conditional_mismatches"],
                  "historical_conditional_mismatch_total")
            chosen_totals = {op: sum(summary["op_confusion"][teacher][op] for teacher in OPS)
                             for op in OPS}
            _need(_same(chosen_totals, selected["expected_selected_ops"]), "historical_selected_ops")
            witness = small_file(game_dir / "first-witness.json", "witness", MAX_WITNESS_BYTES,
                                 receipt["first_witness_sha256"], receipt["first_witness_bytes"])
            _keys(witness, {"schema", "game", "source_sha256", "first_mismatch", "provenance"},
                  "witness_keys")
            provenance = (None if first is None else {key: first[key]
                          for key in ("step", "record_index", "source_line_sha256")})
            expected_witness = {"schema": "saved-student-first-witness-v1", "game": game,
                                "source_sha256": selected["trace_sha256"],
                                "first_mismatch": first, "provenance": provenance}
            _need(_same(witness, expected_witness), "witness_earliest_identity")
            first_metadata = (None if first is None else {
                "step": first["step"], "record_index": first["record_index"],
                "source_line_sha256": first["source_line_sha256"], "flags": first["flags"],
            })
            games.append({"game": game, "seed": selected["seed"], "side": selected["side"],
                          "opponent": selected["opponent"], "turns": 719,
                          "zero_task_turns": sum(turn["task_count"] == 0 for turn in turns),
                          "crop_records": selected["crop_records"],
                          "summary": _safe_marginals(summary), "first_mismatch": first_metadata})
            identities.append({"game": game, "source_sha256": selected["trace_sha256"],
                               "receipt_sha256": bindings[game],
                               "compact_sha256": digest.hexdigest(), "compact_bytes": bytes_read,
                               "first_witness_sha256": receipt["first_witness_sha256"]})
            counters["games_completed"] += 1
        tick()
        _need(counters["games_completed"] == 6 and counters["compact_rows_reserved"] == MAX_ROWS
              and counters["compact_rows_read"] == MAX_ROWS
              and counters["compact_rows_decoded"] == MAX_ROWS
              and counters["turns_verified"] == 4314, "complete_coverage")
        witness_count = sum(game["first_mismatch"] is not None for game in games)
        _need(0 <= witness_count <= 6, "at_most_six_witnesses")
        safe_summary = {
            "schema": "saved-student-safe-aggregate-summary-v1", "games": games,
            "aggregate": _safe_marginals(aggregate), "first_mismatch_games": witness_count,
            "independence_limit": INDEPENDENCE_LIMIT,
        }
        summary_sha = _write_new(folder / "summary.json", safe_summary)
        tick()
        certificate = {
            "schema": "saved-student-independent-compact-check-v1", "passed": True,
            **counters, "summary_sha256": summary_sha, "identities": identities,
            "bound_producer_counts": primary_totals,
            "compact_passes_per_file": 1, "first_mismatch_games": witness_count,
            "checker_source_trace_reads": 0, "checker_replay_reads": 0,
            "checker_old_compact_reads": 0, "checker_model_reads": 0,
            "checker_scoring_calls": 0, "checker_candidate_reconstructions": 0,
            "checker_official_transitions": 0, "checker_new_games": 0,
            "checker_training_runs": 0, "elapsed_seconds": time.monotonic() - started,
            "independence_limit": INDEPENDENCE_LIMIT,
        }
        tick()
        _write_new(folder / "complete.json", certificate)
        tick()
        return certificate
    except BaseException as error:
        if owned:
            failure = {
                "schema": "saved-student-independent-compact-check-failure-v1", "passed": False,
                **counters, "elapsed_seconds": time.monotonic() - started,
                "failure": {"type": type(error).__name__,
                            "code": str(error) if isinstance(error, CheckFailure) else "checker_failed"},
                "independence_limit": INDEPENDENCE_LIMIT,
            }
            _write_new(folder / "failure.json", failure)
        raise
