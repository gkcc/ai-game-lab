"""Six registered tiny synthetic checks; invoked only after source freeze.

No test discovery, original trace, model, replay, policy, or legacy compact is
used. Fixtures live in an owned OS-temp scratch run; only the compact verification
record is retained. Read-back consumption is added to the synthetic cap.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from . import extract


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _candidate(unit, x, y, op):
    return {"id": f"u{unit}:x{x}:y{y}:{op}", "unit": unit,
            "target": [x, y], "op": op, "x": [0.25]}


def _record(step, candidates, chosen, teacher, request):
    selected = candidates[chosen]
    return {
        "source": "callback", "teacher_label": candidates[teacher]["id"],
        "chosen": selected["id"], "unit": selected["unit"],
        "target": selected["target"], "op": selected["op"], "unit_request": request,
        "inputs": {"schema_version": "task-choice-v1", "shared_x": [0.5],
                   "observation": {"step": step, "player": 0},
                   "configuration": {"fixture": True}, "candidates": candidates},
    }


def _turn(step, records=(), crops=0):
    actions = [["PASS"] for _ in range(3)]
    for record in records:
        actions[record["unit"]] = record["unit_request"]
    return {"schema_version": "task-choice-v1", "step": step, "records": list(records),
            "crop_records": [{"x": [0.75], "logits": [0.1, 0.2]} for _ in range(crops)],
            "final_reservations": {"unit_actions": actions},
            "checkpoint_sha256": "b" * 64, "task_checkpoint_sha256": "a" * 64,
            "load_errors": 0, "inference_errors": 0, "fallbacks": 0}


def _caps():
    counts = dict.fromkeys(extract.COUNTER_KEYS, 10 * 1024 * 1024)
    counts.update(source_trace_files=6, compressed_sha_passes=6, gzip_json_passes=6,
                  physical_lines=24, task_callback_objects=48, candidate_objects=96,
                  candidate_x_arrays=96, shared_x_arrays=48, observation_objects=48,
                  configuration_objects=48, crop_records=24, compact_rows=48)
    return counts


def _check_deadline(deadline):
    _assert(time.monotonic() < deadline, "synthetic stage deadline exhausted")


def _empty_result():
    return {"schema": "saved-student-synthetic-verification-v1", "passed": False,
            "cases": [], "counts": dict.fromkeys(extract.COUNTER_KEYS, 0),
            "readback_counts": {"compact_rows": 0, "witness_objects": 0,
                                "bytes": 0, "json_decode_failures": 0},
            "unknown_accounting": [], "failure": None}


def _run_cases(out_dir, deadline):
    """Run exactly six cases inside the caller's already-owned scratch run."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=False)
    result = _empty_result()
    caps = _caps()

    def gate():
        _check_deadline(deadline)
        for key, value in result["counts"].items():
            _assert(value <= caps[key], "synthetic cumulative limit exceeded: " + key)

    def add_receipt(receipt):
        for key in extract.COUNTER_KEYS:
            result["counts"][key] += receipt["counts"][key]
        result["unknown_accounting"].extend(receipt["unknown_accounting"])
        gate()

    def read_json_lines(path, *, witness=False):
        """One bounded read/decode per new synthetic output, never source rereads."""
        values = []
        with path.open("rb") as stream:
            while True:
                gate()
                line = stream.readline(64 * 1024 + 1)
                if not line:
                    break
                result["counts"]["physical_lines"] += 1
                result["readback_counts"]["bytes"] += len(line)
                _assert(len(line) <= 64 * 1024, "synthetic readback line too large")
                try:
                    value = json.loads(line)
                except Exception:
                    result["readback_counts"]["json_decode_failures"] += 1
                    result["unknown_accounting"].append({"reason": "synthetic_readback_json_failed"})
                    raise
                row = value.get("first_mismatch") if witness else value
                if row is not None:
                    result["counts"]["task_callback_objects"] += 1
                    result["counts"]["candidate_objects"] += sum(
                        isinstance(row.get(k), dict) for k in ("teacher", "chosen"))
                if witness:
                    result["readback_counts"]["witness_objects"] += 1
                else:
                    result["readback_counts"]["compact_rows"] += 1
                gate()  # Charge decoded children before deadline/ceiling checks.
                values.append(value)
        return values

    def execute(name, documents, *, byte_limit=False, expire_after_decode=False):
        gate()
        case_dir = directory / name
        case_dir.mkdir()
        payload = b"".join(extract._json_bytes(d) for d in documents)
        compressed = gzip.compress(payload, mtime=0)
        source = case_dir / "synthetic.jsonl.gz"
        with source.open("xb") as output:
            output.write(compressed)
        callbacks = sum(len(d["records"]) for d in documents)
        candidates = sum(len(r["inputs"]["candidates"]) for d in documents for r in d["records"])
        cfg = {"game": name, "seed": 1714414373, "side": 0, "opponent": "v1",
               "trace_sha256": hashlib.sha256(compressed).hexdigest(),
               "compressed_bytes": len(compressed), "turns": len(documents),
               "task_callbacks": callbacks, "task_candidates": candidates,
               "crop_records": sum(len(d["crop_records"]) for d in documents),
               "task_checkpoint_sha256": "a" * 64, "crop_checkpoint_sha256": "b" * 64}
        budget = {k: caps[k] - result["counts"][k] for k in extract.COUNTER_KEYS}
        if byte_limit:
            budget["uncompressed_bytes"] = len(payload) - 1
        expired = False
        real_loads = extract.json.loads

        def after_decode(*args, **kwargs):
            nonlocal expired
            value = real_loads(*args, **kwargs)
            expired = True
            return value

        def clock():
            return deadline + 1 if expired else time.monotonic()

        try:
            if expire_after_decode:
                # A deterministic deadline immediately after json.loads checks
                # that already-decoded children are charged before the stop.
                extract.json.loads = after_decode
            receipt = extract._extract_game(source, case_dir / "output", cfg, budget, deadline,
                                            expected_turns=len(documents), clock=clock)
        finally:
            extract.json.loads = real_loads
        add_receipt(receipt)
        case = {"name": name, "passed": False, "extraction_status": receipt["status"],
                "counts": receipt["counts"], "failure": receipt["failure"]}
        result["cases"].append(case)
        return receipt, case, case_dir / "output", len(payload)

    try:
        a = _candidate(0, 1, 1, "DIG")
        b = _candidate(1, 2, 2, "WATER")
        c = _candidate(2, 1, 1, "DIG")
        first_docs = [
            _turn(0, crops=1),
            _turn(1, [_record(1, [a, b], 0, 0, ["EAST"]),
                      _record(1, [a, b], 1, 0, ["WATER"])], crops=1),
            _turn(2, [_record(2, [a, c], 1, 0, ["PICKUP", "FERTILIZER", 1])]),
        ]
        receipt, case, output, _ = execute("first-overlap-zero-turn", first_docs)
        _assert(receipt["passed"], "first-mismatch fixture rejected")
        rows = read_json_lines(output / "compact.jsonl")
        witnesses = read_json_lines(output / "first-witness.json", witness=True)
        _assert([(r["step"], r["record_index"]) for r in rows] == [(1, 0), (1, 1), (2, 0)],
                "physical callback order changed")
        first = receipt["first_mismatch"]
        _assert(first == rows[1] and witnesses[0]["first_mismatch"] == first,
                "earliest mismatch witness differs")
        _assert(all(first["flags"].values()), "overlapping mismatch flags lost")
        _assert(receipt["turns"][0]["task_count"] == 0 and receipt["turns"][0]["crop_count"] == 1,
                "zero-task turn lost")
        _assert(rows[2]["unit_request"] == ["PICKUP", "FERTILIZER", 1], "saved request order changed")
        _assert(receipt["summary"]["flags"] == {
            "candidate_mismatch": 2, "op_mismatch": 1, "unit_mismatch": 2, "target_mismatch": 1},
            "overlapping mismatch totals differ")
        _assert(receipt["summary"]["phase_teacher_op"]["late"]["PLANT"]["task_callbacks"] == 0,
                "zero marginal cell omitted")
        case["passed"] = True

        same = [_turn(0, [_record(0, [a, b], 0, 0, ["DIG"])], crops=1), _turn(1)]
        receipt, case, output, _ = execute("null-witness", same)
        _assert(receipt["passed"] and receipt["first_mismatch"] is None, "null witness fixture rejected")
        rows = read_json_lines(output / "compact.jsonl")
        witnesses = read_json_lines(output / "first-witness.json", witness=True)
        _assert(len(rows) == 1 and witnesses[0]["first_mismatch"] is None
                and witnesses[0]["provenance"] is None, "null witness manufactured a disagreement")
        case["passed"] = True

        invalid = [_turn(0, [_record(0, [a, b], 0, 0, ["DIG"])], crops=1)]
        wrong_player = deepcopy(invalid)
        wrong_player[0]["records"][0]["inputs"]["observation"]["player"] = 1
        receipt, case, _, _ = execute("wrong-player", wrong_player)
        _assert(not receipt["passed"] and "identity" in receipt["failure"]["message"],
                "wrong source identity accepted")
        _assert(receipt["counts"]["task_callback_objects"] == 1
                and receipt["counts"]["candidate_objects"] == 2
                and receipt["counts"]["crop_records"] == 1, "late identity failure lost decoded children")
        _assert(any(item["reason"] == "gzip_unreturned_buffered_bytes_unknown"
                    for item in receipt["unknown_accounting"]), "early schema stop hid gzip buffering")
        case["passed"] = True

        noncanonical = deepcopy(invalid)
        noncanonical[0]["records"][0]["inputs"]["candidates"][1]["id"] = "u01:x2:y2:WATER"
        receipt, case, _, _ = execute("noncanonical-candidate", noncanonical)
        _assert(not receipt["passed"] and "noncanonical" in receipt["failure"]["message"],
                "noncanonical candidate accepted")
        _assert(receipt["counts"]["candidate_x_arrays"] == 2, "invalid candidate array consumption lost")
        case["passed"] = True

        receipt, case, _, payload_size = execute("bounded-byte-stop", deepcopy(invalid), byte_limit=True)
        _assert(not receipt["passed"] and receipt["counts"]["uncompressed_bytes"] == payload_size - 1
                and receipt["counts"]["physical_lines"] == 1
                and receipt["counts"]["task_callback_objects"] == 0
                and receipt["unknown_accounting"], "byte stop lost consumed/unknown accounting")
        case["passed"] = True

        receipt, case, _, payload_size = execute("post-decode-deadline", deepcopy(invalid), expire_after_decode=True)
        _assert(not receipt["passed"] and "deadline" in receipt["failure"]["message"],
                "post-decode deadline was ignored")
        _assert(receipt["counts"]["physical_lines"] == 1
                and receipt["counts"]["uncompressed_bytes"] == payload_size
                and receipt["counts"]["task_callback_objects"] == 1
                and receipt["counts"]["candidate_objects"] == 2
                and receipt["counts"]["candidate_x_arrays"] == 2
                and receipt["counts"]["shared_x_arrays"] == 1
                and receipt["counts"]["observation_objects"] == 1
                and receipt["counts"]["configuration_objects"] == 1
                and receipt["counts"]["crop_records"] == 1,
                "post-decode deadline lost already-consumed children")
        _assert(any(item["reason"] == "gzip_unreturned_buffered_bytes_unknown"
                    for item in receipt["unknown_accounting"]), "early deadline stop hid gzip buffering")
        case["passed"] = True
        gate()
        result["passed"] = len(result["cases"]) == 6 and all(c["passed"] for c in result["cases"])
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    extract._write_json(directory / "regressions.json", result)
    return result


def run_regressions(out_dir, deadline, *, helper_sha256):
    """Use the user-required pinned temp helper, without enabling its workflow.

    A fresh UUID root confines the helper's startup sweep to this new empty
    directory. No older scratch root is enumerated or adopted. Cleanup is tried
    once; a rejected/locked run retains its original helper ownership marker.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=False)
    result = _empty_result()
    result.update(temporary_created=False, temporary_removed=False, cleanup_failed=False)
    root, scratch, root_identity = None, None, None
    phase = "helper_setup"
    try:
        _check_deadline(deadline)
        codex_home = os.environ.get("CODEX_HOME")
        _assert(codex_home and Path(codex_home).is_absolute(), "CODEX_HOME must be absolute")
        helper_path = Path(codex_home) / "skills" / "agent-workflow" / "scripts" / "artifacts.py"
        helper_bytes = helper_path.read_bytes()
        _assert(hashlib.sha256(helper_bytes).hexdigest() == helper_sha256,
                "registered temp helper SHA256 differs")
        spec = importlib.util.spec_from_file_location("_saved_student_temp_helper", helper_path)
        _assert(spec is not None, "temp helper import specification missing")
        helper = importlib.util.module_from_spec(spec)
        # Execute exactly the bytes just verified, without loading cached pyc or
        # rereading a potentially changed source through the default loader.
        exec(compile(helper_bytes, str(helper_path), "exec"), helper.__dict__)
        _check_deadline(deadline)
        os_temp = Path(tempfile.gettempdir()).resolve(strict=True)
        _assert(os_temp.is_absolute() and os_temp.is_dir(), "OS temp root is invalid")
        root = os_temp / ("saved-student-synthetic-" + uuid.uuid4().hex)
        _assert(root.is_absolute() and root.parent == os_temp
                and root.resolve().parent == os_temp and not os.path.lexists(root),
                "synthetic temp root is not fresh and contained")
        root.mkdir(exist_ok=False)
        result["temporary_created"] = True
        info = root.stat()
        root_identity = (info.st_dev, info.st_ino)
        phase = "scratch_setup"
        with helper.scratch_directory("saved-student-synthetic", root=root) as scratch:
            phase = "synthetic_cases"
            try:
                result = _run_cases(scratch / "cases", deadline)
                result.update(temporary_created=True, temporary_removed=False, cleanup_failed=False)
            finally:
                phase = "scratch_cleanup"
        # The context has already attempted its own cleanup. Never retry it.
        _assert(not os.path.lexists(scratch), "temp helper retained its scratch run")
        phase = "empty_root_cleanup"
        info = root.stat()
        _assert(root.parent == os_temp and root.resolve().parent == os_temp
                and (info.st_dev, info.st_ino) == root_identity,
                "owned empty root identity/containment changed")
        root.rmdir()  # One nonrecursive removal of only our now-empty UUID root.
        result["temporary_removed"] = not os.path.lexists(root) and not os.path.lexists(scratch)
        _assert(result["temporary_removed"], "temporary artifacts remain")
        phase = "completion"
        _check_deadline(deadline)
    except Exception as exc:
        result["passed"] = False
        result["cleanup_failed"] = (
            phase == "scratch_cleanup" and scratch is not None and os.path.lexists(scratch)
        ) or (phase == "empty_root_cleanup" and root is not None and os.path.lexists(root))
        result["failure"] = {"type": type(exc).__name__,
                             "message": "synthetic " + phase + " failed; see private cleanup-local record"}
        # No fallback cleanup, language switch, sweep, or second deletion attempt.
        extract._write_json(directory / "cleanup-local.json", {
            "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
            "retained_root": str(root) if root is not None and os.path.lexists(root) else None,
            "retained_scratch": str(scratch) if scratch is not None and os.path.lexists(scratch) else None,
            "cleanup_retries": 0,
        })
    extract._write_json(directory / "regressions.json", result)
    return result
