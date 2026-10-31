"""Select fixed student/teacher disagreements from one authenticated trace.

The caller authenticates the full compressed trace, source game, seed role and
model identities before supplying the stream. Returned documents contain raw
private observations and must remain in memory or a private local artifact.
This module does not run a game or infer economic value from a disagreement.
"""
import json


WINDOWS = ((0, 239), (240, 479), (480, 718))
ROLES = ("F01", "F02", "F03", "F04")
OPPONENTS = ("market", "round2")
ARMS = ("baseline", "control")
MAX_LINE = 8 * 1024 * 1024
MAX_DECODED = 512 * 1024 * 1024
COUNT_KEYS = ("task_forwards", "task_rows", "crop_forwards", "crop_rows",
              "gate_evaluations", "controller_calls", "teacher_proposals")


def source_coordinates(role, opponent):
    """Choose one arm/seat before reading a trajectory or outcome."""
    if role not in ROLES or opponent not in OPPONENTS:
        raise ValueError("Unknown sample coordinate")
    i, j = ROLES.index(role), OPPONENTS.index(opponent)
    return {"role": role, "opponent": opponent, "side": (i + j) % 2,
            "arm": ARMS[(i // 2 + j) % 2]}


def _invalid_constant(_value):
    raise ValueError("Non-finite trace value")


def scan(stream, *, task_sha256, crop_sha256, guard=lambda: None):
    """Read all 719 lines, choosing the first differing callback per window.

    Each selected value includes a private whole-turn document and the zero-
    based callback ordinal. Empty windows remain None. The selection does not
    filter by operation, candidate count, model outcome or later game result.
    The upstream caller must authenticate the compressed bytes and enforce
    memory/wall-clock budgets; the guard is invoked between physical lines.
    """
    if not callable(guard) or not all(isinstance(x, str) and len(x) == 64
                                      for x in (task_sha256, crop_sha256)):
        raise ValueError("Invalid scan contract")
    selected = [None] * len(WINDOWS)
    selection_work = [{"turns_considered": 0, "callbacks_considered": 0} for _ in WINDOWS]
    previous_counts = {key: 0 for key in COUNT_KEYS}
    previous_document = None
    decoded = 0
    for step in range(719):
        guard()
        raw = stream.readline(MAX_LINE + 1)
        if not isinstance(raw, bytes) or not raw or len(raw) > MAX_LINE:
            raise ValueError("Missing or oversized trace line")
        decoded += len(raw)
        if decoded > MAX_DECODED:
            raise ValueError("Decoded trace budget exceeded")
        doc = json.loads(raw, parse_constant=_invalid_constant)
        if (not isinstance(doc, dict) or doc.get("schema_version") != "task-choice-v1"
                or type(doc.get("step")) is not int or doc["step"] != step
                or doc.get("task_checkpoint_sha256") != task_sha256
                or doc.get("checkpoint_sha256") != crop_sha256
                or any(type(doc.get(k)) is not int or doc[k] != 0
                       for k in ("load_errors", "inference_errors", "fallbacks"))
                or not isinstance(doc.get("records"), list)
                or not isinstance(doc.get("crop_records"), list)):
            raise ValueError("Trace identity, step or runtime status differs")
        counts, turn = doc.get("model_counts"), doc.get("turn_model_counts")
        if (not isinstance(counts, dict) or not isinstance(turn, dict)
                or set(counts) != set(COUNT_KEYS) or set(turn) != set(COUNT_KEYS)
                or any(type(counts[k]) is not int or type(turn[k]) is not int
                       or turn[k] < 0 or counts[k] != previous_counts[k] + turn[k]
                       for k in COUNT_KEYS)
                or turn["controller_calls"] != 1):
            raise ValueError("Trace model budget continuity differs")
        candidates_per_record = []
        for record in doc["records"]:
            if not isinstance(record, dict) or record.get("source") != "callback":
                raise ValueError("Invalid student task callback")
            inputs = record.get("inputs")
            candidates = inputs.get("candidates") if isinstance(inputs, dict) else None
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("Missing candidate list")
            ids = [c.get("id") if isinstance(c, dict) else None for c in candidates]
            teacher, chosen = record.get("teacher_label"), record.get("chosen")
            if (any(not isinstance(i, str) or not i for i in ids)
                    or len(ids) != len(set(ids)) or teacher not in ids or chosen not in ids):
                raise ValueError("Invalid candidate identities")
            candidates_per_record.append(ids)
        if (turn["task_forwards"] != len(doc["records"])
                or turn["task_rows"] != sum(map(len, candidates_per_record))
                or turn["crop_forwards"] != len(doc["crop_records"])
                or turn["crop_rows"] != len(doc["crop_records"])
                or any(not isinstance(r, dict) or r.get("source") != "model"
                       for r in doc["crop_records"])):
            raise ValueError("Trace callbacks do not account for model work")
        window = next(i for i, (start, end) in enumerate(WINDOWS) if start <= step <= end)
        if selected[window] is None:
            selection_work[window]["turns_considered"] += 1
        for ordinal, record in enumerate(doc["records"] if selected[window] is None else []):
            selection_work[window]["callbacks_considered"] += 1
            teacher, chosen = record.get("teacher_label"), record.get("chosen")
            if teacher != chosen:
                ids = candidates_per_record[ordinal]
                third = next((i for i in ids if i != teacher and i != chosen), None)
                selected[window] = {"step": step, "ordinal": ordinal,
                                    "third_candidate": third, "document": doc,
                                    "previous_document": previous_document}
                break
        previous_counts = dict(counts)
        previous_document = {key: doc[key] for key in
            ("step", "task_checkpoint_sha256", "checkpoint_sha256", "load_errors",
             "inference_errors", "fallbacks", "model_counts")}
    guard()
    if stream.read(1):
        raise ValueError("Trace contains extra data")
    return {"selected": selected, "selection_work": selection_work,
            "decoded_bytes": decoded, "lines": 719}
