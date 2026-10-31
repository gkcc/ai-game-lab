"""Tests for selection order, missing windows, and trace identity checks."""
import io
import json
import unittest
from copy import deepcopy

from learning.task_student_continuation_v1 import samples


TASK, CROP = "a" * 64, "b" * 64


def turn(step, records=()):
    counts = {key: 0 for key in samples.COUNT_KEYS}
    counts["controller_calls"] = step + 1
    turn_counts = {key: 0 for key in samples.COUNT_KEYS}
    turn_counts["controller_calls"] = 1
    turn_counts["task_forwards"] = len(records)
    turn_counts["task_rows"] = sum(len(r["inputs"]["candidates"]) for r in records)
    return {"schema_version": "task-choice-v1", "step": step,
            "task_checkpoint_sha256": TASK, "checkpoint_sha256": CROP,
            "load_errors": 0, "inference_errors": 0, "fallbacks": 0,
            "model_counts": counts, "turn_model_counts": turn_counts,
            "records": list(records), "crop_records": []}


def choice(teacher, student, ids):
    return {"teacher_label": teacher, "chosen": student, "source": "callback",
            "inputs": {"candidates": [{"id": x} for x in ids]}}


def stream(documents):
    counts = {key: 0 for key in samples.COUNT_KEYS}
    result = []
    for original in documents:
        doc = deepcopy(original)
        counts = {key: counts[key] + doc["turn_model_counts"][key] for key in counts}
        doc["model_counts"] = dict(counts)
        result.append((json.dumps(doc) + "\n").encode())
    return io.BytesIO(b"".join(result))


class SampleTests(unittest.TestCase):
    def test_predefined_model_and_seat_are_balanced(self):
        values = [samples.source_coordinates(role, opponent)
                  for role in samples.ROLES for opponent in samples.OPPONENTS]
        self.assertEqual({(x["arm"], x["side"]) for x in values},
                         {(a, s) for a in samples.ARMS for s in (0, 1)})
        self.assertEqual(sum(x["arm"] == "baseline" for x in values), 4)
        self.assertEqual(sum(x["side"] == 0 for x in values), 4)

    def test_first_disagreement_and_empty_window(self):
        docs = [turn(i) for i in range(719)]
        docs[10] = turn(10, [choice("x", "x", ["x"]), choice("t", "s", ["s", "t", "z"])])
        docs[11] = turn(11, [choice("a", "b", ["a", "b"])])
        docs[480] = turn(480, [choice("t", "s", ["s", "t"])])
        result = samples.scan(stream(docs), task_sha256=TASK, crop_sha256=CROP)
        first, middle, last = result["selected"]
        self.assertEqual((first["step"], first["ordinal"], first["third_candidate"]), (10, 1, "z"))
        self.assertEqual(result["selection_work"][0], {"turns_considered": 11, "callbacks_considered": 2})
        self.assertEqual(result["selection_work"][1], {"turns_considered": 240, "callbacks_considered": 0})
        self.assertEqual(first["previous_document"]["model_counts"]["controller_calls"], 10)
        self.assertIsNone(middle)
        self.assertEqual((last["step"], last["third_candidate"]), (480, None))

    def test_rejects_wrong_identity_missing_or_extra_line(self):
        docs = [turn(i) for i in range(719)]
        docs[400]["task_checkpoint_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            samples.scan(stream(docs), task_sha256=TASK, crop_sha256=CROP)
        docs[400]["task_checkpoint_sha256"] = TASK
        with self.assertRaises(ValueError):
            samples.scan(stream(docs[:-1]), task_sha256=TASK, crop_sha256=CROP)
        with self.assertRaises(ValueError):
            samples.scan(stream(docs + [turn(719)]), task_sha256=TASK, crop_sha256=CROP)

    def test_disagreement_must_reference_distinct_present_candidates(self):
        docs = [turn(i) for i in range(719)]
        docs[2] = turn(2, [choice("teacher", "student", ["student", "student"])])
        with self.assertRaises(ValueError):
            samples.scan(stream(docs), task_sha256=TASK, crop_sha256=CROP)

    def test_rejects_discontinuous_cross_turn_budget(self):
        docs = [turn(i) for i in range(719)]
        raw = stream(docs).getvalue().splitlines()
        damaged = json.loads(raw[80])
        damaged["model_counts"]["task_rows"] = 1
        raw[80] = json.dumps(damaged).encode()
        with self.assertRaises(ValueError):
            samples.scan(io.BytesIO(b"\n".join(raw) + b"\n"), task_sha256=TASK, crop_sha256=CROP)

    def test_rejects_model_work_missing_from_callback_records(self):
        docs = [turn(i) for i in range(719)]
        docs[80]["turn_model_counts"]["task_forwards"] = 1
        docs[80]["turn_model_counts"]["task_rows"] = 2
        with self.assertRaises(ValueError):
            samples.scan(stream(docs), task_sha256=TASK, crop_sha256=CROP)


if __name__ == "__main__":
    unittest.main()
