"""Guard against counting uncertain or no-effect candidates as evidence."""
import unittest

from learning.task_student_continuation_v1 import analysis, samples


def result(status, request_equal, state_changed):
    checks = {key: True for key in analysis.CHECKS}
    checks["request_equal"] = request_equal
    if status == "unknown":
        checks["next_turn_control_equal"] = False
    return {"status": status, "checks": checks,
            "official": {"factual_reproduced": True, "request_changed": not request_equal,
                         "state_changed": state_changed, "scope": "one_official_transition",
                         "complete_game": False, "future_return_evaluated": False}}


def rows():
    return [{**samples.source_coordinates(role, opponent), "window": index,
             "selection": "empty", "teacher": None, "third": None}
            for role in samples.ROLES for opponent in samples.OPPONENTS for index in range(3)]


def select(row, teacher, third):
    row.update(selection="selected", step=samples.WINDOWS[row["window"]][0], ordinal=0,
               candidate_width=3, teacher=teacher, third=third)


class AnalysisTests(unittest.TestCase):
    def test_gate_needs_qualified_examples_from_both_role_groups(self):
        data = rows()
        for index in (0, 1, 2, 12):
            select(data[index], result("same", True, False), result("different", False, True))
        value = analysis.summarize(data)
        self.assertEqual(value["qualified_windows"], 4)
        self.assertEqual(value["exploratory_gate"], "passed")
        data[12]["third"] = result("different", False, False)
        value = analysis.summarize(data)
        self.assertEqual(value["qualified_windows"], 3)
        self.assertEqual(value["exploratory_gate"], "failed")

    def test_unknown_and_missing_third_never_qualify(self):
        data = rows()
        select(data[0], result("unknown", True, False), result("different", False, True))
        select(data[12], result("same", True, False), None)
        value = analysis.summarize(data)
        self.assertEqual(value["qualified_windows"], 0)
        self.assertEqual(value["teacher_status_counts"]["unknown"], 1)
        self.assertEqual(value["third_status_counts"]["not_attempted"], 1)

    def test_rejects_conflicting_official_observation_and_duplicate_grid(self):
        data = rows()
        select(data[0], result("same", True, True), None)
        with self.assertRaises(ValueError):
            analysis.summarize(data)
        data = rows()
        bad = result("same", True, False)
        bad["checks"]["next_turn_control_equal"] = False
        select(data[0], bad, None)
        with self.assertRaises(ValueError):
            analysis.summarize(data)
        data = rows()
        data[-1] = data[0].copy()
        with self.assertRaises(ValueError):
            analysis.summarize(data)


if __name__ == "__main__":
    unittest.main()
