"""Aggregate sanitized, correlated single-turn comparisons without causality claims."""
from collections import Counter

from .samples import OPPONENTS, ROLES, WINDOWS, source_coordinates

CHECKS = ("request_equal", "reservations_equal", "stable_crop_equal",
          "final_context_equal", "next_turn_control_equal")


def _result(value):
    if type(value) is not dict or value.get("status") not in ("same", "different", "unknown"):
        raise ValueError("Invalid comparison summary")
    checks, official = value.get("checks"), value.get("official")
    if (type(checks) is not dict or set(checks) != set(CHECKS)
            or any(type(checks[k]) is not bool for k in CHECKS)):
        raise ValueError("Missing required controller state comparisons")
    if (type(official) is not dict or official.get("factual_reproduced") is not True
            or type(official.get("request_changed")) is not bool
            or type(official.get("state_changed")) is not bool
            or official.get("scope") != "one_official_transition"
            or official.get("complete_game") is not False
            or official.get("future_return_evaluated") is not False):
        raise ValueError("Missing factual official transition")
    if official["request_changed"] == checks["request_equal"]:
        raise ValueError("Controller and engine request comparisons disagree")
    status = value["status"]
    if status == "same" and (not all(checks.values()) or official["state_changed"]):
        raise ValueError("Same comparison has different official state")
    if status == "different" and checks["request_equal"]:
        raise ValueError("Different comparison has equal request")
    if status == "unknown" and (not checks["request_equal"] or all(checks.values())):
        raise ValueError("Unknown comparison contradicts controller checks")
    return value


def summarize(rows):
    """Report counts and the fixed exploration gate, not significance or value.

    Each row is a sanitized selected/empty window. A contrast means a third
    candidate has both a different request and a different physical next state;
    it does not imply that candidate has a lower value.
    """
    if type(rows) is not list or len(rows) != len(ROLES) * len(OPPONENTS) * len(WINDOWS):
        raise ValueError("Expected the complete fixed sample grid")
    expected = {(role, opponent, index) for role in ROLES for opponent in OPPONENTS
                for index in range(len(WINDOWS))}
    seen, teacher_counts, third_counts = set(), Counter(), Counter()
    qualified, by_role = [], {role: 0 for role in ROLES}
    empty = 0
    for row in rows:
        if type(row) is not dict:
            raise ValueError("Invalid sample row")
        key = (row.get("role"), row.get("opponent"), row.get("window"))
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate sample coordinate")
        seen.add(key)
        source = source_coordinates(key[0], key[1])
        if row.get("side") != source["side"] or row.get("arm") != source["arm"]:
            raise ValueError("Source rotation differs")
        if row.get("selection") == "empty":
            if (row.get("teacher") is not None or row.get("third") is not None
                    or any(row.get(k) is not None for k in ("step", "ordinal", "candidate_width"))):
                raise ValueError("Empty window has comparison data")
            empty += 1
            continue
        if row.get("selection") != "selected":
            raise ValueError("Invalid window selection")
        start, end = WINDOWS[key[2]]
        if (type(row.get("step")) is not int or not start <= row["step"] <= end
                or type(row.get("ordinal")) is not int or row["ordinal"] < 0
                or type(row.get("candidate_width")) is not int or not 2 <= row["candidate_width"] <= 256):
            raise ValueError("Selected sample lies outside its registered window or budget")
        teacher = _result(row.get("teacher"))
        teacher_counts[teacher["status"]] += 1
        third = row.get("third")
        if third is None:
            third_counts["not_attempted"] += 1
            continue
        third = _result(third)
        third_counts[third["status"]] += 1
        if (teacher["status"] == "same" and third["status"] == "different"
                and third["official"]["state_changed"]):
            qualified.append(key)
            by_role[key[0]] += 1
    roles = {key[0] for key in qualified}
    gate = (len(qualified) >= 4 and len(roles) >= 2
            and bool(roles.intersection({"F01", "F02"}))
            and bool(roles.intersection({"F03", "F04"})))
    return {
        "windows": len(rows), "empty_windows": empty, "selected_windows": len(rows) - empty,
        "teacher_status_counts": {k: teacher_counts[k] for k in ("same", "different", "unknown")},
        "third_status_counts": {k: third_counts[k] for k in ("same", "different", "unknown", "not_attempted")},
        "qualified_windows": len(qualified), "qualified_by_role": by_role,
        "exploratory_gate": "passed" if gate else "failed", "training": False,
        "independent_evaluation": False, "statistical_significance": "not_assessed",
        "interpretation": "conditioned same-turn diagnostic; no value or generalization claim",
    }
