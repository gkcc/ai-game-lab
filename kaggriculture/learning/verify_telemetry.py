"""Regression: optimizer calls real game telemetry without importing the engine."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from mechanism_experiment import telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Retained complete game with result.json and replay.json.gz")
    parser.add_argument("--side", type=int, choices=(0, 1), required=True)
    parser.add_argument("--expected-requests", type=int, help="Optional exact request-count assertion, including zero")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite retained evidence"
    assert importlib.util.find_spec("kaggle_environments") is None, "Run in isolated learning venv"
    result = telemetry(args.folder, {"side": args.side})
    assert "kaggle_environments" not in sys.modules
    assert result["execution_label_version"] == 2
    assert result["non_pass_unit_actions"] == result["non_pass_unit_requests"] >= 0
    assert result["action_count_version"] == 1
    if args.expected_requests is not None:
        assert result["non_pass_unit_requests"] == args.expected_requests
    assert len(result["execution_effect_sha256"]) == 64
    report = {"passed": True, "optimizer_engine_available": False,
              "optimizer_engine_imported": False, "official_worker_completed": True,
              "mechanism": result,
              "boundary": "Actual isolated telemetry entry over a complete retained replay; not a new optimization experiment"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
