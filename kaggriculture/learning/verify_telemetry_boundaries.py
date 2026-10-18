"""Real complete official fixtures through the production isolated telemetry entry."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from mechanism_experiment import ROOT, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.exists(), "Preserve previous evidence"
    assert importlib.util.find_spec("kaggle_environments") is None
    python = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    subprocess.run([str(python), "-B", "-m", "decision.telemetry_fixtures", str(out / "fixtures")],
                   cwd=ROOT, check=True, capture_output=True, timeout=300)
    manifest = json.loads((out / "fixtures/manifest.json").read_text())
    rows = []
    for case in manifest:
        folder = out / "fixtures" / case["folder"]
        for side, expected in ((case["side"], case["expected_requests"]), (1-case["side"], 0)):
            result = telemetry(folder, {"side": side})
            assert result["non_pass_unit_requests"] == result["non_pass_unit_actions"] == expected
            assert result["action_count_version"] == 1 and result["execution_label_version"] == 2
            assert hashlib.sha256((folder / "replay.json.gz").read_bytes()).hexdigest() == case["replay_gzip_sha256"]
            rows.append({**case, "evaluated_side": side, "expected_requests": expected,
                         "actual_requests": result["non_pass_unit_requests"], "passed": True})
        print(f"Verified complete fixture {len(rows)//2}/4", flush=True)
    assert "kaggle_environments" not in sys.modules
    report = {"passed": True, "complete_fixture_games": 4, "isolated_calls": 8,
              "optimizer_engine_available": False, "optimizer_engine_imported": False,
              "raw_replays_unchanged": True, "cases": rows,
              "boundary": "Real official step trajectories and full reexecution in production telemetry; no mocks; request counts include no-effect and unknown commands"}
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
