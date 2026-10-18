"""Recheck every retained game through the production isolated telemetry entry."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from mechanism_experiment import ROOT, telemetry

read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert importlib.util.find_spec("kaggle_environments") is None
    root = args.folder.resolve()
    cert = read(args.certificate)
    tagged = [(r, root / "games" / f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}") for r in read(root / "ledger.json")]
    tagged += [(r, root / "parity" / r["policy"]) for r in read(root / "parity.json")["games"]]
    tagged += [(r, root / "reproducibility" / str(r["side"])) for r in read(root / "reproducibility.json")["games"]]
    names = ("decision/telemetry_worker.py", "decision/full_audit.py", "decision/execution.py",
             "decision/semantics.py", "decision/verify_mechanisms.py", "learning/mechanism_experiment.py",
             "learning/verify_telemetry_panel.py")
    sources = {n: sha(ROOT / n) for n in names}
    assert len(tagged) == len(cert["rows"]) == 52
    rows = []
    for (r, folder), expected in zip(tagged, cert["rows"]):
        tag = {k: r[k] for k in ("policy", "opponent", "seed", "side")}
        assert all(expected[k] == v for k, v in tag.items())
        assert sha(folder / "replay.json.gz") == r["replay_gzip_sha256"] == expected["replay_gzip_sha256"]
        result = telemetry(folder, r)
        assert sha(folder / "replay.json.gz") == r["replay_gzip_sha256"]
        assert result["non_pass_unit_actions"] == result["non_pass_unit_requests"] >= 0
        assert result["execution_effect_sha256"] == expected["execution"]["ordered_effect_sha256"]
        assert result["successful_plant_events_in_new_land"] == expected["execution"]["successful_plants_new_land"][r["side"]]
        if len(rows) < 48:
            for k, v in r["mechanism"].items():
                if k not in ("execution_label_version", "execution_effect_sha256"):
                    assert result[k] == v, k
        rows.append(tag | {"passed": True, "replay_gzip_sha256": r["replay_gzip_sha256"],
                           "non_pass_unit_requests": result["non_pass_unit_requests"],
                           "execution_effect_sha256": result["execution_effect_sha256"]})
        print(f"Isolated telemetry {len(rows)}/52", flush=True)
    assert "kaggle_environments" not in sys.modules
    assert all(sha(ROOT / n) == h for n, h in sources.items())
    report = {"passed": True, "games": 52, "source_sha256": sources,
              "reference_certificate_sha256": sha(args.certificate), "action_count_version": 1,
              "optimizer_engine_imported": False, "raw_replays_unchanged": True,
              "all_48_historical_panel_metrics_unchanged": True, "rows": rows,
              "boundary": "Each production telemetry call independently reexecutes the complete official replay; retained games, not new statistical samples"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
