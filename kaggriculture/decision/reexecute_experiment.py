"""Official step-for-step re-execution for the retained 52 land probe games."""
import argparse
import gzip
import json
import hashlib
from pathlib import Path
from decision.full_audit import reexecute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, help="New audit destination; historical audit is never overwritten")
    parser.add_argument("--labels", type=Path, help="Optional PRIVATE compressed execution labels; never publish raw labels")
    args = parser.parse_args()
    root = args.folder.resolve()
    target = args.output or root / "official-reexecution-v2.json"
    if target.exists():
        parser.error("Choose a separate reproduction; do not overwrite retained audit")
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    tagged = [(r, root / "games" / f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}") for r in read("ledger.json")]
    tagged += [(r, root / "parity" / r["policy"]) for r in read("parity.json")["games"]]
    tagged += [(r, root / "reproducibility" / str(r["side"])) for r in read("reproducibility.json")["games"]]
    rows = []
    labels = None
    if args.labels:
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        labels = gzip.open(args.labels, "xt", encoding="utf-8", compresslevel=1)
    source_root = Path(__file__).resolve().parents[1]
    source_names = ("decision/semantics.py", "decision/execution.py", "decision/full_audit.py", "decision/api.py",
                    "decision/reexecute_experiment.py", "decision/verify_audit.py", "decision/verify_mechanisms.py",
                    "decision/verify_experiment.py", "learning/round2_worker.py", "learning/mechanism_experiment.py",
                    "run_match.py", "learning/official_worker.py", "learning/optimize_round2.py")
    sources = {n: hashlib.sha256((source_root / n).read_bytes()).hexdigest() for n in source_names}
    historical = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.json")}
    try:
        for row, folder in tagged:
            raw = (folder / "replay.json.gz").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["replay_gzip_sha256"]
            replay = json.loads(gzip.decompress(raw))
            tag = {k: row[k] for k in ("policy", "opponent", "seed", "side")}
            sink = (lambda record: labels.write(json.dumps({"game": tag, "record": record}, separators=(",", ":")) + "\n")) if labels else None
            result = reexecute(replay, trace=True, label_sink=sink)
            assert result["final_rewards"] == row["banks"] and result["semantic_sha256"] == row["semantic_replay_sha256"]
            if folder.parent.name == "games":
                assert result["execution"]["successful_plants_new_land"][row["side"]] == row["mechanism"]["successful_plant_events_in_new_land"]
            rows.append(tag | {"replay_gzip_sha256": hashlib.sha256(raw).hexdigest()} | result)
            print(f"Official ordered/traced re-execution {len(rows)}/52", flush=True)
    finally:
        if labels:
            labels.close()
    assert len(rows) == 52
    assert all(hashlib.sha256((root / n).read_bytes()).hexdigest() == h for n, h in historical.items())
    assert all(hashlib.sha256((source_root / n).read_bytes()).hexdigest() == h for n, h in sources.items())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"passed": True, "audit_version": 2, "games": 52, "official_transitions": 52 * 719,
                                  "both_public_and_private_checked_each_step": True, "inventory_order_checked_each_step": True,
                                  "source_sha256": sources, "historical_json_sha256": historical,
                                  "historical_results_unchanged": True, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
