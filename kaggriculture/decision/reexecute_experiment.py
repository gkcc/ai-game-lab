"""Official step-for-step re-execution for the retained 52 land probe games."""
import argparse
import gzip
import json
from pathlib import Path
from decision.full_audit import reexecute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    root = args.folder.resolve()
    target = root / "official-reexecution.json"
    if target.exists():
        parser.error("Choose a separate reproduction; do not overwrite retained audit")
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    tagged = [(r, root / "games" / f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}") for r in read("ledger.json")]
    tagged += [(r, root / "parity" / r["policy"]) for r in read("parity.json")["games"]]
    tagged += [(r, root / "reproducibility" / str(r["side"])) for r in read("reproducibility.json")["games"]]
    rows = []
    for row, folder in tagged:
        replay = json.loads(gzip.decompress((folder / "replay.json.gz").read_bytes()))
        result = reexecute(replay)
        assert result["final_rewards"] == row["banks"] and result["semantic_sha256"] == row["semantic_replay_sha256"]
        rows.append({k: row[k] for k in ("policy", "opponent", "seed", "side")} | result)
        print(f"Official re-execution {len(rows)}/52", flush=True)
    assert len(rows) == 52
    target.write_text(json.dumps({"passed": True, "games": 52, "official_transitions": 52 * 719,
                                  "both_public_and_private_checked_each_step": True, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
