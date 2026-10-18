"""Official reexecution and realized fertilizer effects, using retained game actions."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from decision.full_audit import reexecute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("stage", choices=("pilot", "test"))
    args = parser.parse_args()
    root = args.folder.resolve()
    output = root / (args.stage + "-official-audit.json")
    assert not output.exists()
    ledger = json.loads((root / args.stage / "ledger.json").read_text())
    tagged = [(r, root / args.stage / f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}") for r in ledger]
    if args.stage == "test":
        tagged += [(r, root / "repeat" / str(r["side"])) for r in json.loads((root / "reproducibility.json").read_text())["rows"]]
    rows = []
    sources = {n: hashlib.sha256((root / "sources" / n).read_bytes()).hexdigest() for n in ("baseline.py", "candidate.py", "v1.py", "market.py", "round2.py", "worker.py")}
    for row, folder in tagged:
        raw = (folder / "replay.json.gz").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["replay_gzip_sha256"]
        replay = json.loads(gzip.decompress(raw))
        effects = Counter()
        def sink(record):
            if record["player"] != row["side"]:
                return
            if record["phase"] == "unit":
                op = record["effective"][0]
                if op in ("FERTILIZE", "PICKUP"):
                    effects[op + ":" + record["status"]] += 1
            elif record["phase"] == "market":
                request = record["requested"]
                if isinstance(request, list) and request[:2] == ["BUY_PRODUCT", "FERTILIZER"]:
                    effects["fertilizer_bought_units"] += record.get("executed_units", 0)
                    effects["fertilizer_purchase_cost"] -= record.get("cash_delta", 0)
        audit = reexecute(replay, trace=True, label_sink=sink)
        assert audit["final_rewards"] == row["banks"] and audit["semantic_sha256_v2"] == row["semantic_replay_sha256_v2"]
        p = row["side"]
        last = replay["steps"][-1][p]["observation"]["private"]
        effects["unsold_fertilizer_units"] = last["shed"].get("FERTILIZER", 0) + sum(x.get("FERTILIZER", 0) for x in last["inventories"])
        rows.append({k: row[k] for k in ("seed", "side", "policy", "opponent", "replay_gzip_sha256")} | {"audit": audit, "fertilizer_effects": dict(effects)})
        print(f"Official audit {args.stage} {len(rows)}/{len(tagged)}", flush=True)
    assert all(hashlib.sha256((root / "sources" / n).read_bytes()).hexdigest() == h for n,h in sources.items())
    output.write_text(json.dumps({"passed": True, "games": len(rows), "official_transitions": len(rows)*719,
                                  "frozen_strategy_sha256": sources, "rows": rows}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
