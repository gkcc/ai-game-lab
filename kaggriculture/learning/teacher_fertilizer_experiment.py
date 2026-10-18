"""Bounded registered fertilizer teacher evaluation using existing official workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean

from optimize_round2 import Games
from evaluate_holdout import analyze, save, sha


def run(out, stage):
    cfg = json.loads((out / "registration.json").read_text())
    assert sha(out / "registration.json") == (out / "registration.sha256").read_text().strip()
    for n, h in cfg["source_sha256"].items():
        assert sha(out / "sources" / n) == h
    if stage == "test":
        freeze = json.loads((out / "candidate-freeze.json").read_text())
        assert sha(out / "sources/candidate.py") == freeze["candidate_sha256"]
        assert sha(Path(__file__)) == freeze["experiment_runner_sha256"]
    stage_root = out / stage
    stage_root.mkdir(exist_ok=False)
    seeds = cfg["development_seeds"] if stage == "pilot" else cfg["test_seeds"]
    opponents = ["v1", "market"] if stage == "pilot" else cfg["opponents"]
    rows = []
    with Games(out, cfg, stage) as games, ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        jobs = [pool.submit(games.play, out / "sources" / (policy + ".py"), policy, op, seed, side,
                            stage_root / f"{seed}-{op}-{side}-{policy}")
                for seed in seeds for op in opponents for side in (0, 1) for policy in cfg["policies"]]
        for f in as_completed(jobs):
            rows.append(f.result())
            print(f"{stage}: {len(rows)}/{len(jobs)}", flush=True)
        rows.sort(key=lambda r: (r["seed"], r["opponent"], r["side"], r["policy"]))
        save(stage_root / "ledger.json", rows)
        summary, paired = analyze(rows, {"seeds": seeds, "opponents": opponents, "policies": cfg["policies"], "bootstrap": cfg["bootstrap"]})
        summary["helper_gate_not_used"] = summary.pop("statistical_gate_passed")
        margin_delta = mean(r["margin"] for r in rows if r["policy"] == "candidate") - mean(r["margin"] for r in rows if r["policy"] == "baseline")
        summary["mean_margin_delta"] = margin_delta
        summary["competitive_gate"] = (summary["overall"]["profit_95_ci"][0] > 0 and margin_delta > 0
                                       and all(summary["by_opponent"][op]["paired"]["mean_score_delta"] >= 0 for op in opponents))
        summary["scope"] = "Development only" if stage == "pilot" else "Six unseen seed clusters; exploratory fixed-opponent panel, no test-driven changes"
        save(stage_root / "summary.json", summary)
        save(stage_root / "paired-results.json", paired)
        if stage == "test":
            repeats = []
            for side in (1, 0):
                r = games.play(out / "sources/candidate.py", "candidate", "market", seeds[0], side,
                               out / "repeat" / str(side))
                original = next(x for x in rows if x["policy"] == "candidate" and x["opponent"] == "market" and x["side"] == side and x["seed"] == seeds[0])
                assert r["banks"] == original["banks"] and r["semantic_replay_sha256_v2"] == original["semantic_replay_sha256_v2"]
                repeats.append(r)
            save(out / "reproducibility.json", {"passed": True, "rows": repeats})
    print(json.dumps({"stage": stage, "competitive_gate": summary["competitive_gate"], "overall": summary["overall"], "aggregate": summary["aggregate"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.stage)
