"""Pre-register and evaluate one frozen candidate; no training or selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import shutil
from statistics import mean
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = {"v1": "bots/heuristic_v1.py", "v2": "bots/heuristic_v2.py",
         "jit": "bots/heuristic_v1_jit_seeds.py", "market": "bots/heuristic_v1_market.py"}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out, candidate):
    out.mkdir(parents=True, exist_ok=False)
    sources = out / "sources"
    sources.mkdir()
    frozen = {"candidate.py": candidate, "baseline.py": ROOT / PANEL["v1"],
              **{name + ".py": ROOT / path for name, path in PANEL.items()},
              "default.py": ROOT / "main.py", "worker-original.py": ROOT / "learning/official_worker.py",
              "evaluate_holdout.py": Path(__file__)}
    for name, path in frozen.items():
        shutil.copyfile(path, sources / name)
    # A fresh namespace for each player and each game; cache compiled code only.
    worker = (sources / "worker-original.py").read_text(encoding="utf-8")
    old = '''    if digest not in CACHE:
        namespace = {"__name__": "frozen_bot", "__file__": path}
        exec(compile(source, path, "exec"), namespace)
        CACHE[digest] = namespace["agent"]
    return CACHE[digest], digest'''
    new = '''    if digest not in CACHE:
        CACHE[digest] = compile(source, path, "exec")
    namespace = {"__name__": "frozen_bot", "__file__": path}
    exec(CACHE[digest], namespace)
    return namespace["agent"], digest'''
    assert worker.count(old) == 1
    (sources / "worker.py").write_text(worker.replace(old, new), encoding="utf-8")
    seeds = random.Random(2026101802).sample(range(2000000, 3000000), 12)
    config = {
        "experiment": "frozen-candidate-expanded-holdout-v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "no_training_no_retuning": True, "candidate_selection": "Prior experiment champion; no new selection",
        "seed_generator": "Python random.Random(2026101802).sample(range(2000000,3000000),12)",
        "seeds": seeds, "opponents": list(PANEL), "seats": [0, 1],
        "policies": ["baseline", "candidate"], "physical_games": 192,
        "independence": "New numeric seed range disjoint from prior train/test/development panels; no results observed before registration",
        "engine_commit": "f994f2050d58d84f649329c1e35ce09ada7007c5",
        "engine_sha256": "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
        "source_sha256": {p.name: sha(p) for p in sorted(sources.iterdir())},
        "official_gate": "All 192 games: 720 states, 719 actions, DONE/DONE, rewards equal banks, no errors/stderr, both player farming and seed-balance audits pass",
        "primary": "Equal-weight mean paired net-profit difference over seeds, opponents and both seats",
        "bootstrap": {"unit": "entire seed, retaining all opponents and seats", "resamples": 20000, "rng_seed": 2026101803, "ci": 0.95},
        "improvement_gate": "Primary 95% lower bound > 0 AND market-only paired profit 95% lower bound > 0 AND mean match-score difference >= 0 overall AND >= 0 against market; all official and reproducibility gates pass",
        "secondary": "W/D/L, both banks, market-only profit/score differences, exact two-sided sign test; score CI descriptive only",
        "multiplicity": "Overall and market-only tests form a conjunctive gate, not alternative chances to pass; other opponent statistics descriptive",
        "reproducibility": "Repeat seed[0] candidate vs market in both seats sequentially reversed after parallel panel, compare semantic replay hashes and banks",
        "failure_policy": "Report failure or inconclusive without changing candidate, panel, seeds or gate; infrastructure retries separately recorded and never replace completed valid games",
        "scope": "Fixed local opponents only; no claim about unseen opponents or leaderboard; no default change, PPO or GPU training",
        "trusted_code_only": "Local reviewed bots, not a sandbox for untrusted submissions"
    }
    save(out / "registration.json", config)
    (out / "registration.sha256").write_text(sha(out / "registration.json") + "\n")
    print(json.dumps({"prepared": True, "seeds": seeds, "games": 192}), flush=True)


def score(row):
    return 1. if row["margin"] > 0 else .5 if row["margin"] == 0 else 0.


def analyze(rows, config):
    lookup = {(r["policy"], r["opponent"], r["seed"], r["side"]): r for r in rows}
    paired = []
    for seed in config["seeds"]:
        for opponent in config["opponents"]:
            for side in (0, 1):
                a, b = (lookup[p, opponent, seed, side] for p in ("candidate", "baseline"))
                paired.append({"seed": seed, "opponent": opponent, "side": side,
                               "profit_delta": a["net_profit"] - b["net_profit"], "score_delta": score(a) - score(b)})
    def stats(subset):
        clusters = np.array([[mean(r[k] for r in subset if r["seed"] == s) for k in ("profit_delta", "score_delta")]
                             for s in config["seeds"]])
        rng = np.random.default_rng(config["bootstrap"]["rng_seed"])
        dist = clusters[rng.integers(0, len(clusters), size=(20000, len(clusters)))].mean(axis=1)
        ci = np.quantile(dist, [.025, .975], axis=0)
        pos, neg = sum(clusters[:, 0] > 0), sum(clusters[:, 0] < 0)
        n = int(pos + neg)
        p = min(1., 2 * sum(math.comb(n, i) for i in range(int(min(pos, neg)) + 1)) / 2**n) if n else 1.
        return {"mean_profit_delta": float(clusters[:, 0].mean()), "profit_95_ci": ci[:, 0].tolist(),
                "mean_score_delta": float(clusters[:, 1].mean()), "score_95_ci_descriptive": ci[:, 1].tolist(),
                "score_ci_degenerate": bool(ci[0, 1] == ci[1, 1]), "positive_profit_seeds": int(pos),
                "negative_profit_seeds": int(neg), "sign_test_p": p, "seed_profit_deltas": clusters[:, 0].tolist()}
    def aggregate(items):
        return {"games": len(items), "wins": sum(r["margin"] > 0 for r in items), "draws": sum(r["margin"] == 0 for r in items),
                "losses": sum(r["margin"] < 0 for r in items), "mean_net_profit": mean(r["net_profit"] for r in items),
                "mean_opponent_profit": mean(r["opponent_bank"] - 3000 for r in items), "mean_score": mean(score(r) for r in items)}
    overall = stats(paired)
    by_opponent = {op: {"paired": stats([r for r in paired if r["opponent"] == op]),
                        **{p: aggregate([r for r in rows if r["policy"] == p and r["opponent"] == op])
                           for p in config["policies"]}} for op in config["opponents"]}
    market = by_opponent["market"]["paired"]
    passed = (overall["profit_95_ci"][0] > 0 and market["profit_95_ci"][0] > 0
              and overall["mean_score_delta"] >= 0 and market["mean_score_delta"] >= 0)
    return {"physical_games": len(rows), "overall": overall, "by_opponent": by_opponent,
            "aggregate": {p: aggregate([r for r in rows if r["policy"] == p]) for p in config["policies"]},
            "statistical_gate_passed": passed, "score_intervals_descriptive_only": True}, paired


def run(out):
    registration_sha = (out / "registration.sha256").read_text().strip()
    assert sha(out / "registration.json") == registration_sha
    config = json.loads((out / "registration.json").read_text(encoding="utf-8"))
    sources = out / "sources"
    for name, digest in config["source_sha256"].items():
        assert sha(sources / name) == digest
    assert sha(Path(__file__)) == config["source_sha256"]["evaluate_holdout.py"]
    assert not (out / "games").exists(), "Use a fresh registered run; no silent resume or result replacement"
    original = {name: sha(ROOT / path) for name, path in PANEL.items()}
    original["default"] = sha(ROOT / "main.py")
    available, workers, logs = queue.Queue(), [], []
    rows, repeats = [], []
    def request(policy, opponent, seed, side, folder, html=False):
        worker = available.get()
        try:
            req = {"cmd": "play", "bot": str(sources / (policy + ".py")), "rival": str(sources / (opponent + ".py")),
                   "seed": seed, "side": side, "folder": str(folder), "semantic": True, "html": html}
            worker.stdin.write(json.dumps(req) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError("Worker exited; inspect private worker log")
            result = json.loads(line)
            if "error" in result or not result.get("passed"):
                raise RuntimeError("Official game failed; inspect private result/logs")
            assert result["engine_sha256"] == config["engine_sha256"]
            assert result["bot_sha256"] == config["source_sha256"][policy + ".py"]
            assert result["rival_sha256"] == config["source_sha256"][opponent + ".py"]
            # Public ledger is constructed from an explicit field allowlist.
            fields = ["seed", "side", "banks", "bank", "opponent_bank", "net_profit", "margin", "states", "action_turns",
                      "bot_sha256", "rival_sha256", "engine_sha256", "package", "passed", "semantic_replay_sha256"]
            safe = {k: result[k] for k in fields}
            safe.update({"policy": policy, "opponent": opponent, "replay_gzip_sha256": sha(folder / "replay.json.gz"),
                         "final_status": ["DONE", "DONE"], "rewards": result["banks"],
                         "both_player_farming_and_seed_audits_passed": True, "agent_errors": 0})
            return safe
        finally:
            available.put(worker)
    try:
        for i in range(2):
            log = (out / f"worker-{i}.log").open("w", encoding="utf-8")
            logs.append(log)
            worker = subprocess.Popen([str(ROOT / ".venv/Scripts/python.exe"), "-B", "-X", "utf8", str(sources / "worker.py")],
                                      cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                      text=True, encoding="utf-8", bufsize=1,
                                      env={**os.environ, "KAGGRICULTURE_ROOT": str(ROOT)},
                                      creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            workers.append(worker)
            available.put(worker)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            for seed in config["seeds"]:
                for op in config["opponents"]:
                    for side in (0, 1):
                        for policy in config["policies"]:
                            folder = out / "games" / f"{seed}-{op}-{side}-{policy}"
                            jobs.append(pool.submit(request, policy, op, seed, side, folder,
                                                    seed == config["seeds"][0] and op == "market"))
            for job in as_completed(jobs):
                rows.append(job.result())
                save(out / "partial-ledger.json", rows)
                print(f"Completed {len(rows)}/{config['physical_games']}", flush=True)
        for side in (1, 0):
            row = request("candidate", "market", config["seeds"][0], side, out / "reproducibility" / str(side))
            old = next(r for r in rows if r["policy"] == "candidate" and r["opponent"] == "market" and r["seed"] == row["seed"] and r["side"] == side)
            assert row["banks"] == old["banks"] and row["semantic_replay_sha256"] == old["semantic_replay_sha256"]
            repeats.append(row)
        assert len(rows) == 192 and len({(r['policy'], r['opponent'], r['seed'], r['side']) for r in rows}) == 192
        assert original == {**{name: sha(ROOT / path) for name, path in PANEL.items()}, "default": sha(ROOT / "main.py")}
        for name, digest in config["source_sha256"].items():
            assert sha(sources / name) == digest
        rows.sort(key=lambda r: (r["seed"], r["opponent"], r["side"], r["policy"]))
        summary, paired = analyze(rows, config)
        save(out / "ledger.json", rows)
        save(out / "paired-results.json", paired)
        save(out / "summary.json", summary)
        save(out / "reproducibility.json", {"passed": True, "extra_games": 2, "rows": repeats})
    finally:
        for worker in workers:
            if worker.poll() is None:
                try:
                    worker.stdin.write('{"cmd":"close"}\n')
                    worker.stdin.flush()
                    worker.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    worker.terminate()
                    worker.wait(timeout=15)
            if worker.stdin:
                worker.stdin.close()
            if worker.stdout:
                worker.stdout.close()
        for log in logs:
            log.close()
    assert sha(out / "registration.json") == registration_sha
    save(out / "acceptance.json", {"official_games_passed": len(rows) == 192, "extra_reproducibility_games_passed": len(repeats) == 2,
                                   "registration_sha256": registration_sha, "sources_and_default_unchanged": True,
                                   "workers_closed": all(w.poll() is not None for w in workers),
                                   "statistical_gate_passed": summary["statistical_gate_passed"],
                                   "gpu_training": False, "default_changed": False})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.candidate is None:
            parser.error("prepare requires --candidate")
        prepare(args.output.resolve(), args.candidate.resolve())
    else:
        run(args.output.resolve())
