"""Recompute round-two selection/statistics and optionally verify local replays."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "remainingOverageTime"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def verify(folder, replays=None):
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    cfg, selection, summary = (read(n) for n in ("registration.json", "selection.json", "summary.json"))
    seal = next(p for p in [folder / "registration.sha256", folder / "registration-sha256.txt"] if p.exists()).read_text().strip()
    assert sha((folder / "registration.json").read_bytes()) == seal == selection["registration_sha256"]
    development, test = read("development-ledger.json"), read("test-ledger.json")
    trials = read("trials.json")
    assert len(trials) == cfg["trials"] and {t["trial"] for t in trials} == set(range(cfg["trials"]))
    trial_hashes = {t["trial"]: t["bot_sha256"] for t in trials}
    for rows, expected, keys in [
        (development, {(t, s, op, side) for t in range(cfg["trials"]) for s in cfg["development_seeds"] for op in cfg["opponents"] for side in (0, 1)}, ["trial", "seed", "opponent", "side"]),
        (test, {(p, s, op, side) for p in cfg["test_policies"] for s in cfg["test_seeds"] for op in cfg["opponents"] for side in (0, 1)}, ["policy", "seed", "opponent", "side"])
    ]:
        assert len(rows) == len(expected) and {tuple(r[k] for k in keys) for r in rows} == expected
        for r in rows:
            assert r["states"] == 720 and r["action_turns"] == 719 and r["final_status"] == ["DONE", "DONE"]
            assert r["passed"] and r["both_player_farming_and_seed_audits_passed"] and r["agent_errors"] == 0
            assert r["rewards"] == r["banks"]
            assert r["bank"] == r["banks"][r["side"]] and r["opponent_bank"] == r["banks"][1-r["side"]]
            assert r["net_profit"] == r["bank"] - 3000 and r["margin"] == r["bank"] - r["opponent_bank"]
            assert r["engine_sha256"] == cfg["engine_sha256"]
            assert r["rival_sha256"] == cfg["source_sha256"][r["opponent"] + ".py"]
            bot_hash = trial_hashes[r["trial"]] if "trial" in r else selection["champion_sha256"] if r["policy"] == "candidate" else cfg["source_sha256"][r["policy"] + ".py"]
            assert r["bot_sha256"] == bot_hash
    for t in trials:
        rows = [r for r in development if r["trial"] == t["trial"]]
        groups = [mean(r["margin"] / (r["bank"] + r["opponent_bank"] + 6000) for r in rows if r["seed"] == s and r["opponent"] == op)
                  for s in cfg["development_seeds"] for op in cfg["opponents"]]
        assert np.allclose(groups, t["group_values"], rtol=0, atol=1e-12)
        assert abs(mean(groups) - cfg["objective_penalty"] * pstdev(groups) - t["objective"]) < 1e-12
    best = max(trials, key=lambda t: (t["objective"], -t["trial"]))
    assert best["trial"] == selection["trial"] and best["weights"] == selection["params"] and best["bot_sha256"] == selection["champion_sha256"]
    assert selection["selection_used_development_only"] and selection["test_not_started"]
    lookup = {(r["policy"], r["opponent"], r["seed"], r["side"]): r for r in test}
    score = lambda r: float(r["margin"] > 0) + .5 * (r["margin"] == 0)
    paired = read("paired-results.json")
    for comparator in ("baseline", "incumbent"):
        records = paired[comparator]
        expected = {(op, s, side) for op in cfg["opponents"] for s in cfg["test_seeds"] for side in (0, 1)}
        assert len(records) == len(expected) and {(r["opponent"], r["seed"], r["side"]) for r in records} == expected
        for r in records:
            a, b = (lookup[p, r["opponent"], r["seed"], r["side"]] for p in ("candidate", comparator))
            assert r["profit_delta"] == a["net_profit"] - b["net_profit"]
            assert r["score_delta"] == score(a) - score(b)
    statistics, gate = {}, True
    for comparator in ("baseline", "incumbent"):
        statistics[comparator] = {}
        for panel, opponents in [("overall", cfg["opponents"]), ("market", ["market"])]:
            clusters = []
            for seed in cfg["test_seeds"]:
                diffs = []
                for op in opponents:
                    for side in (0, 1):
                        a, b = (lookup[p, op, seed, side] for p in ("candidate", comparator))
                        diffs.append([a["bank"] - b["bank"], score(a) - score(b)])
                clusters.append(np.array(diffs).mean(axis=0))
            a = np.array(clusters)
            rng = np.random.default_rng(cfg["bootstrap"]["rng_seed"])
            ci = np.quantile(a[rng.integers(0, len(a), size=(cfg["bootstrap"]["resamples"], len(a)))].mean(axis=1), [.025, .975], axis=0)
            reported = summary["comparisons"][comparator]["overall"] if panel == "overall" else summary["comparisons"][comparator]["by_opponent"]["market"]["paired"]
            assert np.allclose(ci[:, 0], reported["profit_95_ci"], rtol=0, atol=1e-10)
            assert np.allclose(ci[:, 1], reported["score_95_ci_descriptive"], rtol=0, atol=1e-10)
            assert abs(float(a[:, 0].mean()) - reported["mean_profit_delta"]) < 1e-10
            assert abs(float(a[:, 1].mean()) - reported["mean_score_delta"]) < 1e-10
            gate = gate and bool(ci[0, 0] > 0 and a[:, 1].mean() >= 0)
            statistics[comparator][panel] = {"mean_profit_delta": float(a[:, 0].mean()), "profit_95_ci": ci[:, 0].tolist(),
                                           "mean_score_delta": float(a[:, 1].mean())}
        for op in cfg["opponents"]:
            for policy, label in [(comparator, "baseline"), ("candidate", "candidate")]:
                rows = [r for r in test if r["policy"] == policy and r["opponent"] == op]
                agg = summary["comparisons"][comparator]["by_opponent"][op][label]
                for key, test_margin in [("wins", lambda m: m > 0), ("draws", lambda m: m == 0), ("losses", lambda m: m < 0)]:
                    assert agg[key] == sum(test_margin(r["margin"]) for r in rows)
                assert abs(agg["mean_net_profit"] - mean(r["net_profit"] for r in rows)) < 1e-8
    assert gate == summary["overall_gate_passed"]
    repeats = read("reproducibility.json")
    assert len(repeats["rows"]) == 2 and {r["side"] for r in repeats["rows"]} == {0, 1}
    for r in repeats["rows"]:
        orig = lookup["candidate", "market", cfg["test_seeds"][0], r["side"]]
        assert r["semantic_replay_sha256"] == orig["semantic_replay_sha256"] and r["banks"] == orig["banks"]
    verified = 0
    if replays:
        for name, digest in cfg["source_sha256"].items():
            assert sha((replays / "sources" / name).read_bytes()) == digest
        for t in trials:
            assert sha((replays / "training" / f"trial{t['trial']}" / "agent.py").read_bytes()) == t["bot_sha256"]
        assert sha((replays / "champion.py").read_bytes()) == selection["champion_sha256"]
        tagged = [(r, "development") for r in development] + [(r, "test") for r in test] + [(r, "repeat") for r in repeats["rows"]]
        for r, stage in tagged:
            if stage == "development":
                path = replays / "training" / f"trial{r['trial']}" / f"{r['seed']}-{r['opponent']}-{r['side']}"
            elif stage == "test":
                path = replays / "test" / f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}"
            else:
                path = replays / "reproducibility" / str(r["side"])
            raw = (path / "replay.json.gz").read_bytes()
            assert sha(raw) == r["replay_gzip_sha256"]
            replay = json.loads(gzip.decompress(raw))
            assert len(replay["steps"]) == 720 and replay["info"]["seed"] == r["seed"]
            assert not any(s["status"] in ("ERROR", "INVALID", "TIMEOUT") for frame in replay["steps"] for s in frame)
            last = replay["steps"][-1]
            assert [s["status"] for s in last] == ["DONE", "DONE"]
            assert [s["reward"] for s in last] == r["banks"]
            assert [f["money"] for f in last[0]["observation"]["farms"]] == r["banks"]
            assert sha(json.dumps(clean(replay["steps"]), sort_keys=True, separators=(",", ":")).encode()) == r["semantic_replay_sha256"]
            logs = json.loads((path / "agent-logs.json").read_text(encoding="utf-8"))
            assert not any(e.get("stderr") for frame in logs for e in frame)
            metrics = json.loads((path / "metrics-both.json").read_text(encoding="utf-8"))
            assert len(metrics) == 2 and all(m["farming_preconditions_passed"] and m["seed_balance_matches"] for m in metrics)
            verified += 1
            if verified % 24 == 0:
                print(f"Replays verified {verified}/{len(tagged)}", file=sys.stderr, flush=True)
    return {"passed": True, "development_games": len(development), "test_games": len(test), "reproducibility_games": 2,
            "local_replays_verified": verified, "selection_recomputed": True, "statistical_gate_passed": gate,
            "independently_recomputed_statistics": statistics,
            "boundary": "Anonymous ledger consistency and optional local replay integrity, not external re-execution or proof of all historical non-retuning"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replays", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, args.replays), indent=2))
