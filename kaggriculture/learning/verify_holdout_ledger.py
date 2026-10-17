"""Verify the anonymous ledger, optionally checking private local replay bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "remainingOverageTime"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def verify(folder, replays=None):
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    config, rows, summary = (read(n) for n in ("registration.json", "ledger.json", "summary.json"))
    seals = [folder / "registration.sha256", folder / "registration-sha256.txt"]
    seal = next(p for p in seals if p.exists()).read_text().strip()
    assert sha((folder / "registration.json").read_bytes()) == seal
    expected = {(p, op, seed, side) for p in config["policies"] for op in config["opponents"] for seed in config["seeds"] for side in config["seats"]}
    lookup = {(r["policy"], r["opponent"], r["seed"], r["side"]): r for r in rows}
    assert len(rows) == len(lookup) == len(expected) == config["physical_games"] and set(lookup) == expected
    for r in rows:
        assert r["passed"] and r["both_player_farming_and_seed_audits_passed"] and r["agent_errors"] == 0
        assert r["states"] == 720 and r["action_turns"] == 719 and r["final_status"] == ["DONE", "DONE"]
        assert r["banks"] == r["rewards"]
        assert r["bank"] == r["banks"][r["side"]] and r["opponent_bank"] == r["banks"][1-r["side"]]
        assert r["net_profit"] == r["bank"] - 3000 and r["margin"] == r["bank"] - r["opponent_bank"]
        assert r["engine_sha256"] == config["engine_sha256"]
        assert r["bot_sha256"] == config["source_sha256"][r["policy"] + ".py"]
        assert r["rival_sha256"] == config["source_sha256"][r["opponent"] + ".py"]
    pairs = read("paired-results.json")
    assert len(pairs) == len(rows) // 2
    assert {(r["opponent"], r["seed"], r["side"]) for r in pairs} == {(op, s, side) for op in config["opponents"] for s in config["seeds"] for side in config["seats"]}
    for r in pairs:
        a, b = (lookup[p, r["opponent"], r["seed"], r["side"]] for p in ("candidate", "baseline"))
        score = lambda x: 1. if x["margin"] > 0 else .5 if x["margin"] == 0 else 0.
        assert r["profit_delta"] == a["net_profit"] - b["net_profit"]
        assert r["score_delta"] == score(a) - score(b)
    assert abs(mean(r["profit_delta"] for r in pairs) - summary["overall"]["mean_profit_delta"]) < 1e-8
    for op in config["opponents"]:
        for policy in config["policies"]:
            items = [r for r in rows if r["opponent"] == op and r["policy"] == policy]
            aggregate = summary["by_opponent"][op][policy]
            assert aggregate["wins"] == sum(r["margin"] > 0 for r in items)
            assert aggregate["draws"] == sum(r["margin"] == 0 for r in items)
            assert aggregate["losses"] == sum(r["margin"] < 0 for r in items)
            assert abs(aggregate["mean_net_profit"] - mean(r["net_profit"] for r in items)) < 1e-8
    verified = 0
    if replays:
        repeat = read("reproducibility.json")
        for r, is_repeat in [(r, False) for r in rows] + [(r, True) for r in repeat["rows"]]:
            if not is_repeat:
                name = f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}"
                game = replays / "games" / name
            else:
                game = replays / "reproducibility" / str(r["side"])
            raw = (game / "replay.json.gz").read_bytes()
            assert sha(raw) == r["replay_gzip_sha256"]
            replay = json.loads(gzip.decompress(raw))
            assert len(replay["steps"]) == r["states"]
            assert not any(s["status"] in ("ERROR", "INVALID", "TIMEOUT") for frame in replay["steps"] for s in frame)
            # Official resolve_episode_seed moves the seed to info and clears
            # configuration.seed before exporting the replay.
            assert replay["info"]["seed"] == r["seed"]
            last = replay["steps"][-1]
            assert [s["status"] for s in last] == r["final_status"]
            assert [s["reward"] for s in last] == r["banks"]
            assert [f["money"] for f in last[0]["observation"]["farms"]] == r["banks"]
            semantic = json.dumps(clean(replay["steps"]), sort_keys=True, separators=(",", ":")).encode()
            assert sha(semantic) == r["semantic_replay_sha256"]
            logs = json.loads((game / "agent-logs.json").read_text(encoding="utf-8"))
            assert not any(e.get("stderr") for frame in logs for e in frame)
            metrics = json.loads((game / "metrics-both.json").read_text(encoding="utf-8"))
            assert all(m["farming_preconditions_passed"] and m["seed_balance_matches"] for m in metrics)
            verified += 1
    return {"passed": True, "ledger_games": len(rows), "local_replays_verified": verified,
            "boundary": "Ledger consistency and optional local replay integrity; no proof of historical non-retuning or replay re-execution"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replays", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, args.replays), indent=2))
