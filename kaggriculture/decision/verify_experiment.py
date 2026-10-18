"""Recheck anonymous land statistics and optional official replay integrity."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from decision.semantics import digest as semantic_digest


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if k != "remainingOverageTime"}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(folder, replays=None):
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    registration = "registration-revised.json" if (folder / "registration-revised.json").exists() else "registration.json"
    cfg, ledger, summary, acceptance = read(registration), read("ledger.json"), read("summary.json"), read("acceptance.json")
    assert digest((folder / registration).read_bytes()) == acceptance["registration_sha256"]
    key = lambda r: (r["policy"], r["opponent"], r["seed"], r["side"])
    lookup = {key(r): r for r in ledger}
    expected = {(p, op, s, side) for p in cfg["policies"] for op in cfg["opponents"] for s in cfg["seeds"] for side in cfg["seats"]}
    assert len(ledger) == len(lookup) == 48 and set(lookup) == expected
    parity, repeats = read("parity.json")["games"], read("reproducibility.json")["games"]
    assert len(parity) == len(repeats) == 2
    assert parity[0]["banks"] == parity[1]["banks"] and parity[0]["semantic_replay_sha256"] == parity[1]["semantic_replay_sha256"]
    for r in repeats:
        orig = lookup["land", "market", cfg["seeds"][0], r["side"]]
        assert orig["banks"] == r["banks"] and orig["semantic_replay_sha256"] == r["semantic_replay_sha256"]
    for r in ledger + parity + repeats:
        assert r["passed"] and r["states"] == 720 and r["action_turns"] == 719 and r["final_status"] == ["DONE", "DONE"]
        assert r["bank"] == r["banks"][r["side"]] and r["opponent_bank"] == r["banks"][1-r["side"]]
        assert r["rewards"] == r["banks"] and r["margin"] == r["bank"] - r["opponent_bank"] and r["net_profit"] == r["bank"] - 3000
        assert r["engine_sha256"] == cfg["engine_sha256"]
        assert r["bot_sha256"] == cfg["source_sha256"][r["policy"] + ".py"]
        assert r["rival_sha256"] == cfg["source_sha256"][r["opponent"] + ".py"]
    score = lambda r: float(r["margin"] > 0) + .5 * (r["margin"] == 0)
    stats = {}
    for panel, ops in [("overall", cfg["opponents"]), *[(op, [op]) for op in cfg["opponents"]]]:
        clusters = []
        for s in cfg["seeds"]:
            diffs = []
            for op in ops:
                for side in cfg["seats"]:
                    a, b = lookup["land", op, s, side], lookup["control", op, s, side]
                    diffs.append([a["bank"] - b["bank"], score(a) - score(b)])
            clusters.append(np.array(diffs).mean(axis=0))
        a = np.array(clusters)
        ids = np.random.default_rng(cfg["bootstrap"]["rng_seed"]).integers(0, len(a), (20000, len(a)))
        ci = np.quantile(a[ids].mean(axis=1), [.025, .975], axis=0)
        ref = summary["overall"] if panel == "overall" else summary["by_opponent"][panel]["paired"]
        assert np.allclose(ci[:, 0], ref["profit_95_ci"], rtol=0, atol=1e-9)
        assert abs(a[:, 0].mean() - ref["mean_profit_delta"]) < 1e-9
        assert abs(a[:, 1].mean() - ref["mean_score_delta"]) < 1e-9
        stats[panel] = {"profit_delta": float(a[:, 0].mean()), "profit_95_ci": ci[:, 0].tolist(), "score_delta": float(a[:, 1].mean())}
    gate = stats["overall"]["profit_95_ci"][0] > 0 and all(stats[op]["score_delta"] >= 0 for op in cfg["opponents"])
    assert gate == summary["exploratory_land_support_gate"]
    checked, shop_divergence, expanded_plants = 0, 0, 0
    trajectories = {}
    ordered_hashes = []
    if replays:
        for name, h in cfg["source_sha256"].items():
            assert digest((replays / "sources" / name).read_bytes()) == h
        tagged = [(r, "panel") for r in ledger] + [(r, "parity") for r in parity] + [(r, "repeat") for r in repeats]
        for r, stage in tagged:
            if stage == "panel":
                path = replays / "games" / f"{r['seed']}-{r['opponent']}-{r['side']}-{r['policy']}"
            elif stage == "parity":
                path = replays / "parity" / r["policy"]
            else:
                path = replays / "reproducibility" / str(r["side"])
            raw = (path / "replay.json.gz").read_bytes()
            assert digest(raw) == r["replay_gzip_sha256"]
            d = json.loads(gzip.decompress(raw))
            assert len(d["steps"]) == 720 and d["info"]["seed"] == r["seed"]
            assert digest(json.dumps(clean(d["steps"]), sort_keys=True, separators=(",", ":")).encode()) == r["semantic_replay_sha256"]
            v2 = semantic_digest(d["steps"])
            if "semantic_replay_sha256_v2" in r:
                assert v2 == r["semantic_replay_sha256_v2"]
            ordered_hashes.append({k: r[k] for k in ("policy", "opponent", "seed", "side")} | {"stage": stage, "semantic_sha256_v2": v2})
            assert [s["reward"] for s in d["steps"][-1]] == r["banks"]
            assert [s["status"] for s in d["steps"][-1]] == ["DONE", "DONE"]
            assert not any(s["status"] in ("ERROR", "INVALID", "TIMEOUT") for frame in d["steps"] for s in frame)
            logs = json.loads((path / "agent-logs.json").read_text(encoding="utf-8"))
            assert not any(e.get("stderr") for frame in logs for e in frame)
            metrics = json.loads((path / "metrics-both.json").read_text(encoding="utf-8"))
            assert all(m["seed_balance_matches"] and m["farming_preconditions_passed"] for m in metrics)
            if stage == "panel":
                p = r["side"]
                first = d["steps"][1]
                f = first[0]["observation"]["farms"][p]
                assert f["unlocked_quadrants"] == (["NW", "NE"] if r["policy"] == "land" else ["NW"])
                plants = 0
                for prev, following in zip(d["steps"], d["steps"][1:]):
                    old = prev[0]["observation"]["farms"][p]
                    act = following[p]["action"]
                    seen = set()
                    for (x, y), atom in zip([old["farmer"], *old["hands"]], [act["farmer"], *act["hands"]]):
                        t = following[0]["observation"]["farms"][p]["tiles"][y][x]
                        if atom[0] == "PLANT" and x >= len(old["tiles"]) // 2 and old["tiles"][y][x] is None and isinstance(t, dict) and t.get("crop") == atom[1]:
                            seen.add((x, y))
                    plants += len(seen)
                assert plants == r["mechanism"]["successful_plant_events_in_new_land"]
                expanded_plants += plants
                trajectories[key(r)] = {"first_money": f["money"], "first_action": first[p]["action"],
                                         "town": [frame[0]["observation"]["town"]["unlocked_shops"] for frame in d["steps"]]}
            checked += 1
            if checked % 12 == 0:
                print(f"Verified {checked}/52 replay files", file=sys.stderr, flush=True)
        for s in cfg["seeds"]:
            for op in cfg["opponents"]:
                for side in cfg["seats"]:
                    a, b = trajectories["land", op, s, side], trajectories["control", op, s, side]
                    assert a["first_money"] == b["first_money"] - 1000
                    assert a["first_action"]["farmer"] == b["first_action"]["farmer"] and a["first_action"]["hands"] == b["first_action"]["hands"]
                    assert a["first_action"]["market"] == [["BUY_LAND"], *b["first_action"]["market"]]
                    shop_divergence += a["town"] != b["town"]
    return {"passed": True, "ledger_games": 48, "local_replays_verified": checked, "statistics_recomputed": stats,
            "exploratory_gate": bool(gate), "paired_town_trajectories_differ": shop_divergence if checked else None,
            "expanded_plant_events_verified": expanded_plants if checked else None,
            "audit_version": 2, "ordered_replay_hashes": ordered_hashes,
            "plant_count_boundary": "Unique per-turn tile transitions only; generic execution labels require official per-primitive tracing",
            "boundary": "Shared-RNG and opponent-response effects belong to the treatment; not a pure extra-land-yield estimate or external replay re-execution"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replays", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, args.replays), indent=2))
