"""One registered land counterfactual through the shared decision interface."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
import shutil
from statistics import mean

from evaluate_holdout import analyze, save, sha
from optimize_round2 import Games

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = '''from pathlib import Path
from decision.api import FrozenAdapter, act
_scope = {"__name__": "frozen_baseline", "__file__": str(Path(__file__).with_name("baseline.py"))}
exec(compile(Path(_scope["__file__"]).read_bytes(), _scope["__file__"], "exec"), _scope)
_policy = FrozenAdapter(_scope["agent"], buy_first_land=EXPAND)
def agent(observation, configuration):
    return act(_policy, observation, configuration)
'''


def prepare(out, baseline, reuse=None):
    out.mkdir(parents=True, exist_ok=False)
    src = out / "sources"
    src.mkdir()
    paths = {"baseline.py": baseline, "v1.py": ROOT / "bots/heuristic_v1.py", "market.py": ROOT / "bots/heuristic_v1_market.py",
             "worker.py": ROOT / "learning/round2_worker.py", "mechanism_experiment.py": Path(__file__),
             "optimize_round2.py": ROOT / "learning/optimize_round2.py", "evaluate_holdout.py": ROOT / "learning/evaluate_holdout.py",
             "api.py": ROOT / "decision/api.py", "verify_mechanisms.py": ROOT / "decision/verify_mechanisms.py"}
    for name, path in paths.items():
        shutil.copyfile(path, src / name)
    for name, flag in (("control", False), ("land", True)):
        (src / (name + ".py")).write_text(WRAPPER.replace("EXPAND", repr(flag)), encoding="utf-8")
    config = {"experiment": "mechanism-foundation-one-land-purchase", "registered_utc": datetime.now(timezone.utc).isoformat(),
              "workers": 4, "seeds": random.Random(2026101810).sample(range(6000000, 7000000), 6),
              "opponents": ["v1", "market"], "policies": ["control", "land"], "seats": [0, 1], "evaluation_games": 48,
              "parity_games": 2, "repeat_games": 2, "no_selection_or_retuning": True,
              "intervention": "Frozen round2 controller; append no rules except insert BUY_LAND as first market order at step0; buy exactly NE once; identical controller subsequently",
              "hypothesis": "Land may relieve capacity limits; no claim about optimal expansion timing or all animal/fertilizer strategies",
              "gate": "Exploratory support only if seed-cluster overall profit 95% lower >0 and mean match-score delta nonnegative for each opponent; always report both opponents and both players incomes",
              "bootstrap": {"resamples": 20000, "rng_seed": 2026101811},
              "parity": "Before evaluation compare control-adapter vs bare baseline on first seed vs market side0; same semantic replay and banks required",
              "repeat": "After panel repeat land vs market first seed both seats sequentially; same semantic hashes and banks",
              "frozen_existing": {p: sha(ROOT / p) for p in ("main.py", "bots/heuristic_v1.py", "bots/heuristic_v2.py", "bots/heuristic_v1_jit_seeds.py", "bots/heuristic_v1_market.py")},
              "source_sha256": {p.name: sha(p) for p in src.iterdir()},
              "engine_sha256": "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
              "engine_commit": "f994f2050d58d84f649329c1e35ce09ada7007c5",
              "scope": "Six seed clusters, fixed local opponents, single non-optimized intervention; no PPO, GPU training, default promotion or raw replay publication"}
    if reuse:
        prior = json.loads((reuse / "registration.json").read_text(encoding="utf-8"))
        assert prior["seeds"] == config["seeds"] and prior["source_sha256"]["baseline.py"] == config["source_sha256"]["baseline.py"]
        config["recovery"] = {"prior_registration_sha256": sha(reuse / "registration.json"),
                              "reason": "Initial codec rejected official-valid float order quantities. Compatibility fix only, no strategy/seed/gate changes",
                              "reuse": "24 completed control games and 2 parity games unchanged; initial failed land attempts retained, never replaced as completed results",
                              "fresh_holdout_claim": False}
    save(out / "registration.json", config)
    (out / "registration.sha256").write_text(sha(out / "registration.json") + "\n")
    print(json.dumps({"prepared": True, "seeds": config["seeds"], "games": 52}))


def telemetry(folder, row):
    replay = json.loads(gzip.decompress((folder / "replay.json.gz").read_bytes()))
    p = row["side"]
    expanded_plants = actions = 0
    for before, after in zip(replay["steps"], replay["steps"][1:]):
        farm = before[0]["observation"]["farms"][p]
        positions = [farm["farmer"], *farm["hands"]]
        a = after[p]["action"]
        for pos, atom in zip(positions, [a["farmer"], *a["hands"]]):
            actions += atom[0] != "PASS"
            x, y = pos
            tile = after[0]["observation"]["farms"][p]["tiles"][y][x]
            if atom[0] == "PLANT" and x >= len(farm["tiles"]) // 2 and isinstance(tile, dict) and tile.get("crop") == atom[1] and farm["tiles"][y][x] is None:
                expanded_plants += 1
    first = replay["steps"][1][0]["observation"]["farms"][p]
    last = replay["steps"][-1][0]["observation"]["farms"][p]
    prices = [s[0]["observation"]["market"]["prices"]["MELON"] for s in replay["steps"]]
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    return {"first_post_action_quadrants": first["unlocked_quadrants"], "final_quadrants": last["unlocked_quadrants"],
            "successful_plant_events_in_new_land": expanded_plants, "non_pass_unit_actions": actions,
            "melon_quote_min": min(prices), "melon_quote_mean": mean(prices),
            "simulation_seconds": result["run_seconds"], "both_bot_seconds": sum(result["bot_seconds"])}


def run(out, reuse=None):
    cfg = json.loads((out / "registration.json").read_text(encoding="utf-8"))
    assert sha(out / "registration.json") == (out / "registration.sha256").read_text().strip()
    assert not (out / "games").exists(), "No silent resume or replacement of completed games"
    for name, h in cfg["source_sha256"].items():
        assert sha(out / "sources" / name) == h
    assert sha(ROOT / "decision/api.py") == cfg["source_sha256"]["api.py"]
    rows, repeats = [], []
    with Games(out, cfg, "experiment") as games, ThreadPoolExecutor(max_workers=4) as pool:
        seed = cfg["seeds"][0]
        if reuse:
            assert sha(reuse / "registration.json") == cfg["recovery"]["prior_registration_sha256"]
            shutil.copytree(reuse / "parity", out / "parity")
            parity = json.loads((reuse / "parity.json").read_text(encoding="utf-8"))["games"]
        else:
            parity = [games.play(out / "sources" / (name + ".py"), name, "market", seed, 0, out / "parity" / name)
                      for name in ("baseline", "control")]
        assert parity[0]["banks"] == parity[1]["banks"] and parity[0]["semantic_replay_sha256"] == parity[1]["semantic_replay_sha256"]
        save(out / "parity.json", {"passed": True, "games": parity})
        jobs = []
        for seed in cfg["seeds"]:
            for op in cfg["opponents"]:
                for side in (0, 1):
                    for policy in cfg["policies"]:
                        folder = out / "games" / f"{seed}-{op}-{side}-{policy}"
                        if reuse and policy == "control":
                            original = reuse / "games" / folder.name
                            result = json.loads((original / "result.json").read_text(encoding="utf-8"))
                            assert result["passed"] and result["bot_sha256"] == cfg["source_sha256"]["control.py"]
                            shutil.copytree(original, folder)
                            fields = ("seed", "side", "banks", "bank", "opponent_bank", "net_profit", "margin", "states", "action_turns", "bot_sha256", "rival_sha256", "engine_sha256", "package", "passed", "semantic_replay_sha256")
                            row = {k: result[k] for k in fields}
                            row.update(policy=policy, opponent=op, final_status=["DONE", "DONE"], rewards=result["banks"],
                                       replay_gzip_sha256=sha(folder / "replay.json.gz"), agent_errors=0, both_player_farming_and_seed_audits_passed=True)
                            row["mechanism"] = telemetry(folder, row)
                            row["reused_completed_control"] = True
                            rows.append(row)
                            continue
                        jobs.append((pool.submit(games.play, out / "sources" / (policy + ".py"), policy, op, seed, side, folder,
                                                 seed == cfg["seeds"][0] and op == "market"), folder))
        folders = {future: folder for future, folder in jobs}
        for future in as_completed(folders):
            r = future.result()
            r["mechanism"] = telemetry(folders[future], r)
            assert r["mechanism"]["final_quadrants"] == (["NW", "NE"] if r["policy"] == "land" else ["NW"])
            rows.append(r)
            print(f"Completed {len(rows)}/48", flush=True)
        for side in (1, 0):
            r = games.play(out / "sources/land.py", "land", "market", cfg["seeds"][0], side, out / "reproducibility" / str(side))
            orig = next(x for x in rows if x["policy"] == "land" and x["opponent"] == "market" and x["seed"] == r["seed"] and x["side"] == side)
            assert r["banks"] == orig["banks"] and r["semantic_replay_sha256"] == orig["semantic_replay_sha256"]
            repeats.append(r)
    two = [{**r, "policy": "candidate" if r["policy"] == "land" else "baseline"} for r in rows]
    summary, paired = analyze(two, {"seeds": cfg["seeds"], "opponents": cfg["opponents"], "policies": ["baseline", "candidate"], "bootstrap": cfg["bootstrap"]})
    # This experiment's gate is predeclared separately from the old helper's gate.
    support = summary["overall"]["profit_95_ci"][0] > 0 and all(summary["by_opponent"][op]["paired"]["mean_score_delta"] >= 0 for op in cfg["opponents"])
    summary["exploratory_land_support_gate"] = bool(support)
    summary["helper_gate_not_used_for_this_experiment"] = summary.pop("statistical_gate_passed")
    summary["costs_and_coverage"] = {p: {"expanded_plant_events": sum(r["mechanism"]["successful_plant_events_in_new_land"] for r in rows if r["policy"] == p),
                                       "mean_non_pass_actions": mean(r["mechanism"]["non_pass_unit_actions"] for r in rows if r["policy"] == p),
                                       "mean_simulation_seconds": mean(r["mechanism"]["simulation_seconds"] for r in rows if r["policy"] == p)} for p in cfg["policies"]}
    save(out / "ledger.json", sorted(rows, key=lambda r: (r["seed"], r["opponent"], r["side"], r["policy"])))
    save(out / "paired-results.json", paired)
    save(out / "summary.json", summary)
    save(out / "reproducibility.json", {"passed": True, "games": repeats})
    assert all(sha(ROOT / p) == h for p, h in cfg["frozen_existing"].items())
    for name, h in cfg["source_sha256"].items():
        assert sha(out / "sources" / name) == h
    assert sha(ROOT / "decision/api.py") == cfg["source_sha256"]["api.py"]
    save(out / "acceptance.json", {"complete_official_games": 52, "adapter_semantic_parity": True, "reproducibility": True,
                                   "frozen_sources_and_default_preserved": True, "workers_closed": json.loads((out / "experiment-workers.json").read_text())["closed"],
                                   "registration_sha256": sha(out / "registration.json"), "exploratory_land_support_gate": bool(support)})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.baseline is None:
            parser.error("prepare requires --baseline")
        prepare(args.output.resolve(), args.baseline.resolve(), args.reuse_from.resolve() if args.reuse_from else None)
    else:
        run(args.output.resolve(), args.reuse_from.resolve() if args.reuse_from else None)
