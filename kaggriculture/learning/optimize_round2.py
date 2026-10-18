"""Small multi-opponent parameter search with separately sealed train/test stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import random
import shutil
from statistics import mean, pstdev
import subprocess
import sys

import optuna
from evaluate_holdout import analyze, PANEL, save, sha

ROOT = Path(__file__).resolve().parents[1]
CROPS = ["CARROT", "TOMATO", "STRAWBERRY", "MELON"]


def prepare(out, incumbent):
    out.mkdir(parents=True, exist_ok=False)
    src = out / "sources"
    src.mkdir()
    files = {"incumbent.py": incumbent, "baseline.py": ROOT / PANEL["v1"], "default.py": ROOT / "main.py",
             **{k + ".py": ROOT / v for k, v in PANEL.items()},
             "policy_template.py": ROOT / "learning/policy_template.py",
             "worker.py": ROOT / "learning/round2_worker.py", "evaluate_holdout.py": ROOT / "learning/evaluate_holdout.py",
             "optimize_round2.py": Path(__file__), "requirements-lock.txt": ROOT / "requirements-lock.txt",
             "learning-lock.txt": ROOT / "learning/requirements-lock.txt"}
    for name, path in files.items():
        shutil.copyfile(path, src / name)
    assert sha(src / "incumbent.py") == "2c61c29c146f128f00d3d10dbf6a848e9e3024c5d06d41a984c33960bb2a7cec"
    # Development seeds are already disclosed by the previous holdout. They
    # cannot serve as unseen tests again; selection here is fixed before search.
    config = {"experiment": "round2-multi-opponent-parameters", "registered_utc": datetime.now(timezone.utc).isoformat(),
              "trials": 8, "workers": 4, "sampler": "Optuna 4.5.0 TPESampler", "sampler_seed": 2026101804,
              "startup_trials": 4, "queued_trials": ["incumbent weights", "all weights one"],
              "space": {c: [0.5, 2.0] for c in CROPS}, "scale": "log", "wheat_weight": 1.0,
              "incumbent_weights": {"CARROT": 0.6616130400855971, "TOMATO": 0.7568029243045484,
                                    "STRAWBERRY": 1.0548617343466897, "MELON": 0.5818765524810924},
              "development_seeds": [2903989, 2762846, 2557431],
              "development_seed_status": "Previously disclosed holdout; now explicitly development-only, never claimed unseen",
              "opponents": list(PANEL), "seats": [0, 1],
              "development_games": 192, "test_games": 288, "reproducibility_games": 2,
              "development_objective": "Group both seats for each seed/opponent: x=mean(margin/(own bank+rival bank+6000)); maximize mean(x)-0.25*population_std(x), 12 equally weighted groups",
              "objective_penalty": 0.25,
              "selection": "Highest development objective only; ties choose earliest trial; freeze exactly one finalist before test; no test-driven changes",
              "test_seeds": random.Random(2026101805).sample(range(4000000, 5000000), 12),
              "test_seed_generator": "random.Random(2026101805).sample(range(4000000,5000000),12)",
              "test_policies": ["baseline", "incumbent", "candidate"],
              "bootstrap": {"unit": "whole seed, all opponents and seats together", "resamples": 20000, "rng_seed": 2026101806},
              "improvement_gate": "Against BOTH baseline and incumbent: overall paired profit 95% lower bound > 0 AND market-only paired profit 95% lower bound > 0 AND overall and market-only mean match-score deltas >= 0; all official/reproducibility/source gates pass",
              "multiplicity": "All four profit conditions conjunctive, not alternative chances to pass; other subgroup stats descriptive; score point checks are not noninferiority proofs",
              "official_gate": "All 480 scheduled games and 2 repeats: 720 states,719 actions,DONE/DONE,rewards equal banks,no errors/stderr,both player farming/seed audits pass",
              "repeat_gate": "Repeat test seed[0], candidate vs market, seats 1 then 0 sequentially after the four-worker test; banks and semantic replay hashes identical",
              "failure_policy": "No seed/opponent/budget/gate changes after registration; valid completed games never replaced; report failed gate as failed; any infrastructure retry separately documented",
              "source_sha256": {p.name: sha(p) for p in sorted(src.iterdir())},
              "engine_commit": "f994f2050d58d84f649329c1e35ce09ada7007c5",
              "engine_sha256": "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
              "scope": "Official CPU simulator; no GPU training, PPO, manual strategy changes or default promotion; fixed local opponents only",
              "security": "Trusted local code only; fresh per-player per-game namespace, not a sandbox"}
    assert not set(config["development_seeds"]) & set(config["test_seeds"])
    save(out / "registration.json", config)
    (out / "registration.sha256").write_text(sha(out / "registration.json") + "\n")
    print(json.dumps({"prepared": True, "development_games": 192, "test_games": 288, "test_seeds": config["test_seeds"]}), flush=True)


def frozen(out):
    cfg = json.loads((out / "registration.json").read_text(encoding="utf-8"))
    assert sha(out / "registration.json") == (out / "registration.sha256").read_text().strip()
    for name, digest in cfg["source_sha256"].items():
        assert sha(out / "sources" / name) == digest
    assert sha(Path(__file__)) == cfg["source_sha256"]["optimize_round2.py"]
    assert sha(ROOT / "learning/evaluate_holdout.py") == cfg["source_sha256"]["evaluate_holdout.py"]
    for name, path in {"default": "main.py", **PANEL}.items():
        assert sha(ROOT / path) == cfg["source_sha256"][name + ".py"]
    return cfg


class Games:
    def __init__(self, out, cfg, stage):
        self.out, self.cfg, self.stage = out, cfg, stage
        self.available, self.workers, self.logs = queue.Queue(), [], []

    def __enter__(self):
        try:
            for i in range(self.cfg["workers"]):
                log = (self.out / f"{self.stage}-worker-{i}.log").open("w", encoding="utf-8")
                self.logs.append(log)
                p = subprocess.Popen([str(ROOT / ".venv/Scripts/python.exe"), "-B", "-X", "utf8", str(self.out / "sources/worker.py")],
                                     cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                     text=True, encoding="utf-8", bufsize=1,
                                     env={**os.environ, "KAGGRICULTURE_ROOT": str(ROOT)},
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                self.workers.append(p)
                self.available.put(p)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for p in self.workers:
            if p.poll() is None:
                try:
                    p.stdin.write('{"cmd":"close"}\n')
                    p.stdin.flush()
                    p.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    p.terminate()
                    p.wait(timeout=15)
            p.stdin.close()
            p.stdout.close()
        for log in self.logs:
            log.close()
        save(self.out / f"{self.stage}-workers.json", {"closed": all(p.poll() is not None for p in self.workers), "workers": len(self.workers)})

    def play(self, bot, policy, opponent, seed, side, folder, html=False, trial=None):
        p = self.available.get()
        try:
            request = {"cmd": "play", "bot": str(bot), "rival": str(self.out / "sources" / (opponent + ".py")),
                       "seed": seed, "side": side, "folder": str(folder), "html": html, "semantic": True}
            p.stdin.write(json.dumps(request) + "\n")
            p.stdin.flush()
            response = p.stdout.readline()
            if not response:
                raise RuntimeError("Worker exited; inspect local private logs")
            r = json.loads(response)
            if "error" in r or not r.get("passed"):
                raise RuntimeError("Official game failed; inspect local private logs")
            assert r["bot_sha256"] == sha(bot)
            assert r["rival_sha256"] == self.cfg["source_sha256"][opponent + ".py"]
            assert r["engine_sha256"] == self.cfg["engine_sha256"]
            keys = ["seed", "side", "banks", "bank", "opponent_bank", "net_profit", "margin", "states", "action_turns",
                    "bot_sha256", "rival_sha256", "engine_sha256", "package", "passed", "semantic_replay_sha256"]
            safe = {k: r[k] for k in keys}
            for k in ("semantic_replay_sha256_version", "semantic_replay_sha256_v2", "audit_version"):
                if k in r:
                    safe[k] = r[k]
            safe.update(policy=policy, opponent=opponent, final_status=["DONE", "DONE"], rewards=r["banks"],
                        replay_gzip_sha256=sha(folder / "replay.json.gz"), agent_errors=0,
                        both_player_farming_and_seed_audits_passed=True)
            if trial is not None:
                safe["trial"] = trial
            return safe
        finally:
            self.available.put(p)


def train(out):
    cfg = frozen(out)
    assert not (out / "training").exists() and not (out / "selection.json").exists()
    template = (out / "sources/policy_template.py").read_text(encoding="utf-8")
    study = optuna.create_study(direction="maximize", study_name="round2", storage="sqlite:///" + str(out / "study.sqlite3"),
                                sampler=optuna.samplers.TPESampler(seed=cfg["sampler_seed"], n_startup_trials=cfg["startup_trials"]))
    study.enqueue_trial(cfg["incumbent_weights"])
    study.enqueue_trial({c: 1. for c in CROPS})
    rows = []
    with Games(out, cfg, "train") as games, ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        def objective(trial):
            weights = {c: trial.suggest_float(c, .5, 2., log=True) for c in CROPS}
            folder = out / "training" / f"trial{trial.number}"
            folder.mkdir(parents=True)
            bot = folder / "agent.py"
            bot.write_text(template + "\n# Automatically selected constants; no online adaptation.\nCROP_WEIGHTS = "
                           + repr({"WHEAT": 1., **weights}) + "\n", encoding="utf-8")
            if trial.number == 0:
                assert sha(bot) == cfg["source_sha256"]["incumbent.py"]
            jobs = [pool.submit(games.play, bot, "trial", op, seed, side, folder / f"{seed}-{op}-{side}", False, trial.number)
                    for seed in cfg["development_seeds"] for op in cfg["opponents"] for side in (0, 1)]
            matches = [job.result() for job in jobs]
            groups = [mean(r["margin"] / (r["bank"] + r["opponent_bank"] + 6000) for r in matches if r["seed"] == seed and r["opponent"] == op)
                      for seed in cfg["development_seeds"] for op in cfg["opponents"]]
            value = mean(groups) - cfg["objective_penalty"] * pstdev(groups)
            save(folder / "trial.json", {"trial": trial.number, "weights": weights, "objective": value, "group_values": groups, "bot_sha256": sha(bot)})
            rows.extend(matches)
            save(out / "development-ledger.json", rows)
            print(f"Trial {trial.number + 1}/8: objective={value:.8f}; completed {len(rows)}/192 development games", flush=True)
            return value
        study.optimize(objective, n_trials=cfg["trials"])
    assert len(rows) == cfg["development_games"]
    best = max(study.trials, key=lambda t: (t.value, -t.number))
    shutil.copyfile(out / "training" / f"trial{best.number}" / "agent.py", out / "champion.py")
    save(out / "selection.json", {"trial": best.number, "params": best.params, "objective": best.value,
                                   "champion_sha256": sha(out / "champion.py"), "selected_utc": datetime.now(timezone.utc).isoformat(),
                                   "selection_used_development_only": True, "test_not_started": not (out / "test").exists(),
                                   "registration_sha256": sha(out / "registration.json")})
    with (out / "trials.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "objective", *CROPS])
        writer.writeheader()
        writer.writerows({"trial": t.number, "objective": t.value, **t.params} for t in study.trials)
    frozen(out)
    save(out / "runtime.json", {"python": sys.version.split()[0], "optuna": importlib.metadata.version("optuna"),
                               "numpy": importlib.metadata.version("numpy"), "gpu_training": False, "cpu_workers": cfg["workers"]})
    print("TRAINING COMPLETE; freeze/publish selection before starting test", flush=True)


def test(out):
    cfg = frozen(out)
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert selection["test_not_started"] and selection["selection_used_development_only"]
    assert sha(out / "champion.py") == selection["champion_sha256"]
    assert not (out / "test").exists()
    selection_sha = sha(out / "selection.json")
    rows, repeats = [], []
    paths = {p: out / "sources" / (p + ".py") for p in ("baseline", "incumbent")}
    paths["candidate"] = out / "champion.py"
    with Games(out, cfg, "test") as games, ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        jobs = [pool.submit(games.play, paths[p], p, op, seed, side, out / "test" / f"{seed}-{op}-{side}-{p}",
                            seed == cfg["test_seeds"][0] and op == "market")
                for seed in cfg["test_seeds"] for op in cfg["opponents"] for side in (0, 1) for p in cfg["test_policies"]]
        for job in as_completed(jobs):
            rows.append(job.result())
            if len(rows) % 12 == 0:
                print(f"Test completed {len(rows)}/288", flush=True)
        assert len(rows) == cfg["test_games"]
        for side in (1, 0):
            r = games.play(paths["candidate"], "candidate", "market", cfg["test_seeds"][0], side, out / "reproducibility" / str(side))
            old = next(x for x in rows if x["policy"] == "candidate" and x["opponent"] == "market" and x["seed"] == r["seed"] and x["side"] == side)
            assert r["banks"] == old["banks"] and r["semantic_replay_sha256"] == old["semantic_replay_sha256"]
            repeats.append(r)
    rows.sort(key=lambda r: (r["seed"], r["opponent"], r["side"], r["policy"]))
    save(out / "test-ledger.json", rows)
    save(out / "reproducibility.json", {"passed": True, "extra_games": 2, "rows": repeats})
    comparisons, pairs = {}, {}
    for comparator in ("baseline", "incumbent"):
        two = [{**r, "policy": "baseline" if r["policy"] == comparator else "candidate"} for r in rows if r["policy"] in (comparator, "candidate")]
        comparisons[comparator], pairs[comparator] = analyze(two, {"seeds": cfg["test_seeds"], "opponents": cfg["opponents"],
                                                                 "policies": ["baseline", "candidate"], "bootstrap": cfg["bootstrap"]})
    summary = {"test_games": len(rows), "development_games": cfg["development_games"], "comparisons": comparisons,
               "overall_gate_passed": all(s["statistical_gate_passed"] for s in comparisons.values()),
               "default_changed": False, "gpu_training": False, "test_seed_clusters": 12}
    save(out / "summary.json", summary)
    save(out / "paired-results.json", pairs)
    frozen(out)
    assert sha(out / "champion.py") == selection["champion_sha256"] and sha(out / "selection.json") == selection_sha
    save(out / "acceptance.json", {"official_games_passed": 482, "test_games": 288, "development_games": 192,
                                   "reproducibility_games": 2, "frozen_sources_and_default_preserved": True,
                                   "selection_frozen_before_test": True, "selection_sha256": selection_sha,
                                   "registration_sha256": sha(out / "registration.json"),
                                   "workers_closed": all(json.loads((out / f"{stage}-workers.json").read_text())["closed"] for stage in ("train", "test")),
                                   "statistical_gate_passed": summary["overall_gate_passed"], "gpu_training": False})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.incumbent is None:
            parser.error("prepare requires --incumbent")
        prepare(args.output.resolve(), args.incumbent.resolve())
    else:
        {"train": train, "test": test}[args.mode](args.output.resolve())
