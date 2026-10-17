"""Bounded Optuna experiment; official CPU worker and sealed seed holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import queue
import shutil
from statistics import mean
import subprocess
import sys
import time
import uuid

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[1]
CROPS = ["CARROT", "TOMATO", "STRAWBERRY", "MELON"]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 5 <= args.trials <= 32:
        parser.error("This bounded first experiment accepts 5..32 trials")
    out = args.output_dir.resolve() if args.output_dir else ROOT / "artifacts/learning" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8])
    if out.exists():
        parser.error("Choose a new output directory")
    source = out / "sources"
    source.mkdir(parents=True)
    files = ["main.py", "bots/heuristic_v1.py", "bots/heuristic_v2.py", "bots/heuristic_v1_jit_seeds.py",
             "bots/heuristic_v1_market.py", "learning/policy_template.py", "learning/official_worker.py",
             "learning/optimize.py", "requirements-lock.txt", "learning/requirements-lock.txt"]
    protected = {f: digest(ROOT / f) for f in files}
    for f in files:
        target = source / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, target)
    save(source / "manifest.json", protected)
    # Committed before any training/game result. Test seeds never reach Optuna.
    rng = random.Random(20261018)
    test_seeds = rng.sample(range(900000, 999999), 12)
    config = {"algorithm": "Optuna 4.5.0 TPESampler", "sampler_seed": 20261018,
              "n_startup_trials": 4, "trials": args.trials, "space": {c: [.5, 2.] for c in CROPS},
              "parameterization": "Multiply v1 crop_score by four weights; WHEAT fixed at 1",
              "training_seeds": [200001, 200002], "training_opponents": ["v1", "v2"],
              "test_seeds": test_seeds, "test_opponents": ["v1", "v2", "jit"],
              "test_uses_both_seats": True, "selection": "training score only; exactly one finalist; no test-driven retuning",
              "objective": "mean(margin / (own bank + rival bank + 6000)) on complete games",
              "gpu_used_for_parameter_search": False, "third_party_simulator_used": False,
              "local_cpu_workers": 2,
              "promotion_gate": "seed-cluster bootstrap 95% lower bounds > 0 for both profit delta and match-score delta",
              "command": subprocess.list2cmdline([sys.executable, *sys.argv])}
    save(out / "config.json", config)
    print(f"Artifacts: {out}", flush=True)
    template = (source / "learning/policy_template.py").read_text(encoding="utf-8")
    rivals = {"v1": source / "bots/heuristic_v1.py", "v2": source / "bots/heuristic_v2.py",
              "jit": source / "bots/heuristic_v1_jit_seeds.py"}
    def export(weights, path):
        path.write_text(template + "\n# Automatically selected constants; no online adaptation.\nCROP_WEIGHTS = "
                        + repr({"WHEAT": 1., **weights}) + "\n", encoding="utf-8")
    results = []
    logs, workers = [], []
    available = queue.Queue()
    for number in range(2):
        worker_log = (out / f"worker-{number}.log").open("w", encoding="utf-8")
        logs.append(worker_log)
        worker = subprocess.Popen([str(ROOT / ".venv/Scripts/python.exe"), "-B", "-X", "utf8",
                               str(source / "learning/official_worker.py")], cwd=ROOT,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log,
                              text=True, encoding="utf-8", bufsize=1,
                              env={**os.environ, "KAGGRICULTURE_ROOT": str(ROOT)},
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        workers.append(worker)
        available.put(worker)
    pool = ThreadPoolExecutor(max_workers=2)
    # Worker snapshot is nested in sources; supply actual project root explicitly.
    def request(bot, rival, seed, side, folder, **options):
        req = {"cmd": "play", "bot": str(bot), "rival": str(rival), "seed": seed,
               "side": side, "folder": str(folder), **options}
        worker = available.get()
        try:
            worker.stdin.write(json.dumps(req) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
        finally:
            available.put(worker)
        if not line:
            raise RuntimeError("Official worker exited; inspect worker logs")
        response = json.loads(line)
        if "error" in response or not response.get("passed"):
            raise RuntimeError(response)
        return response
    start = time.perf_counter()
    try:
        # Benchmark excludes import/JSON I/O from simulation throughput, also
        # reports end-to-end time; a separate profiled game is not in rate stats.
        profile = request(rivals["v1"], rivals["v2"], 876543, 0, out / "benchmark/profile", profile=True)
        bench = [request(rivals["v1"], rivals["v2"], 876550 + i, i % 2, out / f"benchmark/game{i}") for i in range(4)]
        run_time = sum(r["run_seconds"] for r in bench)
        end_time = sum(r["total_seconds"] for r in bench)
        bench_summary = {"games": 4, "simulation_games_per_second": 4 / run_time,
                         "simulation_transitions_per_second": 4 * 719 / run_time,
                         "end_to_end_games_per_second_with_replay_audit": 4 / end_time,
                         "mean_bot_seconds_per_game": mean(sum(r["bot_seconds"]) for r in bench),
                         "mean_environment_framework_seconds": mean(r["run_seconds"] - sum(r["bot_seconds"]) for r in bench),
                         "cold_import_seconds": profile["import_seconds"], "worker_processes": 1,
                         "scope": "Sequential complete official games, callable bots; no render, includes both bot inference in simulation rate; end-to-end includes gzip replay and both player audits"}
        save(out / "benchmark.json", bench_summary)
        print("Benchmark: " + json.dumps(bench_summary), flush=True)
        # No-op parameterization must reproduce official v1 replay semantics.
        noop = source / "noop.py"
        export({c: 1. for c in CROPS}, noop)
        check = request(noop, rivals["v2"], 42, 0, out / "parameterization-check", semantic=True)
        assert check["banks"] == [46351., 30471.]
        assert check["semantic_replay_sha256"] == "065fbd1e42b597d07f92c3f290270ced2d896f9fd186e5fe26ce1bc82dee2763"
        study = optuna.create_study(direction="maximize", storage="sqlite:///" + str(out / "study.sqlite3"),
                                    study_name="v1-crop-weights", sampler=optuna.samplers.TPESampler(seed=20261018, n_startup_trials=4))
        study.enqueue_trial({c: 1. for c in CROPS})
        def objective(trial):
            weights = {c: trial.suggest_float(c, .5, 2., log=True) for c in CROPS}
            bot = out / f"training/trial{trial.number}/agent.py"
            bot.parent.mkdir(parents=True)
            export(weights, bot)
            jobs = []
            for opponent in config["training_opponents"]:
                for seed in config["training_seeds"]:
                    for side in (0, 1):
                        jobs.append(dict(bot=bot, rival=rivals[opponent], seed=seed, side=side,
                                         folder=bot.parent / f"{opponent}-seed{seed}-p{side}"))
            games = list(pool.map(lambda job: request(**job), jobs))
            value = mean(r["margin"] / (r["bank"] + r["opponent_bank"] + 6000) for r in games)
            save(bot.parent / "trial.json", {"params": weights, "value": value, "matches": games})
            print(f"Trial {trial.number+1}/{args.trials}: train objective={value:.6f}", flush=True)
            return value
        study.optimize(objective, n_trials=args.trials)
        best = study.best_trial
        champion = out / "champion.py"
        export(best.params, champion)
        save(out / "selection.json", {"trial": best.number, "params": best.params, "training_value": best.value,
                                       "champion_sha256": digest(champion), "selected_before_test": True})
        # Test is evaluated only after selection is persisted; never used in objective.
        for label, bot in (("baseline", rivals["v1"]), ("optimized", champion)):
            for opponent in config["test_opponents"]:
                jobs = []
                for seed in test_seeds:
                    for side in (0, 1):
                        jobs.append(dict(bot=bot, rival=rivals[opponent], seed=seed, side=side,
                                         folder=out / f"test/{label}-{opponent}-seed{seed}-p{side}",
                                         html=(seed == test_seeds[0] and side == 0)))
                for r in pool.map(lambda job: request(**job), jobs):
                        r.update(policy=label, opponent=opponent)
                        results.append(r)
                print(f"Test {label} vs {opponent}: {len(test_seeds)*2} games complete", flush=True)
        save(out / "test-results.json", results)
        by_key = {(r["policy"], r["opponent"], r["seed"], r["side"]): r for r in results}
        score = lambda r: 1. if r["margin"] > 0 else .5 if r["margin"] == 0 else 0.
        paired = []
        seed_clusters = []
        for seed in test_seeds:
            own = []
            for opponent in config["test_opponents"]:
                for side in (0, 1):
                    a = by_key["optimized", opponent, seed, side]
                    b = by_key["baseline", opponent, seed, side]
                    row = {"seed": seed, "opponent": opponent, "side": side,
                           "baseline_profit": b["net_profit"], "optimized_profit": a["net_profit"],
                           "profit_delta": a["net_profit"]-b["net_profit"], "score_delta": score(a)-score(b)}
                    paired.append(row)
                    own.append(row)
            seed_clusters.append([mean(r["profit_delta"] for r in own), mean(r["score_delta"] for r in own)])
        sample = np.asarray(seed_clusters)
        boot_rng = np.random.default_rng(20261018)
        indices = boot_rng.integers(0, len(sample), size=(20000, len(sample)))
        distributions = sample[indices].mean(axis=1)
        cis = np.quantile(distributions, [.025, .975], axis=0)
        summary = {"seed_clusters": len(test_seeds), "test_physical_games": len(results),
                   "training_games": args.trials * 8,
                   "mean_profit_delta": float(sample[:, 0].mean()), "profit_delta_95_ci": cis[:, 0].tolist(),
                   "mean_match_score_delta": float(sample[:, 1].mean()), "match_score_delta_95_ci": cis[:, 1].tolist(),
                   "statistically_credible_improvement_in_this_panel": bool(cis[0, 0] > 0 and cis[0, 1] > 0),
                   "bootstrap": "20000 resamples of entire seed clusters (all opponents and both seats together)",
                   "gpu_used_for_search": False, "default_bot_changed": False,
                   "elapsed_seconds": time.perf_counter()-start}
        def aggregate(g):
            return {"games": len(g), "wins": sum(r["margin"] > 0 for r in g),
                    "draws": sum(r["margin"] == 0 for r in g), "win_rate": mean(r["margin"] > 0 for r in g),
                    "mean_match_score": mean(score(r) for r in g), "mean_net_profit": mean(r["net_profit"] for r in g),
                    "mean_margin": mean(r["margin"] for r in g)}
        summary["overall"] = {p: aggregate([r for r in results if r["policy"] == p]) for p in ("baseline", "optimized")}
        summary["by_opponent"] = {op: {p: aggregate([r for r in results if r["policy"] == p and r["opponent"] == op])
                                       for p in ("baseline", "optimized")} for op in rivals}
        save(out / "paired-test-results.json", paired)
        save(out / "summary.json", summary)
        with (out / "trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["number", "value", *CROPS])
            writer.writeheader()
            writer.writerows({"number": t.number, "value": t.value, **t.params} for t in study.trials)
        assert all(digest(ROOT / f) == h for f, h in protected.items())
        save(out / "acceptance.json", {"passed": True, "all_official_games_complete": True,
                                       "frozen_source_hashes_preserved": True, "test_seeds_disjoint": True,
                                       "test_not_used_for_selection": True, "worker_closed": False,
                                       "statistical_improvement": summary["statistically_credible_improvement_in_this_panel"]})
        write_report(out, config, summary, best.params)
        print(json.dumps({"artifacts": str(out), **summary}, ensure_ascii=False), flush=True)
    finally:
        pool.shutdown(wait=True)
        for worker in workers:
            if worker.poll() is None:
                try:
                    worker.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    worker.stdin.flush()
                    worker.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    worker.terminate()
                    worker.wait(timeout=15)
        for log in logs:
            log.close()
    acceptance = json.loads((out / "acceptance.json").read_text(encoding="utf-8"))
    acceptance["worker_closed"] = all(worker.poll() is not None for worker in workers)
    save(out / "acceptance.json", acceptance)


def write_report(out, config, s, weights):
    lines = ["# 首轮自动优化实验", "", "Optuna/TPE 自动调整第一版 4 个作物评分权重，小麦固定 1；没有继续人工添加策略规则。",
             "只使用官方 Python 引擎。训练/测试均 720 状态、719 动作回合；引擎、默认 Bot 不改。", "",
             f"训练：{s['training_games']} 场，种子 {config['training_seeds']}，对手第一版/第二版；{config['trials']} 次候选中的默认权重是基准试验。",
             f"独立测试：{s['test_physical_games']} 场（原版与唯一入选候选各 72 场），12 个预先固定的新种子，每个对手交换双方。",
             "按需购种版未进入训练对手池。测试只在选择冠军并写入 selection.json 后运行，之后不按测试结果继续调参。", "",
             f"入选参数：`{weights}`。", "", "| 策略 | 胜/平/负 | 胜率 | 平均净收益 | 平均现金差 |",
             "|---|---:|---:|---:|---:|"]
    for p, a in s["overall"].items():
        lines.append(f"| {p} | {a['wins']}/{a['draws']}/{a['games']-a['wins']-a['draws']} | {a['win_rate']:.1%} | {a['mean_net_profit']:.2f} | {a['mean_margin']:+.2f} |")
    lines += ["", f"配对平均收益差：{s['mean_profit_delta']:+.2f}；按完整种子聚类 bootstrap 的 95% 区间 {s['profit_delta_95_ci']}。",
              f"比赛得分差（胜1/平0.5/负0）：{s['mean_match_score_delta']:+.4f}，95% 区间 {s['match_score_delta_95_ci']}。",
              f"预设双指标提升门槛通过：{s['statistically_credible_improvement_in_this_panel']}。该结论只针对这组对手和种子分布。", "",
              "Optuna 参数搜索在 CPU 上运行；本候选没有神经网络，不声称 GPU 加速了搜索或官方仿真。",
              "每场完整 Replay 在 training/、test/ 下（replay.json.gz，可直接用 Python gzip 解压）；测试每个对手第一组另有中文 HTML 回放。",
              "champion.py 是可独立运行的导出 Bot；study.sqlite3、trials.csv、config.json 和源码快照保留。",
              "独立测试按种子聚类，双方席位和多个对手不被伪装成独立样本；没有用训练胜率代替测试胜率。", ""]
    (out / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
