"""Paired official full-game evaluation, including both seats and saved source copies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate(rows):
    profits = [r["bot_profit"] for r in rows]
    return {"games": len(rows), "wins": sum(r["bot_money"] > r["opponent_money"] for r in rows),
            "draws": sum(r["bot_money"] == r["opponent_money"] for r in rows),
            "mean_final_money": statistics.mean(r["bot_money"] for r in rows),
            "mean_net_profit": statistics.mean(profits), "min_net_profit": min(profits),
            "max_net_profit": max(profits),
            "mean_opponent_final_money": statistics.mean(r["opponent_money"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 7, 42, 2026, 314159])
    parser.add_argument("--bot", default="bots/heuristic_v2.py")
    parser.add_argument("--baseline", default="bots/heuristic_v1.py")
    parser.add_argument("--opponent", default="starter")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    out = ROOT / "artifacts" / "evaluations" / name
    (out / "sources").mkdir(parents=True)
    bots, hashes = {}, {}
    for label, value in (("heuristic", args.bot), ("baseline", args.baseline)):
        source = (ROOT / value).resolve()
        snapshot = out / "sources" / (label + ".py")
        shutil.copyfile(source, snapshot)
        hashes[label] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        bots[label] = snapshot
    for filename in ("run_match.py", "replay_metrics.py", "evaluate.py", "localize_replay.py", "viewer_zh.js", "viewer_zh.css"):
        shutil.copyfile(ROOT / filename, out / "sources" / filename)
    rows, references = [], {}
    diagnostics = []

    def play(label, seed, side, repeat=False):
        name = f"{label}-seed{seed}-p{side}" + ("-repeat" if repeat else "")
        command = [sys.executable, "-B", "-X", "utf8", str(ROOT / "run_match.py"),
                   "--bot", str(bots[label]), "--opponent", args.opponent,
                   "--seed", str(seed), "--side", str(side), "--name", name,
                   "--output-root", str(out / "matches")]
        with (out / "runner.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        match = out / "matches" / name
        result = json.loads((match / "result.json").read_text("utf-8"))
        metrics = json.loads((match / "metrics.json").read_text("utf-8"))
        if completed.returncode or not result["passed"] or not metrics["farming_preconditions_passed"]:
            raise RuntimeError(f"Match/audit failed: {match}")
        row = {"label": label, "seed": seed, "side": side,
               "bot_money": result["rewards"][side], "opponent_money": result["rewards"][1 - side],
               "bot_profit": metrics["net_profit"], "hire_cost": metrics["hire_cost"],
               "seed_cost": metrics["seed_cost"], "final_unsold_items": metrics["final_unsold_items"],
               "unused_seed_cost": metrics["unused_seed_cost"],
               "max_same_crop_planted_per_day": metrics["max_same_crop_planted_per_day"],
               "match": str(match.relative_to(out)), "semantic_sha256": result["semantic_replay_sha256"]}
        diagnostics.append({"label": label, "repeat": repeat, **metrics})
        print(f"{name}: {row['bot_money']:.0f} vs {row['opponent_money']:.0f}; passed", flush=True)
        return row

    for seed in args.seeds:
        for side in (0, 1):
            for label in bots:
                row = play(label, seed, side)
                rows.append(row)
                references[label, seed, side] = row
    repeat_checks = []
    for side in (0, 1):
        repeated = play("heuristic", args.seeds[0], side, repeat=True)
        original = references["heuristic", args.seeds[0], side]
        same = repeated["semantic_sha256"] == original["semantic_sha256"]
        repeat_checks.append({"seed": args.seeds[0], "side": side, "matched": same,
                              "original": original["match"], "repeat": repeated["match"],
                              "semantic_sha256": repeated["semantic_sha256"]})
        if not same:
            raise RuntimeError("Fixed-seed replay reproduction failed")
    with (out / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {label: aggregate([r for r in rows if r["label"] == label]) for label in bots}
    improvement = grouped["heuristic"]["mean_net_profit"] - grouped["baseline"]["mean_net_profit"]
    pairs = [{"seed": seed, "side": side,
              "profit_improvement": references["heuristic", seed, side]["bot_profit"] - references["baseline", seed, side]["bot_profit"]}
             for seed in args.seeds for side in (0, 1)]
    summary = {"passed": True, "seeds": args.seeds, "seats": [0, 1], "opponent": args.opponent,
               "bot_sources": {"heuristic": args.bot, "baseline": args.baseline},
               "source_sha256": hashes, "comparative_games": len(rows), "repeat_games": len(repeat_checks),
               "aggregates": grouped, "mean_paired_profit_improvement": improvement,
               "profit_improved": improvement > 0,
               "pairs": pairs, "deterministic_repeats": repeat_checks,
               "scope": "Official default full games. Profit = final bank - 3000. No independent RNG sample claim for paired seats.",
               "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])}
    summary["diagnostics"] = {}
    for label in bots:
        group = [m for m in diagnostics if m["label"] == label and not m["repeat"]]
        summary["diagnostics"][label] = {
            "all_farming_preconditions_passed": all(m["farming_preconditions_passed"] for m in group),
            "all_harvested_products_sold": all(m["harvested_units"] == m["sold_units"] for m in group),
            "all_terminal_products_empty": all(m["final_unsold_items"] == 0 for m in group),
            "all_terminal_seeds_empty": all(not m["final_seeds"] for m in group),
            "all_seed_balances_match": all(m["seed_balance_matches"] for m in group),
            "games_with_unused_seeds": sum(bool(m["final_seeds"]) for m in group),
            "mean_unused_seed_cost": statistics.mean(m["unused_seed_cost"] for m in group),
            "mean_hire_cost": statistics.mean(m["hire_cost"] for m in group),
            "mean_seed_cost": statistics.mean(m["seed_cost"] for m in group),
            "mean_max_same_crop_planted_per_day": statistics.mean(m["max_same_crop_planted_per_day"] for m in group),
            "max_shed_items": max(m["max_shed_items"] for m in group)}
    summary["heuristic_diagnostics"] = summary["diagnostics"]["heuristic"]
    save(out / "summary.json", summary)
    report = ["# 启发式 Bot 新旧版：完整对局评估", "",
              f"新 Bot：`{args.bot}`；旧 Bot：`{args.baseline}`。源码原样保存在 sources/，哈希见 summary.json。",
              f"固定种子：{args.seeds}；每个种子交换双方，对手为 `{args.opponent}`。",
              f"比较 {len(rows)} 场，另有 {len(repeat_checks)} 场固定种子复跑。所有比赛均为官方默认完整赛季。", "",
              "| Bot | 胜/场 | 平均最终余额 | 平均净收益 | 净收益范围 |", "|---|---:|---:|---:|---:|"]
    for label, g in grouped.items():
        report.append(f"| {label} | {g['wins']}/{g['games']} | {g['mean_final_money']:,.2f} | {g['mean_net_profit']:,.2f} | {g['min_net_profit']:,.0f}～{g['max_net_profit']:,.0f} |")
    report += ["", "| 种子 | 新 Bot 位置0余额 | 新 Bot 位置1余额 | 旧 Bot 位置0 / 1余额 |", "|---|---:|---:|---:|"]
    for seed in args.seeds:
        a, b = (references["heuristic", seed, side]["bot_money"] for side in (0, 1))
        c, d = (references["baseline", seed, side]["bot_money"] for side in (0, 1))
        report.append(f"| {seed} | {a:,.0f} | {b:,.0f} | {c:,.0f} / {d:,.0f} |")
    report += ["", "| Bot | 剩种子场数 | 平均剩种子成本 | 平均每日同作物种植峰值 | 平均雇工支出 |", "|---|---:|---:|---:|---:|"]
    for label, d in summary["diagnostics"].items():
        report.append(f"| {label} | {d['games_with_unused_seeds']} | {d['mean_unused_seed_cost']:,.2f} | {d['mean_max_same_crop_planted_per_day']:,.2f} | {d['mean_hire_cost']:,.2f} |")
    report += ["", f"同种子、同位置的平均净收益增量：**{improvement:,.2f}**。",
               "功能验收通过不代表收益提高；收益是否改善以净收益增量为准。",
               "净收益 = 官方最终银行余额 − 初始资金 3000；不是产量或库存估值。",
               "市场共享，Bot 的供给会改变 starter 的售价，因此两组 starter 收益可能不同。",
               "交换位置是公平性检查，不视作两个独立随机样本。仅与 starter 和旧 Bot 比较，不能推断赛事强度。", "",
               "全部对局 DONE/DONE、720 个记录状态、719 次动作、奖励等于银行余额、无 Agent 错误，且引擎字节匹配官方 Commit。",
               "两次复跑去除 remainingOverageTime 后的完整动作/状态 SHA256 与原场相同。",
               "每场 metrics.json 包含全部雇工动作、种植/收获/出售数量、购种和雇工支出、每日资金、操作前置条件与末期库存。",
               "前置条件和实际农地变化审计只覆盖本次作物 Bot；跳过被日终刷新覆盖的变化。它不重新执行官方规则，不等于通用动作验证器。", "",
               "逐场收益见 [results.csv](results.csv)，汇总与哈希见 [summary.json](summary.json)。",
               "回放在 `matches/<Bot>-seed<种子>-p<位置>/replay.html`，JSON、日志和最终棋盘同目录。",
               "`sources/` 保存实际参赛 Bot 和运行脚本快照。开发调试局不计入本表。", "",
               "这套固定种子用于迭代与回归，不是独立的赛事强度测试。",
               "当前只经营初始 25 块地，不养动物、不买地、不做强化学习。", ""]
    (out / "REPORT.zh-CN.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"artifacts": str(out), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
