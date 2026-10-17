"""Three unchanged bots, paired official matches, both seats and shared prices."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import uuid

from replay_metrics import summarize

ROOT = Path(__file__).resolve().parent
BOTS = {"v1": "bots/heuristic_v1.py", "v2": "bots/heuristic_v2.py",
        "v1_jit": "bots/heuristic_v1_jit_seeds.py"}
NAMES = {"v1": "第一版", "v2": "第二版", "v1_jit": "按需购种版"}
PROTECTED = ["main.py", *BOTS.values(), "bots/wheat_baseline.py", "Play.ps1",
             "Evaluate.ps1", "evaluate.py", "run_match.py", "replay_metrics.py",
             "localize_replay.py", "viewer_zh.js", "viewer_zh.css", "requirements-lock.txt"]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in PROTECTED}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def price_metrics(replay, out):
    frames = [s[0]["observation"] for s in replay["steps"]]
    products = list(frames[0]["market"]["prices"])
    rows = [{"step": f["step"], "day": f["day"], "hour": f["hour"],
             **{c: f["market"]["prices"][c] for c in products},
             **{c + "_inventory": f["market"]["inventory"].get(c, 0) for c in products}}
            for f in frames]
    write_csv(out / "market-prices.csv", rows)
    stats = {}
    daily = []
    for c in products:
        quotes = [r[c] for r in rows]
        first = next((r for r in rows if r[c] == 1), None)
        stats[c] = {"initial": quotes[0], "final": quotes[-1], "min": min(quotes),
                    "max": max(quotes), "mean": mean(quotes),
                    "floor_frames": quotes.count(1), "frames": len(quotes),
                    "floor_fraction": quotes.count(1) / len(quotes),
                    "first_floor": {k: first[k] for k in ("step", "day", "hour")} if first else None}
    for day in sorted({r["day"] for r in rows}):
        group = [r for r in rows if r["day"] == day]
        for c in products:
            values = [r[c] for r in group]
            daily.append({"day": day, "product": c, "open": values[0], "close": values[-1],
                          "min": min(values), "max": max(values), "mean": mean(values),
                          "frames": len(values), "floor_frames": values.count(1)})
    dump(out / "market-summary.json", stats)
    write_csv(out / "market-daily.csv", daily)
    return stats, daily


def score(rows):
    wins = sum(r["bank"] > r["opponent_bank"] for r in rows)
    ties = sum(r["bank"] == r["opponent_bank"] for r in rows)
    return {"games": len(rows), "wins": wins, "draws": ties,
            "losses": len(rows) - wins - ties, "win_rate": wins / len(rows),
            **{f"mean_{key}": mean(r[key] for r in rows) for key in
               ("bank", "net_profit", "opponent_bank", "margin", "unused_seed_cost", "seed_cost")},
            "min_bank": min(r["bank"] for r in rows), "max_bank": max(r["bank"] for r in rows),
            "unspent_seed_games": sum(r["unused_seed_cost"] > 0 for r in rows),
            "unsold_product_games": sum(r["final_unsold_items"] > 0 for r in rows)}


def build_summary(out, games, rows, daily, seeds):
    scores = {b: score([r for r in rows if r["bot"] == b]) for b in BOTS}
    pairs = {}
    for a, b in combinations(BOTS, 2):
        key = a + "-vs-" + b
        relevant = [g for g in games if g["pair"] == key]
        products = relevant[0]["market"]
        prices = {c: {"games_at_floor": sum(g["market"][c]["floor_frames"] > 0 for g in relevant),
                      "mean_initial": mean(g["market"][c]["initial"] for g in relevant),
                      "mean_final": mean(g["market"][c]["final"] for g in relevant),
                      "mean_quote": mean(g["market"][c]["mean"] for g in relevant),
                      "min_quote": min(g["market"][c]["min"] for g in relevant),
                      "mean_floor_fraction": mean(g["market"][c]["floor_fraction"] for g in relevant)}
                  for c in products}
        pairs[key] = {"games": len(relevant),
                      "scores": {p: score([r for r in rows if r["pair"] == key and r["bot"] == p])
                                 for p in (a, b)}, "prices": prices,
                      "mean_combined_bank": mean(sum(g["banks"]) for g in relevant)}
    by_seat = {b: {str(s): score([r for r in rows if r["bot"] == b and r["seat"] == s])
                   for s in (0, 1)} for b in BOTS}
    write_csv(out / "results.csv", rows)
    write_csv(out / "market-daily-all.csv", daily)
    prices_flat = [{"pair": pair, "product": c, **stats}
                   for pair, p in pairs.items() for c, stats in p["prices"].items()]
    write_csv(out / "market-comparison.csv", prices_flat)
    summary = {"seeds": seeds, "games": len(games), "scores": scores, "by_seat": by_seat,
               "pairs": pairs, "matches": games, "all_matches_passed": all(g["passed"] for g in games)}
    dump(out / "summary.json", summary)
    text = ["# 三版 Bot 六组种子交叉赛", "",
            f"固定种子 `{seeds}`；三版两两对战，每组交换双方，共 {len(games)} 场。策略及默认 Bot 未改。",
            "每场官方默认 720 个状态、719 个动作回合；双方 DONE，奖励等于最终余额。初始资金均为 3000。", "",
            "## 总成绩", "", "| Bot | 胜/平/负 | 胜率 | 平均余额 | 平均净收益 | 平均现金差 | 平均剩种成本 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for b, s in scores.items():
        text.append(f"| {NAMES[b]} | {s['wins']}/{s['draws']}/{s['losses']} | {s['win_rate']:.1%} | "
                    f"{s['mean_bank']:.2f} | {s['mean_net_profit']:.2f} | {s['mean_margin']:+.2f} | {s['mean_unused_seed_cost']:.2f} |")
    text += ["", f"每版各 {4*len(seeds)} 场，对另两版各 {2*len(seeds)} 场。胜率按余额严格更高计，平局不算胜。", "",
             "## 两两结果与市场", "",
             f"价格是共享市场的逐帧公开报价，不能当作整批成交均价。以下每对阵均值覆盖 {2*len(seeds)} 场、每场 720 帧（含初始）；触底指报价等于 1。"]
    for pair, p in pairs.items():
        a, b = pair.split("-vs-")
        s, t = p["scores"][a], p["scores"][b]
        text += ["", f"### {NAMES[a]} vs {NAMES[b]}", "",
                 f"{NAMES[a]} {s['wins']} 胜、{s['draws']} 平、{s['losses']} 负；平均余额 "
                 f"{s['mean_bank']:.2f} vs {t['mean_bank']:.2f}，现金差 {s['mean_margin']:+.2f}。",
                 f"平均双方余额合计 {p['mean_combined_bank']:.2f}。", "",
                 "| 商品 | 初始价 | 最低价 | 平均报价 | 平均末价 | 触底场数 | 触底帧占比 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for c in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
            q = p["prices"][c]
            text.append(f"| {c} | {q['mean_initial']:.2f} | {q['min_quote']} | {q['mean_quote']:.2f} | "
                        f"{q['mean_final']:.2f} | {q['games_at_floor']}/{p['games']} | {q['mean_floor_fraction']:.1%} |")
    text += ["", "## 逐场双方余额与回放", "",
             "| 对阵 | 种子 | P0 / P1 | P0余额 | P1余额 | 价格与回放 |", "|---|---:|---|---:|---:|---|"]
    for g in games:
        folder = "matches/" + g["name"]
        text.append(f"| {g['pair']} | {g['seed']} | {' / '.join(g['agents'])} | {g['banks'][0]} | "
                    f"{g['banks'][1]} | [HTML]({folder}/replay.html) · [JSON]({folder}/replay.json) · [报价]({folder}/market-prices.csv) |")
    text += ["", "## 解释边界与复现", "",
             "市场供给由双方卖货共同累积，后续报价会受影响；城镇消费也会消耗库存并推高报价。这是有共享市场的竞争，不是零和游戏。",
             "同种子仍会因两方空地数量不同，改变日终杂草试验后的商店抽选序列；各对阵的外生需求轨迹不保证相同。",
             "六组种子属于现有回归集合，交换席位也不是独立随机样本；本报告只描述这三版在此集合上的成绩，不推断正式赛事胜率。",
             "逐场 result.json 记录官方 Commit、Python、依赖、动作状态哈希、实际命令和引擎完整性；metrics-both.json 记录双方农活及收支核验。",
             "sources/ 保留参赛代码、执行脚本、依赖锁与运行前后哈希。市场逐帧/逐日数据在每场及汇总 CSV 中。",
             "在项目目录执行 `./Crossplay.ps1` 可新建一轮同配置比赛；`./Crossplay.ps1 -Seeds 42` 可只跑一个种子的六场。",
             "单场用 `./Play.ps1 -Bot bots/heuristic_v1.py -Opponent bots/heuristic_v2.py -Seed 42 -Side 0`。",
             "本轮不调策略，不改 main.py，不更改 Play.ps1 / Evaluate.ps1 默认行为。", ""]
    (out / "REPORT.zh-CN.md").write_text("\n".join(text), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 7, 42, 2026, 314159])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    out = ROOT / "artifacts" / "crossplay" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8])
    source = out / "sources"
    source.mkdir(parents=True)
    before = hashes()
    for name in [*PROTECTED, "crossplay.py", "Crossplay.ps1"]:
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    dump(source / "protected-before.json", before)
    dump(out / "run.json", {"seeds": args.seeds, "expected_games": 6 * len(args.seeds),
                           "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
                           "bot_sources": BOTS, "status": "running"})
    print(f"Artifacts: {out}", flush=True)
    games, rows, daily = [], [], []
    for a, b in combinations(BOTS, 2):
        pair = a + "-vs-" + b
        for seed in args.seeds:
            for seat in (0, 1):
                name = f"{pair}-seed{seed}-a{seat}"
                cmd = [sys.executable, "-B", "-X", "utf8", str(ROOT / "run_match.py"),
                       "--bot", str(source / BOTS[a]), "--opponent", str(source / BOTS[b]),
                       "--side", str(seat), "--seed", str(seed), "--name", name,
                       "--output-root", str(out / "matches")]
                with (out / "runner.log").open("a", encoding="utf-8") as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                match = out / "matches" / name
                result = json.loads((match / "result.json").read_text(encoding="utf-8"))
                replay = json.loads((match / "replay.json").read_text(encoding="utf-8"))
                agents = [a, b] if seat == 0 else [b, a]
                metrics = [summarize(replay, p) for p in (0, 1)]
                dump(match / "metrics-both.json", metrics)
                prices, days = price_metrics(replay, match)
                okay = result["passed"] and all(m["farming_preconditions_passed"] and m["seed_balance_matches"] for m in metrics)
                game = {"name": name, "pair": pair, "seed": seed, "agents": agents,
                        "banks": result["final_money"], "passed": okay,
                        "semantic_replay_sha256": result["semantic_replay_sha256"], "market": prices}
                games.append(game)
                for p, bot in enumerate(agents):
                    m = metrics[p]
                    rows.append({"match": name, "pair": pair, "seed": seed, "seat": p,
                                 "bot": bot, "opponent": agents[1-p], "bank": result["final_money"][p],
                                 "opponent_bank": result["final_money"][1-p],
                                 "margin": result["final_money"][p] - result["final_money"][1-p],
                                 **{k: m[k] for k in ("net_profit", "seed_cost", "unused_seed_cost", "final_unsold_items")}})
                daily.extend({"pair": pair, "match": name, "seed": seed, "a_seat": seat, **d} for d in days)
                dump(out / "progress.json", {"completed": len(games), "matches": games})
                print(f"{len(games)}/{6*len(args.seeds)} {name}: {game['banks']} passed={okay}", flush=True)
                if not okay:
                    raise RuntimeError(f"Match audit failed: {match}")
                del replay
    summary = build_summary(out, games, rows, daily, args.seeds)
    after = hashes()
    dump(source / "protected-after.json", after)
    passed = summary["all_matches_passed"] and before == after and len(games) == 6 * len(args.seeds)
    dump(out / "acceptance.json", {"passed": passed, "complete_matches": len(games),
                                   "protected_unchanged": before == after, "both_player_audits_passed": summary["all_matches_passed"],
                                   "market_frames_per_game": 720, "price_is_quote_not_realized_execution_price": True})
    dump(out / "run.json", {"seeds": args.seeds, "expected_games": 6 * len(args.seeds),
                           "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
                           "bot_sources": BOTS, "status": "passed" if passed else "failed"})
    print(json.dumps({"artifacts": str(out), "passed": passed, "scores": summary["scores"]}, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
