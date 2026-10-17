"""Frozen v1 market variant versus all three archived bots, with both seats."""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import uuid

from crossplay import BOTS, NAMES, PROTECTED, dump, price_metrics, score, write_csv
from replay_metrics import summarize

ROOT = Path(__file__).resolve().parent
BOT = "bots/heuristic_v1_market.py"
HISTORY = ROOT.parent / "review/evidence/three-bot-crossplay.json"
GUARDED = [*PROTECTED, "Crossplay.ps1", "crossplay.py"]


def hashes(files):
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 7, 42, 2026, 314159])
    parser.add_argument("--output-dir", type=Path, help="New run directory, or an owned development run without a tournament")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    out = (ROOT / args.output_dir).resolve() if args.output_dir else ROOT / "artifacts/market-scheduling" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8])
    source = out / "sources"
    source.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() or (out / "matches").exists():
        parser.error("Tournament already exists; choose a new directory")
    before = hashes(GUARDED)
    existing = source / "protected-before.json"
    if existing.exists():
        assert json.loads(existing.read_text(encoding="utf-8")) == before
    else:
        dump(existing, before)
    participants = {"market": BOT, **BOTS}
    frozen = hashes([*GUARDED, BOT, "market_evaluate.py", "Evaluate-Market.ps1"])
    for f in frozen:
        target = source / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, target)
    dump(source / "frozen-manifest.json", frozen)
    diff = difflib.unified_diff((ROOT / BOTS["v1"]).read_text(encoding="utf-8").splitlines(True),
                               (ROOT / BOT).read_text(encoding="utf-8").splitlines(True),
                               fromfile=BOTS["v1"], tofile=BOT)
    (source / "policy-diff.patch").write_text("".join(diff), encoding="utf-8")
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    dump(out / "run.json", {"seeds": args.seeds, "expected_games": 6 * len(args.seeds), "status": "running", "command": command})
    print(f"Artifacts: {out}", flush=True)
    games, rows, daily = [], [], []
    for opponent, rival_file in BOTS.items():
        for seed in args.seeds:
            for seat in (0, 1):
                name = f"market-vs-{opponent}-seed{seed}-p{seat}"
                cmd = [sys.executable, "-B", "-X", "utf8", str(ROOT / "run_match.py"),
                       "--bot", str(source / BOT), "--opponent", str(source / rival_file),
                       "--side", str(seat), "--seed", str(seed), "--name", name,
                       "--output-root", str(out / "matches")]
                with (out / "runner.log").open("a", encoding="utf-8") as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                match = out / "matches" / name
                result = json.loads((match / "result.json").read_text(encoding="utf-8"))
                replay = json.loads((match / "replay.json").read_text(encoding="utf-8"))
                metrics = [summarize(replay, p) for p in (0, 1)]
                dump(match / "metrics-both.json", metrics)
                prices, days = price_metrics(replay, match)
                passed = result["passed"] and all(m["farming_preconditions_passed"] and m["seed_balance_matches"] for m in metrics)
                agents = ["market", opponent] if seat == 0 else [opponent, "market"]
                game = {"name": name, "opponent": opponent, "seed": seed, "market_side": seat,
                        "agents": agents, "banks": result["final_money"], "passed": passed,
                        "semantic_replay_sha256": result["semantic_replay_sha256"], "prices": prices}
                games.append(game)
                for p, bot in enumerate(agents):
                    m = metrics[p]
                    row = {"match": name, "seed": seed, "seat": p, "bot": bot,
                           "opponent": agents[1-p], "bank": game["banks"][p],
                           "opponent_bank": game["banks"][1-p], "margin": game["banks"][p] - game["banks"][1-p],
                           **{k: m[k] for k in ("net_profit", "unused_seed_cost", "seed_cost", "final_unsold_items")},
                           "melon_planted": m["planted"].get("MELON", 0), "melon_sold": m["sold_units"].get("MELON", 0),
                           "melon_peak_daily_planting": max((d.get("MELON", 0) for d in m["planting_cohorts_by_day"].values()), default=0),
                           "harvested_equals_sold": m["harvested_units"] == m["sold_units"]}
                    rows.append(row)
                daily.extend({"match": name, "opponent": opponent, "seed": seed, "seat": seat, **d} for d in days)
                dump(out / "progress.json", {"completed": len(games), "matches": games})
                print(f"{len(games)}/{6*len(args.seeds)} {name}: {game['banks']} passed={passed}", flush=True)
                del replay
                if not passed:
                    raise RuntimeError(f"Official match/player audit failed: {name}")
    own = [r for r in rows if r["bot"] == "market"]
    opponents = {}
    for opponent in BOTS:
        g = [g for g in games if g["opponent"] == opponent]
        opponents[opponent] = {"market": score([r for r in own if r["opponent"] == opponent]),
                               "rival": score([r for r in rows if r["bot"] == opponent]),
                               "melon_prices": {"floor_games": sum(x["prices"]["MELON"]["floor_frames"] > 0 for x in g),
                                                "mean_floor_frames": mean(x["prices"]["MELON"]["floor_frames"] for x in g),
                                                "mean_final": mean(x["prices"]["MELON"]["final"] for x in g),
                                                "mean_quote": mean(x["prices"]["MELON"]["mean"] for x in g)}}
    baseline_rows, matched = [], {}
    history = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else None
    if history:
        for opponent in ("v2", "v1_jit"):
            pairs = []
            for r in own:
                if r["opponent"] != opponent:
                    continue
                previous = next(g for g in history["matches"] if g["pair"] == "v1-vs-" + opponent
                                and g["seed"] == r["seed"] and g["agents"][r["seat"]] == "v1")
                bank = previous["banks"][r["seat"]]
                pairs.append({"opponent": opponent, "seed": r["seed"], "seat": r["seat"],
                              "old_v1_bank": bank, "market_bank": r["bank"], "delta": r["bank"] - bank})
            baseline_rows.extend(pairs)
            old_prices = history["pairs"]["v1-vs-" + opponent]["prices"]["MELON"]
            matched[opponent] = {"games": len(pairs), "mean_v1_net_profit": mean(p["old_v1_bank"] - 3000 for p in pairs),
                                 "mean_market_net_profit": mean(p["market_bank"] - 3000 for p in pairs),
                                 "mean_profit_delta": mean(p["delta"] for p in pairs),
                                 "improved_games": sum(p["delta"] > 0 for p in pairs),
                                 "old_melon_floor_games": old_prices["games_at_floor"],
                                 "old_mean_floor_frames": old_prices["mean_floor_fraction"] * 720}
        write_csv(out / "paired-v1-comparison.csv", baseline_rows)
        shutil.copyfile(HISTORY, source / "previous-crossplay-summary.json")
    write_csv(out / "results.csv", rows)
    write_csv(out / "market-daily-all.csv", daily)
    summary = {"games": len(games), "seeds": args.seeds, "market_score": score(own),
               "by_opponent": opponents, "by_seat": {str(p): score([r for r in own if r["seat"] == p]) for p in (0, 1)},
               "matched_v1_comparison": matched, "matches": games,
               "market_melon_peak_daily_planting": max(r["melon_peak_daily_planting"] for r in own),
               "all_player_harvested_equals_sold": all(r["harvested_equals_sold"] for r in rows)}
    dump(out / "summary.json", summary)
    report(out, summary)
    after = hashes(GUARDED)
    dump(source / "protected-after.json", after)
    frozen_same = frozen == hashes(frozen)
    passed = before == after and frozen_same and all(g["passed"] for g in games)
    dump(out / "acceptance.json", {"passed": passed, "complete_matches": len(games),
                                   "both_player_audits_passed": True, "protected_unchanged": before == after,
                                   "all_frozen_sources_unchanged": frozen_same,
                                   "all_player_harvested_equals_sold": summary["all_player_harvested_equals_sold"]})
    dump(out / "run.json", {"command": command, "status": "passed" if passed else "failed",
                           "seeds": args.seeds, "expected_games": 6 * len(args.seeds)})
    print(json.dumps({"artifacts": str(out), "passed": passed, "score": summary["market_score"], "opponents": opponents}, ensure_ascii=False))
    return 0 if passed else 1


def report(out, s):
    a = s["market_score"]
    lines = ["# 第一版市场调度改进版：六种子对三版交叉赛", "",
             f"固定种子 `{s['seeds']}`，每个对手交换双方，共 {s['games']} 场官方完整比赛。每场 720 状态、719 动作回合。",
             "新 Bot：`bots/heuristic_v1_market.py`；原版、第二版、按需购种版、默认 main.py 及原有入口均保留。", "",
             "## 改动范围", "",
             "基于第一版独立复制；农活优先级、路径与任务分配、浇水/收获规则、雇工和非甜瓜出售逻辑保留。",
             "甜瓜每天最多种半个仓库对应的产量（默认 8 株），购种量同时受该批次剩余额度限制。",
             "新种甜瓜时计入对手可见作物的上市风险：更早成熟的全计，同批成熟的按一半估计。对手私有仓库、种子和携带物均不读取。",
             "当前价低于基准 55% 时不再新增甜瓜；市场库存高于中性点时，还用每个对手可见工人一株产量作为在途风险余量，检查预估清算价。",
             "对手甜瓜成熟时，己方携带一株甜瓜即可提前运回；普通回合卖单默认至多 25 个，容量压力和终局可越过批量限制。",
             "仅在对手没有成熟甜瓜、现金及空间充裕且一日价格回升估计足够时，暂存至多一株（6 个）；最后三日不新增该缓冲。",
             "这些都是策略估计，不是新 Server；按批卖不能改变总供需，也不能阻止对手砸盘。", "",
             "## 总成绩", "",
             f"新 Bot {a['wins']} 胜 / {a['draws']} 平 / {a['losses']} 负，胜率 {a['win_rate']:.1%}；",
             f"平均最终余额 {a['mean_bank']:.2f}、平均净收益 {a['mean_net_profit']:.2f}。净收益 = 余额 − 初始资金 3000。", "",
             "| 对手 | 新版胜/平/负 | 胜率 | 新版平均净收益 | 对手平均净收益 | 平均现金差 |",
             "|---|---:|---:|---:|---:|---:|"]
    for opponent, data in s["by_opponent"].items():
        m, r = data["market"], data["rival"]
        lines.append(f"| {NAMES[opponent]} | {m['wins']}/{m['draws']}/{m['losses']} | {m['win_rate']:.1%} | "
                     f"{m['mean_net_profit']:.2f} | {r['mean_net_profit']:.2f} | {m['mean_margin']:+.2f} |")
    lines += ["", "## 甜瓜市场", "", "| 对手 | 报价触底场数 | 平均触底帧数 | 平均全程报价 | 平均末价 |",
              "|---|---:|---:|---:|---:|"]
    for opponent, d in s["by_opponent"].items():
        q = d["melon_prices"]
        lines.append(f"| {NAMES[opponent]} | {q['floor_games']}/{d['market']['games']} | {q['mean_floor_frames']:.2f} | {q['mean_quote']:.2f} | {q['mean_final']:.2f} |")
    lines += ["", "触底指公开报价等于 1，每场分母 720 帧（含初始）；公开报价不是整批实际成交均价。", "",
              "## 同对手、同种子、同席位与原第一版对照", "",
              "复用上一轮已保存且源码哈希相同的官方对局；只对第二版、按需购种版做匹配比较。原版对自身的六种子镜像赛不存在，不伪造该基线。", "",
              "| 对手 | 原第一版平均净收益 | 新版平均净收益 | 收益差 | 提高场数 | 旧→新触底场数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for opponent, d in s["matched_v1_comparison"].items():
        lines.append(f"| {NAMES[opponent]} | {d['mean_v1_net_profit']:.2f} | {d['mean_market_net_profit']:.2f} | "
                     f"{d['mean_profit_delta']:+.2f} | {d['improved_games']}/{d['games']} | "
                     f"{d['old_melon_floor_games']}→{s['by_opponent'][opponent]['melon_prices']['floor_games']} |")
    lines += ["", "## 逐场双方余额与回放", "",
              "| 对手 | 种子 | 新版席位 | P0余额 | P1余额 | 回放与报价 |", "|---|---:|---:|---:|---:|---|"]
    for g in s["matches"]:
        p = "matches/" + g["name"]
        lines.append(f"| {NAMES[g['opponent']]} | {g['seed']} | {g['market_side']} | {g['banks'][0]} | {g['banks'][1]} | "
                     f"[HTML]({p}/replay.html) · [JSON]({p}/replay.json) · [报价]({p}/market-prices.csv) |")
    lines += ["", "## 复现与限制", "",
              "执行 `./Evaluate-Market.ps1` 可重新跑新 Bot 对三版的 36 场；`-Seeds 42` 可跑 6 场。",
              "单场：`./Play.ps1 -Bot bots/heuristic_v1_market.py -Opponent bots/heuristic_v1.py -Seed 42 -Side 0`。",
              "开发局在 development/，正式对局只在 matches/；候选开发代码单独归档，正式评估期间锁定同一 SHA256。",
              "种子 42 已参与开发调试；其余五组用于扩展验收。六组都属于反复使用的回归集合，不是未见数据或正式赛事胜率估计。",
              "不同空地数量影响日终共享 RNG 的杂草抽取次数与商店抽选；相同种子不保证不同策略得到相同外生需求轨迹。不能把收益差全部归因于某一个改动。",
              "每场 result.json 记录官方 Commit、Python、包版本、实际命令及引擎一致性；metrics-both.json 核对双方农活、购种及实际状态变化。",
              "原始 Replay JSON、中文 HTML、执行日志、市场逐帧/逐日 CSV、源码及依赖快照全部保留。不修改默认 Bot，不进行 RL 或线上提交。", ""]
    (out / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
