"""Standalone plots from saved optimizer trials and untouched test outcomes."""
import csv
import json
from pathlib import Path
from statistics import mean
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1]).resolve()
with (out / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
paired = json.loads((out / "paired-test-results.json").read_text(encoding="utf-8"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
trial_ids = [int(r["number"]) + 1 for r in rows]
values = [float(r["value"]) for r in rows]
best = [max(values[:i+1]) for i in range(len(values))]
axes[0].plot(trial_ids, values, marker="o", label="本候选训练分")
axes[0].plot(trial_ids, best, label="截至当时的最佳训练分")
axes[0].axhline(values[0], color="gray", linestyle="--", label="原版默认权重")
axes[0].set(title="自动搜索（不含测试结果）", xlabel="候选编号", ylabel="训练面板归一化现金差")
axes[0].legend(fontsize=9)
seeds = sorted({r["seed"] for r in paired})
values = [mean(r["profit_delta"] for r in paired if r["seed"] == seed) for seed in seeds]
axes[1].bar(range(1, len(seeds)+1), values, color=["#247a68" if v >= 0 else "#bf5b50" for v in values])
axes[1].axhline(0, color="gray", linewidth=1)
axes[1].set(title="独立测试：每个新种子的配对收益差", xlabel="预先固定的新种子（排序编号）", ylabel="候选 − 原版；对3对手和2席位取均值")
for ax in axes:
    ax.grid(axis="y", alpha=.2)
fig.savefig(out / "experiment.png", dpi=160)
plt.close(fig)
