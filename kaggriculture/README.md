# Kaggriculture

官方 `kaggle-environments==1.33.0`，游戏引擎固定Commit见根目录 `upstreams.json`。默认720个状态（含初始）、719次动作转移，双人季末现金决定胜负。

从仓库根目录运行：

```powershell
.\kaggriculture\Setup.ps1
.\kaggriculture\Play.ps1 -Bot main.py -Opponent starter -Seed 42
.\kaggriculture\Crossplay.ps1
.\kaggriculture\Evaluate-Market.ps1
.\kaggriculture\Setup-Learning.ps1
.\kaggriculture\Optimize.ps1
```

`bots/` 保留原第一版、第二版、按需购种版、市场版；`models/tuned_v1.py` 为首轮自动调参导出候选，默认入口没有替换。

结果集中到 `artifacts/`，单场有JSON/HTML回放，自动实验保留gzip Replay。所有运行产物只留本地。

[进展与收益](../review/PROGRESS.md) · [实验配置](../review/evidence/experiment-config.json) · [中文规则](中文文档/README.md) · [学习研究](learning/RESEARCH.zh-CN.md)

市场对照脚本使用公开的匿名历史汇总。BC样本提取需要自行生成完整交叉赛Replay，并把交叉赛目录作为 `prepare_bc_probe.py` 的第二个位置参数传入；公开仓库不包含该原始语料。
