# 自动实验

从仓库根目录执行 `kaggriculture/Setup-Learning.ps1`、`kaggriculture/Optimize.ps1`。

优化器环境只安装Optuna及依赖；真实完整比赛由游戏环境中的 `official_worker.py` 执行。默认两个本地CPU worker、8次候选；训练和测试种子预先分开，测试前锁定唯一候选。

搜索只改第一版的4个作物评分权重，小麦固定1，不增加人工策略规则。每轮将配置、候选、SQLite、比赛结果、Replay和统计结果放入本地 `artifacts/learning/`。CLI当前不提供断点续训。

复用本轮测试种子属于复现/回归。若根据其结果修改方法、范围或预算，再证明泛化需要新的预先固定留出种子。

GPU脚本仅为真实教师数据的小型作物分类探针，未作为完整游戏策略。它与CPU参数搜索分别验收。[方法研究](RESEARCH.zh-CN.md)
