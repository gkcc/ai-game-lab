# 自动实验

从仓库根目录执行 `kaggriculture/Setup-Learning.ps1`、`kaggriculture/Optimize.ps1`。

优化器环境只安装Optuna及依赖；真实完整比赛由游戏环境中的 `official_worker.py` 执行。默认两个本地CPU worker、8次候选；训练和测试种子预先分开，测试前锁定唯一候选。

搜索只改第一版的4个作物评分权重，小麦固定1，不增加人工策略规则。每轮将配置、候选、SQLite、比赛结果、Replay和统计结果放入本地 `artifacts/learning/`。CLI当前不提供断点续训。

复用本轮测试种子属于复现/回归。若根据其结果修改方法、范围或预算，再证明泛化需要新的预先固定留出种子。

GPU脚本仅为真实教师数据的小型作物分类探针，未作为完整游戏策略。它与CPU参数搜索分别验收。[方法研究](RESEARCH.zh-CN.md)

新增[冻结候选独立评估](../HOLDOUT.zh-CN.md)：市场版加入4对手面板，12个新种子，交换双方共192场；完整运行通过，预登记统计提升门槛未通过。`evaluate_holdout.py` 将 prepare 和 run 分开，冻结源码并登记配置后才比赛；`verify_holdout_ledger.py` 可核对公开账本，并可选核对本地Replay。仅缓存编译代码的Worker每场每位玩家重建Bot命名空间，历史Worker文件保持不变。

[第二轮多对手参数搜索](../ROUND2.zh-CN.md) 用 `optimize_round2.py` 分开 prepare / train / test 三阶段；192场开发赛后冻结唯一候选，再跑288场独立测试，额外2场复现。`verify_round2.py` 从脱敏账本复算开发目标、入选试验、双方余额、配对收益与bootstrap区间，可用 `--replays` 核对本地482份回放。新模型 `models/tuned_round2.py` 通过运行验收但未通过同时优于原版及上一轮候选的总统计门槛，不更换默认Bot。
