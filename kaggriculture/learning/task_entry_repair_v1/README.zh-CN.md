# 联合任务 BC 官方文件入口兼容修复

这是独立历史复现阶段。只新增兼容包装器，不修改 `task_bc_v1` 的冻结代码、模型、策略、默认入口和旧失败结果。官方文件加载器未提供 `__file__`，但编译代码记录真实文件名；新包装器使用该文件名回退，末尾显式暴露 `submission_agent`。

0训练、0新种子、0新经济面板，最多2次完整比赛尝试：原31205138种子、冻结round2、双席位、原任务模型和原选种模型。任何失败停止，不补赛，不将旧阶段失败改为通过。

准备和独立预审后本地提交登记；实际回归使用官方加载器复现旧错误，并从非项目当前目录连续调用新官方入口、冻结Worker入口、直接模型，比较完整动作和轨迹。回归不执行新官方转移。随后最多两场真实 `run_match.py` 文件入口，要求与旧两场开发轨迹的双方动作、v1/v2摘要、任务/作物记录、模型身份、现金和奖励一致，并经既有官方保序复演。

使用现有官方游戏Python，依次运行：

```text
python -B -m learning.task_entry_repair_v1.experiment prepare --out artifacts/task-entry-repair-v1-20261023
# 独立预审及本地提交登记后：
python -B -m learning.task_entry_repair_v1.verify_entry --out artifacts/task-entry-repair-v1-20261023
python -B -m learning.task_entry_repair_v1.experiment run --out artifacts/task-entry-repair-v1-20261023
```

入口复现只验证部署兼容性，不增加经济统计样本，不改变上一阶段平均净收益比混合参考少155的结论。权重、日志、原始回放和逐步私有状态仅留本地。公开必要源码、聚合结果、哈希和独立审查证据。
