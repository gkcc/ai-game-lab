# 保序审计、执行归因与局内历史修复

本次修复基于 `428448af454b7cc4043d06091268cb988c660b0f`，保留官方引擎、默认 Bot、所有冻结策略和历史实验。没有训练 BC、PPO 或重新挑选策略。验收数据见仓库 `review/evidence/audit-repair-20261018/`。

## 版本化保序审计

`semantics.py` 的 v2 规范化仅将 `observation.private.inventories` 下每单位的背包映射编码为带类型标记的有序键值对。普通 JSON 字段顺序不影响相等和哈希；单位列表、动作和订单的列表顺序保持原样。两版均排除 `remainingOverageTime`。

`full_audit.reexecute()` 逐步比较 v2 状态，终局再核对完整轨迹 v2 哈希。只交换一个背包的物品插入顺序会被拒绝，即使旧排序哈希仍相等。主动 DROP 与日终满仓溢出都由官方函数验证；非满仓时两种顺序可以产生相同商品结余，但审计仍保留完整状态语义。

旧 `semantic_sha256` / `semantic_replay_sha256` 保留为 v1 兼容字段，新增版本声明和 `semantic_sha256_v2` / `semantic_replay_sha256_v2`，不能把旧字段单独当保序保证。现用比赛入口及 Worker 新结果同时保存两版。历史 ledger、回放、登记、首次失败和旧审计文件均不覆盖。历史核验脚本的 v1 结果不会追溯变成 v2，新增 52 场证书才是本轮保序证据。

## 官方执行归因

`execution.py` 是特权离线观察器，不是部署编码器或新 Server。它在独占、单线程复演范围内暂时包装官方函数，每次原样调用原函数一次，转发返回值，并在退出或异常时恢复。不能在同进程混跑未受控的环境线程；并行采集应使用独立进程。

单位记录包括原始请求、官方原子校验后的实际调用、单位序号、位置、调用前后资源和该格状态。按实际效果区分 executed、partial、no_effect、blocked、pass、error；市场订单另有 ignored。no_effect 不等于协议非法，也不等于经济上无价值。只在实际扣除一粒种子且官方生成对应作物时记录一次成功播种。

因此，同格两个播种请求只归因给成功者；先 DIG 再 PLANT 可以计入成功；PLANT→DIG→PLANT 可以记录两次真实成功，不能一律按最终格子去重。旧面板核验器的按回合/玩家/格子去重只用于兼容历史遥测，不充当通用训练标签。新机制实验遥测改用官方复演的实际效果。

市场通过官方解析、逐件成交返回值及雇工/买地前后状态归因：保留订单位置、请求数量、实际成交量、失败成交次数、资金变化、首末报价及资源上下文。部分成交、无钱、仓满、非法订单与订单截断不会被整场 DONE 掩盖。成交交错顺序进入事件摘要哈希；当前订单标签是聚合，不冒称逐件成交的完整训练序列或联合合法性保障。

`reexecute_experiment --labels` 可额外保存私有 gzip JSONL 执行记录。它包含离线特权资源，**不能直接当 actor 输入，也不能上传公开仓库**。本轮不构建训练数据集，公开证据只有白名单结果、来源哈希和聚合。

## 比赛历史隔离

无历史的 `act()` 保持原行为。直接传 previous 时，必须同时传入与旧 View 相同的显式局内 object 标识，并满足同席位和更早步数；缺失、跨标识、换席、乱序和未来帧会拒绝。

运行器可使用 `HistorySession(player)`；每场开始前必须 `reset(player)`，然后调用其 `observe()`。标识由运行器管理，只在内存用于身份比较，不进入 `encode()`，不从 seed、回放或机器信息派生。合法观测自身没有可靠比赛身份，不能声称接口能识别运行器故意沿用同一标识的另一场比赛。重用进程的采集器必须承担 reset 生命周期，并使用部署同一编码器。

针对性回归还只改变对手私有仓储、种子和背包，检查己方编码及冻结策略动作不变。此项是明确的边界保护，不是所有未来特征的不干扰证明。

## 复现

在游戏项目目录运行，输出文件应使用新路径：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m decision.verify_mechanisms --output artifacts/audit-reproduction/contracts.json
.\.venv\Scripts\python.exe -B -X utf8 -m decision.verify_audit --output artifacts/audit-reproduction/regressions.json
.\.venv\Scripts\python.exe -B -X utf8 -m decision.reexecute_experiment artifacts/mechanism-foundation/20261018/land-revised --output artifacts/audit-reproduction/official-reexecution-v2.json
```

最后一条需要本地保留的原始 52 场回放；公开源码仓库不包含这些私有产物。更换结果目录不代表新测试种子，也不代表新策略实验。回放仍为原始 48 场面板 + 2 parity + 2 repeat。外部审查者可复算公开统计和运行场景回归，但没有原始回放时不能声称复演了本地全部 52 场。

修复不改变扩地不晋级的结论，也不构成学习型 Agent 验收。下一阶段仍须另行授权和验收有限 BC，不能用本次执行日志替代模型真实控制完整对局。

## 本轮验收结果

原 29 项机制测试和新增 15 项回归通过；最终源码下 52/52 场、37,388 次官方状态转移逐步一致，含双方背包顺序。旧哈希、双方终局现金及历史结果保留。按官方原语实际效果重新归因的新土地播种仍为 1,274 次；没有发现本批历史计数偏差。公开源码副本也通过 15 项回归及完整比赛入口复演；使用既有隔离引擎运行时，不宣称全新安装验收。默认 Bot 原入口完整比赛亦通过。扩地不晋级结论不变。
