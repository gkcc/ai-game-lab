# 已消费面板的共同前缀诊断

这是一次探索性历史诊断，不是新的留出测试。面板结果已公开、研究者已看过经济汇总；这里固定选择角色 `S01`，不按收益挑选场次。范围为三个既有对手、两个席位、`merged`、`unweighted` 和 `teacher_task` 三臂，共 18 条已有轨迹和 12 个配对。`teacher_task` 仍使用共同作物模型。

只比较同一配对完整历史的共同前缀及首次完整请求分叉的当回合。完整请求包含 `farmer`、`hands` 和保序 `market`；订单列表顺序有意义。v2 状态语义摘要排除计时、保留背包物品顺序，不代表原始文件逐字节一致。对手先分叉、双方同时分叉或状态先不一致均不能把后果归因于本方。每回合的任务选择记录只比较到第一次输入或选择不同；当该回合的完整请求相同且下一回合仍处于共同前缀时，才可重新开始比较。零选择记录、输入不同和记录条数不同单列。`teacher_task` 产生规则选择记录而不调用任务网络；记录配对数不是两侧神经网络调用数，程序单独统计两种来源和零回调回合。

条件教师标签只是当时的参考选择，并非成功、最优或收益标签；同输入标签一致后才比较双方一致情况。两个策略所选工人的请求相同与否不能代替完整请求比较。完整请求差异、未来状态/现金差异以及最终收益差异都不构成单次任务选择的因果证明。一个种子及高度相关的回合不能支持显著性或泛化结论。

程序复用原有只读 Replay 压缩读取、合法观测适配、trace 校验和比较纯函数，核对原登记、脱敏账本、既有验收收据和原始压缩文件摘要。原 Replay 及双方私有状态只在本机用于离线核验，不送入模型或公开。此前的主复演和独立收据仍是同一官方引擎的历史证据；本次不新增比赛、模型调用、训练、GPU 运算或官方状态转移，也不追认任何旧失败实验。

输入为本机已验收面板目录、其本地完成状态及公开脱敏账本。先创建本机私有快照，记录当前收据和 18 对预期输入摘要；运行时重查快照并在读取时验证压缩文件。此快照在事后形成，不能独自证明原始比赛验收当时的文件身份。程序在本机私有内存验证所有 `S01` 行的种子与冻结种子文档一致，以及每条 Replay 的 `info.seed` 与对应行一致，不输出种子值。

累计读取预算为 512 MiB 压缩字节和 16 GiB 解压字节。900 秒截止在读取、处理及返回附近采样，并非操作系统强制的硬截止；一次阻塞操作可能超时后才被发现。解压字节预算不是进程内存限制，需由运行者监测内存。输出写入全新本机目录，不覆盖历史；结果先写入可保留以供恢复的 `result.pending.json`，成功关闭、读回一致后才发布为 `result.json`。只有进程退出码为 0、存在完整结果且无失败记录时才视为成功。缺文件、身份不匹配、trace 未到 EOF、预算耗尽或校验失败须停止并保留失败信息；不能利用部分结果选择策略。

在仓库根目录，把下面的占位符替换为本机路径。私有快照和输出必须在公开仓库与原面板目录之外：

```text
python -B -I kaggriculture/learning/task_prefix_diagnostic_v1/runner.py --panel PANEL_DIR --status LOCAL_STATUS_JSON --manifest PRIVATE_MANIFEST_JSON --prepare-only
python -B -I kaggriculture/learning/task_prefix_diagnostic_v1/runner.py --panel PANEL_DIR --status LOCAL_STATUS_JSON --manifest PRIVATE_MANIFEST_JSON --output NEW_LOCAL_OUTPUT_DIR
python -B -I kaggriculture/learning/task_prefix_diagnostic_v1/test_pairing.py
python -B -I kaggriculture/learning/task_prefix_diagnostic_v1/test_runner.py
```

成功后只发布脱敏配对摘要和结论，不发布本机快照、原始 Replay、trace、真实种子、盐、私有状态、机器路径或原始输入摘要。
