# 教师单标签等价性 V2：固定单状态与协议合同

本轮在已验收的合成协议小修上继续，基线提交为 `1bd2ca626400a597dafb4df342c8f84689af328f`。保留旧 158 份冻结源码、默认 Bot、模型、历史失败与证书。新建 probe、check、experiment、包入口和本合同五个文件；不复制生产 source，不修改官方规则或教师策略。

## 唯一研究问题

固定原训练组 `collect-31317742-v1-0-teacher`，seed 31317742、v1 对手、席位 0、第 15 回合、零基第一个任务回调。它是历史训练来源，不是独立留出样本。旧失败停在前置测试，还没有消费该来源；V2 是新的登记和执行，不是改写或重跑旧 OUT。

先绑定实际保存的训练输入、单标签和原 Replay。原教师与纯教师适配器都必须重现该回合的完整保存动作。然后按基线候选原顺序，逐个替换首回调选择，后缀用全新教师和观测副本重新计算。最多 64 个替代，首次完整动作等价即停；不更换状态或种子。

完整比较包括 farmer、按单位编号排列的 hands，以及有序 market 数组。不同 JSON 类型不相等，数组顺序不丢失。只有单位动作相同而市场不同，单独计数，不当作正例。穷尽后无匹配才是局部负结果；候选过多触及 64 次上限、目标缺失或索引扫描上限分别报告不确定。任何身份、计数、预算或执行故障保留失败，均不重试。

## 数据之前必须通过的七项检查

调用冻结的 `task_bc_teacher_label_protocol_repair_v1.protocol.protocol_checks(deadline, ledger)` 一次，实际执行零 drain、正 drain、中途读取异常三个场景；再调用旧 `task_bc_teacher_label_equivalence_v1.probe.protocol_checks(deadline)` 一次，执行原四项动作／类型／顺序／单次干预检查。旧 source.protocol_checks 不调用。成功总数恰好 **3＋4＝7**，成功合成候选计数 6；不得伪造 2／6 来适配旧核对器。

新配置 `new_synthetic_limits` 与登记计划一致：源协议三个场景、旧 probe 四项、合成 raw 读取最多 131,072 字节、payload 最多 1,024 字节、成功 JSON 解码 2 次。合成数据与真实四个源文件的消费完全分开。

`primary.protocol_checks` 的 schema 为 `teacher-label-equivalence-protocol-v2`，字段为：

- `passed`、`cases`、`synthetic_candidates`、`cases_known_complete`。
- `source`：`started`、`completed`、`result`、`ledger`、`gate_receipt`。
- `probe`：`started`、`completed`、`result`、`partial_consumption_unknown`。

source.ledger 在调用前附着到最终会保存的对象上。成功 result 是修复函数原三字段 `schema/passed/scenario_count`，完整逐项计量仍在 ledger。gate_receipt 是已验收合成父消费者对这份新账本的纯检查，不是另一次执行或独立重现。独立 V2 核对器另行重加各场景和逐源计量，不接受一个 passed 布尔作为全部证据。

probe.result 才包含旧四项检查的结果，不能按平铺结构读取。成功时两部分 started/completed 都为 true，probe.partial_consumption_unknown 为 false，cases_known_complete 与 cases 都为 7。若源协议未返回，已产生的 ledger 仍保存；若旧四项函数开始后异常，其部分进度设为未知，不能预填四项完成或零消费。未整体通过时 cases/synthetic_candidates 为 null，已确证的完成数另列。禁止在七项协议及预算门禁通过前打开真实来源。

修复函数故障场景的预期 OSError、最终完整性 AssertionError 和三个 null 未知量必须保留，且例外只适用于该预定合成场景。它们不能混入生产 source 的失败未知量，也不能放行其他意外故障。两个正常合成前缀的缓冲未知量同样保留。

## 原数据、控制器与计数

生产读取直接复用旧冻结 `source.read_case`。四个原压缩文件各打开一次、各完成原 SHA 校验，共 6,586,966 字节。index 最多解码 256 行；inputs／labels 各只解码目标一条，前缀返回行和原始压缩后缀计量不遗漏。完整 Replay 实际解码的 720 状态／1,440 agent 状态包括未用的内容，也照实计数。关闭前缀 gzip 后只从同一 raw 读取器补齐压缩哈希，不解码后缀。生产缓冲未知量和失败未知量分开。

只有合法的目标玩家观测和官方配置进入教师；不把 Replay.info、对手私有内容或其他回合当作 Agent 观测。原教师 1 次、适配器基线 1 次、替代至多 64 次；每次控制器最多 10 秒、任务回调最多 16、每回调候选最多 512。任务与纯教师作物编码分开计数，不把编码当成模型前向。模型读取、哈希、加载、前向、训练、新比赛和官方转移均为零。

## 独立核对与一次执行

新 primary schema 是 `teacher-label-equivalence-primary-v2`，独立核对输出是 `teacher-label-equivalence-independent-check-v2`，父摘要／执行证书使用 v2。没有变化的私有 case 与生产 source_certificate 保留 v1 schema，不能据此把旧失败改为成功。

独立核对器只各读取本轮新 case 和 primary 一次，合计最多 2 MiB，私有 case 最多 1 MiB；不重新打开原始来源或运行教师。它验证真实保存的标签及输入绑定、原动作重现、唯一替代、后缀重新计算、完整动作等价、早停与穷尽、七项协议及分开计量。任何失败证书优先于成功标记。

连续机器预算 300 秒：主任务阶段到第 220 秒、独立核对到第 270 秒、最终收尾到第 300 秒，每个关闭预算 10 秒均在所属原阶段内。一个所属进程树 worker，单线程数值库，1 GiB 私有提交上限。执行内三次固定控制元数据读取合计最多 256 KiB；所有机器阶段产物（含登记、日志、失败和本地保护清单）最多 4 MiB。具体所有预算值以新 registration.json 为准，不现场延长或重试。

先静态实现，冻结源码与预算，独立静态预审并本地提交，再唯一运行。之后的独立人工结果审查和发布另列阶段；人工最多一次读取本轮私有 case、最多 1 MiB，原计时起点须在首次读取前保留。公开只发布脱敏代码和摘要，原观察、完整动作、case、Replay、模型、日志与本机身份留在本地。

本阶段不改标签、loss、策略或模型，不训练，不增加比赛面板，不启动 DAgger／PPO。任何局部等价结果都不能直接推出收益改善，也不能把不同条件分支的替代标签任意合并。后续任务必须再经独立建议与主代理核实。
