# 收获短窗信号探测（尚未运行真实探测）

此包包含固定阶段约束、历史择点、双分支续行、私有证据 Worker 与注册运行器。当前尚未进行实际准备筛点、真实模型续行或官方转移。具体冻结与停止条件见[本轮登记](../../../review/evidence/task-harvest-window-v1-20261106/REGISTRATION.zh-CN.md)；登记及源码提交完成前不得运行正式探测。

`runner --prepare` 认证已完成的 32 场来源及冻结资产，再扫描固定四个已消费角色各自的一份旧 trace，在预定 240–648 回合窗口取第一个合格机会，并把坐标封存。它不读取预留种子库存、不解码旧 Replay、不运行模型或官方比赛。没有机会就保留缺失，不补点。随后 `--preflight` 只验门禁；正式执行由 `--registration-commit` 绑定干净冻结提交。

阶段最多四个点，每点两臂、每臂 48 步，共不超过 384 次官方转移。所有点处理完且技术验收通过后，`contract.project` 才汇总短窗现金信号；无目标点不算零收益。本轮不训练、不跑完整比赛、不生成新种子，也不替换默认 Bot。

`core.run_pair(replay, trace_factory, *, target, actor_factory, rival_factory, runtime=None, meter, guard, emit, horizon=48)` 接受调用者认证的完整 720 帧 Replay。`trace_factory()` 是上下文管理器，每次提供从第 0 回合开始的历史 trace 字典迭代器。`target` 包含 `side/step/ordinal/input_sha256/original_choice/harvest_choice`，不含私有随机种子或收益。

每臂均构造独立原 `BoundedAgent` 和冻结对手，从第 0 帧重新计算此前合法观测上的请求与稳定 trace。不会直接设置 `previous` 或历史预算。只有前缀通过，才复制原始完整 configuration、info 与双方状态构造官方环境；占位 steps 仅维持框架步号，不运行环境前缀，不使用丢失 info 的 clone。

先运行事实分支的 48 步：必须由重建后的双方策略重新计算请求，并逐帧匹配旧 Replay、背包顺序和 shed 顺序。事实分支全部通过后才开始干预分支；目标 callback 仍调用原网络一次并正常计费，只替换一次既有 HARVEST 候选，然后按新分支观测继续原策略。原 teacher_label 独立保留，不将 HARVEST 伪装成教师标签。

HARVEST 机会检查当前单位与地块相同、同单位原选择、已出现的 HARVEST 候选与正库存产量。官方 HARVEST 没有背包容量限制，本包不添加该伪约束；作物首次产出日等择点资格由登记的纯 metadata 选择器核对并绑定输入摘要。这是研究采样范围，不是永久合法动作屏蔽。

`emit` 是可信离线私有证据入口。Worker 的 `events.jsonl.gz` 对前缀保存请求摘要、保序原 trace 摘要、去推理时延的稳定 trace 摘要及计数；后缀保存实际请求、trace、双方完整帧，包含作物生长、种子和未售库存。`case.local.json` 保存身份、闭合证据与端点现金。策略只获得自己的官方观测，接触不到离线 info 或另一方 private。此隔离是接口约束，不是恶意同进程代码沙箱。

`worker.verify_case` 只读验收证据及原 trace，不重新加载模型、解码旧 Replay 或执行官方转移。每臂旧 trace 正常结束时必须读至第 719 行后的 EOF，完成读取器的后置哈希检查；只读验收另做一次有界 trace 扫描。父运行器单独核对所有者、许可、Job 回收和逐项验收链。

`Meter` 区分构造/调用尝试与返回，累计原 episode 预算，并分别列出前缀和后缀成本。全阶段最多 8 臂、384 次官方 step 调用；原每臂 4000 task、8000 crop、160000 输入行和 719 controller 上限不变。计费前向不代表失败的网络一定成功返回。失败时调用者仍持有 meter，必须保存不完整收据，不重试或换点。

返回结果包含实际双方请求摘要、排除请求字段后的物理状态摘要与逐步差异，而不是仅报告候选编号变化。48 步现金、未售库存及信号门由上层用实际帧计算；本包不估价库存，也不声称终局收益。

当前验收仅为合成回归：核心假时钟检验恢复顺序、计费、隔离、错误停止和摘要；Worker 用合成来源验证真实 gzip/限额写入、EOF 收据和只读消费，临时文件由允许的 `scratch_directory` 管理并实际清理。Worker 测试入口需由该临时上下文提供 `AI_GAME_TEST_SCRATCH`，未提供时明确跳过文件测试。它们不能替代官方规则验收；尚未加载真实权重或执行真实官方 step，48 步恢复仍须登记后实测。
