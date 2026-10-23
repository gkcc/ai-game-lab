# 联合任务 BC V1：一次有界训练与开发闭环

固定原 60 场教师示范（8 个训练种子、2 个已消费验证种子），学习每次从完整候选组中选择工人、地块和任务。选种继续使用冻结第 6 步模型；候选生成、强制运输、导航、市场、经济门控仍由冻结教师提供。真实候选不是官方完整合法动作集合。

数值缓存按决策组保存 275 维共享信息、按真实候选保存 98 维局部信息；标签与离线索引不进网络。保留所有组和候选，报告数值重复与标签冲突，不删除难例。共享归一化按训练决策组，候选归一化按训练真实候选，padding 不参加统计，one-hot/presence 保留原值。完整数值输入一致性上界不是此网络可达能力证明。

单次 MLP 373→128→64→1，组内 softmax 交叉熵；按训练标签操作的逆平方根频率裁剪加权，再归一化平均组权重为 1。Adam 0.0003、梯度范数裁剪 1，有效批 256 个完整组、每个微批最多 65536 个 padding 后候选。微批梯度使用整个有效批权重总和作分母。8 完整轮、最多 3400 次更新、30 分钟同步训练和 45 分钟训练进程上限；任何上限导致未完成即不允许比赛。

离线准入要求：多候选组精确 candidate 命中率至少 60%，较初始增加至少 10 个百分点；跨操作竞争组 macro recall 至少 70%，每类至少 50%；加权验证 CE 低于初始。只在合格完整轮中选择最低 CE，平局择早。初始与所有完整轮均保留，不因结果不佳另跑配置。全验证 GPU 与新进程 NumPy 导出 logits 容差固定 atol=1e-4、rtol=1e-5，candidate ID 必须完全一致；还必须在真实观测证明权重改变动作或资源预留，隔离离线 seed、对手私有数据和回放 info。

依次执行，入口要求已提交且经独立审查的登记。所有产物位于专属 `artifacts/task-bc-v1-20261023-run1/`，不能覆盖旧阶段：

```text
python -B -m learning.task_bc_v1.experiment prepare --out artifacts/task-bc-v1-20261023-run1 --public ../publication/ai-game-lab
python -B -m learning.task_bc_v1.experiment cache --out artifacts/task-bc-v1-20261023-run1
python -B -m learning.task_bc_v1.experiment parity --out artifacts/task-bc-v1-20261023-run1
# 缓存、组合接口独立审查通过后，在现有 WSL GPU 环境：
python -B -m learning.task_bc_v1.train --out artifacts/task-bc-v1-20261023-run1
# 训练与全验证导出审查通过后：
python -B -m learning.task_bc_v1.admission --out artifacts/task-bc-v1-20261023-run1
python -B -m learning.task_bc_v1.games --out artifacts/task-bc-v1-20261023-run1
```

最多 12 场开发比赛：原 2 个验证种子 × 固定 round2 × 双席位 × 教师任务/初始任务/训练任务三个臂，各臂固定同一选种模型。首种子先完成训练臂两席，再完成对照；若训练臂两席平均终局现金不足混合教师参考的 75%，停止第二种子。全部 12 场完成才做 2 场官方文件入口重复，重复不进统计。失败占用预算，不重跑或补种子。两种子单对手不能证明竞技泛化；混合参考不等于原教师。

不改默认 Bot、旧模型、冻结策略和历史结果，不新增示范、不做 DAgger/PPO。原始数组、权重、完整 Replay、私有逐步状态、设备信息和日志仅保留本地；公开代码、脱敏统计、哈希和独立审查证据。

