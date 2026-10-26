# 固定锚定实验的独立重接入口

上一阶段 `task_anchor_once_v1` 在 spawn 启动通信时失败，未进入训练 worker。
失败记录、消耗的一次启动名额、未知计数和原源码全部保留。随后
`task_anchor_spawn_repair_v1` 已用一次无模型的真实 WSL spawn 验证入口修复。
本目录承接独立建议和主代理裁决，获得一个**新阶段、单独预算、附条件的一次 fit**，
不是旧失败退款或原目录重跑。准备、冻结、独立预审和原字节本地提交通过前不得执行。

## 唯一实现变化

复用原版 `task_anchor_once_v1.training._worker`，通过已验收的
`task_anchor_spawn_repair_v1.launcher.launch` 启动。进程创建、序列化、管道、
等待与清理在产物写入 guard 之外；worker 的 512 MiB 写入保护保持原样。
新适配器和启动器的父收据共同占用 64 KiB，不重新获取预算。
Linux 会话监督和 Windows 传输/读取线程退出各有证据，清理异常不覆盖原始错误。
旧 worker 产物中的 anchor-once schema 名称只是方法沿革，**本次身份由新登记 SHA 绑定**。

## 科学问题与预算不变

从原始 S 开始，冻结选种网络、归一化、特征、候选顺序、合法性和控制器。
512 次 Adam，学习率 1e-4、裁剪 1；PCG64(2026102707)，每步原训练与已有
聚合训练各 128 个完整组。唯一候选损失沿用原冻结方法：

`sum_256(w*CE)/sum_256(w) + 1.0*sum_old128(w*KL(p_S||p_C))/sum_old128(w)`。

温度 1，参考 S 停止梯度，只保存最终模型。不改变系数、方向、步数、混合比例、
初始化、检查点或验收门槛。不会读取上一轮失败的 DAgger 候选作为起点。
原验证与已知开发缓存都已经影响设计，仅作已知筛查，不参与梯度、不称独立测试。
四份 CPU 初末全量 logits 和最终已知开发 GPU logits 仅保存在本地。
独立数值核验固定四个 split 各 16 个完整中点组，最多 100000 行模型呈现，
在首次评分前落盘；不看误差换组。

新对局种子沿用 PCG64(2026102718) 的前 10 个合格 uint32，重新盘点并排除
所有可见旧登记、已失败及未使用保留种子；前 2 个开发、后 8 个留出。
只按重复/库存排除，不按结果选种子；库存不覆盖已删除或不可访问历史。

先要求已知开发 CE 比率 ≤0.95、原验证 ≤1.02；CPU/GPU 的 atol=1e-4、
rtol=1e-5、首 argmax 差异为零。通过后最多 24 场新开发；再通过才运行 96 场留出。
S/C/T、round2/market、双方席位以及全部经济门槛、20000 次种子聚类 bootstrap
沿用冻结原登记。新训练局为零。最终只分别判断 S 改善、T 保留和真实动作控制；
不因 fit 完成、已知筛查通过或镜像胜利自动收录教师。

主阶段 3 小时，独立审查另限 1 小时；训练循环 600 秒，进程含清理 1200 秒。
最多 2 个比赛 worker、120 场、174717 个官方步（含主复演和预选独立复演）。
输出总额 16 GiB、空闲底线 4 GiB，所有更细计数与产物分区沿用原登记。
梯度行、参考 S 行、离线行和比赛推理分别记账，失败和在途工作不退款。
Python 写入保护不冒充 OS 配额或 native 文件系统沙箱。

## 入口与停止条件

```text
python -B -m unittest learning.task_anchor_reentry_v1.test_entry
python -B -m learning.task_anchor_reentry_v1.experiment prepare
python -B -m learning.task_anchor_reentry_v1.experiment verify
# 完成独立预审、脱敏本地登记提交与 pre-execution-commit.json 后：
python -B -m learning.task_anchor_reentry_v1.campaign run
```

测试仅使用临时合成字节、实际内存管道和假进程，不读真实模型或数值数组。
临时归属工具按既有环境配置解析，验收要求零跳过；不启用 Agent Workflow。
prepare 不解码模型或数值缓存，复制私有启动配置仅到本地产物目录。
任一入口、存储、不可变性、清理、数值或比赛门槛失败即关闭该阶段，不重试。
完整方法和原门槛可查 [冻结方法说明](../task_anchor_once_v1/README.zh-CN.md)。
原模型、默认 Bot、所有旧源码/证书/结果保持不变；公开仅发布脱敏摘要与哈希。
原始 Replay、逐步状态、特征、logits、权重、日志、设备和启动配置不上传。
