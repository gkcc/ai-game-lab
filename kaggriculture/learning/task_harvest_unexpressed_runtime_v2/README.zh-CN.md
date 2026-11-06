# 同四个冻结点的运行时修复复现

本包承接 `7bdf591cbdb4070be69f7a82d43e8202dadb661c` 的四个已消费目标。旧实验在第一个案例初始化 runtime 时失败，没有模型前向或官方转移。新目录是技术修复复现，不是新样本、新独立测试，也不把旧失败改记成功。

完整登记见 [REGISTRATION.zh-CN.md](../../../review/evidence/task-harvest-unexpressed-runtime-v2-20261106/REGISTRATION.zh-CN.md)。旧策略、模型、默认 Bot、官方规则、四点位置、48 步窗口、现金信号门和所有预算均不变。

## 修复范围

冻结 Worker 原来在 `owned_writes` 保护内惰性导入官方 runtime。其依赖初始化会打开空设备，因越过受保护目录而被正确拒绝。新 Worker 在认证资产后、进入事件输出保护之前准备同一真实 runtime 和既有 `ExecutionTrace` factory，再通过现有参数传给冻结 `core.run_pair`。

准备仍受原 Job、内存和时间门约束，不提前构造模型或环境，不增加允许任意路径写入的例外。目标 tracer 包住的仍是原两臂各自目标回合的一次官方 step，不增加转移或复演。运行时准备收据与原 case 结案共用预留字节限额；原 `original_request`、实际提交 `request`、`trace`、单次 `patch` 和执行消费者协议保持不变。

`runner.py` 在独立 `ModuleType` 中加载 Git 已认证的 v1 wrapper 与原 Windows Job 协调器，显式替换新路径、marker、绑定读取、prepare 和 runtime 接缝。子进程启动新入口；不修改已导入历史模块的全局变量，不重建监督框架。

## 准备与身份边界

`--prepare` 只做以下工作：

1. 核对旧登记 454 项来源、旧 PROGRESS 字节前缀、失败发布五份文档和新增源码。
2. 按固定白名单重哈旧失败 18 份小原件及外置 witness，验证公开盐化承诺、`closed.failed`、`accepted_cases=0`，保留原件。
3. 用 `sources.load(verify_game_files=False)` 认证已完成 32 场的冻结登记、收据和任务身份；另外核对模型、源码资产和默认 Bot 的实际文件哈希。
4. 从旧 binding 原样复制四个 source/selection，保持嵌套 JSON 顺序和每份 selection 的原字节。生成新的盐、源码身份与独立绑定，记录旧失败来源。

该入口不调用选择器，不读取历史 Replay/trace，不解码 Replay，不构造模型或环境。旧失败的 20 字节空事件文件属于 18 份故障原件，可以按字节重哈，不能将它说成一份新轨迹。正式 `--preflight` 和 `--run` 仍会核对完成来源的压缩 Replay/trace 文件身份；准备阶段不能声称重新扫描了这些游戏文件。

## 执行

在仓库根目录、既有游戏虚拟环境中执行：

```text
python -I -B kaggriculture/learning/task_harvest_unexpressed_runtime_v2/runner.py --prepare
```

真实冷进程检查、合成回归和独立复审通过，脱敏并提交登记后，在该干净提交运行：

```text
python -I -B kaggriculture/learning/task_harvest_unexpressed_runtime_v2/runner.py --preflight --registration-commit <登记SHA>
python -I -B kaggriculture/learning/task_harvest_unexpressed_runtime_v2/runner.py --run --registration-commit <登记SHA>
```

必须确认前一步退出 0 才进入下一步。新目录仅允许一次执行；失败封存，不自动重试。每个点先完成并核验事实 48 步，再做单次单位请求覆盖后的 48 步。上限八臂、384 次官方动作转移、零完整比赛、零新种子、零训练和 GPU 计算。

启动和案间三次 6 GiB 可用内存采样，运行底线 4 GiB；每案 900 秒、RSS 2 GiB、commit 3 GiB，阶段两小时。每案输出 128 MiB（保留 1 MiB 结案），四案 512 MiB，私有小记录 16 MiB。准备另有 900 秒合作式检查；它与完整执行进程监督的范围不同。

## 验证与结论限制

本包的入口测试使用实际隔离 `-I -B --help` 与合同快照，并通过内存中的 18 文件夹具验证失败承诺、收据身份、原样继承和不重新择点。真实冷进程测试由 Worker 的针对性回归负责：实际准备 runtime/tracer、在写保护内装卸真实 tracer，以及确认空设备或越界写仍被拒绝；不运行比赛或模型。

这些检查不能替代真实官方连续 48 步验收。完成后仍只报告四个已消费点的截断窗口现金、库存和执行事实，不能推断终局改善或统计显著性。`minimum_label_signal` 沿用原现金门，仅用于进一步研究正负短窗结果；无论是否通过，都不自动生成正式标签、启动 BC/DAgger/PPO 或晋级策略。
