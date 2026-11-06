# 未表达原地收获的四点短窗探测

本包是离线实验入口，不是新 Bot。它在四个已消费来源中各取一个首个合格的原地收获机会，先核对事实臂 48 步，再只覆盖干预臂一个单位的一次提交请求。原模型权重、候选生成器、策略、默认 Bot 和历史结果均保留。

完整登记见 [REGISTRATION.zh-CN.md](../../../review/evidence/task-harvest-unexpressed-v1-20261106/REGISTRATION.zh-CN.md)。运行只允许干净的登记提交；不会重试、补抽或读取未消费种子库存。

## 数据与控制流程

- `selection.py`：仅流式读取四份已有 trace，逐行复用旧输入验证并读完 719 行至 EOF。窗口为 240–648，按回合、单位编号顺序取首点。单位首 callback 时可用、站在成熟且有产量的作物上，本单位本地收获未进入当回合任何候选，且最终请求没有任何单位收获该地块。没有 callback 是未知，不是没有机会；没有背包容量门。
- `adapter.py`：重新核对完整当回合 trace 的资格，保序复制完整原请求，只把目标单位改成 `['HARVEST']`。其他单位与市场订单的内容、数量和顺序必须相同。
- `core.py`：两臂分别新建冻结学生和对手，从合法观测恢复全部策略前缀。事实臂请求、完整状态及库存顺序对齐历史后才允许干预。目标回合用既有 `ExecutionTrace` 包住同一次官方 step，保存实际 requested/effective/执行状态，退出恢复原 hook；不额外复演。
- `worker.py`：复用旧受控 IO、EOF 关闭器、源码资产校验及模型工厂。私有后缀事件完整保存 `original_request`（原策略请求）、`request`（实际提交）、原始 `trace`、单独的 `patch` 和 `target_execution`。前缀在核心已逐步核对之后压缩为摘要与预算，不把摘要说成完整原件。
- `runner.py`：从 Git 认证的旧 harvest 协调器字节在独立 `ModuleType` 执行，只覆写显式入口、路径、合同、来源、marker、prepare 和 runtime 接口；不修改共享历史模块全局。沿用原 Windows Job、内存门、单次许可、失败闭合与验收。根级输出 schema 改为本阶段名称，通用进程 owner 协议沿用原验证器。

准备只绑定 trace 内存在的单位请求，不声称 trace 含完整市场订单。事实续行另从官方 Replay 验证完整原请求。元数据路径直接使用既有完成 32 场的来源读取器，不经过可能读取未消费库存的入口；认证需要读冻结权重的文件字节核哈希，但不构造模型、不解码 Replay、不调用官方引擎。

执行消费者额外流式读旧 trace 至 EOF，并重读本轮事件到 EOF：核对全程原意图与原请求、原生共享时钟、一次且仅一次覆盖、所有其他单位和市场原序、两臂完整目标原请求/稳定 trace/预算相同，以及实际三端点的现金与保序库存。端点由事件保存的官方帧重算。消费者不重新调用官方转移，也不把 HARVEST 请求自动当作成功执行。官方目标单位记录的直接资源变化与整回合端点分别保存。

## 运行与停止

在仓库根目录使用已有游戏虚拟环境，准备入口为：

```text
python -I -B kaggriculture/learning/task_harvest_unexpressed_v1/runner.py --prepare
```

完成独立复审、脱敏检查和登记提交后，在该干净提交执行：

```text
python -I -B kaggriculture/learning/task_harvest_unexpressed_v1/runner.py --preflight --registration-commit <登记SHA>
python -I -B kaggriculture/learning/task_harvest_unexpressed_v1/runner.py --run --registration-commit <登记SHA>
```

必须确认前一入口退出码为 0，才能进入下一步。若四源都无点，不启动执行。运行目录一旦封存，不重用命令重跑。真实执行结果尚待本阶段实际验收；合成回归只验证接口和错误路径，不能证明官方 48 步恢复成功。

最多四点、八臂、384 次官方动作转移，零完整新比赛、零新种子、零训练和 GPU 计算。启动及案间要求 3 次 6 GiB 可用内存采样，运行底线 4 GiB；每案 900 秒、RSS 2 GiB、commit 3 GiB，阶段两小时；每案输出 128 MiB（含保留的 1 MiB 结案记录），总输出 512 MiB。执行使用旧受控写入器；准备元数据则使用有界读取和合作式内存/时间检查，不能称为相同的进程级监控。

## 判读与回归

报告窗口自身和对手现金变化、相对现金差增量、请求/物理变化、库存及指令状态。48 步现金不等于终局净收益，库存不按预测价格折现；四个已消费角色不构成独立测试。缺失不补零，技术失败不输出部分均值作为完整结果。

`minimum_label_signal` 是沿用旧合同的字段名，本轮只表示进一步研究正负短窗结果的信号门。即便通过，也不自动生成正式标签、训练或晋级；全正但未过门仍如实报告，不能解读成机制无价值。

纯内存回归为 `test_selection.py`、`test_adapter.py`、`test_core.py`、`test_runner.py`。`test_worker.py` 只在外部提供的 `AI_GAME_TEST_SCRATCH` 受管临时目录运行：覆盖官方原生共享时钟布局、side 1/hand 提交、市场和其他单位额外修改、原意图/提交区分、EOF 失败、端点账目与顺序、压缩输出预算。启动器须关闭自有子进程并实际清理该目录。所有夹具均为合成数据，不是替代游戏 Server。
