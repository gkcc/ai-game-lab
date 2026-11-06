# 未表达原地收获：初始化失败，尚无收益结论

本轮实际执行了冻结实验，结果是技术失败。没有完成任何官方 48 步窗口，没有比较收益，也没有证明提前收获有益或有害。

登记提交为 `7bdf591cbdb4070be69f7a82d43e8202dadb661c`。准备在四份已消费来源中各找到一个合格点，完整检查 2,876 行 trace；准备和运行前检查均退出 0。登记与全部策略、模型、规则保持原样，运行之后未调整目标或重试。

| 阶段或指标 | 实际结果 |
| --- | --- |
| 事前选定来源／目标 | 4／4 |
| 实际开始的案例 | F01 一案 |
| 通过的案例 | 0 |
| F02、F03、F04 | 未开始，不计为零收益 |
| 模型构造／前向／controller 调用 | 均为 0 |
| 环境构造／官方转移／请求覆盖 | 均为 0 |
| 执行阶段 trace 扫描／有效事件 | 均为 0 |
| F01 压缩事件文件 | 20 字节，可正常解压为空流 |
| 历史 Replay 解码 | 固定控制流证明完成 1 次，非独立计数器实测 |
| 子进程退出码 | 1 |
| 自有进程及 Job 回收 | launcher_reaped、job_members_zero 均为 true |
| 经济增量与信号门 | 未测量，保留 null |

F01 的 case 记录为 `io` 错误，父运行器因子进程未成功而以 `ValueError` 封存。`meter.calls` 为空、所有模型预算为零、没有 trace 扫描。事件文件是在历史 Replay 解码和源身份检查之后创建的，所以不能将执行阶段的全部数据读取成本说成零；上述一次解码明确属于控制流推断。没有模型或官方调用的结论来自计量与失败位置共同核验。

## 已复现的根因

新 Worker 先进入 `owned_writes`，再调用默认 runtime 的 `core.run_pair`。这条真实冷启动路径发生了：

```text
task_harvest_window_v1/core.py:80  official_runtime
task_student_continuation_v1/official_step.py:16  模块导入
decision/verify_mechanisms.py:18  open(os.devnull, "w")
task_anchor_once_v1/storage.py:248 / 215  拒绝保护目录之外的写入
```

空输出设备也属于保护目录之外的写入目标，现有写入限制正确抛出 `UnsafeWriteError`。这发生在创建第一个 Agent 之前。该错误与磁盘字节配额不是同一口径：`refused=0` 只说明没有触发字节配额拒绝，不能排除路径拒绝。没有内存、时间或输出配额耗尽证据。

维护者随后在全新 `-I -B` 进程和独立受管临时目录中，只调用 `core.official_runtime()`，在相同写入保护下实际复现上述异常和调用链。探针没有调用 `run_pair`、创建模型、解码 Replay、创建环境或执行游戏；错误发生时官方 package 尚未导入。探针文件和空父容器已实际删除。这是初始化故障复现，不是对 F01 的第二次实验。

35 项准备前回归确实通过，但 Worker 合成回归替换了真实 runtime 工厂，因而遗漏了这一条冷启动导入路径。不能将先前的回归通过、准备完成或进程回收，表述为官方续行已经验收。

## 证据和边界

保留本轮 18 份原件，共 147,029 字节；未生成成功 result、成功执行 certificate 或 accepted 记录。公开 `certificate.json` 仅为失败档案及诊断的核验证明，`experiment_passed=false`。本地原件集合的盐化承诺为：

`e8882e77576eb946eece64444cb00fb2cc3800b47be8fcb67a3ace56e6b6d627`

承诺由私有 binding 盐、NUL 分隔域 `task-harvest-unexpressed-failure-v1` 和 `json.dumps(witness['files'], sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')` 计算 SHA256；witness 留在集合之外。原始回放、事件、目标输入、实际种子、路径和原件摘要不上传。

454 项登记源码在封存前逐字节核对，默认 Bot 未变；前轮覆盖分析十份原件及其承诺另经重哈保持一致。发布仅追加本轮材料及 PROGRESS，不覆盖旧结论。没有重哈更早的 127／550 份大档案，不能把本轮核验范围夸大为全部历史原件复审。

下一步只修真实初始化路径，另立技术修复复现登记，原样承接四个冻结点。新目录不能冒充新独立测试，也不能用这次技术失败评价收获的经济价值。本轮没有训练、GPU 计算、新种子、完整新比赛或策略晋级。
