# F01 首事实步诊断

本阶段只定位登记 `02f8ef2f9691e5451dcf247f8714797a04d6dcc5` 中 F01 的失败。上轮已返回一次官方 step，但没有成功保存首个 suffix 事件，不能据此判断具体哪项检查失败，更不能称为官方续行或收获收益验收通过。

本包不修改冻结 core、Worker、策略、模型、默认 Bot 或官方规则。只继承原 F01，最多重建 240 步策略前缀并调用目标回合策略一次，执行 **一次事实官方转移**。不覆盖请求、不进入干预、不运行第二步或其他点。

登记见 [REGISTRATION.zh-CN.md](../../../review/evidence/task-harvest-first-step-diagnostic-v1-20261106/REGISTRATION.zh-CN.md)。新目标是诊断位置与有限上下文，不是再次开展四点经济实验。

## 入口与继承

`runner.py` 在独立命名空间复用冻结认证、bootstrap、Windows Job、三采样内存门、存储及关闭函数。单案调度只允许 F01；不会调用旧四案例/48 步统计和现金投影。历史模块的共享全局变量不变。

准备先核对 469 项旧注册来源及 PROGRESS 字节前缀、已发布失败材料，再按固定白名单重哈 v2 的 18 份失败原件（187,746 字节）和外置盐化承诺。确认旧 closed 是 failed、accepted_cases 为 0，case 保留首次官方调用已返回但 core 未完成的事实。

准备仅认证完成 32 场的登记和收据，再真实核对模型/源码资产及默认 Bot。只从已认证旧 binding 原样复制 F01 source/selection，不调用选择器，不打开历史 Replay 或 trace。旧失败事件仅按字节核哈希，不解码为历史游戏输入。正式 preflight/run 继续验证 32 场压缩文件身份。

## 诊断和停止

新 Worker 保留原 case 的 `passed=false`，将诊断字段放进原 1 MiB 结案预留：白名单静态错误编号、注册源码相对位置，以及最多 384 KiB 的相关上下文。原始异常参数、任意局部变量和私人绝对路径不写进公开信息。

可留存的上下文仅用于离线核验：目标原请求和提交请求、实际目标 tracer 记录、实际 after-frame、一个对应历史 expected frame及其保序摘要和有限差异路径。它们不进入 Agent 输入；不因此增加第二次 Replay 解码或官方转移。

诊断只有两种合格结果：

- `located_original_error`：确实到达原目标后的失败位置，静态来源和有限上下文足以核验。
- `first_step_complete`：首个 suffix 通过并成功保存，随即受控停止；仅表示本次未复现，不能抹去旧失败。

定位前失败、资源失败、捕获缺失、预算或来源不符都封存为诊断失败。首个 suffix 完成后也不得进入第二步。无自动重试。

子进程只有在只读消费者验收诊断证据后才退出 0；这里的 0 表示诊断收集完成，原 case 仍是失败。父进程重新核验并生成专门的 diagnostic accepted/result/certificate：`accepted_pairs=0`、`full_horizon_verified=false`、`economic_value_evaluated=false`。不调用旧完整配对消费者冒充成功。

## 预算与运行

最多 1 个案例、1 个事实臂、241 次 controller、241 次对手调用、1 次环境恢复、1 次官方转移、0 次请求覆盖。原 Agent 内部单臂模型限额不变，外层硬限缩至本次诊断。零新种子、零完整比赛、零训练和 GPU 计算。

启动和案前保持三次 6 GiB 可用内存采样，间隔 10 秒；运行底线 4 GiB，每案 900 秒、RSS 2 GiB、commit 3 GiB，总阶段两小时。每案受控输出硬限 128 MiB（含 1 MiB 结案），stage 总量在既有检查点和验收时也必须不超过 128 MiB，私有小记录 16 MiB。父进程在 stage 创建前预留输出空间；已拥有该 stage 的子进程核对实际总量，不重复预留整案 128 MiB。准备另有 900 秒合作式检查。

在仓库根目录使用既有游戏虚拟环境：

```text
python -I -B kaggriculture/learning/task_harvest_first_step_diagnostic_v1/runner.py --prepare
```

独立复审、脱敏检查和登记提交完成后，在该干净提交依次运行，确认前一步退出 0 再继续：

```text
python -I -B kaggriculture/learning/task_harvest_first_step_diagnostic_v1/runner.py --preflight --registration-commit <登记SHA>
python -I -B kaggriculture/learning/task_harvest_first_step_diagnostic_v1/runner.py --run --registration-commit <登记SHA>
```

## 回归范围

`test_runner.py` 用实际隔离 `--help`、stdlib 合同快照、阻私有读的 metadata 导入和内存档案验证原 F01 继承。它还调用实际 child 调度分支（来源、文件和模型操作为内存替身），验证已存在 stage 不重复预留整案空间、只消费 F01、原失败与诊断验收分开，以及资源异常不转成功。

Worker 的针对性回归另检验硬调用上限、静态位置/有限上下文、原错误保留和首 suffix 受控停止。文件类夹具只能在外部提供的受管 `AI_GAME_TEST_SCRATCH` 内运行并清理。合成检查不能证明实际 F01 失败原因；真实一次诊断封存后再决定是否修复，不能直接继续 BC、DAgger 或 PPO。
