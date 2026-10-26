# 标签实验前置协议小修：仅合成数据

上一版在内存 gzip 测试停止，没有进入原始训练样本。它的 `raw_drain_bytes_returned > 0` 要求依赖缓冲预读量，而且异常后没有外传该测试的局部计数。旧失败、旧源码和复审计时例外全部保留。

本阶段只建立替代的前置协议函数 `protocol_checks(deadline, ledger)`。调用者从开始就持有账本；正常或异常退出都能取得已发生的部分工作。复用冻结的旧 `_Meter`、`_CompressedReader`、前缀解码和关闭逻辑，不修改生产读取器、教师、标签、模型或损失函数，也不把它接入旧失败目录重跑。

恰好三个预定合成场景：受控缓冲保证全部压缩原文提前读完（允许零 drain）；限制单次底层读取量，保证仍有余量而完成正 drain；第二次底层读取注入异常，保留首读返回字节、失败未知量和关闭记录，并检查随后完整性断言拒绝时账本仍可见。前两者只解码一行 JSON，不解码后缀；最后一例不应被计作完整哈希或成功解码。此处 raw 打开与字节均为内存对象，真实原始文件读取为零。

只有三个场景全部满足稳定结果与资源不变量，且父入口、所属进程树、源码和历史保护均通过，才接受“前置协议修复通过”。父进程还从新的合成收据检查计数；这不是独立重现。独立子代理另审源码和收据，不访问原始样本或模型。

连续执行上限 60 秒；单 worker 阶段上限 40 秒，其中所属进程树关闭预算 10 秒，工作截止为第 30 秒，与复用 runner 的原有关闭上限一致。内存流各自的关闭检查上限仍为 5 秒，包含在工作截止内。所属进程树内存上限 256 MiB。合成压缩读取合计最多 128 KiB、返回 payload 最多 1 KiB、JSON 解码恰好 2 次，整个阶段产物最多 2 MiB。执行只有一次，先登记、冻结、独立静态预审及本地提交，再运行；失败即保留，不重试。代码生成和静态审查不冒充实际回归通过。

```powershell
.venv/Scripts/python.exe -B -m learning.task_bc_teacher_label_protocol_repair_v1.experiment prepare --out artifacts/task-bc-teacher-label-protocol-repair-v1-20261026
# 完成独立静态预审及精确本地提交后，仅一次：
.venv/Scripts/python.exe -B -m learning.task_bc_teacher_label_protocol_repair_v1.experiment run --out artifacts/task-bc-teacher-label-protocol-repair-v1-20261026
```

本阶段不打开原样本、Replay 或模型，不调用教师、网络、官方状态转移，不训练、不打新比赛。历史任务模型评分仍是 3,921 组／228,505 候选，原模型比赛准入仍失败。通过本阶段也不会产生训练标签等价性或收益结论；继续真实单状态问题需要另一个冻结版本与新登记。
