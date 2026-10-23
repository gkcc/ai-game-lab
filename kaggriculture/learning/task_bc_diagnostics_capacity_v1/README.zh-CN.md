# 任务诊断容量与进程生命周期验证 V1

上一阶段支持分析在 900 秒截止时停止，没有完整索引和评分汇总；原失败永久保留。本阶段只验证分段处理和资源生命周期，不训练、不调用游戏策略、不执行官方状态转移、不处理全部历史或 96 场面板。

准备登记先于真实样本读取。原十个 train/validation 种子各取 v1、seat0 文件组前 128 个物理行，坏行立即失败，不跳过或补取。最多 1,280 个原始调用和 80,000 个候选，超限整次失败，不能先去重再计数。采样文件顺序继承原 manifest；训练支持仅来自其中 1,024 个 train 行，明确 `sample_only`，不代表完整训练支持。

完整样本支持先构建并提交，随后每个文件分片评分，全部使用同一已提交索引。评分复用原 float64 数值函数；float32 支持键保留单独含义。汇总按原 manifest 与行顺序，逐 split 使用原 `SummaryAccumulator`，全局重复键、冲突标签和 unique 分母不按片简单平均。逐行输出与原纯函数顺序的参考路径比较，汇总要求精确一致。

共享真实评分上限为 5,120 组、320,000 候选，覆盖准备参考、准备分片、正式容量验收和独立复核。调度前预占完整任务额度，失败或缺回执不释放。候选缓存命中也计入已评价组数；实际模型 `score` 调用及候选另列。准备计算最多 1,800 秒，正式计算最多 1,800 秒，单任务 900 秒，单 worker、单 BLAS 线程。每阶段只执行一次，不从进度文件恢复。

Windows 执行器通过进程创建时的 Job List 属性纳入受控 Job，主线程在成员检查后才恢复；Job 禁止 breakaway，关闭时终止受控后代，使用实际 ActiveProcesses 和进程句柄证明关闭。它用于可信本地 worker 的进程资源归属，不是防御恶意代码委托系统服务的沙箱。准备阶段固定四个真实 venv 回归：正常退出、合作截止、非合作子孙进程、控制器硬中断；独立的 fixture 哨兵验证没有误杀。各类一次，测试身份和完整日志只留本地。正式数值任务继续通过同一冻结执行器并验证完整退出。

在已有官方 Python 环境下，先准备新输出目录的 `preparation-registration.json` 和保护清单。该登记是公开证据中的完整配置，固定原来源、样本、预算及保留范围；不要复制后直接恢复旧失败目录。

```text
python -B -m learning.task_bc_diagnostics_capacity_v1.experiment freeze --phase preparation --out artifacts/<new-registered-stage>
python -B -m learning.task_bc_diagnostics_capacity_v1.runner run --phase preparation --out artifacts/<new-registered-stage>
# 准备通过，完成独立预审、本地精确提交门控后：
python -B -m learning.task_bc_diagnostics_capacity_v1.experiment freeze --phase formal --out artifacts/<new-registered-stage>
python -B -m learning.task_bc_diagnostics_capacity_v1.runner run --phase formal --out artifacts/<new-registered-stage>
```

准备执行前必须写入绑定准备冻结的独立 `preparation-source-review.json`。正式冻结强制沿用准备测试的同一份实现；正式执行前必须写入绑定 `formal-freeze.json` 的独立 `pre-review.json`，把该冻结、准备登记、预审及全部冻结源码按精确字节提交，再保存本地 `pre-execution-commit.json`；`ready()` 实际核对提交内容。成功也只证明小样本语义和执行生命周期，不能宣称全量吞吐已通过、旧失败已修成成功或任务学生已晋级。完整诊断需另立阶段。

结果和原始数据分离；只发布源码、白名单计数/测量、源码与来源散列、审查回执。样本输入、模型、支持 pickle、逐行评分、日志、PID/创建时间和机器身份不发布。

冻结分别保存运行文件和公开 Git blob 的原始字节 SHA；既有文件若仅 CRLF/LF 不同，显式保留两种 SHA 与 LF 对照 SHA，不改历史文件。其他内容差异拒绝；本阶段新源码要求公开与运行字节一致，正式入口核对提交中的每个实际 blob。
