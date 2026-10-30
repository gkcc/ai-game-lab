# Linux pidfd 自有会话清理：合成阶段

本组件是独立的新守卫，不修改历史 `gpu_guard`；已接入合成阶段的新监督组件，尚未接入正式训练。它在启动前检查 `pidfd_open`、`pidfd_send_signal`、`waitid(P_PIDFD)`、`WNOWAIT`、自身 `/proc` 读取及默认 `SIGCHLD`；不具备条件即失败，不修改系统配置。

守卫在 Popen 前预留一个 CLOEXEC 描述符，创建后释放以缓解本进程 FD 耗尽；这不能保证消除全局 FD 耗尽、并发分配或其他资源错误。它先一次有界读取、按调用方独立固定的 SHA 校验门源码，再以 `-I -S -B` 启动该源码快照。门只继承私有 pipe 读端，父写端不得继承；放行前不 fork、不执行目标命令，EOF 或短超时自行退出。父侧取得并锚定 leader pidfd、核验 PID/SID/starttime 并再次采样取消后，才通过写端放行，目标以同一 PID 执行。

清理时守卫在回收 leader 前保持 pidfd 和 PID 两种 WNOWAIT 锚点，采样 `/proc` 中同 session 的成员，打开各成员 pidfd 后重新核对 PID、SID 和 starttime，使用该 pidfd 发送 TERM 或 KILL。进程消失的 `ESRCH` 可跳过并复扫；权限、扫描、锚点、pidfd 和清理超时分别失败。两次空扫描且 leader 已退出之后才回收直接子进程；不会向数值 PID 或 PGID 发信号。

若 Popen 后无法取得 leader pidfd 或核验身份，父侧先关闭门写端并有限等待门 EOF 自退；不能确认退出时报告清理未确认，不进入正式训练。若门已放行后出现扫描、权限或资源故障，未知后代仍可能存活，不能据此声称整个 session 已清理。检查 `signal.getsignal` 和线程状态也不足以证明没有原生 SA_NOCLDWAIT 或其他并发 reaper；未来入口必须通过专用进程与受信环境保证唯一 child 回收者。

`/proc` 枚举和超时都是采样，fork／退出可能与扫描交错；连续两次空扫描不是抵御恶意并发的形式化证明。调用方必须保证单一子进程所有者及回收者、认证来源并另设完整监督预算。主动 `setsid` 脱离 session、不可中断状态、系统崩溃、强杀监督器或恶意进程均不在清理保证之内；任何不确定性均应停止且标记清理未确认。

工作期限采用保守采样：首次观察到直接子进程退出时，若已达到或超过登记工作期限，结果仍记为 `deadline`；即便该进程实际上提前退出但观察较晚，也会判失败。进程尚在运行时同样采样期限，不能据此承诺硬实时终止。监管计时不包括守卫开始计时之前的源码读取、能力检查等，也不覆盖外层全部启动和收尾。

测试不读取真实数据或模型，不调用 GPU 或比赛。换组测试使用专用隔离 subprocess 和进程级临时 subreaper；合成 leader 与后代具有 EOF／限时自退，并在发起取消前验证、持有各自 pidfd。外层也通过私有 Unix socket 接收这两个已验证 pidfd；如隔离测试超时，只以这些稳定句柄兜底，并检查其退出及原 starttime 的 `/proc` 项消失。若未收到完整句柄或无法确认，测试失败并标记清理未确认，不把自退预期当作已回收证据。测试不向无关进程发送信号。

```text
python -B -I kaggriculture/learning/task_pair_pidfd_guard_v1/test_guard.py
```
