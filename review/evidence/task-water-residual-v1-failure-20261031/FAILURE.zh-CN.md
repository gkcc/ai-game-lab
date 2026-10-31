# 浇水残差 GPU 探针：内存门停止

事前登记提交 `a16306856fbc621a597025565d71ad2aeee62343` 并推送后，仅发起一次受控尝试。Windows 启动前内存门通过；WSL 启动期间，外层收据记录 `memory_gate` 并写入取消标记。实际最低内存采样值未保存，不能据此推断具体数值、原因或责任进程。

本机没有 supervisor 启动标记、worker 启动标记、Linux PIDFD 清理收据、worker 结果或残差模型；supervisor 日志只报告通用失败，没有保存内部异常。按登记源码，supervisor 先创建启动标记才会派发 worker；在现场标记未被外部删除的前提下，证据支持本次没有启动训练 worker。不能把这次尝试说成 GPU 更新或模型训练成功。

host 的原始结果为 `cleanup_unconfirmed`：它在发出取消后约 30 秒仍没有拿到 Linux 清理收据。最外层本次启动进程随后退出码为 1；事后的一次只读观察显示 WSL 没有运行中的发行版、也没有匹配本任务的 Windows transport，但这些后续观察不补成当时的 PIDFD／Linux 会话清理证明，不改写原始状态。没有主动结束其他进程或关闭整个 WSL。

登记源码和训练／开发／基线私有快照仍与绑定一致，已完成来源数据封存可读；原始失败现场保留。此次没有产生新模型、新完整比赛、PPO 或默认 Bot 修改；按事前登记不重试或降低内存门槛。事前固定的开发筛查没有执行，不能判断新策略的离线或竞技表现。

[事前登记](../task-water-residual-v1-20261031/REGISTRATION.zh-CN.md) | [脱敏证据](certificate.json) | [独立复核](REVIEW.zh-CN.md)
