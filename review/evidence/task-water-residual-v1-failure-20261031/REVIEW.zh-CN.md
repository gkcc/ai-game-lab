# 独立复核：GPU 探针停止现场

两名独立子代理只读核对了登记、来源绑定、私有快照和失败标记。host 与取消标记均记录 `memory_gate`，host 原始结果为 `cleanup_unconfirmed`；supervisor stderr 只有一条通用失败记录，没有内部异常类别。文件相对时间与取消后等待约 30 秒仍未确认、随后 supervisor 输出失败相符；这不能证明内存门导致了具体的 supervisor 异常。

两名复核者均未发现 supervisor-attempt、worker-attempt、worker 结果、残差模型或 Linux PIDFD 清理收据。在现场未被外部改动的前提下，冻结代码要求先创建 supervisor-attempt 才能派发 worker，故结果支持 worker 尚未启动；仍不能用缺失文件独立证明过去每个进程行为。事后一次 Windows 进程快照及 WSL 列表只描述当时状态，不能替代原清理收据。

本次未重新启动 WSL、GPU、训练或比赛，没有修改原始现场。登记数据和基线快照身份仍匹配；后续不得将该次失败改称通过或原样重试。复核提出的改进方向是未来单独记录预启动失败类别、最终 transport 回收和内存门采样口径；任何新尝试仍需自己的冻结登记与资源门槛。
