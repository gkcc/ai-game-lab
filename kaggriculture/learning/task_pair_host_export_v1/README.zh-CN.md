# 两臂结果到主机内存的桥接

`copy_pair_to_host` 接受现有 `PairResult` 和调用方指定的 JAX 设备。它先拒绝键数不为六的映射，再核对固定键集合、373→128→64→1 形状、float32、每片数组的单一指定设备、结果字段以及总计 512 KiB 的两臂输出拷贝上限；随后在任何传输前同步全部设备数组，复制成两份独立 NumPy 字典，复核形状、dtype、有限值及两臂不完全相同。该上限约束结果数组字节，不是整个进程的峰值内存上限。字典可以传给现有 `snapshot.export_pair`，不在这里写文件。

此接口无法认证 `PairResult` 的来源或证明训练步数与设备身份；调用方仍须认证训练与初始检查点、实际设备、来源和登记，核对生产实验规定的 64 步及传给导出器的来源信息，并负责进程监管、实际文件保存和 `TaskNetwork` 文件重载。主机数组包含私有权重，不得打印或随意发布；同步与检查不构成设备或文件并发锁。

合成回归使用 CPU 中生成的玩具权重，通过既有导出器生成并验证内存字节；不读取真实数据、真实模型，不使用 GPU 或进行比赛。可设置进程内 `XLA_FLAGS=--xla_force_host_platform_device_count=2` 检查错误 CPU 设备；第二设备不可用时该项会跳过：

```text
python -B -I kaggriculture/learning/task_pair_host_export_v1/test_bridge.py
```
