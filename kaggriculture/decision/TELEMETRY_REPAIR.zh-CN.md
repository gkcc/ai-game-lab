# 遥测消费层兼容修复与动作计数口径

基于 `abfd7405b52e9438ecc3b779dd88c4cf699eb0e3`。只修遥测消费层，不修改官方规则、策略、默认 Bot 或历史结果，不做 BC。

## 请求计数，不是成功执行数

`non_pass_unit_requests` 定义为：每步执行前实际存在的农民/工人槽位上，提交了非空列表、首项为字符串且不等于 `PASS` 的请求数量。

- 未知字符串命令、参数不足、被原子种子校验拒绝、没有产生效果的请求仍计数。它不是合法动作数或成功执行数，也不是耗费资源的次数。
- 非列表或空原子动作、非字符串操作码不计数。
- 非列表 `hands` 按官方解释器约定视为空列表；缺省 farmer 为 PASS，缺省 hands 为空列表。只改变本地读取，不改写回放。
- 超出执行前实际工人数的槽位不计数，即使其请求参与了官方的原子种子需求校验。本指标刻意统计实际单位槽位的请求，不是解释器所有内部调用次数。
- 市场订单不计入，刚在本步雇佣的工人也不计入本步已有单位。

旧字段 `non_pass_unit_actions` 保留为同值兼容别名；新增 `action_count_version=1` 和机器可读口径说明。该版本与 `audit_version=2`、`execution_label_version=2`、历史语义哈希版本 1 是不同概念。

## 验证路径

`learning/verify_telemetry_boundaries.py` 从没有官方引擎依赖的学习环境启动游戏子进程，生成 4 场完整官方对局（双方席位 × 有无工人），随后对每场的双方分别调用正式 `telemetry()`，共 8 次完整隔离遥测。没有使用成功桩，没有修改输入状态或官方引擎。

覆盖 null、整数、字符串、非空对象 hands、缺省 farmer、空/非列表原子动作、未知命令、无种子种植、缺参数 PLACE、超额工人槽位，以及零请求的对手侧。用明确安排的请求断言 3 / 5 / 0 计数，并核对原始 Replay 字节未改。真实发生效果的判断仍由官方复演与执行审计承担。

另分别进行：

1. 全部 52 场官方逐步复演，检查状态、库存顺序、最终奖励与旧哈希。
2. 全部 52 场从隔离学习环境调用生产遥测入口，每次内部再次完整官方复演；核对 48 场面板所有历史遥测指标保持不变，52 场效果摘要与原证书一致。

两条路径重放相同旧比赛，不能当作 104 场独立实验。48 面板 + 2 适配器 + 2 重复检查的统计口径不变，扩地未通过晋级门槛的历史结论不变。

证据：[telemetry-repair-20261018](../../review/evidence/telemetry-repair-20261018)。旧证书目录完整保留。具体验收以 `acceptance.json`、`comparison.json`、`boundary-regressions.json`、`isolated-panel.json` 和源码映射为准。

## 复现

在 `kaggriculture` 目录，先按项目说明建立两个独立虚拟环境。所有输出须选未存在的新路径：

```powershell
.\learning\.venv\Scripts\python.exe -B learning/verify_telemetry_boundaries.py --output artifacts/my-telemetry-boundaries
.\.venv\Scripts\python.exe -B -m decision.reexecute_experiment <留存实验目录> --output artifacts/my-recheck/official-reexecution-v2.json
.\learning\.venv\Scripts\python.exe -B learning/verify_telemetry_panel.py <留存实验目录> --certificate <旧运行证书路径> --output artifacts/my-recheck/isolated-panel.json
```

第一条命令自行生成边界回放，可独立复现；后两条需要本地留存的 52 场数据。原始回放、原始日志及本地路径不公开。当前验收针对已有 Windows 环境，不声称全新 Windows/WSL 安装均已复测。

追踪器独占进程、市场标签聚合、runner 每局 reset 以及私有离线信息不能直接进入 actor 特征的既有边界不变。本轮不开始示范采集或训练。
