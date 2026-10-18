# 执行审计复审修复

本轮基于 `a938ef806334effbb14b14489741bab4f8f482b3`，只修审计与隔离运行入口。不修改官方引擎、策略、默认 Bot 或历史实验，不启动 BC。

## 三项修复

1. 官方解释器将非列表 `hands` 视为空列表。追踪器现在在读取请求时采用相同处理，保留原始输入，不再因 `null`、整数、字符串或对象提前抛错。回归通过真实 `Environment.step()` 比较追踪开关两侧的状态、返回值、原函数调用次数和输入不变性。
2. 动物 PLACE 是否属于新放养，必须同时满足执行前设施匹配且没有动物、执行后确实出现该动物。已有动物时走仓库转入路径，不能用执行后动物仍存在来判断放养。覆盖空设施、已占用设施、全部入库、部分入库和满仓，并保留“新放养忽略数量参数”的既有用例。
3. 优化器父进程通过独立游戏 Python 子进程运行 `decision.telemetry_worker`，只接收 JSON；官方引擎不会导入学习环境。子进程有超时并检查退出码，不吞掉失败。新登记会冻结该 Worker 源码。当前 Windows 隔离学习环境已验证真实完整 Replay 的遥测调用；没有宣称全新 Windows/WSL 安装均已复测。

执行标签版本升为 2；回放语义审计仍为 2，历史语义哈希仍保留为版本 1。三个版本概念不能混用。

## 验收与证据

新证据独立保存于 [audit-followup-20261018](../../review/evidence/audit-followup-20261018)（从仓库根目录访问 `review/evidence/audit-followup-20261018`）。旧 `audit-repair-20261018` 证据不覆盖。

验收包括 29 项官方机制测试、17 项审计回归、隔离环境遥测，以及全部 52 场留存 Replay 的官方逐步重执行。具体数量、结果、历史证书哈希和源码映射以该目录的 `acceptance.json`、`comparison.json`、`source-map.json` 和 `official-reexecution-v2.json` 为准。

52 场是 48 场实验面板加 2 场适配器检查和 2 场重复性检查，不是新的独立测试样本。复核只重放记录动作，不重新调用策略生成动作。每场比较官方公共与私有状态、库存顺序、终局和旧哈希。扩地收益和未通过晋级门槛的结论不变。

## 本地复核命令

先按项目安装说明建立游戏和优化器各自的虚拟环境。以下命令在 `kaggriculture` 目录运行；将 `<留存实验目录>` 和 `<单场目录>` 换成自己的本地数据目录。输出必须使用未存在的新路径。

```powershell
.\.venv\Scripts\python.exe -B -m decision.verify_audit --output artifacts/my-audit/regressions.json
.\.venv\Scripts\python.exe -B -m decision.verify_mechanisms --output artifacts/my-audit/contracts.json
.\learning\.venv\Scripts\python.exe -B learning/verify_telemetry.py <单场目录> --side 0 --output artifacts/my-audit/isolated-telemetry.json
.\.venv\Scripts\python.exe -B -m decision.reexecute_experiment <留存实验目录> --output artifacts/my-audit/official-reexecution-v2.json
```

`verify_telemetry.py` 要求在没有安装官方引擎的学习环境执行，真实调用生产遥测入口。公开仓库不携带原始 Replay；审查者可核对脱敏证书，完整重执行需要自行生成或持有本地回放。

## 仍然保留的边界

追踪器仍要求独占游戏进程；市场记录仍是订单聚合与有序效果摘要；历史隔离仍要求 runner 为每局重置会话。离线审计含权威私有状态，不能直接作为 actor 特征或上传原始标签。此次修复不是全机制合法性证明，也不是学习型 Agent 完成的证据。
