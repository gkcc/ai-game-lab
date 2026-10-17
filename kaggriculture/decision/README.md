# 学习型 Agent 决策基础

先读[官方机制覆盖审计](MECHANISMS.zh-CN.md)，再读[本轮实证与后续路线](EXPERIMENT.zh-CN.md)。本模块不导入、复制或替换官方Server。

## 数据流与接口

```mermaid
flowchart LR
    A[官方单玩家部署观测] --> B[observe 白名单与保序]
    B --> C[View: 全地图/单位/资源/市场/时间/公开历史]
    C --> D[Policy.decide]
    D --> E[Joint: farmer + ordered hands + ordered market]
    C --> F[孤立前置条件与联合约束提示]
    F -.建议，不能当全联合硬mask.-> D
    E --> G[wire 格式检查与原序输出]
    G --> H[官方引擎执行与现金奖励]
```

`api.py` 是轻量标准库接口，不引入新的训练框架。规则策略、参数策略、BC或未来RL只需实现 `decide(view) -> Joint`。`act()`负责观测隔离与序列化；它不会调用教师来覆盖模型选择。`FrozenAdapter`只是本轮对照用的一种Policy，不是所有模型必须经过的控制器。

### 状态

`View.encode()` schema_version=1，包含双方全部公开地块与单位、己方所有官方物品/种子/背包、价格与有符号市场库存、完整城镇列表、时间与游戏配置。原始格子字段保留，可按空间token/稀疏表分头编码；不把模型固定为原43维作物向量。

背包和仓库键顺序保留，并另输出顺序列表，避免JSON消费者排序后丢掉溢出语义。总量、年龄等派生特征可按需计算，不要求现在全部送入神经网络。

不接收Environment或Replay；不包含另一玩家的private、env.info、真实随机种子、未来轨迹。剩余运行时宽限不作为经济/训练状态；以后部署推理预算管理需另用原合法输入中的计时字段。`previous=View`只存更早的同玩家已清理视图；净市场变化不是对手交易真值。对手仓储和种子信念当前明确为unknown，尚未实现数值估计。

该表示是当前固定版本的结构性覆盖，不是声称已经学会经济动力学。未知新版本字段被排除；升级规则必须更新审计、白名单和执行用例。

[固定规则常量目录](rules-catalog.json)提供全部作物/动物、商品价格参数、商店需求和地价等先验；策略可以读取，未强迫它们成为每一轮网络输入。没有把动态经济公式复制为训练用Server。

### 动作与宏观决策

完整单位类别为 NORTH/SOUTH/EAST/WEST/PASS/DROP/PICKUP/PLACE/PLANT/WATER/HARVEST/FERTILIZE/DIG/BUILD_COOP/BUILD_PASTURE/FEED/COLLECT_FERTILIZER/CARE。市场类别为 BUY_SEED/BUY_PRODUCT/BUY_ANIMAL/SELL/HIRE/BUY_LAND。

`Atom`保存类别、物品和数量；`Joint`保存主农夫、任意长度有序工人列表及有序市场队列。数量主域是任意正整数，兼容官方int()接受的数值/字符串并**保留原载荷**；本轮修复了冻结Bot输出2.0被过度严格拒绝的问题。格式错误会明确抛错，不静默用教师动作替代。

`wire()`不按旧策略限制工人数、不排序市场订单，也不裁掉超额hands（它们影响官方种子预检）或订单（由官方按配置忽略）。正式模型通常按真实单位数输出；诊断接口仍能表达官方边界行为。

宏观动作与原语必须分开：

| 模型可学习的偏好/目标 | 确定性执行层可以负责 | 尚待实现 |
|---|---|---|
| 选择作物、动物类型、经营地块、买地/雇工时机 | 按官方格式输出、识别可达位置、保持单位/订单顺序 | 动物/施肥长期任务规划与可切换宏目标调度 |
| 选择哪个单位服务哪个任务、是否施肥/照料 | 指定目标后的导航、确认携带物与已知资源 | 面向多机制的联合资源账本和自回归上下文 |
| 选择交易商品、数量、订单顺序、库存风险 | 明确无效参数检查、可确定预算记录 | 对手交易不确定性与价格条件预算 |

未来宏观解码器应允许任意目标组合，而不是将“选作物”永远编译成旧Bot的固定任务表。策略可以直接输出完整Joint绕过宏任务库。当前唯一宏目标示例是首回合买NE；其他类别已具备原语表达与官方转移验证，但**不声称已经有成熟经营宏策略**。

### 合法性与预算

`assess_unit`返回当前孤立动作前置条件，已对18类原语正反例与官方实际效果核对；`assess_joint`报告全队种子超额、超额hands/订单和顺序依赖；`assess_order`只否定确知的非法商品/已购完土地。它们是诊断，不会重写策略。

当前尚无完整联合精确mask。不能把 `isolated unavailable` 对后续单位永久屏蔽：同格先DIG再PLANT就是反例。未知未来报价也不能用当前价伪装精确预算。上PPO之前需要逐步增加可证明正确的自回归资源约束，或允许官方no-op并将失败原因与原采样动作一起记录，不能把修补后的动作当原策略动作计算logprob。

## 运行验证

在游戏项目目录执行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m decision.verify_mechanisms --output artifacts/mechanism-foundation/contracts-local.json
```

固定版本引擎/源码必须匹配。输出JSON包含29个测试名及结果；正向、no-op、顺序、隐私验证均走官方解释器。完整运行的结果、恢复记录与复现命令见实证报告。

多文件本地Policy用新入口启动一局，并自动做官方逐步复演（游戏目录内）：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m decision.play --bot models/land_probe/land.py --opponent bots/heuristic_v1_market.py --seed 6774936 --output artifacts/mechanism-foundation/one-match
```

输出目录须不存在，生成Replay gzip、日志及审计JSON。这里只执行可信本地代码。旧Play.ps1保持原状；官方裸文件加载器不提供__file__，而本地多文件模型依赖旁边的baseline.py，所以该模型使用decision.play或已登记Worker。没有改官方加载器，也没有声称多文件包已经通过Kaggle提交验收。
