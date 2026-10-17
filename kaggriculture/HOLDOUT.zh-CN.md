# 冻结候选：扩展独立评估

日期：2026-10-18。没有训练、调参、修改策略或更换默认 Bot，没有 PPO。

## 先登记，后比赛

登记提交：`c1d63f31f4b51bc9b94831aed6b4b5d925f26273`，先于本轮任何比赛推送。
候选固定为上轮 champion，SHA256 `2c61c29c146f128f00d3d10dbf6a848e9e3024c5d06d41a984c33960bb2a7cec`。
12 个新种子，第一版/第二版/按需购种版/市场版四个冻结对手，双方位置互换；原版与候选各 96 场，共 192 场。另加 2 场复现检查，不计入统计。
种子、源码哈希和全部标准见 [registration.json](../review/evidence/holdout-20261018/registration.json)。新种子数值区间与历史面板不重合。

预定统计门槛：整体及市场版各自的配对净收益 95% 种子聚类 bootstrap 区间下界均大于零，并且整体及市场版的平均比赛得分差均不小于零。两项收益条件必须同时满足；得分条件只是面板内点估计约束，不构成未知对手上的非劣证明。
所有官方完整性、冻结哈希与复现检查也必须通过。比赛得分区间仅作描述，不用退化区间证明泛化。

## 完整比赛结果

以下收益均为最终现金减初始 3000；双方收益都来自同一场官方比赛。

| 对手 | 策略 | 胜/平/负 | 自方平均净收益 | 对方平均净收益 |
|---|---|---:|---:|---:|
| v1 | baseline | 2/20/2 | 35052.92 | 35052.92 |
| v1 | candidate | 16/0/8 | 35104.42 | 35130.83 |
| v2 | baseline | 24/0/0 | 44576.88 | 27933.50 |
| v2 | candidate | 24/0/0 | 44817.00 | 27236.83 |
| jit | baseline | 22/0/2 | 34572.29 | 33884.75 |
| jit | candidate | 16/0/8 | 34824.46 | 34280.46 |
| market | baseline | 8/0/16 | 37099.00 | 37958.50 |
| market | candidate | 10/0/14 | 37080.54 | 37172.79 |

| 配对面板 | 平均净收益差 | 95% 种子聚类区间 | 平均比赛得分差 |
|---|---:|---|---:|
| 整体 | +131.33 | [-341.21, 574.63] | +0.0000 |
| 市场版 | -18.46 | [-648.42, 498.04] | +0.0833 |

统计按完整种子为单位重采样 20,000 次，同一种子的所有对手和双方位置一起采样。有效独立种子数为 12，不能把 192 场视为 192 个独立样本。

**预登记统计门槛：未通过。** 官方完整比赛、源文件保留、双席位复现和 194 份本地回放核对均通过。
候选整体平均收益略高，但置信区间跨零；整体平均比赛得分与原版相同。对按需购种版，原版 22 胜 2 负，候选 16 胜 8 负；对市场版候选虽多赢 2 场，配对平均净收益却略低。不能把胜场数增加当作全面提升，也不能以本轮区间跨零断言策略完全等效或必然更差。
该判断仅适用于本次固定对手面板。没有新增第三方强 Bot，也没有证明对未知强对手或公开排行榜泛化；市场版是已知行为不同的挑战者，不称其为已证实的普遍强对手。

## 审计与隐私

[逐场账本](../review/evidence/holdout-20261018/ledger.json) 列出双方余额、奖励、终局、Bot/对手/引擎哈希及回放摘要；[源码映射](../review/evidence/holdout-20261018/source-map.json) 将登记哈希映射到公开文件。
全部 194 份 Replay、双方动作审计和日志，以及 4 份 HTML 样例留在本地运行目录。公开库没有上传原始回放、机器配置、账号、绝对路径或执行日志。
Replay gzip 哈希绑定保存的具体文件；semantic 哈希只剔除 remainingOverageTime，另用于确定性复现。公开账本可核对统计和哈希引用，但外部没有原始轨迹，不能声称已独立验证每次状态转移；也不能仅凭先登记提交证明所有历史上从未人工复测。
新 Worker 只缓存编译代码，每场每位玩家执行到独立命名空间。已用有状态计数器验证隔离；再将首种子候选对市场版以相反席位顺序逐场重跑，两场语义哈希和余额均与并行面板一致。这不是对任意有外部文件状态的 Bot 的沙箱保证。

## 如何复核和复现

从仓库根目录，公开账本核对只需标准库：

```powershell
.\kaggriculture\.venv\Scripts\python.exe -B .\kaggriculture\learning\verify_holdout_ledger.py .\review\evidence\holdout-20261018
```

重新跑比赛需要先完成游戏和 learning 环境安装，然后分两步执行；prepare 生成登记文件，核对后再 run。输出目录必须不存在。

```powershell
.\kaggriculture\learning\.venv\Scripts\python.exe -B .\kaggriculture\learning\evaluate_holdout.py prepare --output .\kaggriculture\artifacts\learning\holdout-reproduction --candidate .\kaggriculture\models\tuned_v1.py
.\kaggriculture\learning\.venv\Scripts\python.exe -B .\kaggriculture\learning\evaluate_holdout.py run --output .\kaggriculture\artifacts\learning\holdout-reproduction
```

这会复用本轮种子，属于复现，不是新的未见留出集。`--replays` 可让账本核对工具检查本地运行目录中的全部回放。
正式评估仅用 CPU，不涉及新的 GPU 运算；此前 GPU BC 探针的边界不变。暂不进入 PPO，也不自动晋级默认 Bot。

## 唯一推荐下一步

将市场版纳入开发对手池，开展第二轮小预算、面向多对手的参数搜索，并另行预登记全新留出集；本轮公开种子以后只能作开发回归或复现，不能再作为未见测试。先验证能否修复跨对手退步，再考虑 BC 或 PPO。
