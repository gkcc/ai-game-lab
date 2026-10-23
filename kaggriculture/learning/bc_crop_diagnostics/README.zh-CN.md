# 有限选种 BC：误差定位与单一纠错探针

此阶段保持既有模型、默认 Bot、教师与规则不变，不训练模型。目标是先判断选种误差是否可能解释经济差距，再决定是否值得补数据。独立规划与审查由子代理完成，主代理逐条核对；不冒充外部 WebChat 裁决。

## 既有轨迹诊断

`analyze_existing` 只读原 60 场训练/验证示范，以及 2 场训练种子上的学生开发对局。重建 44,578 个决策回合、11,795 次选种，检查原动作和稳定遥测一致。这个步骤没有重新执行官方环境。

这批样本没有出现相同编码对应不同教师标签；从部署特征恢复的五类教师评分与原控制器一致。这只覆盖被检查记录，不能证明所有官方状态信息无损。官方正常观测不含市场参数，冻结教师使用自己的默认价格参数；这不是读取了官方隐含参数。

验证准确率 87.35% 混合了两类调用：1,032 次的完整特征向量在训练中出现过，准确率 94.96%；另外 1,324 次为 81.42%。整种子分割仍然成立，但不能把所有验证调用称作新状态泛化。

预写规则按每个训练种子的“该类错误教师评分差总和 / 所有购种或种植表达次数”计算，再对八种子等权平均，选中草莓分层。教师评分差是启发式偏好，**不是官方奖励、现金损失或真实 regret**。该层 160 个错误仅对应 23 个完全不同的编码，其中 128 次来自在全部八种子反复出现的七个编码。“八种子都有”不等于八种不同局势的证据。连续决策链也不是独立样本。

## 开发探针

`StrawberryCorrection` 每次仍计算原模型。只有当前局面下条件教师选择草莓、模型选择其他作物时，临时覆盖为草莓。原始模型 logits、原预测、是否纠错和实际选择分别记录。`logits` 是覆盖后的选择信号，`model_logits` 才是冻结网络原输出。

这是借助教师标签的反事实诊断，**不是新学生、容错回退或可晋级策略**。除了这一个内层选种表达，其余冻结控制器保留；干预当然可能影响同回合后续资源与选择，以及后续市场和对手行为，不能承诺整局其他动作不变。

最多两个新开发种子、三个冻结对手、双席位、教师/原学生/探针，共 36 场。运行前登记代码、种子、统计、门槛并本地提交；不冒充第三方时间戳。两种子不做显著性宣称，也不根据结果增加版本或扩大预算。门槛仅用于判断是否值得继续调查，不用于教师或模型晋级。

## 本地重现

以下在 Kaggriculture 项目根目录运行。旧实验目录 `artifacts/bc-crop-v1-20261023-run1` 仅存在于实际本地部署；公开库不包含原始回放、私有遥测或权重。公开文件中的哈希不等于外部审查者已经独立取得原始状态。

```powershell
.venv/Scripts/python.exe -B -m learning.bc_crop_diagnostics.verify_probe --source artifacts/bc-crop-v1-20261023-run1 --out artifacts/bc-diagnostics-v1-20261023/probe-regression.json
.venv/Scripts/python.exe -B -m learning.bc_crop_diagnostics.panel prepare --source artifacts/bc-crop-v1-20261023-run1 --out artifacts/bc-diagnostics-v1-20261023 --public ../publication/ai-game-lab
# 完成代码、登记、冻结的本地提交后：
learning/.venv/Scripts/python.exe -B -m learning.bc_crop_diagnostics.panel games --out artifacts/bc-diagnostics-v1-20261023
.venv/Scripts/python.exe -B -m learning.bc_crop_diagnostics.panel audit --out artifacts/bc-diagnostics-v1-20261023
.venv/Scripts/python.exe -B -m learning.bc_crop_diagnostics.panel analyze --out artifacts/bc-diagnostics-v1-20261023
```

这些命令是本轮执行记录，已有目标文件时会拒绝覆盖；不要删除历史结果后重跑来制造新证据。`Games` 使用既有学习环境中的依赖启动官方比赛 Worker；Agent 及环境运行在游戏虚拟环境。比赛和诊断均为 CPU，GPU 此轮没有参与训练。

`audit` 逐场重新加载决策代码和权重，复现双方 719 次动作及稳定遥测，再调用固定官方引擎核对所有公开/私有状态、背包顺序、v1/v2 摘要和终局奖励。销量、销售收入、购种量来自官方追踪的实际成交；选择调用、种植请求、成功种植和纠错次数分开报告。禁止把分类错误数直接解释为经济损失。
