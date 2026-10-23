# 固定数据的输出层微调 V1

本阶段只检验一个问题：保持原示范、编码和隐藏层不变，能否让网络学会前一阶段定位的、已有示范覆盖的开局选种。此前的教师纠错探针不是学生；本轮候选必须由网络权重自行选择，不允许教师覆盖或回退。

## 冻结设置与停止条件

- 原训练8种子共9,127条；原验证2种子共2,356条。这批验证已用于研发诊断，只称开发验证。
- 从原学生检查点初始化，冻结 `w0/b0/w1/b1/mean/scale`、289维编码与原教师控制器，仅训练 `w2/b2`。
- 一次训练：Adam、学习率0.0003、batch256、20epoch、720次更新，随机种子2026102321；保留原类别权重和逐epoch随机遍历，不补示范、不重采样、不去重或增加教师评分输入。
- 训练用GPU最高FP32矩阵精度；选模统一用实际部署float64的未加权开发验证交叉熵。epoch0也是候选，严格改善才替换，平局保留更早模型。不能根据准确率门槛另挑epoch。
- 离线门槛：选择非epoch0；固定的160条训练草莓错例至少纠正一半；开发验证准确率下降不超过1个百分点；未在训练精确出现的1,324条验证调用准确率下降不超过2个百分点；冻结、真实GPU梯度、导出校验全部通过。
- 失败即停止，不换学习率、优化器或继续拟合，不运行比赛。后续方法需要另立阶段。

若离线通过，先在新原生进程重载核查并冻结唯一模型，再运行候选12场：沿用上一阶段两个开发种子、三个冻结对手、双席位。另有两场真实官方文件入口复现，最多14场新比赛。原学生的12场对照复用上一阶段账本，逐项绑定模型、引擎和源码身份；不冒充新对照或独立测试。原教师只用于既有背景，不追加新教师比赛。

开发推进门槛为：候选相对原学生在两个完整种子上的平均净收益均为正，整体现金差和得分不下降，对市场对手的收益、现金差和得分均不下降。只作继续研究依据，不宣称统计显著或晋级。正式新留出面板不在本阶段运行。

## 执行入口

在项目根目录，用已有隔离环境执行。`RUN` 是尚不存在的新实验目录；已有登记、模型和账本都拒绝覆盖。WSL使用现有独立GPU环境，不改Lux环境或驱动。

```text
游戏Python -B -m learning.bc_head_v1.verify_metrics
游戏Python -B -m learning.bc_head_v1.experiment prepare --out RUN --public PUBLIC
# 本地提交源码、登记和freeze后，WSL独立GPU环境运行：
python -B -m learning.bc_head_v1.fit --out RUN
# 新Windows进程，无论离线门槛正负，都验证导出：
游戏Python -B -m learning.bc_head_v1.verify_export --out RUN
# 仅离线通过后：
游戏Python -B -m learning.bc_head_v1.experiment freeze-candidate --out RUN
# 本地提交检查点哈希和候选冻结，然后：
学习Python -B -m learning.bc_head_v1.experiment games --out RUN
游戏Python -B -m learning.bc_head_v1.experiment entries --out RUN
游戏Python -B -m learning.bc_crop.audit --out RUN --stages dev entry-dev
游戏Python -B -m learning.bc_head_v1.experiment analyze --out RUN
```

复用原Worker、控制器、NumPy推理和官方保序复演器。输出层训练的GPU证据与CPU官方模拟分开报告；不把GPU可见当作训练成功，也不声称比CPU快。公开库不含原始数据、私有遥测或模型权重；原始文件保存在本地实验目录，公开哈希不能代替外部取得原始状态。
