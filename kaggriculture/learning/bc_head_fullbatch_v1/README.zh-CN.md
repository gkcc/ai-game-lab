# 全批次输出层 V1：一次训练、最多20次更新

依据已完成的固定模型诊断，只尝试一个全批次方案。保持原模型前两层、编码、归一化、9,127条训练示范、类别权重、Adam及学习率0.0003，仅更新w2/b2。原训练是20轮×36小批次＝720更新；本轮20轮×1全批次＝20更新，两者数据遍历次数相同、优化器更新次数不同。不声称严格只隔离了一个优化器因素。

从epoch0到20每步均保存完整输出头、11,483行预测与SHA、全部开发指标、训练/验证各类别混淆、每个种子、既定160错例、训练精确未见验证子集及草莓减甜瓜margin。正margin不代表在5类中选草莓。头和逐行预测仅本地保存；不复制21份固定隐藏层。

选模始终取部署float64未加权开发验证CE的严格最小值，平局取早。留存其他epoch不是事后换选模规则。离线必须选非0、固定160错误≤80、验证准确率下降≤1个百分点、训练未见验证子集下降≤2个百分点，且真实GPU更新、冻结数组及导出通过。

离线失败即停止，不增加步数、换学习率或优化器补考。通过才冻结候选，沿用两个开发种子、三冻结对手、双席位12场，另加2场官方文件入口复现。复用原12场学生作冻结对照；每个完整种子净收益增量为正、总体及市场对手现金差/得分不降。均是开发证据，不是独立显著性或晋级证据。所有比赛调用官方引擎，720状态/719动作/DONE、零fallback、保序复演。

运行顺序（从项目根目录，训练用既有独立GPU环境）：

```powershell
python -B -m learning.bc_head_fullbatch_v1.verify_metrics
python -B -m learning.bc_head_fullbatch_v1.experiment prepare --out artifacts/bc-head-fullbatch-v1-20261023 --public <公开工作树>
# 独立预审，通过后本地提交登记，写入pre-training-commit.json；不是外部时间戳。
python -B -m learning.bc_head_fullbatch_v1.fit --out artifacts/bc-head-fullbatch-v1-20261023
# 新Windows进程重载全部21个头，再核对选模；不重新训练。
python -B -m learning.bc_head_fullbatch_v1.verify_export --out artifacts/bc-head-fullbatch-v1-20261023
```

仅离线通过时依次执行本模块experiment的 `freeze-candidate`、`games`、`entries`；运行已有 `learning.bc_crop.audit` 的dev/entry-dev官方复演后，执行 `analyze`。实验门禁和比赛实现复用冻结 `bc_head_v1`，新注册身份由本模块额外验证；不得绕过门禁。原模型、默认Bot、历史数据及旧结果不修改。不得启动旧正式测试或预留192场，不做DAgger/PPO。
