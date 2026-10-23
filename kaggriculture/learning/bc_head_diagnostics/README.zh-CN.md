# 固定模型零更新诊断

本模块只计算原学生输出层的局部损失、解析梯度和固定方向曲率。不调用优化器，不改变真实参数，不运行比赛，不导出检查点。合成回归中的有限差分只作用于人工小矩阵。

**数值修正：最终消费 `results-corrected.json`。** 初始 `experiment.py`、`math.py` 与 `results.json` 是已冻结的原始记录，保留用于核验。独立复算发现初算曲率的权重分母使用float32求和；追加的 `curvature_repair.direction_stats` 统一float64，不修改初始证据。后续调用该修正版函数。修正仅涉及76个加权目标曲率，方向导数、符号及其余字段均严格不变。

从项目根目录使用已有 Python 环境：

```powershell
python -B -m learning.bc_head_diagnostics.regression
python -B -m learning.bc_head_diagnostics.experiment prepare --out artifacts/bc-head-diagnostics-v1-20261023 --public <公开工作树>
```

登记与源码必须本地提交，记录 `pre-analysis-commit.json` 后才能执行：

```powershell
python -B -m learning.bc_head_diagnostics.experiment analyze --out artifacts/bc-head-diagnostics-v1-20261023
```

之后执行修复回归和登记，提交修复源码与登记并记录 `pre-repair-commit.json`，再重算曲率：

```powershell
python -B -m learning.bc_head_diagnostics.curvature_repair regression
python -B -m learning.bc_head_diagnostics.curvature_repair prepare --out artifacts/bc-head-diagnostics-v1-20261023
python -B -m learning.bc_head_diagnostics.curvature_repair apply --out artifacts/bc-head-diagnostics-v1-20261023
```

各结果文件均拒绝覆盖。完整复现同时保留初始结果与修正证书；修正阶段绑定原结果、原冻结、修复代码和登记SHA，不重新选模型或改变分组。

分析固定父模型，使用部署float64隐藏表示；这不是重建上轮GPU FP32的720次更新轨迹。把偏置拼入设计矩阵只影响浮点累加次序，须与部署逐行预测完全相同，logits在1e-12绝对/相对容差内一致。原训练float32类别权重转为float64使用。

对于样本权重a、带偏置隐藏向量H、类别概率p和方向d：

- 梯度为 `H.T @ (a*(p-onehot)) / sum(a)`。
- 方向导数为梯度与d的内积。
- 曲率为各样本 `H@d` 在类别概率p下方差的加权平均。
- Adam方向是零一阶/二阶矩条件下、偏置修正后的理想首步 `-lr*g/(abs(g)+eps)`。保持原lr0.0003和eps1e-8，不进行学习率扫描。
- 原首轮36批分别在**同一个原模型**计算。没有权重或动量传递，不得称为一次epoch实测。

只使用旧训练、开发验证数据。旧探针的52次纠正用于12种输入的近邻描述，不加入训练或验证损失。原正式测试逐场结果不参与本模块。最近邻池是训练集唯一原始float32输入，距离为64维隐藏激活的欧氏距离；另报原归一化289维输入距离，二者不混合。按距离再按输入SHA确定同距结果，无事后阈值。精确隐藏碰撞统一正负零；没有异标签碰撞不能证明线性可分。

输出仅含聚合量、特征SHA和距离，不输出原始x、隐藏向量、逐步私有状态、梯度数组或检查点。历史源码、数据、模型及结果由SHA核查保留。公开代码不会自动下载私有本地输入；完整实证复算需要本地保留数据。

局部下降方向不等于实际训练或游戏收益改善；曲率不是执行一步后的损失。该阶段最多指导下一次另行登记的小实验，不能把开发数据结论称为独立泛化，也不能因原一次失败就宣布需要解冻隐藏层、扩网或PPO。
