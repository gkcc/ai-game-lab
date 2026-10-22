# 有限选种 BC：运行与边界

该模块训练五分类小网络，只替换冻结终局施肥教师在分配 PLANT 任务后、检查种子前的选种表达式。其他经营代码共用同一份经哈希锁定的教师源码。它是混合 Agent，尚不是完整神经策略。

## 运行已经部署的学生

在 Kaggriculture 项目目录执行：

```powershell
.venv/Scripts/python.exe -B run_match.py --bot bots/bc_crop_student.py --opponent bots/heuristic_v1_market.py --seed 42
```

新比赛自动存入 `artifacts/matches/`；打开该场的 `replay.html` 查看，JSON 和结果文件在同一目录。默认 `main.py` 保持原样。学生从 `models/bc_crop_v1/checkpoint.json` 读取本地权重身份；可通过 `BC_CROP_MODEL_CARD` 指定另一张模型卡。没有检查点会直接报错，不会回退教师。

公开仓库不附带本机训练数据、检查点、逐帧遥测或原始回放。新机器需要自行生成示范和训练；公开的模型哈希用于审查本轮身份，不意味着其他依赖环境一定逐字节复现权重。

## 接管和保留的范围

`agent.py` 对固定教师 AST 仅插入选种回调和返回时遥测回调；两处数量和教师 SHA 均检查。先编码，再记录教师标签，避免答案进入输入。

保留教师的种植机会生成、作物经济拒绝门控、任务排序、施肥、及时收获、导航、雇工、采购预算、卖出、终局规则。经济拒绝是这次混合控制器的边界，不是官方合法性屏蔽。模型不能主动创建种植机会；缺种也不屏蔽作物，而是继续原来的购种路径。

编码器复用 `decision.api.observe` 的合法观测过滤，白名单提取 289 维特征，包括市场、公开农场、己方库存及前序单位的资源预留。它不使用种子元数据、对手私有库存、未来价格、最终得分、教师评分或排名。官方回放只在离线复核模块使用。

每次调用重新开始当前回合的记录，无跨局隐藏历史。缺权重、身份错误、schema 不匹配及非有限值均失败。NumPy CPU 推理使用导出的 float32 权重，以 float64 运算；平局按固定作物顺序选第一个。

## 重现本轮流程

以下 `RUN` 指一个新的实验目录；已有目录中的结果不覆盖。准备阶段需要指定公开 Git 审查副本，用于记录 HEAD、历史保护哈希和种子使用检查。新实验不能继续把本轮已披露种子称为留出测试；应另行登记新种子。

1. 用游戏环境运行 `python -B -m learning.bc_crop.verify`。用现有学习环境运行 `experiment prepare` 和 `experiment parity`，确认双方席位的完整教师轨迹一致。
2. `experiment collect-smoke`；游戏环境运行 `experiment dataset-smoke`。在独立 GPU 训练环境运行 `train --smoke`；学习环境运行 `experiment smoke`。随后采集剩余示范：`experiment collect`、`experiment dataset`。
3. 独立 GPU 环境运行 `python -B -m learning.bc_crop.train --out RUN`。`requirements-gpu.txt` 记录实际 JAX/Optax 主依赖版本。本轮因官方 PyTorch 大包传输受阻，在拟合前登记切换 JAX；独立 WSL 环境使用已有可用依赖的物理副本，没有修改 Lux 环境或驱动。
4. 新进程运行 `reload --out RUN --scratch <已拥有的临时测试目录>`，核验保存权重、框架/部署输出及失败契约。`experiment deploy` 仅创建新学生的本地模型卡，不更改默认 Bot。
5. `experiment trained-smoke` 后用游戏环境运行 `entries --source-stage trained-smoke`，验证实际文件入口。将代码、初始/训练权重、分区和统计冻结并本地提交后，才运行 `experiment test` 的 144 场。
6. 游戏环境运行 `audit --stages test`，`entries`，`audit --stages entry-test`；最后运行 `python -B -m learning.bc_crop.report --out RUN`。其他命令均附 `--out RUN`。其他示范及冒烟账本也可分别传给 `audit --stages`。`report` 为冻结统计器补充最后打印需要的标准库 `json` 名称；冻结计算和校验代码保持原字节，补充前后的两份统计文件逐字节相同。

采集沿用现有 `round2_worker.py` 和 `Games`；复核沿用 `full_audit.reexecute` 与官方执行追踪器，不提供替代仿真器。日志和 Replay 留本地。审计将真实采样、重新加载模型后的选择、完整动作、官方成功种植及购种成交逐一关联。

## 如何解释指标

“选种调用”包含采购规划、移动中的计划和经济拒绝，不等于成功种植。“请求 PLANT”和“官方成功 PLANT”分开计数。相对于教师的选择差异在模型当前资源上下文计算；不是把不同比赛的状态假装成相同状态。

工程闭环 A 要求权重确实控制动作、完整官方对局、两种入口一致、零回退和保序复演。模仿效果 B 另要求验证准确率胜过初始模型及多数类，且测试中相对初始模型的净收益聚类区间下限为正、比赛得分均值不下降。教师差距单独报告；并不要求首版学生击败教师。

统计按完整种子聚类，双方席位和全部对手一起重采样 20,000 次。最终测试只作结论，不选择模型或 epoch。本轮结束后不自动开展 DAgger、PPO、自博弈或新教师研究。
