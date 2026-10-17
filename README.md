# AI Game Lab

Kaggriculture / Lux AI S3 的本地竞赛与自动学习研究。此仓库是经过筛选和脱敏的公开审查副本。

先读 [研发进展与结果](review/PROGRESS.md)，再读 [审查入口](review/WEBCHAT_REVIEW.md)。机器配置、账号配置、原始日志、原始回放、虚拟环境和第三方完整仓库不公开；结果由匿名汇总和源码版本记录支撑。

## 当前进度

- Kaggriculture：官方完整对局链路、中文回放、三版启发式交叉赛完成。默认 `main.py` 保留原第一版。
- 市场调度版：36场18胜18负，全部甜瓜报价未触底1；对第二版仍全败，不能称为全面升级。
- 自动参数搜索：8个候选、64场训练，随后144场独立对照。自动候选72胜；原版48胜24平，其中24平是原版镜像自战。配对平均净收益增加752.04，种子聚类95%区间585.54～964.28。
- 新增冻结候选独立评估：先公开登记，再用12个新种子加入市场版，双方位置共192场；另2场复现。全部完整运行通过，**统计提升门槛未通过**：整体配对收益差+131.33，95%区间[-341.21,574.63]；对市场版-18.46，区间[-648.42,498.04]。默认Bot不变。[完整结果与脱敏逐场账本](kaggriculture/HOLDOUT.zh-CN.md)。
- 第二轮多对手参数搜索：8候选、192场开发赛，选择记录先于288场独立测试发布。新候选78胜18负，相对原版配对收益+571.89、区间[339.82,870.23]；相对上一轮候选+209.50、区间跨零，且对市场版少赢2场，**总验收门槛未通过**。[第二轮结果与复现](kaggriculture/ROUND2.zh-CN.md)。原版、上一轮候选和默认Bot均保留。
- GPU：真实对局样本的小型行为克隆训练探针完成，但未接入正式Bot。自动参赛候选来自CPU参数搜索，官方Python仿真仍在CPU上运行。
- Lux：官方Starter CPU/GPU完整链路曾验收。公开版的可配置GPU安装适配尚未在全新机器上重新验收。

## 从仓库根目录开始

需要 Git、Python 3.12、PowerShell。两个游戏和优化器分别使用独立环境。

```powershell
.\kaggriculture\Setup.ps1
.\kaggriculture\Play.ps1 -Seed 42
# 导出的自动候选：不改变默认Bot
.\kaggriculture\Play.ps1 -Bot models/tuned_v1.py -Opponent bots/heuristic_v1.py -Seed 42
# 重跑自动实验；同配置重跑属于复现
.\kaggriculture\Setup-Learning.ps1
.\kaggriculture\Optimize.ps1
```

安装会获取 [upstreams.json](upstreams.json) 中固定Commit的官方源码，用于核验规则；不会修改官方规则。

Lux：运行 `.\lux-s3\Setup.ps1`，再运行 `.\lux-s3\Play.ps1`。GPU需要独立配置WSL Python环境，见 [Lux说明](lux-s3/README.md)。这些脚本不安装系统驱动。

## 审查资料

- [实验配置](review/evidence/experiment-config.json)、[逐次搜索结果](review/evidence/trials.csv)、[配对独立测试](review/evidence/paired-test-results.json)、[汇总](review/evidence/automatic-summary.json)。
- [市场版结果](review/evidence/market-summary.json)、[三版交叉赛](review/evidence/three-bot-crossplay.json)。
- [新评估预登记](review/evidence/holdout-20261018/registration.json)、[192场匿名账本](review/evidence/holdout-20261018/ledger.json)、[统计结果](review/evidence/holdout-20261018/summary.json)、[回放核验记录](review/evidence/holdout-20261018/replay-verification.json)。
- [第二轮预登记](review/evidence/round2-20261018/registration.json)、[开发账本](review/evidence/round2-20261018/development-ledger.json)、[测试账本](review/evidence/round2-20261018/test-ledger.json)、[冻结选择](review/evidence/round2-20261018/selection.json)、[复算与回放核验](review/evidence/round2-20261018/verification.json)。
- [学习方法与两个公开实现的研究](kaggriculture/learning/RESEARCH.zh-CN.md)、[中文游戏规则与接口](kaggriculture/中文文档/README.md)。
- [公开边界与脱敏说明](PUBLICATION.md)、[来源与公开版适配](review/source-provenance.json)、[第三方许可说明](THIRD_PARTY_NOTICES.md)。

原始Replay保留在私有工作区，新运行也会生成本地Replay；Git忽略全部运行产物。公开汇总不包含完整状态轨迹，不能单凭它独立验证每个状态转移。
