# 完整机制基础：执行验证与最小实证

日期：2026-10-18。基于公开仓库初始HEAD `2253662`，初始工作区干净。官方1.33.0的规则文件与提交 `f994f2050d58d84f649329c1e35ce09ada7007c5` 逐字节匹配；没有改Server、默认Bot、两轮候选或历史实验。

## 实际交付

- [机制覆盖表](MECHANISMS.zh-CN.md)：12机制族的官方依据、资源/时间/联合约束、原Bot缺口与价值证据。
- [共享状态/动作/约束/Policy接口](README.md)：全部18个单位、6个市场原语，合法公共/己方信息和保序库存；策略可以绕过教师直接返回Joint。
- `verify_mechanisms.py`：29个自动用例，验证真实官方转移、全部原语的孤立正反例，以及一场完整运行中的双方信息隔离；不是只检验动作名字。
- `smoke_full_agent.py`：独立于作物教师的一只鹅执行程序，通过同一Policy接口完成720状态，卖出54蛋和29肥料，最终现金7150，对手PASS现金3000。不是竞技收益证据，不将它晋级策略。
- `full_audit.py` / `reexecute_experiment.py`：用记录动作回到官方引擎逐步复演，避免旧作物审计器把合法动物收获当作无效动作。动物程序719次转移一致；扩地52场共37388次转移，双方公开/私有状态、余额、最终结果逐步一致。

## 首次运行失败与透明恢复

初始登记提交 `a4a1661`。首次编码器只接受Python int，冻结Bot在资金不足时可能产生2.0等数量，官方int()本可执行，编码器却拒绝。首次24场control和2场parity已完成，24次land尝试中断。原始数据与异常日志保留，没有把它们藏成成功结果。
修订提交 `9ef19f2` 在补跑前登记：只修正数量兼容，原数值/字符串载荷保序不改，不改变策略、种子、对手、预算或门槛。复用已完成24场control与2场parity，再补24场land及2场复现。不会声称修复后的这批种子是全新未见测试。首次27用例记录与修订29用例记录都保留在公开证据目录。

## 单一扩张干预

冻结第二轮候选为经营控制器。control经过新接口但不买地；land只在step0将BUY_LAND插在市场队首，支付1000购买NE，之后控制器完全相同。没有搜索买地时点/数量，也没有改7工人上限。
6个固定种子×第一版/市场版×双方席位×control/land = 48场评价；2场裸Bot/透明对照parity和2场重复检查不进入收益统计。每对首回合的非买地动作相同，首回合现金严格相差1000，均由官方回放确认。

| 对手 | 方案 | 胜/平/负 | 自方平均净收益 | 对方平均净收益 |
|---|---|---:|---:|---:|
| v1 | control | 12/0/0 | 32346.00 | 31451.00 |
| v1 | land | 0/0/12 | 29069.67 | 46523.67 |
| market | control | 2/0/10 | 33331.00 | 36255.17 |
| market | land | 12/0/0 | 50502.83 | 40564.67 |

| 面板 | 配对自方净收益差 | 95%种子聚类bootstrap区间 | 比赛得分差 |
|---|---:|---|---:|
| 整体 | +6947.75 | [625.00,11589.25] | -0.0833 |
| v1 | -3276.33 | [-8375.33,357.17] | -1.0000 |
| market | +17171.83 | [9643.50,23111.50] | +0.8333 |

原计划的探索性支持条件是整体收益区间下界大于0，且每个对手的平均比赛得分不下降；**该门槛未通过**。总体平均自方净收益从32838.50升至39786.25，但比赛得分从0.5833降至0.5000。不能因赚得更多宣称更强。
只有6个独立种子。整体符号检验5正1负，双侧p=0.21875，bootstrap区间对如此小面板可能不稳；分组结果只作探索，不能声称普遍统计优越。summary里旧helper的门槛另存 helper_gate_not_used_for_this_experiment，不作为本轮决策。

## 真正新增了哪些能力，以及因果边界

- 回放验证了 **1274次新土地成功播种**。不是只发出BUY_LAND名称：从状态读取、策略目标、编码、官方执行到最终结算已贯通。
- 平均非PASS单位动作由2113.17增加到2653.33；mean simulation时间约2.65秒→3.06秒，仅为本轮并行运行观测，不是严格单线程性能benchmark。固定资本成本1000，净收益已扣该真实支付。
- **全部24/24对** 同种子control/land的城镇序列不同。扩地改变可长草地数，官方共用RNG也改变商店抽取；对手随价格/状态响应。该干预的真实整体效应包含这些变化，不能分离归因成纯粹的额外种植面积收益，更不能为配平而修改官方RNG。
- 旧replay_metrics是作物审计，sales_income字段没有加回买地成本，本轮不使用它作营业收入依据；净收益/现金差均直接取官方余额。完整新机制动作使用官方逐步复演审计，不能沿用crop-only合法性假设。
- 买地具有强烈对手依赖：值得作为可学习选择，却没有证据支持无条件开局扩张。动物完整程序证明可执行，不证明强于作物Bot；施肥、动物组合、择时买地、更多工人的竞技价值仍未知。

## 后续学习判断

原教师是9机制族部分覆盖、3族未建模（扩地/动物/施肥），不是全游戏专家。当前动作表示足以逐步表达本版本完整游戏，但没有完整的联合精确mask、成熟动物/施肥宏任务、数值对手信念或学习型比赛Bot。
下一步优先补**多机制示范采集、执行标签与联合资源上下文**，然后接入有限BC学生。不是继续无限补手写规则，也不是立刻上大网络/PPO。DAgger必须包含能回答新增机制的示范来源，否则仍会复制教师盲点。GPU可以加速后续网络批推理/梯度，不能加速这里的Python官方规则；本轮无新GPU训练或驱动/WSL修改。
完整分工、数据隔离、独立评估与GPU边界见[后续路线](LEARNING_ROADMAP.zh-CN.md)。

## 重跑与证据

从仓库根目录进入游戏目录：

```powershell
Push-Location kaggriculture
.\.venv\Scripts\python.exe -B -X utf8 -m decision.verify_mechanisms --output artifacts/mechanism-foundation/contracts-local.json
.\.venv\Scripts\python.exe -B -X utf8 -m decision.smoke_full_agent --output artifacts/mechanism-foundation/animal-reproduction
.\learning\.venv\Scripts\python.exe -B -X utf8 learning/mechanism_experiment.py prepare --output artifacts/mechanism-foundation/land-reproduction --baseline models/tuned_round2.py
# 先核对并冻结该目录registration.json，再执行
.\learning\.venv\Scripts\python.exe -B -X utf8 learning/mechanism_experiment.py run --output artifacts/mechanism-foundation/land-reproduction
Pop-Location
```

新目录必须不存在。修复后复现不需要旧失败目录或--reuse-from；复用的是同一已公开种子面板，只称复现/回归。原版与两个参数候选均未覆盖，默认Bot不变。

[脱敏逐场账本](../../review/evidence/mechanism-foundation-20261018/ledger.json)、[修订登记](../../review/evidence/mechanism-foundation-20261018/registration-revised.json)、[恢复记录](../../review/evidence/mechanism-foundation-20261018/recovery.json)、[统计与回放核对](../../review/evidence/mechanism-foundation-20261018/verification.json)、[官方逐步复演](../../review/evidence/mechanism-foundation-20261018/official-reexecution.json)。
全部完整回放、HTML样例、原始日志及失败记录只留本地；公开只有白名单代码/模型/规则常量、账本与汇总，不含个人机器和账号配置。官方逐步复演是本地证据，外部若没有原始轨迹仍需自行复跑，不能把公开汇总当成外部独立执行验收。
