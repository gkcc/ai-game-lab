# 收获决策的信息与候选表达

核对基于项目提交 `596c2767342e8311b1272a417c2a83782c29423a` 和固定官方引擎 `f994f2050d58d84f649329c1e35ce09ada7007c5`。这是在已知上一轮四点结果后、本轮事件消费者首次扫描前固定的源码映射；不宣称未见结果的独立特征选择，也不代表模型已学会使用它们。

下文 O 指官方 `kaggle_environments/envs/kaggriculture/kaggriculture.py`，F 指本项目 `learning/task_choice_v1/features.py`，C 指 `learning/bc_crop/features.py`，T 指 `bots/teacher_fertilizer_terminal_v2.py`。项目文件均在 `kaggriculture/` 下。

当前 `TaskNetwork` 实际读取 **275 个共享特征与 98 个候选特征**（`learning/task_bc_v1/model.py:51–55,71–90`）。完整合法 observation、configuration、planned 和顺序数组虽传给 selector（F:110–119），但该网络仅取 `shared_x` 和候选 `x`。保留原始结构不等于网络已经读取全部信息。

|需要的信息|合法来源与官方依据|现有数值编码／缺口|
|---|---|---|
|当前作物与可收数量|公开地块 crop/yield_units，O:215–225,446–468|`target_tile.crop.MELON`、`target_tile.yield_units`、presence 与脚下对应字段已编码；F:37–47,98–105。|
|年龄和生命周期|公开 planted_day/max_lifespan_step、day/step，O:11–16,224,752–766|原始字段及时间已编码，可合法派生年龄和剩余时间；ongoing、max_yield、first/max_yield_day 及预期边际产量未单列。真实未来产量不能作为部署输入。|
|浇水、肥效、枯萎风险|公开 watered_today/consecutive_unwatered/fertilized_until_day，O:432–443,475–481,777–787|目标／脚下数值及双方作物 dry/fertilized 汇总已编码，F:37–47,C:56–57。一次 WATER 的即时边际量可据固定规则合法计算，当前未单列。|
|背包与仓储|本方 private.inventories/shed 和 shedCapacity；O:169–174,299–300,343–355,843–857|carried_unit/all、shed、projected_shed 与仓容已编码（C:10–11,25–26,60–64,F:84–95）。官方没有背包容量门；真实容量限制在仓库。插入顺序只在结构化数组中，网络未读（F:118–119）。|
|运输与剩余兑现时间|公开位置、地图、时间；日终自动入仓与单位重置 O:843–882，单位先市场后 O:935–946|单位／目标位置、到目标曼哈顿距离与剩余回合已编码（F:84–105,C:25–38）；收获→入仓→出售时间未单列，全体单位位置和逐单位预分配明细仅结构化保留；动作类别计数已由 `planning.prior_actions.*` 编码（C:37–38）。派生成本仍是估计，不保证未来成交。|
|当前市场压力|公开 market.inventory/prices/params；O:192–212,544–560,590–628|五种作物库存、价格及曲线参数列存在（C:65–78）；默认市场不带 params 时参数列是 0/false 填充值，未展开官方默认常量（O:178–184）。不能把填充值当真实曲线参数。|
|需求与未来供给代理|公开已解锁商店，O:730–749,886–891；本方种子、库存、作物与自己计划|商店种类计数、townShopSellInterval 已编码；townCenterSellInterval、townShopUnlockInterval 仅结构化保留（C:10–11,39–40）。planned_supply、采购预留已编码（C:66–71,T:129–141,251–259），是教师估计，不是官方未来产量。|
|对手公开竞争状态|公开对手现金、作物、地块、位置，O:261–273|rival_public 的现金、雇工、土地、作物产量／年龄／肥效等已汇总（C:41–59）；逐地块布局与单位坐标仅结构化保留。对手背包、仓库、种子、未来订单不是公开真值。|
|历史与不确定性|允许的过去公开观测，`decision/api.py:87–94`|历史接口区分市场净变化与真实成交；当前 `TaskAgent` 未传 previous（`task_choice_v1/agent.py:93–100`），`BoundedAgent.previous` 只查回合连续性（`task_dagger_once_v1/policy.py:49–66,81`），网络未用该历史。|

甜瓜是非持续作物（O:16）。年龄 6–12 天内，后续 WATER 当步才可能增产，受肥效和上限约束（O:432–443）；日终对非持续作物跳过生产（O:786–788）。收获会移除该株（O:464–468）。因此不能把“继续等待”说成必然获得夜间产量。

更直接的动作限制在编码之前：T:177–188 的 `needs_bonus` 分支会只放 WATER，而不同时放同株 HARVEST。之后还有运输预分配及时间、资源筛选（T:195–240）。F:61–76 只编码已有 choices，selector 也只能选择 lookup 中的 ID（`task_choice_v1/agent.py:64–78`）。这些教师经营偏好不是官方完整合法动作屏蔽。

HARVEST 已有编码，地块状态也已有表达，最小扩展不必重建框架。但加入此前未表达的候选会改变决策分布，现有权重可计算不代表能可靠决策。应先结合本轮保存轨迹识别具体需要，再另立最小扩展实验；不直接增加大量特征或开始训练。

本轮研究者可以核对已保存双方状态和反事实现金；这些事后真值、未来价格、对手私有库存和反事实标签永远不得作为部署输入。
