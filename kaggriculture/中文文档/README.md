# Kaggriculture 中文资料入口

这是一套两人对战的农场经营游戏。参赛者编写 Bot，由程序安排农夫、工人、种植、养殖和交易，以季末现金决定胜负。

## 按这个顺序阅读

1. [官方完整游戏规则·中文版](游戏规则.md)：任务目标、全部动作、作物和动物数值、城镇需求、价格公式、结算顺序、状态结构及默认配置。
2. [官方开发与参赛指南·中文版](开发与参赛指南.md)：策略函数、输入输出接口、示例代码、本地测试，以及官方文档中的登录、参赛、提交、下载回放和日志步骤。
3. [接口与通信说明](接口与通信说明.md)：本地程序怎样交换数据、全部字段和指令中文对照、公开与私有信息、时间限制及常见误区。
4. [回放怎么看](回放怎么看.md)：人物编号、动作气泡、日终回仓、市场订单和播放控制。
5. [机器接口定义·中文注释版](接口定义.zh-CN.json)：与官方 JSON 相同的字段、类型、默认值与约束，全部说明文字已翻译。

## 直接看中文回放

- [完整对局·中文界面与人物动作](../../review/PROGRESS.md)
- [之前那场回放·已汉化](../../review/PROGRESS.md)

## 资料版本与范围

前两份是本地官方源码快照中对应文件的中文译本，保留全部章节主题、数值表、接口字段和命令；代码注释译为中文。第三、四份是本项目补充说明。字段名、命令、文件名保持原样，以便直接对应程序。

对应官方源码提交：`f994f2050d58d84f649329c1e35ce09ada7007c5`，本地安装包：`kaggle-environments==1.33.0`。译本描述这次部署的版本，不表示已核对当前线上赛事政策。

- [英文规则原文](https://github.com/Kaggle/kaggle-environments/blob/f994f2050d58d84f649329c1e35ce09ada7007c5/kaggle_environments/envs/kaggriculture/README.md)
- [英文开发指南原文](https://github.com/Kaggle/kaggle-environments/blob/f994f2050d58d84f649329c1e35ce09ada7007c5/kaggle_environments/envs/kaggriculture/AGENTS.md)
- [原始机器接口定义](https://github.com/Kaggle/kaggle-environments/blob/f994f2050d58d84f649329c1e35ce09ada7007c5/kaggle_environments/envs/kaggriculture/kaggriculture.json)
- [官方引擎源码](https://github.com/Kaggle/kaggle-environments/blob/f994f2050d58d84f649329c1e35ce09ada7007c5/kaggle_environments/envs/kaggriculture/kaggriculture.py)

本地比赛不需要登录。开发指南里的登录和提交章节只用于阅读；本次汉化不会登录、接受赛事条款或提交比赛。
