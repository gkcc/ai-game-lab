# Windows 到 WSL 的一次性外层候选

`host.py` 尚无正式登记，也未执行真实训练。调用方须在运行前独立认证 Windows host 文件、系统 WSL 可执行文件、解释器、登记提交以及私有合同身份；程序自身的源码检查发生在程序启动之后，不等于启动前认证。缺登记时入口在读取私有合同或启动 WSL 前失败。

在不把本机信息发布到仓库的前提下，调用方需要事前独立固定一份本地 host 合同及其 SHA-256。合同包含 `distribution`、`windows_game_root`、`windows_training_contract`、`windows_wsl_executable`、`wsl_game_root`、`wsl_training_contract`、`wsl_repo_root`、`wsl_venv_executable`、`training_contract_sha256` 及固定 schema；路径和发行版名称都是私有值。host 验证 Windows 的 WSL 程序位于系统目录，并核验训练私有合同的预固定摘要及其中未解析的虚拟环境 Python 路径；不声称验证 Windows 二进制签名或第三方供应链。合同摘要不能从被验证合同在运行时临时取代独立期望值。

host 启动前要求固定的 Linux 监督目录和子进程输出目录均不存在，并排他创建独立的本机 host 记录，写入一次性随机 nonce。它使用 argv 和 `shell=False` 调用一次 `wsl.exe --exec`，不调用 shell，不打印命令、路径、nonce、合同摘要或输出。WSL Python 首先运行包含在已登记 host 源码中的短 bootstrap：它有界读取一次 Linux launcher 的源码，核对登记 SHA 后直接执行同一快照，并设置 launcher 的 `__file__` 和参数；WSL 路径误指向不同字节时不会执行该文件。此检查不认证解释器启动时的代码或抵御同用户恶意并发。Linux 启动记录写入同一 nonce、登记 commit 和训练合同摘要；Windows 只有在这些值都匹配本次运行时，才向该 Linux 私有目录写 `cancel.request`。若路径映射错误或启动记录尚未出现，host 不会猜测其他目录，最多在取消等待窗口内重复检查绑定。

host 以登记的外层总时限采样等待；到期或等待过程收到 KeyboardInterrupt 时，仅请求取消并再等待最多 60 秒，不调用 `kill`、`terminate`，也不终止整个 WSL。若进程构造异常，不能断言 WSL 没有启动：host 只在有限时间和次数内尝试对本次 nonce 绑定的目录请求取消，结果仍为 unknown。启动已发生后若等待、计时或睡眠发生异常，会先尝试对已绑定目录请求取消；计时设施不能继续工作时立即标记 unknown。没收到真实 WSL 退出或 Linux 会话清理证据时，结果为 unknown，不自动重试。已经观测到超时、取消或等待异常时，即使随后 Linux 返回零退出且清理已确认，host 也不会将该次外层运行标为 passed。宿主断电、进程被外部强杀、文件系统卡住和 Linux 会话逃逸仍可能留下无法确认的状态。

运行期限依赖标准 Python `time.monotonic` 正常提供有限且单调的时间值。独立复审用可注入的测试时钟构造过非数字、NaN 等反例；没有证据表明本机正常的标准时钟会返回这些值，也不能把该测试接口当成对任意自定义时钟的保证。

正常退出后，host 读取有界的 Linux 启动、监督结果、原始规范收据和私有摘要，检查 nonce、令牌、步数门槛、三项 pidfd 清理标记及技术结果；它不读取模型或重新评估开发数据。通过只说明对应的本机过程证据齐备，开发筛查仍单列 passed、failed 或 unknown，不说明官方比赛效果。私人 `host-result.local.json` 的 `host_exit_confirmation` 固定为 false：文件本身不能代替 Windows host 实际退出结果；若最终写入期间超时，会另写本机迟到标记并返回失败。

目前验收仅包括 Windows 与 WSL 的内存合成、短命进程等待、注入超时/取消/无证据失败，以及一次从 Windows 用 argv 运行 WSL `/bin/true` 的零退出探针；其中监督证据为合成，没有发起训练、读取真实数据或模型、使用 GPU、触发比赛或清理任何临时目录。未经独立复审、完整登记和真实运行验收，不得把候选状态称为训练通过。
