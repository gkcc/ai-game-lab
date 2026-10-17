# Lux AI Season 3

官方CLI与Python Starter Kit。环境与Kaggriculture隔离；CPU/GPU都保持官方源代码，不改系统驱动。

```powershell
.\lux-s3\Setup.ps1
.\lux-s3\Play.ps1 -Seed 42
```

Setup会获取固定官方Commit，在自己的虚拟环境构建源码wheel并核验安装后的Python文件。Bot位于 `bot/agent.py`。本地比赛结果和Replay在 `artifacts/`，不会提交到Git。

GPU适配需要先自行配置WSL2 NVIDIA支持，再将 `wsl-config.example.json` 复制为不入Git的 `wsl-config.json`，填写自己的发行版、用户、Linux Python路径和缓存路径。确认目标是专用虚拟环境后再运行 `Setup-GPU.ps1`、`Check-GPU.ps1`。

原私有环境的GPU矩阵运算、官方小规模吞吐与完整CLI对局已验收；公开版移除了主机配置，将设备检查泛化为CUDA后端，其安装适配未在全新机器上重新验收。不要仅凭GPU可见就声称运算或整条训练链已经加速。
