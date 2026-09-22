# API 与安全说明

当前正式运行位于 `outputs/gmc_material_discovery/run_20260920_020815/`，强制使用真实 DeepSeek API。该运行包含 12 个决策步骤、17 次 API 交换和 17 个唯一 `response_id`，总计 233,244 tokens；`run_manifest.json` 明确记录 `llm_required=true`、`deterministic_policy_fallback=false` 和启动后无人工干预。

DeepSeek 对原有固定流水线是可选策略，但对 `scripts/25_run_autonomous_shield_agent.py` 是强制策略。密钥按以下顺序自动解析：进程环境变量 `DEEPSEEK_API_KEY` → 仓库根目录 `.env` 文件（脚本自动加载，已被 `.gitignore` 排除）→ 交互式隐藏输入（不回显、不落盘、结束后清除）。密钥不会进入配置、manifest、命令、日志或错误消息。

自主屏蔽 Agent 只能看到候选签名、编译器生成的合法交换摘要、各保真度风险、物理诊断、预算和当前可执行工具白名单；不能读取任意文件，也不能修改几何约束、源、材料预算、目标或 Verifier 阈值。空正文、截断 JSON 或不可执行工具会触发同一个 DeepSeek 策略的有限恢复重试，所有尝试均记录在 `llm_exchange`；API、网络或非受控工具故障会直接终止运行，**不存在确定性回退冒充 Agent**。

安全规则：

- 不把真实密钥写入 `.env.example`、配置、命令行参数或 notebook；
- `.gitignore` 排除 `.env`、`.env.*` 和 `secrets/`；
- 提交自检扫描常见 `sk-...` 模式，只报告数量，不回显内容；
- 发布前轮换任何曾在聊天、截图或终端历史中出现的密钥；
- 公共仓库只保留变量名和空示例。

截至 2026 年 9 月 20 日，交互式入口仍以隐藏输入读取密钥，运行轨迹只记录 `secret_recorded: false`，不记录密钥值。公开仓库为 https://github.com/haohaoxuexi101/GenVR-Fusion-Agent。任何曾出现在聊天明文、截图或终端历史中的密钥仍应立即撤销并轮换。
