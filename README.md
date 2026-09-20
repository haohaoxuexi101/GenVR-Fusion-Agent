# GenShield-Agent：GMC 驱动的自主屏蔽材料发现与 Monte Carlo 认证

参赛队伍：小手震  
成员：沈绍宁（清华大学）  
当前提交版本：2026-09-20

## 1. 项目定位

GenShield-Agent 是一个面向固定材料预算屏蔽布局的工具调用智能体。它在二维 `7×7` 宏网格中保持中心源、材料种类和 **11 个强吸收块**不变，自主决定：

- 选择哪些父代继续生成；
- 使用哪些物理启发算子以及探索/利用比例；
- 哪些候选值得消耗 GMC 预算；
- 何时进入高成本 Monte Carlo（MC）认证；
- 在证据不足时继续搜索或以 `inconclusive` 停止。

当前提交只聚焦以下科学闭环：

```text
约束结构生成
    ↓
GMC 快速筛选并输出候选通量场
    ↓
GMC 证据改变父代、算子和下一代结构
    ↓
GMC 复筛与候选排序
    ↓
冠军 GMC 通量构造权窗
    ↓
Analog MC + 权窗 MC 高精度认证
    ↓
独立 Verifier 接受或否决最终声明
```

本 README 只描述当前 `gmc_only` 生产主线。仓库中的其他探索、训练和历史复现脚本不作为本次提交入口，也不参与当前结论。

## 2. 智能体、物理工具与验证器的职责

### DeepSeek API 智能体

API 模型负责高层科学决策：选择工具、父代、算子组合、筛选批次、MC 资源和停止时机。它不能直接填写材料坐标、修改源、增加屏蔽材料、改变目标函数或修改 Verifier 阈值。

### 约束结构编译器

编译器根据 Agent 选择的策略产生合法结构。每个候选只能执行“一块吸收材料移出、一块吸收材料移入”的等质量交换，因此材料总量始终固定。

### GMC 快速物理评估

GMC 使用缓存的 GenVR/CFM 响应算子计算候选的全场标量通量。本提交配置采用：

- 空间网格：`112×112`；
- 界面位置状态：`n_pos=4`；
- 极角状态：`n_mu=4`；
- 方位角状态：`n_phi=32`；
- 总界面状态数：`512`。

每次 `screen_gmc` 完成后直接复用本次求解结果，不增加新的输运计算，立即保存候选材料排布和 GMC 通量场。

### Monte Carlo 认证

最终候选与基准各自使用对应的 GMC 通量构造权窗。认证同时运行 Analog MC 和权窗 MC；屏蔽结论使用远场绝对通量、根历史标准误、95% 置信区间和一致性检验。权窗只提高计算效率，FOM 不作为“材料结构更好”的判据。

### 独立 Verifier

Verifier 的阈值在运行前冻结。LLM 只能请求完成，不能批准自己的结论。即使 GMC 和 MC 显示大幅下降，只要一致性、置信区间或统计预算未通过，最终声明仍必须是 `inconclusive`。

## 3. 当前优化目标

远场区域定义为距中心源盒最远的约四分之一网格，并划分为 8 个角向扇区。当前 GMC 排序使用预先冻结的项目级综合风险：

```text
GMC risk =
    0.40 × 远场 CVaR90 相对值
  + 0.20 × 远场最大通量相对值
  + 0.15 × 远场平均通量相对值
  + 0.20 × 最差扇区平均通量相对值
  + 0.025 × 热点比相对值
  + 0.025 × 角向不均衡度相对值
```

所有分量均为“候选/基准”，因此 `risk < 1` 表示综合风险优于基准，数值越低越好。该标量化具有明确物理动机，但权重属于本项目预先设定的工程偏好，不应表述为通用核工程标准。最终科学结论不直接使用该综合分数，而由 MC 的远场绝对通量统计检验决定。

实现位置：[`ray_agent/shield_design.py`](ray_agent/shield_design.py)；生产配置：[`configs/gmc_material_discovery_agent.json`](configs/gmc_material_discovery_agent.json)。

## 4. 生产配置与硬门槛

当前配置固定为：

| 项目 | 配置 |
|---|---:|
| 发现模式 | `gmc_only` |
| 最多生成结构 | 96 |
| 最多 GMC 候选 | 16 |
| 单次最多 GMC 候选 | 4 |
| MC 前最少 GMC 候选 | 10 |
| MC 前最少 GMC 引导后代 | 4 |
| MC 网格 | `56×56` |
| 独立重复 | 4 |
| 每个设计、每种方法最少根历史 | 20,000 |
| 总 MC 根历史预算 | 80,000 |
| Analog/权窗一致性门槛 | `|z| ≤ 2.0` |
| 候选优于基准的显著性门槛 | `z ≤ -1.96` |
| 远场 95% 置信区间分离 | 必须 |
| 投影局部 MC 审计 | 默认关闭 |

Agent 至少必须经历一次真实的：

```text
生成 → GMC 筛选 → 使用 GMC 场再生成 → GMC 复筛
```

之后才允许请求 MC。

## 5. 环境安装

前置要求：Conda。GPU 可选；无 NVIDIA GPU 时使用 CPU 环境。

```bash
cd /home/ssn/GenVR
bash scripts/setup_env.sh
```

PowerShell：

```powershell
.\scripts\setup_env.ps1
```

环境规格分别位于 [`environment.yml`](environment.yml) 和 [`environment_cpu.yml`](environment_cpu.yml)。

## 6. API Key

自主 Agent 必须使用真实 DeepSeek API，不存在确定性策略回退。密钥按以下顺序解析：

1. 环境变量 `DEEPSEEK_API_KEY`；
2. 仓库根目录 `.env`；
3. Bash/PowerShell 启动脚本的隐藏交互输入。

推荐直接运行脚本并在提示时粘贴密钥；密钥不会回显或写入日志。也可以使用环境变量：

```bash
export DEEPSEEK_API_KEY="sk-xxxx"
```

或创建不会被 Git 提交的 `.env`：

```text
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

任何曾出现在聊天、截图或终端历史中的真实密钥都应立即撤销并重新生成。

## 7. 运行正式 Agent

Linux/WSL 推荐命令：

```bash
cd /home/ssn/GenVR

RUN_DIR="outputs/gmc_material_discovery/run_$(date +%Y%m%d_%H%M%S)"

MPLCONFIGDIR=/tmp/genvr-mpl \
RAY_AGENT_ENV=openmc-env \
bash scripts/run_gmc_material_discovery_agent.sh \
  --outdir "$RUN_DIR"
```

PowerShell：

```powershell
.\scripts\run_autonomous_shield_agent.ps1 `
  -Config configs/gmc_material_discovery_agent.json
```

输出目录必须为空。不要将新的正式运行写入旧的 `outputs/gmc_material_discovery/trial/`。

## 8. 运行中实时查看

每个新 GMC 候选完成后都会立即更新：

```text
<run_dir>/gmc_screening/latest.png
```

同时保存：

```text
gmc_screening/candidate_*.png    # 候选材料排布 + GMC 标量通量场
gmc_screening/candidate_*.npz    # 原始 GMC 通量与几何数组
gmc_screening/manifest.json      # 全部候选的筛选顺序、风险和文件索引
gmc_screening/latest.json        # 最新候选元数据
trajectory.jsonl                 # API、决策、工具调用和结果的完整轨迹
```

该实时渲染复用已经计算出的 GMC 场，不会增加新的 GMC 求解。

## 9. 运行结束后的核心产物

```text
summary.json                     # 最终声明、Verifier 结果、预算与候选档案
summary.md                       # 人类可读摘要
run_manifest.json                # 模型、环境和关键文件 SHA-256
trajectory.jsonl                 # 完整 API 与工具调用证据链
agent_search_dashboard.png       # 搜索预算与门槛
agent_decision_timeline.png      # Agent 假设、决策和证据时间线
gmc_discovery_storyboard.png     # GMC 发现闭环总览
gmc_screening_reel.gif           # 候选筛选过程动画
candidate_evidence_chain.png     # 候选结构与证据链
mc_certification.png             # Analog/权窗 MC 通量和置信区间
mc_population_control.png        # GMC 引导场、权窗级别、访问与分裂行为
response_fields.npz/json         # 基准与最终候选 GMC 响应场
```

只有当 Verifier 接受 `verified_improvement` 时才生成 `verified_design.png` 和 `best_design.json`。缺少这两个文件并不表示程序失败，可能表示证据不足或认证被拒绝。

## 10. API 确实参与决策的审计证据

每次 API 请求与响应都记录为 `llm_exchange`，包括：

- 时间戳、步骤和重试次数；
- provider、model、base URL；
- 发给模型的观测、记忆和工具白名单；
- API 返回的原始 `assistant_content`；
- `response_id`、`system_fingerprint`、`finish_reason`；
- prompt/completion/total token 用量；
- 响应是否成功解析为可执行工具动作；
- `secret_recorded: false`。

同一步随后记录：

```text
llm_exchange → agent_output → decision → tool_call → tool_result
```

因此可以逐步核对“API 返回了什么动作”以及“程序实际执行了什么工具”。Manifest 固定声明 `llm_required=true`、`deterministic_policy_fallback=false`；删除 API Key 或阻断网络会使运行失败，不会用脚本策略伪装成 LLM 决策。

需要更强的第三方证明时，可将 `response_id`、时间戳和 token 总量与 API 提供方账户的同期用量记录配对。公开提交前不得公开 API Key。

## 11. 当前可审计运行

当前运行目录：

```text
outputs/gmc_material_discovery/run_20260920_020815/
```

运行事实：

- DeepSeek API 决策步骤：12；
- API 交换：17 次，包含解析失败后的同模型恢复重试；
- 不同 API `response_id`：17 个；
- 总 token：233,244；
- 生成结构：59；
- GMC 候选：16；
- GMC 引导且完成复筛的候选：12；
- MC 根历史预算使用：80,000 / 80,000；
- 确定性策略回退：无。

候选 `d7d960548ee7` 的 GMC 综合风险为 `0.07593`。MC 远场候选/基准比为 `0.10999`，差异 `z=-50.80`，两侧 95% 置信区间分离。然而基准设计的 Analog/权窗远场一致性为 `z=-2.1477`，超过冻结门槛 `|z|≤2.0`。Verifier 因此拒绝 `verified_improvement`；MC 预算耗尽后，Agent 最终以 `inconclusive` 停止。

这次运行不能被描述为“已经通过最终物理认证”，但它完整展示了：API 自主决策、GMC 反馈再生成、GMC 权窗 MC、Verifier 拒绝错误声明，以及 Agent 在无剩余认证预算时诚实停止。

## 12. 从已有运行重绘

重绘不会再次调用 LLM，也不会重新运行 MC：

```bash
MPLCONFIGDIR=/tmp/genvr-mpl \
conda run --no-capture-output -n openmc-env \
python scripts/26_plot_autonomous_shield_agent.py \
  --run-dir outputs/gmc_material_discovery/run_20260920_020815
```

只有旧运行缺少 `response_fields.npz` 时才使用 `--recompute-responses`。该参数会从缓存响应算子重建确定性 GMC 场，但仍不会重跑 LLM 或 MC。

## 13. 测试

核心 Agent 与报告测试：

```bash
MPLCONFIGDIR=/tmp/genvr-mpl \
conda run --no-capture-output -n openmc-env \
python -m pytest -q \
  tests/test_shield_autonomous_agent.py \
  tests/test_shield_agent_reporting.py
```

当前开发环境中的相关测试结果为：`15 passed`。

## 14. 代码入口

```text
configs/gmc_material_discovery_agent.json   # 当前生产配置
scripts/run_gmc_material_discovery_agent.sh # Linux/WSL 主入口
scripts/run_autonomous_shield_agent.ps1     # PowerShell 入口
scripts/25_run_autonomous_shield_agent.py   # Agent 驱动与产物写出
scripts/26_plot_autonomous_shield_agent.py  # 无求解重绘
ray_agent/shield_autonomous_agent.py        # Agent 观测—决策—工具循环
ray_agent/shield_agent_protocol.py          # DeepSeek API 与动作协议
ray_agent/shield_agent_tools.py             # 结构、GMC、MC 工具环境
ray_agent/shield_live_reporting.py          # 运行中候选几何与 GMC 场
ray_agent/shield_agent_verifier.py          # 独立冻结阈值验证器
ray_agent/shield_certification.py           # GMC 权窗与 MC 认证
ray_agent/shield_agent_reporting.py         # 运行结束后的完整图集
gmc/                                        # GMC 数值内核与响应算子
```

## 15. 结论边界

当前实现证明的是冻结二维 benchmark 内的自主闭环能力，不是通用工程智能：

- 几何固定为二维 `7×7` 宏网格；
- 材料库、源位置和吸收块总量固定；
- GMC 综合风险的权重属于项目定义；
- 生产结论仍依赖有限历史 MC 与统计门槛；
- 尚未纳入连续几何、能群、真实核数据库、三维泄漏、热工、结构和制造约束；
- 当前可审计运行的最终结论是 `inconclusive`，不得宣传为已认证工程设计。

## 16. 许可

代码按 MIT License 发布。第三方依赖与自产模型权重说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
