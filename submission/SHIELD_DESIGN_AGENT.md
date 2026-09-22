# GenShield-Agent：固定材料预算下的智能屏蔽逆设计

更新日期：2026-09-19。

## 1. 科学问题

本模块研究一个有意收窄、可否证的问题：在 `7×7` lattice 宏网格、中心 `1×1 cm` 体源、两种既有材料以及**恰好 11 个强吸收块**不变时，怎样重新布置吸收材料，降低距源最远四分之一网格中的全场中子通量、尾部热点和最差角向泄漏？

它不是任意 CAD 生成器，也不允许智能体增加屏蔽质量、移动源或改变评价目标。所有候选都由约束编译器执行一次“移除一个吸收块、加入一个吸收块”的合法交换。

## 2. 智能体为什么不是普通参数搜索

### 2.1 分布式伴随灵敏度

快速扩散模型满足

```text
A(phi) = q
A^T(psi) = r_far
```

其中 `r_far` 同时覆盖远端全场和当前最热的 10% 尾部网格。吸收截面的一阶响应灵敏度近似为

```text
S(x) ∝ phi(x) * psi(x)
```

因此候选生成不是简单地把材料移向“正向通量最高处”，而是优先覆盖既有粒子到达、又能影响远端目标的传输通道。该量在 [`ray_agent/shield_design.py`](../ray_agent/shield_design.py) 的 `solve_far_field_adjoint` 与 `PhysicsGuidedProposalModel` 中计算。

### 2.2 角向泄漏与热点风险

远端区域被分成 8 个扇区。目标同时包含：

- 远端平均通量；
- 远端 P90、CVaR90 与最大通量；
- 最差扇区平均通量；
- 单元热点比和角向不均衡度。

这避免优化器只压低总体平均值，却把泄漏集中到一个窄通道。

### 2.3 固定流水线中的证据驱动策略

无 API 时，`EvidenceDrivenShieldStrategist` 也会根据每个变异算子的历史改进、尝试次数和不确定性奖励自适应调整搜索；近期停滞时提高探索比例，有持续改进时收缩探索。可选 LLM 只能输出：

- 预声明算子的权重；
- 8 个扇区之一或不指定扇区；
- 探索比例；
- 可证伪的本轮假设与决策理由。

LLM 不能直接输出材料图，也不能修改质量预算、源、目标或验证阈值。在 `24_*` 固定流水线中，非法 JSON 会被拒绝并回退到确定性科学策略；因此该入口是稳定的自动化搜索器，而不是本文所称的“完全自主 Agent”。

### 2.4 当前主线：GMC 驱动的闭环材料发现

自主 Agent 不再把 GMC 当作只执行一次的“晋级关卡”，而把它作为高通量发现引擎。生产配置使用 `gmc_only`，主流程为：

1. 约束编译器生成一批满足固定材料预算的合法结构；
2. 缓存 GenVR/CFM 局部响应算子以 `112×112` 空间网格和 `n_pos=4,n_mu=4,n_phi=32`（512 个界面状态）执行 GMC 快速评估；
3. 将 GMC 优胜结构作为父代，用其通量场、源距离、路径连续性和泄漏扇区生成下一代结构；
4. 再次用同一 GMC 保真度复筛，只把少量优胜候选送入无偏全场 Monte Carlo；
5. 用基准与候选各自的 GMC 通量生成权窗，并由独立 Verifier 裁决 MC 结果。

生产主线不调用中子扩散模型；扩散评估只保留为历史配置的兼容模式。每一级都重新计算相对 GMC 基准的风险。配置最多筛选 16 个 GMC 候选，要求在 MC 之前至少完成 10 个候选的 GMC 筛选，并至少有 4 个由 GMC 场指导生成的后代完成 GMC 复筛。这个硬门槛保证 Agent 至少经历真实的 `提出结构 → 获取 GMC 证据 → 改变生成策略和下一代 → 再评价` 闭环，而不是第一轮生成后直接调用昂贵 MC。若后代没有超过父代，Agent 仍可回选已验证的早期结构；探索闭环不强迫接受更差设计。

### 2.5 GMC 同时服务发现与 MC 权窗

GMC 在搜索阶段负责快速比较大量布局；在认证阶段，最终 MC 使用基准与候选各自的 GMC 通量构造 MAGIC 型反比通量权窗：粗估通量越低，重要性越高、目标粒子权重越低。冻结认证下限为 `56×56` MC 网格、4 次独立重复、每个设计/方法合计 20,000 根历史，总根历史预算 80,000。所有分裂后代先聚合回独立根历史，再计算远端区域均值、标准误和 95% 置信区间。

屏蔽优劣由**远端绝对通量和置信区间**判断；FOM 只衡量 analog 与权窗方法获得同一无偏估计的效率，不能替代屏蔽目标。

### 2.6 真正的工具调用 Agent

[`scripts/25_run_autonomous_shield_agent.py`](../scripts/25_run_autonomous_shield_agent.py) 将决策权提升到科学工具层。每一轮 DeepSeek 都收到压缩后的高优先级候选、待补齐的保真度证据队列、剩余预算、当前可执行工具和近期摘要，并且必须只选择一个动作：

1. `generate_structures`：选择父代、算子权重、重点扇区、探索比例和本轮预算；
2. `screen_gmc`：自主选择哪些候选进入 GMC 批量评估；
3. `audit_projected_mc`：可选地对 GMC 通过者执行匹配投影局部 MC 学习核审计；
4. `certify_mc`：为一个 GMC 通过者分配网格、根历史、重复数和权窗层级；
5. `finish`：提出最终声明，但无权批准声明。

旧名称 `run_diffusion_batch`、`promote_genvr`、`promote_exact`、`run_unbiased_mc` 仍可用于历史轨迹回放，但进入环境后统一映射为上述新语义。完整工具结果进入不可删改的轨迹日志，下一轮只接收关键风险、远端指标、候选谱系、GMC 引导状态和近期摘要，形成 `观察 → 假设 → 工具调用 → 定量证据 → 再决策`。几何、源、材料总量、目标函数和验收阈值均为保护字段；所有候选仍由合法 swap 编译器生成。

独立 [`ShieldAgentVerifier`](../ray_agent/shield_agent_verifier.py) 要求同一候选通过 GMC 与无偏 MC，并检查：

- 最低 GMC 候选批量和 GMC 引导再生成是否完成；
- baseline 与 candidate 的 analog/权窗远端估计一致性；
- 每种方法的最低独立根历史数和重复数；
- 最低 MC 空间网格是否达到冻结精度下限；
- 候选/基准远端通量比是否达到冻结阈值；
- 远端差值 z-score 与 95% 区间是否分离；
- MC 报告中的设计签名是否与待认证候选一致。

若配置显式设置 `require_projected_mc_audit=true`，Verifier 才额外要求候选通过投影局部 MC 审计。该审计只回答“学习响应算子是否偏离匹配投影 MC”，不替代全局 MC 物理认证。

FOM 只进入诊断字段，不进入接受条件。每个候选最多执行一次最终 MC，避免通过重复抽样进行可选停止。Agent 的过早 `finish` 会被拒绝并反馈；达到硬步数上限时，Verifier 只允许输出 `inconclusive`，不会替 Agent 合成未经请求的科学发现。空正文、截断 JSON 和不可执行工具会由同一个 DeepSeek 策略有限重试；格式失败不占用科学工具步，连续失败达到独立上限后同样只能硬停止为 `inconclusive`，不存在确定性策略代答。

## 3. 运行方式

快速离线搜索：

```bash
python scripts/24_run_shield_design_agent.py \
  --fast-only --rounds 2 --beam-width 3 --proposals-per-parent 12 \
  --outdir outputs/shield_design_agent/fast
```

完整“扩散 → GenVR/CFM 投影响应 → 投影局部 MC 响应”验证：

```bash
python scripts/24_run_shield_design_agent.py \
  --rounds 4 --beam-width 4 --proposals-per-parent 24 \
  --high-fidelity-top-k 2 --exact-top-k 1 \
  --outdir outputs/shield_design_agent/latest
```

增加最终无偏 MC 认证：

```bash
python scripts/24_run_shield_design_agent.py \
  --rounds 4 --beam-width 4 --proposals-per-parent 24 \
  --high-fidelity-top-k 2 --exact-top-k 1 \
  --mc-certify --mc-nx 28 --mc-ny 28 \
  --mc-histories 1000 --mc-replicates 2 \
  --outdir outputs/shield_design_agent/latest
```

若启用外部策略模型，再添加 `--use-llm --model <模型名>`；不启用 LLM 时，证据驱动策略仍完整运行。

当前 GMC 材料发现 Agent：

```bash
bash scripts/run_gmc_material_discovery_agent.sh
```

PowerShell：

```powershell
.\scripts\run_autonomous_shield_agent.ps1
```

也可直接运行：

```bash
export DEEPSEEK_API_KEY="sk-xxxx"
conda run -n openmc-env python scripts/25_run_autonomous_shield_agent.py \
  --config configs/gmc_material_discovery_agent.json
```

该生产入口强制使用真实 DeepSeek API，不提供确定性策略回退。是否构成正式 Agent 结果必须以对应运行目录中的 `run_manifest.json`、`trajectory.jsonl`、`discovery_progress`、MC 认证文件和 Verifier 最终决定为准；仅有 API 调用、GMC 排名或投影局部 MC 响应通过都不能冒充无偏 MC 已认证结论。

## 4. 输出与审计

主目录包含：

- `summary.json`：全部候选、算子记忆、策略轨迹和逐级否决结果；
- `best_design.json`：最终通过最高验证级别的设计，或被否决后的原基准；
- `trajectory.jsonl`：配置、策略、候选、LLM 交换和验证事件；
- `shield_design_comparison.png`：共同绝对色标下的材料图、通量图和搜索轨迹；
- `mc_certification.json`：远端根历史置信区间、analog/VR 一致性及 FOM；
- `mc_certification_fields.npz`：通量、误差、权窗、分裂和区域掩膜；
- `mc_certification.png`：无偏通量、设计比值、重要性级别与远端置信区间。

`25_*` 自主入口按 UTC 时间创建 `outputs/autonomous_shield_agent/<run_id>/`，其中包含：

- `config.json` 与 `run_manifest.json`：有效配置、模型名和关键文件 SHA-256；
- `trajectory.jsonl`：逐轮 Agent 输入、输出、工具调用、结果、环境反馈和裁判决定；
- `summary.json` / `summary.md`：候选档案、预算、记忆和最终声明；
- `mc/<序号>_<签名>/`：最终无偏 MC 的 JSON 与 NPZ；
- `agent_search_dashboard.png`：工具序列、搜索风险、跨保真晋级、预算使用和 Verifier 检查；
- `gmc_screening_reel.gif`：路演动图，逐个展示候选 GMC 测量、实时排行榜、GMC 反馈驱动再生成和最终 MC 晋级；
- `gmc_discovery_storyboard.png`：决赛路演主图，串联 Agent 决策、GMC 代际筛选、算子学习、通量场/通量比和最终 MC 显著性；
- `agent_decision_timeline.png`：逐轮假设、工具选择与预期观测；
- `candidate_evidence_chain.png`：候选材料编辑、GMC 主评估场、代际风险和区域 MC 证据；
- `response_method_comparison.png`：仅在执行可选投影审计时生成，对照同一 `n_pos/n_mu/n_phi` 下的 GMC 与投影局部 MC 响应；
- `response_fields.npz/json`：上述响应场与相空间参数，供后续无求解重绘；
- `verified_design.png`：仅在 Verifier 接受改进时生成，主场图固定使用最终无偏权窗 MC；
- `mc_certification.png`：Analog/权窗通量、逐格误差收益、区域置信区间和一致性诊断；
- `mc_population_control.png`：引导通量、重要性级别、访问增益和分裂位置；
- `best_design.json`：仅在 Verifier 接受候选时生成。

历史自主运行无需重新调用 LLM 或输运求解器，可直接从 `summary.json`、`trajectory.jsonl` 和 MC NPZ 重绘：

```bash
conda run -n openmc-env python scripts/26_plot_autonomous_shield_agent.py \
  --run-dir outputs/autonomous_shield_agent/<run_id>
```

早期运行若没有 `response_fields.npz`，可显式增加 `--recompute-responses`，从原运行记录的 CFM 与投影局部 MC 响应库重新执行确定性全局响应求解；不会重跑 LLM 或 MC 认证。

## 5. 结论边界

当前实现可验证的是冻结 benchmark 内的闭环完备性：Agent 能自主提出结构、读取 GMC 证据、改变下一代、选择精算对象、调用 GMC 权窗 MC，并接受独立 Verifier 的通过或否决。它不等于通用工程智能；当前结果只支持“给定二维 benchmark、材料库和固定 11 块预算下的屏蔽布局逆设计”。进入工程设计前还必须增加连续几何、能群、真实材料核数据、热工/结构/制造约束、三维泄漏路径以及独立 OpenMC 等生产级输运认证。

## 6. 2026-09-18 正式运行结果

运行目录：`outputs/shield_design_agent/latest/`；运行 ID：`shield-20260918T064134Z`。

- 最终设计签名：`d7d960548ee7`；相对基准将外侧 `(3,5)`、`(5,5)` 两块移至源右侧 `(3,4)` 和源下侧 `(4,3)`，材料总量仍为 11 块。
- 鲁棒扩散风险：`0.07355`；CFM 风险：`0.07107`；独立投影局部 MC 响应风险：`0.11029`，两级响应近似均接受该候选。
- 无偏 MC 使用每个设计、每种方法 `2×1000` 个根历史。权窗与 analog 的远端均值差分别为 `0.79σ`（基准）和 `1.40σ`（候选），未发现统计不一致。
- 权窗估计的远端平均通量由 `2.3098e-3 ± 1.3412e-4` 降至 `2.1456e-4 ± 2.2500e-5`，估计下降 `90.71%`，两侧 95% 区间分离。
- 同一设计内，远端均值 FOM 的权窗/analog 增益为 `3.28×`（基准）和 `3.66×`（候选）。
- 固定为“基准远端最低通量四分位”的更深区域只观察到 `30.4%` 下降，差异约 `-1.96σ`，95% 区间尚未分离；因此当前不能声称所有最深穿透网格都已获得同等幅度改善。

上述数值来自 `24_*` 固定流水线，可从 `summary.json.final_verified` 与 `mc_certification.json` 直接复核；图见 `shield_design_comparison.png` 和 `mc_certification.png`。它们不能冒充 `25_*` 自主 Agent 的正式运行结果。
