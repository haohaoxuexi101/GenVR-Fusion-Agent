# GenShield-Agent 可追溯链

当前口径日期：2026-09-20。正式证据目录为 `outputs/gmc_material_discovery/run_20260920_020815/`。

## 1. 总体证据链

```text
生产配置 + 关键源码/响应库 SHA-256
  -> DeepSeek API 观测与原始响应
  -> Agent 假设、预期观测和工具选择
  -> 受约束结构编译
  -> GMC 候选场、指标与反馈再生成
  -> 候选专属 GMC 权窗
  -> Analog MC + 权窗 MC
  -> 独立 Verifier 接受或拒绝声明
  -> summary.json / 图件 / 报告
```

## 2. 版本、队伍与数据

- 队伍：小手震
- 公开仓库：https://github.com/haohaoxuexi101/GenVR-Fusion-Agent
- tag：`v0.1.0`
- commit：`ca885735e09cb46a6afba29c1a47f19d0fd8eda1`
- 数据来源：全部由项目代码生成，不使用外部训练数据集。

边界与内部响应训练样本分别由 `scripts/01_generate_boundary_data.py` 和 `scripts/08_generate_internal_data.py` 生成；底层 benchmark 与随机飞行位于 `gmc/benchmarks2d.py` 和 `gmc/mc_cell.py`。大型检查点和响应缓存属于自产、可再生成中间产物。

## 3. 运行时冻结

`run_manifest.json` 记录：

- 创建时间、Python 与操作系统；
- Agent 类、provider、model、是否必须调用 LLM；
- `discovery_mode=gmc_only` 与 `diffusion_model_used=false`；
- GMC 空间与界面状态；
- 配置、Agent、Verifier、认证模块和响应库的字节数与 SHA-256；
- `human_intervention=None after launch`；
- `deterministic_policy_fallback=false`。

## 4. API 决策链

`trajectory.jsonl` 每行包含 `schema_version`、`run_id`、UTC 时间和 `event_type`。正式运行事件计数为：

- 17 个 `llm_exchange`；
- 12 个 `agent_input` 与 12 个 `agent_output`；
- 12 个 `decision`、12 个 `tool_call`、12 个 `tool_result`；
- 11 个 `environment_feedback`；
- 2 个 `verifier_decision`；
- 1 个 `run_started`、1 个 `run_completed`、1 个 `report_generated`。

每一步均可按以下顺序对照：

```text
llm_exchange -> agent_output -> decision -> tool_call -> tool_result
```

API 记录含 `response_id`、`system_fingerprint`、token 用量、原始响应、解析状态和 `secret_recorded=false`。17 次交换使用 17 个不同 `response_id`。

## 5. GMC 发现链

- `gmc_screening/manifest.json`：16 个候选的实际筛选顺序、风险、运行时间与文件索引；
- `gmc_screening/candidate_*.png`：候选几何和 GMC 标量通量场；
- `gmc_screening/candidate_*.npz`：原始几何和 GMC 数组；
- `gmc_screening_reel.gif`：按真实完成顺序生成的路演动画；
- `summary.json.candidate_archive`：父代、材料交换、代数、GMC 证据和最高验证等级。

## 6. MC 与 Verifier 链

- `mc/01_d7d960548ee7/certification.json`：基准/候选 × Analog/权窗的区域统计、置信区间、一致性和设计比较；
- `mc/01_d7d960548ee7/fields.npz`：通量、相对误差、权窗、重要性级别和区域掩膜；
- `mc_certification.png` 与 `mc_population_control.png`：认证和人口控制可视化；
- 第 10 步 `verifier_decision`：拒绝 `verified_improvement`；
- 第 12 步 `verifier_decision`：接受最终 `inconclusive`。

## 7. 报告派生

`scripts/27_build_scientific_finding_report.py` 读取 Markdown、正式轨迹和现有图件，生成：

- `submission/assets/api_participation_audit.png`
- `output/pdf/GenShield-Agent_科学发现与环境定义报告.pdf`
- 旧文件名兼容副本 `output/pdf/GenVR-Fusion_科学发现与环境定义报告.pdf`

报告重建不会调用 API，也不会重跑 GMC 或 MC。
