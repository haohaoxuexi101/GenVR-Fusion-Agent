# official_outputs —— 完整运行波次与推荐结果

本目录是 2026-09-03 的完整运行波次产物，涵盖官方消融复现（flash 与 pro 跨模型）、112×112 开放探索与模型矩阵验证。**推荐结论在文末，给出了具体的实验名称与文件路径。**

## 目录结构

```text
official_outputs/
├── README.md                    # 本文件：波次汇总与推荐结果
├── REPORT_CONTENT.md            # 科学发现与环境定义报告正文（五个必答问题已填）
├── SLIDES_CONTENT.md            # 方案说明 PPT 逐页内容
├── OPENxxx_RayLab-GMC_slides.pptx  # 生成的答辩 PPT
├── official_llm_flash/          # 官方消融复现（deepseek-v4-flash）
├── official_llm_pro/            # 官方消融复现（deepseek-v4-pro，跨模型复核）
├── open_explore_lattice_112/    # 开放探索（LLM 自主提议，112×112）
└── model_matrix/                # DeepSeek 模型矩阵验证
```

## 各波结果

| 波次 | 目录                        | 模型              | 结果                                                                                                                                                                                                                                     |
| ---- | --------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `official_llm_flash/`       | deepseek-v4-flash | 4 实验消融 + 3 随机参照；角向在 128 预算下射线指标较位置低 **43.5%**；联合加密的局部 Exact-MC 指标较粗网格低 **57.3%**，CFM 残差 **5.0%**                                                                                                |
| 2    | `official_llm_pro/`         | deepseek-v4-pro   | 与波 1 数字逐位一致（环境确定性，模型只影响顺序）；结论相同                                                                                                                                                                              |
| 3    | `open_explore_lattice_112/` | deepseek-v4-flash | 5 个 LLM 提议 + 2 随机参照；最终复验中 LLM 最佳提议 `proposal_p2_m4_f32`（256 状态）较基线低 **33.6%**（射线指数 0.003195），仍略高于随机参照最优 0.002976；注意该模式提议序列有随机波动（另一次运行最佳为 `proposal_p2_m2_f16`、+4.6%） |
| 4    | `model_matrix/`             | flash + pro       | 两模型均通过白名单，且都选择 `angular_refined`（flash 1.59s、pro 4.49s）                                                                                                                                                                 |

## 推荐结果（具体名称 + 路径）

### 1. 最省钱缓解策略 —— 推荐 `angular_refined`

- **实验名**：`angular_refined`（n_pos=1, n_mu=4, n_phi=32，128 状态）
- **依据**：同为 128 状态预算，CFM 射线指标 0.002501，比 `position_refined` 的 0.004431 低 43.5%，是本次所有实验里「同预算下射线指标最低」的配置。
- **证据文件**（flash 波次）：
  - 指标：`official_outputs/official_llm_flash/experiments/angular_refined/metrics.json`
  - 诊断：`official_outputs/official_llm_flash/experiments/angular_refined/diagnostics.json`
  - 通量图：`official_outputs/official_llm_flash/experiments/angular_refined/lattice_stage8_ultra.png`
- **推荐文件**：`official_outputs/official_llm_flash/recommendation.json`（`overall_recommendation` 字段）

### 2. 最干净参照（预算更高）—— 推荐 `combined_refined`

- **实验名**：`combined_refined`（n_pos=4, n_mu=4, n_phi=32，512 状态）
- **依据**：局部 Exact-MC 相对固定 Dense 的射线指标 0.000911，较 `coarse` 的 0.002136 低 57.3%；代价是 4 倍预算，且 CFM 相对 Exact-MC 仍保留 5.0% 归一化 L1 残差。
- **证据文件**：
  - `official_outputs/official_llm_flash/experiments/combined_refined/metrics.json`
  - `official_outputs/official_llm_flash/experiments/combined_refined/diagnostics.json`
  - `official_outputs/official_llm_flash/experiments/combined_refined/lattice_stage8_ultra.png`

### 3. 开放探索（LLM 自主提议）—— 诚实负结果

- 最终复验中 LLM 在 5 次提议内的最佳配置为 `proposal_p2_m4_f32`（256 状态，射线指标 0.003195，较基线低 33.6%），但仍**未超过随机参照** `random_1_p2_m2_f64`（0.002976）。
- **结论与波动**：该模式提议序列有随机性——另一次独立运行的 LLM 最佳是 `proposal_p2_m2_f16`（仅 +4.6%）；两次都未在预算内稳定发现角向最优解（`n_mu=4` 高角向组合），说明「预声明白名单消融 + 开放提议」应配合使用，且开放探索边界仍需扩大。
- **证据**：`official_outputs/open_explore_lattice_112/summary.json` 的 `best_config`/`recommendation` 字段；轨迹 `trajectory.jsonl` 的 `llm_exchange`/`decision` 事件。

## 指标速查

- **`ray_index`**（越低越好）：残差 2D FFT 后「方向集中度 × 残差 RMS」。一条空间直线射线在谱上表现为与其方向垂直的窄角带功率集中；相乘同时看方向性与幅度。`43.5%` 即用 `cfm_ray_vs_dense.ray_index` 比较得出。
- **`normalized_rel_l1`**（越低越好）：`Σ|候选−参照|/Σ|参照|`，归一化形状的平均绝对误差；`5.0%` 学习算子残差即 `cfm_vs_oracle.normalized_rel_l1`。
- **`log10_rmse`**（越低越好）：对数坐标下的 RMS，突出弱通量区保真度。
- **`corr`**（越接近 1 越好）：两场逐格皮尔逊相关，形状一致度。
- **`n_state = n_pos×n_mu×n_phi`**：每个界面的相空间状态数（预算）；`wall_s = cfm_build_s + cfm_solve_s`；`cfm_residual` 是求解器收敛残差（非物理误差）。
- 详细推导与物理含义见 [`REPORT_CONTENT.md`](official_outputs/REPORT_CONTENT.md) 附录 A 与 [`SLIDES_CONTENT.md`](official_outputs/SLIDES_CONTENT.md) 第 6/7/16 页。

## 跨模型一致性

`official_llm_flash` 与 `official_llm_pro` 的四个实验指标逐位一致（0.003843 / 0.004431 / 0.002501 / 0.003126），科学声明完全相同——结论由确定性环境与预声明预算决定，不依赖具体模型；两模型仅决策顺序与推理文本不同（见各自 `trajectory.jsonl`）。

## 复现命令

```powershell
.\scripts\reproduce_core.ps1                          # 波 1（默认 flash）
.\scripts\reproduce_core.ps1 -Model deepseek-v4-pro   # 波 2
.\scripts\explore_open.ps1                            # 波 3
.\scripts\validate_deepseek_models.ps1                # 波 4
```

各波次对应的冻结配置、代码哈希与随机种子见各目录 `run_manifest.json` 与 `config.json`。
