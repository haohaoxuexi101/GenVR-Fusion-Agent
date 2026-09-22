# 科学发现与环境定义报告（正文，含指标词典）

> 对应 `submission/复赛提交说明.md` 第四节五个必答问题。正式采用的运行：`submission/runs/official_llm/`；本波次复现证据在 `official_outputs/`。提交前需替换队号 `OPENxxx` 并填入公开仓库链接与冻结 tag。

## 1. 科学问题与发现目标

**科学问题**：经典离散纵标射线效应源于连续角空间被有限方向替代。本项目研究的新问题是——随机粒子（MC）生成的局部响应在被压缩为有限界面状态（CFM 条件流匹配学习）并反复全局组合后，是否**重新获得方向性伪影**？

**为什么值得研究**：它提供一条可证伪、可量化的桥梁，把「MC/生成模型的连续样本质量」与「最终离散全局解质量」分开，并回答「有限的界面状态预算应优先分配给位置还是角向」。

**期望 Agent 发现**：哪一种界面相空间离散化机制主导射线伪影，以及最便宜的缓解配置。

**核心 Scientific Claim**（完整复现确认，flash 与 pro 逐位一致）：
> 在单一预声明种子、二维非均匀 lattice、给定检查点与采样预算下，把 128 个界面状态分配给角向离散（`angular_refined`）比只增加面位置节点（`position_refined`）使 CFM 相对固定 Dense 的谱射线指数低 **43.5%**；联合加密（`combined_refined`）使局部 Exact-MC 相对固定 Dense 的射线指数较粗网格降低 **57.3%**，但联合加密后的 CFM 相对其 Exact-MC 仍保留 **5.0%** 归一化 L1 误差——离散化加密能缓解伪影，但不能单独消除学习响应算子的残差。

## 2. Environment / Benchmark 定义

- **Observation**：`ray_index`、`normalized_rel_l1`、`corr`、`log10_rmse`（定义见附录 A）、实验耗时 `wall_s`、状态数 `n_state`，以及上一轮实验的完整历史。
- **Action**：模式 B 从 4 个预声明实验白名单中选择；模式 C 在声明边界内自由提议 `{n_pos, n_mu, n_phi}`。
- **Tools**：`scripts/30_stage8_ultra_benchmark.py`（执行 CFM / Exact-MC / 固定 Dense 三方诊断并落盘 metrics/npz/png）。
- **数据与资源**：benchmark 几何与材料由 `gmc/benchmarks2d.py` 程序化生成，无外部测试数据集；训练数据自产（`outputs/hpc_v2_data/`），检查点为自产 `outputs/hpc_v2_models/*_900k_100ep/best.pt`（MIT 再分发，训练管线见 `scripts/01/02/08/09/32`）。
- **验证机制**：CFM / Exact-MC / 固定 Dense 三方交叉比较 + 3 组随机参照（换种子）。**无隐藏 Ground Truth 或测试集**，Agent 可见信息 = 历史指标 + 声明边界，不存在信息泄漏。

## 3. Agent 如何完成发现

- 模式 B（预声明消融）：`BootstrapPolicy` 跑粗基线 → `DeepSeekPolicy` 依据反馈从白名单选择 → 环境执行 → 反馈回 LLM，共 3 轮 LLM 决策；随后 `RandomPolicy` 跑 3 组随机参照。`llm_required=true`，API 失败即终止、不回退。
- 模式 C（开放探索）：`OpenProposer` 每轮返回 JSON 提议，越界/超预算/重复提议被拒绝并把理由送回重提（每轮最多 3 次）；若 JSON 被 token 截断，先用正则抢救数值字段，失败再重提。
- **与轨迹对应**：每个决策/提议、每次 API 交换（`llm_exchange`）、每次工具调用（`tool_call`/`tool_result`）、每次环境反馈（`environment_feedback`）都写入运行目录的 `trajectory.jsonl`。
- **如实说明多次运行**：本波次 4 个正式运行目录见下表；历史 `submission/runs/` 另有多次记录。**无人工干预、无 Best-of-N 筛选**：每个因果实验只跑一次。开放探索具有随机性——两次独立运行提议不同（一次最佳 `proposal_p2_m2_f16` 仅 +4.6%，一次最佳 `proposal_p2_m4_f32` −33.6%），这是如实记录的探索波动。

| 运行目录                                     | 模型  | 用途         |
| -------------------------------------------- | ----- | ------------ |
| `submission/runs/official_llm/`              | flash | **正式采用** |
| `official_outputs/official_llm_flash/`       | flash | 完整复现     |
| `official_outputs/official_llm_pro/`         | pro   | 跨模型复核   |
| `official_outputs/open_explore_lattice_112/` | flash | 开放探索     |

## 4. 发现结果与验证

**核心发现**（flash 与 pro 两波逐位一致）：

| 实验             | n_pos | n_mu | n_phi | 状态数 | CFM ray_index（vs Dense） | 说明                 |
| ---------------- | ----- | ---- | ----- | ------ | ------------------------- | -------------------- |
| coarse           | 1     | 2    | 16    | 32     | 0.003843                  | 粗基线               |
| angular_refined  | 1     | 4    | 32    | 128    | **0.002501**              | 角向细化，同预算最优 |
| position_refined | 4     | 2    | 16    | 128    | 0.004431                  | 只加位置节点反而变差 |
| combined_refined | 4     | 4    | 32    | 512    | 0.003126                  | 局部 Exact-MC 最优   |

- **独立验证**：`deepseek-v4-pro` 复现出与 flash 完全相同的数字（模型只影响决策顺序与推理文本）；
- **随机种子**：3 组随机参照指标落在 coarse/position 水平，随机最优 0.002095，证明 43.5% 的差距不是种子波动；
- **对照**：固定 Dense S_N 参照（`dense_n_mu=4, dense_n_phi=32`）作为不随实验变化的公共比较；
- **稳定性**：结论由确定性环境与预声明预算决定，跨模型、跨次运行稳定（两次 flash + 一次 pro 数字一致）；开放探索的提议序列有波动（见第 3 节）；
- **limitation / false positive**：射线指标是诊断代理量而非通用物理量；单种子、未做匹配多种子置信区间；开放探索在有限预算内未能稳定超过随机参照最优（0.002976）。

## 5. 开源资源链接

- 代码仓库：**待填**（GitHub/GitLab/Gitee），冻结 tag/commit：**待填**
- 数据 / Benchmark：几何与训练数据为程序化自产；权重 `outputs/hpc_v2_models/*_900k_100ep/best.pt` 建议挂 Zenodo/Hugging Face 并提供 SHA-256（见各 `run_manifest.json`）；构建流程见 `scripts/01/02/08/09/32`。
- 第三方依赖版本与 License：`environment.yml`、`THIRD_PARTY_NOTICES.md`。
- 复现入口：`README.md`（一键 setup / smoke / reproduce / explore）。

---

## 附录 A：指标词典与物理含义

> 所有指标在 [`ray_agent/metrics.py`](ray_agent/metrics.py:1) 中定义；前处理统一为 `normalize_flux`：把通量场按最大绝对值缩放到 [-1,1]，消除量纲，专注**形状**比较。

### A.1 谱射线指数（ray_index）—— 核心诊断量

`ray_index = residual_rms × directional_concentration`

- **计算**（[`spectral_ray_metrics`](ray_agent/metrics.py:29)）：取「候选场 − 参照场」的残差 → 2D Hanning 窗 → 2D FFT 功率谱 → 在半径 0.03~0.45 的环带内（剔除直流与奈奎斯特边缘），把功率按方向角 θ（模 π，共 36 个 bin）累加。
- **`directional_concentration`**：功率最强的 4 个角向 bin 占总功率的比例。**物理含义**：一条空间上的直线射线，其傅里叶功率集中在与射线方向垂直的窄角带内；若残差各向同性，该比例趋近 4/36≈0.111。值越高 = 残差越「像射线」。
- **`residual_rms`**：残差的均方根，表示误差的**幅度**（归一化后无量纲）。
- **为什么相乘**：一个很小但高度方向性的残差不该被过度惩罚，一个很大但各向同性的残差也不该被误判为射线——`ray_index` 同时看「方向性」和「幅度」。**越低越好**；本报告用它比较同预算下的不同离散化。
- 附带输出 `directional_peak_to_mean`（最强角向 bin / 平均 bin），越大说明方向性越尖锐。

### A.2 通量形状三件套（flux_shape_metrics）

`corr`、`normalized_rel_l1`、`log10_rmse`（[`flux_shape_metrics`](ray_agent/metrics.py:17)），比较候选场与参照场的**整体形状**：

- **`corr`（皮尔逊相关系数）**：两场逐格取值（展平后）的线性相关，∈[-1,1]。**物理含义**：空间分布形态是否一致；越接近 1 越好。它对整体亮度不敏感（已归一化）。
- **`normalized_rel_l1`（归一化相对 L1 误差）**：`Σ|候选−参照| / Σ|参照|`。**物理含义**：相对总通量的平均绝对偏差，是「总量级」上的形状误差；越低越好。本报告的 **5.0% 学习算子残差**即 `cfm_vs_oracle.normalized_rel_l1`。
- **`log10_rmse`（对数 RMS 误差）**：对两场逐格取 log10（仅在有值的格子）后的 RMSE。**物理含义**：用对数坐标衡量误差，对**弱通量区域**（对数空间差距被放大）敏感；越低越好，反映暗区的保真度。

### A.3 三方交叉比较的含义

每个实验产出三组标量场（npz 中的 `cfm`、`oracle`、`dense`），两两比较得到：

| 比较                                      | 名称                   | 物理含义                                                |
| ----------------------------------------- | ---------------------- | ------------------------------------------------------- |
| `cfm_vs_dense` / `cfm_ray_vs_dense`       | CFM vs 固定 Dense      | 学习算子 + 离散化的**总误差**；射线指标用于判定伪影强弱 |
| `oracle_vs_dense` / `oracle_ray_vs_dense` | Exact-MC vs 固定 Dense | 「真值」局部响应算子本身的投影/离散误差（下界参照）     |
| `cfm_vs_oracle`                           | CFM vs Exact-MC        | **纯学习算子残差**（5.0% 就来自这里）                   |

- **Dense** 是固定确定性 S_N 场（`dense_n_mu=4, dense_n_phi=32`），不是连续方程真值，只作不随实验变化的公共参照。
- **Exact-MC（oracle）** 是每个局部算子 1000 样本的蒙特卡洛响应，仍带有限样本误差。

### A.4 工程/成本指标（diagnose_experiment）

- **`n_state = n_pos × n_mu × n_phi`**：每个界面上的总相空间状态数（位置节点 × 极角 × 方位角）。**物理含义**：离散化自由度预算；本报告「同 128 状态预算」即由此定义。
- **`n_angle`**：界面角向状态数（来自 metrics.json）。
- **`cfm_build_s`**：局部 CFM 响应算子构建耗时；**`cfm_solve_s`**：全局响应矩阵确定性求解耗时；`wall_s = build + solve`。
- **`cfm_residual`**：全局响应矩阵线性求解器的残差范数，表示**数值收敛**程度（不是物理误差），应远小于物理指标。

### A.5 如何解读「43.5% / 57.3% / 5.0%」

- **43.5%**：同为 128 状态，`angular_refined` 的 `cfm_ray_vs_dense.ray_index`（0.002501）比 `position_refined`（0.004431）低 43.5% → 资源应优先给角向离散。
- **57.3%**：`combined_refined` 的 `oracle_ray_vs_dense.ray_index`（0.000911）比 `coarse`（0.002136）低 57.3% → 联合加密能让「真值」算子的伪影大幅下降。
- **5.0%**：`combined_refined` 的 `cfm_vs_oracle.normalized_rel_l1` = 0.050 → 学习算子相对其 Exact-MC 仍有 5% 形状误差，说明离散化加密缓解了伪影，但学习残差仍存。
