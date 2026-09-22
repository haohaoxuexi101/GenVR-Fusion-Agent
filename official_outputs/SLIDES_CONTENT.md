# 方案说明 PPT 逐页内容

> 建议 17 页；文件名 `OPENxxx_RayLab-GMC_slides.pptx`（已生成）。每页标题 + 要点 + 图。

**第 1 页 · 封面**
- 标题：GenVR-Fusion：面向聚变堆深穿透屏蔽的生成式稀有事件输运探索环境
- 副标题：GMC 响应算子射线伪影的因果审计智能体
- 队伍：小手震 · 沈绍宁（清华大学）

**第 2 页 · 问题定义（看见了什么问题）**
- 数值输运中，随机粒子生成的局部响应被压缩为有限界面状态并反复全局组合后，重新出现方向性射线伪影
- 量化：粗基线 `coarse` 的 CFM 射线指数 0.003843、归一化 L1 0.0797
- 图：`official_outputs/official_llm_flash/experiments/coarse/lattice_stage8_ultra.png`

**第 3 页 · 问题定义（为什么 AI 可以介入 + 边界）**
- 假设空间小（离散化配置组合）、实验反馈快（秒级~40s）、决策可白名单约束、探索-验证闭环可由轨迹审计
- 边界：二维非均匀 lattice、给定检查点、单种子、射线指标是诊断代理量

**第 4 页 · 探索环境设计（固定什么 / 可探索什么）**
- 固定：环境、900k 检查点、种子 260827、采样预算 1000、固定 Dense 参照、搜索边界
- 可探索：模式 B 白名单消融（4 动作）；模式 C LLM 自由提议 `{n_pos, n_mu, n_phi}`

**第 5 页 · 反馈机制与验证**
- `ray_index = FFT 方向集中度 × 残差 RMS`；归一化 L1；对数 RMSE；相关系数
- CFM / Exact-MC / 固定 Dense 三方交叉诊断；3 组随机参照换种子；无隐藏 Ground Truth

**第 6 页 · 指标含义（一）：ray_index 的物理意义**
- 残差做 2D FFT：一条空间直线射线的功率集中在与方向垂直的窄角带
- `directional_concentration` = 最强 4 个角向 bin 的功率占比（各向同性≈0.111，越高越像射线）
- `residual_rms` = 误差幅度；`ray_index = 两者相乘`：既看方向性又看幅度，越低越好

**第 7 页 · 指标含义（二）：三方交叉比较**
- `cfm_vs_dense`：学习算子+离散化总误差（判伪影）
- `oracle_vs_dense`：真值算子的投影误差（下界参照）
- `cfm_vs_oracle`：纯学习算子残差（5.0% 就来自这里）
- `normalized_rel_l1`=平均绝对偏差；`log10_rmse`=对数误差，突出弱通量区；`corr`=形状相关

**第 8 页 · 探索过程（Agent 工作流）**
- 模式 B：粗基线 → DeepSeek 白名单决策 → 环境执行 → 反馈 → 再决策（3 轮）+ 3 随机参照
- 模式 C：LLM 提议 → 边界校验（拒绝/截断即自愈重提）→ 执行 → 反馈 → 再提议
- 图：`official_outputs/official_llm_flash/trajectory.jsonl`（事件链示意）

**第 9 页 · 发现信号（正结果）**
- 128 状态预算：`angular_refined` 射线指数 0.002501 vs `position_refined` 0.004431 —— **低 43.5%**
- 联合加密 `combined_refined` 局部 Exact-MC 指标较 coarse **低 57.3%**
- 图：`official_outputs/official_llm_flash/ablation_summary.png`

**第 10 页 · 发现信号（负结果与异常）**
- `position_refined` 只加位置节点反而变差（0.004431 > 0.003843）
- 联合加密 CFM 相对 Exact-MC 仍保留 5.0% L1 残差——学习算子误差未消除
- 开放探索两次运行提议不同：一次最佳 `proposal_p2_m2_f16`（+4.6%），一次最佳 `proposal_p2_m4_f32`（−33.6%），均未超过随机参照最优 0.002976——探索具有随机波动，需扩大边界

**第 11 页 · 参照系对比（排除随机波动）**
- 3 组随机参照指标落在 coarse/position 水平，随机最优 0.002095
- 43.5% 的角向优势不是种子波动

**第 12 页 · 推荐策略（本报告的核心交付）**
- 最省钱缓解：**`angular_refined`**（n_pos=1, n_mu=4, n_phi=32，128 状态）
- 证据路径：`official_outputs/official_llm_flash/experiments/angular_refined/`（metrics.json、diagnostics.json、png）

**第 13 页 · 复现情况**
- 一键 `setup_env` → 离线 `smoke_test` → `reproduce_core`（真实 API）
- 两次 flash + 一次 pro 独立复现，数字逐位一致；GTX 1660 SUPER 约 4 分钟、成本 <0.002 美元；无 GPU 退 CPU

**第 14 页 · 开源情况**
- 仓库 URL：**待填**；冻结 tag：**待填**
- MIT License；可复用件：环境包（gmc/ray_agent）、四类入口脚本、轨迹 JSONL 审计工具

**第 15 页 · 已知限制**
- 单种子、未做匹配多种子置信区间；射线指标是诊断代理量；二维 lattice 单几何；未证明对所有几何/材料普适

**第 16 页 · 指标速查（评审用）**
- ray_index 低 = 伪影弱；normalized_rel_l1 低 = 形状准；log10_rmse 低 = 暗区保真；corr 高 = 形态一致
- n_state = n_pos×n_mu×n_phi（预算）；wall_s = cfm_build_s + cfm_solve_s；cfm_residual = 求解器收敛（非物理误差）

**第 17 页 · 结语**
- 问题定义成立：MC 质量 ≠ 离散解质量，可证伪桥梁
- 发现可信：环境确定性 + 跨模型 + 随机参照三重印证
- 下一步：多种子置信区间、旋转角向集、CFM 方向平滑、混合源分离
