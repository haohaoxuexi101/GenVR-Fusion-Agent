# GenVR 数值方法与基准口径

截至 2026-09-18，仓库中曾同时使用 `baseline`、`reference`、`oracle`、`exact-MC`、Dense `S_N`、GMC 和 diffusion 等词。它们不在同一层级。本文给出唯一推荐口径；历史 JSON/NPZ 字段继续兼容，但不再把任何有限离散或有限样本结果称为“精确真值”。

## 一、先区分三件事

1. **物理 benchmark**：`lattice`、`hohlraum`、ITER 几何定义的是材料、源和边界条件，不是数值真值。
2. **数值方法/参照**：CFM、投影局部 MC、Dense `S_N`、全局历史 MC、扩散分别求解不同近似层级。
3. **实验 baseline**：`coarse`、`position_refined`、`angular_refined`、`combined_refined` 是相空间配置消融；屏蔽设计中的 baseline 则是原始几何。二者都不是一种输运算法。

## 二、方法层级

| 规范 ID | 历史字段 | 实际计算 | 主要用途 | 不能宣称什么 |
|---|---|---|---|---|
| `genvr_cfm_response` | `cfm` | CFM 生成局部出射样本，投影为有限界面响应算子，再做确定性全局求解 | 被评估的 GenVR 方法 | 不是全局粒子 MC，也不是无偏真值 |
| `projected_local_mc_response` | `oracle` | 每个有限界面状态用随机飞行 MC 估计局部算子，再用与 CFM 相同的全局离散求解器组合 | 隔离“神经核误差”和“共同投影误差” | 不是 `Exact-MC`；仍有局部采样误差和相空间离散偏差 |
| `dense_sn_reference` | `dense` | 固定空间网格与有限角度求积上的 diamond-difference `S_N` 扫描，并对负出射角通量施加守恒 zero-flux fixup | 稳定、无抽样空格的有限网格诊断 | 不是连续输运方程精确解，仍需空间与角度收敛检查 |
| `global_history_mc` | `mc` | 从源到终止跟踪完整粒子历史；当前实现采用隐式俘获与 track-length tally | 独立随机输运参照 | 有有限历史统计误差，且权重截断可引入极小偏差 |
| `global_cell_response_mc` | 函数名含 `oracle` | 每次跨单元直接采样连续随机飞行局部响应，并继续完整全局历史 | 检验 GMC 组合逻辑而不引入神经网络误差 | 仍是有限历史 MC，不是解析真值 |
| `gmc_particle_transport` | `GMC` | 用学习到的单元出射分布推进完整粒子，必要时回退直接 MC | 加速型随机输运候选 | 可能含学习模型偏差 |
| `diffusion_proxy` | diffusion | 扩散方程正向/伴随近似 | 屏蔽候选快速筛选、重要性初值 | 不适合认证深穿透流线、空腔与强各向异性输运 |
| `importance_vr_mc` | VR MC | 在直接输运中做守权分裂和轮盘 | 降低远端/全场 tally 方差 | 只有守权和无偏性检验通过后才能与普通 MC 比较 |

机器可读版本位于 `gmc/method_taxonomy.py`，新 `metrics.json` 会嵌入同一份 `method_taxonomy`。

## 三、为什么旧 `Exact-MC` 名称不准确

Stage-8 的历史 `oracle` 不是“从源到探测终止的无限精度 Monte Carlo”。它先对每个局部入射状态做有限次随机飞行，再把出射结果投影到有限的 `n_pos × n_mu × n_phi` 界面状态，最后由确定性响应方程组合全局场。因此它适合回答：

- 在**相同界面离散**下，CFM 学习核比直接局部 MC 核多了多少误差？
- 增加界面角向/位置自由度后，共同的投影伪影是否下降？

它不能回答“连续输运真值是多少”。新图、日志和指标统一称为 **Projected local-MC response**；`oracle` 仅作为旧产物兼容键保留。

## 四、当前最佳 CFM 检查点

主 benchmark 现在默认使用：

| 模型 | 路径 | 训练/验证样本 | 最佳 epoch | 验证损失 | SHA1 前 12 位 | 路径参数化 |
|---|---|---:|---:|---:|---|---|
| Boundary | `outputs/hpc_v2_models/boundary_900k_100ep/best.pt` | 900k / 100k | 100 | 0.7213677764 | `5f816a4032cd` | `excess_chord` |
| Internal | `outputs/hpc_v2_models/internal_900k_100ep/best.pt` | 900k / 100k | 94 | 0.5778295398 | `2e0c41f7562b` | `excess_chord` |

旧默认 `models/boundary_stage9_120k_30ep.pt` 和 `models/internal_stage9_ep9.pt` 使用 legacy `absolute` 路径参数化，其中内部模型只训练到 epoch 9。它们仍可用于历史复核，但 Stage-8 默认拒绝加载；必须显式传入 `--allow-legacy-checkpoints` 才能运行。

每次新运行的 `metrics.json` 会记录检查点路径、SHA1、类型、epoch、验证损失、训练/验证样本数、网络结构和 `path_mode`，避免“文件名叫 best，但实际加载了什么无法确认”。

## 五、推荐验证阶梯

1. **单元核验证**：比较 CFM 与连续随机飞行局部样本，检查概率守恒、出射分布、轨长和吸收。
2. **同投影对照**：比较 `genvr_cfm_response` 与 `projected_local_mc_response`，归因学习核误差。
3. **离散化对照**：比较二者与固定 `dense_sn_reference`，诊断有限方向/位置投影造成的射线伪影。
4. **独立全局复核**：用多批 `global_history_mc` 报告均值、方差和置信区间，而不是单次场图。
5. **深穿透效率验证**：在同一源历史预算、同一 tally 区域下比较普通 MC 与 `importance_vr_mc` 的偏差、相对误差和 FOM。
6. **屏蔽设计认证**：diffusion 只负责廉价搜索；候选必须再经过输运级求解和独立统计验证。

## 六、新旧产物兼容

新 NPZ 同时写入规范键和历史键：

```text
genvr_cfm_response              <-> cfm
projected_local_mc_response     <-> oracle
dense_sn_reference              <-> dense
global_history_mc               <-> mc
```

`ray_agent.metrics.load_fluxes` 会双向解析并检查重复别名是否一致、非空通量是否为二维且形状相同。历史正式运行目录不被重写，以保留原始 provenance。
