# GenVR 权窗与重要性减方差独立验证

## 科学问题

本验证在现有二维 Monte Carlo 输运核上加入三种彼此区分的无偏减方差目标：

1. **全场通量场均匀化**：直接读取仓库已有的 GenVR/CFM 粗通量场，令低通量区具有高重要性；评价时不把源区的大通量、低误差网格与深穿透网格混合平均，而以吸收体之后的远端区域为主指标；
2. **局部探测器响应**：对指定边界 detector 使用扩散伴随重要性，重点降低单一深穿透响应的方差。
3. **响应—吸收体双目标**：以 detector 伴随为全域基线，仅在强吸收材料内叠加 GenVR 反比通量重要性，在保留目标响应效率的同时显著增加吸收体内部历史数。

这三类目标不能混为一谈。`1/粗通量` 适合全场通量的相对误差均匀化，其误差和 FOM 必须在预声明的远端区域比较；特定 detector 的最优重要性还需要目标响应的伴随信息；双目标模式则明确接受少量 detector 效率损失，以换取强吸收体内部统计质量。

核心实现位于 `gmc/importance_transport.py`，独立入口为 `scripts/20_validate_importance_vr.py`，高分辨率绘图入口为 `scripts/21_plot_importance_vr_cloudmaps.py`。`scripts/22_build_hohlraum_genvr_adjoint.py` 可从右边界反向源生成真正的 GenVR 伴随代理；`scripts/23_compare_hohlraum_importance_strategies.py` 用于比较 detector-only 与 absorber-aware 两种策略。

## GenVR 粗通量到权窗

对经过重采样、正值截断和对数域平滑的 GenVR 粗通量 `phi_hat(x)`，基础重要性严格定义为：

```text
I_phi(x) = [phi_source / phi_hat(x)]^alpha
```

对应目标粒子权重：

```text
w_target(x) ∝ 1 / I_phi(x) ∝ phi_hat(x)^alpha
```

所以关系被明确锁定为：

- 粗估通量低 → 重要性大；
- 重要性大 → 目标粒子权重小；
- 当前粒子权重高于窗口上界 → splitting；
- 当前粒子权重低于窗口下界 → Russian roulette。

lattice 使用中心体源归一化，源区固定为 `w_target=1 / level=0`；四周低通量区由 GenVR 场自然获得较高重要性。强吸收材料额外乘入材料增益：

```text
I_total(x) = I_phi(x) * I_contrast(x) * 2^(b * A_strong(x))
```

其中 `b=absorber_boost_levels`，`A_strong` 是由 `Sigma_a / Sigma_t` 判定的强吸收区掩膜。该乘法保留吸收块内部由 GenVR 给出的通量梯度，而不是把整块材料压成同一个 level；因此粒子会随粗通量继续降低而逐级分裂。可选入口 halo 也改为每向外一层降低一级的渐变 halo，避免大片普通介质被无差别推到最高重要性。对于 lattice 的全场通量目标，默认不把 all-boundary detector 强制钉在最高 level。

对于局部 detector，可选乘入响应因子：

```text
I_total(x) = I_phi(x) * max[I_response(x) / I_response(source), 1]^beta
```

`max(..., 1)` 保证响应项只能继续增大重要性，不能把低通量区的重要性压低。

## 无偏性

权窗只改变粒子数和单粒子权重，不改变物理输运核。若当前权重 `w` 高于目标权重，则分裂为 `m` 个后代：

```text
w_child = w / m
```

后代总权重严格等于父权重。若 `w` 低于窗口下界，则以 `w / w_target` 的概率存活，存活权重提升为 `w_target`，条件期望仍为 `w`。

每个源粒子的全部后代贡献先聚合为一个根历史样本。对 detector 响应为标量：

```text
Y_i = sum(weight of every detector-crossing descendant of root i)
```

最终方差只对独立的 `Y_i` 计算，不能把相关 clone 当成独立样本。

对全场 track-length 通量，则每个网格 `c` 都有独立的根历史样本：

```text
Y_i,c = sum(track-length contribution of every descendant of root i in cell c)
R_c = sqrt[s_c² / N] / |mean(Y_i,c)|
FOM_c = 1 / (R_c² * T_transport)
```

`T_transport` 只计 `run_detector_mc` 输运阶段，因此包含 splitting / roulette 的运行开销，但不包含离线生成 GenVR 场、读取文件和编译权窗的预处理成本。正式汇总报告远端网格上的 `median(R_c)` 与成对逐格 `median(FOM_VR,c / FOM_A,c)`；总泄漏标量只用于无偏性和权重守恒审计。

### 可视化口径（2026-09-19 修订）

- Analog、GenVR/CFM、固定 Dense `S_N` 和权窗 MC 通量图使用同一绝对 `log10(phi)` 色标；不能再通过各自归一化后的颜色直接比较绝对通量。
- 与固定 Dense `S_N` 的局部误差图比较的是各自峰值归一化后的**形状误差**；Dense `S_N` 只是有限角度确定性参照，不是连续输运真值。
- 全场目标用绿色轮廓标记预声明远场评价区，不再把全边界泄漏 tally 画成主要 detector；边界响应目标仍用红色 detector 标记。
- FOM 云图只显示两种方法均有有限误差且至少有两个非零贡献根历史的网格；灰色表示该图不提供可靠的逐格 FOM 比较。标题中的覆盖率仍单独报告全部远场网格的非零计分覆盖。
- 材料底图改为 `Sigma_a / Sigma_t`，直接对应代码中的强吸收判据；它不等价于完整光学厚度，仍需结合几何尺寸和 `Sigma_t` 解读。

## 复现

### Lattice：GenVR 全场通量权窗，112×112

```bash
MPLCONFIGDIR=/tmp/mplconfig conda run -n openmc-env \
  python scripts/20_validate_importance_vr.py \
  --case lattice --objective full-field-flux --nx 112 --ny 112 \
  --histories 5000 --replicates 4 \
  --importance-kind genvr-flux \
  --levels 14 --flux-exponent 0.75 \
  --outdir outputs/importance_vr_validation/highres/lattice_112_full_field_genvr

MPLCONFIGDIR=/tmp/mplconfig conda run -n openmc-env \
  python scripts/21_plot_importance_vr_cloudmaps.py \
  --case lattice \
  --input-dir outputs/importance_vr_validation/highres/lattice_112_full_field_genvr \
  --dpi 300
```

`alpha=0.75` 现已是 lattice 全场模式的自动默认值；level 自动推导为 13。上面的 L14 是参数筛选时使用的保守上限，但该场实际最高仅到 level 13，因此 L13/L14 生成同一运行权窗。全场模式关闭 detector pin、单向单调约束、额外 absorber boost、contrast heuristic 和平坦 halo，只让 GenVR 通量本身决定目标权重。绘图命令同时生成通量、远端不确定度与人口控制图。

默认读取开放探索中最佳 LLM 提案的 CFM 通量：

`official_outputs/open_explore_lattice_112/experiments/proposal_p2_m4_f32/lattice_stage8_ultra.npz` 中的 `cfm` 字段；同文件的 `dense` 字段只作为确定性形状参照，不宣称是精确连续解。

lattice 的中心体源位于 `3 cm <= x,y < 4 cm`。远端主区域定义为网格中心到该源盒几何距离最高的四分位，即最远的 `3,080` 个网格；其中独立 `dense` 形状参照通量最低的四分位（`770` 个网格）作为更严格的深穿透子区域。这样 GenVR `cfm` 只用于生成权窗，不参与定义自己的评价区域。中心源区和近端高通量区不参与主误差与主 FOM 汇总。

在 `56×56`、`4,000` 根历史的顺序参数筛选中，`alpha=0.70/L13`、`0.75/L14`、`0.80/L15` 的远端成对逐格 FOM 中位增益分别为 `6.56×`、`7.08×`、`6.95×`。因此正式计算选用 `alpha=0.75/L14`，而不是盲目使用完整 `alpha=1` 动态范围。

### Hohlraum：GenVR 全场通量权窗，112×112

```bash
MPLCONFIGDIR=/tmp/mplconfig conda run -n openmc-env \
  python scripts/20_validate_importance_vr.py \
  --case hohlraum --objective full-field-flux \
  --nx 112 --ny 112 --histories 2500 --replicates 2 \
  --outdir outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr

MPLCONFIGDIR=/tmp/mplconfig conda run -n openmc-env \
  python scripts/21_plot_importance_vr_cloudmaps.py \
  --case hohlraum \
  --input-dir outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr \
  --dpi 300
```

该模式直接使用 `outputs/hohlraum/hohlraum_stage8_ultra.npz` 的 `cfm` 场生成 MAGIC 型反比通量权窗，自动关闭 detector pin、单向单调约束、额外 absorber boost 和 contrast heuristic。对于左边界源，主评价区预定义为吸收体之后的下游内部区域：

```text
0.65W <= x < 0.96W, 0.04H <= y < 0.96H
```

远端内按同一 NPZ 的独立 `dense` 形状参照取最低通量四分位，作为更严格的深穿透子区域。GenVR `cfm` 只用于生成权窗，不用于定义自己的评价区域。源区和近端高通量区仍显示在全场云图中，但不参与主误差与主 FOM 的汇总。

### Hohlraum：右边界深穿透响应，112×112（不同科学目标）

```bash
MPLCONFIGDIR=/tmp/mplconfig conda run -n openmc-env \
  python scripts/20_validate_importance_vr.py \
  --case hohlraum --nx 112 --ny 112 \
  --histories 5000 --replicates 4 \
  --importance-kind diffusion --levels 10 --split-factor 2 \
  --outdir outputs/importance_vr_validation/highres/hohlraum_112_diffusion_L10
```

这里保留扩散伴随作为正式权窗，因为评价目标是单一右边界响应，而不是全域通量图。正向 GenVR 通量的倒数适合全场相对误差均衡，但不是该单一响应的伴随重要性；可重复生成的反向 GenVR 伴随代理与扩散伴随的对数场相关系数达到 `0.99415`、线性相关系数达到 `0.99959`，因此当前 detector-only 正式结果采用更简单、稳定且无需额外模型推理的扩散 L10。

若同时要求中央强吸收体有足够统计，可运行双目标模式：

```bash
MPLCONFIGDIR=/tmp/mplconfig conda run -n openmc-env \
  python scripts/20_validate_importance_vr.py \
  --case hohlraum --nx 112 --ny 112 \
  --histories 5000 --replicates 4 \
  --importance-kind response-absorber \
  --levels 10 --flux-exponent 0.20 \
  --absorber-peak-fraction 0.0625 \
  --outdir outputs/importance_vr_validation/highres/hohlraum_112_response_absorber_L10

MPLCONFIGDIR=/tmp/mplconfig conda run -n openmc-env \
  python scripts/23_compare_hohlraum_importance_strategies.py
```

双目标场使用：

```text
I_hybrid(x) = max[I_response(x), c * I_reciprocal_flux(x)]
```

其中第二项只作用于 `Sigma_a / Sigma_t >= 0.8` 的强吸收单元；`c=0.0625` 控制吸收体峰值相对 detector 重要性的强度。这样不会把整个普通介质无差别抬高，也不会破坏 splitting / roulette 的无偏性。

## 2026-09-17 正式结果

### Lattice：GenVR 全场通量，源外远端为主评价区

本结果使用 `112×112` 网格和 `20,000` 个独立根历史（`4×5,000`）。中心源区保持 level 0，配置上限为 L14，实际运行权窗最高为 level 13；基础映射满足：

```text
I(x) ∝ phi_GenVR(x)^(-0.75)
```

Analog 输运墙钟时间为 `24.247 s`，GenVR 权窗输运为 `64.078 s`，时间比为 `2.64×`；所有远端 FOM 均已计入这部分人口控制开销，但不含离线 GenVR 场生成与权窗预处理成本。

| 远端主指标 | Analog MC | GenVR Weight Window |
|---|---:|---:|
| 有独立根历史贡献的网格覆盖率 | 100% | 100% |
| 网格中位相对误差 | 66.64% | 10.49% |
| 网格中位贡献根历史数 | 59 | 181 |

- `3,080` 个远端网格中，`100%` 的网格相对误差下降；
- 远端成对逐格 FOM 中位增益为 `14.30×`；用区域中位误差计算的 FOM 增益为 `15.27×`；
- 远端最低通量四分位的成对逐格 FOM 中位增益为 `20.96×`，区域中位误差 FOM 增益为 `22.66×`；
- 权窗平均输运 `14.62` 个粒子/根历史，其中 `13.62` 个为分裂子代；未发生 split-cap 截断；
- 强吸收材料访问量达到 analog 的 `2.36×`，`30.26%` 的分裂子代产生在强吸收材料内，但最高 level 只覆盖 `0.59%` 网格，未把整块材料粗暴压到同一最高重要性；
- 总泄漏估计为 `0.03409` 与 `0.03296`，仅相差 `0.842` 个联合标准差；VR 权重平衡为 `0.99981`。因此 GenVR 只指导无偏 splitting/roulette，不替代 Monte Carlo 计分。

相对同一 `dense` 通量形状参照：

| 区域 | Analog normalized L1 | GenVR-WW normalized L1 | 误差降低 |
|---|---:|---:|---:|
| 全域 | 0.0964 | 0.0740 | 23.20% |
| 强吸收单元 | 0.2109 | 0.2047 | 2.90% |
| 深吸收四分位 | 0.7196 | 0.4800 | 33.29% |
| 最低通量四分位 | 0.7735 | 0.2164 | 72.03% |

证据：

- `outputs/importance_vr_validation/highres/lattice_112_full_field_genvr/comparison.json`
- `outputs/importance_vr_validation/highres/lattice_112_full_field_genvr/comparison.npz`
- `outputs/importance_vr_validation/highres/lattice_112_full_field_genvr/comparison.png`
- `outputs/importance_vr_validation/highres/lattice_112_full_field_genvr/far_field_uncertainty_maps.png`
- `outputs/importance_vr_validation/highres/lattice_112_full_field_genvr/flux_uncertainty_maps.png`
- `outputs/importance_vr_validation/highres/lattice_112_full_field_genvr/flux_cloudmaps.png`
- `outputs/importance_vr_validation/highres/lattice_112_full_field_genvr/population_control_maps.png`

### Hohlraum：GenVR 全场通量，远端为主评价区

本结果使用 `112×112` 网格和 `5,000` 个独立根历史（`2×2,500`）。Analog 输运墙钟时间为 `8.638 s`，GenVR 权窗输运为 `128.060 s`，时间比为 `14.83×`；所有 FOM 均已计入这部分人口控制开销，但不含离线 GenVR 场生成与权窗预处理成本。

| 远端主指标 | Analog MC | GenVR Weight Window |
|---|---:|---:|
| 有独立根历史贡献的网格覆盖率 | 94.81% | 100% |
| 网格中位相对误差 | 99.52% | 10.80% |
| 网格中位贡献根历史数 | 3 | 190 |

- 在两种方法均可比较的 `3,451` 个远端网格中，`100%` 的网格相对误差下降；
- 远端成对逐格 FOM 中位增益为 `5.20×`；用区域中位误差计算的 FOM 增益为 `5.72×`；
- 远端最低通量四分位的成对逐格 FOM 中位增益为 `6.36×`，区域中位误差 FOM 增益为 `6.74×`；
- 总泄漏标量 FOM 为 `0.07×`，但它衡量的是全边界泄漏，不是本次“远端全场通量”优化目标，只保留作守恒辅助诊断。

证据：

- `outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr/comparison.json`
- `outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr/comparison.npz`
- `outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr/comparison.png`
- `outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr/far_field_uncertainty_maps.png`
- `outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr/flux_uncertainty_maps.png`
- `outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr/flux_cloudmaps.png`
- `outputs/importance_vr_validation/highres/hohlraum_112_full_field_genvr/population_control_maps.png`

### Hohlraum：右边界扩散伴随重要性 L10（不同科学目标）

20,000 个独立根历史：

| 指标 | Analog MC | Importance VR |
|---|---:|---:|
| detector estimate | 3.5453e-6 | 3.3228e-6 |
| standard error | 1.8680e-6 | 1.8000e-7 |
| relative error | 52.69% | 5.42% |
| contributing roots | 74 | 1,761 |
| root ESS | 3.60 | 335.07 |
| runtime | 20.84 s | 107.65 s |
| FOM | 0.173 | 3.165 |

- 根历史方差降低 `107.69×`，相对方差降低 `94.60×`；
- 墙钟 FOM 提升 `18.32×`；
- 两估计值仅相差 `0.119` 个联合标准差；
- 每根历史平均产生 `14.71` 个子代，未发生 split-cap 截断；
- 上下绕流通道访问量达到 analog 的 `13.42×`，中央吸收体后方区域达到 `65.15×`；
- 中央强吸收体访问量保持为 analog 的 `0.95×`，仅 `15` 个子代在其中产生。对右边界响应而言，这不是“分裂不足”，而是伴随重要性主动避开几乎不贡献响应的死亡路径。

56×56、8,000 根历史的 level 扫描进一步说明不能只追求更多分裂：

| Levels | 子代/根历史 | 相对方差降低 | FOM 提升 |
|---:|---:|---:|---:|
| 8 | 7.45 | 35.10× | 11.71× |
| 10 | 16.92 | 81.82× | **16.25×** |
| 12 | 51.21 | 141.50× | 13.76× |

L12 虽继续降低方差，但粒子人口增长过快，墙钟效率反而低于 L10。因此 L10 是当前“统计质量—计算成本”的甜点，而不是简单把最高 level 无限增大。

证据：

- `outputs/importance_vr_validation/highres/hohlraum_112_diffusion_L10/comparison.json`
- `outputs/importance_vr_validation/highres/hohlraum_112_diffusion_L10/comparison.npz`
- `outputs/importance_vr_validation/highres/hohlraum_112_diffusion_L10/flux_cloudmaps.png`
- `outputs/importance_vr_validation/highres/hohlraum_112_diffusion_L10/population_control_maps.png`

### Hohlraum：响应—吸收体双目标 L10（不同科学目标）

用户若不仅关心右边界标量响应，还要求中央蓝色强吸收体内具备足够的粒子统计，应采用这一模式。GenVR 正向粗通量仅在强吸收材料内提供反比通量增强，外部空间仍由右边界伴随控制。

20,000 个独立根历史的正式结果：

| 指标 | Detector-only L10 | 双目标 L10 |
|---|---:|---:|
| 中央吸收体平均 level | 0.002 | 6.77 |
| 中央吸收体 level 范围 | 0–1 | 5–8 |
| 中央吸收体访问量 / analog | 0.95× | 77.68× |
| 吸收体内分裂子代占比 | 0.005% | 33.44% |
| 上下绕流通道访问量 / analog | 13.42× | 13.61× |
| 子代 / 根历史 | 14.71 | 22.26 |
| 右边界相对误差 | 5.42% | 5.55% |
| 相对方差降低 | 94.60× | 90.20× |
| FOM 提升 | 18.32× | 15.04× |

结论是：中央重要性可以显著提高，而且不会破坏右边界响应估计；代价是 detector-only FOM 下降约 `18%`。这不是一个方案取代另一个，而是两个清晰的优化目标：

- 只追求右边界深穿透响应：使用 `diffusion L10`；
- 同时要求中央吸收体/极低通量区统计：使用 `response-absorber L10`。

双目标模式相对同一 `dense` 场参照，使深吸收四分位误差降低 `31.12%`，最低通量四分位误差降低 `71.87%`。

证据：

- `outputs/importance_vr_validation/highres/hohlraum_112_response_absorber_L10/comparison.json`
- `outputs/importance_vr_validation/highres/hohlraum_112_response_absorber_L10/comparison.npz`
- `outputs/importance_vr_validation/highres/hohlraum_112_response_absorber_L10/flux_cloudmaps.png`
- `outputs/importance_vr_validation/highres/hohlraum_112_response_absorber_L10/population_control_maps.png`
- `outputs/importance_vr_validation/highres/hohlraum_importance_strategy_comparison.png`
- `outputs/importance_vr_validation/hohlraum_reverse_genvr/adjoint_compare.png`

## 审计检查

- Transparent benchmark 上，analog 与 VR 的根历史 detector score 守恒；
- GenVR 基础重要性单测明确检查“低通量 → 高重要性 → 低目标权重”；
- source region、perimeter、吸收区平均 level 均写入或可由 `comparison.npz` 复算；
- `absorbed_weight + weighted_leakage` 用于权重守恒审计；
- 记录每个网格的访问次数、分裂事件、分裂子代、分裂上限命中和轮盘死亡；
- 2026-09-17：`conda run -n openmc-develop python -m pytest -q`，`28 passed`。

## 与 Agent 的接口

Agent 不应直接修改粒子权重，而应输出可审计的权窗提案：

- 选择 GenVR 通量文件和字段；
- 选择 `flux_exponent`、level cap、窗口宽度；
- 选择局部凹陷尺度、吸收材料额外 level 和渐变入口 halo；
- 对局部响应决定是否加入伴随响应项；
- 对多目标问题选择 detector-only 或 response-absorber，并控制 `absorber_peak_fraction`；
- 通过无偏性、权重平衡、根历史方差、区域通量误差和墙钟 FOM 验证后才能接受。

主要接口为 `WeightWindowMap.target_weights`、`WeightWindowMap.levels` 和 `run_detector_mc(..., weight_windows=...)`。物理输运核无需改变。
