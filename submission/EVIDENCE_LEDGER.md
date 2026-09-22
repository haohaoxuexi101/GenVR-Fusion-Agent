# GenShield-Agent 证据台账

当前口径日期：2026-09-20。本台账只登记当前 `gmc_only` 正式运行能够支持或不能支持的声明。

## 1. 内部实测证据

| ID | 证据 | 支持的陈述 | 限制 |
|---|---|---|---|
| I1 | `run_manifest.json` | 正式策略为 `DeepSeekShieldToolAgent`，`llm_required=true`，无确定性回退，启动后无人工干预 | 不能单独证明每次 API 决策质量 |
| I2 | `trajectory.jsonl` 中 17 个 `llm_exchange` | 真实 API 参与 12 步工具决策；17 个唯一 `response_id`，总计 233,244 tokens | 单次运行，不能估计跨模型稳定性 |
| I3 | `discovery_progress` 与候选谱系 | 生成 59 个结构，GMC 筛选 16 个，12 个 GMC 引导后代完成复筛 | 不是随机策略的配对胜率实验 |
| I4 | `gmc_screening/manifest.json` | GMC 风险覆盖 0.07593–1.09295，正负候选均保留 | GMC 是学习响应近似，不是最终真值 |
| I5 | 候选 `d7d960548ee7` 的 GMC 场 | 近源右侧/下侧材料迁移对应显著降低的远场综合风险 | 结构规律尚未跨 seed 复现 |
| I6 | `mc/01_d7d960548ee7/certification.json` | 权窗 MC 远场候选/基准为 0.109986，差异 z=-50.802，95% 区间分离 | 基准 Analog/权窗一致性未通过 |
| I7 | 第 10、12 步 `verifier_decision` | Verifier 拒绝过早成功声明，并最终接受 `inconclusive` | 证明门槛有效，不证明候选无改进 |
| I8 | 生成脚本与 `THIRD_PARTY_NOTICES.md` | 训练、benchmark、GMC 和 MC 数据均为项目自产，无外部训练数据集 | 大型中间产物仍需随包提供或重新生成 |

## 2. 可以支持的声明

- API 确实参与了父代、算子、筛选批次、MC 候选和停止决策；
- GMC 结果确实改变了后续结构生成；
- GMC 场确实被转换为候选专属 MC 权窗；
- 固定 benchmark 内已形成完整的自主提出—测量—再生成—认证闭环；
- 候选 `d7d960548ee7` 存在值得追加 MC 复验的强远场下降信号；
- 独立 Verifier 阻止了未满足全部门槛的成功声明。

## 3. 不能支持的声明

- 候选 `d7d960548ee7` 已通过最终物理认证；
- 89% 下降可以直接外推到真实聚变堆或三维工程装置；
- 当前 GMC 综合风险权重是通用核工程标准；
- Agent 已经统计稳定地优于随机搜索或人工专家；
- 单次正式运行足以证明跨模型、跨 seed 的稳定成功率；
- GMC 或权窗 MC 是无误差真值。

## 4. 最终声明等级

- **Agent 闭环完备性**：已支持。
- **候选物理改进信号**：强信号。
- **最终材料发现认证**：`inconclusive`。

公开仓库：https://github.com/haohaoxuexi101/GenVR-Fusion-Agent  
代码基线：tag `v0.1.0`，commit `ca885735e09cb46a6afba29c1a47f19d0fd8eda1`。
