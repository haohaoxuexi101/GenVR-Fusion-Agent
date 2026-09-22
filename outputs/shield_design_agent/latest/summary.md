# GenShield-Agent 运行摘要

- 运行 ID：`shield-20260918T064134Z`
- 扩散筛选最优风险：`0.073553`
- 最终验证级别：`exact-local-mc-response`
- 最终验证风险：`0.110285`
- 最终设计签名：`d7d960548ee7`
- 是否保留基准：`False`

## 验证链

- `genvr-cfm-response`：risk=`0.071075`，accepted=`True`；accepted d7d960548ee7: verified risk 0.0710747 < 1
- `exact-local-mc-response`：risk=`0.110285`，accepted=`True`；accepted d7d960548ee7: verified risk 0.110285 < 1

## 无偏 MC 认证

- 远端平均通量估计下降：`90.71%`
- 候选 95% 上界低于基准 95% 下界：`True`
- FOM 只用于评价验证效率，不是屏蔽优化目标。

## 文件

- `summary.json`：完整机器可读结果
- `best_design.json`：最终材料布局
- `trajectory.jsonl`：搜索与验证轨迹
- `shield_design_comparison.png`：共同色标设计对比
- `mc_certification.json` / `mc_certification.png`：无偏统计证据
