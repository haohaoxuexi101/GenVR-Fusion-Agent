# 推荐结果：official_llm_pro

## 推荐策略（最省钱缓解）：`angular_refined`

angular_refined keeps the same 128-state budget and lowers the CFM-versus-dense ray index to 0.00250119 versus 0.00443073 for position_refined (43.5% lower). Recommended as the cheapest conservative mitigation.

- 指标文件：`G:/ssn_gmc_llmapi/official_outputs/official_llm_pro/experiments/angular_refined/metrics.json`
- 诊断文件：`G:/ssn_gmc_llmapi/official_outputs/official_llm_pro/experiments/angular_refined/diagnostics.json`
- 通量图：`G:/ssn_gmc_llmapi/official_outputs/official_llm_pro/experiments/angular_refined/lattice_stage8_ultra.png`

## 最干净参照（预算更高）：`combined_refined`

combined_refined reduces the local Exact-MC-versus-dense ray index to 0.000911311 versus 0.00213594 for coarse (57.3% lower), at the cost of a 512-state budget and a remaining 5.0% normalized L1 CFM-versus-Exact-MC residual.

- 指标文件：`G:/ssn_gmc_llmapi/official_outputs/official_llm_pro/experiments/combined_refined/metrics.json`
- 诊断文件：`G:/ssn_gmc_llmapi/official_outputs/official_llm_pro/experiments/combined_refined/diagnostics.json`
- 通量图：`G:/ssn_gmc_llmapi/official_outputs/official_llm_pro/experiments/combined_refined/lattice_stage8_ultra.png`

## 总推荐

Adopt angular_refined as the operational mitigation: angular_refined keeps the same 128-state budget and lowers the CFM-versus-dense ray index to 0.00250119 versus 0.00443073 for position_refined (43.5% lower). Recommended as the cheapest conservative mitigation.