# 复赛提交材料索引

当前口径日期：2026-09-20。唯一正式科学主线为 `gmc_only` 的 GenShield-Agent：受约束结构生成 → GMC 筛选与反馈再生成 → GMC 权窗 MC → 独立 Verifier。旧射线效应归因材料仅为历史探索，不得作为本版结论。

| 要求 | 当前交付 | 状态 |
|---|---|---|
| 1. 项目展示 PPT | 尚未生成 | 待办 |
| 2. 科学发现与环境定义报告 PDF | `../output/pdf/GenShield-Agent_科学发现与环境定义报告.pdf` | 完成 |
| 3. 最小可运行探索环境 | `../environment.yml`、`../requirements.txt`、`../configs/gmc_material_discovery_agent.json`、`../scripts/run_gmc_material_discovery_agent.sh` | 完成 |
| 4. 完整 JSON/JSONL 日志 | `../outputs/gmc_material_discovery/run_20260920_020815/` | 完成 |
| 5. Reference Benchmark | `REFERENCE_BENCHMARK_DESIGN.md`：固定基准 `ded6840cea70`、完整候选分布及 Analog/权窗 MC 对照 | 完成；多 Agent seed 仍为局限 |
| 6. 公开代码仓库与 README | https://github.com/haohaoxuexi101/GenVR-Fusion-Agent、根目录 `README.md`、MIT License、测试与重绘入口 | 完成 |

## 当前报告

- Markdown 源稿：`SCIENTIFIC_FINDING_AND_ENVIRONMENT.md`
- 可复现构建器：`../scripts/27_build_scientific_finding_report.py`
- 正式 PDF：`../output/pdf/GenShield-Agent_科学发现与环境定义报告.pdf`
- 旧文件名兼容副本：`../output/pdf/GenVR-Fusion_科学发现与环境定义报告.pdf`
- API 参与审计图：`assets/api_participation_audit.png`

## 正式证据入口

- 运行清单：`../outputs/gmc_material_discovery/run_20260920_020815/run_manifest.json`
- 完整轨迹：`../outputs/gmc_material_discovery/run_20260920_020815/trajectory.jsonl`
- 机器可读总结：`../outputs/gmc_material_discovery/run_20260920_020815/summary.json`
- GMC 候选清单：`../outputs/gmc_material_discovery/run_20260920_020815/gmc_screening/manifest.json`
- MC 认证：`../outputs/gmc_material_discovery/run_20260920_020815/mc/01_d7d960548ee7/certification.json`
- 发现总览：`../outputs/gmc_material_discovery/run_20260920_020815/gmc_discovery_storyboard.png`
- MC 图：`../outputs/gmc_material_discovery/run_20260920_020815/mc_certification.png`
- GMC 路演动图：`../outputs/gmc_material_discovery/run_20260920_020815/gmc_screening_reel.gif`

## 正式运行口径

- 真实 DeepSeek API 决策 12 步，17 次 API 交换，17 个唯一 `response_id`；
- 生成 59 个结构，GMC 筛选 16 个，其中 12 个为 GMC 引导后代；
- MC 根历史预算 80,000/80,000；
- 候选 `d7d960548ee7` 显示远场下降强信号；
- 基准 Analog/权窗一致性 `z=-2.1477` 未通过冻结门槛；
- Verifier 最终接受的声明为 `inconclusive`。

## 开源与数据

- 队伍名称：小手震
- 公开仓库：https://github.com/haohaoxuexi101/GenVR-Fusion-Agent
- 冻结 tag：`v0.1.0`
- 冻结 commit：`ca885735e09cb46a6afba29c1a47f19d0fd8eda1`
- 数据来源：全部由项目代码生成，不使用外部训练数据集。
- 生成流程：`scripts/01_generate_boundary_data.py`、`scripts/08_generate_internal_data.py`、`gmc/benchmarks2d.py`、`gmc/mc_cell.py`。
- 数据交付：正式运行日志、通量场和 MC 认证产物随报告与日志提交包提供；大型检查点和响应缓存为可再生成的自产中间产物。

## 提交前仍需填写

- 参赛队号：`OPENxxx`
- 最终 PPT：待生成

## 文件名建议

- PPT：`OPENxxx_GenShield-Agent_项目展示.pptx`
- 报告：`OPENxxx_GenShield-Agent_科学发现与环境定义报告.pdf`
- 报告与日志包：`OPENxxx_GenShield-Agent_submission.zip`

历史文件 `OPENxxx_RayLab-GMC_报告与日志_非PDF版.zip` 与 `bundle_manifest.json` 对应旧射线效应提交，不应直接用于本版提交。公开材料不得包含 API Key、访问令牌或未确认可再分发的数据与权重。
