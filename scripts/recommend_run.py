#!/usr/bin/env python3
"""Emit an explicit recommendation for a completed ray-artifact run.

Reads <run-dir>/summary.json and writes <run-dir>/recommendation.json and
<run-dir>/recommendation.md with the recommended experiment (concrete
experiment_id plus the exact artifact paths to inspect).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json


def _fmt(path: Path) -> str:
    return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend the best experiment of a completed ablation run.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if str(summary.get("schema_version")) == "3.0":
        certified = summary.get("recommendation")
        recommendation = {
            "schema_version": "2.0",
            "source_schema_version": "3.0",
            "run_dir": _fmt(run_dir),
            "source": "summary.json",
            "selection_rule": (
                "Cheapest Pareto-nondominated candidate passing learning-fidelity, rotation, "
                "angular-convergence, repeatability, and global-MC gates."
            ),
            "artifact_attribution": summary.get("artifact_attribution"),
            "certified_candidate": certified,
            "recommended_experiment_id": (
                certified["representative_experiment"] if certified is not None else None
            ),
            "overall_recommendation": summary.get("scientific_claim"),
        }
        (run_dir / "recommendation.json").write_text(
            json.dumps(recommendation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [f"# 推荐结果：{run_dir.name}", "", summary.get("scientific_claim", "")]
        if certified is None:
            lines.extend(
                [
                    "",
                    "没有候选同时通过全部科学门槛，因此不做配置推荐。",
                ]
            )
        else:
            experiment_id = certified["representative_experiment"]
            lines.extend(
                [
                    "",
                    f"## 认证候选：`{certified['candidate_id']}`",
                    "",
                    f"- 配置：`{certified['configuration']}`",
                    f"- 证据实验：`{', '.join(certified['experiments'])}`",
                    f"- 指标文件：`experiments/{experiment_id}/metrics.json`",
                    f"- 诊断文件：`experiments/{experiment_id}/diagnostics.json`",
                    "- 选择依据：通过全部门槛后，在 Pareto 前沿中选择观测成本最低者。",
                ]
            )
        (run_dir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(recommendation, ensure_ascii=False, indent=2))
        return

    causal = summary.get("causal_experiments", [])
    by_id = {x["experiment_id"]: x for x in causal}

    recommendation: dict = {
        "run_dir": _fmt(run_dir),
        "source": "summary.json",
        "cheapest_mitigation": None,
        "cleanest_reference": None,
        "overall_recommendation": None,
    }
    notes: list[str] = []

    position = by_id.get("position_refined")
    angular = by_id.get("angular_refined")
    combined = by_id.get("combined_refined")
    coarse = by_id.get("coarse")

    if angular and position:
        pos_ray = position["cfm_ray_vs_dense"]["ray_index"]
        ang_ray = angular["cfm_ray_vs_dense"]["ray_index"]
        if ang_ray < pos_ray:
            picked, other, pct = angular, position, 100.0 * (pos_ray - ang_ray) / pos_ray
            reason = (
                f"angular_refined keeps the same 128-state budget and lowers the CFM-versus-dense ray index "
                f"to {ang_ray:.6g} versus {pos_ray:.6g} for position_refined ({pct:.1f}% lower). "
                "Recommended as the cheapest conservative mitigation."
            )
        else:
            picked, other, pct = position, angular, 100.0 * (ang_ray - pos_ray) / ang_ray
            reason = (
                f"position_refined achieves ray index {pos_ray:.6g} versus {ang_ray:.6g} for angular_refined "
                f"({pct:.1f}% lower). Recommended as the cheapest conservative mitigation."
            )
        recommendation["cheapest_mitigation"] = {
            "recommended_experiment_id": picked["experiment_id"],
            "reason": reason,
            "metrics_file": _fmt(run_dir / "experiments" / picked["experiment_id"] / "metrics.json"),
            "diagnostics_file": _fmt(run_dir / "experiments" / picked["experiment_id"] / "diagnostics.json"),
            "figure_file": _fmt(run_dir / "experiments" / picked["experiment_id"] / "lattice_stage8_ultra.png"),
        }

    if (
        combined
        and coarse
        and "oracle_ray_vs_dense" in combined
        and "oracle_ray_vs_dense" in coarse
        and "cfm_vs_oracle" in combined
    ):
        base_ray = coarse["oracle_ray_vs_dense"]["ray_index"]
        comb_ray = combined["oracle_ray_vs_dense"]["ray_index"]
        pct = 100.0 * (base_ray - comb_ray) / base_ray if base_ray else 0.0
        recommendation["cleanest_reference"] = {
            "recommended_experiment_id": combined["experiment_id"],
            "reason": (
                f"combined_refined reduces the projected local-MC-versus-dense ray index to {comb_ray:.6g} "
                f"versus {base_ray:.6g} for coarse ({pct:.1f}% lower), at the cost of a 512-state budget "
                f"and a remaining {100.0 * combined['cfm_vs_oracle']['normalized_rel_l1']:.1f}% "
                "normalized L1 CFM-versus-projected-local-MC residual."
            ),
            "metrics_file": _fmt(run_dir / "experiments" / combined["experiment_id"] / "metrics.json"),
            "diagnostics_file": _fmt(run_dir / "experiments" / combined["experiment_id"] / "diagnostics.json"),
            "figure_file": _fmt(run_dir / "experiments" / combined["experiment_id"] / "lattice_stage8_ultra.png"),
        }

    if recommendation["cheapest_mitigation"]:
        rec = recommendation["cheapest_mitigation"]
        recommendation["overall_recommendation"] = (
            f"Adopt {rec['recommended_experiment_id']} as the operational mitigation: {rec['reason']}"
        )
        notes.append(rec["reason"])
    if recommendation["cleanest_reference"]:
        notes.append(recommendation["cleanest_reference"]["reason"])

    (run_dir / "recommendation.json").write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 推荐结果：{run_dir.name}", ""]
    if recommendation["cheapest_mitigation"]:
        r = recommendation["cheapest_mitigation"]
        lines += [
            "## 推荐策略（最省钱缓解）：`" + r["recommended_experiment_id"] + "`",
            "",
            r["reason"],
            "",
            f"- 指标文件：`{r['metrics_file']}`",
            f"- 诊断文件：`{r['diagnostics_file']}`",
            f"- 通量图：`{r['figure_file']}`",
            "",
        ]
    if recommendation["cleanest_reference"]:
        r = recommendation["cleanest_reference"]
        lines += [
            "## 最干净参照（预算更高）：`" + r["recommended_experiment_id"] + "`",
            "",
            r["reason"],
            "",
            f"- 指标文件：`{r['metrics_file']}`",
            f"- 诊断文件：`{r['diagnostics_file']}`",
            f"- 通量图：`{r['figure_file']}`",
            "",
        ]
    if recommendation["overall_recommendation"]:
        lines += ["## 总推荐", "", recommendation["overall_recommendation"]]
    (run_dir / "recommendation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
