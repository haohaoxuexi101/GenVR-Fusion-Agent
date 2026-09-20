from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np

from gmc.method_taxonomy import get_method_taxonomy

from .metrics import finite_json, flux_shape_metrics, load_fluxes, select_reference_flux, spectral_ray_metrics
from .scientific import (
    attribute_artifacts,
    build_candidate_assessments,
    build_controlled_comparisons,
)


def _pct(before: float, after: float) -> float:
    if before == 0.0:
        return 0.0
    return 100.0 * (before - after) / before


def _limits(config: Any | None) -> dict[str, Any]:
    return {
        "learning_shape_l1": float(getattr(config, "learning_shape_l1_limit", 0.08)),
        "rotation_shape_l1": float(getattr(config, "rotation_shape_l1_limit", 0.03)),
        "angular_convergence_shape_l1": float(
            getattr(config, "angular_convergence_shape_l1_limit", 0.03)
        ),
        "repeatability_shape_l1": float(
            getattr(config, "repeatability_shape_l1_limit", 0.03)
        ),
        "global_mc_integral_rel": float(
            getattr(config, "global_mc_integral_rel_limit", 0.10)
        ),
        "require_global_mc_certification": bool(
            getattr(config, "require_global_mc_certification", False)
        ),
        "require_angular_convergence": bool(
            getattr(config, "require_angular_convergence", False)
        ),
        "require_repeatability_for_attribution": bool(
            getattr(config, "require_repeatability_for_attribution", False)
        ),
        "require_candidate_repeatability": bool(
            getattr(config, "require_candidate_repeatability", False)
        ),
    }


def _claim(
    case: str,
    attribution: dict[str, Any],
    recommendation: dict[str, Any] | None,
) -> str:
    conclusions = {
        "dense_sn_is_more_rotation_sensitive": (
            "Controlled azimuthal perturbations indicate that the finite-angle dense S_N branch is "
            "more orientation-sensitive than the CFM/interface branch."
        ),
        "shared_interface_projection_is_rotation_sensitive": (
            "CFM and projected local-MC respond similarly to interface-angle rotation, indicating "
            "a shared finite interface-projection artifact rather than a learned-kernel artifact."
        ),
        "learned_operator_is_more_rotation_sensitive": (
            "CFM is more orientation-sensitive than its matched projected local-MC calculation, "
            "indicating an additional learned-operator contribution."
        ),
        "rotation_sensitivity_not_cleanly_separated": (
            "The controlled rotations do not cleanly separate S_N, interface-projection, and learned-model effects."
        ),
        "insufficient_controlled_rotation_evidence": (
            "The run lacks enough matched rotation controls to assign the directional residual to one method."
        ),
        "insufficient_repeatability_evidence": (
            "The run has rotation controls but lacks a matched repeatability experiment needed to separate intervention sensitivity from sampling variation."
        ),
    }
    statement = conclusions[attribution["conclusion"]]
    if recommendation is None:
        decision = "No configuration is certified because the required robustness evidence or acceptance gates are incomplete."
    else:
        cfg = recommendation["configuration"]
        decision = (
            f"The cheapest non-dominated configuration passing the declared gates is "
            f"n_pos={cfg['n_pos']}, n_mu={cfg['n_mu']}, n_phi={cfg['n_phi']}."
        )
    return f"For the declared {case} experiment family: {statement} {decision}"


def _legacy_diagnostic_summary(
    experiments: list[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    baseline = experiments[0]
    final = next((item for item in experiments if "combined" in item["experiment_id"]), experiments[-1])
    baseline_cfm = baseline.get("cfm_ray_vs_dense", {}).get("ray_index")
    final_cfm = final.get("cfm_ray_vs_dense", {}).get("ray_index")
    baseline_projected = baseline.get("oracle_ray_vs_dense", {}).get("ray_index")
    final_projected = final.get("oracle_ray_vs_dense", {}).get("ray_index")
    dense_change = {
        "combined_cfm_percent": (
            _pct(float(baseline_cfm), float(final_cfm))
            if baseline_cfm is not None and final_cfm is not None
            else None
        ),
        "combined_projected_local_mc_percent": (
            _pct(float(baseline_projected), float(final_projected))
            if baseline_projected is not None and final_projected is not None
            else None
        ),
        "combined_oracle_percent": (
            _pct(float(baseline_projected), float(final_projected))
            if baseline_projected is not None and final_projected is not None
            else None
        ),
    }
    return {}, dense_change


def _plot_summary(
    run_dir: Path,
    experiments: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
) -> None:
    labels = [item["experiment_id"] for item in experiments]
    learning = [
        item.get("cfm_learning_error", {}).get("shape", {}).get("normalized_rel_l1", np.nan)
        for item in experiments
    ]
    disagreement = [
        item.get(
            "cfm_directional_disagreement_vs_dense_sn",
            item.get("cfm_ray_vs_dense", {}),
        ).get("ray_index", np.nan)
        for item in experiments
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    xpos = np.arange(len(labels))
    axes[0, 0].bar(xpos, learning, color="#4C78A8")
    axes[0, 0].set_ylabel("Shape L1")
    axes[0, 0].set_title("Learning error: CFM vs matched projected local-MC")
    axes[0, 0].set_xticks(xpos, [label.replace("_", "\n") for label in labels], fontsize=7)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(xpos, disagreement, color="#B5B5B5")
    axes[0, 1].set_ylabel("Directional residual index")
    axes[0, 1].set_title("CFM–S_N disagreement (diagnostic, not objective)")
    axes[0, 1].set_xticks(xpos, [label.replace("_", "\n") for label in labels], fontsize=7)
    axes[0, 1].grid(axis="y", alpha=0.25)

    controlled = [
        comparison
        for comparison in comparisons
        if comparison["comparison_type"]
        in {
            "interface_rotation",
            "dense_rotation",
            "interface_angular_refinement",
            "dense_angular_refinement",
            "repeatability",
        }
    ][:12]
    if controlled:
        control_labels = [
            f"{item['comparison_type']}\n{item['baseline_experiment']}→{item['intervention_experiment']}"
            for item in controlled
        ]
        control_x = np.arange(len(controlled))
        width = 0.25
        for offset, method, color in (
            (-width, "cfm", "#4C78A8"),
            (0.0, "projected_local_mc", "#F58518"),
            (width, "dense_sn", "#54A24B"),
        ):
            values = [
                item["method_sensitivity"].get(method, {}).get("shape", {}).get(
                    "normalized_rel_l1",
                    np.nan,
                )
                for item in controlled
            ]
            axes[1, 0].bar(
                control_x + offset,
                values,
                width,
                label=method.replace("_", " "),
                color=color,
            )
        axes[1, 0].set_xticks(control_x, control_labels, fontsize=6, rotation=20, ha="right")
        axes[1, 0].set_ylabel("Shape L1 under intervention")
        axes[1, 0].legend(fontsize=7)
    else:
        axes[1, 0].text(0.5, 0.5, "No matched controlled pairs", ha="center", va="center")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
    axes[1, 0].set_title("Controlled rotation, refinement, and repeatability tests")
    axes[1, 0].grid(axis="y", alpha=0.25)

    complete = [item for item in assessments if item["evidence_complete"]]
    if complete:
        colors = [float(item["learning_shape_l1"]) for item in complete]
        color_max = max(max(colors), 1.0e-12)
        for item, color in zip(complete, colors):
            marker = "o" if item["eligible"] else "x"
            axes[1, 1].scatter(
                item["process_elapsed_s"],
                item["interface_rotation_shape_l1"],
                c=[color],
                cmap="viridis",
                vmin=0.0,
                vmax=color_max,
                s=90,
                marker=marker,
            )
            axes[1, 1].annotate(
                item["candidate_id"],
                (item["process_elapsed_s"], item["interface_rotation_shape_l1"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
        axes[1, 1].set_xlabel("Observed process time [s]")
        axes[1, 1].set_ylabel("Interface-rotation shape L1")
        axes[1, 1].grid(alpha=0.25)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No candidate has complete rotation evidence",
            ha="center",
            va="center",
        )
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    axes[1, 1].set_title("Robustness–cost candidate map")
    fig.suptitle("GenVR numerical-artifact attribution agent")
    fig.savefig(run_dir / "ablation_summary.png", dpi=180)
    plt.close(fig)


def finalize_run(
    run_dir: Path,
    case: str,
    observations: list[dict[str, Any]],
    random_observations: list[dict[str, Any]],
    config: Any | None = None,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("at least one causal observation is required")

    final_observation = observations[-1]
    final_fluxes = load_fluxes(run_dir / final_observation["artifacts"]["fluxes"])
    common_method, common_reference = select_reference_flux(
        final_fluxes,
        preference=("projected_local_mc_response", "global_history_mc", "dense_sn_reference"),
    )
    common_name = f"{final_observation['experiment_id']}:{common_method}"

    def enrich(observation: dict[str, Any]) -> dict[str, Any]:
        fluxes = load_fluxes(run_dir / observation["artifacts"]["fluxes"])
        enriched = {
            **observation,
            "common_reference": common_name,
            "cfm_vs_common": flux_shape_metrics(fluxes["genvr_cfm_response"], common_reference),
            "cfm_ray_vs_common": spectral_ray_metrics(fluxes["genvr_cfm_response"], common_reference),
        }
        projected = fluxes["projected_local_mc_response"]
        if projected.size:
            enriched["oracle_vs_common"] = flux_shape_metrics(projected, common_reference)
            enriched["oracle_ray_vs_common"] = spectral_ray_metrics(projected, common_reference)
        return enriched

    causal = [enrich(observation) for observation in observations]
    random = [enrich(observation) for observation in random_observations]
    all_experiments = causal + random
    comparisons = build_controlled_comparisons(run_dir, all_experiments)
    limits = _limits(config)
    assessments, pareto_front, recommendation = build_candidate_assessments(
        causal,
        comparisons,
        limits,
    )
    attribution = attribute_artifacts(
        comparisons,
        require_repeatability=limits["require_repeatability_for_attribution"],
    )
    claim = _claim(case, attribution, recommendation)
    legacy_reductions, legacy_dense_reductions = _legacy_diagnostic_summary(causal)

    summary = {
        "schema_version": "3.0",
        "agent_objective": "causal_numerical_artifact_attribution_and_robust_discretization_selection",
        "method_taxonomy": get_method_taxonomy(
            [
                "genvr_cfm_response",
                "projected_local_mc_response",
                "dense_sn_reference",
                "global_history_mc",
            ]
        ),
        "scientific_claim": claim,
        "claim_scope": (
            "The declared two-dimensional benchmark family, controlled action set, checkpoints, "
            "sample counts, angular rotations, and random seeds only."
        ),
        "acceptance_limits": limits,
        "artifact_attribution": attribution,
        "controlled_comparisons": comparisons,
        "candidate_assessments": assessments,
        "pareto_front": pareto_front,
        "recommendation": recommendation,
        "causal_experiments": causal,
        "random_reference_experiments": random,
        "common_reference": common_name,
        "directional_disagreement_vs_fixed_dense": legacy_dense_reductions,
        "ray_index_reduction": legacy_reductions,
        "ray_index_reduction_vs_fixed_dense": legacy_dense_reductions,
        "limitations": [
            "A directional residual between two fields does not identify which field contains the artifact.",
            "Rotation sensitivity is a causal numerical diagnostic, not a universal physical observable.",
            "Projected local-MC shares the same finite interface phase space and isolates learned-kernel error only.",
            "Dense S_N is a finite-angle deterministic diagnostic and is not treated as continuum truth.",
            "Global history MC is a physical certification reference only at statistically supported tallies.",
            "A recommendation is withheld unless matched rotation evidence and all declared acceptance gates pass.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(finite_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_summary(run_dir, causal, comparisons, assessments)

    md = [
        "# GenVR numerical-artifact attribution report",
        "",
        f"**Scientific claim (scoped):** {claim}",
        "",
        "The CFM–S_N directional residual is reported only as cross-method disagreement; it is not optimized as an intrinsic CFM ray score.",
        "",
        "## Candidate evidence",
        "",
        "| Candidate | States | Learning L1 | Rotation L1 | Angular gap | Repeat L1 | MC integral error | Eligible | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for assessment in assessments:
        def value(name: str) -> str:
            item = assessment[name]
            return "n/a" if item is None else f"{float(item):.5g}"

        md.append(
            f"| {assessment['candidate_id']} | {assessment['n_state']} | "
            f"{value('learning_shape_l1')} | {value('interface_rotation_shape_l1')} | "
            f"{value('angular_convergence_shape_l1')} | {value('repeatability_shape_l1')} | "
            f"{value('global_mc_integral_relative_error')} | "
            f"{'yes' if assessment['eligible'] else 'no'} | {assessment['process_elapsed_s']:.3f} |"
        )
    md.extend(
        [
            "",
            "## Controlled comparisons",
            "",
            "| Type | Baseline | Intervention | CFM L1 | Projected-MC L1 | Dense S_N L1 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for comparison in comparisons:
        def sensitivity(method: str) -> str:
            value = comparison["method_sensitivity"].get(method, {}).get("shape", {}).get(
                "normalized_rel_l1"
            )
            return "n/a" if value is None else f"{float(value):.5g}"

        md.append(
            f"| {comparison['comparison_type']} | {comparison['baseline_experiment']} | "
            f"{comparison['intervention_experiment']} | {sensitivity('cfm')} | "
            f"{sensitivity('projected_local_mc')} | {sensitivity('dense_sn')} |"
        )
    md.extend(["", "## Limitations", ""])
    md.extend(f"- {item}" for item in summary["limitations"])
    md.extend(
        [
            "",
            "See `trajectory.jsonl` for every action, tool call, observation, and policy decision.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary
