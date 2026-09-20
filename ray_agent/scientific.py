from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import comparison_metrics, load_fluxes


METHOD_FIELDS = {
    "cfm": "genvr_cfm_response",
    "projected_local_mc": "projected_local_mc_response",
    "dense_sn": "dense_sn_reference",
    "global_mc": "global_history_mc",
}

CONTROL_LABELS = {
    "interface_phi_offset_fraction": "interface_rotation",
    "dense_phi_offset_fraction": "dense_rotation",
    "n_phi": "interface_angular_refinement",
    "n_mu": "interface_polar_refinement",
    "n_pos": "interface_position_refinement",
    "dense_n_phi": "dense_angular_refinement",
    "dense_n_mu": "dense_polar_refinement",
    "cfm_samples": "cfm_sampling_refinement",
    "oracle_samples": "projected_mc_sampling_refinement",
    "internal_samples": "cfm_internal_sampling_refinement",
    "oracle_internal_samples": "projected_mc_internal_sampling_refinement",
    "mc_histories": "global_mc_sampling_refinement",
    "seed": "repeatability",
}


def experiment_parameters(observation: dict[str, Any]) -> dict[str, Any]:
    action = observation.get("action", {})
    return {
        "n_pos": int(action.get("n_pos", 0)),
        "n_mu": int(action.get("n_mu", 0)),
        "n_phi": int(action.get("n_phi", 0)),
        "interface_phi_offset_fraction": float(
            action.get(
                "interface_phi_offset_fraction",
                observation.get("interface_phi_offset_fraction", 0.0),
            )
        ),
        "dense_n_mu": int(observation.get("dense_n_mu", action.get("dense_n_mu") or 0)),
        "dense_n_phi": int(observation.get("dense_n_phi", action.get("dense_n_phi") or 0)),
        "dense_phi_offset_fraction": float(
            action.get(
                "dense_phi_offset_fraction",
                observation.get("dense_phi_offset_fraction", 0.0),
            )
        ),
        "cfm_samples": int(action.get("cfm_samples", 0)),
        "oracle_samples": int(action.get("oracle_samples", 0)),
        "internal_samples": int(action.get("internal_samples", 0)),
        "oracle_internal_samples": int(action.get("oracle_internal_samples", 0)),
        "mc_histories": int(action.get("mc_histories") or 0),
        "seed": int(action.get("seed", 0)),
        "cfm_mode": str(action.get("cfm_mode", "direct")),
    }


def candidate_key(observation: dict[str, Any]) -> tuple[Any, ...]:
    params = experiment_parameters(observation)
    return (
        params["n_pos"],
        params["n_mu"],
        params["n_phi"],
        params["cfm_mode"],
        params["cfm_samples"],
        params["oracle_samples"],
        params["internal_samples"],
        params["oracle_internal_samples"],
    )


def candidate_id(observation: dict[str, Any]) -> str:
    params = experiment_parameters(observation)
    return (
        f"p{params['n_pos']}_m{params['n_mu']}_f{params['n_phi']}_"
        f"{params['cfm_mode']}"
    )


def _different_controls(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    differences = []
    for name in first:
        left = first[name]
        right = second[name]
        if isinstance(left, float) or isinstance(right, float):
            if not np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-12):
                differences.append(name)
        elif left != right:
            differences.append(name)
    return differences


def _ordered_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    factor: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first_value = experiment_parameters(first)[factor]
    second_value = experiment_parameters(second)[factor]
    if factor == "seed":
        return (first, second) if first_value <= second_value else (second, first)
    return (first, second) if float(first_value) <= float(second_value) else (second, first)


def build_controlled_comparisons(
    run_dir: Path,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flux_cache: dict[str, dict[str, np.ndarray]] = {}

    def fluxes(observation: dict[str, Any]) -> dict[str, np.ndarray]:
        experiment_id = str(observation["experiment_id"])
        if experiment_id not in flux_cache:
            flux_cache[experiment_id] = load_fluxes(
                run_dir / observation["artifacts"]["fluxes"]
            )
        return flux_cache[experiment_id]

    comparisons: list[dict[str, Any]] = []
    for first_index, first in enumerate(observations):
        first_params = experiment_parameters(first)
        for second in observations[first_index + 1 :]:
            second_params = experiment_parameters(second)
            differences = _different_controls(first_params, second_params)
            if len(differences) != 1 or differences[0] not in CONTROL_LABELS:
                continue
            factor = differences[0]
            baseline, intervention = _ordered_pair(first, second, factor)
            baseline_fluxes = fluxes(baseline)
            intervention_fluxes = fluxes(intervention)
            method_metrics = {}
            for method, field in METHOD_FIELDS.items():
                baseline_field = baseline_fluxes[field]
                intervention_field = intervention_fluxes[field]
                if baseline_field.size and intervention_field.size:
                    method_metrics[method] = comparison_metrics(
                        intervention_field,
                        baseline_field,
                    )
            comparisons.append(
                {
                    "comparison_type": CONTROL_LABELS[factor],
                    "controlled_factor": factor,
                    "baseline_experiment": baseline["experiment_id"],
                    "intervention_experiment": intervention["experiment_id"],
                    "baseline_value": experiment_parameters(baseline)[factor],
                    "intervention_value": experiment_parameters(intervention)[factor],
                    "method_sensitivity": method_metrics,
                }
            )
    return comparisons


def _comparison_values(
    comparisons: list[dict[str, Any]],
    comparison_type: str,
    method: str,
    candidate_experiments: set[str] | None = None,
) -> list[float]:
    values = []
    for comparison in comparisons:
        if comparison["comparison_type"] != comparison_type:
            continue
        if candidate_experiments is not None and not {
            comparison["baseline_experiment"],
            comparison["intervention_experiment"],
        } <= candidate_experiments:
            continue
        metrics = comparison["method_sensitivity"].get(method)
        if metrics is not None:
            values.append(float(metrics["shape"]["normalized_rel_l1"]))
    return values


def attribute_artifacts(
    comparisons: list[dict[str, Any]],
    require_repeatability: bool = False,
) -> dict[str, Any]:
    sn_rotation = _comparison_values(comparisons, "dense_rotation", "dense_sn")
    cfm_rotation = _comparison_values(comparisons, "interface_rotation", "cfm")
    projected_rotation = _comparison_values(
        comparisons,
        "interface_rotation",
        "projected_local_mc",
    )
    cfm_repeatability = _comparison_values(comparisons, "repeatability", "cfm")
    projected_repeatability = _comparison_values(
        comparisons,
        "repeatability",
        "projected_local_mc",
    )

    evidence = {
        "dense_sn_rotation_shape_l1": max(sn_rotation, default=None),
        "cfm_interface_rotation_shape_l1": max(cfm_rotation, default=None),
        "projected_mc_interface_rotation_shape_l1": max(projected_rotation, default=None),
        "cfm_repeatability_shape_l1": max(cfm_repeatability, default=None),
        "projected_mc_repeatability_shape_l1": max(projected_repeatability, default=None),
    }
    available = [value for value in evidence.values() if value is not None]
    if not sn_rotation or not cfm_rotation:
        conclusion = "insufficient_controlled_rotation_evidence"
    elif require_repeatability and not (cfm_repeatability or projected_repeatability):
        conclusion = "insufficient_repeatability_evidence"
    else:
        noise_floor = max(cfm_repeatability + projected_repeatability, default=0.0)
        sn_signal = max(sn_rotation)
        cfm_signal = max(cfm_rotation)
        projected_signal = max(projected_rotation, default=0.0)
        competing_signal = max(cfm_signal, projected_signal, noise_floor, 1.0e-12)
        if sn_signal > 1.5 * competing_signal:
            conclusion = "dense_sn_is_more_rotation_sensitive"
        elif projected_signal > 1.5 * max(noise_floor, 1.0e-12) and cfm_signal <= 1.5 * projected_signal:
            conclusion = "shared_interface_projection_is_rotation_sensitive"
        elif cfm_signal > 1.5 * max(projected_signal, noise_floor, 1.0e-12):
            conclusion = "learned_operator_is_more_rotation_sensitive"
        else:
            conclusion = "rotation_sensitivity_not_cleanly_separated"
    return {
        "conclusion": conclusion,
        "evidence": evidence,
        "controlled_rotation_measurements": len(sn_rotation) + len(cfm_rotation),
        "maximum_observed_shape_l1": max(available, default=None),
    }


def _pareto_frontier(rows: list[dict[str, Any]], objectives: tuple[str, ...]) -> list[str]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if row is other:
                continue
            no_worse = all(float(other[name]) <= float(row[name]) for name in objectives)
            strictly_better = any(float(other[name]) < float(row[name]) for name in objectives)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(row["candidate_id"]))
    return frontier


def build_candidate_assessments(
    observations: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    limits: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[candidate_key(observation)].append(observation)

    assessments = []
    for group in grouped.values():
        representative = min(
            group,
            key=lambda item: (
                abs(experiment_parameters(item)["interface_phi_offset_fraction"]),
                experiment_parameters(item)["seed"],
            ),
        )
        experiment_ids = {str(item["experiment_id"]) for item in group}
        learning = representative.get("cfm_learning_error", {}).get("shape", {}).get(
            "normalized_rel_l1"
        )
        physical_observation = next(
            (item for item in group if "cfm_vs_global_mc" in item),
            representative,
        )
        physical_ratio = physical_observation.get("cfm_vs_global_mc", {}).get("level", {}).get(
            "integral_ratio"
        )
        rotation_values = _comparison_values(
            comparisons,
            "interface_rotation",
            "cfm",
            experiment_ids,
        )
        projected_rotation_values = _comparison_values(
            comparisons,
            "interface_rotation",
            "projected_local_mc",
            experiment_ids,
        )
        repeatability_values = _comparison_values(
            comparisons,
            "repeatability",
            "cfm",
            experiment_ids,
        )
        angular_values = []
        for comparison in comparisons:
            if comparison["comparison_type"] != "interface_angular_refinement":
                continue
            if experiment_ids.isdisjoint(
                {
                    comparison["baseline_experiment"],
                    comparison["intervention_experiment"],
                }
            ):
                continue
            metrics = comparison["method_sensitivity"].get("cfm")
            if metrics is not None:
                angular_values.append(float(metrics["shape"]["normalized_rel_l1"]))

        rotation = max(rotation_values, default=None)
        projected_rotation = max(projected_rotation_values, default=None)
        angular = min(angular_values, default=None)
        repeatability = max(repeatability_values, default=None)
        physical_error = abs(float(physical_ratio) - 1.0) if physical_ratio is not None else None
        evidence_complete = learning is not None and rotation is not None
        passes = {
            "learning_fidelity": learning is not None and learning <= limits["learning_shape_l1"],
            "rotation_robustness": rotation is not None and rotation <= limits["rotation_shape_l1"],
            "angular_convergence": (
                angular is not None and angular <= limits["angular_convergence_shape_l1"]
                if limits.get("require_angular_convergence", False)
                else angular is None or angular <= limits["angular_convergence_shape_l1"]
            ),
            "repeatability": (
                repeatability is not None and repeatability <= limits["repeatability_shape_l1"]
                if limits.get("require_candidate_repeatability", False)
                else repeatability is None or repeatability <= limits["repeatability_shape_l1"]
            ),
            "global_mc_integral": (
                physical_error is not None and physical_error <= limits["global_mc_integral_rel"]
                if limits.get("require_global_mc_certification", False)
                else physical_error is None or physical_error <= limits["global_mc_integral_rel"]
            ),
        }
        dense_disagreement = representative.get(
            "cfm_directional_disagreement_vs_dense_sn",
            representative.get("cfm_ray_vs_dense", {}),
        ).get("ray_index")
        assessments.append(
            {
                "candidate_id": candidate_id(representative),
                "representative_experiment": representative["experiment_id"],
                "experiments": sorted(experiment_ids),
                "configuration": {
                    key: experiment_parameters(representative)[key]
                    for key in ("n_pos", "n_mu", "n_phi", "cfm_mode")
                },
                "n_state": int(representative["n_state"]),
                "process_elapsed_s": float(
                    representative.get("process_elapsed_s", representative.get("wall_s", 0.0))
                ),
                "learning_shape_l1": learning,
                "interface_rotation_shape_l1": rotation,
                "projected_mc_rotation_shape_l1": projected_rotation,
                "angular_convergence_shape_l1": angular,
                "repeatability_shape_l1": repeatability,
                "global_mc_integral_relative_error": physical_error,
                "directional_disagreement_vs_dense_sn": dense_disagreement,
                "evidence_complete": evidence_complete,
                "passes": passes,
                "eligible": evidence_complete and all(passes.values()),
            }
        )

    eligible = [row for row in assessments if row["eligible"]]
    objectives = ("learning_shape_l1", "interface_rotation_shape_l1", "process_elapsed_s")
    frontier = _pareto_frontier(eligible, objectives)
    frontier_rows = [row for row in eligible if row["candidate_id"] in frontier]
    recommendation = min(frontier_rows, key=lambda row: row["process_elapsed_s"], default=None)
    return assessments, frontier, recommendation
