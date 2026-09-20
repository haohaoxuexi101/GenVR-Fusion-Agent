#!/usr/bin/env python3
"""Compare analog MC with importance splitting on an existing 2-D benchmark."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gmc.benchmarks2d import make_hohlraum_problem, make_lattice_problem
from gmc.importance_transport import (
    BoundaryDetector,
    combine_response_and_absorber_importance,
    DetectorTransportResult,
    make_adjoint_importance_map,
    make_all_boundary_importance_map,
    make_diffusion_adjoint_importance_map,
    make_inverse_flux_weight_windows,
    make_reciprocal_flux_importance,
    make_right_boundary_importance_map,
    prepare_coarse_flux_field,
    run_detector_mc,
    solve_diffusion_adjoint_importance,
)


DEFAULT_GENVR_FLUX_FILES = {
    "lattice": ROOT
    / "official_outputs/open_explore_lattice_112/experiments/"
    "proposal_p2_m4_f32/lattice_stage8_ultra.npz",
    "hohlraum": ROOT / "outputs/hohlraum/hohlraum_stage8_ultra.npz",
}

DEFAULT_ADJOINT_FILES = {
    "hohlraum": ROOT
    / "outputs/importance_vr_validation/hohlraum_reverse_genvr/adjoint_proxy.npz",
}


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _flux_shape_summary(
    candidate: np.ndarray,
    reference: np.ndarray,
    problem,
) -> dict[str, object]:
    candidate_scale = max(float(np.max(np.abs(candidate))), 1.0e-300)
    reference_scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    normalized_candidate = candidate / candidate_scale
    normalized_reference = reference / reference_scale
    correlation = float(
        np.corrcoef(normalized_candidate.ravel(), normalized_reference.ravel())[0, 1]
    )
    low_flux_mask = reference <= float(np.quantile(reference, 0.25))
    absorber_mask = problem.sigma_a > 0.0
    deep_absorber_mask = np.zeros(reference.shape, dtype=bool)
    if np.any(absorber_mask):
        absorber_threshold = float(np.quantile(reference[absorber_mask], 0.25))
        deep_absorber_mask = absorber_mask & (reference <= absorber_threshold)
    masks = {
        "all_cells": np.ones(reference.shape, dtype=bool),
        "absorber_cells": absorber_mask,
        "deep_absorber_quartile": deep_absorber_mask,
        "lowest_flux_quartile": low_flux_mask,
    }
    regions: dict[str, object] = {}
    for name, mask in masks.items():
        denominator = max(
            float(np.sum(np.abs(normalized_reference[mask]))),
            1.0e-300,
        )
        regions[name] = {
            "normalized_rel_l1": float(
                np.sum(
                    np.abs(normalized_candidate[mask] - normalized_reference[mask])
                )
                / denominator
            ),
            "zero_fraction": float(np.mean(candidate[mask] == 0.0)),
        }
    positive_mask = (normalized_candidate > 0.0) | (normalized_reference > 0.0)
    log_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.log10(np.maximum(normalized_candidate[positive_mask], 1.0e-15))
                    - np.log10(np.maximum(normalized_reference[positive_mask], 1.0e-15))
                )
                ** 2
            )
        )
    )
    return {
        "correlation": correlation,
        "log10_rmse": log_rmse,
        "regions": regions,
    }


def _error_reduction_percent(
    baseline: dict[str, object],
    candidate: dict[str, object],
    region: str,
) -> float:
    baseline_regions = baseline["regions"]
    candidate_regions = candidate["regions"]
    baseline_error = float(baseline_regions[region]["normalized_rel_l1"])
    candidate_error = float(candidate_regions[region]["normalized_rel_l1"])
    if baseline_error <= 0.0:
        return 0.0
    return 100.0 * (baseline_error - candidate_error) / baseline_error


def _run_summary(result: DetectorTransportResult, runtime_s: float) -> dict[str, object]:
    relative_error = result.relative_error
    fom = 0.0
    if math.isfinite(relative_error) and relative_error > 0.0 and runtime_s > 0.0:
        fom = 1.0 / (relative_error * relative_error * runtime_s)
    return {
        "estimate": result.estimate,
        "sample_variance": result.sample_variance,
        "standard_error": result.standard_error,
        "relative_error": _finite(relative_error),
        "ci95": [
            result.estimate - 1.96 * result.standard_error,
            result.estimate + 1.96 * result.standard_error,
        ],
        "runtime_s": runtime_s,
        "fom": fom,
        "histories": result.histories,
        "steps": result.steps,
        "transported_particles": result.transported_particles,
        "detector_crossings": result.detector_crossings,
        "contributing_roots": result.contributing_roots,
        "zero_score_fraction": result.zero_score_fraction,
        "root_ess": result.root_ess,
        "max_root_fraction": result.max_root_fraction,
        "weighted_leakage": result.weighted_leakage,
        "absorbed_weight": result.absorbed_weight,
        "track_length": result.track_length,
        "split_events": result.split_events,
        "split_children": result.split_children,
        "split_cap_hits": result.split_cap_hits,
        "roulette_survivals": result.roulette_survivals,
        "roulette_kills": result.roulette_kills,
        "cutoff_survivals": result.cutoff_survivals,
        "cutoff_kills": result.cutoff_kills,
        "max_bank_size": result.max_bank_size,
    }


def _combine(results: list[DetectorTransportResult]) -> np.ndarray:
    return np.concatenate([result.root_scores for result in results])


def _aggregate(results: list[DetectorTransportResult], runtimes: list[float]) -> dict[str, object]:
    scores = _combine(results)
    histories = len(scores)
    estimate = float(np.mean(scores))
    variance = float(np.var(scores, ddof=1)) if histories > 1 else float("nan")
    standard_error = float(np.sqrt(variance / histories)) if histories > 1 else float("nan")
    relative_error = standard_error / abs(estimate) if estimate != 0.0 else float("inf")
    runtime_s = float(np.sum(runtimes))
    fom = 0.0
    if math.isfinite(relative_error) and relative_error > 0.0 and runtime_s > 0.0:
        fom = 1.0 / (relative_error * relative_error * runtime_s)
    denominator = float(np.dot(scores, scores))
    root_ess = float(np.sum(scores) ** 2 / denominator) if denominator > 0.0 else 0.0
    total_score = float(np.sum(scores))
    weighted_leakage = float(np.mean([result.weighted_leakage for result in results]))
    absorbed_weight = float(np.mean([result.absorbed_weight for result in results]))
    return {
        "estimate": estimate,
        "sample_variance": variance,
        "standard_error": standard_error,
        "relative_error": _finite(relative_error),
        "ci95": [estimate - 1.96 * standard_error, estimate + 1.96 * standard_error],
        "runtime_s": runtime_s,
        "fom": fom,
        "histories": histories,
        "steps": int(sum(result.steps for result in results)),
        "transported_particles": int(sum(result.transported_particles for result in results)),
        "split_events": int(sum(result.split_events for result in results)),
        "split_children": int(sum(result.split_children for result in results)),
        "split_cap_hits": int(sum(result.split_cap_hits for result in results)),
        "roulette_survivals": int(sum(result.roulette_survivals for result in results)),
        "roulette_kills": int(sum(result.roulette_kills for result in results)),
        "cutoff_survivals": int(sum(result.cutoff_survivals for result in results)),
        "cutoff_kills": int(sum(result.cutoff_kills for result in results)),
        "max_bank_size": int(max(result.max_bank_size for result in results)),
        "detector_crossings": int(sum(result.detector_crossings for result in results)),
        "contributing_roots": int(np.count_nonzero(scores)),
        "zero_score_fraction": float(np.mean(scores == 0.0)),
        "root_ess": root_ess,
        "max_root_fraction": float(np.max(scores) / total_score) if total_score > 0.0 else 0.0,
        "weighted_leakage": weighted_leakage,
        "absorbed_weight": absorbed_weight,
        "weight_balance": weighted_leakage + absorbed_weight,
        "replicate_estimates": [result.estimate for result in results],
        "replicate_runtimes_s": runtimes,
    }


def _aggregate_flux_statistics(
    results: list[DetectorTransportResult],
) -> dict[str, np.ndarray | int]:
    histories = int(sum(result.histories for result in results))
    score_sum = np.sum([result.flux_score_sum for result in results], axis=0)
    score_sum_sq = np.sum([result.flux_score_sum_sq for result in results], axis=0)
    contributing_roots = np.sum(
        [result.flux_contributing_roots for result in results],
        axis=0,
    )
    mean = score_sum / float(histories)
    if histories > 1:
        centered_sum_sq = score_sum_sq - score_sum * score_sum / float(histories)
        sample_variance = np.maximum(centered_sum_sq / float(histories - 1), 0.0)
        standard_error = np.sqrt(sample_variance / float(histories))
    else:
        sample_variance = np.full_like(mean, np.nan)
        standard_error = np.full_like(mean, np.nan)
    relative_error = np.full_like(mean, np.inf)
    np.divide(
        standard_error,
        np.abs(mean),
        out=relative_error,
        where=mean != 0.0,
    )
    return {
        "histories": histories,
        "mean": mean,
        "score_sum": score_sum,
        "score_sum_sq": score_sum_sq,
        "contributing_roots": contributing_roots,
        "sample_variance": sample_variance,
        "standard_error": standard_error,
        "relative_error": relative_error,
    }


def _full_field_masks(
    problem,
    guide_flux: np.ndarray,
    strong_absorber_ratio: float,
) -> dict[str, np.ndarray]:
    sigma_total = problem.sigma_a + problem.sigma_s
    absorption_ratio = np.divide(
        problem.sigma_a,
        sigma_total,
        out=np.zeros_like(problem.sigma_a),
        where=sigma_total > 0.0,
    )
    strong_absorber = absorption_ratio >= strong_absorber_ratio
    low_flux = guide_flux <= float(np.quantile(guide_flux, 0.25))
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)
    if problem.source_kind == "left_boundary":
        far_field = (
            (x_grid >= 0.65 * problem.width)
            & (x_grid < 0.96 * problem.width)
            & (y_grid >= 0.04 * problem.height)
            & (y_grid < 0.96 * problem.height)
        )
    elif problem.source_kind == "volume_box" and problem.source_box is not None:
        xmin, xmax, ymin, ymax = problem.source_box
        delta_x = np.maximum.reduce((xmin - x_grid, np.zeros_like(x_grid), x_grid - xmax))
        delta_y = np.maximum.reduce((ymin - y_grid, np.zeros_like(y_grid), y_grid - ymax))
        source_distance = np.hypot(delta_x, delta_y)
        non_source = source_distance > 0.0
        distance_threshold = float(np.quantile(source_distance[non_source], 0.75))
        far_field = source_distance >= distance_threshold
    else:
        raise ValueError(f"unsupported source definition for far-field masks: {problem.source_kind}")
    far_field_low_flux = np.zeros_like(far_field)
    if np.any(far_field):
        far_field_flux_threshold = float(np.quantile(guide_flux[far_field], 0.25))
        far_field_low_flux = far_field & (guide_flux <= far_field_flux_threshold)
    deep_absorber = np.zeros_like(strong_absorber)
    if np.any(strong_absorber):
        threshold = float(np.quantile(guide_flux[strong_absorber], 0.25))
        deep_absorber = strong_absorber & (guide_flux <= threshold)
    return {
        "far_field_cells": far_field,
        "far_field_low_flux_quartile": far_field_low_flux,
        "all_cells": np.ones_like(strong_absorber),
        "strong_absorber_cells": strong_absorber,
        "deep_absorber_quartile": deep_absorber,
        "lowest_flux_quartile": low_flux,
    }


def _full_field_summary(
    statistics: dict[str, np.ndarray | int],
    runtime_s: float,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    mean = np.asarray(statistics["mean"], dtype=np.float64)
    relative_error = np.asarray(statistics["relative_error"], dtype=np.float64)
    contributing_roots = np.asarray(
        statistics["contributing_roots"],
        dtype=np.int64,
    )
    regions: dict[str, object] = {}
    for name, mask in masks.items():
        cells = int(np.count_nonzero(mask))
        valid = mask & np.isfinite(relative_error) & (mean != 0.0)
        values = relative_error[valid]
        if values.size:
            region_fom = 1.0 / np.maximum(values * values * runtime_s, 1.0e-300)
            median_relative_error = float(np.median(values))
            p90_relative_error = float(np.quantile(values, 0.90))
            p95_relative_error = float(np.quantile(values, 0.95))
            median_fom = float(np.median(region_fom))
        else:
            median_relative_error = None
            p90_relative_error = None
            p95_relative_error = None
            median_fom = None
        regions[name] = {
            "cells": cells,
            "scored_cell_fraction": (
                float(np.mean(contributing_roots[mask] > 0)) if cells else None
            ),
            "finite_relative_error_fraction": (
                float(np.count_nonzero(valid) / cells) if cells else None
            ),
            "median_relative_error": median_relative_error,
            "p90_relative_error": p90_relative_error,
            "p95_relative_error": p95_relative_error,
            "median_contributing_roots": (
                float(np.median(contributing_roots[mask])) if cells else None
            ),
            "median_fom": median_fom,
        }
    return {
        "histories": int(statistics["histories"]),
        "runtime_s": runtime_s,
        "regions": regions,
    }


def _compare_full_field_statistics(
    analog_statistics: dict[str, np.ndarray | int],
    vr_statistics: dict[str, np.ndarray | int],
    analog_runtime_s: float,
    vr_runtime_s: float,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    analog_error = np.asarray(analog_statistics["relative_error"], dtype=np.float64)
    vr_error = np.asarray(vr_statistics["relative_error"], dtype=np.float64)
    comparison: dict[str, object] = {}
    for name, mask in masks.items():
        valid = (
            mask
            & np.isfinite(analog_error)
            & np.isfinite(vr_error)
            & (analog_error > 0.0)
            & (vr_error > 0.0)
        )
        if not np.any(valid):
            comparison[name] = {
                "comparable_cells": 0,
                "fraction_cells_with_lower_relative_error": None,
                "median_cellwise_relative_error_ratio_analog_over_vr": None,
                "median_cellwise_fom_gain": None,
            }
            continue
        relative_error_ratio = analog_error[valid] / vr_error[valid]
        cellwise_fom_gain = (
            analog_error[valid] ** 2 * analog_runtime_s
        ) / (vr_error[valid] ** 2 * vr_runtime_s)
        analog_region_error = float(np.median(analog_error[valid]))
        vr_region_error = float(np.median(vr_error[valid]))
        comparison[name] = {
            "comparable_cells": int(np.count_nonzero(valid)),
            "fraction_cells_with_lower_relative_error": float(
                np.mean(vr_error[valid] < analog_error[valid])
            ),
            "median_cellwise_relative_error_ratio_analog_over_vr": float(
                np.median(relative_error_ratio)
            ),
            "p10_cellwise_relative_error_ratio_analog_over_vr": float(
                np.quantile(relative_error_ratio, 0.10)
            ),
            "median_cellwise_fom_gain": float(np.median(cellwise_fom_gain)),
            "region_median_relative_error_fom_gain": float(
                (analog_region_error / vr_region_error) ** 2
                * analog_runtime_s
                / vr_runtime_s
            ),
        }
    return comparison


def _plot(
    outpath: Path,
    problem,
    importance_levels: np.ndarray,
    analog_flux_statistics: dict[str, np.ndarray | int],
    vr_flux_statistics: dict[str, np.ndarray | int],
    analog_scores: np.ndarray,
    vr_scores: np.ndarray,
    analog_summary: dict[str, object],
    vr_summary: dict[str, object],
    objective: str,
    far_field_mask: np.ndarray,
) -> None:
    analog_flux = np.asarray(analog_flux_statistics["mean"], dtype=np.float64)
    vr_flux = np.asarray(vr_flux_statistics["mean"], dtype=np.float64)
    positive_flux = np.concatenate(
        [analog_flux[analog_flux > 0.0], vr_flux[vr_flux > 0.0]]
    )
    flux_floor = max(float(np.quantile(positive_flux, 0.005)), 1.0e-300)
    log_floor = float(np.log10(flux_floor))
    log_ceiling = float(np.log10(np.max(positive_flux)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    extent = [0.0, problem.width, 0.0, problem.height]
    im0 = axes[0, 0].imshow(
        np.log10(np.maximum(analog_flux, flux_floor)),
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=log_floor,
        vmax=log_ceiling,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("Analog MC log10 track-length flux")
    fig.colorbar(
        im0,
        ax=axes[0, 0],
        label="$\\log_{10}(\\phi)$ (shared with VR)",
        extend="min",
    )
    axes[0, 1].imshow(
        np.log10(np.maximum(vr_flux, flux_floor)),
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=log_floor,
        vmax=log_ceiling,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 1].set_title("Importance VR log10 track-length flux")
    im2 = axes[0, 2].imshow(
        importance_levels,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 2].set_title("Cellwise importance levels")
    fig.colorbar(im2, ax=axes[0, 2], label="importance level")
    for axis in axes[0]:
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")

    methods = ["analog", "importance"]
    if objective == "full-field-flux":
        far_field_label = (
            "Downstream far field"
            if problem.source_kind == "left_boundary"
            else "Source-distant far field"
        )
        analog_error = np.asarray(
            analog_flux_statistics["relative_error"],
            dtype=np.float64,
        )
        vr_error = np.asarray(
            vr_flux_statistics["relative_error"],
            dtype=np.float64,
        )
        analog_valid = (
            far_field_mask & np.isfinite(analog_error) & (analog_error > 0.0)
        )
        vr_valid = far_field_mask & np.isfinite(vr_error) & (vr_error > 0.0)
        common = analog_valid & vr_valid
        paired = bool(np.any(common))
        analog_metric_mask = common if paired else analog_valid
        vr_metric_mask = common if paired else vr_valid

        if np.any(analog_valid):
            axes[1, 0].hist(
                np.log10(analog_error[analog_valid]),
                bins=30,
                alpha=0.65,
                label="analog",
            )
        if np.any(vr_valid):
            axes[1, 0].hist(
                np.log10(vr_error[vr_valid]),
                bins=30,
                alpha=0.65,
                label="importance",
            )
        axes[1, 0].set_title(f"{far_field_label} cell errors")
        axes[1, 0].set_xlabel("log10(cell relative error)")
        if np.any(analog_valid) or np.any(vr_valid):
            axes[1, 0].legend()

        median_errors = [
            (
                float(np.median(analog_error[analog_metric_mask]))
                if np.any(analog_metric_mask)
                else 0.0
            ),
            (
                float(np.median(vr_error[vr_metric_mask]))
                if np.any(vr_metric_mask)
                else 0.0
            ),
        ]
        axes[1, 1].bar(methods, median_errors)
        axes[1, 1].set_title("Far-field median relative error")

        analog_runtime = float(analog_summary["runtime_s"])
        vr_runtime = float(vr_summary["runtime_s"])
        analog_cell_fom = (
            1.0 / (analog_error[analog_metric_mask] ** 2 * analog_runtime)
            if np.any(analog_metric_mask)
            else np.empty(0, dtype=np.float64)
        )
        vr_cell_fom = (
            1.0 / (vr_error[vr_metric_mask] ** 2 * vr_runtime)
            if np.any(vr_metric_mask)
            else np.empty(0, dtype=np.float64)
        )
        median_foms = [
            float(np.median(analog_cell_fom)) if analog_cell_fom.size else 0.0,
            float(np.median(vr_cell_fom)) if vr_cell_fom.size else 0.0,
        ]
        gain_text = "paired gain unavailable"
        if paired:
            paired_analog_fom = 1.0 / (analog_error[common] ** 2 * analog_runtime)
            paired_vr_fom = 1.0 / (vr_error[common] ** 2 * vr_runtime)
            gain_text = (
                f"median paired gain = "
                f"{np.median(paired_vr_fom / paired_analog_fom):.2f}×"
            )
        axes[1, 2].bar(
            methods,
            median_foms,
        )
        axes[1, 2].set_title(
            "Far-field median transport-stage cell FOM\n"
            f"{gain_text}"
        )

        for axis in axes[0]:
            axis.contour(
                far_field_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="lime",
                linewidths=1.2,
            )
        fig.suptitle(
            f"Full-field flux objective: {far_field_label.lower()} uncertainty and FOM\n"
            "Total boundary leakage is retained only as an unbiasedness diagnostic",
            fontsize=13,
        )
    else:
        positive_analog = analog_scores[analog_scores > 0.0]
        positive_vr = vr_scores[vr_scores > 0.0]
        if len(positive_analog):
            axes[1, 0].hist(
                np.log10(positive_analog),
                bins=30,
                alpha=0.65,
                label="analog",
            )
        if len(positive_vr):
            axes[1, 0].hist(
                np.log10(positive_vr),
                bins=30,
                alpha=0.65,
                label="importance",
            )
        axes[1, 0].set_title("Positive root detector scores")
        axes[1, 0].set_xlabel("log10(root score)")
        axes[1, 0].legend()

        rel_errors = [analog_summary["relative_error"], vr_summary["relative_error"]]
        axes[1, 1].bar(
            methods,
            [float(value) if value is not None else 0.0 for value in rel_errors],
        )
        axes[1, 1].set_title("Detector relative error (lower is better)")

        foms = [float(analog_summary["fom"]), float(vr_summary["fom"])]
        axes[1, 2].bar(methods, foms)
        axes[1, 2].set_title("Detector transport FOM = 1 / (R² · transport time)")

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("lattice", "hohlraum"), default="lattice")
    parser.add_argument(
        "--objective",
        choices=("auto", "boundary-response", "full-field-flux"),
        default="auto",
        help=(
            "auto treats GenVR-flux windows as a full-field flux objective and "
            "adjoint importance as a boundary-response objective"
        ),
    )
    parser.add_argument("--nx", type=int, default=28)
    parser.add_argument("--ny", type=int, default=28)
    parser.add_argument("--histories", type=int, default=5000, help="root histories per replicate")
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help=(
            "auto derives the required level range from the GenVR flux for a "
            "full-field objective; otherwise uses 6 for lattice and 10 for hohlraum"
        ),
    )
    parser.add_argument("--split-factor", type=int, default=2)
    parser.add_argument(
        "--detector-face",
        choices=("auto", "left", "right", "bottom", "top", "all"),
        default="auto",
        help="auto selects all boundaries for lattice and right boundary for hohlraum",
    )
    parser.add_argument(
        "--importance-kind",
        choices=(
            "auto",
            "linear",
            "diffusion",
            "genvr-flux",
            "adjoint-file",
            "response-absorber",
        ),
        default="auto",
        help=(
            "auto uses GenVR reciprocal-flux windows for lattice and a diffusion "
            "adjoint for the one-sided hohlraum response"
        ),
    )
    parser.add_argument("--genvr-flux-file", default=None)
    parser.add_argument("--genvr-flux-key", default="cfm")
    parser.add_argument("--reference-flux-key", default="dense")
    parser.add_argument("--adjoint-file", default=None)
    parser.add_argument("--adjoint-key", default="genvr_adjoint")
    parser.add_argument(
        "--adjoint-blend",
        type=float,
        default=1.0,
        help="geometric blend fraction for external adjoint versus diffusion adjoint",
    )
    parser.add_argument("--adjoint-smoothing-sigma", type=float, default=0.0)
    parser.add_argument(
        "--absorber-peak-fraction",
        type=float,
        default=0.125,
        help="peak absorber importance relative to normalized detector importance",
    )
    parser.add_argument(
        "--flux-exponent",
        type=float,
        default=None,
        help=(
            "auto uses 0.75 for lattice full-field flux, 1.0 for hohlraum "
            "full-field flux, and 0.3 otherwise"
        ),
    )
    parser.add_argument(
        "--response-exponent",
        type=float,
        default=None,
        help="auto uses 0 for lattice and 1 for the right-boundary hohlraum response",
    )
    parser.add_argument(
        "--contrast-exponent",
        type=float,
        default=None,
        help="auto disables the extra local-depression heuristic for full-field flux",
    )
    parser.add_argument("--contrast-scale-fraction", type=float, default=0.05)
    parser.add_argument(
        "--importance-halo-fraction",
        type=float,
        default=None,
        help="auto uses no flat minimum-filter halo for full-field flux",
    )
    parser.add_argument("--strong-absorber-ratio", type=float, default=0.8)
    parser.add_argument(
        "--absorber-boost-levels",
        type=float,
        default=None,
        help=(
            "extra continuous importance levels inside strong absorbers; auto "
            "uses 2 for lattice and 0 for hohlraum"
        ),
    )
    parser.add_argument(
        "--absorber-min-level",
        type=int,
        default=None,
        help="optional hard floor for strong absorbers; auto leaves the gradient unflattened",
    )
    parser.add_argument(
        "--absorber-halo-fraction",
        type=float,
        default=0.0,
        help=(
            "graded pre-splitting halo around strong absorbers; each mesh layer "
            "outward drops one importance level"
        ),
    )
    parser.add_argument(
        "--monotone-to-detector",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="auto enables a no-reverse-gradient constraint for one-sided detectors",
    )
    parser.add_argument(
        "--pin-detector-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "force detector-adjacent cells to the maximum level; auto disables "
            "this for the lattice all-boundary full-field objective"
        ),
    )
    parser.add_argument("--flux-floor-quantile", type=float, default=0.005)
    parser.add_argument("--flux-smoothing-sigma", type=float, default=1.25)
    parser.add_argument(
        "--flux-symmetry",
        choices=("auto", "none", "x"),
        default="auto",
        help="auto enforces the known left-right symmetry of the lattice benchmark",
    )
    parser.add_argument("--window-lower-ratio", type=float, default=0.1)
    parser.add_argument("--window-upper-ratio", type=float, default=1.25)
    parser.add_argument("--max-window-split", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260917)
    parser.add_argument("--weight-cutoff", type=float, default=1.0e-12)
    parser.add_argument("--outdir", default="outputs/importance_vr_validation")
    args = parser.parse_args()

    problem = (
        make_lattice_problem(args.nx, args.ny)
        if args.case == "lattice"
        else make_hohlraum_problem(args.nx, args.ny)
    )
    importance_kind = args.importance_kind
    if importance_kind == "auto":
        importance_kind = (
            "genvr-flux"
            if args.objective == "full-field-flux" or args.case == "lattice"
            else "diffusion"
        )
    objective = args.objective
    if objective == "auto":
        objective = (
            "full-field-flux"
            if importance_kind == "genvr-flux"
            else "boundary-response"
        )
    detector_face = args.detector_face
    if detector_face == "auto":
        detector_face = (
            "all"
            if objective == "full-field-flux" or args.case == "lattice"
            else "right"
        )
    detector = BoundaryDetector(face=detector_face)
    n_levels = args.levels
    if n_levels is None and objective != "full-field-flux":
        n_levels = 6 if args.case == "lattice" else 10
    flux_exponent = (
        args.flux_exponent
        if args.flux_exponent is not None
        else (
            (0.75 if args.case == "lattice" else 1.0)
            if objective == "full-field-flux"
            else 0.3
        )
    )
    contrast_exponent = (
        args.contrast_exponent
        if args.contrast_exponent is not None
        else (0.0 if objective == "full-field-flux" else 0.5)
    )
    importance_halo_fraction = (
        args.importance_halo_fraction
        if args.importance_halo_fraction is not None
        else (0.0 if objective == "full-field-flux" else 0.02)
    )
    if n_levels is None and importance_kind not in {"genvr-flux", "response-absorber"}:
        n_levels = 6 if args.case == "lattice" else 10
    importance = None
    weight_windows = None
    coarse_flux = None
    reference_flux = None
    response_importance = None
    reciprocal_flux_importance = None
    genvr_flux_path = None
    adjoint_path = None
    external_adjoint = None
    if importance_kind in {"genvr-flux", "response-absorber"}:
        genvr_flux_path = (
            Path(args.genvr_flux_file)
            if args.genvr_flux_file is not None
            else DEFAULT_GENVR_FLUX_FILES[args.case]
        )
        if not genvr_flux_path.exists():
            raise FileNotFoundError(
                f"GenVR flux file not found: {genvr_flux_path}; "
                "pass --genvr-flux-file explicitly"
            )
        with np.load(genvr_flux_path) as flux_data:
            if args.genvr_flux_key not in flux_data.files:
                raise KeyError(
                    f"flux key {args.genvr_flux_key!r} not found in {genvr_flux_path}; "
                    f"available keys: {flux_data.files}"
                )
            raw_coarse_flux = np.asarray(flux_data[args.genvr_flux_key], dtype=np.float64)
            if importance_kind == "genvr-flux" and args.reference_flux_key not in flux_data.files:
                raise KeyError(
                    f"reference key {args.reference_flux_key!r} not found in "
                    f"{genvr_flux_path}; available keys: {flux_data.files}"
                )
            raw_reference_flux = (
                np.asarray(flux_data[args.reference_flux_key], dtype=np.float64)
                if args.reference_flux_key in flux_data.files
                else None
            )
        symmetrize_x = args.flux_symmetry == "x" or (
            args.flux_symmetry == "auto" and args.case == "lattice"
        )
        coarse_flux = prepare_coarse_flux_field(
            problem,
            raw_coarse_flux,
            floor_quantile=args.flux_floor_quantile,
            smoothing_sigma=args.flux_smoothing_sigma,
            symmetrize_x=symmetrize_x,
        )
        if n_levels is None:
            raw_importance = make_reciprocal_flux_importance(
                problem,
                coarse_flux,
                flux_exponent=flux_exponent,
            )
            required_levels = int(
                math.ceil(
                    math.log(
                        max(float(np.max(raw_importance)), 1.0),
                        float(args.split_factor),
                    )
                )
            )
            n_levels = min(max(required_levels, 1), 36)
        if raw_reference_flux is not None:
            reference_flux = prepare_coarse_flux_field(
                problem,
                raw_reference_flux,
                floor_quantile=0.0,
                smoothing_sigma=0.0,
                symmetrize_x=symmetrize_x,
            )
        if importance_kind == "response-absorber":
            response_importance = solve_diffusion_adjoint_importance(problem, detector)
            importance_potential, reciprocal_flux_importance = (
                combine_response_and_absorber_importance(
                    problem,
                    response_importance,
                    coarse_flux,
                    flux_exponent=flux_exponent,
                    absorber_peak_fraction=args.absorber_peak_fraction,
                    strong_absorber_ratio=args.strong_absorber_ratio,
                )
            )
            importance = make_adjoint_importance_map(
                problem,
                importance_potential,
                detector=detector,
                n_levels=n_levels,
                split_factor=args.split_factor,
            )
            importance_levels = importance.levels
        else:
            reciprocal_flux_importance = make_reciprocal_flux_importance(
                problem,
                coarse_flux,
                flux_exponent=flux_exponent,
            )
        response_exponent = (
            args.response_exponent
            if args.response_exponent is not None
            else (
                0.0
                if objective == "full-field-flux"
                else (1.0 if args.case == "hohlraum" else 0.0)
            )
        )
        if importance_kind == "genvr-flux" and response_exponent > 0.0:
            response_importance = solve_diffusion_adjoint_importance(problem, detector)
        monotone_to_detector = (
            args.monotone_to_detector
            if args.monotone_to_detector is not None
            else objective != "full-field-flux" and detector.face != "all"
        )
        absorber_min_level = (
            args.absorber_min_level
            if args.absorber_min_level is not None
            else 0
        )
        absorber_boost_levels = (
            args.absorber_boost_levels
            if args.absorber_boost_levels is not None
            else (
                0.0
                if objective == "full-field-flux"
                else (2.0 if args.case == "lattice" else 0.0)
            )
        )
        pin_detector_window = (
            args.pin_detector_window
            if args.pin_detector_window is not None
            else objective != "full-field-flux" and detector.face != "all"
        )
        if importance_kind == "genvr-flux":
            weight_windows = make_inverse_flux_weight_windows(
                problem,
                coarse_flux,
                detector=detector if pin_detector_window else None,
                response_importance=response_importance,
                n_levels=n_levels,
                split_factor=args.split_factor,
                flux_exponent=flux_exponent,
                response_exponent=response_exponent,
                contrast_exponent=contrast_exponent,
                contrast_scale_fraction=args.contrast_scale_fraction,
                halo_fraction=importance_halo_fraction,
                strong_absorber_ratio=args.strong_absorber_ratio,
                absorber_boost_levels=absorber_boost_levels,
                absorber_min_level=absorber_min_level,
                absorber_halo_fraction=args.absorber_halo_fraction,
                enforce_monotone_to_detector=monotone_to_detector,
                lower_ratio=args.window_lower_ratio,
                upper_ratio=args.window_upper_ratio,
                max_split=args.max_window_split,
            )
            importance_levels = weight_windows.levels
            importance_potential = weight_windows.importance
    elif importance_kind == "adjoint-file":
        if not 0.0 <= args.adjoint_blend <= 1.0:
            raise ValueError("--adjoint-blend must lie in [0, 1]")
        default_adjoint_path = DEFAULT_ADJOINT_FILES.get(args.case)
        adjoint_path = (
            Path(args.adjoint_file)
            if args.adjoint_file is not None
            else default_adjoint_path
        )
        if adjoint_path is None or not adjoint_path.exists():
            raise FileNotFoundError(
                f"adjoint field not found: {adjoint_path}; pass --adjoint-file explicitly"
            )
        with np.load(adjoint_path) as adjoint_data:
            if args.adjoint_key not in adjoint_data.files:
                raise KeyError(
                    f"adjoint key {args.adjoint_key!r} not found in {adjoint_path}; "
                    f"available keys: {adjoint_data.files}"
                )
            raw_external_adjoint = np.asarray(
                adjoint_data[args.adjoint_key],
                dtype=np.float64,
            )
        external_adjoint = prepare_coarse_flux_field(
            problem,
            raw_external_adjoint,
            floor_quantile=0.0,
            smoothing_sigma=args.adjoint_smoothing_sigma,
            symmetrize_x=False,
        )
        external_adjoint /= float(np.max(external_adjoint))
        if args.adjoint_blend < 1.0:
            diffusion_adjoint = solve_diffusion_adjoint_importance(problem, detector)
            diffusion_adjoint /= float(np.max(diffusion_adjoint))
            importance_potential = np.exp(
                (1.0 - args.adjoint_blend)
                * np.log(np.maximum(diffusion_adjoint, np.finfo(np.float64).tiny))
                + args.adjoint_blend
                * np.log(np.maximum(external_adjoint, np.finfo(np.float64).tiny))
            )
        else:
            importance_potential = external_adjoint.copy()
        importance = make_adjoint_importance_map(
            problem,
            importance_potential,
            detector=detector,
            n_levels=n_levels,
            split_factor=args.split_factor,
        )
        importance_levels = importance.levels
    elif importance_kind == "diffusion":
        importance_potential = solve_diffusion_adjoint_importance(problem, detector)
        importance = make_diffusion_adjoint_importance_map(
            problem,
            detector=detector,
            n_levels=n_levels,
            split_factor=args.split_factor,
        )
        importance_levels = importance.levels
    else:
        if detector.face == "all":
            importance = make_all_boundary_importance_map(
                problem,
                n_levels=n_levels,
                split_factor=args.split_factor,
            )
        elif detector.face == "right":
            importance = make_right_boundary_importance_map(
                problem,
                n_levels=n_levels,
                split_factor=args.split_factor,
            )
        else:
            raise ValueError(
                "linear importance currently supports right or all-boundary detectors; "
                "use --importance-kind diffusion for other detector faces"
            )
        importance_potential = np.power(
            float(args.split_factor), importance.levels.astype(np.float64)
        )
        importance_levels = importance.levels
    importance_potential /= float(np.max(importance_potential))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    analog_results: list[DetectorTransportResult] = []
    vr_results: list[DetectorTransportResult] = []
    analog_runtimes: list[float] = []
    vr_runtimes: list[float] = []
    runs: list[dict[str, object]] = []

    for replicate in range(args.replicates):
        run_seed = args.seed + 104729 * replicate
        start = time.perf_counter()
        analog = run_detector_mc(
            problem,
            n_particles=args.histories,
            detector=detector,
            importance=None,
            seed=run_seed,
            weight_cutoff=args.weight_cutoff,
        )
        analog_runtime = time.perf_counter() - start

        start = time.perf_counter()
        vr = run_detector_mc(
            problem,
            n_particles=args.histories,
            detector=detector,
            importance=importance,
            weight_windows=weight_windows,
            seed=run_seed,
            weight_cutoff=args.weight_cutoff,
        )
        vr_runtime = time.perf_counter() - start

        analog_results.append(analog)
        vr_results.append(vr)
        analog_runtimes.append(analog_runtime)
        vr_runtimes.append(vr_runtime)
        run_record = {
            "replicate": replicate,
            "seed": run_seed,
            "analog": _run_summary(analog, analog_runtime),
            "importance": _run_summary(vr, vr_runtime),
        }
        runs.append(run_record)
        print(
            f"replicate={replicate} "
            f"analog={analog.estimate:.6g}±{analog.standard_error:.3g} "
            f"importance={vr.estimate:.6g}±{vr.standard_error:.3g}",
            flush=True,
        )

    analog_summary = _aggregate(analog_results, analog_runtimes)
    vr_summary = _aggregate(vr_results, vr_runtimes)
    analog_flux_statistics = _aggregate_flux_statistics(analog_results)
    vr_flux_statistics = _aggregate_flux_statistics(vr_results)
    analog_variance = float(analog_summary["sample_variance"])
    vr_variance = float(vr_summary["sample_variance"])
    variance_reduction = analog_variance / vr_variance if vr_variance > 0.0 else float("inf")
    analog_relative_error = analog_summary["relative_error"]
    vr_relative_error = vr_summary["relative_error"]
    relative_variance_reduction = float("inf")
    if analog_relative_error is not None and vr_relative_error not in (None, 0.0):
        relative_variance_reduction = (
            float(analog_relative_error) / float(vr_relative_error)
        ) ** 2
    fom_gain = (
        float(vr_summary["fom"]) / float(analog_summary["fom"])
        if float(analog_summary["fom"]) > 0.0
        else float("inf")
    )
    difference = float(vr_summary["estimate"]) - float(analog_summary["estimate"])
    combined_se = math.sqrt(
        float(vr_summary["standard_error"]) ** 2
        + float(analog_summary["standard_error"]) ** 2
    )
    consistency_z = difference / combined_se if combined_se > 0.0 else 0.0
    analog_mean_flux = np.asarray(analog_flux_statistics["mean"], dtype=np.float64)
    vr_mean_flux = np.asarray(vr_flux_statistics["mean"], dtype=np.float64)
    analog_cell_visits = np.sum(
        [result.cell_visits for result in analog_results],
        axis=0,
    )
    vr_cell_visits = np.sum(
        [result.cell_visits for result in vr_results],
        axis=0,
    )
    split_events_map = np.sum(
        [result.split_events_map for result in vr_results],
        axis=0,
    )
    split_children_map = np.sum(
        [result.split_children_map for result in vr_results],
        axis=0,
    )
    split_cap_hits_map = np.sum(
        [result.split_cap_hits_map for result in vr_results],
        axis=0,
    )
    roulette_kills_map = np.sum(
        [result.roulette_kills_map for result in vr_results],
        axis=0,
    )

    summary = {
        "schema_version": "1.0",
        "case": args.case,
        "objective": objective,
        "grid": [args.nx, args.ny],
        "detector": {"face": detector.face, "lower": None, "upper": None},
        "histories_per_replicate": args.histories,
        "replicates": args.replicates,
        "importance": {
            "kind": importance_kind,
            "requested_kind": args.importance_kind,
            "n_levels": n_levels,
            "split_factor": args.split_factor,
            "levels": importance_levels.tolist(),
        },
        "analog": analog_summary,
        "importance_vr": vr_summary,
        "comparison": {
            "role": (
                "auxiliary total-leakage diagnostic; not the optimization metric"
                if objective == "full-field-flux"
                else "primary boundary-response metric"
            ),
            "estimate_difference": difference,
            "unpaired_consistency_z": consistency_z,
            "root_score_variance_ratio": _finite(variance_reduction),
            "relative_variance_reduction_factor": _finite(relative_variance_reduction),
            "fom_gain": _finite(fom_gain),
            "step_ratio_vr_over_analog": (
                int(vr_summary["steps"]) / int(analog_summary["steps"])
                if int(analog_summary["steps"]) > 0
                else None
            ),
        },
        "runs": runs,
    }
    sigma_total = problem.sigma_a + problem.sigma_s
    absorption_ratio = np.divide(
        problem.sigma_a,
        sigma_total,
        out=np.zeros_like(problem.sigma_a),
        where=sigma_total > 0.0,
    )
    strong_absorber_mask = absorption_ratio >= args.strong_absorber_ratio
    nonabsorber_mask = ~strong_absorber_mask
    guide_flux = (
        reference_flux
        if reference_flux is not None
        else (coarse_flux if coarse_flux is not None else analog_mean_flux)
    )
    region_mask_flux_key = (
        args.reference_flux_key
        if reference_flux is not None
        else (args.genvr_flux_key if coarse_flux is not None else "analog_flux")
    )
    full_field_masks = _full_field_masks(
        problem,
        guide_flux,
        args.strong_absorber_ratio,
    )
    summary["full_field_flux"] = {
        "definition": (
            "Cellwise track-length scores are first summed over every descendant "
            "of one source history; cell variances are then estimated across "
            "independent root histories."
        ),
        "primary_region": "far_field_cells",
        "deep_penetration_region": "far_field_low_flux_quartile",
        "region_mask_flux_key": region_mask_flux_key,
        "region_definitions": {
            "far_field_cells": (
                "For a left-boundary source: downstream interior cells with "
                "0.65W <= x < 0.96W and 0.04H <= y < 0.96H. For a volume "
                "source: the outer quartile of geometric distance from the source box."
            ),
            "far_field_low_flux_quartile": (
                "The lowest evaluation-reference-flux quartile restricted to "
                "far_field_cells."
            ),
        },
        "analog": _full_field_summary(
            analog_flux_statistics,
            float(analog_summary["runtime_s"]),
            full_field_masks,
        ),
        "importance_vr": _full_field_summary(
            vr_flux_statistics,
            float(vr_summary["runtime_s"]),
            full_field_masks,
        ),
        "comparison": _compare_full_field_statistics(
            analog_flux_statistics,
            vr_flux_statistics,
            float(analog_summary["runtime_s"]),
            float(vr_summary["runtime_s"]),
            full_field_masks,
        ),
    }
    primary_region = summary["full_field_flux"]["primary_region"]
    deep_region = summary["full_field_flux"]["deep_penetration_region"]
    summary["primary_results"] = {
        "objective": "full-field track-length flux",
        "primary_region": primary_region,
        "far_field_analog": summary["full_field_flux"]["analog"]["regions"][
            primary_region
        ],
        "far_field_importance_vr": summary["full_field_flux"]["importance_vr"][
            "regions"
        ][primary_region],
        "far_field_comparison": summary["full_field_flux"]["comparison"][
            primary_region
        ],
        "deep_penetration_comparison": summary["full_field_flux"]["comparison"][
            deep_region
        ],
    }

    def visit_ratio(mask: np.ndarray) -> float | None:
        analog_visits = int(np.sum(analog_cell_visits[mask]))
        if analog_visits == 0:
            return None
        return float(np.sum(vr_cell_visits[mask]) / analog_visits)

    total_histories = int(vr_summary["histories"])
    total_split_children = int(np.sum(split_children_map))
    summary["population_control"] = {
        "transported_particles_per_root": (
            float(vr_summary["transported_particles"]) / total_histories
        ),
        "split_children_per_root": (
            float(vr_summary["split_children"]) / total_histories
        ),
        "split_cap_hits_per_root": (
            float(vr_summary["split_cap_hits"]) / total_histories
        ),
        "strong_absorber_visit_ratio_vr_over_analog": visit_ratio(
            strong_absorber_mask
        ),
        "nonabsorber_visit_ratio_vr_over_analog": visit_ratio(nonabsorber_mask),
        "split_children_in_strong_absorbers": int(
            np.sum(split_children_map[strong_absorber_mask])
        ),
        "split_children_in_strong_absorbers_fraction": (
            float(np.sum(split_children_map[strong_absorber_mask]))
            / total_split_children
            if total_split_children > 0
            else 0.0
        ),
        "maximum_level_cell_fraction": float(
            np.mean(importance_levels == np.max(importance_levels))
        ),
    }
    if args.case == "hohlraum":
        centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
        centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
        x_grid, y_grid = np.meshgrid(centers_x, centers_y)
        bypass_corridor_mask = (
            (x_grid >= 0.45)
            & (x_grid <= 0.85)
            & (y_grid > 0.05)
            & (y_grid < problem.height - 0.05)
            & ((y_grid < 0.34) | (y_grid > 0.96))
        )
        downstream_mask = (
            (x_grid > 0.85)
            & (x_grid < problem.width - 0.05)
            & (y_grid > 0.05)
            & (y_grid < problem.height - 0.05)
        )
        summary["population_control"].update(
            {
                "bypass_corridor_visit_ratio_vr_over_analog": visit_ratio(
                    bypass_corridor_mask
                ),
                "downstream_visit_ratio_vr_over_analog": visit_ratio(
                    downstream_mask
                ),
                "split_children_in_bypass_corridors": int(
                    np.sum(split_children_map[bypass_corridor_mask])
                ),
                "split_children_in_bypass_corridors_fraction": (
                    float(np.sum(split_children_map[bypass_corridor_mask]))
                    / total_split_children
                    if total_split_children > 0
                    else 0.0
                ),
            }
        )
    if reference_flux is not None:
        analog_flux_validation = _flux_shape_summary(
            analog_mean_flux,
            reference_flux,
            problem,
        )
        vr_flux_validation = _flux_shape_summary(
            vr_mean_flux,
            reference_flux,
            problem,
        )
        summary["flux_validation"] = {
            "reference_key": args.reference_flux_key,
            "reference_note": (
                "Deterministic dense field from the same benchmark artifact; "
                "used as a shape reference, not claimed as an exact continuum solution."
            ),
            "analog": analog_flux_validation,
            "genvr_weight_window": vr_flux_validation,
            "error_reduction_percent": {
                region: _error_reduction_percent(
                    analog_flux_validation,
                    vr_flux_validation,
                    region,
                )
                for region in (
                    "all_cells",
                    "absorber_cells",
                    "deep_absorber_quartile",
                    "lowest_flux_quartile",
                )
            },
        }
    if weight_windows is not None:
        summary["importance"].update(
            {
                "genvr_flux_file": str(genvr_flux_path),
                "genvr_flux_key": args.genvr_flux_key,
                "reference_flux_key": args.reference_flux_key,
                "flux_exponent": flux_exponent,
                "response_exponent": response_exponent,
                "contrast_exponent": contrast_exponent,
                "contrast_scale_fraction": args.contrast_scale_fraction,
                "importance_halo_fraction": importance_halo_fraction,
                "strong_absorber_ratio": args.strong_absorber_ratio,
                "absorber_boost_levels": absorber_boost_levels,
                "absorber_min_level": absorber_min_level,
                "absorber_halo_fraction": args.absorber_halo_fraction,
                "pin_detector_window": pin_detector_window,
                "monotone_to_detector": monotone_to_detector,
                "flux_floor_quantile": args.flux_floor_quantile,
                "flux_smoothing_sigma": args.flux_smoothing_sigma,
                "flux_symmetry": args.flux_symmetry,
                "window_lower_ratio": args.window_lower_ratio,
                "window_upper_ratio": args.window_upper_ratio,
                "max_window_split": args.max_window_split,
                "target_weight_min": float(np.min(weight_windows.target_weights)),
                "target_weight_max": float(np.max(weight_windows.target_weights)),
                "reciprocal_flux_importance_min": float(
                    np.min(reciprocal_flux_importance)
                ),
                "reciprocal_flux_importance_max": float(
                    np.max(reciprocal_flux_importance)
                ),
                "absorber_mean_level": float(
                    np.mean(weight_windows.levels[problem.sigma_a > 0.0])
                ),
                "nonabsorber_mean_level": float(
                    np.mean(weight_windows.levels[problem.sigma_a == 0.0])
                ),
            }
        )
    if external_adjoint is not None:
        summary["importance"].update(
            {
                "adjoint_file": str(adjoint_path),
                "adjoint_key": args.adjoint_key,
                "adjoint_blend": args.adjoint_blend,
                "adjoint_smoothing_sigma": args.adjoint_smoothing_sigma,
            }
        )
    if importance_kind == "response-absorber":
        summary["importance"].update(
            {
                "genvr_flux_file": str(genvr_flux_path),
                "genvr_flux_key": args.genvr_flux_key,
                "flux_exponent": flux_exponent,
                "absorber_peak_fraction": args.absorber_peak_fraction,
                "strong_absorber_ratio": args.strong_absorber_ratio,
            }
        )
    (outdir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    arrays = {
        "analog_root_scores": _combine(analog_results),
        "importance_root_scores": _combine(vr_results),
        "analog_flux": analog_mean_flux,
        "importance_flux": vr_mean_flux,
        "analog_flux_sample_variance": analog_flux_statistics["sample_variance"],
        "importance_flux_sample_variance": vr_flux_statistics["sample_variance"],
        "analog_flux_standard_error": analog_flux_statistics["standard_error"],
        "importance_flux_standard_error": vr_flux_statistics["standard_error"],
        "analog_flux_relative_error": analog_flux_statistics["relative_error"],
        "importance_flux_relative_error": vr_flux_statistics["relative_error"],
        "analog_flux_contributing_roots": analog_flux_statistics["contributing_roots"],
        "importance_flux_contributing_roots": vr_flux_statistics["contributing_roots"],
        "far_field_mask": full_field_masks["far_field_cells"],
        "far_field_low_flux_mask": full_field_masks[
            "far_field_low_flux_quartile"
        ],
        "importance_levels": importance_levels,
        "importance_potential": importance_potential,
        "analog_cell_visits": analog_cell_visits,
        "importance_cell_visits": vr_cell_visits,
        "split_events_map": split_events_map,
        "split_children_map": split_children_map,
        "split_cap_hits_map": split_cap_hits_map,
        "roulette_kills_map": roulette_kills_map,
    }
    if coarse_flux is not None:
        arrays["coarse_flux"] = coarse_flux
    if reciprocal_flux_importance is not None:
        arrays["reciprocal_flux_importance"] = reciprocal_flux_importance
    if reference_flux is not None:
        arrays["reference_flux"] = reference_flux
    if response_importance is not None:
        arrays["response_importance"] = response_importance
    if external_adjoint is not None:
        arrays["external_adjoint"] = external_adjoint
    if weight_windows is not None:
        arrays["target_weights"] = weight_windows.target_weights
    np.savez_compressed(
        outdir / "comparison.npz",
        **arrays,
    )
    _plot(
        outdir / "comparison.png",
        problem,
        importance_levels,
        analog_flux_statistics,
        vr_flux_statistics,
        _combine(analog_results),
        _combine(vr_results),
        analog_summary,
        vr_summary,
        objective,
        full_field_masks["far_field_cells"],
    )

    printed_comparison = (
        summary["primary_results"]
        if objective == "full-field-flux"
        else summary["comparison"]
    )
    print(json.dumps({
        "analog": analog_summary,
        "importance_vr": vr_summary,
        "comparison": printed_comparison,
        "artifacts": {
            "json": str(outdir / "comparison.json"),
            "npz": str(outdir / "comparison.npz"),
            "figure": str(outdir / "comparison.png"),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
