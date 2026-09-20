from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

import numpy as np

from gmc.importance_transport import (
    BoundaryDetector,
    DetectorTransportResult,
    WeightWindowMap,
    make_inverse_flux_weight_windows,
    make_reciprocal_flux_importance,
    prepare_coarse_flux_field,
    run_detector_mc,
)

from .shield_design import (
    ShieldDesign,
    materialize_lattice_design,
    source_distant_mask,
)


@dataclass(frozen=True)
class WeightWindowSettings:
    n_levels: int
    split_factor: int = 2
    flux_exponent: float = 0.75
    lower_ratio: float = 0.10
    upper_ratio: float = 1.25
    max_split: int = 16

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_levels": self.n_levels,
            "split_factor": self.split_factor,
            "flux_exponent": self.flux_exponent,
            "lower_ratio": self.lower_ratio,
            "upper_ratio": self.upper_ratio,
            "max_split": self.max_split,
        }


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _automatic_level_count(
    problem,
    coarse_flux: np.ndarray,
    split_factor: int,
    flux_exponent: float,
    maximum_levels: int,
) -> int:
    importance = make_reciprocal_flux_importance(
        problem,
        coarse_flux,
        flux_exponent=flux_exponent,
    )
    required = int(
        math.ceil(
            math.log(
                max(float(np.max(importance)), 1.0),
                float(split_factor),
            )
        )
    )
    return min(max(required, 1), maximum_levels)


def build_design_weight_windows(
    design: ShieldDesign,
    guide_flux: np.ndarray,
    nx: int,
    ny: int,
    n_levels: int | None = None,
    maximum_levels: int = 12,
    split_factor: int = 2,
    flux_exponent: float = 0.75,
    lower_ratio: float = 0.10,
    upper_ratio: float = 1.25,
    max_split: int = 16,
) -> tuple[WeightWindowMap, np.ndarray, WeightWindowSettings]:
    problem = materialize_lattice_design(design, nx, ny)
    coarse_flux = prepare_coarse_flux_field(
        problem,
        np.asarray(guide_flux, dtype=np.float64),
        floor_quantile=0.005,
        smoothing_sigma=1.25,
        symmetrize_x=False,
    )
    levels = (
        int(n_levels)
        if n_levels is not None
        else _automatic_level_count(
            problem,
            coarse_flux,
            split_factor,
            flux_exponent,
            maximum_levels,
        )
    )
    settings = WeightWindowSettings(
        n_levels=levels,
        split_factor=split_factor,
        flux_exponent=flux_exponent,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        max_split=max_split,
    )
    windows = make_inverse_flux_weight_windows(
        problem,
        coarse_flux,
        detector=None,
        n_levels=settings.n_levels,
        split_factor=settings.split_factor,
        flux_exponent=settings.flux_exponent,
        response_exponent=0.0,
        contrast_exponent=0.0,
        halo_fraction=0.0,
        absorber_boost_levels=0.0,
        absorber_min_level=0,
        absorber_halo_fraction=0.0,
        enforce_monotone_to_detector=False,
        lower_ratio=settings.lower_ratio,
        upper_ratio=settings.upper_ratio,
        max_split=settings.max_split,
    )
    return windows, coarse_flux, settings


def _region_masks(problem, guide_flux: np.ndarray) -> dict[str, np.ndarray]:
    far_field = source_distant_mask(problem)
    far_values = np.asarray(guide_flux, dtype=np.float64)[far_field]
    low_threshold = float(np.quantile(far_values, 0.25))
    return {
        "far_field": far_field,
        "deep_far_field": far_field & (guide_flux <= low_threshold),
    }


def _scalar_summary(scores: np.ndarray, runtime_s: float) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    histories = int(values.size)
    estimate = float(np.mean(values))
    variance = float(np.var(values, ddof=1)) if histories > 1 else float("nan")
    standard_error = (
        float(math.sqrt(variance / histories)) if histories > 1 else float("nan")
    )
    relative_error = (
        standard_error / abs(estimate) if estimate != 0.0 else float("inf")
    )
    fom = (
        1.0 / (relative_error * relative_error * runtime_s)
        if relative_error > 0.0 and math.isfinite(relative_error) and runtime_s > 0.0
        else 0.0
    )
    return {
        "estimate": estimate,
        "sample_variance": _finite(variance),
        "standard_error": _finite(standard_error),
        "relative_error": _finite(relative_error),
        "ci95": (
            [
                max(0.0, estimate - 1.96 * standard_error),
                estimate + 1.96 * standard_error,
            ]
            if math.isfinite(standard_error)
            else [None, None]
        ),
        "fom": fom,
        "histories": histories,
        "runtime_s": runtime_s,
    }


def _aggregate_method(
    results: list[DetectorTransportResult],
    runtimes: list[float],
    masks: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    histories = int(sum(result.histories for result in results))
    runtime_s = float(sum(runtimes))
    score_sum = np.sum([result.flux_score_sum for result in results], axis=0)
    score_sum_sq = np.sum([result.flux_score_sum_sq for result in results], axis=0)
    flux = score_sum / float(histories)
    if histories > 1:
        centered = score_sum_sq - score_sum * score_sum / float(histories)
        variance = np.maximum(centered / float(histories - 1), 0.0)
        standard_error = np.sqrt(variance / float(histories))
    else:
        standard_error = np.full_like(flux, np.nan)
    relative_error = np.full_like(flux, np.inf)
    np.divide(
        standard_error,
        np.abs(flux),
        out=relative_error,
        where=flux != 0.0,
    )
    regions: dict[str, Any] = {}
    for name, mask in masks.items():
        scores = np.concatenate([result.region_scores[name] for result in results])
        valid = mask & np.isfinite(relative_error) & (flux != 0.0)
        region_summary = _scalar_summary(scores, runtime_s)
        region_summary.update(
            {
                "cells": int(np.count_nonzero(mask)),
                "finite_cell_fraction": float(np.mean(valid[mask])),
                "median_cell_relative_error": (
                    float(np.median(relative_error[valid])) if np.any(valid) else None
                ),
                "p90_cell_relative_error": (
                    float(np.quantile(relative_error[valid], 0.90))
                    if np.any(valid)
                    else None
                ),
            }
        )
        regions[name] = region_summary
    summary = {
        "histories": histories,
        "runtime_s": runtime_s,
        "regions": regions,
        "transported_particles": int(
            sum(result.transported_particles for result in results)
        ),
        "transported_particles_per_root": float(
            sum(result.transported_particles for result in results) / histories
        ),
        "split_events": int(sum(result.split_events for result in results)),
        "split_children": int(sum(result.split_children for result in results)),
        "split_cap_hits": int(sum(result.split_cap_hits for result in results)),
        "roulette_kills": int(sum(result.roulette_kills for result in results)),
        "max_bank_size": int(max(result.max_bank_size for result in results)),
    }
    arrays = {
        "flux": flux,
        "relative_error": relative_error,
        "cell_visits": np.sum([result.cell_visits for result in results], axis=0),
        "split_children_map": np.sum(
            [result.split_children_map for result in results],
            axis=0,
        ),
    }
    return summary, arrays


def _run_method(
    problem,
    masks: dict[str, np.ndarray],
    histories: int,
    replicates: int,
    seed: int,
    weight_windows: WeightWindowMap | None,
    weight_cutoff: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    results = []
    runtimes = []
    for replicate in range(replicates):
        start = perf_counter()
        result = run_detector_mc(
            problem,
            histories,
            detector=BoundaryDetector(face="all"),
            weight_windows=weight_windows,
            seed=seed + replicate * 100_003,
            weight_cutoff=weight_cutoff,
            tally_masks=masks,
        )
        runtimes.append(perf_counter() - start)
        results.append(result)
    return _aggregate_method(results, runtimes, masks)


def _method_consistency(
    analog: dict[str, Any],
    variance_reduced: dict[str, Any],
    region: str,
) -> dict[str, Any]:
    analog_region = analog["regions"][region]
    variance_reduced_region = variance_reduced["regions"][region]
    difference = float(variance_reduced_region["estimate"]) - float(
        analog_region["estimate"]
    )
    standard_errors = (
        float(analog_region["standard_error"] or 0.0),
        float(variance_reduced_region["standard_error"] or 0.0),
    )
    combined_standard_error = math.hypot(*standard_errors)
    analog_fom = float(analog_region["fom"])
    return {
        "estimate_difference": difference,
        "unpaired_consistency_z": (
            difference / combined_standard_error
            if combined_standard_error > 0.0
            else 0.0
        ),
        "fom_gain": (
            float(variance_reduced_region["fom"]) / analog_fom
            if analog_fom > 0.0
            else None
        ),
    }


def certify_shield_pair(
    baseline_design: ShieldDesign,
    candidate_design: ShieldDesign,
    baseline_guide_flux: np.ndarray,
    candidate_guide_flux: np.ndarray,
    nx: int = 28,
    ny: int = 28,
    histories: int = 1000,
    replicates: int = 2,
    n_levels: int | None = None,
    maximum_levels: int = 12,
    split_factor: int = 2,
    flux_exponent: float = 0.75,
    lower_ratio: float = 0.10,
    upper_ratio: float = 1.25,
    max_split: int = 16,
    seed: int = 260918,
    weight_cutoff: float = 1.0e-12,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if histories < 2 or replicates < 1:
        raise ValueError("MC certification requires at least two histories and one replicate")
    designs = {
        "baseline": (baseline_design, baseline_guide_flux),
        "candidate": (candidate_design, candidate_guide_flux),
    }
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "objective": "source-distant full-field track-length flux",
        "grid": [nx, ny],
        "histories_per_replicate": histories,
        "replicates": replicates,
        "designs": {},
    }
    arrays: dict[str, np.ndarray] = {}
    prepared: dict[
        str,
        tuple[Any, WeightWindowMap, np.ndarray, WeightWindowSettings],
    ] = {}
    for name, (design, guide_flux) in designs.items():
        problem = materialize_lattice_design(design, nx, ny)
        windows, prepared_guide, settings = build_design_weight_windows(
            design,
            guide_flux,
            nx,
            ny,
            n_levels=n_levels,
            maximum_levels=maximum_levels,
            split_factor=split_factor,
            flux_exponent=flux_exponent,
            lower_ratio=lower_ratio,
            upper_ratio=upper_ratio,
            max_split=max_split,
        )
        prepared[name] = (problem, windows, prepared_guide, settings)
    baseline_problem, _, baseline_prepared_guide, _ = prepared["baseline"]
    masks = _region_masks(baseline_problem, baseline_prepared_guide)
    summary["region_definition"] = {
        "far_field": "outer quartile of geometric distance from the fixed source box",
        "deep_far_field": (
            "lowest baseline-guide-flux quartile within the fixed far-field mask; "
            "the identical cells are used for both designs"
        ),
    }

    for design_index, (name, (design, _)) in enumerate(designs.items()):
        problem, windows, prepared_guide, settings = prepared[name]
        run_seed = seed + design_index * 1_000_003
        analog, analog_arrays = _run_method(
            problem,
            masks,
            histories,
            replicates,
            run_seed,
            None,
            weight_cutoff,
        )
        variance_reduced, variance_reduced_arrays = _run_method(
            problem,
            masks,
            histories,
            replicates,
            run_seed + 500_009,
            windows,
            weight_cutoff,
        )
        summary["designs"][name] = {
            "design": design.to_dict(),
            "weight_window": settings.to_dict(),
            "analog": analog,
            "variance_reduced": variance_reduced,
            "unbiased_consistency": {
                region: _method_consistency(analog, variance_reduced, region)
                for region in masks
            },
        }
        arrays[f"{name}_guide_flux"] = prepared_guide
        arrays[f"{name}_weight_window"] = windows.target_weights
        arrays[f"{name}_importance_levels"] = windows.levels
        arrays[f"{name}_far_mask"] = masks["far_field"]
        arrays[f"{name}_deep_far_mask"] = masks["deep_far_field"]
        for method_name, method_arrays in (
            ("analog", analog_arrays),
            ("variance_reduced", variance_reduced_arrays),
        ):
            for array_name, values in method_arrays.items():
                arrays[f"{name}_{method_name}_{array_name}"] = values

    baseline_region = summary["designs"]["baseline"]["variance_reduced"]["regions"]
    candidate_region = summary["designs"]["candidate"]["variance_reduced"]["regions"]
    comparisons = {}
    for region in ("far_field", "deep_far_field"):
        baseline_estimate = float(baseline_region[region]["estimate"])
        candidate_estimate = float(candidate_region[region]["estimate"])
        baseline_se = float(baseline_region[region]["standard_error"] or 0.0)
        candidate_se = float(candidate_region[region]["standard_error"] or 0.0)
        difference = candidate_estimate - baseline_estimate
        combined_se = math.hypot(baseline_se, candidate_se)
        baseline_ci = baseline_region[region]["ci95"]
        candidate_ci = candidate_region[region]["ci95"]
        comparisons[region] = {
            "candidate_over_baseline": (
                candidate_estimate / baseline_estimate
                if baseline_estimate != 0.0
                else None
            ),
            "estimated_reduction_fraction": (
                1.0 - candidate_estimate / baseline_estimate
                if baseline_estimate != 0.0
                else None
            ),
            "difference": difference,
            "difference_z": difference / combined_se if combined_se > 0.0 else 0.0,
            "candidate_upper_below_baseline_lower": (
                candidate_ci[1] is not None
                and baseline_ci[0] is not None
                and float(candidate_ci[1]) < float(baseline_ci[0])
            ),
        }
    summary["design_comparison"] = comparisons
    summary["interpretation"] = {
        "shielding_claim_uses": (
            "variance-reduced absolute far-field flux estimates and root-history "
            "confidence intervals"
        ),
        "variance_reduction_claim_uses": (
            "analog-versus-weight-window relative error and FOM for the same design"
        ),
        "fom_is_not_the_shielding_objective": True,
    }
    return summary, arrays
