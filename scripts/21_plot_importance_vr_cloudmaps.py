#!/usr/bin/env python3
"""Create publication-style cloud maps from importance-VR comparison outputs."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
import numpy as np

from gmc.benchmarks2d import make_hohlraum_problem, make_lattice_problem


FULL_FIELD_OBJECTIVE = "full-field-flux"
FOM_GAIN_LOG2_LIMIT = 6.0
ROOT_GAIN_LOG2_LIMIT = 8.0
SHAPE_IMPROVEMENT_LIMIT = 2.0


def _problem(case: str, nx: int, ny: int):
    if case == "lattice":
        return make_lattice_problem(nx, ny)
    if case == "hohlraum":
        return make_hohlraum_problem(nx, ny)
    raise ValueError(case)


def _detector_marker(axis, problem, detector_face: str) -> None:
    segments = {
        "left": ([0.0, 0.0], [0.0, problem.height]),
        "right": ([problem.width, problem.width], [0.0, problem.height]),
        "bottom": ([0.0, problem.width], [0.0, 0.0]),
        "top": ([0.0, problem.width], [problem.height, problem.height]),
    }
    faces = tuple(segments) if detector_face == "all" else (detector_face,)
    for index, face in enumerate(faces):
        xs, ys = segments[face]
        axis.plot(
            xs,
            ys,
            color="red",
            linewidth=4,
            solid_capstyle="butt",
            label="all-boundary detector" if detector_face == "all" and index == 0 else (
                f"{detector_face}-boundary detector" if index == 0 else None
            ),
        )


def _source_marker(axis, problem) -> None:
    if problem.source_box is None:
        if problem.source_kind == "left_boundary":
            ymin, ymax = problem.boundary_source_y_range or (0.0, problem.height)
            axis.plot(
                [0.0, 0.0],
                [ymin, ymax],
                color="cyan",
                linewidth=4,
                solid_capstyle="butt",
                label="source boundary",
            )
        return
    xmin, xmax, ymin, ymax = problem.source_box
    axis.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        color="cyan",
        linewidth=1.8,
        linestyle="--",
        label="source region",
    )


def _objective(metadata: dict[str, object]) -> str:
    value = metadata.get("objective")
    if value:
        return str(value)
    importance = metadata.get("importance", {})
    if isinstance(importance, dict) and importance.get("kind") == "genvr-flux":
        return FULL_FIELD_OBJECTIVE
    return "boundary-response"


def _absorption_ratio(problem) -> np.ndarray:
    sigma_total = problem.sigma_a + problem.sigma_s
    return np.divide(
        problem.sigma_a,
        sigma_total,
        out=np.zeros_like(problem.sigma_a, dtype=np.float64),
        where=sigma_total > 0.0,
    )


def _strong_absorber_threshold(metadata: dict[str, object]) -> float:
    importance = metadata.get("importance", {})
    if isinstance(importance, dict):
        return float(importance.get("strong_absorber_ratio", 0.8))
    return 0.8


def _default_far_field_mask(problem) -> np.ndarray:
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)
    if problem.source_kind == "left_boundary":
        return (
            (x_grid >= 0.65 * problem.width)
            & (x_grid < 0.96 * problem.width)
            & (y_grid >= 0.04 * problem.height)
            & (y_grid < 0.96 * problem.height)
        )
    if problem.source_kind == "volume_box" and problem.source_box is not None:
        xmin, xmax, ymin, ymax = problem.source_box
        delta_x = np.maximum.reduce(
            (xmin - x_grid, np.zeros_like(x_grid), x_grid - xmax)
        )
        delta_y = np.maximum.reduce(
            (ymin - y_grid, np.zeros_like(y_grid), y_grid - ymax)
        )
        source_distance = np.hypot(delta_x, delta_y)
        non_source = source_distance > 0.0
        distance_threshold = float(np.quantile(source_distance[non_source], 0.75))
        return source_distance >= distance_threshold
    raise ValueError(
        f"unsupported source definition for far-field mask: {problem.source_kind}"
    )


def _far_field_mask(data, problem) -> np.ndarray:
    if "far_field_mask" not in data.files:
        return _default_far_field_mask(problem)
    mask = np.asarray(data["far_field_mask"], dtype=bool)
    if mask.shape != (problem.ny, problem.nx):
        raise ValueError(
            f"far_field_mask shape {mask.shape} does not match "
            f"problem shape {(problem.ny, problem.nx)}"
        )
    return mask


def _normalized_log_fields(
    *fields: np.ndarray,
    floor_quantile: float = 0.005,
) -> tuple[list[np.ndarray], float]:
    if not fields:
        raise ValueError("at least one field is required")
    normalized_fields: list[np.ndarray] = []
    for values in fields:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("flux fields must be finite and non-negative")
        scale = max(float(np.max(array)), 1.0e-300)
        normalized_fields.append(array / scale)
    positive_parts = [values[values > 0.0] for values in normalized_fields]
    positive_parts = [values for values in positive_parts if values.size]
    if not positive_parts:
        raise ValueError("at least one field must contain a positive value")
    positive = np.concatenate(positive_parts)
    floor = max(float(np.quantile(positive, floor_quantile)), 1.0e-12)
    return (
        [np.log10(np.maximum(values, floor)) for values in normalized_fields],
        np.log10(floor),
    )


def _absolute_log_fields(
    *fields: np.ndarray,
    floor_quantile: float = 0.005,
) -> tuple[list[np.ndarray], float, float]:
    if not fields:
        raise ValueError("at least one field is required")
    arrays = [np.asarray(values, dtype=np.float64) for values in fields]
    if any(not np.all(np.isfinite(values)) or np.any(values < 0.0) for values in arrays):
        raise ValueError("flux fields must be finite and non-negative")
    positive_parts = [values[values > 0.0] for values in arrays]
    positive_parts = [values for values in positive_parts if values.size]
    if not positive_parts:
        raise ValueError("at least one field must contain a positive value")
    positive = np.concatenate(positive_parts)
    floor = max(float(np.quantile(positive, floor_quantile)), 1.0e-300)
    ceiling = max(float(np.max(positive)), floor * 10.0)
    return (
        [np.log10(np.maximum(values, floor)) for values in arrays],
        np.log10(floor),
        np.log10(ceiling),
    )


def _objective_overlay(
    axis,
    problem,
    objective: str,
    detector_face: str,
    far_field_mask: np.ndarray | None = None,
) -> None:
    _source_marker(axis, problem)
    if objective == FULL_FIELD_OBJECTIVE:
        if far_field_mask is not None and np.any(far_field_mask):
            axis.contour(
                far_field_mask,
                levels=[0.5],
                origin="lower",
                extent=[0.0, problem.width, 0.0, problem.height],
                colors="lime",
                linewidths=1.15,
                alpha=0.9,
            )
        return
    _detector_marker(axis, problem, detector_face)


def _format_spatial_axis(axis, problem) -> None:
    axis.set_xlabel("x [cm]")
    axis.set_ylabel("y [cm]")
    axis.set_xlim(0.0, problem.width)
    axis.set_ylim(0.0, problem.height)


def _overlay_legend(axis, problem, objective: str, detector_face: str) -> None:
    handles: list[Line2D] = []
    if problem.source_box is not None or problem.source_kind == "left_boundary":
        handles.append(
            Line2D(
                [0],
                [0],
                color="cyan",
                linewidth=2.0,
                linestyle="--" if problem.source_box is not None else "-",
                label="source",
            )
        )
    if objective == FULL_FIELD_OBJECTIVE:
        handles.append(
            Line2D([0], [0], color="lime", linewidth=1.5, label="far-field metric")
        )
    else:
        detector_label = (
            "all-boundary detector"
            if detector_face == "all"
            else f"{detector_face}-boundary detector"
        )
        handles.append(
            Line2D([0], [0], color="red", linewidth=3.0, label=detector_label)
        )
    axis.legend(handles=handles, loc="upper left", fontsize=7)


def _far_field_label(problem) -> str:
    return (
        "Downstream far field"
        if problem.source_kind == "left_boundary"
        else "Source-distant far field"
    )


def _plot_flux_uncertainty_maps(
    metadata: dict[str, object],
    data,
    problem,
    output_path: Path,
    dpi: int,
) -> bool:
    objective = _objective(metadata)
    if objective != FULL_FIELD_OBJECTIVE:
        return False
    required = {
        "analog_flux_relative_error",
        "importance_flux_relative_error",
        "analog_flux_contributing_roots",
        "importance_flux_contributing_roots",
        "importance_levels",
    }
    if not required.issubset(data.files):
        return False

    detector_face = str(metadata.get("detector", {}).get("face", "right"))
    analog_error = np.asarray(data["analog_flux_relative_error"], dtype=np.float64)
    importance_error = np.asarray(
        data["importance_flux_relative_error"],
        dtype=np.float64,
    )
    analog_roots = np.asarray(data["analog_flux_contributing_roots"], dtype=np.float64)
    importance_roots = np.asarray(
        data["importance_flux_contributing_roots"],
        dtype=np.float64,
    )
    levels = np.asarray(data["importance_levels"], dtype=np.float64)
    far_field_mask = _far_field_mask(data, problem)
    if not np.any(far_field_mask):
        return False
    far_field_low_flux_mask = (
        np.asarray(data["far_field_low_flux_mask"], dtype=bool)
        if "far_field_low_flux_mask" in data.files
        else np.zeros_like(far_field_mask)
    )
    if far_field_low_flux_mask.shape != far_field_mask.shape:
        raise ValueError("far_field_low_flux_mask does not match the problem mesh")

    analog_valid = (
        np.isfinite(analog_error) & (analog_error > 0.0) & (analog_roots > 0.0)
    )
    importance_valid = (
        np.isfinite(importance_error)
        & (importance_error > 0.0)
        & (importance_roots > 0.0)
    )
    valid_error = np.concatenate(
        [
            analog_error[analog_valid],
            importance_error[importance_valid],
        ]
    )
    if valid_error.size == 0:
        return False

    error_floor = 10.0 ** np.floor(
        np.log10(max(float(np.quantile(valid_error, 0.01)), 1.0e-6))
    )
    error_ceiling = 10.0 ** np.ceil(
        np.log10(max(float(np.quantile(valid_error, 0.99)), 1.0))
    )

    def log_error(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        plotted = np.full_like(values, np.nan)
        plotted[valid] = np.log10(np.clip(values[valid], error_floor, error_ceiling))
        return plotted

    common = analog_valid & importance_valid
    map_common = common & (analog_roots >= 2.0) & (importance_roots >= 2.0)
    analog_runtime = float(metadata["analog"]["runtime_s"])
    importance_runtime = float(metadata["importance_vr"]["runtime_s"])
    paired_log2_fom_gain = np.full_like(analog_error, np.nan)
    paired_log2_fom_gain[common] = (
        2.0 * np.log2(analog_error[common] / importance_error[common])
        + np.log2(analog_runtime / importance_runtime)
    )
    plotted_fom_gain = np.where(map_common, paired_log2_fom_gain, np.nan)
    root_gain = np.clip(
        np.log2((importance_roots + 1.0) / (analog_roots + 1.0)),
        -ROOT_GAIN_LOG2_LIMIT,
        ROOT_GAIN_LOG2_LIMIT,
    )
    extent = [0.0, problem.width, 0.0, problem.height]
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad(color="lightgray")
    gain_cmap = plt.get_cmap("RdBu_r").copy()
    gain_cmap.set_bad(color="lightgray")
    absorption_ratio = _absorption_ratio(problem)
    absorber_mask = absorption_ratio >= _strong_absorber_threshold(metadata)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    material = axes[0, 0].imshow(
        absorption_ratio,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("Absorption probability per collision")
    fig.colorbar(material, ax=axes[0, 0], label="$\\Sigma_a/\\Sigma_t$")

    level_image = axes[0, 1].imshow(
        levels,
        origin="lower",
        extent=extent,
        cmap="plasma",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 1].set_title("Cellwise weight-window importance levels")
    fig.colorbar(level_image, ax=axes[0, 1], label="importance level")

    root_image = axes[0, 2].imshow(
        root_gain,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-ROOT_GAIN_LOG2_LIMIT,
        vmax=ROOT_GAIN_LOG2_LIMIT,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 2].set_title("Independent root-history coverage gain (clipped)")
    fig.colorbar(
        root_image,
        ax=axes[0, 2],
        label="$\\log_2[(N_{root,VR}+1)/(N_{root,A}+1)]$",
        extend="both",
    )

    error_images = []
    for axis, values, title in (
        (axes[1, 0], (analog_error, analog_valid), "Analog cellwise relative error"),
        (
            axes[1, 1],
            (importance_error, importance_valid),
            "Weight-window cellwise relative error",
        ),
    ):
        image = axis.imshow(
            log_error(*values),
            origin="lower",
            extent=extent,
            cmap=error_cmap,
            vmin=np.log10(error_floor),
            vmax=np.log10(error_ceiling),
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)
        error_images.append(image)
    fig.colorbar(
        error_images[0],
        ax=axes[1, :2],
        label="$\\log_{10}(R)$",
        extend="both",
    )

    gain_image = axes[1, 2].imshow(
        plotted_fom_gain,
        origin="lower",
        extent=extent,
        cmap=gain_cmap,
        vmin=-FOM_GAIN_LOG2_LIMIT,
        vmax=FOM_GAIN_LOG2_LIMIT,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 2].set_title(
        "Cellwise transport-stage FOM gain, paired cells with ≥2 roots/method"
    )
    fig.colorbar(
        gain_image,
        ax=axes[1, 2],
        label="$\\log_2(FOM^{transport}_{VR}/FOM^{transport}_A)$",
        extend="both",
    )

    for axis in axes.ravel():
        if np.any(absorber_mask) and np.any(~absorber_mask):
            axis.contour(
                absorber_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="cyan" if axis is axes[0, 0] else "black",
                linewidths=0.7,
                alpha=0.8,
            )
        _objective_overlay(
            axis,
            problem,
            objective,
            detector_face,
            far_field_mask,
        )
        if np.any(far_field_low_flux_mask):
            axis.contour(
                far_field_low_flux_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="white",
                linewidths=0.55,
                linestyles="--",
                alpha=0.8,
            )
        _format_spatial_axis(axis, problem)
    _overlay_legend(axes[0, 0], problem, objective, detector_face)

    far_field_label = _far_field_label(problem)
    far_common = common & far_field_mask
    if not np.any(far_common):
        plt.close(fig)
        return False
    far_analog_error = float(np.median(analog_error[far_common]))
    far_importance_error = float(np.median(importance_error[far_common]))
    far_fom_gain = float(np.median(np.exp2(paired_log2_fom_gain[far_common])))
    far_analog_coverage = float(np.mean(analog_roots[far_field_mask] > 0.0))
    far_importance_coverage = float(np.mean(importance_roots[far_field_mask] > 0.0))
    far_map_support = float(np.mean(map_common[far_field_mask]))
    fig.suptitle(
        f"{problem.name}: {far_field_label.lower()} flux is the primary metric "
        f"({objective})\n"
        f"far-field median R: {far_analog_error:.3f} → {far_importance_error:.3f}; "
        f"median transport-stage FOM gain (all paired scored cells): "
        f"{far_fom_gain:.2f}×; "
        f"coverage: {100.0 * far_analog_coverage:.1f}% → "
        f"{100.0 * far_importance_coverage:.1f}%; "
        f"FOM-map support (≥2 roots/method): {100.0 * far_map_support:.1f}%",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    far_indices = np.argwhere(far_field_mask)
    iy_min, ix_min = np.min(far_indices, axis=0)
    iy_max, ix_max = np.max(far_indices, axis=0)
    x_min = ix_min * problem.dx
    x_max = (ix_max + 1) * problem.dx
    y_min = iy_min * problem.dy
    y_max = (iy_max + 1) * problem.dy
    zoom_path = output_path.with_name("far_field_uncertainty_maps.png")
    region_aspect = (x_max - x_min) / max(y_max - y_min, np.finfo(np.float64).tiny)
    zoom_width = 10.0 if region_aspect < 0.6 else 16.0
    zoom_fig, zoom_axes = plt.subplots(
        1,
        3,
        figsize=(zoom_width, 5.2),
        constrained_layout=True,
    )
    zoom_error_images = []
    for axis, values, title in (
        (
            zoom_axes[0],
            log_error(analog_error, analog_valid),
            "Analog relative error",
        ),
        (
            zoom_axes[1],
            log_error(importance_error, importance_valid),
            "Weight-window relative error",
        ),
    ):
        far_field_values = np.where(far_field_mask, values, np.nan)
        image = axis.imshow(
            far_field_values,
            origin="lower",
            extent=extent,
            cmap=error_cmap,
            vmin=np.log10(error_floor),
            vmax=np.log10(error_ceiling),
            interpolation="nearest",
            aspect="equal",
        )
        zoom_error_images.append(image)
        if np.any(far_field_low_flux_mask):
            axis.contour(
                far_field_low_flux_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="white",
                linewidths=0.8,
                linestyles="--",
            )
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
        axis.set_title(title)
    zoom_fig.colorbar(
        zoom_error_images[0],
        ax=zoom_axes[:2],
        label="$\\log_{10}(R)$",
        extend="both",
    )
    zoom_gain = zoom_axes[2].imshow(
        np.where(far_field_mask, plotted_fom_gain, np.nan),
        origin="lower",
        extent=extent,
        cmap=gain_cmap,
        vmin=-FOM_GAIN_LOG2_LIMIT,
        vmax=FOM_GAIN_LOG2_LIMIT,
        interpolation="nearest",
        aspect="equal",
    )
    if np.any(far_field_low_flux_mask):
        zoom_axes[2].contour(
            far_field_low_flux_mask,
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="cyan",
            linewidths=0.8,
            linestyles="--",
        )
    zoom_axes[2].set_xlim(x_min, x_max)
    zoom_axes[2].set_ylim(y_min, y_max)
    zoom_axes[2].set_xlabel("x [cm]")
    zoom_axes[2].set_ylabel("y [cm]")
    zoom_axes[2].set_title(
        "Transport-stage FOM gain\n(≥2 nonzero contributing roots/method)"
    )
    zoom_fig.colorbar(
        zoom_gain,
        ax=zoom_axes[2],
        label="$\\log_2(FOM^{transport}_{VR}/FOM^{transport}_A)$",
        extend="both",
    )
    zoom_fig.suptitle(
        f"{far_field_label}: median R {far_analog_error:.3f} → "
        f"{far_importance_error:.3f}, median transport FOM gain {far_fom_gain:.2f}×; "
        f"FOM-map support {100.0 * far_map_support:.1f}%",
        fontsize=14,
    )
    zoom_fig.savefig(zoom_path, dpi=dpi, bbox_inches="tight")
    plt.close(zoom_fig)
    return True


def _plot_population_control_maps(
    metadata: dict[str, object],
    data,
    problem,
    detector_face: str,
    output_path: Path,
    dpi: int,
) -> bool:
    required = {
        "analog_cell_visits",
        "importance_cell_visits",
        "split_events_map",
        "split_children_map",
        "roulette_kills_map",
    }
    if not required.issubset(data.files):
        return False

    objective = _objective(metadata)
    far_field_mask = (
        _far_field_mask(data, problem) if objective == FULL_FIELD_OBJECTIVE else None
    )
    extent = [0.0, problem.width, 0.0, problem.height]
    analog_visits = np.asarray(data["analog_cell_visits"], dtype=np.float64)
    importance_visits = np.asarray(data["importance_cell_visits"], dtype=np.float64)
    split_events = np.asarray(data["split_events_map"], dtype=np.float64)
    split_children = np.asarray(data["split_children_map"], dtype=np.float64)
    split_cap_hits = (
        np.asarray(data["split_cap_hits_map"], dtype=np.float64)
        if "split_cap_hits_map" in data.files
        else np.zeros_like(split_children)
    )
    roulette_kills = np.asarray(data["roulette_kills_map"], dtype=np.float64)
    levels = np.asarray(data["importance_levels"], dtype=np.float64)
    histories = max(int(metadata.get("importance_vr", {}).get("histories", 1)), 1)
    visit_gain = np.clip(
        np.log2((importance_visits + 1.0) / (analog_visits + 1.0)),
        -ROOT_GAIN_LOG2_LIMIT,
        ROOT_GAIN_LOG2_LIMIT,
    )
    split_cap_field = np.log10(1.0 + split_cap_hits / histories)
    split_cap_ceiling = max(float(np.max(split_cap_field)), 1.0e-6)
    population = metadata.get("population_control", {})
    absorption_ratio = _absorption_ratio(problem)
    absorber_mask = absorption_ratio >= _strong_absorber_threshold(metadata)

    if np.any(split_cap_hits):
        middle_field = split_cap_field
        middle_title = "Split-cap hits per source history"
        middle_label = "$\\log_{10}(1+N_{cap}/N_{source})$"
        middle_limits = (0.0, split_cap_ceiling)
    else:
        middle_field = np.log10(1.0 + split_events / histories)
        middle_title = "Split events per source history (no cap hits)"
        middle_label = "$\\log_{10}(1+N_{event}/N_{source})$"
        middle_limits = None

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    panels = (
        (
            axes[0, 0],
            absorption_ratio,
            "Absorption probability per collision",
            "magma",
            "$\\Sigma_a/\\Sigma_t$",
            (0.0, 1.0),
        ),
        (
            axes[0, 1],
            levels,
            "Compiled importance levels",
            "plasma",
            "level",
            None,
        ),
        (
            axes[0, 2],
            visit_gain,
            "All-particle cell-visit gain (clipped)",
            "RdBu_r",
            "$\\log_2[(N_{VR}+1)/(N_A+1)]$",
            (-ROOT_GAIN_LOG2_LIMIT, ROOT_GAIN_LOG2_LIMIT),
        ),
        (
            axes[1, 0],
            np.log10(1.0 + split_children / histories),
            "Split children created per source history",
            "inferno",
            "$\\log_{10}(1+N_{child}/N_{source})$",
            None,
        ),
        (
            axes[1, 1],
            middle_field,
            middle_title,
            "inferno",
            middle_label,
            middle_limits,
        ),
        (
            axes[1, 2],
            np.log10(1.0 + roulette_kills / histories),
            "Roulette deaths per source history",
            "cividis",
            "$\\log_{10}(1+N_{kill}/N_{source})$",
            None,
        ),
    )
    for axis, field, title, cmap, label, limits in panels:
        kwargs = {}
        if limits is not None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            **kwargs,
        )
        if np.any(absorber_mask) and np.any(~absorber_mask):
            axis.contour(
                absorber_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="white" if cmap in {"magma", "inferno", "cividis"} else "black",
                linewidths=0.65,
                alpha=0.8,
            )
        _objective_overlay(
            axis,
            problem,
            objective,
            detector_face,
            far_field_mask,
        )
        axis.set_title(title)
        _format_spatial_axis(axis, problem)
        fig.colorbar(
            image,
            ax=axis,
            label=label,
            extend="both" if limits == (-ROOT_GAIN_LOG2_LIMIT, ROOT_GAIN_LOG2_LIMIT) else "neither",
        )
    _overlay_legend(axes[0, 0], problem, objective, detector_face)

    children_per_root = population.get("split_children_per_root")
    corridor_gain = population.get("bypass_corridor_visit_ratio_vr_over_analog")
    downstream_gain = population.get("downstream_visit_ratio_vr_over_analog")
    absorber_gain = population.get("strong_absorber_visit_ratio_vr_over_analog")
    importance_kind = str(metadata.get("importance", {}).get("kind", "unknown"))
    if (
        importance_kind == "response-absorber"
        and absorber_gain is not None
        and corridor_gain is not None
        and children_per_root is not None
    ):
        title = (
            "Population-control audit"
            f" | absorber visits={float(absorber_gain):.2f}×"
            f" | bypass visits={float(corridor_gain):.2f}×"
            f" | children/source={float(children_per_root):.2f}"
        )
    elif corridor_gain is not None and downstream_gain is not None and children_per_root is not None:
        title = (
            "Population-control audit"
            f" | bypass visits={float(corridor_gain):.2f}×"
            f" | downstream visits={float(downstream_gain):.2f}×"
            f" | children/source={float(children_per_root):.2f}"
        )
    elif absorber_gain is not None and children_per_root is not None:
        title = (
            "Population-control audit"
            f" | absorber visits={float(absorber_gain):.2f}×"
            f" | children/source={float(children_per_root):.2f}"
        )
    else:
        title = "Population-control audit"
    fig.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_genvr_weight_windows(
    case: str,
    metadata: dict[str, object],
    data,
    problem,
    detector_face: str,
    output_path: Path,
    dpi: int,
) -> None:
    objective = _objective(metadata)
    far_field_mask = (
        _far_field_mask(data, problem) if objective == FULL_FIELD_OBJECTIVE else None
    )
    extent = [0.0, problem.width, 0.0, problem.height]
    coarse_flux = np.asarray(data["coarse_flux"], dtype=np.float64)
    reference_flux = np.asarray(data["reference_flux"], dtype=np.float64)
    target_weights = np.asarray(data["target_weights"], dtype=np.float64)
    importance_levels = np.asarray(data["importance_levels"], dtype=np.float64)
    analog_flux = np.asarray(data["analog_flux"], dtype=np.float64)
    vr_flux = np.asarray(data["importance_flux"], dtype=np.float64)

    absolute_logs, flux_floor, flux_ceiling = _absolute_log_fields(
        coarse_flux,
        reference_flux,
        analog_flux,
        vr_flux,
    )
    coarse_log, reference_log, analog_log, vr_log = absolute_logs
    shape_logs, _ = _normalized_log_fields(reference_flux, analog_flux, vr_flux)
    reference_shape_log, analog_shape_log, vr_shape_log = shape_logs
    shape_valid = (reference_flux > 0.0) & (analog_flux > 0.0) & (vr_flux > 0.0)
    has_root_support = {
        "analog_flux_contributing_roots",
        "importance_flux_contributing_roots",
    }.issubset(data.files)
    if has_root_support:
        analog_roots = np.asarray(
            data["analog_flux_contributing_roots"], dtype=np.float64
        )
        vr_roots = np.asarray(
            data["importance_flux_contributing_roots"], dtype=np.float64
        )
        shape_valid &= (analog_roots >= 2.0) & (vr_roots >= 2.0)
    shape_support_note = (
        "≥2 nonzero roots/method"
        if has_root_support
        else "legacy output: positive cells only"
    )
    analog_log_error = np.abs(analog_shape_log - reference_shape_log)
    vr_log_error = np.abs(vr_shape_log - reference_shape_log)
    error_improvement = np.where(
        shape_valid,
        analog_log_error - vr_log_error,
        np.nan,
    )
    improvement_cmap = plt.get_cmap("RdBu_r").copy()
    improvement_cmap.set_bad(color="lightgray")
    absorption_ratio = _absorption_ratio(problem)
    absorber_mask = absorption_ratio >= _strong_absorber_threshold(metadata)

    validation = metadata.get("flux_validation", {})
    reductions = validation.get("error_reduction_percent", {}) if isinstance(validation, dict) else {}
    analog_validation = validation.get("analog", {}) if isinstance(validation, dict) else {}
    vr_validation = validation.get("genvr_weight_window", {}) if isinstance(validation, dict) else {}
    analog_regions = analog_validation.get("regions", {}) if isinstance(analog_validation, dict) else {}
    vr_regions = vr_validation.get("regions", {}) if isinstance(vr_validation, dict) else {}
    analog_all = analog_regions.get("all_cells", {}).get("normalized_rel_l1", float("nan"))
    vr_all = vr_regions.get("all_cells", {}).get("normalized_rel_l1", float("nan"))
    primary_results = metadata.get("primary_results", {})
    if not isinstance(primary_results, dict):
        primary_results = {}
    far_analog = primary_results.get("far_field_analog", {})
    far_vr = primary_results.get("far_field_importance_vr", {})
    far_analog_r = (
        float(far_analog.get("median_relative_error", float("nan")))
        if isinstance(far_analog, dict)
        else float("nan")
    )
    far_vr_r = (
        float(far_vr.get("median_relative_error", float("nan")))
        if isinstance(far_vr, dict)
        else float("nan")
    )

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
    material = axes[0, 0].imshow(
        absorption_ratio,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("Absorption probability per collision")
    fig.colorbar(material, ax=axes[0, 0], label="$\\Sigma_a/\\Sigma_t$")

    coarse_image = axes[0, 1].imshow(
        coarse_log,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=flux_floor,
        vmax=flux_ceiling,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 1].set_title("Saved GenVR/CFM guide flux")
    fig.colorbar(
        coarse_image,
        ax=axes[0, 1],
        label="$\\log_{10}(\\phi)$ (shared by all four flux panels)",
        extend="min",
    )

    target_image = axes[0, 2].imshow(
        np.log10(np.maximum(target_weights, np.finfo(np.float64).tiny)),
        origin="lower",
        extent=extent,
        cmap="cividis_r",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 2].set_title("Target particle-weight window")
    fig.colorbar(target_image, ax=axes[0, 2], label="$\\log_{10}(w_{target})$")

    level_image = axes[0, 3].imshow(
        importance_levels,
        origin="lower",
        extent=extent,
        cmap="plasma",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 3].set_title("Reciprocal-flux importance levels")
    fig.colorbar(level_image, ax=axes[0, 3], label="importance level")

    flux_fields = (
        (
            axes[1, 0],
            reference_log,
            "Fixed dense finite-angle $S_N$ comparison\n(not exact transport truth)",
        ),
        (
            axes[1, 1],
            analog_log,
            (
                f"Analog MC track-length flux\nshape L1={float(analog_all):.3f}; "
                f"far-field median R={far_analog_r:.3f}"
                if objective == FULL_FIELD_OBJECTIVE
                else (
                    "Analog MC track-length flux\n"
                    f"detector R={float(metadata['analog']['relative_error']):.3f}"
                )
            ),
        ),
        (
            axes[1, 2],
            vr_log,
            (
                f"GenVR weight-window MC\nshape L1={float(vr_all):.3f}; "
                f"far-field median R={far_vr_r:.3f}"
                if objective == FULL_FIELD_OBJECTIVE
                else (
                    "GenVR weight-window MC\n"
                    f"detector R={float(metadata['importance_vr']['relative_error']):.3f}"
                )
            ),
        ),
    )
    for axis, field, title in flux_fields:
        axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=flux_floor,
            vmax=flux_ceiling,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)

    improvement = axes[1, 3].imshow(
        error_improvement,
        origin="lower",
        extent=extent,
        cmap=improvement_cmap,
        vmin=-SHAPE_IMPROVEMENT_LIMIT,
        vmax=SHAPE_IMPROVEMENT_LIMIT,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 3].set_title(
        "Shape-error reduction vs fixed dense $S_N$ (red = VR closer)\n"
        f"all={float(reductions.get('all_cells', float('nan'))):.1f}%, "
        f"absorber={float(reductions.get('absorber_cells', float('nan'))):.1f}%, "
        f"low-flux={float(reductions.get('lowest_flux_quartile', float('nan'))):.1f}%; "
        f"support={100.0 * float(np.mean(shape_valid)):.1f}% ({shape_support_note})"
    )
    fig.colorbar(
        improvement,
        ax=axes[1, 3],
        label="decrease in $|\\Delta\\log_{10}(\\phi/\\phi_{max})|$",
        extend="both",
    )

    for axis in axes.ravel():
        if np.any(absorber_mask) and np.any(~absorber_mask):
            axis.contour(
                absorber_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="white" if axis is axes[0, 0] else "black",
                linewidths=0.55,
                alpha=0.7,
            )
        _objective_overlay(
            axis,
            problem,
            objective,
            detector_face,
            far_field_mask,
        )
        _format_spatial_axis(axis, problem)
    _overlay_legend(axes[0, 0], problem, objective, detector_face)
    fig.suptitle(
        f"{case.capitalize()}: GenVR guide flux → reciprocal importance → "
        "unbiased weight-window MC\n"
        "All flux panels share one absolute log scale; shape error uses "
        "individually normalized fields",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_gmc_weight_window_mc_field(
    case: str,
    metadata: dict[str, object],
    data,
    problem,
    detector_face: str,
    output_path: Path,
    dpi: int,
) -> bool:
    required = {
        "coarse_flux",
        "target_weights",
        "importance_flux",
        "importance_flux_relative_error",
        "importance_flux_contributing_roots",
        "analog_flux_relative_error",
        "analog_flux_contributing_roots",
    }
    if _objective(metadata) != FULL_FIELD_OBJECTIVE or not required.issubset(data.files):
        return False

    extent = [0.0, problem.width, 0.0, problem.height]
    far_field_mask = _far_field_mask(data, problem)
    coarse_flux = np.asarray(data["coarse_flux"], dtype=np.float64)
    target_weights = np.asarray(data["target_weights"], dtype=np.float64)
    mc_flux = np.asarray(data["importance_flux"], dtype=np.float64)
    mc_relative_error = np.asarray(
        data["importance_flux_relative_error"], dtype=np.float64
    )
    mc_roots = np.asarray(
        data["importance_flux_contributing_roots"], dtype=np.float64
    )
    analog_relative_error = np.asarray(
        data["analog_flux_relative_error"], dtype=np.float64
    )
    analog_roots = np.asarray(
        data["analog_flux_contributing_roots"], dtype=np.float64
    )

    flux_logs, flux_floor, flux_ceiling = _absolute_log_fields(coarse_flux, mc_flux)
    coarse_log, mc_log = flux_logs
    target_log = np.log10(np.maximum(target_weights, np.finfo(np.float64).tiny))
    root_log = np.log10(1.0 + mc_roots)

    mc_error_valid = np.isfinite(mc_relative_error) & (mc_relative_error > 0.0)
    paired_valid = (
        mc_error_valid
        & np.isfinite(analog_relative_error)
        & (analog_relative_error > 0.0)
        & (mc_roots >= 2.0)
        & (analog_roots >= 2.0)
    )
    if not np.any(mc_error_valid) or not np.any(paired_valid):
        return False
    valid_errors = mc_relative_error[mc_error_valid]
    error_floor = max(float(np.quantile(valid_errors, 0.01)), 1.0e-6)
    error_ceiling = max(float(np.quantile(valid_errors, 0.99)), error_floor * 10.0)
    plotted_error = np.full_like(mc_relative_error, np.nan)
    plotted_error[mc_error_valid] = np.log10(
        np.clip(mc_relative_error[mc_error_valid], error_floor, error_ceiling)
    )
    precision_gain = np.full_like(mc_relative_error, np.nan)
    precision_gain[paired_valid] = np.log2(
        analog_relative_error[paired_valid] / mc_relative_error[paired_valid]
    )
    gain_limit = min(
        ROOT_GAIN_LOG2_LIMIT,
        max(1.0, float(np.ceil(np.quantile(np.abs(precision_gain[paired_valid]), 0.98)))),
    )

    primary = metadata.get("primary_results", {})
    if not isinstance(primary, dict):
        primary = {}
    far_analog = primary.get("far_field_analog", {})
    far_mc = primary.get("far_field_importance_vr", {})
    far_comparison = primary.get("far_field_comparison", {})
    far_analog_r = float(far_analog.get("median_relative_error", float("nan")))
    far_mc_r = float(far_mc.get("median_relative_error", float("nan")))
    far_analog_roots = float(far_analog.get("median_contributing_roots", float("nan")))
    far_mc_roots = float(far_mc.get("median_contributing_roots", float("nan")))
    far_precision_gain = float(
        far_comparison.get(
            "median_cellwise_relative_error_ratio_analog_over_vr",
            float("nan"),
        )
    )
    importance_metadata = metadata.get("importance", {})
    if not isinstance(importance_metadata, dict):
        importance_metadata = {}
    flux_exponent = float(importance_metadata.get("flux_exponent", float("nan")))
    n_levels = int(importance_metadata.get("n_levels", int(np.nanmax(data["importance_levels"]))))
    split_factor = int(importance_metadata.get("split_factor", 2))
    histories = int(metadata.get("importance_vr", {}).get("histories", 0))

    absorption_ratio = _absorption_ratio(problem)
    absorber_mask = absorption_ratio >= _strong_absorber_threshold(metadata)
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad(color="lightgray")
    gain_cmap = plt.get_cmap("RdBu_r").copy()
    gain_cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    guide_image = axes[0, 0].imshow(
        coarse_log,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=flux_floor,
        vmax=flux_ceiling,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("1  Fast GMC/CFM guide flux\n(not the final physical tally)")

    target_image = axes[0, 1].imshow(
        target_log,
        origin="lower",
        extent=extent,
        cmap="cividis_r",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 1].set_title(
        "2  GMC-compiled target particle weight\n"
        f"$w_{{target}}\\propto\\phi_{{GMC}}^{{{flux_exponent:g}}}$"
    )
    fig.colorbar(
        target_image,
        ax=axes[0, 1],
        label="$\\log_{10}(w_{target})$  (lower = more particles)",
    )

    roots_image = axes[0, 2].imshow(
        root_log,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 2].set_title(
        "3  Independent MC roots contributing per cell\n"
        f"far-field median: {far_analog_roots:.0f} analog → {far_mc_roots:.0f} GMC-WW"
    )
    fig.colorbar(roots_image, ax=axes[0, 2], label="$\\log_{10}(1+N_{root})$")

    axes[1, 0].imshow(
        mc_log,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=flux_floor,
        vmax=flux_ceiling,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 0].set_title(
        "4  Unbiased weight-window MC physical flux\n"
        "GMC changes sampling; MC remains the estimator"
    )
    fig.colorbar(
        guide_image,
        ax=[axes[0, 0], axes[1, 0]],
        label="$\\log_{10}(\\phi)$  (shared absolute scale)",
        extend="min",
    )

    error_image = axes[1, 1].imshow(
        plotted_error,
        origin="lower",
        extent=extent,
        cmap=error_cmap,
        vmin=np.log10(error_floor),
        vmax=np.log10(error_ceiling),
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 1].set_title(
        "5  MC cellwise relative uncertainty\n"
        f"far-field median R: {far_analog_r:.3f} analog → {far_mc_r:.3f} GMC-WW"
    )
    fig.colorbar(error_image, ax=axes[1, 1], label="$\\log_{10}(R)$", extend="both")

    gain_image = axes[1, 2].imshow(
        precision_gain,
        origin="lower",
        extent=extent,
        cmap=gain_cmap,
        vmin=-gain_limit,
        vmax=gain_limit,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 2].set_title(
        "6  Precision amplification from the GMC window\n"
        f"far-field median $R_A/R_{{GMC-WW}}$ = {far_precision_gain:.2f}×"
    )
    fig.colorbar(
        gain_image,
        ax=axes[1, 2],
        label="$\\log_2(R_{analog}/R_{GMC-WW})$  (red = lower uncertainty)",
        extend="both",
    )

    for axis in axes.ravel():
        if np.any(absorber_mask) and np.any(~absorber_mask):
            axis.contour(
                absorber_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="black",
                linewidths=0.5,
                alpha=0.55,
            )
        _objective_overlay(
            axis,
            problem,
            FULL_FIELD_OBJECTIVE,
            detector_face,
            far_field_mask,
        )
        _format_spatial_axis(axis, problem)
    _overlay_legend(axes[0, 0], problem, FULL_FIELD_OBJECTIVE, detector_face)

    fig.suptitle(
        f"{case.capitalize()}: GMC guide → weight window → high-precision physical MC field\n"
        f"$I=(\\phi_{{source}}/\\phi_{{GMC}})^{{{flux_exponent:g}}}$, "
        f"{n_levels} levels, split factor {split_factor}, {histories:,} independent root histories; "
        "split descendants are recombined by root before uncertainty estimation",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_cloudmaps(
    case: str,
    input_dir: Path,
    output_path: Path,
    population_output_path: Path,
    uncertainty_output_path: Path,
    gmc_mc_output_path: Path,
    dpi: int = 300,
) -> bool:
    metadata = json.loads((input_dir / "comparison.json").read_text(encoding="utf-8"))
    data = np.load(input_dir / "comparison.npz")
    analog_flux = np.asarray(data["analog_flux"], dtype=np.float64)
    importance_flux = np.asarray(data["importance_flux"], dtype=np.float64)
    importance_levels = np.asarray(data["importance_levels"], dtype=np.float64)
    if "importance_potential" in data:
        importance_potential = np.asarray(data["importance_potential"], dtype=np.float64)
    else:
        importance_potential = np.power(2.0, importance_levels)
        importance_potential /= float(np.max(importance_potential))
    ny, nx = analog_flux.shape
    problem = _problem(case, nx, ny)
    importance_kind = str(metadata.get("importance", {}).get("kind", "unknown"))
    detector_face = str(metadata.get("detector", {}).get("face", "right"))
    objective = _objective(metadata)
    far_field_mask = (
        _far_field_mask(data, problem) if objective == FULL_FIELD_OBJECTIVE else None
    )
    _plot_flux_uncertainty_maps(
        metadata,
        data,
        problem,
        uncertainty_output_path,
        dpi,
    )
    if (
        importance_kind == "genvr-flux"
        and "coarse_flux" in data.files
        and "reference_flux" in data.files
        and "target_weights" in data.files
    ):
        _plot_genvr_weight_windows(
            case,
            metadata,
            data,
            problem,
            detector_face,
            output_path,
            dpi,
        )
        _plot_gmc_weight_window_mc_field(
            case,
            metadata,
            data,
            problem,
            detector_face,
            gmc_mc_output_path,
            dpi,
        )
        return _plot_population_control_maps(
            metadata,
            data,
            problem,
            detector_face,
            population_output_path,
            dpi,
        )

    extent = [0.0, problem.width, 0.0, problem.height]
    flux_logs, log_floor, log_ceiling = _absolute_log_fields(
        analog_flux,
        importance_flux,
    )
    analog_log, importance_log = flux_logs
    flux_floor = 10.0 ** log_floor
    difference_valid = (analog_flux > 0.0) & (importance_flux > 0.0)
    has_root_support = {
        "analog_flux_contributing_roots",
        "importance_flux_contributing_roots",
    }.issubset(data.files)
    if has_root_support:
        difference_valid &= (
            np.asarray(data["analog_flux_contributing_roots"], dtype=np.float64)
            >= 2.0
        ) & (
            np.asarray(data["importance_flux_contributing_roots"], dtype=np.float64)
            >= 2.0
        )
    difference_support_note = (
        "≥2 nonzero roots/method"
        if has_root_support
        else "legacy output: no cellwise uncertainty stored"
    )
    scale = np.maximum(0.5 * (analog_flux + importance_flux), flux_floor)
    symmetric_relative_difference = np.full_like(analog_flux, np.nan)
    symmetric_relative_difference[difference_valid] = (
        (importance_flux[difference_valid] - analog_flux[difference_valid])
        / scale[difference_valid]
    )
    difference_cmap = plt.get_cmap("coolwarm").copy()
    difference_cmap.set_bad(color="lightgray")
    absorption_ratio = _absorption_ratio(problem)
    absorber_mask = absorption_ratio >= _strong_absorber_threshold(metadata)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    material = axes[0, 0].imshow(
        absorption_ratio,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("Absorption probability per collision")
    fig.colorbar(material, ax=axes[0, 0], label="$\\Sigma_a/\\Sigma_t$")

    importance_normalized = importance_potential / max(
        float(np.max(importance_potential)),
        np.finfo(np.float64).tiny,
    )
    positive_importance = importance_normalized[importance_normalized > 0.0]
    importance_floor = max(float(np.quantile(positive_importance, 0.005)), 1.0e-300)
    importance_image = axes[0, 1].imshow(
        np.log10(np.maximum(importance_normalized, importance_floor)),
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=np.log10(importance_floor),
        vmax=0.0,
        interpolation="nearest",
        aspect="equal",
    )
    importance_title = (
        "Diffusion-adjoint importance potential"
        if importance_kind == "diffusion"
        else (
            "Response adjoint + GenVR absorber importance"
            if importance_kind == "response-absorber"
            else (
                "GenVR reciprocal-flux importance"
                if importance_kind == "genvr-flux"
                else "Normalized geometric importance proxy"
            )
        )
    )
    axes[0, 1].set_title(importance_title)
    fig.colorbar(importance_image, ax=axes[0, 1], label="$\\log_{10}(I/I_{max})$")

    level_image = axes[0, 2].imshow(
        importance_levels,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 2].set_title("Compiled splitting levels")
    fig.colorbar(level_image, ax=axes[0, 2], label="importance level")

    if objective == FULL_FIELD_OBJECTIVE:
        primary_results = metadata.get("primary_results", {})
        if not isinstance(primary_results, dict):
            primary_results = {}
        analog_region = primary_results.get("far_field_analog", {})
        importance_region = primary_results.get("far_field_importance_vr", {})
        analog_metric = (
            float(analog_region.get("median_relative_error", float("nan")))
            if isinstance(analog_region, dict)
            else float("nan")
        )
        importance_metric = (
            float(importance_region.get("median_relative_error", float("nan")))
            if isinstance(importance_region, dict)
            else float("nan")
        )
        analog_title = (
            "Analog MC track-length flux\n"
            f"far-field median R={analog_metric:.3f}"
        )
        importance_flux_title = (
            "Importance-VR track-length flux\n"
            f"far-field median R={importance_metric:.3f}"
        )
    else:
        analog_title = (
            "Analog MC track-length flux\n"
            f"detector R={float(metadata['analog']['relative_error']):.3f}"
        )
        importance_flux_title = (
            "Importance-VR track-length flux\n"
            f"detector R={float(metadata['importance_vr']['relative_error']):.3f}"
        )

    analog = axes[1, 0].imshow(
        analog_log,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=log_floor,
        vmax=log_ceiling,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 0].set_title(analog_title)

    importance = axes[1, 1].imshow(
        importance_log,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=log_floor,
        vmax=log_ceiling,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 1].set_title(importance_flux_title)
    fig.colorbar(
        analog,
        ax=axes[1, :2],
        label="$\\log_{10}(\\phi)$ (shared absolute scale)",
        extend="min",
    )

    difference = axes[1, 2].imshow(
        symmetric_relative_difference,
        origin="lower",
        extent=extent,
        cmap=difference_cmap,
        norm=SymLogNorm(
            linthresh=0.05,
            linscale=1.0,
            vmin=-2.0,
            vmax=2.0,
            base=10,
        ),
        interpolation="nearest",
        aspect="equal",
    )
    contour_levels = np.unique(importance_levels)
    if contour_levels.size > 9:
        contour_levels = np.linspace(
            float(np.min(importance_levels)),
            float(np.max(importance_levels)),
            9,
        )
    contours = axes[1, 2].contour(
        importance_levels,
        levels=contour_levels,
        origin="lower",
        extent=extent,
        colors="black",
        linewidths=0.5,
        alpha=0.55,
    )
    axes[1, 2].clabel(contours, inline=True, fontsize=7, fmt="L%.0f")
    axes[1, 2].set_title(
        "Descriptive paired flux difference with importance contours\n"
        f"support={100.0 * np.mean(difference_valid):.1f}% "
        f"({difference_support_note}); not a significance test"
    )
    fig.colorbar(
        difference,
        ax=axes[1, 2],
        label="$2(\\phi_{VR}-\\phi_A)/(\\phi_{VR}+\\phi_A)$",
        extend="both",
    )

    for axis in axes.ravel():
        if np.any(absorber_mask) and np.any(~absorber_mask):
            axis.contour(
                absorber_mask,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors="white" if axis is axes[0, 0] else "black",
                linewidths=0.55,
                alpha=0.7,
            )
        _objective_overlay(
            axis,
            problem,
            objective,
            detector_face,
            far_field_mask,
        )
        _format_spatial_axis(axis, problem)
    _overlay_legend(axes[0, 0], problem, objective, detector_face)

    fig.suptitle(
        f"{case.capitalize()} benchmark: analog vs importance-splitting Monte Carlo "
        f"({objective})",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return _plot_population_control_maps(
        metadata,
        data,
        problem,
        detector_face,
        population_output_path,
        dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("lattice", "hohlraum"), required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--population-output", default=None)
    parser.add_argument("--uncertainty-output", default=None)
    parser.add_argument("--gmc-mc-output", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output) if args.output else input_dir / "flux_cloudmaps.png"
    population_output = (
        Path(args.population_output)
        if args.population_output
        else input_dir / "population_control_maps.png"
    )
    uncertainty_output = (
        Path(args.uncertainty_output)
        if args.uncertainty_output
        else input_dir / "flux_uncertainty_maps.png"
    )
    gmc_mc_output = (
        Path(args.gmc_mc_output)
        if args.gmc_mc_output
        else input_dir / "gmc_weight_window_mc_field.png"
    )
    population_created = plot_cloudmaps(
        args.case,
        input_dir,
        output,
        population_output,
        uncertainty_output,
        gmc_mc_output,
        dpi=args.dpi,
    )
    print(output)
    if uncertainty_output.exists():
        print(uncertainty_output)
        far_field_output = uncertainty_output.with_name("far_field_uncertainty_maps.png")
        if far_field_output.exists():
            print(far_field_output)
    if population_created:
        print(population_output)
    if gmc_mc_output.exists():
        print(gmc_mc_output)


if __name__ == "__main__":
    main()
