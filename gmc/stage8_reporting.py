from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np

from .benchmarks2d import Structured2DProblem, material_rgb


CASE_DISPLAY_NAMES = {
    "lattice": "Heterogeneous lattice",
    "hohlraum": "Linearized hohlraum",
    "iter_r": "ITER radial benchmark",
    "iter_a": "ITER axial benchmark",
}

BACKGROUND = "#f7f8fb"
PANEL_BACKGROUND = "#eef1f5"
TEXT = "#182230"
MUTED_TEXT = "#667085"
SPINE = "#cbd2dc"


def _styled_material_rgb(problem: Structured2DProblem) -> np.ndarray:
    original = material_rgb(problem)
    styled = np.full_like(original, (0.965, 0.973, 0.984))
    palette = (
        ((0.0, 0.0, 0.0), (0.16, 0.19, 0.24)),
        ((0.9, 0.05, 0.05), (0.86, 0.25, 0.27)),
        ((0.1, 0.6, 0.1), (0.19, 0.58, 0.36)),
        ((0.05, 0.05, 0.8), (0.18, 0.33, 0.72)),
        ((0.8, 0.05, 0.8), (0.69, 0.20, 0.62)),
        ((1.0, 1.0, 1.0), (0.965, 0.973, 0.984)),
    )
    for source, target in palette:
        mask = np.all(np.isclose(original, source, atol=1.0e-5), axis=-1)
        styled[mask] = target
    return styled


def _material_labels(problem: Structured2DProblem) -> np.ndarray:
    material_pairs = np.stack((problem.sigma_a, problem.sigma_s), axis=-1)
    _, inverse = np.unique(material_pairs.reshape(-1, 2), axis=0, return_inverse=True)
    return inverse.reshape(problem.ny, problem.nx)


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL_BACKGROUND)
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT,
        labelsize=8,
        length=3.0,
        width=0.7,
        direction="out",
        pad=2,
    )
    for spine in axis.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)


def _set_panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(
        f"({letter}) {title}",
        loc="left",
        color=TEXT,
        fontsize=10.5,
        fontweight="semibold",
        pad=8,
    )


def _metric_box(
    axis: plt.Axes,
    text: str,
    *,
    x: float = 0.025,
    y: float = 0.025,
    horizontal_alignment: str = "left",
    vertical_alignment: str = "bottom",
) -> None:
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        color="#344054",
        fontsize=8.0,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d0d5dd",
            "linewidth": 0.7,
            "alpha": 0.90,
        },
    )


def _comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    flat_reference = np.asarray(reference, dtype=np.float64).ravel()
    flat_candidate = np.asarray(candidate, dtype=np.float64).ravel()
    normalized_reference = flat_reference / max(float(np.max(flat_reference)), 1.0e-300)
    normalized_candidate = flat_candidate / max(float(np.max(flat_candidate)), 1.0e-300)
    if np.std(normalized_reference) == 0.0 or np.std(normalized_candidate) == 0.0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(normalized_reference, normalized_candidate)[0, 1])
    relative_l1 = float(
        np.sum(np.abs(normalized_candidate - normalized_reference))
        / max(np.sum(np.abs(normalized_reference)), 1.0e-300)
    )
    return correlation, relative_l1


def plot_stage8_case(
    problem: Structured2DProblem,
    cfm: np.ndarray,
    projected_mc: np.ndarray | None,
    dense: np.ndarray | None,
    global_mc: np.ndarray | None,
    output_dir: str | Path,
    case: str,
    *,
    n_pos: int | None = None,
    n_mu: int | None = None,
    n_phi: int | None = None,
    mc_histories: int | None = None,
    dense_note: str | None = None,
    dense_spatial_scheme: str = "diamond",
    dpi: int = 190,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("cfm", np.asarray(cfm, dtype=np.float64)),
        ("dense", None if dense is None else np.asarray(dense, dtype=np.float64)),
        (
            "global_mc",
            None if global_mc is None else np.asarray(global_mc, dtype=np.float64),
        ),
    ]
    available = [values for _, values in fields if values is not None]
    if not available:
        raise ValueError("at least one Stage-8 flux field is required")
    expected_shape = available[0].shape
    if any(values.ndim != 2 or values.shape != expected_shape for values in available):
        raise ValueError("all Stage-8 flux fields must be two-dimensional with equal shapes")
    if any(not np.all(np.isfinite(values)) or np.any(values < 0.0) for values in available):
        raise ValueError("Stage-8 flux fields must be finite and nonnegative")
    if dense_spatial_scheme == "diamond":
        dense_title = r"Dense diamond-difference $S_N$"
        dense_footer = (
            r"Dense $S_N$ uses diamond difference with conservative zero-flux fixup "
            "and still requires grid convergence; "
        )
    elif dense_spatial_scheme == "upwind":
        dense_title = r"Dense positive-upwind $S_N$"
        dense_footer = (
            r"Dense $S_N$ uses the legacy positive-upwind closure and still requires "
            "grid convergence; "
        )
    else:
        raise ValueError("dense_spatial_scheme must be 'diamond' or 'upwind'")

    common_scale = max(max(float(np.max(values)) for values in available), 1.0e-300)
    normalized = {
        method: None if values is None else values / common_scale
        for method, values in fields
    }
    scale_fields = [
        normalized[method]
        for method in ("cfm", "dense")
        if normalized[method] is not None
    ]
    if not scale_fields:
        scale_fields = [values / common_scale for values in available]
    positive_parts = [values[values > 0.0] for values in scale_fields if np.any(values > 0.0)]
    positive = np.concatenate(positive_parts) if positive_parts else np.array([1.0e-10, 1.0])
    raw_floor = max(float(np.quantile(positive, 1.0e-3)), 1.0e-300)
    flux_vmin = max(10.0 ** np.floor(np.log10(raw_floor)), 1.0e-10)
    flux_vmin = min(flux_vmin, 1.0e-1)
    extent = [0.0, problem.width, 0.0, problem.height]

    figure = plt.figure(figsize=(14.8, 8.8), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        left=0.065,
        right=0.94,
        bottom=0.115,
        top=0.835,
        wspace=0.18,
        hspace=0.28,
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[0, column]) for column in range(3)],
            [figure.add_subplot(grid[1, column]) for column in range(3)],
        ],
        dtype=object,
    )
    flux_color_axis = figure.add_subplot(grid[0, 3])
    ratio_color_axis = figure.add_subplot(grid[1, 3])

    _set_panel_title(axes[0, 0], "a", "Material layout")
    axes[0, 0].imshow(
        _styled_material_rgb(problem),
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )

    field_axes = (axes[0, 1], axes[0, 2], axes[1, 0])
    panel_titles = {
        "cfm": ("b", "GenVR / CFM"),
        "dense": ("c", dense_title),
        "global_mc": (
            "d",
            "Global MC"
            + (f" · {mc_histories:,} histories" if mc_histories else " · finite histories"),
        ),
    }
    flux_cmap = plt.get_cmap("cividis").copy()
    flux_cmap.set_bad(PANEL_BACKGROUND)
    flux_image = None
    material_labels = _material_labels(problem)
    x_centers = (np.arange(problem.nx) + 0.5) * problem.dx
    y_centers = (np.arange(problem.ny) + 0.5) * problem.dy
    boundary_levels = np.arange(int(np.max(material_labels))) + 0.5

    for axis, (method, values) in zip(field_axes, fields, strict=True):
        letter, title = panel_titles[method]
        _set_panel_title(axis, letter, title)
        if values is None:
            axis.text(
                0.5,
                0.5,
                "Not generated",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color=MUTED_TEXT,
                fontsize=10,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            continue
        shown = np.ma.masked_less_equal(normalized[method], 0.0)
        flux_image = axis.imshow(
            shown,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            cmap=flux_cmap,
            norm=LogNorm(vmin=flux_vmin, vmax=1.0, clip=True),
        )
        if boundary_levels.size:
            axis.contour(
                x_centers,
                y_centers,
                material_labels,
                levels=boundary_levels,
                colors="white",
                linewidths=0.35,
                alpha=0.20,
            )
        if method == "global_mc":
            nonzero_fraction = float(np.mean(values > 0.0))
            _metric_box(axis, f"positive tallies  {100.0 * nonzero_fraction:.1f}%")
        elif method == "dense" and dense_note:
            _metric_box(axis, dense_note)

    if flux_image is None:
        flux_color_axis.set_axis_off()
    else:
        flux_colorbar = figure.colorbar(flux_image, cax=flux_color_axis)
        flux_colorbar.set_label(
            "Scalar flux / shared maximum",
            color="#344054",
            fontsize=9,
            labelpad=10,
        )
        minimum_exponent = int(np.floor(np.log10(flux_vmin)))
        exponent_step = max(1, int(np.ceil(-minimum_exponent / 5)))
        tick_exponents = list(range(minimum_exponent, 1, exponent_step))
        if 0 not in tick_exponents:
            tick_exponents.append(0)
        flux_colorbar.set_ticks(10.0 ** np.asarray(sorted(set(tick_exponents))))
        flux_colorbar.ax.tick_params(colors=MUTED_TEXT, labelsize=8, width=0.7, length=3)
        flux_colorbar.outline.set_edgecolor(SPINE)
        flux_colorbar.outline.set_linewidth(0.8)

    comparison = None if dense is None else np.asarray(dense, dtype=np.float64)
    diagnostic_axis = axes[1, 1]
    agreement_axis = axes[1, 2]
    _set_panel_title(diagnostic_axis, "e", r"CFM / Dense $S_N$ shape ratio")
    _set_panel_title(agreement_axis, "f", "Cellwise shape agreement")
    if comparison is None:
        for axis in (diagnostic_axis, agreement_axis):
            axis.text(
                0.5,
                0.5,
                r"Dense $S_N$ not generated",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color=MUTED_TEXT,
                fontsize=10,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        ratio_color_axis.set_axis_off()
    else:
        normalized_cfm = np.asarray(cfm, dtype=np.float64) / max(
            float(np.max(cfm)), 1.0e-300
        )
        normalized_comparison = comparison / max(float(np.max(comparison)), 1.0e-300)
        comparison_floor = flux_vmin
        signal = np.maximum(normalized_cfm, normalized_comparison)
        log_ratio = np.log10(
            (normalized_cfm + comparison_floor)
            / (normalized_comparison + comparison_floor)
        )
        low_signal = signal < flux_vmin
        shown_ratio = np.ma.array(log_ratio, mask=low_signal)
        visible_ratio = np.abs(log_ratio[~low_signal])
        raw_limit = float(np.quantile(visible_ratio, 0.99)) if visible_ratio.size else 0.25
        ratio_limit = min(2.0, max(0.25, np.ceil(raw_limit * 4.0) / 4.0))
        ratio_cmap = plt.get_cmap("RdBu_r").copy()
        ratio_cmap.set_bad(PANEL_BACKGROUND)
        ratio_image = diagnostic_axis.imshow(
            shown_ratio,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            cmap=ratio_cmap,
            norm=TwoSlopeNorm(vmin=-ratio_limit, vcenter=0.0, vmax=ratio_limit),
        )
        if boundary_levels.size:
            diagnostic_axis.contour(
                x_centers,
                y_centers,
                material_labels,
                levels=boundary_levels,
                colors="#475467",
                linewidths=0.35,
                alpha=0.22,
            )
        correlation, relative_l1 = _comparison_metrics(comparison, np.asarray(cfm))
        correlation_text = "n/a" if not np.isfinite(correlation) else f"{correlation:.4f}"
        _metric_box(
            diagnostic_axis,
            f"corr  {correlation_text}\nshape L1  {100.0 * relative_l1:.1f}%",
        )
        ratio_colorbar = figure.colorbar(ratio_image, cax=ratio_color_axis)
        ratio_colorbar.set_label(
            r"$\log_{10}$(CFM shape / Dense $S_N$ shape)",
            color="#344054",
            fontsize=9,
            labelpad=10,
        )
        ratio_colorbar.set_ticks(np.linspace(-ratio_limit, ratio_limit, 5))
        ratio_colorbar.ax.tick_params(colors=MUTED_TEXT, labelsize=8, width=0.7, length=3)
        ratio_colorbar.outline.set_edgecolor(SPINE)
        ratio_colorbar.outline.set_linewidth(0.8)

        agreement_mask = (
            (~low_signal)
            & (normalized_cfm > 0.0)
            & (normalized_comparison > 0.0)
        )
        dense_shape = np.clip(normalized_comparison[agreement_mask], flux_vmin, 1.0)
        cfm_shape = np.clip(normalized_cfm[agreement_mask], flux_vmin, 1.0)
        guide = np.geomspace(flux_vmin, 1.0, 256)
        agreement_axis.fill_between(
            guide,
            np.maximum(flux_vmin, guide / 1.25),
            np.minimum(1.0, guide * 1.25),
            color="#12b76a",
            alpha=0.10,
            linewidth=0.0,
        )
        agreement_axis.scatter(
            dense_shape,
            cfm_shape,
            s=3.0,
            color="#4169a1",
            alpha=0.12,
            edgecolors="none",
            rasterized=True,
        )
        agreement_axis.plot(guide, guide, color="#344054", linewidth=1.2)
        agreement_axis.set_xscale("log")
        agreement_axis.set_yscale("log")
        agreement_axis.set_xlim(flux_vmin, 1.0)
        agreement_axis.set_ylim(flux_vmin, 1.0)
        agreement_axis.grid(True, which="major", color="#d9dee7", linewidth=0.6, alpha=0.65)
        agreement_axis.set_xlabel(r"Normalized Dense $S_N$", color="#344054", fontsize=9)
        agreement_axis.set_ylabel("Normalized GenVR / CFM", color="#344054", fontsize=9)
        agreement_axis.text(
            0.03,
            0.97,
            "green band: within 25%",
            transform=agreement_axis.transAxes,
            ha="left",
            va="top",
            color=MUTED_TEXT,
            fontsize=7.8,
        )
        peak_ratio = float(np.max(cfm)) / max(float(np.max(comparison)), 1.0e-300)
        integral_ratio = float(np.sum(cfm)) / max(float(np.sum(comparison)), 1.0e-300)
        _metric_box(
            agreement_axis,
            f"peak ratio  {peak_ratio:.3f}\nintegral ratio  {integral_ratio:.3f}",
            x=0.975,
            horizontal_alignment="right",
        )

    spatial_axes = (
        (axes[0, 0], 0, 0),
        (axes[0, 1], 0, 1),
        (axes[0, 2], 0, 2),
        (axes[1, 0], 1, 0),
        (diagnostic_axis, 1, 1),
    )
    for axis, row, column in spatial_axes:
        _style_axis(axis)
        if axis.axison and len(axis.get_xticks()) > 0:
            axis.set_xlim(0.0, problem.width)
            axis.set_ylim(0.0, problem.height)
            if row == 1:
                axis.set_xlabel("x [cm]", color="#344054", fontsize=9, labelpad=3)
            else:
                axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel("y [cm]", color="#344054", fontsize=9, labelpad=3)
            else:
                axis.tick_params(labelleft=False)
    _style_axis(agreement_axis)

    phase_space = ""
    if n_pos is not None and n_mu is not None and n_phi is not None:
        phase_space = f"Interface phase space {n_pos} × {n_mu} × {n_phi}"
    history_summary = (
        f"Global MC {mc_histories:,} histories" if mc_histories else "Finite-history global MC"
    )
    metadata = "  ·  ".join(
        item
        for item in (
            "Shared absolute flux scale",
            f"Diagnostic comparison: {dense_title}",
            phase_space,
            history_summary,
        )
        if item
    )
    case_name = CASE_DISPLAY_NAMES.get(case, case)
    figure.text(
        0.065,
        0.955,
        f"Stage-8 transport comparison — {case_name}",
        ha="left",
        va="top",
        color=TEXT,
        fontsize=17,
        fontweight="semibold",
    )
    figure.text(
        0.065,
        0.910,
        metadata,
        ha="left",
        va="top",
        color=MUTED_TEXT,
        fontsize=10,
    )
    figure.text(
        0.065,
        0.035,
        "Light gray marks zero-tally MC cells or below-scale ratio cells. "
        + dense_footer
        + "global MC is qualitative at the displayed history count.",
        ha="left",
        va="bottom",
        color=MUTED_TEXT,
        fontsize=8.5,
    )

    output_path = output_dir / f"{case}_stage8_ultra.png"
    figure.savefig(
        output_path,
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.16,
    )
    plt.close(figure)
    return output_path
