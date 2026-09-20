#!/usr/bin/env python3
"""Compare detector-only and absorber-aware hohlraum importance strategies."""
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
import numpy as np

from gmc.benchmarks2d import make_hohlraum_problem


def _load(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    with np.load(directory / "comparison.npz") as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return metadata, arrays


def _visit_gain(arrays: dict[str, np.ndarray]) -> np.ndarray:
    analog = arrays["analog_cell_visits"].astype(np.float64)
    variance_reduced = arrays["importance_cell_visits"].astype(np.float64)
    return np.log2((variance_reduced + 1.0) / (analog + 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector-only-dir",
        type=Path,
        default=ROOT
        / "outputs/importance_vr_validation/highres/hohlraum_112_diffusion_L10",
    )
    parser.add_argument(
        "--absorber-aware-dir",
        type=Path,
        default=ROOT
        / "outputs/importance_vr_validation/highres/hohlraum_112_response_absorber_L10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/importance_vr_validation/highres/"
        "hohlraum_importance_strategy_comparison.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    detector_metadata, detector_arrays = _load(args.detector_only_dir)
    hybrid_metadata, hybrid_arrays = _load(args.absorber_aware_dir)
    detector_levels = detector_arrays["importance_levels"].astype(np.float64)
    hybrid_levels = hybrid_arrays["importance_levels"].astype(np.float64)
    if detector_levels.shape != hybrid_levels.shape:
        raise ValueError("the two strategies must use the same mesh")

    ny, nx = detector_levels.shape
    problem = make_hohlraum_problem(nx, ny)
    extent = [0.0, problem.width, 0.0, problem.height]
    detector_gain = _visit_gain(detector_arrays)
    hybrid_gain = _visit_gain(hybrid_arrays)
    visit_limit = max(
        float(np.quantile(np.abs(np.concatenate([detector_gain.ravel(), hybrid_gain.ravel()])), 0.98)),
        1.0,
    )
    detector_population = detector_metadata["population_control"]
    hybrid_population = hybrid_metadata["population_control"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    material = axes[0, 0].imshow(
        problem.sigma_a,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("Material absorption $\\Sigma_a$")
    fig.colorbar(material, ax=axes[0, 0], label="$\\Sigma_a$")

    for axis, levels, title in (
        (
            axes[0, 1],
            detector_levels,
            "Detector-only diffusion adjoint\n"
            f"absorber visits={float(detector_population['strong_absorber_visit_ratio_vr_over_analog']):.2f}×",
        ),
        (
            axes[0, 2],
            hybrid_levels,
            "Response + GenVR absorber objective\n"
            f"absorber visits={float(hybrid_population['strong_absorber_visit_ratio_vr_over_analog']):.2f}×",
        ),
    ):
        image = axis.imshow(
            levels,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=max(float(np.max(detector_levels)), float(np.max(hybrid_levels))),
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label="importance level")

    delta = hybrid_levels - detector_levels
    delta_limit = max(float(np.max(np.abs(delta))), 1.0)
    delta_image = axes[1, 0].imshow(
        delta,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-delta_limit,
        vmax=delta_limit,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1, 0].set_title("Importance-level change\n(absorber-aware − detector-only)")
    fig.colorbar(delta_image, ax=axes[1, 0], label="$\\Delta$ level")

    for axis, gain, title in (
        (
            axes[1, 1],
            detector_gain,
            "Detector-only cell-visit gain\n"
            f"FOM gain={float(detector_metadata['comparison']['fom_gain']):.2f}×",
        ),
        (
            axes[1, 2],
            hybrid_gain,
            "Absorber-aware cell-visit gain\n"
            f"FOM gain={float(hybrid_metadata['comparison']['fom_gain']):.2f}×",
        ),
    ):
        image = axis.imshow(
            gain,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-visit_limit,
            vmax=visit_limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label="$\\log_2[(N_{VR}+1)/(N_A+1)]$")

    absorber_mask = problem.sigma_a / np.maximum(
        problem.sigma_a + problem.sigma_s,
        np.finfo(np.float64).tiny,
    ) >= 0.8
    for axis in axes.ravel():
        axis.contour(
            absorber_mask,
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="white" if axis in (axes[0, 0], axes[1, 1], axes[1, 2]) else "black",
            linewidths=0.8,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.suptitle(
        "Hohlraum importance strategy trade-off: detector efficiency vs absorber statistics",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
