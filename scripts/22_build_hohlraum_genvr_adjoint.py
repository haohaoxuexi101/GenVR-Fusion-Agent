#!/usr/bin/env python3
"""Build a GenVR adjoint proxy for the hohlraum right-boundary response."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gmc.benchmarks2d import Structured2DProblem, make_hohlraum_problem
from gmc.importance_transport import BoundaryDetector, solve_diffusion_adjoint_importance
from gmc.response_operator_ultra import (
    UltraPhaseSpace,
    load_ultra_library,
    solve_global_source_iteration,
)


DEFAULT_CACHE = (
    ROOT
    / "outputs/response_cache_hpc/"
    "hohlraum_cfm_150x150_p4_m4_f32_s10000_rk12_5f816a4032cd_direct.npz"
)


def _reverse_problem(problem: Structured2DProblem) -> Structured2DProblem:
    return Structured2DProblem(
        name=f"{problem.name}_right_boundary_reverse",
        nx=problem.nx,
        ny=problem.ny,
        width=problem.width,
        height=problem.height,
        sigma_a=np.fliplr(problem.sigma_a).copy(),
        sigma_s=np.fliplr(problem.sigma_s).copy(),
        source_kind="left_boundary",
        boundary_source_y_range=(0.0, problem.height),
    )


def _normalized(field: np.ndarray) -> np.ndarray:
    return field / max(float(np.max(field)), np.finfo(np.float64).tiny)


def _plot(
    problem: Structured2DProblem,
    diffusion: np.ndarray,
    genvr: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    diffusion_normalized = _normalized(diffusion)
    genvr_normalized = _normalized(genvr)
    log_diffusion = np.log10(np.maximum(diffusion_normalized, 1.0e-15))
    log_genvr = np.log10(np.maximum(genvr_normalized, 1.0e-15))
    correction = log_genvr - log_diffusion
    correction_limit = max(float(np.quantile(np.abs(correction), 0.995)), 0.1)
    extent = [0.0, problem.width, 0.0, problem.height]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    panels = (
        (axes[0], problem.sigma_a, "Material absorption", "viridis", None),
        (axes[1], log_diffusion, "Diffusion adjoint", "viridis", (-15.0, 0.0)),
        (axes[2], log_genvr, "Reverse GenVR adjoint", "viridis", (-15.0, 0.0)),
        (
            axes[3],
            correction,
            "GenVR log correction",
            "viridis",
            (-correction_limit, correction_limit),
        ),
    )
    for axis, field, title, cmap, limits in panels:
        kwargs = {}
        if limits is not None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,
            **kwargs,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=150)
    parser.add_argument("--ny", type=int, default=150)
    parser.add_argument("--n-pos", type=int, default=4)
    parser.add_argument("--n-mu", type=int, default=4)
    parser.add_argument("--n-phi", type=int, default=32)
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--max-iters", type=int, default=6000)
    parser.add_argument("--check-every", type=int, default=4)
    parser.add_argument("--solver", choices=("source", "bicgstab", "auto"), default="source")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs/importance_vr_validation/hohlraum_reverse_genvr",
    )
    args = parser.parse_args()

    cache_path = args.cache_file if args.cache_file.is_absolute() else ROOT / args.cache_file
    if not cache_path.exists():
        raise FileNotFoundError(f"response cache not found: {cache_path}")

    problem = make_hohlraum_problem(args.nx, args.ny)
    reverse_problem = _reverse_problem(problem)
    discretization = UltraPhaseSpace(args.n_pos, args.n_mu, args.n_phi)
    library = load_ultra_library(str(cache_path))
    expected_state_count = discretization.n_state
    if any(operator.R.shape != (expected_state_count, expected_state_count) for operator in library.values()):
        raise ValueError("response cache phase-space shape does not match requested discretization")
    if any(
        not np.isclose(operator.dx, problem.dx) or not np.isclose(operator.dy, problem.dy)
        for operator in library.values()
    ):
        raise ValueError("response cache mesh spacing does not match requested hohlraum grid")

    start = time.perf_counter()
    reverse_result = solve_global_source_iteration(
        reverse_problem,
        library,
        discretization,
        rtol=args.rtol,
        max_iters=args.max_iters,
        device=args.device,
        dtype=args.dtype,
        check_every=args.check_every,
        solver=args.solver,
    )
    runtime_s = time.perf_counter() - start
    genvr_adjoint = np.fliplr(reverse_result.flux).copy()
    diffusion_adjoint = solve_diffusion_adjoint_importance(
        problem,
        BoundaryDetector(face="right"),
    )
    normalized_genvr = _normalized(genvr_adjoint)
    normalized_diffusion = _normalized(diffusion_adjoint)
    log_correlation = float(
        np.corrcoef(
            np.log(np.maximum(normalized_genvr, 1.0e-15)).ravel(),
            np.log(np.maximum(normalized_diffusion, 1.0e-15)).ravel(),
        )[0, 1]
    )
    linear_correlation = float(
        np.corrcoef(normalized_genvr.ravel(), normalized_diffusion.ravel())[0, 1]
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outdir / "adjoint_proxy.npz",
        genvr_adjoint=genvr_adjoint,
        reverse_forward_flux=reverse_result.flux,
        diffusion_adjoint=diffusion_adjoint,
        log_correction=(
            np.log10(np.maximum(normalized_genvr, 1.0e-15))
            - np.log10(np.maximum(normalized_diffusion, 1.0e-15))
        ),
    )
    metadata = {
        "case": "hohlraum",
        "response": "right-boundary leakage",
        "construction": (
            "flip the material map left-right, inject the original detector as a "
            "full-height left-boundary source, solve with the cached GenVR/CFM "
            "response operator, then flip the scalar field back"
        ),
        "grid": [args.nx, args.ny],
        "phase_space": {
            "n_pos": args.n_pos,
            "n_mu": args.n_mu,
            "n_phi": args.n_phi,
            "n_state": discretization.n_state,
        },
        "cache_file": str(cache_path.relative_to(ROOT) if cache_path.is_relative_to(ROOT) else cache_path),
        "solver": reverse_result.method,
        "iterations": reverse_result.iterations,
        "converged": reverse_result.converged,
        "relative_residual": reverse_result.relative_residual,
        "runtime_s": runtime_s,
        "log_field_correlation_with_diffusion": log_correlation,
        "linear_field_correlation_with_diffusion": linear_correlation,
    }
    (args.outdir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot(
        problem,
        diffusion_adjoint,
        genvr_adjoint,
        args.outdir / "adjoint_compare.png",
        args.dpi,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
