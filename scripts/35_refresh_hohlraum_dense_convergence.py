#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmc.benchmarks2d import make_hohlraum_problem  # noqa: E402
from gmc.response_matrix2d import solve_response_matrix  # noqa: E402
from gmc.response_operator_ultra import flux_shape_metrics  # noqa: E402


DEFAULT_SPATIAL_MESHES = (65, 130, 260, 390, 520)
DEFAULT_ANGULAR_ORDERS = ((2, 16), (4, 16), (4, 32), (4, 64), (6, 48), (8, 64))


def _backup_once(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _solve_record(
    nx: int,
    ny: int,
    n_mu: int,
    n_phi: int,
    *,
    rtol: float,
    max_iters: int,
    cfm: np.ndarray,
    aggregate_mc: np.ndarray,
) -> dict[str, object]:
    problem = make_hohlraum_problem(nx, ny)
    start = time.perf_counter()
    result = solve_response_matrix(
        problem,
        n_mu=n_mu,
        n_phi=n_phi,
        rtol=rtol,
        max_iters=max_iters,
        spatial_scheme="diamond",
    )
    runtime = time.perf_counter() - start
    if not result.converged:
        raise RuntimeError(
            f"Dense DD did not converge for {nx}x{ny}, angular order {n_mu}x{n_phi}"
        )

    record: dict[str, object] = {
        "nx": nx,
        "ny": ny,
        "n_mu": n_mu,
        "n_phi": n_phi,
        "spatial_scheme": result.spatial_scheme,
        "runtime_s": runtime,
        "iterations": result.iterations,
        "converged": result.converged,
        "final_residual": float(result.residual_history[-1]),
        "peak": float(np.max(result.flux)),
        "integrated_track": float(np.sum(result.flux) * problem.volume),
        "min_positive": result.min_flux,
        "nonzero_fraction": result.nonzero_fraction,
        "fixup_fraction": result.fixup_fraction,
        "double_fixup_fraction": result.double_fixup_fraction,
        "geometry_boundaries_cell_aligned": nx == ny and nx % 130 == 0,
    }
    if result.flux.shape == cfm.shape:
        record["vs_cfm"] = flux_shape_metrics(result.flux, cfm)
    if result.flux.shape == aggregate_mc.shape:
        record["vs_mc"] = flux_shape_metrics(result.flux, aggregate_mc)
    print(
        f"[Dense DD] {nx}x{ny}, {n_mu}x{n_phi}: "
        f"track={record['integrated_track']:.9f}, "
        f"fixup={100.0 * result.fixup_fraction:.3f}%, runtime={runtime:.2f}s",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Hohlraum Dense diamond-difference convergence data"
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/hohlraum")
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--max-iters", type=int, default=2000)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    stage8_path = run_dir / "hohlraum_stage8_ultra.npz"
    mc_fields_path = run_dir / "mc_convergence.npz"
    output_path = run_dir / "dense_convergence.json"
    for path in (stage8_path, mc_fields_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(stage8_path) as archive:
        cfm = np.asarray(archive["genvr_cfm_response"], dtype=np.float64)
    with np.load(mc_fields_path) as archive:
        aggregate_mc = np.asarray(archive["aggregate_flux"], dtype=np.float64)

    _backup_once(output_path, ".positive_upwind")
    cache: dict[tuple[int, int, int, int], dict[str, object]] = {}

    def solve(nx: int, ny: int, n_mu: int, n_phi: int) -> dict[str, object]:
        key = (nx, ny, n_mu, n_phi)
        if key not in cache:
            cache[key] = _solve_record(
                nx,
                ny,
                n_mu,
                n_phi,
                rtol=args.rtol,
                max_iters=args.max_iters,
                cfm=cfm,
                aggregate_mc=aggregate_mc,
            )
        return cache[key]

    spatial_records = [solve(mesh, mesh, 4, 32) for mesh in DEFAULT_SPATIAL_MESHES]
    angular_records = [
        solve(150, 150, n_mu, n_phi) for n_mu, n_phi in DEFAULT_ANGULAR_ORDERS
    ]
    report = {
        "schema_version": "2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": "hohlraum",
        "spatial_scheme": "diamond",
        "fixup": "conservative_zero_flux",
        "rtol": args.rtol,
        "max_iters": args.max_iters,
        "spatial_records": spatial_records,
        "angular_records": angular_records,
        "records": spatial_records + angular_records,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "records": len(cache)}, indent=2))


if __name__ == "__main__":
    main()
