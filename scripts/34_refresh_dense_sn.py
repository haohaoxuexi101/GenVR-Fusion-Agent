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

from gmc.benchmarks2d import (  # noqa: E402
    make_hohlraum_problem,
    make_lattice_problem,
    make_tokamak_coils_problem,
    make_tokamak_square_discrete_problem,
)
from gmc.method_taxonomy import get_method_taxonomy  # noqa: E402
from gmc.response_matrix2d import solve_response_matrix  # noqa: E402
from gmc.response_operator_ultra import flux_shape_metrics  # noqa: E402


def _problem(case: str, nx: int, ny: int):
    if case == "lattice":
        return make_lattice_problem(nx, ny)
    if case == "hohlraum":
        return make_hohlraum_problem(nx, ny)
    if case == "iter_r":
        return make_tokamak_square_discrete_problem(nx, ny)
    if case == "iter_a":
        return make_tokamak_coils_problem(nx, ny)
    raise ValueError(f"unsupported Stage-8 case {case!r}")


def _load(archive: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    raise KeyError(f"none of {names!r} found in archive")


def _backup_once(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def refresh(run_dir: Path, scheme: str, dense_rtol: float, dense_max_iters: int) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics["config"]
    case = str(config["case"])
    nx = int(config["nx"])
    ny = int(config["ny"])
    n_mu = int(config.get("dense_n_mu") or config["n_mu"])
    n_phi = int(config.get("dense_n_phi") or config["n_phi"])
    npz_path = run_dir / f"{case}_stage8_ultra.npz"

    backup_suffix = ".positive_upwind"
    _backup_once(metrics_path, backup_suffix)
    _backup_once(npz_path, backup_suffix)
    _backup_once(run_dir / f"{case}_stage8_ultra.png", backup_suffix)

    with np.load(npz_path) as archive:
        fields = {name: np.asarray(archive[name]) for name in archive.files}
        cfm = _load(archive, "genvr_cfm_response", "cfm").astype(np.float64)
        global_mc = _load(archive, "global_history_mc", "mc").astype(np.float64)

    problem = _problem(case, nx, ny)
    start = time.perf_counter()
    dense_result = solve_response_matrix(
        problem,
        n_mu=n_mu,
        n_phi=n_phi,
        max_iters=dense_max_iters,
        rtol=dense_rtol,
        spatial_scheme=scheme,
    )
    runtime = time.perf_counter() - start
    if not dense_result.converged:
        raise RuntimeError(f"dense solve did not converge for {case}")

    fields["dense_sn_reference"] = dense_result.flux
    fields["dense"] = dense_result.flux
    np.savez_compressed(npz_path, **fields)

    case_metrics = metrics["cases"][case]
    cfm_vs_dense = flux_shape_metrics(cfm, dense_result.flux)
    case_metrics.update(
        {
            "cfm_vs_dense": cfm_vs_dense,
            "genvr_cfm_vs_dense_sn": cfm_vs_dense,
            "dense_runtime_s": runtime,
            "dense_iterations": dense_result.iterations,
            "dense_converged": dense_result.converged,
            "dense_residual": float(dense_result.residual_history[-1]),
            "dense_angular_order": list(dense_result.angular_order),
            "dense_spatial_scheme": dense_result.spatial_scheme,
            "dense_fixup_fraction": dense_result.fixup_fraction,
            "dense_double_fixup_fraction": dense_result.double_fixup_fraction,
            "dense_nonzero_fraction": dense_result.nonzero_fraction,
            "dense_min_positive_flux": dense_result.min_flux,
        }
    )
    if global_mc.size:
        dense_vs_mc = flux_shape_metrics(dense_result.flux, global_mc)
        case_metrics["dense_vs_mc"] = dense_vs_mc
        case_metrics["dense_sn_vs_global_history_mc"] = dense_vs_mc
    config["dense_spatial_scheme"] = scheme
    config["dense_rtol"] = dense_rtol
    config["dense_max_iters"] = dense_max_iters
    metrics["method_taxonomy"] = get_method_taxonomy()
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "case": case,
        "run_dir": str(run_dir),
        "mesh": [nx, ny],
        "angular_order": [n_mu, n_phi],
        "spatial_scheme": scheme,
        "runtime_s": runtime,
        "iterations": dense_result.iterations,
        "residual": float(dense_result.residual_history[-1]),
        "fixup_fraction": dense_result.fixup_fraction,
        "double_fixup_fraction": dense_result.double_fixup_fraction,
        "cfm_vs_dense": cfm_vs_dense,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh saved Stage-8 Dense S_N fields without rerunning CFM or global MC"
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--scheme", choices=("diamond", "upwind"), default="diamond")
    parser.add_argument("--dense-rtol", type=float, default=1.0e-9)
    parser.add_argument("--dense-max-iters", type=int, default=2000)
    args = parser.parse_args()

    reports = []
    for raw_path in args.run_dirs:
        run_dir = Path(raw_path)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        reports.append(
            refresh(run_dir, args.scheme, args.dense_rtol, args.dense_max_iters)
        )
    output = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
    }
    (ROOT / "outputs" / "dense_refresh_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
