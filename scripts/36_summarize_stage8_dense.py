#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmc.benchmarks2d import (  # noqa: E402
    make_hohlraum_problem,
    make_lattice_problem,
    make_tokamak_coils_problem,
    make_tokamak_square_discrete_problem,
)


DEFAULT_RUN_DIRS = (
    "outputs/lattice",
    "outputs/hohlraum",
    "outputs/iter_axial",
    "outputs/iter_radial",
)


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


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_field(archive: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name], dtype=np.float64)
    raise KeyError(f"none of {names!r} found in archive")


def _summarize_case(run_dir: Path) -> tuple[str, dict[str, object], str | None]:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics["config"]
    case = str(config["case"])
    case_metrics = metrics["cases"][case]
    nx = int(config["nx"])
    ny = int(config["ny"])
    npz_path = run_dir / f"{case}_stage8_ultra.npz"
    with np.load(npz_path) as archive:
        cfm = _load_field(archive, "genvr_cfm_response", "cfm")
        dense = _load_field(archive, "dense_sn_reference", "dense")

    problem = _problem(case, nx, ny)
    shape_metrics = case_metrics["cfm_vs_dense"]
    summary = {
        "case_id": case,
        "output_directory": _relative(run_dir),
        "mesh": [nx, ny],
        "phase_space": [
            int(config["n_pos"]),
            int(config["n_mu"]),
            int(config["n_phi"]),
        ],
        "cfm_samples": int(config.get("cfm_samples", 0)),
        "global_mc_histories": int(config.get("mc_histories", 0)),
        "dense_spatial_scheme": str(config["dense_spatial_scheme"]),
        "dense_fixup": "conservative_zero_flux"
        if config["dense_spatial_scheme"] == "diamond"
        else None,
        "dense_fixup_fraction": float(case_metrics.get("dense_fixup_fraction", 0.0)),
        "dense_double_fixup_fraction": float(
            case_metrics.get("dense_double_fixup_fraction", 0.0)
        ),
        "correlation": float(shape_metrics["corr"]),
        "normalized_shape_l1": float(shape_metrics["normalized_rel_l1"]),
        "log10_rmse": float(shape_metrics["log10_rmse"]),
        "peak_ratio_cfm_over_dense": float(np.max(cfm) / np.max(dense)),
        "integral_ratio_cfm_over_dense": float(
            (np.sum(cfm) * problem.volume) / (np.sum(dense) * problem.volume)
        ),
        "cfm_converged": bool(case_metrics.get("cfm_converged", True)),
        "dense_converged": bool(case_metrics["dense_converged"]),
        "dense_iterations": int(case_metrics["dense_iterations"]),
        "dense_residual": float(case_metrics["dense_residual"]),
    }
    audit_path = run_dir / "reference_audit.json"
    audit_reference = None
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("audited_dense_spatial_scheme") == config["dense_spatial_scheme"]:
            audit_reference = _relative(audit_path)
    return run_dir.name, summary, audit_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage-8 Dense S_N comparisons")
    parser.add_argument("run_dirs", nargs="*", default=DEFAULT_RUN_DIRS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/stage8_dense_summary.json",
    )
    args = parser.parse_args()

    cases = {}
    reference_audits = {}
    schemes = set()
    for raw_path in args.run_dirs:
        run_dir = Path(raw_path)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        name, summary, audit_reference = _summarize_case(run_dir)
        cases[name] = summary
        schemes.add(summary["dense_spatial_scheme"])
        if audit_reference is not None:
            reference_audits[name] = audit_reference
    if len(schemes) != 1:
        raise ValueError(f"mixed Dense spatial schemes are not comparable: {sorted(schemes)}")

    scheme = schemes.pop()
    comparison_method = {
        "diamond": "dense_diamond_difference_sn",
        "upwind": "dense_positive_upwind_sn",
    }[scheme]
    report = {
        "schema_version": "2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "comparison_method": comparison_method,
        "comparison_role": "finite_grid_diagnostic_requires_convergence",
        "reference_audits": reference_audits,
    }
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
