#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmc.benchmarks2d import (  # noqa: E402
    make_hohlraum_problem,
    make_lattice_problem,
    make_tokamak_coils_problem,
    make_tokamak_square_discrete_problem,
)
from gmc.checkpoints import (  # noqa: E402
    DEFAULT_BOUNDARY_CHECKPOINT,
    DEFAULT_INTERNAL_CHECKPOINT,
    require_recommended_path_mode,
)
from gmc.method_taxonomy import get_method_taxonomy  # noqa: E402
from gmc.stage8_reporting import plot_stage8_case  # noqa: E402
from gmc.response_matrix2d import solve_response_matrix  # noqa: E402
from gmc.response_operator_ultra import (  # noqa: E402
    UltraPhaseSpace,
    build_boundary_response_ballistic_split_cfm_ultra,
    build_internal_response_cfm_ultra,
    build_internal_response_mc_ultra,
    build_internal_response_mc_ultra_vectorized,
    build_library,
    flux_shape_metrics,
    internal_operator_metrics,
    load_ultra_internal,
    load_ultra_library,
    operator_metrics,
    save_ultra_internal,
    save_ultra_library,
    solve_global_source_iteration,
    unique_materials,
)
from gmc.transport2d import BoundaryGMCSampler, InternalGMCSampler, run_standard_mc  # noqa: E402


def _file_sig(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _offset_tag(value: float) -> str:
    return f"o{int(round(float(value) * 1_000_000)):06d}"


def _cache_path(args, case, kind, disc, model_sig="none") -> Path:
    root = ROOT / args.response_cache_dir
    root.mkdir(parents=True, exist_ok=True)
    samples = args.cfm_samples if kind == "cfm" else args.oracle_samples
    tag = (
        f"{case}_{kind}_{args.nx}x{args.ny}_p{disc.n_pos}_m{disc.n_mu}_f{disc.n_phi}"
        f"_{_offset_tag(disc.phi_offset_fraction)}"
        f"_s{samples}_seed{args.seed}_rk{args.rk4_steps}_{model_sig}_{args.cfm_mode}"
    )
    return root / f"{tag}.npz"


def _internal_cache_path(args, case, kind, disc, model_sig="none") -> Path:
    root = ROOT / args.response_cache_dir
    root.mkdir(parents=True, exist_ok=True)
    samples = args.internal_samples if kind == "cfm" else args.oracle_internal_samples
    return root / (
        f"{case}_{kind}_internal_{args.nx}x{args.ny}_p{disc.n_pos}_m{disc.n_mu}_f{disc.n_phi}"
        f"_{_offset_tag(disc.phi_offset_fraction)}"
        f"_s{samples}_seed{args.seed}_rk{args.rk4_steps}_{model_sig}.npz"
    )


def make_prob(case, nx, ny):
    if case == "lattice":
        return make_lattice_problem(nx, ny)
    if case == "iter_r":
        return make_tokamak_square_discrete_problem(nx, ny)
    if case == "iter_a":
        return make_tokamak_coils_problem(nx, ny)
    return make_hohlraum_problem(nx, ny)


def build_cfm_lib(prob, disc, args, boundary, internal):
    if args.cfm_mode == "direct":
        return build_library(
            prob,
            disc,
            "cfm",
            boundary_sampler=boundary,
            samples_per_state=args.cfm_samples,
            seed=args.seed,
            qmc=True,
            inference_batch_size=args.inference_batch_size,
        )
    library = {}
    for material_index, (sigma_s, sigma_a) in enumerate(unique_materials(prob)):
        library[(sigma_s, sigma_a)] = build_boundary_response_ballistic_split_cfm_ultra(
            sigma_s,
            sigma_a,
            prob.dx,
            prob.dy,
            internal,
            disc,
            scattered_samples_per_state=args.cfm_samples,
            seed=args.seed + 100003 * material_index,
            inference_batch_size=args.inference_batch_size,
        )
    return library


def run_case(case, args, boundary, internal, disc, out):
    problem = make_prob(case, args.nx, args.ny)
    print(f"[{case}] building/loading GenVR/CFM operators; states={disc.n_state}", flush=True)
    model_signature = (
        _file_sig(ROOT / args.boundary_ckpt)
        if args.cfm_mode == "direct"
        else _file_sig(ROOT / args.internal_ckpt)
    )
    cfm_cache = _cache_path(args, case, "cfm", disc, model_signature)
    start = time.perf_counter()
    if cfm_cache.exists() and not args.rebuild_response:
        cfm_library = load_ultra_library(str(cfm_cache))
        print(f"[{case}] GenVR/CFM response cache hit: {cfm_cache}", flush=True)
    else:
        cfm_library = build_cfm_lib(problem, disc, args, boundary, internal)
        save_ultra_library(str(cfm_cache), cfm_library)
    cfm_boundary_build_s = time.perf_counter() - start

    cfm_internal = None
    cfm_internal_build_s = 0.0
    if case in ("lattice", "iter_r"):
        start = time.perf_counter()
        cfm_internal_cache = _internal_cache_path(
            args,
            case,
            "cfm",
            disc,
            _file_sig(ROOT / args.internal_ckpt),
        )
        if cfm_internal_cache.exists() and not args.rebuild_response:
            cfm_internal = load_ultra_internal(str(cfm_internal_cache))
            print(f"[{case}] internal GenVR/CFM cache hit: {cfm_internal_cache}", flush=True)
        else:
            cfm_internal = build_internal_response_cfm_ultra(
                1.0,
                0.0,
                problem.dx,
                problem.dy,
                internal,
                disc,
                samples=args.internal_samples,
                seed=args.seed + 787,
                inference_batch_size=args.inference_batch_size,
                qmc=True,
            )
            save_ultra_internal(str(cfm_internal_cache), cfm_internal)
        cfm_internal_build_s = time.perf_counter() - start

    cfm_build_s = cfm_boundary_build_s + cfm_internal_build_s
    start = time.perf_counter()
    cfm_result = solve_global_source_iteration(
        problem,
        cfm_library,
        disc,
        cfm_internal,
        rtol=args.rtol,
        max_iters=args.max_iters,
        device=args.device,
        dtype=args.dtype,
        check_every=args.check_every,
        solver=args.solver,
    )
    cfm_solve_s = time.perf_counter() - start

    projected_mc_result = None
    projected_mc_library = None
    projected_mc_internal = None
    projected_mc_build_s = None
    projected_mc_solve_s = None
    if args.oracle_samples > 0:
        print(f"[{case}] building/loading projected local-MC response", flush=True)
        projected_cache = _cache_path(args, case, "oracle", disc, "exact")
        start = time.perf_counter()
        if projected_cache.exists() and not args.rebuild_response:
            projected_mc_library = load_ultra_library(str(projected_cache))
            print(f"[{case}] projected local-MC cache hit: {projected_cache}", flush=True)
        else:
            projected_mc_library = build_library(
                problem,
                disc,
                "mc-vectorized" if args.oracle_backend != "numpy" else "mc",
                samples_per_state=args.oracle_samples,
                seed=args.seed + 99991,
                mc_device=args.oracle_backend,
                mc_dtype=args.oracle_mc_dtype,
            )
            save_ultra_library(str(projected_cache), projected_mc_library)
        projected_mc_boundary_build_s = time.perf_counter() - start
        projected_mc_internal_build_s = 0.0
        if case in ("lattice", "iter_r"):
            start = time.perf_counter()
            projected_internal_cache = _internal_cache_path(args, case, "oracle", disc, "exact")
            if projected_internal_cache.exists() and not args.rebuild_response:
                projected_mc_internal = load_ultra_internal(str(projected_internal_cache))
            else:
                if args.oracle_backend == "numpy":
                    projected_mc_internal = build_internal_response_mc_ultra(
                        1.0,
                        0.0,
                        problem.dx,
                        problem.dy,
                        disc,
                        samples=args.oracle_internal_samples,
                        seed=args.seed + 199991,
                    )
                else:
                    projected_mc_internal = build_internal_response_mc_ultra_vectorized(
                        1.0,
                        0.0,
                        problem.dx,
                        problem.dy,
                        disc,
                        samples=args.oracle_internal_samples,
                        seed=args.seed + 199991,
                        device=args.oracle_backend,
                        mc_dtype=args.oracle_mc_dtype,
                    )
                save_ultra_internal(str(projected_internal_cache), projected_mc_internal)
            projected_mc_internal_build_s = time.perf_counter() - start
        projected_mc_build_s = projected_mc_boundary_build_s + projected_mc_internal_build_s
        start = time.perf_counter()
        projected_mc_result = solve_global_source_iteration(
            problem,
            projected_mc_library,
            disc,
            projected_mc_internal,
            rtol=args.rtol,
            max_iters=args.max_iters,
            device=args.device,
            dtype=args.dtype,
            check_every=args.check_every,
            solver=args.solver,
        )
        projected_mc_solve_s = time.perf_counter() - start

    dense_result = None
    dense_runtime_s = None
    if not args.skip_dense:
        dense_n_mu = args.dense_n_mu if args.dense_n_mu is not None else args.n_mu
        dense_n_phi = args.dense_n_phi if args.dense_n_phi is not None else args.n_phi
        print(
            f"[{case}] dense finite-angle S_N reference ({dense_n_mu} x {dense_n_phi}, "
            f"offset={args.dense_phi_offset_fraction:g} bin)",
            flush=True,
        )
        start = time.perf_counter()
        dense_result = solve_response_matrix(
            problem,
            n_mu=dense_n_mu,
            n_phi=dense_n_phi,
            max_iters=args.dense_max_iters,
            rtol=args.dense_rtol,
            spatial_scheme=args.dense_spatial_scheme,
            phi_offset_fraction=args.dense_phi_offset_fraction,
        )
        dense_runtime_s = time.perf_counter() - start

    global_mc_result = None
    global_mc_runtime_s = None
    if args.mc_histories > 0:
        print(f"[{case}] global history MC; histories={args.mc_histories}", flush=True)
        start = time.perf_counter()
        global_mc_result = run_standard_mc(problem, args.mc_histories, seed=args.seed + 31337)
        global_mc_runtime_s = time.perf_counter() - start

    projected_flux = projected_mc_result.flux if projected_mc_result is not None else np.array([])
    dense_flux = dense_result.flux if dense_result is not None else np.array([])
    global_mc_flux = global_mc_result.flux if global_mc_result is not None else np.array([])
    metrics = {
        "n_state": disc.n_state,
        "n_angle": disc.n_angle,
        "interface_phi_offset_fraction": disc.phi_offset_fraction,
        "cfm_mode": args.cfm_mode,
        "solver": cfm_result.method,
        "boundary_path_mode": getattr(boundary.transform, "path_mode", "n/a") if boundary is not None else "n/a",
        "internal_path_mode": getattr(internal.transform, "path_mode", "n/a"),
        "cfm_build_s": cfm_build_s,
        "cfm_boundary_build_s": cfm_boundary_build_s,
        "cfm_internal_build_s": cfm_internal_build_s,
        "cfm_solve_s": cfm_solve_s,
        "cfm_iterations": cfm_result.iterations,
        "cfm_converged": cfm_result.converged,
        "cfm_residual": cfm_result.relative_residual,
        "cfm_nonzero_fraction": cfm_result.nonzero_fraction,
        "cfm_min_positive_flux": cfm_result.min_positive_flux,
    }
    if dense_result is not None:
        cfm_vs_dense = flux_shape_metrics(cfm_result.flux, dense_flux)
        dense_final_residual = float(dense_result.residual_history[-1]) if dense_result.residual_history.size else None
        metrics.update(
            {
                "cfm_vs_dense": cfm_vs_dense,
                "genvr_cfm_vs_dense_sn": cfm_vs_dense,
                "dense_runtime_s": dense_runtime_s,
                "dense_iterations": dense_result.iterations,
                "dense_converged": dense_result.converged,
                "dense_residual": dense_final_residual,
                "dense_angular_order": list(dense_result.angular_order),
                "dense_spatial_scheme": dense_result.spatial_scheme,
                "dense_phi_offset_fraction": dense_result.phi_offset_fraction,
                "dense_fixup_fraction": dense_result.fixup_fraction,
                "dense_double_fixup_fraction": dense_result.double_fixup_fraction,
                "dense_nonzero_fraction": dense_result.nonzero_fraction,
                "dense_min_positive_flux": dense_result.min_flux,
            }
        )
    if projected_mc_result is not None:
        cfm_vs_projected = flux_shape_metrics(cfm_result.flux, projected_flux)
        metrics.update(
            {
                "oracle_build_s": projected_mc_build_s,
                "oracle_boundary_build_s": projected_mc_boundary_build_s,
                "oracle_internal_build_s": projected_mc_internal_build_s,
                "oracle_solve_s": projected_mc_solve_s,
                "oracle_iterations": projected_mc_result.iterations,
                "oracle_solver": projected_mc_result.method,
                "oracle_residual": projected_mc_result.relative_residual,
                "cfm_vs_oracle": cfm_vs_projected,
                "projected_local_mc_build_s": projected_mc_build_s,
                "projected_local_mc_boundary_build_s": projected_mc_boundary_build_s,
                "projected_local_mc_internal_build_s": projected_mc_internal_build_s,
                "projected_local_mc_solve_s": projected_mc_solve_s,
                "projected_local_mc_iterations": projected_mc_result.iterations,
                "projected_local_mc_converged": projected_mc_result.converged,
                "projected_local_mc_solver": projected_mc_result.method,
                "projected_local_mc_residual": projected_mc_result.relative_residual,
                "genvr_cfm_vs_projected_local_mc": cfm_vs_projected,
            }
        )
        if dense_result is not None:
            projected_vs_dense = flux_shape_metrics(projected_flux, dense_flux)
            metrics["oracle_vs_dense"] = projected_vs_dense
            metrics["projected_local_mc_vs_dense_sn"] = projected_vs_dense
        local_metrics = {}
        for key in cfm_library:
            local_metrics[f"ss={key[0]:g},sa={key[1]:g}"] = operator_metrics(
                cfm_library[key], projected_mc_library[key]
            )
        metrics["local_operator_metrics"] = local_metrics
        metrics["cfm_vs_projected_local_mc_operator_metrics"] = local_metrics
        if case in ("lattice", "iter_r") and cfm_internal is not None and projected_mc_internal is not None:
            internal_metrics = internal_operator_metrics(cfm_internal, projected_mc_internal)
            metrics["internal_source_operator_metrics"] = internal_metrics
            metrics["cfm_vs_projected_local_mc_internal_metrics"] = internal_metrics
    if global_mc_result is not None:
        cfm_vs_global = flux_shape_metrics(cfm_result.flux, global_mc_flux)
        metrics.update(
            {
                "mc_histories": args.mc_histories,
                "mc_runtime_s": global_mc_runtime_s,
                "mc_steps": global_mc_result.steps,
                "mc_leakage_histories": global_mc_result.leakage,
                "mc_absorbed_weight": global_mc_result.absorbed_weight,
                "mc_total_track_length": global_mc_result.track_length,
                "cfm_vs_mc": cfm_vs_global,
                "global_history_mc_histories": args.mc_histories,
                "global_history_mc_runtime_s": global_mc_runtime_s,
                "genvr_cfm_vs_global_history_mc": cfm_vs_global,
            }
        )
        if dense_result is not None:
            dense_vs_global = flux_shape_metrics(dense_flux, global_mc_flux)
            metrics["dense_vs_mc"] = dense_vs_global
            metrics["dense_sn_vs_global_history_mc"] = dense_vs_global
        if projected_mc_result is not None:
            projected_vs_global = flux_shape_metrics(projected_flux, global_mc_flux)
            metrics["oracle_vs_mc"] = projected_vs_global
            metrics["projected_local_mc_vs_global_history_mc"] = projected_vs_global

    np.savez_compressed(
        out / f"{case}_stage8_ultra.npz",
        genvr_cfm_response=cfm_result.flux,
        projected_local_mc_response=projected_flux,
        dense_sn_reference=dense_flux,
        global_history_mc=global_mc_flux,
        cfm=cfm_result.flux,
        oracle=projected_flux,
        dense=dense_flux,
        mc=global_mc_flux,
    )
    plot_stage8_case(
        problem,
        cfm_result.flux,
        None,
        dense_result.flux if dense_result is not None else None,
        global_mc_result.flux if global_mc_result is not None else None,
        out,
        case,
        n_pos=disc.n_pos,
        n_mu=disc.n_mu,
        n_phi=disc.n_phi,
        mc_histories=args.mc_histories if global_mc_result is not None else None,
        dense_spatial_scheme=args.dense_spatial_scheme,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end Stage 8 Ultra benchmark")
    parser.add_argument("--case", choices=["lattice", "hohlraum", "iter_r", "iter_a", "bench"], default="bench")
    parser.add_argument("--nx", type=int, default=112)
    parser.add_argument("--ny", type=int, default=112)
    parser.add_argument("--n-pos", type=int, default=4)
    parser.add_argument("--n-mu", type=int, default=4)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--phi-offset-fraction", type=float, default=0.0)
    parser.add_argument("--cfm-mode", choices=["direct", "ballistic-split"], default="direct")
    parser.add_argument("--boundary-ckpt", default=DEFAULT_BOUNDARY_CHECKPOINT)
    parser.add_argument("--internal-ckpt", default=DEFAULT_INTERNAL_CHECKPOINT)
    parser.add_argument("--allow-legacy-checkpoints", action="store_true")
    parser.add_argument("--cfm-samples", type=int, default=2048)
    parser.add_argument(
        "--projected-local-mc-samples",
        "--oracle-samples",
        dest="oracle_samples",
        type=int,
        default=2048,
        help="finite local-MC samples per projected interface state; 0 disables this reference",
    )
    parser.add_argument("--internal-samples", type=int, default=32768)
    parser.add_argument(
        "--projected-local-mc-internal-samples",
        "--oracle-internal-samples",
        dest="oracle_internal_samples",
        type=int,
        default=65536,
    )
    parser.add_argument(
        "--projected-local-mc-backend",
        "--oracle-backend",
        dest="oracle_backend",
        choices=["numpy", "cpu", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--projected-local-mc-dtype",
        "--oracle-mc-dtype",
        dest="oracle_mc_dtype",
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--mc-histories", type=int, default=0, help="0 disables global history MC")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--inference-batch-size", type=int, default=65536)
    parser.add_argument("--rk4-steps", type=int, default=12)
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--max-iters", type=int, default=6000)
    parser.add_argument("--solver", choices=["source", "bicgstab", "auto"], default="auto")
    parser.add_argument("--dense-rtol", type=float, default=2.0e-8)
    parser.add_argument("--dense-max-iters", type=int, default=1200)
    parser.add_argument("--dense-n-mu", type=int, default=None)
    parser.add_argument("--dense-n-phi", type=int, default=None)
    parser.add_argument("--dense-phi-offset-fraction", type=float, default=0.0)
    parser.add_argument(
        "--dense-spatial-scheme",
        choices=["diamond", "upwind"],
        default="diamond",
        help="diamond difference with conservative zero-flux fixup, or legacy positive upwind",
    )
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--check-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260827)
    parser.add_argument("--response-cache-dir", default="outputs/response_cache")
    parser.add_argument("--rebuild-response", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="highest")
    parser.add_argument("--outdir", default="outputs/stage8_ultra_benchmark")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.matmul_precision != "highest"
        torch.backends.cudnn.allow_tf32 = args.matmul_precision != "highest"

    disc = UltraPhaseSpace(
        args.n_pos,
        args.n_mu,
        args.n_phi,
        phi_offset_fraction=args.phi_offset_fraction,
    )
    print(f"device={args.device}; n_angle={disc.n_angle}; n_state={disc.n_state}", flush=True)
    boundary = (
        BoundaryGMCSampler(ROOT / args.boundary_ckpt, device=args.device, n_steps=args.rk4_steps)
        if args.cfm_mode == "direct"
        else None
    )
    internal = InternalGMCSampler(ROOT / args.internal_ckpt, device=args.device, n_steps=args.rk4_steps)
    if not args.allow_legacy_checkpoints:
        if boundary is not None:
            require_recommended_path_mode(boundary.checkpoint_metadata)
        require_recommended_path_mode(internal.checkpoint_metadata)
    if args.torch_compile:
        if boundary is not None:
            boundary.model = torch.compile(boundary.model, mode="reduce-overhead")
        internal.model = torch.compile(internal.model, mode="reduce-overhead")

    checkpoint_provenance = {
        "boundary": boundary.checkpoint_metadata if boundary is not None else {
            "configured_path": args.boundary_ckpt,
            "used": False,
        },
        "internal": internal.checkpoint_metadata,
        "legacy_path_mode_allowed": bool(args.allow_legacy_checkpoints),
    }
    print(json.dumps({"checkpoints": checkpoint_provenance}, indent=2), flush=True)

    cases = ["lattice", "hohlraum"] if args.case == "bench" else [args.case]
    summary = {
        "schema_version": "2.0",
        "config": vars(args),
        "checkpoints": checkpoint_provenance,
        "method_taxonomy": get_method_taxonomy(
            [
                "genvr_cfm_response",
                "projected_local_mc_response",
                "dense_sn_reference",
                "global_history_mc",
            ]
        ),
        "n_angle": disc.n_angle,
        "n_state": disc.n_state,
        "cases": {},
    }
    for case in cases:
        summary["cases"][case] = run_case(case, args, boundary, internal, disc, out)
        print(json.dumps(summary["cases"][case], indent=2), flush=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
