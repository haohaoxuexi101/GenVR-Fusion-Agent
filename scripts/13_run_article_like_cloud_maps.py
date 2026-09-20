#!/usr/bin/env python
"""Run article-like 112x112 2D lattice and hohlraum cloud-map comparisons.

This script uses the batched GMC transport driver so that MC/GMC flux-map
comparisons are feasible on CPU.  It preserves the article-level dimensions and
material data used in the reproduction code:
  * lattice:   7 x 7 cm, 112 x 112, internal 1 x 1 cm source
  * hohlraum:  1.3 x 1.3 cm, 112 x 112, left boundary source
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)

from gmc.benchmarks2d import make_lattice_problem, make_hohlraum_problem, material_rgb
from gmc.transport2d import (
    BoundaryGMCSampler,
    InternalGMCSampler,
    run_standard_mc,
    run_gmc_full_transport_batched,
    save_result_npz,
)


def rel_l1(ref: np.ndarray, pred: np.ndarray, eps: float = 1e-30) -> float:
    return float(np.sum(np.abs(pred - ref)) / (np.sum(np.abs(ref)) + eps))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = a.ravel(), b.ravel()
    if np.std(av) == 0 or np.std(bv) == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def make_comparison_figure(problem, mc, gmc, outdir: Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)
    eps = 1e-30
    extent = [0, problem.width, 0, problem.height]
    err = np.abs(gmc.flux - mc.flux)

    # Common color limits for MC and GMC panels.
    combined = np.concatenate([np.log10(mc.flux.ravel() + eps), np.log10(gmc.flux.ravel() + eps)])
    finite = combined[np.isfinite(combined)]
    vmin, vmax = np.quantile(finite, [0.02, 0.995]) if finite.size else (-12, 0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(material_rgb(problem), origin="lower", extent=extent, aspect="equal")
    ax.set_title("Material layout")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")

    ax = axes[0, 1]
    im = ax.imshow(np.log10(mc.flux + eps), origin="lower", extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title("Standard MC log10 flux")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    fig.colorbar(im, ax=ax, label="log10 scalar flux")

    ax = axes[1, 0]
    im = ax.imshow(np.log10(gmc.flux + eps), origin="lower", extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title("GMC log10 flux")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    fig.colorbar(im, ax=ax, label="log10 scalar flux")

    ax = axes[1, 1]
    efinite = np.log10(err.ravel() + eps)
    evmin, evmax = np.quantile(efinite[np.isfinite(efinite)], [0.02, 0.995])
    im = ax.imshow(np.log10(err + eps), origin="lower", extent=extent, aspect="equal", vmin=evmin, vmax=evmax)
    ax.set_title("|GMC - MC| log10 abs error")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    fig.colorbar(im, ax=ax, label="log10 abs difference")
    fig.suptitle(f"{prefix}: article-like 112x112 cloud-map comparison")
    fig.savefig(outdir / f"{prefix}_cloudmap_comparison.png", dpi=220)
    plt.close(fig)

    # Lineouts through source/center, like the paper figure caption describes.
    iy = problem.ny // 2
    ix = problem.nx // 2
    xs = (np.arange(problem.nx) + 0.5) * problem.dx
    ys = (np.arange(problem.ny) + 0.5) * problem.dy
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(xs, mc.flux[iy] + eps, label="MC horizontal")
    plt.semilogy(xs, gmc.flux[iy] + eps, label="GMC horizontal")
    plt.semilogy(ys, mc.flux[:, ix] + eps, "--", label="MC vertical")
    plt.semilogy(ys, gmc.flux[:, ix] + eps, "--", label="GMC vertical")
    plt.xlabel("position [cm]")
    plt.ylabel("scalar flux")
    plt.title(f"{prefix}: centered lineouts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_lineouts.png", dpi=220)
    plt.close()

    save_result_npz(outdir / f"{prefix}_mc_result.npz", mc)
    save_result_npz(outdir / f"{prefix}_gmc_result.npz", gmc)


def run_one(name: str, problem, boundary, internal, n: int, outdir: Path, seed_base: int, max_steps: int, batch: int):
    t0 = time.time()
    mc = run_standard_mc(problem, n_particles=n, seed=seed_base, max_steps_per_history=max_steps)
    t_mc = time.time() - t0
    t0 = time.time()
    gmc = run_gmc_full_transport_batched(
        problem,
        boundary,
        internal if problem.source_kind == "volume_box" else None,
        n_particles=n,
        seed=seed_base + 1,
        max_steps_per_history=max_steps,
        inference_batch_size=batch,
    )
    t_gmc = time.time() - t0
    make_comparison_figure(problem, mc, gmc, outdir, name)
    m = {
        "benchmark": name,
        "n_particles": n,
        "mesh": [problem.nx, problem.ny],
        "domain_cm": [problem.width, problem.height],
        "mc_steps": mc.steps,
        "gmc_steps": gmc.steps,
        "mc_leakage": mc.leakage,
        "gmc_leakage": gmc.leakage,
        "mc_track_length": mc.track_length,
        "gmc_track_length": gmc.track_length,
        "track_length_ratio_gmc_over_mc": gmc.track_length / mc.track_length if mc.track_length else None,
        "flux_relative_l1_gmc_vs_mc": rel_l1(mc.flux, gmc.flux),
        "flux_correlation_gmc_vs_mc": corr(mc.flux, gmc.flux),
        "mc_wall_seconds": t_mc,
        "gmc_wall_seconds": t_gmc,
    }
    with open(outdir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary-ckpt", required=True)
    ap.add_argument("--internal-ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--outdir", default="outputs/stage4_article_like_cloudmaps")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--inference-batch-size", type=int, default=8192)
    ap.add_argument("--skip-hohlraum", action="store_true")
    ap.add_argument("--skip-lattice", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    boundary = BoundaryGMCSampler(args.boundary_ckpt, device=args.device)
    internal = InternalGMCSampler(args.internal_ckpt, device=args.device)

    metrics = []
    if not args.skip_lattice:
        metrics.append(run_one("lattice", make_lattice_problem(112, 112), boundary, internal, args.n, outdir, 101, args.max_steps, args.inference_batch_size))
    if not args.skip_hohlraum:
        metrics.append(run_one("hohlraum", make_hohlraum_problem(112, 112), boundary, internal, args.n, outdir, 201, args.max_steps, args.inference_batch_size))
    with open(outdir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
