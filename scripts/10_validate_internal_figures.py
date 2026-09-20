#!/usr/bin/env python
"""Validate a trained internal GMC model against MC CDFs and direction scatters."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
if not torch.cuda.is_available():
    torch.set_num_threads(1)

from gmc.mc_cell import generate_internal_dataset
from gmc.model import BoundaryVelocityNet, ModelConfig, sample_rk4
from gmc.transforms import InternalTransform


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(ckpt["model_config"])
    transform = InternalTransform.from_dict(ckpt["transform"])
    model = BoundaryVelocityNet(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, transform, ckpt


@torch.no_grad()
def sample_gmc(model, transform, conditions: np.ndarray, batch_size: int, steps: int, device: torch.device) -> np.ndarray:
    outs = []
    for i in range(0, conditions.shape[0], batch_size):
        c_np = transform.encode_conditions(conditions[i : i + batch_size])
        c = torch.from_numpy(c_np).to(device)
        y_enc = sample_rk4(model, c, n_steps=steps)
        y = transform.decode_targets_np(y_enc.detach().cpu().numpy(), conditions[i : i + batch_size])
        outs.append(y)
    return np.concatenate(outs, axis=0)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a)
    b = np.sort(b)
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def plot_cdf_grid(values_mc, values_gmc, dims, xlabel, out_path, logx=False):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharey=True)
    for ax, (W, H), mc, gmc in zip(axes.flat, dims, values_mc, values_gmc):
        for arr, label in [(mc, "MC"), (gmc, "GMC")]:
            arr = np.sort(arr)
            cdf = np.linspace(0.0, 1.0, arr.size, endpoint=False)
            ax.plot(arr, cdf, label=label, linewidth=1.2)
        ax.set_title(f"W={W:g}, H={H:g}", fontsize=9)
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("CDF")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_direction_grid(values_mc, values_gmc, dims, out_path, max_points=3000):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True, sharey=True)
    circle = np.linspace(0.0, 2.0 * np.pi, 256)
    for ax, (W, H), mc, gmc in zip(axes.flat, dims, values_mc, values_gmc):
        n = min(max_points, mc.shape[0], gmc.shape[0])
        ax.scatter(mc[:n, 1], mc[:n, 2], s=2, alpha=0.25, label="MC")
        ax.scatter(gmc[:n, 1], gmc[:n, 2], s=2, alpha=0.25, label="GMC")
        ax.plot(np.cos(circle), np.sin(circle), linestyle="--", linewidth=0.8)
        ax.set_title(f"W={W:g}, H={H:g}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("u_exit")
    for ax in axes[:, 0]:
        ax.set_ylabel("v_exit")
    axes[0, 0].legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/internal_cfm.pt")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--outdir", type=str, default="outputs/validation_internal")
    parser.add_argument("--seed", type=int, default=3026)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, _ = load_model(args.ckpt, device)

    grid = [0.015, 0.05, 0.5, 5.0]
    dims = [(W, H) for W in grid for H in grid]
    mc_targets = []
    gmc_targets = []
    rows = []

    for j, (W, H) in enumerate(dims):
        print(f"validating internal W={W}, H={H}", flush=True)
        batch = generate_internal_dataset(args.n, seed=args.seed + j, fixed_W=W, fixed_H=H, progress_every=0)
        gmc = sample_gmc(model, transform, batch.conditions, args.batch_size, args.steps, device)
        mc_targets.append(batch.targets)
        gmc_targets.append(gmc)
        rows.append(
            [
                W,
                H,
                ks_distance(batch.targets[:, 0], gmc[:, 0]),
                ks_distance(batch.targets[:, 3], gmc[:, 3]),
                float(np.mean(batch.targets[:, 3])),
                float(np.mean(gmc[:, 3])),
            ]
        )

    plot_cdf_grid(
        [x[:, 0] for x in mc_targets],
        [x[:, 0] for x in gmc_targets],
        dims,
        "exit perimeter coordinate p",
        outdir / "internal_exit_perimeter_cdf.png",
        logx=False,
    )
    plot_cdf_grid(
        [x[:, 3] for x in mc_targets],
        [x[:, 3] for x in gmc_targets],
        dims,
        "path length / time of flight",
        outdir / "internal_path_length_cdf.png",
        logx=True,
    )
    plot_direction_grid(mc_targets, gmc_targets, dims, outdir / "internal_exit_direction_scatter.png")
    metrics = np.array(rows, dtype=float)
    np.savetxt(
        outdir / "internal_validation_metrics.csv",
        metrics,
        delimiter=",",
        header="W,H,ks_p_exit,ks_path_length,mean_L_mc,mean_L_gmc",
        comments="",
    )
    print(f"saved internal validation figures and metrics to {outdir}")


if __name__ == "__main__":
    main()
