#!/usr/bin/env python
"""Figure-3.2-style single-cell runtime scaling: MC vs GMC as optical size grows."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import torch
from gmc.mc_cell import generate_boundary_dataset
from gmc.transport2d import BoundaryGMCSampler


def time_mc(W: float, H: float, n: int, seed: int) -> float:
    t0 = time.time()
    _ = generate_boundary_dataset(n=n, seed=seed, fixed_W=W, fixed_H=H, progress_every=0)
    return time.time() - t0


def time_gmc(sampler: BoundaryGMCSampler, W: float, H: float, n: int, seed: int, batch: int) -> float:
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, H, size=n)
    # Cosine boundary source: mu=sqrt(xi), azimuth uniform.
    mu = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    eta = rng.uniform(0.0, 2*np.pi, size=n)
    rt = np.sqrt(np.maximum(0.0, 1.0 - mu*mu))
    raw = np.stack([np.full(n, W), np.full(n, H), y, mu, rt*np.cos(eta)], axis=1).astype(np.float32)
    if sampler.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    _ = sampler.sample_batch(raw, batch_size=batch)
    if sampler.device == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary-ckpt", required=True)
    ap.add_argument("--outdir", default="outputs/runtime_scaling")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-list", default="5000,20000,80000,320000")
    ap.add_argument("--sizes", default="0.015,0.05,0.1,0.5,1,5,10,50,100")
    ap.add_argument("--batch", type=int, default=65536)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sampler = BoundaryGMCSampler(args.boundary_ckpt, device=args.device)
    sizes = [float(x) for x in args.sizes.split(',')]
    n_list = [int(x) for x in args.n_list.split(',')]
    rows=[]
    for n in n_list:
        for W in sizes:
            H=W
            mc_t = time_mc(W,H,n,seed=1000+n+int(W*1000))
            gmc_t = time_gmc(sampler,W,H,n,seed=2000+n+int(W*1000),batch=args.batch)
            rows.append({"n":n,"W":W,"H":H,"mc_seconds":mc_t,"gmc_seconds":gmc_t,"speedup_mc_over_gmc":mc_t/gmc_t if gmc_t>0 else None})
            print(rows[-1], flush=True)
    with open(outdir/"runtime_scaling.json","w") as f: json.dump(rows,f,indent=2)
    plt.figure(figsize=(7,5))
    for n in n_list:
        sub=[r for r in rows if r['n']==n]
        plt.loglog([r['W'] for r in sub],[r['mc_seconds'] for r in sub],'o-',label=f"MC N={n}")
        plt.loglog([r['W'] for r in sub],[r['gmc_seconds'] for r in sub],'s--',label=f"GMC N={n}")
    plt.xlabel("Cell optical width W=H [mean free paths]")
    plt.ylabel("wall time [s]")
    plt.title("Runtime scaling: standard MC vs GMC cell transmission")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir/"runtime_scaling.png",dpi=220)

if __name__ == '__main__':
    main()
