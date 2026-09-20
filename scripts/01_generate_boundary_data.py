#!/usr/bin/env python
"""Generate exact pure-scattering MC data for the boundary GMC model.

Use ``--backend cuda`` to advance many random flights concurrently on the GPU.
The output law is the same optical-coordinate boundary random flight used by the
CPU reference generator; only execution is vectorized.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from gmc.mc_cell import generate_boundary_dataset,save_npz
from gmc.mc_cell_gpu import generate_boundary_dataset_torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=50_000); p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--out',default='outputs/boundary_raw_50k.npz')
    p.add_argument('--wmin',type=float,default=0.015); p.add_argument('--wmax',type=float,default=5.0)
    p.add_argument('--hmin',type=float,default=0.015); p.add_argument('--hmax',type=float,default=5.0)
    p.add_argument('--direction-mode',choices=['cosine','uniform'],default='cosine')
    p.add_argument('--backend',choices=['auto','cpu','cuda','numpy'],default='auto')
    p.add_argument('--chunk-size',type=int,default=262144)
    p.add_argument('--mc-dtype',choices=['float32','float64'],default='float64')
    p.add_argument('--progress-every',type=int,default=5000)
    p.add_argument('--compress',action='store_true',help='use slower np.savez_compressed for archival')
    a=p.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter()
    if a.backend=='numpy':
        batch=generate_boundary_dataset(a.n,a.seed,(a.wmin,a.wmax),(a.hmin,a.hmax),a.direction_mode,progress_every=a.progress_every)
    else:
        batch=generate_boundary_dataset_torch(a.n,a.seed,(a.wmin,a.wmax),(a.hmin,a.hmax),a.direction_mode,
            backend=a.backend,chunk_size=a.chunk_size,dtype=a.mc_dtype,progress=True)
    if a.compress: save_npz(str(out),batch)
    else: np.savez(str(out),conditions=batch.conditions,targets=batch.targets)
    print(f'saved {a.n} samples to {out}; elapsed={time.perf_counter()-t:.3f}s')

if __name__=='__main__': main()
