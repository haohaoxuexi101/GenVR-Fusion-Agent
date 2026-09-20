#!/usr/bin/env python
"""Generate exact internal-source random-flight training data, CUDA-vectorized."""
from __future__ import annotations
import argparse,time,sys
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from gmc.mc_cell import generate_internal_dataset,save_npz
from gmc.mc_cell_gpu import generate_internal_dataset_torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--n',type=int,default=110000); p.add_argument('--seed',type=int,default=4321)
    p.add_argument('--out',default='outputs/internal_raw_110k.npz')
    p.add_argument('--wmin',type=float,default=0.015); p.add_argument('--wmax',type=float,default=5.0)
    p.add_argument('--hmin',type=float,default=0.015); p.add_argument('--hmax',type=float,default=5.0)
    p.add_argument('--backend',choices=['auto','cpu','cuda','numpy'],default='auto')
    p.add_argument('--chunk-size',type=int,default=262144); p.add_argument('--mc-dtype',choices=['float32','float64'],default='float64')
    p.add_argument('--compress',action='store_true',help='use slower np.savez_compressed for archival')
    a=p.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); t=time.perf_counter()
    if a.backend=='numpy':
        batch=generate_internal_dataset(a.n,a.seed,(a.wmin,a.wmax),(a.hmin,a.hmax),progress_every=max(1,a.n//20))
    else:
        batch=generate_internal_dataset_torch(a.n,a.seed,(a.wmin,a.wmax),(a.hmin,a.hmax),backend=a.backend,
            chunk_size=a.chunk_size,dtype=a.mc_dtype,progress=True)
    if a.compress: save_npz(str(out),batch)
    else: np.savez(str(out),conditions=batch.conditions,targets=batch.targets)
    print(f'saved {a.n} internal samples to {out}; elapsed={time.perf_counter()-t:.3f}s')

if __name__=='__main__': main()
