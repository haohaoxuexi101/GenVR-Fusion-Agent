#!/usr/bin/env python
"""Figure-3.2a-style statistical convergence for lattice and hohlraum."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import torch
from gmc.benchmarks2d import make_lattice_problem, make_hohlraum_problem
from gmc.transport2d import BoundaryGMCSampler, InternalGMCSampler, run_standard_mc, run_gmc_full_transport_batched


def avg_cell_std(fluxes):
    arr=np.stack(fluxes,axis=0)
    return float(np.mean(np.std(arr,axis=0,ddof=1)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--boundary-ckpt',required=True)
    ap.add_argument('--internal-ckpt',required=True)
    ap.add_argument('--outdir',default='outputs/convergence')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n-list',default='100000,200000,400000,800000,1600000,3200000,6400000,12800000')
    ap.add_argument('--runs',type=int,default=5)
    ap.add_argument('--max-steps',type=int,default=200000)
    ap.add_argument('--batch',type=int,default=65536)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    boundary=BoundaryGMCSampler(args.boundary_ckpt,device=args.device)
    internal=InternalGMCSampler(args.internal_ckpt,device=args.device)
    n_list=[int(x) for x in args.n_list.split(',')]
    problems={'lattice':make_lattice_problem(112,112),'hohlraum':make_hohlraum_problem(112,112)}
    rows=[]
    for pname,prob in problems.items():
        for n in n_list:
            mc_flux=[]; gmc_flux=[]
            for k in range(args.runs):
                t0=time.time(); mc=run_standard_mc(prob,n,seed=10000+17*k+n,max_steps_per_history=args.max_steps); tmc=time.time()-t0
                t0=time.time(); gmc=run_gmc_full_transport_batched(prob,boundary,internal if prob.source_kind=='volume_box' else None,n,seed=20000+19*k+n,max_steps_per_history=args.max_steps,inference_batch_size=args.batch); tgmc=time.time()-t0
                mc_flux.append(mc.flux); gmc_flux.append(gmc.flux)
                rows.append({'benchmark':pname,'n':n,'run':k,'mc_steps':mc.steps,'gmc_steps':gmc.steps,'mc_seconds':tmc,'gmc_seconds':tgmc})
                print(rows[-1], flush=True)
            rows.append({'benchmark':pname,'n':n,'run':'summary','mc_avg_cell_std':avg_cell_std(mc_flux),'gmc_avg_cell_std':avg_cell_std(gmc_flux)})
    with open(outdir/'convergence_raw.json','w') as f: json.dump(rows,f,indent=2)
    plt.figure(figsize=(7,5))
    for pname in problems:
        sub=[r for r in rows if r.get('run')=='summary' and r['benchmark']==pname]
        plt.loglog([r['n'] for r in sub],[r['mc_avg_cell_std'] for r in sub],'o-',label=f'MC {pname}')
        plt.loglog([r['n'] for r in sub],[r['gmc_avg_cell_std'] for r in sub],'s--',label=f'GMC {pname}')
    n0=n_list[0]
    # reference slope line scaled to first available MC point
    first=[r for r in rows if r.get('run')=='summary'][0]
    c=first['mc_avg_cell_std']*(n0**0.5)
    plt.loglog(n_list,[c/np.sqrt(n) for n in n_list],':',label=r'$N^{-1/2}$')
    plt.xlabel('Number of particles')
    plt.ylabel('cell-averaged standard deviation')
    plt.title('Statistical convergence')
    plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(outdir/'convergence.png',dpi=220)

if __name__=='__main__':
    main()
