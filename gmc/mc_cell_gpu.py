"""Vectorized direct single-cell random-flight generators for CUDA/CPU torch.

These routines generate the *same physical training law* as :mod:`gmc.mc_cell`:
optical-coordinate pure scattering (Sigma_s=1), isotropic 3-D scattering, and
exit targets ``(p_exit, u_exit, v_exit, path_length)``.  The difference is only
execution: thousands to millions of independent histories are advanced in SIMD
batches on a torch device.

The global transport method remains deterministic-response based. This module
is only a direct local-MC data/reference accelerator; finite samples remain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
import torch

from .mc_cell import BoundaryBatch, InternalBatch


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be float32 or float64")


def _rand(shape, *, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def _log_uniform(n: int, lo: float, hi: float, *, generator, device, dtype) -> torch.Tensor:
    u = _rand((n,), generator=generator, device=device, dtype=dtype)
    l0 = math.log10(lo); l1 = math.log10(hi)
    return torch.pow(torch.tensor(10.0, device=device, dtype=dtype), l0 + (l1-l0)*u)


def _sample_isotropic(n: int, *, generator, device, dtype) -> torch.Tensor:
    z = 2.0*_rand((n,), generator=generator, device=device, dtype=dtype)-1.0
    phi = (2.0*math.pi)*_rand((n,), generator=generator, device=device, dtype=dtype)
    r = torch.sqrt(torch.clamp(1.0-z*z, min=0.0))
    return torch.stack((r*torch.cos(phi), r*torch.sin(phi), z), dim=1)


def _sample_left_boundary(n: int, mode: str, *, generator, device, dtype) -> torch.Tensor:
    u = _rand((n,), generator=generator, device=device, dtype=dtype)
    if mode == "cosine":
        mu = torch.sqrt(u)
    elif mode == "uniform":
        mu = u
    else:
        raise ValueError("mode must be cosine or uniform")
    eta = (2.0*math.pi)*_rand((n,), generator=generator, device=device, dtype=dtype)
    rt = torch.sqrt(torch.clamp(1.0-mu*mu, min=0.0))
    return torch.stack((mu, rt*torch.cos(eta), rt*torch.sin(eta)), dim=1)


def _distance_to_boundary(x, y, u, v, W, H) -> torch.Tensor:
    inf = torch.full_like(x, float("inf"))
    eps = 1.0e-14 if x.dtype == torch.float64 else 1.0e-7
    sx = torch.where(u > eps, (W-x)/u, torch.where(u < -eps, -x/u, inf))
    sy = torch.where(v > eps, (H-y)/v, torch.where(v < -eps, -y/v, inf))
    return torch.minimum(sx, sy)


def _perimeter_coordinate_vec(x, y, W, H) -> torch.Tensor:
    # Match mc_cell.perimeter_coordinate: choose closest face at corners.
    dist = torch.stack((torch.abs(y), torch.abs(x-W), torch.abs(y-H), torch.abs(x)), dim=1)
    face = torch.argmin(dist, dim=1)
    perim = 2.0*(W+H)
    s = torch.empty_like(x)
    m = face == 0
    s[m] = torch.clamp(x[m], min=0.0)  # upper clamp applied below with W
    s[m] = torch.minimum(s[m], W[m])
    m = face == 1
    yy = torch.minimum(torch.clamp(y[m], min=0.0), H[m]); s[m] = W[m] + yy
    m = face == 2
    xx = torch.minimum(torch.clamp(x[m], min=0.0), W[m]); s[m] = W[m] + H[m] + (W[m]-xx)
    m = face == 3
    yy = torch.minimum(torch.clamp(y[m], min=0.0), H[m]); s[m] = 2.0*W[m] + H[m] + (H[m]-yy)
    return torch.remainder(s/perim, 1.0)


@torch.no_grad()
def _advance_random_flights(
    W: torch.Tensor,
    H: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    dirs: torch.Tensor,
    *,
    generator: torch.Generator,
    max_collisions: int = 2_000_000,
) -> torch.Tensor:
    """Advance independent pure-scattering histories until each exits.

    Returns target tensor ``[p,u,v,L]`` on the same device/dtype.
    """
    n = W.numel(); device=W.device; dtype=W.dtype
    u=dirs[:,0].clone(); v=dirs[:,1].clone()
    L=torch.zeros(n,device=device,dtype=dtype)
    out=torch.empty((n,4),device=device,dtype=dtype)
    active=torch.ones(n,device=device,dtype=torch.bool)
    collisions=0
    tiny = torch.finfo(dtype).tiny
    while bool(active.any()):
        ids=torch.nonzero(active,as_tuple=False).squeeze(1)
        na=ids.numel()
        # Exact Exp(1) inverse-CDF, clamped only against log(0).
        r=_rand((na,),generator=generator,device=device,dtype=dtype)
        sc=-torch.log(torch.clamp(1.0-r,min=tiny))
        xa=x[ids]; ya=y[ids]; ua=u[ids]; va=v[ids]; Wa=W[ids]; Ha=H[ids]
        sb=_distance_to_boundary(xa,ya,ua,va,Wa,Ha)
        exiting = sc >= sb
        step=torch.minimum(sc,sb)
        xnew=xa+step*ua; ynew=ya+step*va
        x[ids]=xnew; y[ids]=ynew; L[ids]+=step
        if bool(exiting.any()):
            eid=ids[exiting]
            p=_perimeter_coordinate_vec(xnew[exiting],ynew[exiting],Wa[exiting],Ha[exiting])
            out[eid,0]=p; out[eid,1]=ua[exiting]; out[eid,2]=va[exiting]; out[eid,3]=L[eid]
            active[eid]=False
        coll=~exiting
        if bool(coll.any()):
            cid=ids[coll]
            nd=_sample_isotropic(cid.numel(),generator=generator,device=device,dtype=dtype)
            u[cid]=nd[:,0]; v[cid]=nd[:,1]
        collisions += 1
        if collisions >= max_collisions:
            remaining=int(active.sum().item())
            raise RuntimeError(f"GPU random flight exceeded max_collisions={max_collisions}; remaining={remaining}")
    return out


def _device_from_backend(backend: str) -> torch.device:
    if backend == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(backend)


def generate_boundary_dataset_torch(
    n: int,
    seed: int = 1234,
    W_range: Tuple[float,float] = (0.015,5.0),
    H_range: Tuple[float,float] = (0.015,5.0),
    direction_mode: str = "cosine",
    backend: str = "auto",
    chunk_size: int = 262_144,
    dtype: str = "float64",
    progress: bool = True,
) -> BoundaryBatch:
    device=_device_from_backend(backend); td=_torch_dtype(dtype)
    cond=np.empty((n,5),dtype=np.float32); targ=np.empty((n,4),dtype=np.float32)
    gen=torch.Generator(device=device); gen.manual_seed(int(seed))
    done=0
    while done<n:
        m=min(chunk_size,n-done)
        W=_log_uniform(m,*W_range,generator=gen,device=device,dtype=td)
        H=_log_uniform(m,*H_range,generator=gen,device=device,dtype=td)
        yin=_rand((m,),generator=gen,device=device,dtype=td)*H
        dirs=_sample_left_boundary(m,direction_mode,generator=gen,device=device,dtype=td)
        x=torch.zeros(m,device=device,dtype=td); y=yin.clone()
        out=_advance_random_flights(W,H,x,y,dirs,generator=gen)
        c=torch.stack((W,H,yin,dirs[:,0],dirs[:,1]),dim=1)
        cond[done:done+m]=c.float().cpu().numpy(); targ[done:done+m]=out.float().cpu().numpy()
        done+=m
        if progress: print(f"generated {done}/{n} on {device}",flush=True)
    return BoundaryBatch(cond,targ)


def generate_internal_dataset_torch(
    n: int,
    seed: int = 4321,
    W_range: Tuple[float,float] = (0.015,5.0),
    H_range: Tuple[float,float] = (0.015,5.0),
    backend: str = "auto",
    chunk_size: int = 262_144,
    dtype: str = "float64",
    progress: bool = True,
) -> InternalBatch:
    device=_device_from_backend(backend); td=_torch_dtype(dtype)
    cond=np.empty((n,6),dtype=np.float32); targ=np.empty((n,4),dtype=np.float32)
    gen=torch.Generator(device=device); gen.manual_seed(int(seed))
    done=0
    while done<n:
        m=min(chunk_size,n-done)
        W=_log_uniform(m,*W_range,generator=gen,device=device,dtype=td)
        H=_log_uniform(m,*H_range,generator=gen,device=device,dtype=td)
        xin=_rand((m,),generator=gen,device=device,dtype=td)*W
        yin=_rand((m,),generator=gen,device=device,dtype=td)*H
        dirs=_sample_isotropic(m,generator=gen,device=device,dtype=td)
        out=_advance_random_flights(W,H,xin.clone(),yin.clone(),dirs,generator=gen)
        c=torch.stack((W,H,xin,yin,dirs[:,0],dirs[:,1]),dim=1)
        cond[done:done+m]=c.float().cpu().numpy(); targ[done:done+m]=out.float().cpu().numpy()
        done+=m
        if progress: print(f"generated {done}/{n} on {device}",flush=True)
    return InternalBatch(cond,targ)

@torch.no_grad()
def sample_boundary_conditions_torch(
    raw_conditions: np.ndarray,
    seed: int = 1,
    backend: str = 'auto',
    dtype: str = 'float64',
) -> np.ndarray:
    """Exact random flights for explicit boundary conditions [W,H,y,u,v]."""
    device=_device_from_backend(backend); td=_torch_dtype(dtype)
    raw=torch.as_tensor(raw_conditions,dtype=td,device=device)
    W,H,yin,u,v=[raw[:,i] for i in range(5)]
    w=torch.sqrt(torch.clamp(1.0-u*u-v*v,min=0.0)); dirs=torch.stack((u,v,w),dim=1)
    gen=torch.Generator(device=device); gen.manual_seed(int(seed))
    out=_advance_random_flights(W,H,torch.zeros_like(W),yin.clone(),dirs,generator=gen)
    return out.float().cpu().numpy()


@torch.no_grad()
def sample_internal_conditions_torch(
    raw_conditions: np.ndarray,
    seed: int = 1,
    backend: str = 'auto',
    dtype: str = 'float64',
) -> np.ndarray:
    """Exact random flights for explicit internal conditions [W,H,x,y,u,v]."""
    device=_device_from_backend(backend); td=_torch_dtype(dtype)
    raw=torch.as_tensor(raw_conditions,dtype=td,device=device)
    W,H,x,y,u,v=[raw[:,i] for i in range(6)]
    w=torch.sqrt(torch.clamp(1.0-u*u-v*v,min=0.0)); dirs=torch.stack((u,v,w),dim=1)
    gen=torch.Generator(device=device); gen.manual_seed(int(seed))
    out=_advance_random_flights(W,H,x.clone(),y.clone(),dirs,generator=gen)
    return out.float().cpu().numpy()
