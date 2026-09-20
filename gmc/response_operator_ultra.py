"""Stage 8 Ultra: high-fidelity CFM -> response-operator -> deterministic transport.

The original Stage 8 idea is preserved exactly:

    Farmer-style conditional cell sampler
        -> many OFFLINE conditional samples
        -> expected local boundary / track response operator
        -> deterministic global composition of expected interface current.

This module focuses on numerical correctness rather than a minimal teaching
implementation.  The important upgrades are:

* high-order S2-derived interface angular states instead of four hand-picked rays;
* linear conservative interpolation in face position;
* barycentric conservative interpolation in projected direction space;
* sample-by-sample implicit-capture attenuation;
* hard geometric repair including L >= entry/exit chord length;
* scrambled-Sobol Gaussian latents for low-noise CFM response integration;
* direct local-MC operator backend with the IDENTICAL phase-space projection;
* matrix-free deterministic global solve on CPU or CUDA;
* strict conservation and local-operator diagnostics.

The global solver contains no online particle histories.  Monte Carlo is used only
as a direct local-response validation backend; CFM can replace it as the fast local
response generator.  Finite sampling and finite interface projection remain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence, Tuple, List
import math
import numpy as np
import torch
from scipy.spatial import Delaunay

from .benchmarks2d import Structured2DProblem
from .mc_cell import (
    random_flight_boundary_sample,
    random_flight_sample_from_state,
    sample_isotropic_direction,
)
from .model import integrate_rk4_from_latent, integrate_rk4_from_latent_cached, integrate_rk4_from_latent_prepared, prepare_condition_cache, gather_condition_cache, sobol_standard_normal
from .mc_cell_gpu import sample_boundary_conditions_torch, sample_internal_conditions_torch
from .transport2d import (
    BoundaryGMCSampler,
    InternalGMCSampler,
    canonicalize_boundary_state,
    invert_canonical_exit,
    p_to_canonical_point,
)

Face = Literal["left", "right", "bottom", "top"]
FACES: Tuple[Face, ...] = ("left", "right", "bottom", "top")
FACE_TO_ID = {f: i for i, f in enumerate(FACES)}
OPPOSITE: Dict[Face, Face] = {"left": "right", "right": "left", "bottom": "top", "top": "bottom"}


def _product_s2_projected_half_quadrature(
    n_mu: int,
    n_phi: int,
    phi_offset_fraction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unique projected (normal,tangent) nodes on one inward hemisphere.

    We start from product Gauss-Legendre x uniform-azimuth S2 quadrature.  Because
    the 2-D transport geometry is invariant in z, +/- z directions with identical
    projected (x,y) are merged and their solid-angle weights are added.
    """
    if n_mu < 2 or n_phi < 8 or n_phi % 4 != 0:
        raise ValueError("use n_mu>=2 and n_phi>=8 divisible by 4")
    if not math.isfinite(phi_offset_fraction) or not 0.0 <= phi_offset_fraction < 1.0:
        raise ValueError("phi_offset_fraction must be finite and in [0,1)")
    muz, wmuz = np.polynomial.legendre.leggauss(n_mu)
    phis = (
        np.arange(n_phi, dtype=np.float64) + 0.5 + phi_offset_fraction
    ) * (2.0 * np.pi / n_phi)
    if np.any(np.abs(np.cos(phis)) <= 1.0e-13):
        raise ValueError(
            "phi_offset_fraction places nodes tangent to a face; choose a non-half-bin offset"
        )
    wphi = 2.0 * np.pi / n_phi
    merged: Dict[Tuple[float, float], float] = {}
    for mz, wz in zip(muz, wmuz):
        rho = math.sqrt(max(0.0, 1.0 - float(mz*mz)))
        for ph in phis:
            normal = rho * math.cos(float(ph))
            tang = rho * math.sin(float(ph))
            if normal <= 1.0e-13:
                continue
            key = (round(normal, 13), round(tang, 13))
            merged[key] = merged.get(key, 0.0) + float(wz*wphi)
    nodes = np.asarray(list(merged.keys()), dtype=np.float64)
    weights = np.asarray([merged[k] for k in merged], dtype=np.float64)
    # Stable ordering: increasing polar angle, then projected radius.
    ang = np.arctan2(nodes[:, 1], nodes[:, 0])
    rad = np.linalg.norm(nodes, axis=1)
    order = np.lexsort((rad, ang))
    return nodes[order], weights[order]


@dataclass
class UltraPhaseSpace:
    """Nodal face-position / projected-direction phase space.

    ``n_mu=4, n_phi=16`` produces 16 unique projected inward directions per face
    after z-symmetry merging.  With ``n_pos=4`` this gives 256 boundary states,
    versus only 32 in the original Stage 8 teaching discretization.
    """
    n_pos: int = 4
    n_mu: int = 4
    n_phi: int = 16
    phi_offset_fraction: float = 0.0
    angle_nodes: np.ndarray = field(init=False, repr=False)
    angle_weights: np.ndarray = field(init=False, repr=False)
    _tri: Delaunay = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_pos < 1:
            raise ValueError("n_pos must be >=1")
        self.angle_nodes, self.angle_weights = _product_s2_projected_half_quadrature(
            self.n_mu,
            self.n_phi,
            self.phi_offset_fraction,
        )
        if len(self.angle_nodes) < 3:
            raise RuntimeError("insufficient angular nodes")
        self._tri = Delaunay(self.angle_nodes)

    @property
    def n_angle(self) -> int:
        return int(self.angle_nodes.shape[0])

    @property
    def n_state(self) -> int:
        return 4 * self.n_pos * self.n_angle

    @property
    def position_nodes(self) -> np.ndarray:
        return (np.arange(self.n_pos, dtype=np.float64) + 0.5) / self.n_pos

    def state_index(self, face: Face, pos: int, angle: int) -> int:
        return (FACE_TO_ID[face] * self.n_pos + int(pos)) * self.n_angle + int(angle)

    def decode_state(self, idx: int) -> Tuple[Face, int, int]:
        a = idx % self.n_angle
        q = idx // self.n_angle
        p = q % self.n_pos
        return FACES[q // self.n_pos], int(p), int(a)

    def face_position(self, face: Face, pos: int, dx: float, dy: float) -> Tuple[float, float]:
        f = float(self.position_nodes[pos])
        if face == "left": return 0.0, f*dy
        if face == "right": return dx, f*dy
        if face == "bottom": return f*dx, 0.0
        return f*dx, dy

    def direction_for_entry(self, face: Face, angle: int) -> Tuple[float, float]:
        normal, tang = map(float, self.angle_nodes[angle])
        if face == "left": return normal, tang
        if face == "right": return -normal, tang
        if face == "bottom": return tang, normal
        return tang, -normal

    def _canonical_direction(self, face: Face, u: float, v: float) -> Tuple[float, float]:
        if face == "left": return float(u), float(v)
        if face == "right": return float(-u), float(v)
        if face == "bottom": return float(v), float(u)
        return float(-v), float(u)

    def project_position(self, face: Face, x: float, y: float, dx: float, dy: float) -> List[Tuple[int, float]]:
        if self.n_pos == 1:
            return [(0, 1.0)]
        frac = float(y/dy if face in ("left", "right") else x/dx)
        frac = min(1.0, max(0.0, frac))
        nodes = self.position_nodes
        if frac <= nodes[0]: return [(0, 1.0)]
        if frac >= nodes[-1]: return [(self.n_pos-1, 1.0)]
        hi = int(np.searchsorted(nodes, frac, side="right"))
        lo = hi - 1
        den = float(nodes[hi] - nodes[lo])
        whi = (frac - float(nodes[lo])) / den
        return [(lo, 1.0-whi), (hi, whi)]

    def project_angle(self, entry_face: Face, u: float, v: float) -> List[Tuple[int, float]]:
        normal, tang = self._canonical_direction(entry_face, u, v)
        # The outgoing state enters the neighboring cell, hence normal should be >0.
        normal = max(normal, 1.0e-12)
        q = np.array([normal, tang], dtype=np.float64)
        simplex = int(self._tri.find_simplex(q))
        if simplex >= 0:
            T = self._tri.transform[simplex]
            bary12 = T[:2] @ (q - T[2])
            bary = np.array([bary12[0], bary12[1], 1.0 - bary12.sum()], dtype=np.float64)
            verts = self._tri.simplices[simplex]
            # Numerical roundoff at triangle edges only.
            bary = np.maximum(bary, 0.0)
            sb = float(bary.sum())
            if sb > 0.0:
                bary /= sb
                return [(int(vv), float(ww)) for vv, ww in zip(verts, bary) if ww > 1.0e-14]
        # Projected directions very near the unit-disk rim can lie just outside the
        # convex hull of Gauss nodes.  A smooth 4-nearest inverse-distance blend is
        # much less ray-prone than nearest-neighbor snapping and remains conservative.
        d2 = np.sum((self.angle_nodes - q[None, :])**2, axis=1)
        k = min(4, self.n_angle)
        ids = np.argpartition(d2, k-1)[:k]
        if float(np.min(d2[ids])) < 1.0e-24:
            j = int(ids[np.argmin(d2[ids])]); return [(j, 1.0)]
        w = 1.0 / np.maximum(d2[ids], 1.0e-18)
        w /= np.sum(w)
        return [(int(j), float(ww)) for j, ww in zip(ids, w)]

    def source_angle_probabilities(self) -> np.ndarray:
        """Discrete probabilities for a cosine/current-weighted left-face source."""
        mu = np.maximum(self.angle_nodes[:, 0], 0.0)
        p = self.angle_weights * mu
        p /= np.sum(p)
        return p


def _force_outward(face: Face, u: float, v: float, min_normal: float = 1.0e-10) -> Tuple[float, float]:
    if face == "left": u = -max(abs(float(u)), min_normal)
    elif face == "right": u = max(abs(float(u)), min_normal)
    elif face == "bottom": v = -max(abs(float(v)), min_normal)
    else: v = max(abs(float(v)), min_normal)
    r2 = u*u + v*v
    if r2 >= 1.0:
        scale = (1.0 - 1.0e-9) / math.sqrt(r2)
        u *= scale; v *= scale
    return float(u), float(v)


def _track_from_lopt(Lopt: np.ndarray, sigma_s: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
    Lopt = np.maximum(np.asarray(Lopt, dtype=np.float64), 0.0)
    if sigma_s <= 0.0:
        raise ValueError("Stage 8 optical response requires sigma_s > 0")
    kappa = sigma_a / sigma_s
    survival = np.exp(-kappa * Lopt)
    if sigma_a > 0.0:
        track = (1.0 - survival) / sigma_a
    else:
        track = Lopt / sigma_s
    return survival, track


def _canonical_output_geometry(p: float, W: float, H: float) -> Tuple[float, float, Face]:
    return p_to_canonical_point(float(min(1.0-1e-12, max(0.0, p))), float(W), float(H))


def _decode_perimeter_vectorized(p: np.ndarray, W: np.ndarray, H: np.ndarray):
    p=np.clip(np.asarray(p,dtype=np.float64),0.0,1.0-1e-12)
    W=np.asarray(W,dtype=np.float64); H=np.asarray(H,dtype=np.float64)
    s=p*(2.0*(W+H))
    xo=np.empty_like(s); yo=np.empty_like(s); face=np.empty(s.shape,dtype=np.int8)
    m0=s<W
    m1=(~m0)&(s<W+H)
    m2=(~m0)&(~m1)&(s<2.0*W+H)
    m3=~(m0|m1|m2)
    xo[m0]=s[m0]; yo[m0]=0.0; face[m0]=2  # bottom
    xo[m1]=W[m1]; yo[m1]=s[m1]-W[m1]; face[m1]=1  # right
    xo[m2]=2.0*W[m2]+H[m2]-s[m2]; yo[m2]=H[m2]; face[m2]=3  # top
    xo[m3]=0.0; yo[m3]=2.0*(W[m3]+H[m3])-s[m3]; face[m3]=0  # left
    return p,xo,yo,face


def _vectorized_outward(u: np.ndarray,v: np.ndarray,face: np.ndarray):
    u=np.asarray(u,dtype=np.float64).copy(); v=np.asarray(v,dtype=np.float64).copy()
    u[face==0]=-np.maximum(np.abs(u[face==0]),1e-10)
    u[face==1]= np.maximum(np.abs(u[face==1]),1e-10)
    v[face==2]=-np.maximum(np.abs(v[face==2]),1e-10)
    v[face==3]= np.maximum(np.abs(v[face==3]),1e-10)
    r2=u*u+v*v; bad=r2>=1.0
    if np.any(bad):
        sc=(1.0-1e-9)/np.sqrt(r2[bad]); u[bad]*=sc; v[bad]*=sc
    return u,v


def _repair_boundary_decoded(raw: np.ndarray, decoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized physical projection for boundary CFM samples."""
    out=np.asarray(decoded,dtype=np.float64).copy()
    good=np.all(np.isfinite(out),axis=1)&np.all(np.isfinite(raw),axis=1)
    if not np.any(good): return out,good
    ids=np.where(good)[0]; W=raw[ids,0].astype(np.float64); H=raw[ids,1].astype(np.float64); yin=raw[ids,2].astype(np.float64)
    pp,xo,yo,face=_decode_perimeter_vectorized(out[ids,0],W,H)
    u,v=_vectorized_outward(out[ids,1],out[ids,2],face)
    chord=np.hypot(xo,yo-yin); L=np.maximum.reduce([out[ids,3].astype(np.float64),chord*(1.0+1e-10),np.full_like(chord,1e-12)])
    out[ids,0]=pp; out[ids,1]=u; out[ids,2]=v; out[ids,3]=L
    good &= np.all(np.isfinite(out),axis=1)&(out[:,3]>0.0)
    return out,good


def _repair_internal_decoded(raw: np.ndarray, decoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out=np.asarray(decoded,dtype=np.float64).copy()
    good=np.all(np.isfinite(out),axis=1)&np.all(np.isfinite(raw),axis=1)
    if not np.any(good): return out,good
    ids=np.where(good)[0]; W=raw[ids,0].astype(np.float64); H=raw[ids,1].astype(np.float64); xin=raw[ids,2].astype(np.float64); yin=raw[ids,3].astype(np.float64)
    pp,xo,yo,face=_decode_perimeter_vectorized(out[ids,0],W,H)
    u,v=_vectorized_outward(out[ids,1],out[ids,2],face)
    chord=np.hypot(xo-xin,yo-yin); L=np.maximum.reduce([out[ids,3].astype(np.float64),chord*(1.0+1e-10),np.full_like(chord,1e-12)])
    out[ids,0]=pp; out[ids,1]=u; out[ids,2]=v; out[ids,3]=L
    good &= np.all(np.isfinite(out),axis=1)&(out[:,3]>0.0)
    return out,good

def _qmc_cfm_samples(sampler, raw: np.ndarray, seed: int, batch_size: int) -> np.ndarray:
    """Evaluate a trained CFM with scrambled-Sobol standard-normal latent points."""
    n = int(raw.shape[0])
    c_np = sampler.transform.encode_conditions(raw)
    out_chunks=[]
    # One global Sobol sequence; chunks only control GPU memory.
    z_all = sobol_standard_normal(n, sampler.config.y_dim, seed=seed, device="cpu", dtype=torch.float32)
    for start in range(0,n,batch_size):
        stop=min(n,start+batch_size)
        c=torch.as_tensor(c_np[start:stop],dtype=torch.float32,device=sampler.device)
        z=z_all[start:stop].to(device=sampler.device)
        yenc=integrate_rk4_from_latent(sampler.model,c,z,n_steps=sampler.n_steps)
        out_chunks.append(sampler.transform.decode_targets_np(yenc.cpu().numpy(), raw[start:stop]))
    return np.concatenate(out_chunks,axis=0)


def _project_one_output(
    disc: UltraPhaseSpace,
    entry_face: Face,
    p: float,
    uc: float,
    vc: float,
    dxc: float,
    dyc: float,
    dx: float,
    dy: float,
) -> List[Tuple[int,float]]:
    xc,yc,_ = p_to_canonical_point(float(p), dxc, dyc)
    x,y,u,v,exit_face = invert_canonical_exit(entry_face,xc,yc,float(uc),float(vc),dx,dy)
    u,v=_force_outward(exit_face,u,v)
    neighbor_entry=OPPOSITE[exit_face]
    pw=disc.project_position(exit_face,x,y,dx,dy)
    aw=disc.project_angle(neighbor_entry,u,v)
    terms=[]
    s=0.0
    for ip,wp in pw:
        for ia,wa in aw:
            ww=float(wp*wa)
            if ww<=0: continue
            terms.append((disc.state_index(exit_face,ip,ia),ww)); s+=ww
    if not terms or s<=0:
        raise RuntimeError("empty phase-space projection")
    # Exact conservation even after floating-point barycentric clipping.
    return [(j,w/s) for j,w in terms]


@dataclass
class UltraCellResponse:
    sigma_s: float
    sigma_a: float
    dx: float
    dy: float
    R: np.ndarray
    P: np.ndarray
    track: np.ndarray
    samples_per_state: int
    backend: str
    invalid_fraction: float
    chord_repair_fraction: float

    @property
    def survival(self) -> np.ndarray:
        return np.sum(self.R,axis=0)


@dataclass
class UltraInternalResponse:
    outgoing: np.ndarray
    track: float
    samples: int
    backend: str
    invalid_fraction: float
    chord_repair_fraction: float


@dataclass
class UltraGlobalResult:
    flux: np.ndarray
    method: str
    iterations: int
    converged: bool
    relative_residual: float
    nonzero_fraction: float
    min_positive_flux: float
    leakage: float
    total_track: float


def _input_condition(disc: UltraPhaseSpace, state: int, sigma_s: float, dx: float, dy: float):
    face,pos,ang=disc.decode_state(state)
    x,y=disc.face_position(face,pos,dx,dy)
    u,v=disc.direction_for_entry(face,ang)
    raw,dxc,dyc=canonicalize_boundary_state(face,x,y,u,v,dx,dy,sigma_s)
    return raw,face,dxc,dyc


def build_boundary_response_cfm_ultra(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    sampler: BoundaryGMCSampler,
    disc: UltraPhaseSpace,
    samples_per_state: int = 2048,
    seed: int = 1000,
    inference_batch_size: int = 32768,
    qmc: bool = True,
) -> UltraCellResponse:
    ns=disc.n_state
    P=np.zeros((ns,ns),dtype=np.float64)
    R=np.zeros_like(P)
    track=np.zeros(ns,dtype=np.float64)
    invalid=0; total=0; repaired=0
    for i in range(ns):
        raw1,entry_face,dxc,dyc=_input_condition(disc,i,sigma_s,dx,dy)
        raw=np.repeat(raw1,samples_per_state,axis=0)
        if qmc:
            dec=_qmc_cfm_samples(sampler,raw,seed+7919*i,inference_batch_size)
        else:
            dec=sampler.sample_batch(raw,seed=seed+7919*i,batch_size=inference_batch_size)
        before=np.asarray(dec[:,3],dtype=np.float64).copy()
        dec,good=_repair_boundary_decoded(raw,dec)
        repaired += int(np.sum(dec[:,3] > before*(1.0+1e-12)))
        total += samples_per_state
        invalid += int(np.sum(~good))
        if not np.all(good):
            # Invalid neural values should not silently remove probability mass.
            # Replace them by fresh iid draws until every requested sample is valid.
            bad=np.where(~good)[0]
            attempts=0
            while len(bad)>0 and attempts<8:
                rr=np.repeat(raw1,len(bad),axis=0)
                fresh=sampler.sample_batch(rr,seed=seed+104729*i+attempts*100003,batch_size=inference_batch_size)
                fresh,g2=_repair_boundary_decoded(rr,fresh)
                dec[bad[g2]]=fresh[g2]
                bad=bad[~g2]
                attempts+=1
            if len(bad)>0:
                raise RuntimeError(f"CFM produced {len(bad)} unrecoverable samples in state {i}")
        surv,tr=_track_from_lopt(dec[:,3],sigma_s,sigma_a)
        track[i]=float(np.mean(tr))
        for m in range(samples_per_state):
            terms=_project_one_output(disc,entry_face,float(dec[m,0]),float(dec[m,1]),float(dec[m,2]),dxc,dyc,dx,dy)
            for j,wproj in terms:
                P[j,i]+=wproj/samples_per_state
                R[j,i]+=float(surv[m])*wproj/samples_per_state
        # Enforce probability conservation at machine precision.  R is not
        # separately normalized: its column sum must remain the sample mean survival.
        ps=float(np.sum(P[:,i]))
        if ps<=0: raise RuntimeError("zero response probability column")
        P[:,i]/=ps
    return UltraCellResponse(sigma_s,sigma_a,dx,dy,R,P,track,samples_per_state,
                             "cfm-qmc" if qmc else "cfm-iid",invalid/max(total,1),repaired/max(total,1))




def _project_outputs_vectorized(
    disc: UltraPhaseSpace,
    entry_face: Face,
    p: np.ndarray,
    uc: np.ndarray,
    vc: np.ndarray,
    dxc: float,
    dyc: float,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Conservatively project many continuous exits to discrete states at once.

    This is numerically equivalent to repeated ``_project_one_output`` calls but
    removes the Python/SciPy-per-sample bottleneck.  Each sample contributes to at
    most 2 position nodes x 4 angular nodes = 8 discrete states; projection weights
    sum to one sample-by-sample.
    """
    p=np.asarray(p,dtype=np.float64); uc=np.asarray(uc,dtype=np.float64); vc=np.asarray(vc,dtype=np.float64)
    n=p.size
    Dxc=np.full(n,float(dxc)); Dyc=np.full(n,float(dyc))
    _,xc,yc,_cface=_decode_perimeter_vectorized(p,Dxc,Dyc)
    # Canonical -> physical local coordinates/directions.
    if entry_face=='left':
        x,y,u,v=xc,yc,uc.copy(),vc.copy()
    elif entry_face=='right':
        x,y,u,v=dx-xc,yc,-uc,vc.copy()
    elif entry_face=='bottom':
        x,y,u,v=yc,xc,vc,uc
    elif entry_face=='top':
        x,y,u,v=yc,dy-xc,vc,-uc
    else: raise ValueError(entry_face)
    x=np.clip(x,0.0,dx); y=np.clip(y,0.0,dy)
    dist=np.stack((np.abs(x),np.abs(x-dx),np.abs(y),np.abs(y-dy)),axis=1)
    face_id=np.argmin(dist,axis=1).astype(np.int64)  # left,right,bottom,top
    u,v=_vectorized_outward(u,v,face_id.astype(np.int8))

    # Position interpolation (uniform cell-face nodes).
    if disc.n_pos==1:
        pos_ids=np.zeros((n,2),dtype=np.int64); pos_w=np.zeros((n,2)); pos_w[:,0]=1.0
    else:
        frac=np.where((face_id==0)|(face_id==1),y/dy,x/dx)
        frac=np.clip(frac,0.0,1.0); nodes=disc.position_nodes
        hi=np.searchsorted(nodes,frac,side='right'); lo=hi-1
        pos_ids=np.empty((n,2),dtype=np.int64); pos_w=np.zeros((n,2),dtype=np.float64)
        below=frac<=nodes[0]; above=frac>=nodes[-1]; mid=~(below|above)
        pos_ids[below,0]=0; pos_ids[below,1]=0; pos_w[below,0]=1.0
        pos_ids[above,0]=disc.n_pos-1; pos_ids[above,1]=disc.n_pos-1; pos_w[above,0]=1.0
        if np.any(mid):
            lm=lo[mid]; hm=hi[mid]; den=nodes[hm]-nodes[lm]; wh=(frac[mid]-nodes[lm])/den
            pos_ids[mid,0]=lm; pos_ids[mid,1]=hm; pos_w[mid,0]=1.0-wh; pos_w[mid,1]=wh

    # Outgoing physical direction is incoming direction of the neighbor.  Map to
    # that neighbor face's canonical inward (normal,tangent) coordinates.
    neigh=np.empty_like(face_id)
    neigh[face_id==0]=1; neigh[face_id==1]=0; neigh[face_id==2]=3; neigh[face_id==3]=2
    normal=np.empty(n); tang=np.empty(n)
    m=neigh==0; normal[m]=u[m]; tang[m]=v[m]       # left
    m=neigh==1; normal[m]=-u[m]; tang[m]=v[m]      # right
    m=neigh==2; normal[m]=v[m]; tang[m]=u[m]       # bottom
    m=neigh==3; normal[m]=-v[m]; tang[m]=u[m]      # top
    normal=np.maximum(normal,1.0e-12)
    q=np.column_stack((normal,tang))

    # Batched Delaunay barycentric interpolation; near the convex-hull rim use
    # exactly the same conservative 4-nearest inverse-distance fallback.
    aid=np.full((n,4),-1,dtype=np.int64); aw=np.zeros((n,4),dtype=np.float64)
    simplex=disc._tri.find_simplex(q)
    inside=simplex>=0
    if np.any(inside):
        ii=np.where(inside)[0]; simp=simplex[ii]; T=disc._tri.transform[simp]
        b12=np.einsum('nij,nj->ni',T[:,:2,:],q[ii]-T[:,2,:])
        bary=np.column_stack((b12,1.0-b12.sum(axis=1))); bary=np.maximum(bary,0.0)
        sb=bary.sum(axis=1); good=sb>0.0; bary[good]/=sb[good,None]
        verts=disc._tri.simplices[simp]
        aid[ii,:3]=verts; aw[ii,:3]=bary
        # Degenerate edge case falls through to kNN.
        inside[ii[~good]]=False
    outside=~inside
    if np.any(outside):
        oo=np.where(outside)[0]; qq=q[oo]
        d2=np.sum((qq[:,None,:]-disc.angle_nodes[None,:,:])**2,axis=2)
        k=min(4,disc.n_angle); ids=np.argpartition(d2,k-1,axis=1)[:,:k]
        dd=np.take_along_axis(d2,ids,axis=1)
        exact=np.min(dd,axis=1)<1.0e-24
        if np.any(exact):
            rr=np.where(exact)[0]; jj=np.argmin(dd[rr],axis=1); aid[oo[rr],0]=ids[rr,jj]; aw[oo[rr],0]=1.0
        if np.any(~exact):
            rr=np.where(~exact)[0]; ww=1.0/np.maximum(dd[rr],1.0e-18); ww/=ww.sum(axis=1,keepdims=True)
            aid[np.ix_(oo[rr],np.arange(k))]=ids[rr]; aw[np.ix_(oo[rr],np.arange(k))]=ww

    # Cartesian product of position and angular projection.  Invalid slots retain
    # weight zero and are assigned a harmless index 0.
    out_idx=np.zeros((n,8),dtype=np.int64); out_w=np.zeros((n,8),dtype=np.float64)
    t=0
    for ip in range(2):
        for ia in range(4):
            ww=pos_w[:,ip]*aw[:,ia]; jj=np.maximum(aid[:,ia],0)
            out_idx[:,t]=((face_id*disc.n_pos+pos_ids[:,ip])*disc.n_angle+jj)
            out_w[:,t]=ww; t+=1
    sw=out_w.sum(axis=1)
    if np.any(sw<=0.0): raise RuntimeError('empty vectorized phase-space projection')
    out_w/=sw[:,None]
    return out_idx,out_w


def build_boundary_response_cfm_ultra_batched(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    sampler: BoundaryGMCSampler,
    disc: UltraPhaseSpace,
    samples_per_state: int = 2048,
    seed: int = 1000,
    inference_batch_size: int = 65536,
    qmc: bool = True,
    symmetry_dedup: bool = True,
) -> UltraCellResponse:
    """High-throughput CFM response builder.

    Two exact reductions are used before any neural inference:

    * canonical-condition deduplication: equivalent physical entry faces share the
      same CFM conditional law.  Square cells reduce 4 face families to 1 (4x fewer
      neural samples); rectangular cells reduce to 2 (2x fewer samples).
    * fully vectorized conservative exit projection and histogramming.

    Neither reduction changes the represented local response law.
    """
    ns=disc.n_state; n=int(samples_per_state)
    state_raw=np.empty((ns,5),dtype=np.float32); faces=[]; dxcs=np.empty(ns); dycs=np.empty(ns)
    for i in range(ns):
        raw1,face,dxc,dyc=_input_condition(disc,i,sigma_s,dx,dy)
        state_raw[i]=raw1[0]; faces.append(face); dxcs[i]=dxc; dycs[i]=dyc
    if symmetry_dedup:
        uniq,inv=np.unique(state_raw,axis=0,return_inverse=True)
    else:
        uniq=state_raw.copy(); inv=np.arange(ns,dtype=np.int64)
    nu=len(uniq); total=nu*n
    raw=np.repeat(uniq,n,axis=0)
    if qmc:
        # Encode/precompute each unique conditional state exactly once.  Thousands
        # of latent samples for the same response column then gather that cache.
        cu_np=sampler.transform.encode_conditions(uniq)
        cu=torch.as_tensor(cu_np,dtype=torch.float32,device=sampler.device)
        cache_u=prepare_condition_cache(sampler.model,cu)
        z0=sobol_standard_normal(n,sampler.config.y_dim,seed=seed,device='cpu').numpy()
        z=np.tile(z0,(nu,1)); gids=np.repeat(np.arange(nu,dtype=np.int64),n)
        dec_chunks=[]
        for st in range(0,total,inference_batch_size):
            en=min(total,st+inference_batch_size)
            ids=torch.as_tensor(gids[st:en],dtype=torch.long,device=sampler.device)
            cache=gather_condition_cache(cache_u,ids)
            zz=torch.as_tensor(z[st:en],dtype=torch.float32,device=sampler.device)
            enc=integrate_rk4_from_latent_prepared(sampler.model,cache,zz,n_steps=sampler.n_steps)
            dec_chunks.append(sampler.transform.decode_targets_np(enc.cpu().numpy(), raw[st:en]))
        dec=np.concatenate(dec_chunks,axis=0)
    else:
        dec=sampler.sample_batch(raw,seed=seed,batch_size=inference_batch_size)
    before=dec[:,3].astype(np.float64).copy(); dec,good=_repair_boundary_decoded(raw,dec)
    invalid=int(np.sum(~good)); repaired=int(np.sum(dec[:,3]>before*(1+1e-12)))
    if invalid:
        bad=np.where(~good)[0]; attempts=0
        while len(bad) and attempts<8:
            fresh=sampler.sample_batch(raw[bad],seed=seed+1000003+attempts*104729,batch_size=inference_batch_size)
            fresh,g2=_repair_boundary_decoded(raw[bad],fresh)
            dec[bad[g2]]=fresh[g2]; bad=bad[~g2]; attempts+=1
        if len(bad): raise RuntimeError(f'{len(bad)} unrecoverable CFM samples')

    P=np.zeros((ns,ns),dtype=np.float64); R=np.zeros_like(P); track=np.zeros(ns)
    # Survival/track depend only on canonical condition, so calculate once per unique
    # condition and reuse for its symmetry-equivalent physical entry faces.
    surv_u=[]; tr_u=[]
    for g in range(nu):
        di=dec[g*n:(g+1)*n]; su,tr=_track_from_lopt(di[:,3],sigma_s,sigma_a); surv_u.append(su); tr_u.append(tr)
    for i in range(ns):
        g=int(inv[i]); di=dec[g*n:(g+1)*n]; surv=surv_u[g]; tr=tr_u[g]; track[i]=float(np.mean(tr))
        idx,w=_project_outputs_vectorized(disc,faces[i],di[:,0],di[:,1],di[:,2],float(dxcs[i]),float(dycs[i]),dx,dy)
        jf=idx.reshape(-1); wf=w.reshape(-1)
        P[:,i]=np.bincount(jf,weights=wf,minlength=ns)/n
        R[:,i]=np.bincount(jf,weights=(w*surv[:,None]).reshape(-1),minlength=ns)/n
        ps=float(P[:,i].sum())
        if ps<=0: raise RuntimeError(f'zero response column {i}')
        P[:,i]/=ps
    factor=float(ns)/float(nu)
    backend=('cfm-qmc-hpc' if qmc else 'cfm-iid-hpc')+f'-dedup{factor:.1f}x'
    return UltraCellResponse(sigma_s,sigma_a,dx,dy,R,P,track,n,backend,
                             invalid/max(total,1),repaired/max(total,1))

def build_boundary_response_mc_ultra(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    disc: UltraPhaseSpace,
    samples_per_state: int = 8192,
    seed: int = 2000,
) -> UltraCellResponse:
    rng=np.random.default_rng(seed)
    ns=disc.n_state
    P=np.zeros((ns,ns),dtype=np.float64); R=np.zeros_like(P); track=np.zeros(ns)
    W=sigma_s*dx; H=sigma_s*dy
    total=0
    for i in range(ns):
        raw1,entry_face,dxc,dyc=_input_condition(disc,i,sigma_s,dx,dy)
        yopt=float(raw1[0,2]); uc=float(raw1[0,3]); vc=float(raw1[0,4])
        wc=math.sqrt(max(0.0,1.0-uc*uc-vc*vc))
        L=np.empty(samples_per_state,dtype=np.float64)
        samples=[]
        for m in range(samples_per_state):
            p,uo,vo,ll=random_flight_boundary_sample(W,H,yopt,np.array([uc,vc,wc]),rng)
            samples.append((p,uo,vo)); L[m]=ll; total+=1
        surv,tr=_track_from_lopt(L,sigma_s,sigma_a); track[i]=float(np.mean(tr))
        for m,(p,uo,vo) in enumerate(samples):
            terms=_project_one_output(disc,entry_face,p,uo,vo,dxc,dyc,dx,dy)
            for j,wproj in terms:
                P[j,i]+=wproj/samples_per_state
                R[j,i]+=float(surv[m])*wproj/samples_per_state
        P[:,i]/=float(np.sum(P[:,i]))
    return UltraCellResponse(sigma_s,sigma_a,dx,dy,R,P,track,samples_per_state,"direct-local-mc",0.0,0.0)


def _sobol_internal_raw_and_latent(W: float,H: float,n: int,latent_dim: int,seed: int):
    eng=torch.quasirandom.SobolEngine(dimension=4+latent_dim,scramble=True,seed=int(seed))
    q=eng.draw(n).numpy().astype(np.float64)
    x=q[:,0]*W; y=q[:,1]*H
    muz=2.0*q[:,2]-1.0; phi=2.0*np.pi*q[:,3]
    rho=np.sqrt(np.maximum(0.0,1.0-muz*muz))
    u=rho*np.cos(phi); v=rho*np.sin(phi)
    raw=np.column_stack([np.full(n,W),np.full(n,H),x,y,u,v]).astype(np.float32)
    u0=np.clip(q[:,4:],1e-7,1-1e-7)
    z=np.sqrt(2.0)*torch.erfinv(torch.as_tensor(2*u0-1,dtype=torch.float32)).numpy()
    return raw,z.astype(np.float32)


def _classify_internal(disc: UltraPhaseSpace,p:float,u:float,v:float,dx:float,dy:float) -> List[Tuple[int,float]]:
    x,y,exit_face=p_to_canonical_point(p,dx,dy)
    u,v=_force_outward(exit_face,u,v)
    neighbor=OPPOSITE[exit_face]
    pw=disc.project_position(exit_face,x,y,dx,dy); aw=disc.project_angle(neighbor,u,v)
    terms=[]; s=0.0
    for ip,wp in pw:
        for ia,wa in aw:
            ww=wp*wa; terms.append((disc.state_index(exit_face,ip,ia),ww)); s+=ww
    return [(j,float(w/s)) for j,w in terms]


def build_internal_response_cfm_ultra(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    sampler: InternalGMCSampler,
    disc: UltraPhaseSpace,
    samples: int = 32768,
    seed: int = 3000,
    inference_batch_size: int = 32768,
    qmc: bool = True,
) -> UltraInternalResponse:
    W,H=sigma_s*dx,sigma_s*dy
    if qmc:
        raw,z=_sobol_internal_raw_and_latent(W,H,samples,sampler.config.y_dim,seed)
        c_np=sampler.transform.encode_conditions(raw); chunks=[]
        for st in range(0,samples,inference_batch_size):
            en=min(samples,st+inference_batch_size)
            c=torch.as_tensor(c_np[st:en],dtype=torch.float32,device=sampler.device)
            zz=torch.as_tensor(z[st:en],dtype=torch.float32,device=sampler.device)
            enc=integrate_rk4_from_latent_cached(sampler.model,c,zz,n_steps=sampler.n_steps)
            chunks.append(sampler.transform.decode_targets_np(enc.cpu().numpy(), raw[st:en]))
        dec=np.concatenate(chunks,axis=0)
    else:
        rng=np.random.default_rng(seed); x=rng.uniform(0,W,samples); y=rng.uniform(0,H,samples); d=sample_isotropic_direction(rng,samples)
        raw=np.column_stack([np.full(samples,W),np.full(samples,H),x,y,d[:,0],d[:,1]]).astype(np.float32)
        dec=sampler.sample_batch(raw,seed=seed+17,batch_size=inference_batch_size)
    before=np.asarray(dec[:,3],dtype=np.float64).copy(); dec,good=_repair_internal_decoded(raw,dec)
    if not np.all(good):
        raise RuntimeError(f"internal CFM produced {np.sum(~good)} invalid samples")
    repaired=float(np.mean(dec[:,3] > before*(1+1e-12)))
    surv,tr=_track_from_lopt(dec[:,3],sigma_s,sigma_a)
    idx,wproj=_project_outputs_vectorized(disc,'left',dec[:,0],dec[:,1],dec[:,2],dx,dy,dx,dy)
    vec=np.bincount(idx.reshape(-1),weights=(wproj*surv[:,None]).reshape(-1),minlength=disc.n_state)/samples
    return UltraInternalResponse(vec,float(np.mean(tr)),samples,"cfm-qmc-hpc" if qmc else "cfm-iid-hpc",0.0,repaired)


def build_internal_response_mc_ultra(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    disc: UltraPhaseSpace,
    samples: int = 65536,
    seed: int = 4000,
) -> UltraInternalResponse:
    rng=np.random.default_rng(seed); W,H=sigma_s*dx,sigma_s*dy
    vec=np.zeros(disc.n_state,dtype=np.float64); L=np.empty(samples)
    outs=[]
    for m in range(samples):
        x=float(rng.uniform(0,W)); y=float(rng.uniform(0,H)); d=sample_isotropic_direction(rng,1)[0]
        p,u,v,ll=random_flight_sample_from_state(W,H,x,y,d,rng); L[m]=ll; outs.append((p,u,v))
    surv,tr=_track_from_lopt(L,sigma_s,sigma_a)
    for m,(p,u,v) in enumerate(outs):
        for j,wproj in _classify_internal(disc,p,u,v,dx,dy):
            vec[j]+=float(surv[m])*wproj/samples
    return UltraInternalResponse(vec,float(np.mean(tr)),samples,"direct-local-mc",0.0,0.0)



def build_boundary_response_mc_ultra_vectorized(
    sigma_s: float, sigma_a: float, dx: float, dy: float, disc: UltraPhaseSpace,
    samples_per_state: int = 8192, seed: int = 2000, device: str = 'auto',
    mc_dtype: str = 'float64', symmetry_dedup: bool = True,
) -> UltraCellResponse:
    """Direct local-MC reference using batched torch random flights and symmetries."""
    ns=disc.n_state; n=int(samples_per_state)
    state_raw=np.empty((ns,5),dtype=np.float32); faces=[]; dxcs=np.empty(ns); dycs=np.empty(ns)
    for i in range(ns):
        raw1,face,dxc,dyc=_input_condition(disc,i,sigma_s,dx,dy)
        state_raw[i]=raw1[0]; faces.append(face); dxcs[i]=dxc; dycs[i]=dyc
    if symmetry_dedup: uniq,inv=np.unique(state_raw,axis=0,return_inverse=True)
    else: uniq=state_raw.copy(); inv=np.arange(ns,dtype=np.int64)
    nu=len(uniq); raw=np.repeat(uniq,n,axis=0)
    dec=sample_boundary_conditions_torch(raw,seed=seed,backend=device,dtype=mc_dtype)
    P=np.zeros((ns,ns),dtype=np.float64); R=np.zeros_like(P); track=np.zeros(ns)
    surv_u=[]; tr_u=[]
    for g in range(nu):
        di=dec[g*n:(g+1)*n]; su,tr=_track_from_lopt(di[:,3],sigma_s,sigma_a); surv_u.append(su); tr_u.append(tr)
    for i in range(ns):
        g=int(inv[i]); di=dec[g*n:(g+1)*n]; su=surv_u[g]; track[i]=float(np.mean(tr_u[g]))
        idx,w=_project_outputs_vectorized(disc,faces[i],di[:,0],di[:,1],di[:,2],float(dxcs[i]),float(dycs[i]),dx,dy)
        P[:,i]=np.bincount(idx.reshape(-1),weights=w.reshape(-1),minlength=ns)/n
        R[:,i]=np.bincount(idx.reshape(-1),weights=(w*su[:,None]).reshape(-1),minlength=ns)/n
        P[:,i]/=P[:,i].sum()
    return UltraCellResponse(sigma_s,sigma_a,dx,dy,R,P,track,n,f'direct-local-mc-vectorized-{device}',0.0,0.0)


def build_internal_response_mc_ultra_vectorized(
    sigma_s: float, sigma_a: float, dx: float, dy: float, disc: UltraPhaseSpace,
    samples: int = 65536, seed: int = 4000, device: str = 'auto', mc_dtype: str = 'float64',
) -> UltraInternalResponse:
    """Direct internal-source MC reference with vectorized random flights."""
    # Source states are generated with torch for full parallelism.
    dev = torch.device('cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device))
    td = torch.float64 if mc_dtype=='float64' else torch.float32
    gen=torch.Generator(device=dev); gen.manual_seed(int(seed))
    W=sigma_s*dx; H=sigma_s*dy
    q=torch.rand((samples,4),generator=gen,device=dev,dtype=td)
    x=q[:,0]*W; y=q[:,1]*H; muz=2.0*q[:,2]-1.0; phi=2.0*math.pi*q[:,3]
    rho=torch.sqrt(torch.clamp(1.0-muz*muz,min=0.0)); u=rho*torch.cos(phi); v=rho*torch.sin(phi)
    raw=torch.stack((torch.full_like(x,W),torch.full_like(y,H),x,y,u,v),dim=1).float().cpu().numpy()
    dec=sample_internal_conditions_torch(raw,seed=seed+17,backend=str(dev),dtype=mc_dtype)
    surv,tr=_track_from_lopt(dec[:,3],sigma_s,sigma_a)
    idx,w=_project_outputs_vectorized(disc,'left',dec[:,0],dec[:,1],dec[:,2],dx,dy,dx,dy)
    vec=np.bincount(idx.reshape(-1),weights=(w*surv[:,None]).reshape(-1),minlength=disc.n_state)/samples
    return UltraInternalResponse(vec,float(np.mean(tr)),samples,f'direct-local-mc-vectorized-{dev}',0.0,0.0)

def unique_materials(problem: Structured2DProblem) -> Sequence[Tuple[float,float]]:
    x=np.column_stack([problem.sigma_s.ravel(),problem.sigma_a.ravel()])
    return [tuple(map(float,r)) for r in np.unique(x,axis=0)]


def build_library(problem:Structured2DProblem,disc:UltraPhaseSpace,backend:str,
                  boundary_sampler:Optional[BoundaryGMCSampler]=None,samples_per_state:int=2048,
                  seed:int=5000,qmc:bool=True,inference_batch_size:int=32768,
                  mc_device:str='cpu',mc_dtype:str='float64') -> Dict[Tuple[float,float],UltraCellResponse]:
    lib={}
    for k,(ss,sa) in enumerate(unique_materials(problem)):
        if ss<=1.0e-14:
            op=build_boundary_response_streaming_ultra(sa,problem.dx,problem.dy,disc)
            lib[(ss,sa)]=op
            continue
        if backend=="cfm":
            if boundary_sampler is None: raise ValueError("boundary_sampler required")
            op=build_boundary_response_cfm_ultra_batched(ss,sa,problem.dx,problem.dy,boundary_sampler,disc,
                samples_per_state,seed+100003*k,inference_batch_size,qmc)
        elif backend=="mc":
            op=build_boundary_response_mc_ultra(ss,sa,problem.dx,problem.dy,disc,samples_per_state,seed+100003*k)
        elif backend=="mc-vectorized":
            op=build_boundary_response_mc_ultra_vectorized(ss,sa,problem.dx,problem.dy,disc,samples_per_state,seed+100003*k,device=mc_device,mc_dtype=mc_dtype)
        else: raise ValueError(backend)
        lib[(ss,sa)]=op
    return lib


def _source_mask(problem:Structured2DProblem):
    mask=np.zeros((problem.ny,problem.nx),dtype=bool); w=np.zeros_like(mask,dtype=np.float64)
    if problem.source_kind!="volume_box": return mask,w
    xmin,xmax,ymin,ymax=problem.source_box
    xc=(np.arange(problem.nx)+0.5)*problem.dx; yc=(np.arange(problem.ny)+0.5)*problem.dy
    X,Y=np.meshgrid(xc,yc); mask=(X>=xmin)&(X<xmax)&(Y>=ymin)&(Y<ymax)
    w[mask]=1.0/np.count_nonzero(mask); return mask,w


def _external_boundary_source_np(problem:Structured2DProblem,disc:UltraPhaseSpace) -> np.ndarray:
    b=np.zeros((problem.ny,problem.nx,disc.n_state),dtype=np.float64)
    if problem.source_kind!="left_boundary": return b
    ymin,ymax=problem.boundary_source_y_range or (0.0,problem.height)
    yc=(np.arange(problem.ny)+0.5)*problem.dy; active=np.where((yc>=ymin)&(yc<=ymax))[0]
    ap=disc.source_angle_probabilities()
    for iy in active:
        wc=1.0/len(active)
        for p in range(disc.n_pos):
            for a,pa in enumerate(ap):
                b[iy,0,disc.state_index("left",p,a)] += wc*float(pa)/disc.n_pos
    return b


def _route_np(out:np.ndarray,disc:UltraPhaseSpace) -> Tuple[np.ndarray,float]:
    ny,nx,ns=out.shape; q=out.reshape(ny,nx,4,disc.n_pos,disc.n_angle); nxt=np.zeros_like(q); leak=0.0
    # left exits -> left neighbor, enters its right face
    nxt[:, :-1, FACE_TO_ID["right"]] += q[:,1:,FACE_TO_ID["left"]]
    leak += float(np.sum(q[:,0,FACE_TO_ID["left"]]))
    nxt[:, 1:, FACE_TO_ID["left"]] += q[:,:-1,FACE_TO_ID["right"]]
    leak += float(np.sum(q[:,-1,FACE_TO_ID["right"]]))
    nxt[:-1,:,FACE_TO_ID["top"]] += q[1:,:,FACE_TO_ID["bottom"]]
    leak += float(np.sum(q[0,:,FACE_TO_ID["bottom"]]))
    nxt[1:,:,FACE_TO_ID["bottom"]] += q[:-1,:,FACE_TO_ID["top"]]
    leak += float(np.sum(q[-1,:,FACE_TO_ID["top"]]))
    return nxt.reshape(ny,nx,ns),leak


def _apply_local_np(x:np.ndarray,problem:Structured2DProblem,lib:Dict[Tuple[float,float],UltraCellResponse]) -> np.ndarray:
    out=np.zeros_like(x)
    for key,op in lib.items():
        ss,sa=key; mask=np.isclose(problem.sigma_s,ss)&np.isclose(problem.sigma_a,sa)
        if np.any(mask): out[mask]=x[mask]@op.R.T
    return out


def solve_global_source_iteration(
    problem:Structured2DProblem,
    lib:Dict[Tuple[float,float],UltraCellResponse],
    disc:UltraPhaseSpace,
    internal:Optional[UltraInternalResponse]=None,
    rtol:float=1e-10,
    atol:float=1e-14,
    max_iters:int=5000,
    device:str="cpu",
    dtype:str="float64",
    check_every:int=4,
    solver:str="source",
) -> UltraGlobalResult:
    """Solve the global deterministic response equation.

    The equation is ``x = b + K x`` with ``K = P R``.  Two matrix-free solvers are
    provided:

    * ``source``: positivity-preserving fixed-point/source iteration.  This remains
      the correctness reference.
    * ``bicgstab``: Krylov solve of ``(I-K)x=b``.  It usually needs far fewer local
      response applications when the spectral radius of K is close to one.
    * ``auto``: try BiCGSTAB first and fall back to source iteration if the Krylov
      solve fails, stagnates, or returns materially negative interface current.

    All large phase-space fields and response matrices stay resident on CUDA when a
    CUDA device is selected.  Host synchronization is limited to periodic residual
    checks.
    """
    solver=str(solver).lower()
    if solver not in ('source','bicgstab','auto'):
        raise ValueError(f'unknown solver={solver}')
    b=_external_boundary_source_np(problem,disc); direct_track=np.zeros((problem.ny,problem.nx),dtype=np.float64)
    if problem.source_kind=="volume_box":
        if internal is None: raise ValueError("internal response required")
        _,sw=_source_mask(problem); direct_track += sw*internal.track
        outsrc=sw[:,:,None]*internal.outgoing[None,None,:]
        routed,_=_route_np(outsrc,disc); b += routed
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    tdtype=torch.float64 if dtype=="float64" else torch.float32
    dev=torch.device(device)
    bt=torch.as_tensor(b,dtype=tdtype,device=dev)
    # Pre-upload material index lists and local response matrices once.
    mats=[]
    ssf=problem.sigma_s.ravel(); saf=problem.sigma_a.ravel()
    for key,op in lib.items():
        ids=np.where(np.isclose(ssf,key[0])&np.isclose(saf,key[1]))[0]
        mats.append((torch.as_tensor(ids,dtype=torch.long,device=dev),torch.as_tensor(op.R,dtype=tdtype,device=dev)))

    def apply_K(z: torch.Tensor) -> torch.Tensor:
        zf=z.reshape(-1,disc.n_state); of=torch.zeros_like(zf)
        for ids,R in mats:
            if ids.numel():
                of[ids]=zf[ids]@R.T
        q=of.reshape(problem.ny,problem.nx,4,disc.n_pos,disc.n_angle)
        n=torch.zeros_like(q)
        n[:,:-1,FACE_TO_ID["right"]] += q[:,1:,FACE_TO_ID["left"]]
        n[:,1:,FACE_TO_ID["left"]] += q[:,:-1,FACE_TO_ID["right"]]
        n[:-1,:,FACE_TO_ID["top"]] += q[1:,:,FACE_TO_ID["bottom"]]
        n[1:,:,FACE_TO_ID["bottom"]] += q[:-1,:,FACE_TO_ID["top"]]
        return n.reshape(problem.ny,problem.nx,disc.n_state)

    def apply_A(z: torch.Tensor) -> torch.Tensor:
        return z-apply_K(z)

    bnorm=max(float(torch.linalg.vector_norm(bt).item()),1e-300)
    check_every=max(1,int(check_every))

    def fixed_point(x0: Optional[torch.Tensor]=None):
        x=bt.clone() if x0 is None else x0.clone()
        rel=math.inf; conv=False; it=0
        for it in range(1,max_iters+1):
            xn=bt+apply_K(x)
            if it % check_every == 0 or it == 1 or it == max_iters:
                diff=torch.linalg.vector_norm(xn-x)
                dv=float(diff.item()); rel=dv/bnorm
                x=xn
                if dv<atol or rel<rtol:
                    conv=True; break
            else:
                x=xn
        resid=apply_A(x)-bt
        rel=float(torch.linalg.vector_norm(resid).item())/bnorm
        return x,it,conv,rel

    def _safe_scalar(z: torch.Tensor, tiny: float=1e-300) -> torch.Tensor:
        # Avoid host synchronization on scalar breakdown checks.  Preserve sign when
        # possible; a true zero is replaced by +tiny.
        az=torch.abs(z)
        eps=torch.as_tensor(tiny,device=z.device,dtype=z.dtype)
        sign=torch.where(z<0,-torch.ones_like(z),torch.ones_like(z))
        return torch.where(az>eps,z,sign*eps)

    def bicgstab(x0: Optional[torch.Tensor]=None):
        # Matrix-free BiCGSTAB for (I-K)x=b.  Coefficients remain 0-D CUDA tensors,
        # so only the periodic residual checks synchronize with the host.
        x=bt.clone() if x0 is None else x0.clone()
        r=bt-apply_A(x); rhat=r.clone()
        p=torch.zeros_like(r); v=torch.zeros_like(r)
        one=torch.ones((),device=dev,dtype=tdtype)
        rho_old=one.clone(); alpha=one.clone(); omega=one.clone()
        rel=float(torch.linalg.vector_norm(r).item())/bnorm
        if rel<rtol: return x,0,True,rel
        conv=False; it=0
        for it in range(1,max_iters+1):
            rho=torch.sum(rhat*r)
            beta=(rho/_safe_scalar(rho_old))*(alpha/_safe_scalar(omega))
            p=r+beta*(p-omega*v)
            v=apply_A(p)
            alpha=rho/_safe_scalar(torch.sum(rhat*v))
            s=r-alpha*v
            t=apply_A(s)
            tt=torch.sum(t*t)
            omega=torch.sum(t*s)/_safe_scalar(tt)
            x=x+alpha*p+omega*s
            r=s-omega*t
            rho_old=rho
            if it % check_every == 0 or it == 1 or it == max_iters:
                rel=float(torch.linalg.vector_norm(r).item())/bnorm
                if not math.isfinite(rel):
                    break
                if rel<rtol or float(torch.linalg.vector_norm(r).item())<atol:
                    conv=True; break
        resid=apply_A(x)-bt
        rel=float(torch.linalg.vector_norm(resid).item())/bnorm
        conv=bool(conv and math.isfinite(rel) and rel < max(rtol*5.0,1e-14))
        return x,it,conv,rel

    used=solver
    if solver=='source':
        x,it,conv,rel=fixed_point()
    else:
        x,it,conv,rel=bicgstab()
        # A materially negative solution is not acceptable for this positive
        # transport problem.  ``auto`` falls back to the monotone reference solve.
        scale=max(float(torch.max(torch.abs(x)).item()),1e-300)
        xmin=float(torch.min(x).item())
        bad_negative=xmin < -1e-10*scale
        if solver=='auto' and (not conv or bad_negative or not math.isfinite(rel)):
            x,it2,conv,rel=fixed_point(x0=torch.clamp(x,min=0.0))
            it += it2; used='auto->source'
        else:
            used='bicgstab'

    xnp=x.detach().cpu().numpy()
    tl=direct_track.copy()
    for key,op in lib.items():
        mask=np.isclose(problem.sigma_s,key[0])&np.isclose(problem.sigma_a,key[1])
        tl[mask]+=xnp[mask]@op.track
    # Leakage from all boundary exits produced by total x + direct internal source.
    out=_apply_local_np(xnp,problem,lib); _,leak=_route_np(out,disc)
    if problem.source_kind=="volume_box":
        _,sw=_source_mask(problem); outsrc=sw[:,:,None]*internal.outgoing[None,None,:]; _,lk0=_route_np(outsrc,disc); leak+=lk0
    flux=tl/problem.volume; pos=flux>0
    return UltraGlobalResult(flux,used,it,conv,rel,float(np.mean(pos)),
        float(np.min(flux[pos])) if np.any(pos) else 0.0,float(leak),float(np.sum(tl)))


def internal_operator_metrics(a:UltraInternalResponse,b:UltraInternalResponse) -> Dict[str,float]:
    av=np.asarray(a.outgoing,dtype=np.float64); bv=np.asarray(b.outgoing,dtype=np.float64)
    denom=max(float(np.sum(np.abs(bv))),1e-30)
    return {
        "attenuated_outgoing_l1":float(np.sum(np.abs(av-bv))),
        "attenuated_outgoing_rel_l1":float(np.sum(np.abs(av-bv))/denom),
        "outgoing_max_abs_element":float(np.max(np.abs(av-bv))),
        "mean_survival_a":float(np.sum(av)),
        "mean_survival_b":float(np.sum(bv)),
        "survival_abs_error":float(abs(np.sum(av)-np.sum(bv))),
        "track_rel_error":float(abs(a.track-b.track)/max(abs(b.track),1e-30)),
        "a_invalid_fraction":float(a.invalid_fraction),
        "a_chord_repair_fraction":float(a.chord_repair_fraction),
    }

def operator_metrics(a:UltraCellResponse,b:UltraCellResponse) -> Dict[str,float]:
    P1,P2=a.P,b.P; R1,R2=a.R,b.R
    col_l1=np.sum(np.abs(P1-P2),axis=0); rcol=np.sum(np.abs(R1-R2),axis=0)
    return {
        "prob_l1_mean_per_input":float(np.mean(col_l1)),
        "prob_l1_p95_per_input":float(np.quantile(col_l1,0.95)),
        "prob_l1_max_per_input":float(np.max(col_l1)),
        "prob_max_abs_element":float(np.max(np.abs(P1-P2))),
        "attenuated_l1_mean_per_input":float(np.mean(rcol)),
        "track_rel_l1":float(np.sum(np.abs(a.track-b.track))/max(np.sum(np.abs(b.track)),1e-30)),
        "a_probability_conservation_max":float(np.max(np.abs(np.sum(P1,axis=0)-1.0))),
        "b_probability_conservation_max":float(np.max(np.abs(np.sum(P2,axis=0)-1.0))),
        "a_invalid_fraction":float(a.invalid_fraction),
        "a_chord_repair_fraction":float(a.chord_repair_fraction),
        "a_mean_survival":float(np.mean(np.sum(R1,axis=0))),
        "b_mean_survival":float(np.mean(np.sum(R2,axis=0))),
    }


def flux_shape_metrics(a:np.ndarray,b:np.ndarray) -> Dict[str,float]:
    aa=np.asarray(a,dtype=np.float64); bb=np.asarray(b,dtype=np.float64)
    an=aa/max(float(np.max(aa)),1e-300); bn=bb/max(float(np.max(bb)),1e-300)
    mask=(an>0)|(bn>0)
    return {
        "corr":float(np.corrcoef(an.ravel(),bn.ravel())[0,1]),
        "normalized_rel_l1":float(np.sum(np.abs(an-bn))/max(np.sum(np.abs(bn)),1e-300)),
        "log10_rmse":float(np.sqrt(np.mean((np.log10(np.maximum(an[mask],1e-15))-np.log10(np.maximum(bn[mask],1e-15)))**2))),
        "a_nonzero_fraction":float(np.mean(aa>0)),
        "b_nonzero_fraction":float(np.mean(bb>0)),
    }

# -----------------------------------------------------------------------------
# Exact ballistic singular-component split
# -----------------------------------------------------------------------------
from .mc_cell import distance_to_rectangle_boundary, perimeter_coordinate


def _conditional_first_collision_distance(u: np.ndarray, d0: float) -> np.ndarray:
    """Sample Exp(1) conditioned on collision before distance d0."""
    if d0 <= 0.0:
        return np.zeros_like(u,dtype=np.float64)
    one_minus = -math.expm1(-d0)  # 1-exp(-d0), stable for small d0
    return -np.log1p(-np.asarray(u,dtype=np.float64)*one_minus)


def build_boundary_response_ballistic_split_cfm_ultra(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    internal_sampler: InternalGMCSampler,
    disc: UltraPhaseSpace,
    scattered_samples_per_state: int = 2048,
    seed: int = 6000,
    inference_batch_size: int = 32768,
) -> UltraCellResponse:
    """Boundary response with the zero-collision point mass treated analytically.

    For an incoming state, the no-scatter probability exp(-d0) and ballistic exit
    are exact.  Conditional on at least one scatter, the first collision distance
    is sampled exactly and only the *post-first-collision* transmission is supplied
    by the smoother internal CFM.  This removes the hardest singular component from
    the neural density while preserving the Stage-8 response-matrix construction.
    """
    ns=disc.n_state; P=np.zeros((ns,ns)); R=np.zeros_like(P); track=np.zeros(ns)
    W,H=sigma_s*dx,sigma_s*dy; kappa=sigma_a/sigma_s
    repaired=0; total=0
    for i in range(ns):
        raw1,entry_face,dxc,dyc=_input_condition(disc,i,sigma_s,dx,dy)
        yin=float(raw1[0,2]); uc=float(raw1[0,3]); vc=float(raw1[0,4]); wc=math.sqrt(max(0.,1.-uc*uc-vc*vc))
        d0=distance_to_rectangle_boundary(0.0,yin,uc,vc,W,H)
        p0=math.exp(-d0)
        # exact ballistic response
        xb=uc*d0; yb=yin+vc*d0; pb=perimeter_coordinate(xb,yb,W,H)
        bterms=_project_one_output(disc,entry_face,pb,uc,vc,dxc,dyc,dx,dy)
        surv0=math.exp(-kappa*d0)
        if sigma_a>0: tr0=(1.0-surv0)/sigma_a
        else: tr0=d0/sigma_s
        for j,wproj in bterms:
            P[j,i]+=p0*wproj; R[j,i]+=p0*surv0*wproj
        if p0 >= 1.0-1e-14:
            track[i]=tr0; P[:,i]/=P[:,i].sum(); continue
        n=scattered_samples_per_state
        # Sobol dimensions: first-collision U, isotropic direction (2), CFM latent (4).
        eng=torch.quasirandom.SobolEngine(dimension=3+internal_sampler.config.y_dim,scramble=True,seed=int(seed+7919*i))
        q=eng.draw(n).numpy().astype(np.float64)
        sc=_conditional_first_collision_distance(q[:,0],d0)
        xin=uc*sc; ycin=yin+vc*sc
        muz=2.0*q[:,1]-1.0; phi=2*np.pi*q[:,2]; rho=np.sqrt(np.maximum(0.,1.-muz*muz))
        us=rho*np.cos(phi); vs=rho*np.sin(phi)
        raw=np.column_stack([np.full(n,W),np.full(n,H),xin,ycin,us,vs]).astype(np.float32)
        u0=np.clip(q[:,3:],1e-7,1-1e-7)
        z=(np.sqrt(2.0)*torch.erfinv(torch.as_tensor(2*u0-1,dtype=torch.float32))).numpy()
        c_np=internal_sampler.transform.encode_conditions(raw); chunks=[]
        for st in range(0,n,inference_batch_size):
            en=min(n,st+inference_batch_size)
            c=torch.as_tensor(c_np[st:en],dtype=torch.float32,device=internal_sampler.device)
            zz=torch.as_tensor(z[st:en],dtype=torch.float32,device=internal_sampler.device)
            enc=integrate_rk4_from_latent(internal_sampler.model,c,zz,n_steps=internal_sampler.n_steps)
            chunks.append(internal_sampler.transform.decode_targets_np(enc.cpu().numpy(), raw[st:en]))
        dec=np.concatenate(chunks,axis=0); before=dec[:,3].astype(np.float64).copy(); dec,good=_repair_internal_decoded(raw,dec)
        if not np.all(good): raise RuntimeError(f"ballistic-split internal CFM invalid samples: {np.sum(~good)}")
        repaired += int(np.sum(dec[:,3]>before*(1+1e-12))); total += n
        Ltot=sc+dec[:,3]
        surv,tr=_track_from_lopt(Ltot,sigma_s,sigma_a)
        ps=1.0-p0
        track[i]=p0*tr0 + ps*float(np.mean(tr))
        for m in range(n):
            terms=_project_one_output(disc,entry_face,float(dec[m,0]),float(dec[m,1]),float(dec[m,2]),dxc,dyc,dx,dy)
            for j,wproj in terms:
                P[j,i]+=ps*wproj/n; R[j,i]+=ps*float(surv[m])*wproj/n
        P[:,i]/=float(P[:,i].sum())
    return UltraCellResponse(sigma_s,sigma_a,dx,dy,R,P,track,scattered_samples_per_state,
                             "cfm-ballistic-split",0.0,repaired/max(total,1))


def build_boundary_response_ballistic_split_mc_ultra(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    disc: UltraPhaseSpace,
    scattered_samples_per_state: int = 4096,
    seed: int = 7000,
) -> UltraCellResponse:
    """Direct local-MC validation of the same analytic-ballistic decomposition."""
    rng=np.random.default_rng(seed); ns=disc.n_state; P=np.zeros((ns,ns)); R=np.zeros_like(P); track=np.zeros(ns)
    W,H=sigma_s*dx,sigma_s*dy; kappa=sigma_a/sigma_s
    for i in range(ns):
        raw1,entry_face,dxc,dyc=_input_condition(disc,i,sigma_s,dx,dy)
        yin=float(raw1[0,2]); uc=float(raw1[0,3]); vc=float(raw1[0,4])
        d0=distance_to_rectangle_boundary(0.,yin,uc,vc,W,H); p0=math.exp(-d0)
        xb=uc*d0; yb=yin+vc*d0; pb=perimeter_coordinate(xb,yb,W,H)
        surv0=math.exp(-kappa*d0); tr0=(1-surv0)/sigma_a if sigma_a>0 else d0/sigma_s
        for j,wproj in _project_one_output(disc,entry_face,pb,uc,vc,dxc,dyc,dx,dy):
            P[j,i]+=p0*wproj; R[j,i]+=p0*surv0*wproj
        if p0>=1-1e-14: track[i]=tr0; P[:,i]/=P[:,i].sum(); continue
        n=scattered_samples_per_state; Ltot=np.empty(n); outs=[]
        udraw=rng.random(n); sc=_conditional_first_collision_distance(udraw,d0)
        dirs=sample_isotropic_direction(rng,n)
        for m in range(n):
            xin=uc*sc[m]; ycin=yin+vc*sc[m]
            p,u,v,Lrem=random_flight_sample_from_state(W,H,float(xin),float(ycin),dirs[m],rng)
            outs.append((p,u,v)); Ltot[m]=sc[m]+Lrem
        surv,tr=_track_from_lopt(Ltot,sigma_s,sigma_a); ps=1-p0; track[i]=p0*tr0+ps*float(np.mean(tr))
        for m,(p,u,v) in enumerate(outs):
            for j,wproj in _project_one_output(disc,entry_face,p,u,v,dxc,dyc,dx,dy):
                P[j,i]+=ps*wproj/n; R[j,i]+=ps*float(surv[m])*wproj/n
        P[:,i]/=P[:,i].sum()
    return UltraCellResponse(sigma_s,sigma_a,dx,dy,R,P,track,scattered_samples_per_state,"mc-ballistic-split",0.,0.)

# -----------------------------------------------------------------------------
# Analytic streaming fallback (Sigma_s = 0 or deliberately excluded from CFM)
# -----------------------------------------------------------------------------

def _distance_local_boundary(x:float,y:float,u:float,v:float,dx:float,dy:float):
    cand=[]; eps=1e-14
    if u>eps: cand.append(((dx-x)/u,'right'))
    elif u<-eps: cand.append((-x/u,'left'))
    if v>eps: cand.append(((dy-y)/v,'top'))
    elif v<-eps: cand.append((-y/v,'bottom'))
    if not cand: raise RuntimeError('zero projected direction has no 2D boundary crossing')
    L,face=min(cand,key=lambda q:q[0]); return max(float(L),0.0),face


def build_boundary_response_streaming_ultra(
    sigma_a:float,dx:float,dy:float,disc:UltraPhaseSpace
) -> UltraCellResponse:
    """Exact no-scattering cell operator, useful for void/streaming materials."""
    ns=disc.n_state; P=np.zeros((ns,ns)); R=np.zeros_like(P); track=np.zeros(ns)
    for i in range(ns):
        face,pos,ang=disc.decode_state(i); x,y=disc.face_position(face,pos,dx,dy); u,v=disc.direction_for_entry(face,ang)
        L,exit_face=_distance_local_boundary(x,y,u,v,dx,dy)
        xo=x+L*u; yo=y+L*v; uo,vo=_force_outward(exit_face,u,v); neighbor=OPPOSITE[exit_face]
        terms=[]; s=0.0
        for ip,wp in disc.project_position(exit_face,xo,yo,dx,dy):
            for ia,wa in disc.project_angle(neighbor,uo,vo):
                ww=wp*wa; terms.append((disc.state_index(exit_face,ip,ia),ww)); s+=ww
        surv=math.exp(-sigma_a*L)
        track[i]=(1-surv)/sigma_a if sigma_a>0 else L
        for j,w in terms:
            w/=s; P[j,i]+=w; R[j,i]+=surv*w
    return UltraCellResponse(0.0,sigma_a,dx,dy,R,P,track,1,'analytic-streaming',0.0,0.0)

# -----------------------------------------------------------------------------
# Fast persistent response-library cache
# -----------------------------------------------------------------------------
def save_ultra_library(path: str, lib: Dict[Tuple[float,float],UltraCellResponse]) -> None:
    """Persist a material response library without compression overhead."""
    from pathlib import Path
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(lib.keys()); ops=[lib[k] for k in keys]
    np.savez(str(p),
        materials=np.asarray(keys,dtype=np.float64),
        R=np.stack([o.R for o in ops]), P=np.stack([o.P for o in ops]), track=np.stack([o.track for o in ops]),
        sigma_s=np.asarray([o.sigma_s for o in ops]), sigma_a=np.asarray([o.sigma_a for o in ops]),
        dx=np.asarray([o.dx for o in ops]), dy=np.asarray([o.dy for o in ops]),
        samples=np.asarray([o.samples_per_state for o in ops],dtype=np.int64),
        backend=np.asarray([o.backend for o in ops]), invalid=np.asarray([o.invalid_fraction for o in ops]),
        repaired=np.asarray([o.chord_repair_fraction for o in ops]))


def load_ultra_library(path: str) -> Dict[Tuple[float,float],UltraCellResponse]:
    d=np.load(path,allow_pickle=False); lib={}
    for i,k in enumerate(d['materials']):
        key=(float(k[0]),float(k[1]))
        lib[key]=UltraCellResponse(float(d['sigma_s'][i]),float(d['sigma_a'][i]),float(d['dx'][i]),float(d['dy'][i]),
            np.asarray(d['R'][i],dtype=np.float64),np.asarray(d['P'][i],dtype=np.float64),np.asarray(d['track'][i],dtype=np.float64),
            int(d['samples'][i]),str(d['backend'][i]),float(d['invalid'][i]),float(d['repaired'][i]))
    return lib


def save_ultra_internal(path: str, op: UltraInternalResponse) -> None:
    from pathlib import Path
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(p),outgoing=op.outgoing,track=np.asarray(op.track),samples=np.asarray(op.samples),
        backend=np.asarray(op.backend),invalid=np.asarray(op.invalid_fraction),repaired=np.asarray(op.chord_repair_fraction))


def load_ultra_internal(path: str) -> UltraInternalResponse:
    d=np.load(path,allow_pickle=False)
    return UltraInternalResponse(np.asarray(d['outgoing'],dtype=np.float64),float(d['track']),int(d['samples']),str(d['backend']),float(d['invalid']),float(d['repaired']))
