"""2D structured-grid MC/GMC transport drivers for stage-2 reproduction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple
import json
import os

import numpy as np
import torch

from .benchmarks2d import Structured2DProblem
from .checkpoints import load_cfm_checkpoint
from .mc_cell import sample_isotropic_direction, sample_left_boundary_direction, distance_to_rectangle_boundary
from .model import BoundaryVelocityNet, ModelConfig, sample_rk4, sample_rk4_cached
from .transforms import BoundaryTransform, InternalTransform


EntryFace = Literal["left", "right", "bottom", "top"]


@dataclass
class TransportResult:
    flux: np.ndarray
    leakage: int
    absorbed_weight: float
    track_length: float
    histories: int
    steps: int


def _distance_to_cell_boundary(xloc: float, yloc: float, u: float, v: float, dx: float, dy: float) -> Tuple[float, EntryFace]:
    eps = 1.0e-14
    candidates: list[Tuple[float, EntryFace]] = []
    if u > eps:
        candidates.append(((dx - xloc) / u, "right"))
    elif u < -eps:
        candidates.append((-xloc / u, "left"))
    if v > eps:
        candidates.append(((dy - yloc) / v, "top"))
    elif v < -eps:
        candidates.append((-yloc / v, "bottom"))
    if not candidates:
        return np.inf, "right"
    s, face = min(candidates, key=lambda z: z[0])
    return float(max(0.0, s)), face


def _nudge_after_face(x: float, y: float, face: EntryFace, eps: float = 1e-12) -> Tuple[float, float]:
    if face == "right":
        return x + eps, y
    if face == "left":
        return x - eps, y
    if face == "top":
        return x, y + eps
    return x, y - eps


def _project_direction_out_of_face(face: EntryFace, u: float, v: float, min_normal: float = 1.0e-7) -> Tuple[float, float]:
    """Project a decoded exit direction onto the physically admissible half space.

    The true MC exit direction must point out of the cell through the sampled
    exit face.  A smooth neural density can occasionally decode a direction that
    violates this hard geometric constraint, especially near corners or when the
    model is undertrained.  Farmer's description explicitly includes
    deterministic post-processing to enforce hard constraints after decoding; this
    projection is the minimal constraint needed by the structured-cell transport
    driver.
    """
    u = float(u); v = float(v)
    if face == "right":
        u = max(abs(u), min_normal)
    elif face == "left":
        u = -max(abs(u), min_normal)
    elif face == "top":
        v = max(abs(v), min_normal)
    elif face == "bottom":
        v = -max(abs(v), min_normal)
    r2 = u*u + v*v
    if r2 > 1.0:
        scale = (1.0 - 1.0e-8) / np.sqrt(r2)
        u *= scale; v *= scale
    return u, v


def _sample_is_invalid(p: float, u: float, v: float, Lopt: float) -> bool:
    return (not np.isfinite(p)) or (not np.isfinite(u)) or (not np.isfinite(v)) or (not np.isfinite(Lopt)) or Lopt <= 0.0 or p < -1e-6 or p > 1.0 + 1e-6


def _track_and_weight(w: float, sigma_a: float, L: float) -> Tuple[float, float, float]:
    """Return TL, absorbed weight, surviving weight after distance L."""
    if L <= 0.0:
        return 0.0, 0.0, w
    if sigma_a > 0.0:
        atten = float(np.exp(-sigma_a * L))
        wout = w * atten
        absorbed = w - wout
        tl = absorbed / sigma_a
    else:
        wout = w
        absorbed = 0.0
        tl = w * L
    return tl, absorbed, wout


def _initial_particle(problem: Structured2DProblem, rng: np.random.Generator, direction_mode: str) -> Tuple[float, float, float, float]:
    if problem.source_kind == "left_boundary":
        yrange = getattr(problem, "boundary_source_y_range", None)
        if yrange is None:
            ymin, ymax = 0.0, problem.height
        else:
            ymin, ymax = yrange
        y = float(rng.uniform(ymin, ymax))
        d = sample_left_boundary_direction(rng, 1, mode=direction_mode)[0]
        return 1.0e-12, y, float(d[0]), float(d[1])
    if problem.source_kind == "volume_box":
        if problem.source_box is None:
            raise ValueError("volume_box source requires problem.source_box")
        xmin, xmax, ymin, ymax = problem.source_box
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        d = sample_isotropic_direction(rng, 1)[0]
        return x, y, float(d[0]), float(d[1])
    raise ValueError(f"unknown source_kind={problem.source_kind}")


def run_standard_mc(
    problem: Structured2DProblem,
    n_particles: int,
    seed: int = 1,
    direction_mode: str = "cosine",
    weight_cutoff: float = 1.0e-12,
    max_steps_per_history: int = 200_000,
) -> TransportResult:
    """Implicit-capture, scattering-distance MC on a structured 2D grid."""
    rng = np.random.default_rng(seed)
    flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    leakage = 0
    absorbed_weight = 0.0
    total_tl = 0.0
    total_steps = 0
    w0 = 1.0 / float(n_particles)

    for _ in range(n_particles):
        x, y, u, v = _initial_particle(problem, rng, direction_mode)
        w = w0
        for _step in range(max_steps_per_history):
            total_steps += 1
            idx = problem.cell_indices(x, y)
            if idx is None or w < weight_cutoff:
                leakage += int(idx is None)
                break
            ix, iy = idx
            x0 = ix * problem.dx
            y0 = iy * problem.dy
            xloc = min(problem.dx, max(0.0, x - x0))
            yloc = min(problem.dy, max(0.0, y - y0))
            ss = float(problem.sigma_s[iy, ix])
            sa = float(problem.sigma_a[iy, ix])
            sc = float(rng.exponential(1.0 / ss)) if ss > 0.0 else np.inf
            sb, face = _distance_to_cell_boundary(xloc, yloc, u, v, problem.dx, problem.dy)
            L = min(sc, sb)
            tl, absorbed, w = _track_and_weight(w, sa, L)
            flux_tl[iy, ix] += tl
            absorbed_weight += absorbed
            total_tl += tl
            x += L * u
            y += L * v
            if sc < sb:
                d = sample_isotropic_direction(rng, 1)[0]
                u, v = float(d[0]), float(d[1])
            else:
                x, y = _nudge_after_face(x, y, face)
        else:
            # Treat nonterminated histories as leaked for diagnostics.
            leakage += 1
    flux = flux_tl / problem.volume
    return TransportResult(flux, leakage, absorbed_weight, total_tl, n_particles, total_steps)


def canonicalize_boundary_state(
    face: EntryFace,
    xloc: float,
    yloc: float,
    u: float,
    v: float,
    dx: float,
    dy: float,
    sigma_s: float,
) -> Tuple[np.ndarray, float, float]:
    """Map arbitrary cell entry face to the model's left-entry frame.

    Returns raw condition row [Wopt,Hopt,yin_opt,u_can,v_can] and the canonical
    physical dimensions (dx_can, dy_can) used by inverse mapping.
    """
    if face == "left":
        Wc, Hc = sigma_s * dx, sigma_s * dy
        yin = sigma_s * yloc
        uc, vc = u, v
        dxc, dyc = dx, dy
    elif face == "right":
        Wc, Hc = sigma_s * dx, sigma_s * dy
        yin = sigma_s * yloc
        uc, vc = -u, v
        dxc, dyc = dx, dy
    elif face == "bottom":
        Wc, Hc = sigma_s * dy, sigma_s * dx
        yin = sigma_s * xloc
        uc, vc = v, u
        dxc, dyc = dy, dx
    elif face == "top":
        Wc, Hc = sigma_s * dy, sigma_s * dx
        yin = sigma_s * xloc
        uc, vc = -v, u
        dxc, dyc = dy, dx
    else:
        raise ValueError(face)
    return np.array([[Wc, Hc, yin, uc, vc]], dtype=np.float32), dxc, dyc


def p_to_canonical_point(p: float, dxc: float, dyc: float) -> Tuple[float, float, EntryFace]:
    """Convert perimeter fraction to canonical physical point and exit face."""
    perim = 2.0 * (dxc + dyc)
    s = (p % 1.0) * perim
    if s < dxc:
        return s, 0.0, "bottom"
    s -= dxc
    if s < dyc:
        return dxc, s, "right"
    s -= dyc
    if s < dxc:
        return dxc - s, dyc, "top"
    s -= dxc
    return 0.0, dyc - s, "left"


def invert_canonical_exit(
    entry_face: EntryFace,
    xc: float,
    yc: float,
    uc: float,
    vc: float,
    dx: float,
    dy: float,
) -> Tuple[float, float, float, float, EntryFace]:
    """Map canonical physical exit point/direction back to local cell coordinates."""
    tol = 1e-10
    if entry_face == "left":
        xloc, yloc, u, v = xc, yc, uc, vc
    elif entry_face == "right":
        xloc, yloc, u, v = dx - xc, yc, -uc, vc
    elif entry_face == "bottom":
        xloc, yloc, u, v = yc, xc, vc, uc
    elif entry_face == "top":
        xloc, yloc, u, v = yc, dy - xc, vc, -uc
    else:
        raise ValueError(entry_face)

    # Determine physical exit face from local point.
    distances = {
        "left": abs(xloc),
        "right": abs(xloc - dx),
        "bottom": abs(yloc),
        "top": abs(yloc - dy),
    }
    face = min(distances, key=distances.get)  # type: ignore[arg-type]
    xloc = min(dx, max(0.0, xloc))
    yloc = min(dy, max(0.0, yloc))
    return xloc, yloc, float(u), float(v), face  # type: ignore[return-value]


class BoundaryGMCSampler:
    """Load and apply a trained stage-1 boundary CFM model."""
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", n_steps: int = 12):
        ckpt, self.checkpoint_metadata = load_cfm_checkpoint(checkpoint_path, "boundary", map_location=device)
        self.transform = BoundaryTransform.from_dict(ckpt["transform"])
        self.config = ModelConfig.from_dict(ckpt.get("model_config", {"c_dim": 5, "y_dim": 4}))
        self.model = BoundaryVelocityNet(self.config).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device
        self.n_steps = n_steps

    @torch.no_grad()
    def sample_one(self, raw_condition: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        c_np = self.transform.encode_conditions(raw_condition)
        c = torch.tensor(c_np, dtype=torch.float32, device=self.device)
        y_enc = sample_rk4(self.model, c, n_steps=self.n_steps, seed=seed)
        dec=self.transform.decode_targets_np(y_enc.cpu().numpy(), raw_condition)
        return _enforce_decoded_exit_constraints(raw_condition,dec)[0]


class InternalGMCSampler:
    """Load and apply a trained internal-source CFM model."""
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", n_steps: int = 12):
        ckpt, self.checkpoint_metadata = load_cfm_checkpoint(checkpoint_path, "internal", map_location=device)
        self.transform = InternalTransform.from_dict(ckpt["transform"])
        self.config = ModelConfig.from_dict(ckpt.get("model_config", {"c_dim": 6, "y_dim": 4}))
        self.model = BoundaryVelocityNet(self.config).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device
        self.n_steps = n_steps

    @torch.no_grad()
    def sample_one(self, raw_condition: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        c_np = self.transform.encode_conditions(raw_condition)
        c = torch.tensor(c_np, dtype=torch.float32, device=self.device)
        y_enc = sample_rk4(self.model, c, n_steps=self.n_steps, seed=seed)
        dec=self.transform.decode_targets_np(y_enc.cpu().numpy(), raw_condition)
        return _enforce_decoded_exit_constraints(raw_condition,dec)[0]


def _infer_entry_face_from_position_direction(xloc: float, yloc: float, u: float, v: float, dx: float, dy: float) -> EntryFace:
    # Prefer geometric boundary location; fall back to incoming direction.
    distances = [(abs(xloc), "left"), (abs(xloc - dx), "right"), (abs(yloc), "bottom"), (abs(yloc - dy), "top")]
    d, f = min(distances, key=lambda z: z[0])
    if d < 1e-8:
        return f  # type: ignore[return-value]
    # The entry face is opposite the direction into the cell.
    if abs(u) >= abs(v):
        return "left" if u > 0 else "right"
    return "bottom" if v > 0 else "top"


def run_gmc_boundary_transport(
    problem: Structured2DProblem,
    sampler: BoundaryGMCSampler,
    n_particles: int,
    seed: int = 2,
    direction_mode: str = "cosine",
    fallback: Literal["mc_cell", "stream"] = "mc_cell",
    optical_min: float = 1.0e-4,
    weight_cutoff: float = 1.0e-12,
    max_steps_per_history: int = 100_000,
) -> TransportResult:
    """Run a boundary-model GMC multi-cell transport simulation.

    Volumetric source histories use standard MC inside the birth cell until the
    first boundary crossing, then switch to GMC for boundary-to-boundary traversals.
    This is deliberately marked as an approximation until an internal-source GMC
    model is trained.
    """
    rng = np.random.default_rng(seed)
    flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    leakage = 0
    absorbed_weight = 0.0
    total_tl = 0.0
    total_steps = 0
    w0 = 1.0 / float(n_particles)

    for ip in range(n_particles):
        x, y, u, v = _initial_particle(problem, rng, direction_mode)
        w = w0
        # Volumetric birth is not a boundary condition.  Use MC until the first
        # cell crossing; after that every state is a boundary entry.
        force_mc_step = problem.source_kind == "volume_box"
        for _step in range(max_steps_per_history):
            total_steps += 1
            idx = problem.cell_indices(x, y)
            if idx is None or w < weight_cutoff:
                leakage += int(idx is None)
                break
            ix, iy = idx
            x0 = ix * problem.dx
            y0 = iy * problem.dy
            xloc = min(problem.dx, max(0.0, x - x0))
            yloc = min(problem.dy, max(0.0, y - y0))
            ss = float(problem.sigma_s[iy, ix])
            sa = float(problem.sigma_a[iy, ix])

            # Fall back to direct cell MC if there is no scattering, the state is outside the stable optical range,
            # or for the first step from an internal volumetric source.
            if force_mc_step or ss <= 0.0 or ss * problem.dx < optical_min or ss * problem.dy < optical_min:
                sc = float(rng.exponential(1.0 / ss)) if ss > 0.0 else np.inf
                sb, face = _distance_to_cell_boundary(xloc, yloc, u, v, problem.dx, problem.dy)
                L = min(sc, sb)
                tl, absorbed, w = _track_and_weight(w, sa, L)
                flux_tl[iy, ix] += tl
                absorbed_weight += absorbed
                total_tl += tl
                x += L * u
                y += L * v
                if sc < sb:
                    d = sample_isotropic_direction(rng, 1)[0]
                    u, v = float(d[0]), float(d[1])
                else:
                    x, y = _nudge_after_face(x, y, face)
                    force_mc_step = False
                continue

            entry_face = _infer_entry_face_from_position_direction(xloc, yloc, u, v, problem.dx, problem.dy)
            raw_c, dxc, dyc = canonicalize_boundary_state(entry_face, xloc, yloc, u, v, problem.dx, problem.dy, ss)
            # Use torch randomness by not fixing seed; global run remains stochastic.
            p, uc, vc, Lopt = sampler.sample_one(raw_c)
            L = float(max(0.0, Lopt / ss))
            tl, absorbed, w = _track_and_weight(w, sa, L)
            flux_tl[iy, ix] += tl
            absorbed_weight += absorbed
            total_tl += tl
            xc, yc, canon_exit_face = p_to_canonical_point(float(p), dxc, dyc)
            uc, vc = _project_direction_out_of_face(canon_exit_face, float(uc), float(vc))
            xloc2, yloc2, u, v, face = invert_canonical_exit(entry_face, xc, yc, uc, vc, problem.dx, problem.dy)
            x = x0 + xloc2
            y = y0 + yloc2
            x, y = _nudge_after_face(x, y, face)
        else:
            leakage += 1
    flux = flux_tl / problem.volume
    return TransportResult(flux, leakage, absorbed_weight, total_tl, n_particles, total_steps)




def run_gmc_full_transport(
    problem: Structured2DProblem,
    boundary_sampler: BoundaryGMCSampler,
    internal_sampler: Optional[InternalGMCSampler],
    n_particles: int,
    seed: int = 3,
    direction_mode: str = "cosine",
    optical_min: float = 1.0e-4,
    weight_cutoff: float = 1.0e-12,
    max_steps_per_history: int = 100_000,
) -> TransportResult:
    """Run a GMC multi-cell simulation with optional internal-source model.

    If ``internal_sampler`` is provided, volumetric source births are advanced by
    the internal GMC model in their birth cell.  Otherwise the first birth-cell
    step falls back to direct cell MC, matching the stage-2 bridge implementation. All
    subsequent boundary-to-boundary traversals use the boundary GMC model when
    the cell optical dimensions are within a stable range.
    """
    rng = np.random.default_rng(seed)
    flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    leakage = 0
    absorbed_weight = 0.0
    total_tl = 0.0
    total_steps = 0
    w0 = 1.0 / float(n_particles)

    for _ip in range(n_particles):
        x, y, u, v = _initial_particle(problem, rng, direction_mode)
        w = w0
        internal_available_for_birth = (problem.source_kind == "volume_box" and internal_sampler is not None)
        force_mc_step = (problem.source_kind == "volume_box" and internal_sampler is None)

        for _step in range(max_steps_per_history):
            total_steps += 1
            idx = problem.cell_indices(x, y)
            if idx is None or w < weight_cutoff:
                leakage += int(idx is None)
                break
            ix, iy = idx
            x0 = ix * problem.dx
            y0 = iy * problem.dy
            xloc = min(problem.dx, max(0.0, x - x0))
            yloc = min(problem.dy, max(0.0, y - y0))
            ss = float(problem.sigma_s[iy, ix])
            sa = float(problem.sigma_a[iy, ix])

            can_use_gmc = ss > 0.0 and ss * problem.dx >= optical_min and ss * problem.dy >= optical_min

            if internal_available_for_birth and can_use_gmc:
                raw_c = np.array([[ss * problem.dx, ss * problem.dy, ss * xloc, ss * yloc, u, v]], dtype=np.float32)
                p, ug, vg, Lopt = internal_sampler.sample_one(raw_c)
                L = float(max(0.0, Lopt / ss))
                tl, absorbed, w = _track_and_weight(w, sa, L)
                flux_tl[iy, ix] += tl
                absorbed_weight += absorbed
                total_tl += tl
                xc, yc, exit_face_can = p_to_canonical_point(float(p), problem.dx, problem.dy)
                # Internal model has no entry canonicalization; canonical cell == physical local cell.
                ug, vg = _project_direction_out_of_face(exit_face_can, float(ug), float(vg))
                xloc2, yloc2, u, v = xc, yc, ug, vg
                distances = {
                    "left": abs(xloc2),
                    "right": abs(xloc2 - problem.dx),
                    "bottom": abs(yloc2),
                    "top": abs(yloc2 - problem.dy),
                }
                face = min(distances, key=distances.get)  # type: ignore[arg-type]
                x = x0 + min(problem.dx, max(0.0, xloc2))
                y = y0 + min(problem.dy, max(0.0, yloc2))
                x, y = _nudge_after_face(x, y, face)  # type: ignore[arg-type]
                internal_available_for_birth = False
                continue

            if force_mc_step or not can_use_gmc:
                sc = float(rng.exponential(1.0 / ss)) if ss > 0.0 else np.inf
                sb, face = _distance_to_cell_boundary(xloc, yloc, u, v, problem.dx, problem.dy)
                L = min(sc, sb)
                tl, absorbed, w = _track_and_weight(w, sa, L)
                flux_tl[iy, ix] += tl
                absorbed_weight += absorbed
                total_tl += tl
                x += L * u
                y += L * v
                if sc < sb:
                    d = sample_isotropic_direction(rng, 1)[0]
                    u, v = float(d[0]), float(d[1])
                else:
                    x, y = _nudge_after_face(x, y, face)
                    force_mc_step = False
                internal_available_for_birth = False
                continue

            entry_face = _infer_entry_face_from_position_direction(xloc, yloc, u, v, problem.dx, problem.dy)
            raw_c, dxc, dyc = canonicalize_boundary_state(entry_face, xloc, yloc, u, v, problem.dx, problem.dy, ss)
            p, uc, vc, Lopt = boundary_sampler.sample_one(raw_c)
            L = float(max(0.0, Lopt / ss))
            tl, absorbed, w = _track_and_weight(w, sa, L)
            flux_tl[iy, ix] += tl
            absorbed_weight += absorbed
            total_tl += tl
            xc, yc, canon_exit_face = p_to_canonical_point(float(p), dxc, dyc)
            uc, vc = _project_direction_out_of_face(canon_exit_face, float(uc), float(vc))
            xloc2, yloc2, u, v, face = invert_canonical_exit(entry_face, xc, yc, uc, vc, problem.dx, problem.dy)
            x = x0 + xloc2
            y = y0 + yloc2
            x, y = _nudge_after_face(x, y, face)
        else:
            leakage += 1
    flux = flux_tl / problem.volume
    return TransportResult(flux, leakage, absorbed_weight, total_tl, n_particles, total_steps)


def save_result_npz(path: str | Path, result: TransportResult) -> None:
    np.savez_compressed(
        path,
        flux=result.flux,
        leakage=result.leakage,
        absorbed_weight=result.absorbed_weight,
        track_length=result.track_length,
        histories=result.histories,
        steps=result.steps,
    )

# -----------------------------------------------------------------------------
# Stage-4 batched inference utilities.
# These are intentionally appended rather than replacing the simple scalar drivers
# above, so the earlier educational implementation remains readable.
# -----------------------------------------------------------------------------

def _enforce_decoded_exit_constraints(raw_conditions: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    """Apply the hard geometric constraints described for GMC decoding.

    ``p_exit`` already parameterizes the rectangle boundary exactly.  The remaining
    hard constraint is that the projected outgoing direction must lie in the outward
    half-space of the decoded face.  Farmer explicitly separates this deterministic
    physical post-processing from the learned smooth density.  Applying it here keeps
    single-cell validation and full transport on the same physical sampler.
    """
    out=np.asarray(decoded,dtype=np.float32).copy()
    for i in range(len(out)):
        W=float(raw_conditions[i,0]); H=float(raw_conditions[i,1]); p=float(out[i,0])
        _x,_y,face=p_to_canonical_point(p,W,H)
        u,v=_project_direction_out_of_face(face,float(out[i,1]),float(out[i,2]))
        out[i,1]=u; out[i,2]=v
    return out


def _sampler_sample_batch(self, raw_conditions: np.ndarray, seed: Optional[int] = None, batch_size: int = 8192, enforce_hard_constraints: bool = True) -> np.ndarray:
    """Vectorized version of ``sample_one`` for many independent cell traversals."""
    if raw_conditions.shape[0] == 0:
        return np.zeros((0, self.config.y_dim), dtype=np.float32)
    outputs = []
    base_seed = seed
    for start in range(0, raw_conditions.shape[0], batch_size):
        stop = min(raw_conditions.shape[0], start + batch_size)
        c_np = self.transform.encode_conditions(raw_conditions[start:stop])
        c = torch.tensor(c_np, dtype=torch.float32, device=self.device)
        # Use different deterministic seeds for chunks only when a seed is provided.
        chunk_seed = None if base_seed is None else int(base_seed + start)
        y_enc = sample_rk4_cached(self.model, c, n_steps=self.n_steps, seed=chunk_seed)
        dec=self.transform.decode_targets_np(y_enc.cpu().numpy(), raw_conditions[start:stop])
        if enforce_hard_constraints:
            dec=_enforce_decoded_exit_constraints(raw_conditions[start:stop],dec)
        outputs.append(dec)
    return np.concatenate(outputs, axis=0)


BoundaryGMCSampler.sample_batch = _sampler_sample_batch  # type: ignore[attr-defined]
InternalGMCSampler.sample_batch = _sampler_sample_batch  # type: ignore[attr-defined]


def run_gmc_full_transport_batched(
    problem: Structured2DProblem,
    boundary_sampler: BoundaryGMCSampler,
    internal_sampler: Optional[InternalGMCSampler],
    n_particles: int,
    seed: int = 3,
    direction_mode: str = "cosine",
    optical_min: float = 1.0e-4,
    weight_cutoff: float = 1.0e-12,
    max_steps_per_history: int = 100_000,
    inference_batch_size: int = 8192,
) -> TransportResult:
    """Batched GMC structured-grid transport.

    The scalar ``run_gmc_full_transport`` function is useful for understanding the
    algorithm, but it performs one neural ODE solve per cell crossing.  This batched
    version groups all currently active cell traversals into boundary/internal calls,
    which is much closer to the execution model described in the dissertation and is
    required for practical cloud-map comparisons on CPU.
    """
    rng = np.random.default_rng(seed)
    flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    leakage = 0
    absorbed_weight = 0.0
    total_tl = 0.0
    total_steps = 0
    w0 = 1.0 / float(n_particles)

    x = np.empty(n_particles, dtype=np.float64)
    y = np.empty(n_particles, dtype=np.float64)
    u = np.empty(n_particles, dtype=np.float64)
    v = np.empty(n_particles, dtype=np.float64)
    w = np.full(n_particles, w0, dtype=np.float64)
    alive = np.ones(n_particles, dtype=bool)
    step_counts = np.zeros(n_particles, dtype=np.int32)
    internal_available = np.zeros(n_particles, dtype=bool)

    for i in range(n_particles):
        xi, yi, ui, vi = _initial_particle(problem, rng, direction_mode)
        x[i], y[i], u[i], v[i] = xi, yi, ui, vi
        internal_available[i] = (problem.source_kind == "volume_box" and internal_sampler is not None)

    _debug = os.environ.get("GMC_DEBUG", "0") == "1"
    _loop_iter = 0
    while bool(np.any(alive)):
        _loop_iter += 1
        if _debug and (_loop_iter <= 5 or _loop_iter % 25 == 0):
            print(f"[gmc-batch] iter={_loop_iter} alive={int(np.sum(alive))} max_step={int(step_counts.max())} total_steps={int(total_steps)}", flush=True)
        active = np.flatnonzero(alive)
        boundary_rows = []
        boundary_meta = []
        internal_rows = []
        internal_meta = []
        exact_ids = []

        for pid in active:
            idx = problem.cell_indices(float(x[pid]), float(y[pid]))
            if idx is None or w[pid] < weight_cutoff or step_counts[pid] >= max_steps_per_history:
                if idx is None or step_counts[pid] >= max_steps_per_history:
                    leakage += 1
                alive[pid] = False
                continue
            step_counts[pid] += 1
            total_steps += 1
            ix, iy = idx
            x0 = ix * problem.dx
            y0 = iy * problem.dy
            xloc = min(problem.dx, max(0.0, float(x[pid] - x0)))
            yloc = min(problem.dy, max(0.0, float(y[pid] - y0)))
            ss = float(problem.sigma_s[iy, ix])
            sa = float(problem.sigma_a[iy, ix])
            can_use_gmc = ss > 0.0 and ss * problem.dx >= optical_min and ss * problem.dy >= optical_min

            if internal_available[pid] and can_use_gmc:
                internal_rows.append([ss * problem.dx, ss * problem.dy, ss * xloc, ss * yloc, u[pid], v[pid]])
                internal_meta.append((pid, ix, iy, x0, y0, ss, sa))
            elif can_use_gmc:
                entry_face = _infer_entry_face_from_position_direction(xloc, yloc, float(u[pid]), float(v[pid]), problem.dx, problem.dy)
                raw_c, dxc, dyc = canonicalize_boundary_state(entry_face, xloc, yloc, float(u[pid]), float(v[pid]), problem.dx, problem.dy, ss)
                boundary_rows.append(raw_c[0])
                boundary_meta.append((pid, ix, iy, x0, y0, ss, sa, entry_face, dxc, dyc))
            else:
                exact_ids.append((pid, ix, iy, x0, y0, xloc, yloc, ss, sa))

        # Rare fallback path: direct cell MC for non-scattering or outside the training regime.
        for pid, ix, iy, x0, y0, xloc, yloc, ss, sa in exact_ids:
            sc = float(rng.exponential(1.0 / ss)) if ss > 0.0 else np.inf
            sb, face = _distance_to_cell_boundary(xloc, yloc, float(u[pid]), float(v[pid]), problem.dx, problem.dy)
            L = min(sc, sb)
            tl, absorbed, wout = _track_and_weight(float(w[pid]), sa, L)
            flux_tl[iy, ix] += tl
            absorbed_weight += absorbed
            total_tl += tl
            w[pid] = wout
            x[pid] += L * u[pid]
            y[pid] += L * v[pid]
            internal_available[pid] = False
            if sc < sb:
                d = sample_isotropic_direction(rng, 1)[0]
                u[pid], v[pid] = float(d[0]), float(d[1])
            else:
                x[pid], y[pid] = _nudge_after_face(float(x[pid]), float(y[pid]), face)

        if internal_rows:
            raw = np.asarray(internal_rows, dtype=np.float32)
            out = internal_sampler.sample_batch(raw, batch_size=inference_batch_size)  # type: ignore[union-attr]
            add_i = []
            add_j = []
            add_tl = []
            for k, (pid, ix, iy, x0, y0, ss, sa) in enumerate(internal_meta):
                p, ug, vg, Lopt = map(float, out[k])
                L = max(0.0, Lopt / ss)
                tl, absorbed, wout = _track_and_weight(float(w[pid]), sa, L)
                add_i.append(iy); add_j.append(ix); add_tl.append(tl)
                absorbed_weight += absorbed
                total_tl += tl
                w[pid] = wout
                xc, yc, exit_face_can = p_to_canonical_point(p, problem.dx, problem.dy)
                ug, vg = _project_direction_out_of_face(exit_face_can, ug, vg)
                xloc2, yloc2 = xc, yc
                u[pid], v[pid] = ug, vg
                distances = {
                    "left": abs(xloc2),
                    "right": abs(xloc2 - problem.dx),
                    "bottom": abs(yloc2),
                    "top": abs(yloc2 - problem.dy),
                }
                face = min(distances, key=distances.get)  # type: ignore[arg-type]
                x[pid] = x0 + min(problem.dx, max(0.0, xloc2))
                y[pid] = y0 + min(problem.dy, max(0.0, yloc2))
                x[pid], y[pid] = _nudge_after_face(float(x[pid]), float(y[pid]), face)  # type: ignore[arg-type]
                internal_available[pid] = False
            np.add.at(flux_tl, (np.asarray(add_i), np.asarray(add_j)), np.asarray(add_tl))

        if boundary_rows:
            raw = np.asarray(boundary_rows, dtype=np.float32)
            out = boundary_sampler.sample_batch(raw, batch_size=inference_batch_size)
            add_i = []
            add_j = []
            add_tl = []
            for k, (pid, ix, iy, x0, y0, ss, sa, entry_face, dxc, dyc) in enumerate(boundary_meta):
                p, uc, vc, Lopt = map(float, out[k])
                L = max(0.0, Lopt / ss)
                tl, absorbed, wout = _track_and_weight(float(w[pid]), sa, L)
                add_i.append(iy); add_j.append(ix); add_tl.append(tl)
                absorbed_weight += absorbed
                total_tl += tl
                w[pid] = wout
                xc, yc, canon_exit_face = p_to_canonical_point(p, dxc, dyc)
                uc, vc = _project_direction_out_of_face(canon_exit_face, uc, vc)
                xloc2, yloc2, ui, vi, face = invert_canonical_exit(entry_face, xc, yc, uc, vc, problem.dx, problem.dy)
                x[pid] = x0 + xloc2
                y[pid] = y0 + yloc2
                u[pid], v[pid] = ui, vi
                x[pid], y[pid] = _nudge_after_face(float(x[pid]), float(y[pid]), face)
            np.add.at(flux_tl, (np.asarray(add_i), np.asarray(add_j)), np.asarray(add_tl))

    flux = flux_tl / problem.volume
    return TransportResult(flux, leakage, absorbed_weight, total_tl, n_particles, int(total_steps))

# -----------------------------------------------------------------------------
# Stage-6 guarded hybrid GMC/MC driver.
# -----------------------------------------------------------------------------
@dataclass
class GuardedTransportResult(TransportResult):
    gmc_accepted: int = 0
    mc_fallback: int = 0
    invalid_gmc: int = 0
    watchdog_fallback: int = 0


def _boundary_chord_lower_bound(entry_face: EntryFace, xloc: float, yloc: float, p: float, dxc: float, dyc: float, dx: float, dy: float) -> float:
    """Physical lower bound on path length: projected displacement from entry to exit."""
    xc, yc, _ = p_to_canonical_point(float(p), dxc, dyc)
    if entry_face == "left":
        x2, y2 = xc, yc
    elif entry_face == "right":
        x2, y2 = dx - xc, yc
    elif entry_face == "bottom":
        x2, y2 = yc, xc
    elif entry_face == "top":
        x2, y2 = yc, dy - xc
    else:
        x2, y2 = xloc, yloc
    return float(np.hypot(x2 - xloc, y2 - yloc))


def _internal_chord_lower_bound(xloc: float, yloc: float, p: float, dx: float, dy: float) -> float:
    x2, y2, _ = p_to_canonical_point(float(p), dx, dy)
    return float(np.hypot(x2 - xloc, y2 - yloc))


def _exact_mc_one_cell_step(
    rng: np.random.Generator,
    x: float,
    y: float,
    u: float,
    v: float,
    w: float,
    ix: int,
    iy: int,
    x0: float,
    y0: float,
    xloc: float,
    yloc: float,
    ss: float,
    sa: float,
    problem: Structured2DProblem,
) -> Tuple[float, float, float, float, float, float, float, bool]:
    """Direct random-flight MC step within one grid cell."""
    sc = float(rng.exponential(1.0 / ss)) if ss > 0.0 else np.inf
    sb, face = _distance_to_cell_boundary(xloc, yloc, u, v, problem.dx, problem.dy)
    L = min(sc, sb)
    tl, absorbed, wout = _track_and_weight(w, sa, L)
    xnew = x + L * u
    ynew = y + L * v
    unew, vnew = u, v
    crossed = not (sc < sb)
    if sc < sb:
        d = sample_isotropic_direction(rng, 1)[0]
        unew, vnew = float(d[0]), float(d[1])
    else:
        xnew, ynew = _nudge_after_face(float(xnew), float(ynew), face, eps=max(problem.dx, problem.dy) * 1.0e-10)
    return xnew, ynew, unew, vnew, wout, tl, absorbed, crossed


def run_gmc_guarded_hybrid_transport(
    problem: Structured2DProblem,
    boundary_sampler: BoundaryGMCSampler,
    internal_sampler: Optional[InternalGMCSampler],
    n_particles: int,
    seed: int = 7,
    direction_mode: str = "cosine",
    optical_gmc_min: float = 0.15,
    optical_gmc_max: float = 5.0,
    weight_cutoff: float = 1.0e-12,
    max_steps_per_history: int = 200_000,
    watchdog_gmc_after: int = 500,
    inference_batch_size: int = 4096,
    chord_tolerance: float = 1.0e-8,
    L_cap_factor: float = 200.0,
) -> GuardedTransportResult:
    """Quality-first guarded hybrid GMC/MC transport.

    This implements a conservative version of the paper's hybrid idea: use the
    learned response only where the optical cell dimensions are inside a safe
    training regime and the decoded sample passes physical checks.  Otherwise,
    take the exact one-cell MC step.  This prevents rare malformed neural samples
    from creating long traversal tails or empty/biased cloud maps.
    """
    rng = np.random.default_rng(seed)
    flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    leakage = 0
    absorbed_weight = 0.0
    total_tl = 0.0
    total_steps = 0
    gmc_accepted = 0
    mc_fallback = 0
    invalid_gmc = 0
    watchdog_fallback = 0
    w0 = 1.0 / float(n_particles)

    x = np.empty(n_particles, dtype=np.float64)
    y = np.empty(n_particles, dtype=np.float64)
    u = np.empty(n_particles, dtype=np.float64)
    v = np.empty(n_particles, dtype=np.float64)
    w = np.full(n_particles, w0, dtype=np.float64)
    alive = np.ones(n_particles, dtype=bool)
    step_counts = np.zeros(n_particles, dtype=np.int32)
    internal_available = np.zeros(n_particles, dtype=bool)

    for i in range(n_particles):
        xi, yi, ui, vi = _initial_particle(problem, rng, direction_mode)
        x[i], y[i], u[i], v[i] = xi, yi, ui, vi
        internal_available[i] = (problem.source_kind == "volume_box" and internal_sampler is not None)

    while bool(np.any(alive)):
        active = np.flatnonzero(alive)
        boundary_rows: list[list[float]] = []
        boundary_meta = []
        internal_rows: list[list[float]] = []
        internal_meta = []
        exact_meta = []

        for pid in active:
            idx = problem.cell_indices(float(x[pid]), float(y[pid]))
            if idx is None or w[pid] < weight_cutoff or step_counts[pid] >= max_steps_per_history:
                if idx is None or step_counts[pid] >= max_steps_per_history:
                    leakage += 1
                alive[pid] = False
                continue
            step_counts[pid] += 1
            total_steps += 1
            ix, iy = idx
            x0 = ix * problem.dx
            y0 = iy * problem.dy
            xloc = min(problem.dx, max(0.0, float(x[pid] - x0)))
            yloc = min(problem.dy, max(0.0, float(y[pid] - y0)))
            ss = float(problem.sigma_s[iy, ix])
            sa = float(problem.sigma_a[iy, ix])
            Wopt = ss * problem.dx
            Hopt = ss * problem.dy
            opt_safe = (ss > 0.0 and Wopt >= optical_gmc_min and Hopt >= optical_gmc_min and Wopt <= optical_gmc_max and Hopt <= optical_gmc_max)
            # After many cell crossings, stop asking the learned model and force direct cell MC.
            if step_counts[pid] > watchdog_gmc_after:
                opt_safe = False
                watchdog_fallback += 1
            if internal_available[pid] and internal_sampler is not None and opt_safe:
                internal_rows.append([Wopt, Hopt, ss * xloc, ss * yloc, float(u[pid]), float(v[pid])])
                internal_meta.append((pid, ix, iy, x0, y0, xloc, yloc, ss, sa))
            elif opt_safe:
                entry_face = _infer_entry_face_from_position_direction(xloc, yloc, float(u[pid]), float(v[pid]), problem.dx, problem.dy)
                raw_c, dxc, dyc = canonicalize_boundary_state(entry_face, xloc, yloc, float(u[pid]), float(v[pid]), problem.dx, problem.dy, ss)
                boundary_rows.append(raw_c[0].tolist())
                boundary_meta.append((pid, ix, iy, x0, y0, xloc, yloc, ss, sa, entry_face, dxc, dyc))
            else:
                exact_meta.append((pid, ix, iy, x0, y0, xloc, yloc, ss, sa))

        # Direct cell-MC fallback steps.
        for pid, ix, iy, x0, y0, xloc, yloc, ss, sa in exact_meta:
            mc_fallback += 1
            xnew, ynew, unew, vnew, wout, tl, absorbed, crossed = _exact_mc_one_cell_step(
                rng, float(x[pid]), float(y[pid]), float(u[pid]), float(v[pid]), float(w[pid]),
                ix, iy, x0, y0, xloc, yloc, ss, sa, problem
            )
            flux_tl[iy, ix] += tl
            absorbed_weight += absorbed
            total_tl += tl
            x[pid], y[pid], u[pid], v[pid], w[pid] = xnew, ynew, unew, vnew, wout
            internal_available[pid] = False

        if internal_rows:
            raw = np.asarray(internal_rows, dtype=np.float32)
            out = internal_sampler.sample_batch(raw, batch_size=inference_batch_size)  # type: ignore[union-attr]
            for k, (pid, ix, iy, x0, y0, xloc, yloc, ss, sa) in enumerate(internal_meta):
                p, ug, vg, Lopt = map(float, out[k])
                L = Lopt / ss if ss > 0.0 else np.inf
                chord = _internal_chord_lower_bound(xloc, yloc, p, problem.dx, problem.dy)
                bad = _sample_is_invalid(p, ug, vg, Lopt) or (L + chord_tolerance < chord) or (L > L_cap_factor * max(problem.dx, problem.dy))
                if bad:
                    invalid_gmc += 1
                    mc_fallback += 1
                    xnew, ynew, unew, vnew, wout, tl, absorbed, crossed = _exact_mc_one_cell_step(
                        rng, float(x[pid]), float(y[pid]), float(u[pid]), float(v[pid]), float(w[pid]),
                        ix, iy, x0, y0, xloc, yloc, ss, sa, problem
                    )
                    flux_tl[iy, ix] += tl
                    absorbed_weight += absorbed
                    total_tl += tl
                    x[pid], y[pid], u[pid], v[pid], w[pid] = xnew, ynew, unew, vnew, wout
                    internal_available[pid] = False
                    continue
                gmc_accepted += 1
                tl, absorbed, wout = _track_and_weight(float(w[pid]), sa, L)
                flux_tl[iy, ix] += tl
                absorbed_weight += absorbed
                total_tl += tl
                w[pid] = wout
                xc, yc, exit_face_can = p_to_canonical_point(p, problem.dx, problem.dy)
                ug, vg = _project_direction_out_of_face(exit_face_can, ug, vg)
                distances = {"left": abs(xc), "right": abs(xc - problem.dx), "bottom": abs(yc), "top": abs(yc - problem.dy)}
                face = min(distances, key=distances.get)  # type: ignore[arg-type]
                x[pid] = x0 + min(problem.dx, max(0.0, xc))
                y[pid] = y0 + min(problem.dy, max(0.0, yc))
                u[pid], v[pid] = ug, vg
                x[pid], y[pid] = _nudge_after_face(float(x[pid]), float(y[pid]), face, eps=max(problem.dx, problem.dy) * 1.0e-10)  # type: ignore[arg-type]
                internal_available[pid] = False

        if boundary_rows:
            raw = np.asarray(boundary_rows, dtype=np.float32)
            out = boundary_sampler.sample_batch(raw, batch_size=inference_batch_size)
            for k, (pid, ix, iy, x0, y0, xloc, yloc, ss, sa, entry_face, dxc, dyc) in enumerate(boundary_meta):
                p, uc, vc, Lopt = map(float, out[k])
                L = Lopt / ss if ss > 0.0 else np.inf
                chord = _boundary_chord_lower_bound(entry_face, xloc, yloc, p, dxc, dyc, problem.dx, problem.dy)
                bad = _sample_is_invalid(p, uc, vc, Lopt) or (L + chord_tolerance < chord) or (L > L_cap_factor * max(problem.dx, problem.dy))
                if bad:
                    invalid_gmc += 1
                    mc_fallback += 1
                    xnew, ynew, unew, vnew, wout, tl, absorbed, crossed = _exact_mc_one_cell_step(
                        rng, float(x[pid]), float(y[pid]), float(u[pid]), float(v[pid]), float(w[pid]),
                        ix, iy, x0, y0, xloc, yloc, ss, sa, problem
                    )
                    flux_tl[iy, ix] += tl
                    absorbed_weight += absorbed
                    total_tl += tl
                    x[pid], y[pid], u[pid], v[pid], w[pid] = xnew, ynew, unew, vnew, wout
                    internal_available[pid] = False
                    continue
                gmc_accepted += 1
                tl, absorbed, wout = _track_and_weight(float(w[pid]), sa, L)
                flux_tl[iy, ix] += tl
                absorbed_weight += absorbed
                total_tl += tl
                w[pid] = wout
                xc, yc, canon_exit_face = p_to_canonical_point(p, dxc, dyc)
                uc, vc = _project_direction_out_of_face(canon_exit_face, uc, vc)
                xloc2, yloc2, ui, vi, face = invert_canonical_exit(entry_face, xc, yc, uc, vc, problem.dx, problem.dy)
                x[pid] = x0 + xloc2
                y[pid] = y0 + yloc2
                u[pid], v[pid] = ui, vi
                x[pid], y[pid] = _nudge_after_face(float(x[pid]), float(y[pid]), face, eps=max(problem.dx, problem.dy) * 1.0e-10)
                internal_available[pid] = False

    flux = flux_tl / problem.volume
    return GuardedTransportResult(
        flux=flux,
        leakage=leakage,
        absorbed_weight=absorbed_weight,
        track_length=total_tl,
        histories=n_particles,
        steps=int(total_steps),
        gmc_accepted=int(gmc_accepted),
        mc_fallback=int(mc_fallback),
        invalid_gmc=int(invalid_gmc),
        watchdog_fallback=int(watchdog_fallback),
    )


def save_guarded_result_npz(path: str | Path, result: GuardedTransportResult) -> None:
    np.savez_compressed(
        path,
        flux=result.flux,
        leakage=result.leakage,
        absorbed_weight=result.absorbed_weight,
        track_length=result.track_length,
        histories=result.histories,
        steps=result.steps,
        gmc_accepted=result.gmc_accepted,
        mc_fallback=result.mc_fallback,
        invalid_gmc=result.invalid_gmc,
        watchdog_fallback=result.watchdog_fallback,
    )

# -----------------------------------------------------------------------------
# Stage-6 direct cell-response MC driver (historical API name retained).
# -----------------------------------------------------------------------------
def run_oracle_cell_response_transport(
    problem: Structured2DProblem,
    n_particles: int,
    seed: int = 11,
    direction_mode: str = "cosine",
    weight_cutoff: float = 1.0e-12,
    max_steps_per_history: int = 200_000,
) -> TransportResult:
    """Global cell-response MC using direct random-flight local responses.

    The historical function name contains ``oracle`` for compatibility.  This is
    not an exact continuum truth: it samples full source-to-termination histories
    and therefore retains finite-history uncertainty and cutoff effects.  It does
    avoid neural-surrogate error by drawing each pure-scattering cell response
    directly in continuous optical coordinates, then applying analytic absorption.
    """
    from .mc_cell import random_flight_boundary_sample, random_flight_sample_from_state

    rng = np.random.default_rng(seed)
    flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    leakage = 0
    absorbed_weight = 0.0
    total_tl = 0.0
    total_steps = 0
    w0 = 1.0 / float(n_particles)

    for _ip in range(n_particles):
        x, y, u, v = _initial_particle(problem, rng, direction_mode)
        w = w0
        internal_available = problem.source_kind == "volume_box"
        for _step in range(max_steps_per_history):
            total_steps += 1
            idx = problem.cell_indices(x, y)
            if idx is None or w < weight_cutoff:
                leakage += int(idx is None)
                break
            ix, iy = idx
            x0 = ix * problem.dx
            y0 = iy * problem.dy
            xloc = min(problem.dx, max(0.0, x - x0))
            yloc = min(problem.dy, max(0.0, y - y0))
            ss = float(problem.sigma_s[iy, ix])
            sa = float(problem.sigma_a[iy, ix])
            if ss <= 0.0:
                # Pure streaming cell: one deterministic boundary crossing.
                sb, face = _distance_to_cell_boundary(xloc, yloc, u, v, problem.dx, problem.dy)
                L = sb
                tl, absorbed, w = _track_and_weight(w, sa, L)
                flux_tl[iy, ix] += tl
                absorbed_weight += absorbed
                total_tl += tl
                x += L * u
                y += L * v
                x, y = _nudge_after_face(x, y, face, eps=max(problem.dx, problem.dy) * 1e-10)
                internal_available = False
                continue
            if internal_available:
                Wopt, Hopt = ss * problem.dx, ss * problem.dy
                p, ug, vg, Lopt = random_flight_sample_from_state(
                    Wopt, Hopt, ss * xloc, ss * yloc, np.array([u, v, 0.0]), rng
                )
                L = Lopt / ss
                tl, absorbed, w = _track_and_weight(w, sa, L)
                flux_tl[iy, ix] += tl
                absorbed_weight += absorbed
                total_tl += tl
                xc, yc, face = p_to_canonical_point(p, problem.dx, problem.dy)
                ug, vg = _project_direction_out_of_face(face, ug, vg)
                x = x0 + xc
                y = y0 + yc
                u, v = ug, vg
                x, y = _nudge_after_face(x, y, face, eps=max(problem.dx, problem.dy) * 1e-10)
                internal_available = False
                continue

            entry_face = _infer_entry_face_from_position_direction(xloc, yloc, u, v, problem.dx, problem.dy)
            raw_c, dxc, dyc = canonicalize_boundary_state(entry_face, xloc, yloc, u, v, problem.dx, problem.dy, ss)
            Wc, Hc, yin, uc, vc = map(float, raw_c[0])
            p, uco, vco, Lopt = random_flight_boundary_sample(Wc, Hc, yin, np.array([uc, vc, 0.0]), rng)
            L = Lopt / ss
            tl, absorbed, w = _track_and_weight(w, sa, L)
            flux_tl[iy, ix] += tl
            absorbed_weight += absorbed
            total_tl += tl
            xc, yc, canon_exit_face = p_to_canonical_point(p, dxc, dyc)
            uco, vco = _project_direction_out_of_face(canon_exit_face, uco, vco)
            xloc2, yloc2, u, v, face = invert_canonical_exit(entry_face, xc, yc, uco, vco, problem.dx, problem.dy)
            x = x0 + xloc2
            y = y0 + yloc2
            x, y = _nudge_after_face(x, y, face, eps=max(problem.dx, problem.dy) * 1e-10)
        else:
            leakage += 1
    flux = flux_tl / problem.volume
    return TransportResult(flux, leakage, absorbed_weight, total_tl, n_particles, total_steps)
