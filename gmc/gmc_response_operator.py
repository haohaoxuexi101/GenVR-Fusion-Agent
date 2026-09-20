"""Phase-space response operators built from the GMC CFM sampler.

This module turns Farmer-style *random cell transmission samples* into a reusable
linear response operator.  It is deliberately separate from the paper's online
history-based GMC algorithm: the trained CFM model is queried offline in batches,
and its conditional samples are histogrammed into an expected boundary-current
operator.  The global solve then propagates *expected current*, not particle
histories, so there are no empty tally cells simply because a rare history was not
sampled online.

For one material/cell type we construct

    J_out = R J_in,
    Phi_cell += T J_in,

where R includes continuous-absorption attenuation sample-by-sample and T is the
expected weight-integrated physical track length per unit incoming weight.
For an internal volumetric source we similarly construct an outgoing source vector
and an in-cell track-length response.

This is an exploratory deterministic extension of the published GMC sampler, not a
claim that Farmer et al. solved the benchmarks this way.  Its purpose is to retain
the learned GMC cell physics while exploiting response-matrix composition for deep
penetration / globally populated flux fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

from .benchmarks2d import Structured2DProblem
from .mc_cell import random_flight_boundary_sample, random_flight_sample_from_state, sample_isotropic_direction
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


@dataclass(frozen=True)
class PhaseSpaceDiscretization:
    """Compact face-position-angle discretization used by the response operator."""

    n_pos: int = 2
    # (normal component, tangential component).  Components are x/y projected
    # direction cosines; the missing z component is immaterial to 2D geometry.
    angle_components: Tuple[Tuple[float, float], ...] = (
        (0.86, -0.30),
        (0.86, +0.30),
        (0.48, -0.72),
        (0.48, +0.72),
    )

    @property
    def n_angle(self) -> int:
        return len(self.angle_components)

    @property
    def n_state(self) -> int:
        return 4 * self.n_pos * self.n_angle

    def state_index(self, face: Face, pos_bin: int, angle_bin: int) -> int:
        return (FACE_TO_ID[face] * self.n_pos + pos_bin) * self.n_angle + angle_bin

    def decode_state(self, idx: int) -> Tuple[Face, int, int]:
        angle = idx % self.n_angle
        q = idx // self.n_angle
        pos = q % self.n_pos
        face = FACES[q // self.n_pos]
        return face, pos, angle

    def face_position(self, face: Face, pos_bin: int, dx: float, dy: float) -> Tuple[float, float]:
        f = (pos_bin + 0.5) / self.n_pos
        if face == "left":
            return 0.0, f * dy
        if face == "right":
            return dx, f * dy
        if face == "bottom":
            return f * dx, 0.0
        return f * dx, dy

    def position_bin(self, face: Face, x: float, y: float, dx: float, dy: float) -> int:
        frac = y / dy if face in ("left", "right") else x / dx
        return int(np.clip(np.floor(frac * self.n_pos), 0, self.n_pos - 1))

    def direction_for_entry(self, face: Face, angle_bin: int) -> Tuple[float, float]:
        normal, tang = self.angle_components[angle_bin]
        if face == "left":
            return normal, tang
        if face == "right":
            return -normal, tang
        if face == "bottom":
            return tang, normal
        return tang, -normal

    def nearest_angle_for_entry(self, face: Face, u: float, v: float) -> int:
        reps = np.asarray([self.direction_for_entry(face, a) for a in range(self.n_angle)], dtype=np.float64)
        q = np.array([u, v], dtype=np.float64)
        # Normalize only for comparison so variation in projected radius does not
        # spuriously select a grazing bin.
        nr = np.linalg.norm(reps, axis=1)
        nq = max(float(np.linalg.norm(q)), 1.0e-12)
        scores = (reps @ q) / (nr * nq)
        return int(np.argmax(scores))


@dataclass
class CellResponseOperator:
    sigma_s: float
    sigma_a: float
    dx: float
    dy: float
    R: np.ndarray                 # shape (n_state_out, n_state_in)
    track: np.ndarray             # expected physical weight-integrated path per incoming state
    raw_exit_probability: np.ndarray  # unattenuated histogram, useful for diagnostics
    samples_per_state: int
    backend: str
    invalid_fraction: float

    @property
    def row_survival(self) -> np.ndarray:
        return np.sum(self.R, axis=0)


@dataclass
class InternalSourceResponse:
    outgoing: np.ndarray          # expected attenuated outgoing state weights per source birth
    track: float                  # expected physical weight-integrated path per source birth
    samples: int
    backend: str
    invalid_fraction: float


@dataclass
class GlobalResponseResult:
    flux: np.ndarray
    iterations: int
    residual_history: np.ndarray
    leaked_weight: float
    remaining_weight: float
    nonzero_fraction: float
    min_positive_flux: float
    source_strength: float
    total_track_length: float


def _force_outward(face: Face, u: float, v: float) -> Tuple[float, float]:
    """Hard geometric direction constraint matching the paper's post-processing idea."""
    if face == "left":
        u = -abs(u)
    elif face == "right":
        u = abs(u)
    elif face == "bottom":
        v = -abs(v)
    else:
        v = abs(v)
    r2 = u*u + v*v
    if r2 > 1.0:
        s = 1.0 / np.sqrt(r2)
        u *= s
        v *= s
    return float(u), float(v)


def _track_response_from_Lopt(Lopt: np.ndarray, sigma_s: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return surviving weight factor and physical integrated-track response."""
    if sigma_s <= 0.0:
        raise ValueError("GMC optical response requires sigma_s>0")
    kappa = sigma_a / sigma_s
    survive = np.exp(-kappa * np.maximum(Lopt, 0.0))
    if sigma_a > 0.0:
        track = (1.0 - survive) / sigma_a
    else:
        track = np.maximum(Lopt, 0.0) / sigma_s
    return survive, track


def _canonical_condition_for_state(
    disc: PhaseSpaceDiscretization,
    state: int,
    sigma_s: float,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, Face, float, float, float, float]:
    face, pos_bin, angle_bin = disc.decode_state(state)
    x, y = disc.face_position(face, pos_bin, dx, dy)
    u, v = disc.direction_for_entry(face, angle_bin)
    raw, dxc, dyc = canonicalize_boundary_state(face, x, y, u, v, dx, dy, sigma_s)
    return raw, face, x, y, dxc, dyc


def _map_sample_to_output_state(
    disc: PhaseSpaceDiscretization,
    entry_face: Face,
    p: float,
    uc: float,
    vc: float,
    dxc: float,
    dyc: float,
    dx: float,
    dy: float,
) -> Tuple[int, bool]:
    if not np.isfinite(p + uc + vc):
        return 0, False
    xc, yc, _ = p_to_canonical_point(float(p), dxc, dyc)
    x, y, u, v, exit_face = invert_canonical_exit(entry_face, xc, yc, float(uc), float(vc), dx, dy)
    u, v = _force_outward(exit_face, u, v)
    # The same outgoing direction becomes an incoming direction in the neighbor.
    neighbor_entry = OPPOSITE[exit_face]
    apos = disc.position_bin(exit_face, x, y, dx, dy)
    aang = disc.nearest_angle_for_entry(neighbor_entry, u, v)
    # Store the *exit face* for routing; the angle label is already expressed as
    # the matching incoming class of the neighboring cell.
    out_state = disc.state_index(exit_face, apos, aang)
    return out_state, True


def build_boundary_response_cfm(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    sampler: BoundaryGMCSampler,
    disc: PhaseSpaceDiscretization = PhaseSpaceDiscretization(),
    samples_per_state: int = 256,
    seed: int = 1000,
    inference_batch_size: int = 8192,
) -> CellResponseOperator:
    """Histogram the CFM cell sampler into a deterministic response matrix."""
    ns = disc.n_state
    R = np.zeros((ns, ns), dtype=np.float64)
    P = np.zeros((ns, ns), dtype=np.float64)
    track = np.zeros(ns, dtype=np.float64)
    invalid = 0
    total = 0
    for i in range(ns):
        raw1, entry_face, _x, _y, dxc, dyc = _canonical_condition_for_state(disc, i, sigma_s, dx, dy)
        raw = np.repeat(raw1, samples_per_state, axis=0)
        out = sampler.sample_batch(raw, seed=seed + i * samples_per_state, batch_size=inference_batch_size)
        survive, tr = _track_response_from_Lopt(out[:, 3].astype(np.float64), sigma_s, sigma_a)
        track[i] = float(np.mean(tr[np.isfinite(tr)])) if np.any(np.isfinite(tr)) else 0.0
        for m in range(samples_per_state):
            total += 1
            if not np.all(np.isfinite(out[m])) or out[m, 3] <= 0.0:
                invalid += 1
                continue
            j, ok = _map_sample_to_output_state(
                disc, entry_face, float(out[m, 0]), float(out[m, 1]), float(out[m, 2]),
                dxc, dyc, dx, dy,
            )
            if not ok:
                invalid += 1
                continue
            P[j, i] += 1.0 / samples_per_state
            R[j, i] += float(survive[m]) / samples_per_state
    return CellResponseOperator(
        sigma_s=sigma_s, sigma_a=sigma_a, dx=dx, dy=dy, R=R, track=track,
        raw_exit_probability=P, samples_per_state=samples_per_state, backend="cfm",
        invalid_fraction=invalid / max(total, 1),
    )


def build_boundary_response_mc(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    disc: PhaseSpaceDiscretization = PhaseSpaceDiscretization(),
    samples_per_state: int = 4096,
    seed: int = 2000,
) -> CellResponseOperator:
    """Exact random-flight reference for the same discretized local operator."""
    rng = np.random.default_rng(seed)
    ns = disc.n_state
    R = np.zeros((ns, ns), dtype=np.float64)
    P = np.zeros((ns, ns), dtype=np.float64)
    track = np.zeros(ns, dtype=np.float64)
    invalid = 0
    total = 0
    W = sigma_s * dx
    H = sigma_s * dy
    for i in range(ns):
        face, pos_bin, angle_bin = disc.decode_state(i)
        x, y = disc.face_position(face, pos_bin, dx, dy)
        u, v = disc.direction_for_entry(face, angle_bin)
        raw, dxc, dyc = canonicalize_boundary_state(face, x, y, u, v, dx, dy, sigma_s)
        yopt = float(raw[0, 2])
        uc, vc = float(raw[0, 3]), float(raw[0, 4])
        wc = np.sqrt(max(0.0, 1.0 - uc*uc - vc*vc))
        Ls = np.zeros(samples_per_state, dtype=np.float64)
        outs = []
        for m in range(samples_per_state):
            total += 1
            p, uo, vo, L = random_flight_boundary_sample(W, H, yopt, np.array([uc, vc, wc]), rng)
            Ls[m] = L
            j, ok = _map_sample_to_output_state(disc, face, p, uo, vo, dxc, dyc, dx, dy)
            outs.append((j, ok))
        survive, tr = _track_response_from_Lopt(Ls, sigma_s, sigma_a)
        track[i] = float(np.mean(tr))
        for m, (j, ok) in enumerate(outs):
            if not ok:
                invalid += 1
                continue
            P[j, i] += 1.0 / samples_per_state
            R[j, i] += float(survive[m]) / samples_per_state
    return CellResponseOperator(
        sigma_s=sigma_s, sigma_a=sigma_a, dx=dx, dy=dy, R=R, track=track,
        raw_exit_probability=P, samples_per_state=samples_per_state, backend="mc",
        invalid_fraction=invalid / max(total, 1),
    )


def _classify_internal_output(
    disc: PhaseSpaceDiscretization,
    p: float, u: float, v: float,
    dx: float, dy: float,
) -> Tuple[int, bool]:
    if not np.isfinite(p + u + v):
        return 0, False
    x, y, exit_face = p_to_canonical_point(float(p), dx, dy)
    u, v = _force_outward(exit_face, float(u), float(v))
    neighbor_entry = OPPOSITE[exit_face]
    pos = disc.position_bin(exit_face, x, y, dx, dy)
    ang = disc.nearest_angle_for_entry(neighbor_entry, u, v)
    return disc.state_index(exit_face, pos, ang), True


def build_internal_source_response_cfm(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    sampler: InternalGMCSampler,
    disc: PhaseSpaceDiscretization = PhaseSpaceDiscretization(),
    samples: int = 8192,
    seed: int = 3000,
    inference_batch_size: int = 8192,
) -> InternalSourceResponse:
    rng = np.random.default_rng(seed)
    W, H = sigma_s * dx, sigma_s * dy
    x = rng.uniform(0.0, W, size=samples)
    y = rng.uniform(0.0, H, size=samples)
    dirs = sample_isotropic_direction(rng, samples)
    raw = np.column_stack([np.full(samples, W), np.full(samples, H), x, y, dirs[:, 0], dirs[:, 1]]).astype(np.float32)
    out = sampler.sample_batch(raw, seed=seed + 17, batch_size=inference_batch_size)
    survive, tr = _track_response_from_Lopt(out[:, 3].astype(np.float64), sigma_s, sigma_a)
    vec = np.zeros(disc.n_state, dtype=np.float64)
    invalid = 0
    for m in range(samples):
        if not np.all(np.isfinite(out[m])) or out[m, 3] <= 0.0:
            invalid += 1
            continue
        j, ok = _classify_internal_output(disc, float(out[m, 0]), float(out[m, 1]), float(out[m, 2]), dx, dy)
        if not ok:
            invalid += 1
            continue
        vec[j] += float(survive[m]) / samples
    return InternalSourceResponse(vec, float(np.nanmean(tr)), samples, "cfm", invalid/max(samples,1))


def build_internal_source_response_mc(
    sigma_s: float,
    sigma_a: float,
    dx: float,
    dy: float,
    disc: PhaseSpaceDiscretization = PhaseSpaceDiscretization(),
    samples: int = 32768,
    seed: int = 4000,
) -> InternalSourceResponse:
    rng = np.random.default_rng(seed)
    W, H = sigma_s * dx, sigma_s * dy
    vec = np.zeros(disc.n_state, dtype=np.float64)
    Ls = np.zeros(samples, dtype=np.float64)
    js = np.zeros(samples, dtype=np.int64)
    oks = np.ones(samples, dtype=bool)
    for m in range(samples):
        x = float(rng.uniform(0.0, W)); y = float(rng.uniform(0.0, H))
        d = sample_isotropic_direction(rng, 1)[0]
        p, u, v, L = random_flight_sample_from_state(W, H, x, y, d, rng)
        Ls[m] = L
        js[m], oks[m] = _classify_internal_output(disc, p, u, v, dx, dy)
    survive, tr = _track_response_from_Lopt(Ls, sigma_s, sigma_a)
    for m in range(samples):
        if oks[m]:
            vec[js[m]] += float(survive[m]) / samples
    return InternalSourceResponse(vec, float(np.mean(tr)), samples, "mc", float(np.mean(~oks)))


def unique_materials(problem: Structured2DProblem) -> Sequence[Tuple[float, float]]:
    pairs = np.column_stack([problem.sigma_s.ravel(), problem.sigma_a.ravel()])
    return [tuple(map(float, x)) for x in np.unique(pairs, axis=0)]


def build_problem_operator_library(
    problem: Structured2DProblem,
    boundary_sampler: Optional[BoundaryGMCSampler],
    disc: PhaseSpaceDiscretization,
    backend: Literal["cfm", "mc"] = "cfm",
    samples_per_state: int = 256,
    seed: int = 5000,
) -> Dict[Tuple[float, float], CellResponseOperator]:
    lib: Dict[Tuple[float, float], CellResponseOperator] = {}
    for k, (ss, sa) in enumerate(unique_materials(problem)):
        if backend == "cfm":
            if boundary_sampler is None:
                raise ValueError("boundary_sampler required for cfm backend")
            op = build_boundary_response_cfm(ss, sa, problem.dx, problem.dy, boundary_sampler, disc,
                                             samples_per_state=samples_per_state, seed=seed + 100000*k)
        else:
            op = build_boundary_response_mc(ss, sa, problem.dx, problem.dy, disc,
                                            samples_per_state=samples_per_state, seed=seed + 100000*k)
        lib[(ss, sa)] = op
    return lib


def _route_outgoing(
    outgoing: np.ndarray,
    disc: PhaseSpaceDiscretization,
) -> Tuple[np.ndarray, float]:
    """Route cell-exit state weights to the neighboring cell's incoming state."""
    ny, nx, ns = outgoing.shape
    nxt = np.zeros_like(outgoing)
    leaked = 0.0
    for s in range(ns):
        exit_face, pos, ang = disc.decode_state(s)
        neighbor_face = OPPOSITE[exit_face]
        sin = disc.state_index(neighbor_face, pos, ang)
        a = outgoing[:, :, s]
        if exit_face == "right":
            nxt[:, 1:, sin] += a[:, :-1]
            leaked += float(np.sum(a[:, -1]))
        elif exit_face == "left":
            nxt[:, :-1, sin] += a[:, 1:]
            leaked += float(np.sum(a[:, 0]))
        elif exit_face == "top":
            nxt[1:, :, sin] += a[:-1, :]
            leaked += float(np.sum(a[-1, :]))
        else:
            nxt[:-1, :, sin] += a[1:, :]
            leaked += float(np.sum(a[0, :]))
    return nxt, leaked


def _boundary_source_injection(problem: Structured2DProblem, disc: PhaseSpaceDiscretization) -> np.ndarray:
    inc = np.zeros((problem.ny, problem.nx, disc.n_state), dtype=np.float64)
    if problem.source_kind != "left_boundary":
        return inc
    ymin, ymax = problem.boundary_source_y_range or (0.0, problem.height)
    yc = (np.arange(problem.ny) + 0.5) * problem.dy
    active = np.where((yc >= ymin) & (yc <= ymax))[0]
    if active.size == 0:
        raise RuntimeError("empty boundary source")
    # Unit total incoming current.  Distribute across position/angle classes using
    # representative normal current, which approximates cosine incidence.
    aw = np.asarray([max(disc.direction_for_entry("left", a)[0], 0.0) for a in range(disc.n_angle)], dtype=np.float64)
    aw /= np.sum(aw)
    for iy in active:
        cell_weight = 1.0 / active.size
        # Boundary source spans the whole cell face; distribute equally over pos bins.
        for p in range(disc.n_pos):
            for a in range(disc.n_angle):
                s = disc.state_index("left", p, a)
                inc[iy, 0, s] += cell_weight * aw[a] / disc.n_pos
    return inc


def _source_mask_and_weights(problem: Structured2DProblem) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((problem.ny, problem.nx), dtype=bool)
    weights = np.zeros_like(mask, dtype=np.float64)
    if problem.source_kind != "volume_box":
        return mask, weights
    xmin, xmax, ymin, ymax = problem.source_box  # type: ignore[misc]
    xc = (np.arange(problem.nx) + 0.5) * problem.dx
    yc = (np.arange(problem.ny) + 0.5) * problem.dy
    X, Y = np.meshgrid(xc, yc)
    mask = (X >= xmin) & (X < xmax) & (Y >= ymin) & (Y < ymax)
    if not np.any(mask):
        raise RuntimeError("empty volumetric source mask")
    weights[mask] = 1.0 / np.count_nonzero(mask)
    return mask, weights


def solve_global_response(
    problem: Structured2DProblem,
    library: Dict[Tuple[float, float], CellResponseOperator],
    disc: PhaseSpaceDiscretization,
    internal_source_response: Optional[InternalSourceResponse] = None,
    max_iters: int = 2000,
    weight_tol: float = 1.0e-13,
) -> GlobalResponseResult:
    """Compose local response matrices globally without launching histories."""
    ny, nx, ns = problem.ny, problem.nx, disc.n_state
    incoming = _boundary_source_injection(problem, disc)
    flux_tl = np.zeros((ny, nx), dtype=np.float64)
    leaked = 0.0
    source_strength = 1.0

    # Volumetric source is handled by the internal GMC response: tally its in-cell
    # path immediately, then inject its expected boundary leakage into neighbors.
    if problem.source_kind == "volume_box":
        if internal_source_response is None:
            raise ValueError("internal_source_response required for volumetric source")
        mask, sw = _source_mask_and_weights(problem)
        flux_tl += sw * internal_source_response.track
        out_src = sw[:, :, None] * internal_source_response.outgoing[None, None, :]
        incoming_src, leak0 = _route_outgoing(out_src, disc)
        incoming += incoming_src
        leaked += leak0

    # Material masks allow vectorized application of one small response matrix to
    # thousands of cells sharing the same local physics.
    mats: Dict[Tuple[float, float], np.ndarray] = {}
    for key in library:
        ss, sa = key
        mats[key] = np.isclose(problem.sigma_s, ss) & np.isclose(problem.sigma_a, sa)

    residuals = []
    total_track = float(np.sum(flux_tl))
    for it in range(1, max_iters + 1):
        active_weight = float(np.sum(incoming))
        residuals.append(active_weight)
        if active_weight < weight_tol:
            break
        outgoing = np.zeros_like(incoming)
        for key, op in library.items():
            mask = mats[key]
            X = incoming[mask]          # (ncells, ns)
            if X.size == 0:
                continue
            # track per cell: sum_i J_i T_i
            tl = X @ op.track
            flux_tl[mask] += tl
            total_track += float(np.sum(tl))
            # outgoing row vector = incoming row vector @ R.T
            outgoing[mask] = X @ op.R.T
        incoming, dl = _route_outgoing(outgoing, disc)
        leaked += dl
    else:
        it = max_iters

    flux = flux_tl / problem.volume
    pos = flux > 0.0
    return GlobalResponseResult(
        flux=flux,
        iterations=it,
        residual_history=np.asarray(residuals),
        leaked_weight=leaked,
        remaining_weight=float(np.sum(incoming)),
        nonzero_fraction=float(np.mean(pos)),
        min_positive_flux=float(np.min(flux[pos])) if np.any(pos) else 0.0,
        source_strength=source_strength,
        total_track_length=total_track,
    )


def response_operator_metrics(a: CellResponseOperator, b: CellResponseOperator) -> Dict[str, float]:
    """Compare two local response operators (typically CFM vs direct local MC)."""
    pa, pb = a.raw_exit_probability, b.raw_exit_probability
    Ra, Rb = a.R, b.R
    return {
        "prob_l1_mean_per_input": float(np.mean(np.sum(np.abs(pa-pb), axis=0))),
        "prob_max_abs": float(np.max(np.abs(pa-pb))),
        "attenuated_l1_mean_per_input": float(np.mean(np.sum(np.abs(Ra-Rb), axis=0))),
        "track_rel_l1": float(np.sum(np.abs(a.track-b.track)) / max(np.sum(np.abs(b.track)), 1e-30)),
        "cfm_invalid_fraction": float(a.invalid_fraction),
        "mc_invalid_fraction": float(b.invalid_fraction),
        "cfm_mean_survival": float(np.mean(np.sum(Ra, axis=0))),
        "mc_mean_survival": float(np.mean(np.sum(Rb, axis=0))),
    }
