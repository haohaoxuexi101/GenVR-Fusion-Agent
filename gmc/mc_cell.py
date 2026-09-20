"""Monte Carlo random-flight data generation for GMC cell-transmission models.

This module implements the 2D rectangular-cell boundary and internal-source
training datasets used in the stage-1/stage-3 reproduction.  The geometry is
represented in optical coordinates, so the scattering cross section is one and
collision distances are Exp(1).  Particles scatter isotropically in 3D; only the
x-y projection moves particles in the 2D cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundaryBatch:
    """Raw boundary-model samples.

    conditions columns:
        W, H, y_in, u_in, v_in
    targets columns:
        p_exit, u_exit, v_exit, path_length
    """

    conditions: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class InternalBatch:
    """Raw internal-source model samples.

    conditions columns:
        W, H, x_in, y_in, u_in, v_in
    targets columns:
        p_exit, u_exit, v_exit, path_length
    """

    conditions: np.ndarray
    targets: np.ndarray


def sample_isotropic_direction(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n directions uniformly on S^2.

    Returns columns [u, v, w].  Only u and v move particles in the 2D cell; the
    full 3D direction is still needed because isotropic scattering in 3D does not
    induce a uniform distribution in the projected unit disk.
    """

    z = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def sample_left_boundary_direction(
    rng: np.random.Generator,
    n: int,
    mode: str = "cosine",
) -> np.ndarray:
    """Sample directions entering through the left face x=0.

    mode="cosine" is the natural surface-crossing distribution for an isotropic
    angular flux: pdf(mu)=2*mu on the inward hemisphere.  mode="uniform" samples
    uniformly over the inward hemisphere.  Both are useful sensitivity checks
    because papers often state only that a left-face boundary source is used.
    """

    if mode == "cosine":
        mu = np.sqrt(rng.uniform(0.0, 1.0, size=n))  # x direction cosine, mu>0
    elif mode == "uniform":
        mu = rng.uniform(0.0, 1.0, size=n)
    else:
        raise ValueError("mode must be 'cosine' or 'uniform'")
    eta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rt = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    return np.stack([mu, rt * np.cos(eta), rt * np.sin(eta)], axis=1)


def distance_to_rectangle_boundary(x: float, y: float, u: float, v: float, W: float, H: float) -> float:
    """Distance along direction (u,v) to the first rectangle boundary."""

    eps = 1.0e-14
    sx = np.inf
    sy = np.inf
    if u > eps:
        sx = (W - x) / u
    elif u < -eps:
        sx = -x / u
    if v > eps:
        sy = (H - y) / v
    elif v < -eps:
        sy = -y / v
    return float(min(sx, sy))


def perimeter_coordinate(x: float, y: float, W: float, H: float) -> float:
    """Map a boundary point to an unwrapped perimeter coordinate p in [0, 1)."""

    # Choose the closest face; this is robust at corners where two faces meet.
    distances = np.array([abs(y), abs(x - W), abs(y - H), abs(x)])
    face = int(np.argmin(distances))
    perim = 2.0 * (W + H)
    if face == 0:  # bottom, left -> right
        s = np.clip(x, 0.0, W)
    elif face == 1:  # right, bottom -> top
        s = W + np.clip(y, 0.0, H)
    elif face == 2:  # top, right -> left
        s = W + H + (W - np.clip(x, 0.0, W))
    else:  # left, top -> bottom
        s = W + H + W + (H - np.clip(y, 0.0, H))
    return float((s / perim) % 1.0)


def random_flight_sample_from_state(
    W: float,
    H: float,
    x_in: float,
    y_in: float,
    direction: np.ndarray,
    rng: np.random.Generator,
    max_collisions: int = 2_000_000,
) -> Tuple[float, float, float, float]:
    """Run one pure-scattering in-cell random flight until boundary exit.

    Returns (p_exit, u_exit, v_exit, path_length) in optical coordinates.
    """

    x = float(x_in)
    y = float(y_in)
    u, v, _w = map(float, direction)
    L = 0.0

    for _ in range(max_collisions):
        sc = float(rng.exponential(scale=1.0))
        sb = distance_to_rectangle_boundary(x, y, u, v, W, H)
        if not np.isfinite(sb):
            # Direction is almost purely out-of-plane; it cannot hit a 2D side
            # before the next collision, so force a collision.
            sb = np.inf
        if sc < sb:
            x += sc * u
            y += sc * v
            L += sc
            u, v, _w = map(float, sample_isotropic_direction(rng, 1)[0])
        else:
            x += sb * u
            y += sb * v
            L += sb
            p = perimeter_coordinate(x, y, W, H)
            return p, u, v, L

    raise RuntimeError(
        f"random flight exceeded max_collisions={max_collisions}; "
        f"W={W}, H={H}, last L={L}"
    )


def random_flight_boundary_sample(
    W: float,
    H: float,
    y_in: float,
    direction: np.ndarray,
    rng: np.random.Generator,
    max_collisions: int = 2_000_000,
) -> Tuple[float, float, float, float]:
    """Boundary-to-boundary random flight from the canonical left face."""

    return random_flight_sample_from_state(W, H, 0.0, y_in, direction, rng, max_collisions)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), size=n)


def generate_boundary_dataset(
    n: int,
    seed: int = 1234,
    W_range: Tuple[float, float] = (0.015, 5.0),
    H_range: Tuple[float, float] = (0.015, 5.0),
    direction_mode: str = "cosine",
    fixed_W: Optional[float] = None,
    fixed_H: Optional[float] = None,
    progress_every: int = 0,
) -> BoundaryBatch:
    """Generate n boundary-model training or validation samples."""

    rng = np.random.default_rng(seed)
    if fixed_W is None:
        Ws = _log_uniform(rng, W_range[0], W_range[1], n)
    else:
        Ws = np.full(n, float(fixed_W))
    if fixed_H is None:
        Hs = _log_uniform(rng, H_range[0], H_range[1], n)
    else:
        Hs = np.full(n, float(fixed_H))

    y_in = rng.uniform(0.0, 1.0, size=n) * Hs
    dirs = sample_left_boundary_direction(rng, n, mode=direction_mode)

    conditions = np.zeros((n, 5), dtype=np.float32)
    targets = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        p, uo, vo, L = random_flight_boundary_sample(
            float(Ws[i]), float(Hs[i]), float(y_in[i]), dirs[i], rng
        )
        conditions[i] = (Ws[i], Hs[i], y_in[i], dirs[i, 0], dirs[i, 1])
        targets[i] = (p, uo, vo, L)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"generated {i + 1}/{n}", flush=True)

    return BoundaryBatch(conditions=conditions, targets=targets)


def generate_internal_dataset(
    n: int,
    seed: int = 4321,
    W_range: Tuple[float, float] = (0.015, 5.0),
    H_range: Tuple[float, float] = (0.015, 5.0),
    fixed_W: Optional[float] = None,
    fixed_H: Optional[float] = None,
    progress_every: int = 0,
) -> InternalBatch:
    """Generate n internal-source model samples.

    The internal source is uniform in the cell and isotropic in 3D direction.
    This model is needed for volumetric source histories such as the lattice
    benchmark.  Its target is the same physical exit tuple used by the boundary
    model: perimeter coordinate, outgoing projected direction, and path length.
    """

    rng = np.random.default_rng(seed)
    if fixed_W is None:
        Ws = _log_uniform(rng, W_range[0], W_range[1], n)
    else:
        Ws = np.full(n, float(fixed_W))
    if fixed_H is None:
        Hs = _log_uniform(rng, H_range[0], H_range[1], n)
    else:
        Hs = np.full(n, float(fixed_H))

    x_in = rng.uniform(0.0, 1.0, size=n) * Ws
    y_in = rng.uniform(0.0, 1.0, size=n) * Hs
    dirs = sample_isotropic_direction(rng, n)

    conditions = np.zeros((n, 6), dtype=np.float32)
    targets = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        p, uo, vo, L = random_flight_sample_from_state(
            float(Ws[i]), float(Hs[i]), float(x_in[i]), float(y_in[i]), dirs[i], rng
        )
        conditions[i] = (Ws[i], Hs[i], x_in[i], y_in[i], dirs[i, 0], dirs[i, 1])
        targets[i] = (p, uo, vo, L)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"generated {i + 1}/{n}", flush=True)

    return InternalBatch(conditions=conditions, targets=targets)


def save_npz(path: str, batch: BoundaryBatch | InternalBatch) -> None:
    np.savez_compressed(path, conditions=batch.conditions, targets=batch.targets)


def load_boundary_npz(path: str) -> BoundaryBatch:
    data = np.load(path)
    return BoundaryBatch(conditions=data["conditions"], targets=data["targets"])


def load_internal_npz(path: str) -> InternalBatch:
    data = np.load(path)
    return InternalBatch(conditions=data["conditions"], targets=data["targets"])


# Backward-compatible name used by stage-1/stage-2 scripts.
def load_npz(path: str) -> BoundaryBatch:
    return load_boundary_npz(path)
