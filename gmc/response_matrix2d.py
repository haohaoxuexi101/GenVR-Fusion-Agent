"""Deterministic dense-response transport for the 2D Farmer benchmarks.

This module deliberately separates *response construction* from Monte Carlo
history sampling.  Each angular direction and cell is treated as a local linear
response operator mapping upwind face angular fluxes and an isotropic in-cell
source to the cell/outgoing angular flux.  Global source iteration composes those
local responses over the full mesh.

The default spatial closure is diamond difference.  With
``ax=|Omega_x|/dx`` and ``ay=|Omega_y|/dy``, it uses

    psi_c = (q + 2 ax psi_x,in + 2 ay psi_y,in)
            / (Sigma_t + 2 ax + 2 ay)
    psi_x,out = 2 psi_c - psi_x,in
    psi_y,out = 2 psi_c - psi_y,in.

Unmodified diamond difference can produce negative outgoing angular fluxes in
optically thick cells.  This implementation applies a conservative zero-flux
fixup only in those cells: a negative outgoing face is set to zero and the cell
balance is re-solved using diamond closure on the remaining face.  The previous
positive-upwind closure remains available explicitly for regression studies.

It is not claimed to be Farmer's neural CFM sampler.  It is the deterministic
response-operator branch needed to test the user's key hypothesis: if the local
cell response is represented as a full operator rather than one random exit draw,
deep/low-flux regions remain populated deterministically.  The result still has
finite spatial and angular discretization error and must not be described as an
exact transport solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numba import njit

from .benchmarks2d import Structured2DProblem


SpatialScheme = Literal["diamond", "upwind"]


@dataclass
class ResponseSolveResult:
    flux: np.ndarray
    iterations: int
    converged: bool
    residual_history: np.ndarray
    angular_order: Tuple[int, int]
    source_strength: float
    min_flux: float
    nonzero_fraction: float
    spatial_scheme: str = "diamond"
    phi_offset_fraction: float = 0.0
    fixup_fraction: float = 0.0
    double_fixup_fraction: float = 0.0


def product_s2_quadrature(
    n_mu: int = 4,
    n_phi: int = 16,
    phi_offset_fraction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Product quadrature on S^2 for problems invariant in z.

    mu is the z-direction cosine.  phi is uniform in [0,2pi).  The resulting
    (u,v) are the active x/y direction cosines.  Weights sum to 4*pi.
    """
    if n_mu < 2 or n_phi < 4:
        raise ValueError("use n_mu>=2 and n_phi>=4")
    if not np.isfinite(phi_offset_fraction) or not 0.0 <= phi_offset_fraction < 1.0:
        raise ValueError("phi_offset_fraction must be finite and in [0,1)")
    mu, wmu = np.polynomial.legendre.leggauss(n_mu)
    phis = (
        np.arange(n_phi, dtype=np.float64) + 0.5 + phi_offset_fraction
    ) * (2.0 * np.pi / n_phi)
    wphi = 2.0 * np.pi / n_phi
    u = []
    v = []
    w = []
    for m, wm in zip(mu, wmu):
        rho = np.sqrt(max(0.0, 1.0 - m*m))
        for ph in phis:
            u.append(rho * np.cos(ph))
            v.append(rho * np.sin(ph))
            w.append(wm * wphi)
    return np.asarray(u), np.asarray(v), np.asarray(w)


@njit(cache=True)
def _response_sweep(
    sigma_t: np.ndarray,
    q_iso: np.ndarray,
    dx: float,
    dy: float,
    us: np.ndarray,
    vs: np.ndarray,
    weights: np.ndarray,
    left_in: np.ndarray,
    right_in: np.ndarray,
    bottom_in: np.ndarray,
    top_in: np.ndarray,
) -> np.ndarray:
    """One full directional sweep returning scalar flux.

    Boundary arrays are angular flux values with shape (ndir, transverse_cells).
    They are zero for vacuum and nonzero only for prescribed incoming directions.
    """
    ny, nx = sigma_t.shape
    ndir = us.size
    phi = np.zeros((ny, nx), dtype=np.float64)

    for n in range(ndir):
        u = us[n]
        v = vs[n]
        wt = weights[n]
        ax = abs(u) / dx
        ay = abs(v) / dy

        # Incoming angular fluxes from the x-upwind and y-upwind faces are
        # propagated cell-by-cell. xface is indexed by y; yface by x.
        xface = np.empty(ny, dtype=np.float64)
        yface = np.empty(nx, dtype=np.float64)

        if u > 0.0:
            for iy in range(ny):
                xface[iy] = left_in[n, iy]
            ix0, ix1, dix = 0, nx, 1
        else:
            for iy in range(ny):
                xface[iy] = right_in[n, iy]
            ix0, ix1, dix = nx - 1, -1, -1

        if v > 0.0:
            for ix in range(nx):
                yface[ix] = bottom_in[n, ix]
            iy0, iy1, diy = 0, ny, 1
        else:
            for ix in range(nx):
                yface[ix] = top_in[n, ix]
            iy0, iy1, diy = ny - 1, -1, -1

        # Four sweep quadrants are handled by the index signs above.  The
        # positive local operator guarantees psi>=0 whenever inputs are >=0.
        iy = iy0
        while iy != iy1:
            ix = ix0
            while ix != ix1:
                den = ax + ay + sigma_t[iy, ix]
                # q_iso is angular source (per steradian) in the cell.
                psi = (ax * xface[iy] + ay * yface[ix] + q_iso[iy, ix]) / den
                if psi < 0.0:
                    psi = 0.0
                phi[iy, ix] += wt * psi
                xface[iy] = psi
                yface[ix] = psi
                ix += dix
            iy += diy

    return phi


@njit(cache=True)
def _diamond_difference_sweep(
    sigma_t: np.ndarray,
    q_iso: np.ndarray,
    dx: float,
    dy: float,
    us: np.ndarray,
    vs: np.ndarray,
    weights: np.ndarray,
    left_in: np.ndarray,
    right_in: np.ndarray,
    bottom_in: np.ndarray,
    top_in: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """Diamond-difference sweep with conservative zero-flux fixup.

    The fixup is local and balance preserving.  If one DD outgoing face is
    negative, that face is set to zero and the cell balance is solved again with
    diamond closure on the other face.  If the remaining outgoing face is also
    negative, both are set to zero and the cell-average flux follows directly
    from balance.  A degenerate zero-total-cross-section cell falls back to the
    positive upwind closure.
    """
    ny, nx = sigma_t.shape
    ndir = us.size
    phi = np.zeros((ny, nx), dtype=np.float64)
    fixup_count = 0
    double_fixup_count = 0
    tiny = 1.0e-14

    for n in range(ndir):
        u = us[n]
        v = vs[n]
        wt = weights[n]
        ax = abs(u) / dx
        ay = abs(v) / dy

        xface = np.empty(ny, dtype=np.float64)
        yface = np.empty(nx, dtype=np.float64)

        if u > 0.0:
            for iy in range(ny):
                xface[iy] = left_in[n, iy]
            ix0, ix1, dix = 0, nx, 1
        else:
            for iy in range(ny):
                xface[iy] = right_in[n, iy]
            ix0, ix1, dix = nx - 1, -1, -1

        if v > 0.0:
            for ix in range(nx):
                yface[ix] = bottom_in[n, ix]
            iy0, iy1, diy = 0, ny, 1
        else:
            for ix in range(nx):
                yface[ix] = top_in[n, ix]
            iy0, iy1, diy = ny - 1, -1, -1

        iy = iy0
        while iy != iy1:
            ix = ix0
            while ix != ix1:
                sigma = sigma_t[iy, ix]
                source = q_iso[iy, ix]
                incoming_x = xface[iy]
                incoming_y = yface[ix]

                denominator = sigma + 2.0 * ax + 2.0 * ay
                cell_flux = (
                    source + 2.0 * ax * incoming_x + 2.0 * ay * incoming_y
                ) / denominator
                outgoing_x = 2.0 * cell_flux - incoming_x
                outgoing_y = 2.0 * cell_flux - incoming_y

                if outgoing_x < 0.0 or outgoing_y < 0.0:
                    fixup_count += 1
                    if sigma <= tiny:
                        upwind_denominator = sigma + ax + ay
                        cell_flux = (
                            source + ax * incoming_x + ay * incoming_y
                        ) / max(upwind_denominator, tiny)
                        outgoing_x = cell_flux
                        outgoing_y = cell_flux
                    elif outgoing_x < 0.0 and outgoing_y >= 0.0:
                        outgoing_x = 0.0
                        cell_flux = (
                            source + ax * incoming_x + 2.0 * ay * incoming_y
                        ) / (sigma + 2.0 * ay)
                        outgoing_y = 2.0 * cell_flux - incoming_y
                        if outgoing_y < 0.0:
                            outgoing_y = 0.0
                            cell_flux = (
                                source + ax * incoming_x + ay * incoming_y
                            ) / sigma
                            double_fixup_count += 1
                    elif outgoing_y < 0.0 and outgoing_x >= 0.0:
                        outgoing_y = 0.0
                        cell_flux = (
                            source + 2.0 * ax * incoming_x + ay * incoming_y
                        ) / (sigma + 2.0 * ax)
                        outgoing_x = 2.0 * cell_flux - incoming_x
                        if outgoing_x < 0.0:
                            outgoing_x = 0.0
                            cell_flux = (
                                source + ax * incoming_x + ay * incoming_y
                            ) / sigma
                            double_fixup_count += 1
                    else:
                        outgoing_x = 0.0
                        outgoing_y = 0.0
                        cell_flux = (
                            source + ax * incoming_x + ay * incoming_y
                        ) / sigma
                        double_fixup_count += 1

                phi[iy, ix] += wt * cell_flux
                xface[iy] = outgoing_x
                yface[ix] = outgoing_y
                ix += dix
            iy += diy

    return phi, fixup_count, double_fixup_count


def _external_volume_source(problem: Structured2DProblem) -> np.ndarray:
    """Total isotropic source Q integrated over angle, normalized to unit strength."""
    q = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    if problem.source_kind != "volume_box":
        return q
    if problem.source_box is None:
        raise ValueError("volume source missing source_box")
    xmin, xmax, ymin, ymax = problem.source_box
    xs = (np.arange(problem.nx) + 0.5) * problem.dx
    ys = (np.arange(problem.ny) + 0.5) * problem.dy
    X, Y = np.meshgrid(xs, ys)
    mask = (X >= xmin) & (X < xmax) & (Y >= ymin) & (Y < ymax)
    area = float(mask.sum()) * problem.volume
    if area <= 0:
        raise RuntimeError("volume source mask is empty")
    q[mask] = 1.0 / area
    return q


def _incoming_boundary_arrays(
    problem: Structured2DProblem,
    us: np.ndarray,
    vs: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    ndir = us.size
    left = np.zeros((ndir, problem.ny), dtype=np.float64)
    right = np.zeros((ndir, problem.ny), dtype=np.float64)
    bottom = np.zeros((ndir, problem.nx), dtype=np.float64)
    top = np.zeros((ndir, problem.nx), dtype=np.float64)
    if problem.source_kind != "left_boundary":
        return left, right, bottom, top, 0.0

    # Farmer Fig. 3.1d shows the active left boundary source on y=[0.25,1.05].
    yrange = getattr(problem, "boundary_source_y_range", None)
    if yrange is None:
        yrange = (0.0, problem.height)
    ymin, ymax = yrange
    yc = (np.arange(problem.ny) + 0.5) * problem.dy
    active = (yc >= ymin) & (yc <= ymax)
    active_length = float(active.sum()) * problem.dy
    # Normalize incoming current \int_A \int_{u>0} u psi dOmega dA = 1.
    current_quad = float(np.sum(weights[us > 0.0] * us[us > 0.0]))
    if active_length <= 0 or current_quad <= 0:
        raise RuntimeError("invalid boundary source normalization")
    psi0 = 1.0 / (active_length * current_quad)
    for n in range(ndir):
        if us[n] > 0.0:
            left[n, active] = psi0
    return left, right, bottom, top, 1.0


def solve_response_matrix(
    problem: Structured2DProblem,
    n_mu: int = 4,
    n_phi: int = 16,
    max_iters: int = 500,
    rtol: float = 2.0e-7,
    atol: float = 1.0e-14,
    relaxation: float = 1.0,
    initial_flux: Optional[np.ndarray] = None,
    spatial_scheme: SpatialScheme = "diamond",
    phi_offset_fraction: float = 0.0,
) -> ResponseSolveResult:
    """Solve the 2D fixed-source problem by deterministic local-response sweeps.

    The scattering source is iterated to self-consistency.  Because each sweep
    transports *expected angular flux* rather than one random history, the result
    contains no MC empty-tally cells.
    """
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0,1]")
    spatial_scheme = str(spatial_scheme).lower()  # type: ignore[assignment]
    if spatial_scheme not in ("diamond", "upwind"):
        raise ValueError("spatial_scheme must be 'diamond' or 'upwind'")
    us, vs, weights = product_s2_quadrature(n_mu, n_phi, phi_offset_fraction)
    left, right, bottom, top, boundary_strength = _incoming_boundary_arrays(problem, us, vs, weights)
    qext = _external_volume_source(problem)
    sigma_t = np.asarray(problem.sigma_a + problem.sigma_s, dtype=np.float64)
    sigma_s = np.asarray(problem.sigma_s, dtype=np.float64)
    if initial_flux is None:
        phi = np.zeros_like(sigma_t)
    else:
        phi = np.asarray(initial_flux, dtype=np.float64).copy()
    residuals = []
    converged = False
    fixup_count = 0
    double_fixup_count = 0

    for it in range(1, max_iters + 1):
        # Isotropic source per steradian: (Sigma_s phi + Q)/(4*pi).
        q_iso = (sigma_s * phi + qext) / (4.0 * np.pi)
        if spatial_scheme == "diamond":
            phi_sweep, fixup_count, double_fixup_count = _diamond_difference_sweep(
                sigma_t,
                q_iso,
                problem.dx,
                problem.dy,
                us,
                vs,
                weights,
                left,
                right,
                bottom,
                top,
            )
        else:
            phi_sweep = _response_sweep(
                sigma_t,
                q_iso,
                problem.dx,
                problem.dy,
                us,
                vs,
                weights,
                left,
                right,
                bottom,
                top,
            )
            fixup_count = 0
            double_fixup_count = 0
        phi_new = relaxation * phi_sweep + (1.0 - relaxation) * phi
        diff = float(np.max(np.abs(phi_new - phi)))
        scale = max(float(np.max(np.abs(phi_new))), 1.0e-300)
        rel = diff / scale
        residuals.append(rel)
        phi = phi_new
        if diff < atol or rel < rtol:
            converged = True
            break

    nz = phi > 0.0
    min_flux = float(np.min(phi[nz])) if np.any(nz) else 0.0
    sweep_cell_count = max(int(us.size) * problem.nx * problem.ny, 1)
    return ResponseSolveResult(
        flux=phi,
        iterations=it,
        converged=converged,
        residual_history=np.asarray(residuals, dtype=np.float64),
        angular_order=(n_mu, n_phi),
        source_strength=1.0 if problem.source_kind == "volume_box" else boundary_strength,
        min_flux=min_flux,
        nonzero_fraction=float(np.count_nonzero(nz) / phi.size),
        spatial_scheme=spatial_scheme,
        phi_offset_fraction=float(phi_offset_fraction),
        fixup_fraction=float(fixup_count / sweep_cell_count),
        double_fixup_fraction=float(double_fixup_count / sweep_cell_count),
    )
