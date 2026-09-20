from __future__ import annotations

import numpy as np
import pytest

from gmc.benchmarks2d import make_hohlraum_problem, make_lattice_problem
from gmc.response_matrix2d import product_s2_quadrature, solve_response_matrix


def test_azimuth_offset_rotates_quadrature_without_changing_weight() -> None:
    u0, v0, w0 = product_s2_quadrature(2, 16, phi_offset_fraction=0.0)
    u1, v1, w1 = product_s2_quadrature(2, 16, phi_offset_fraction=0.25)

    assert np.isclose(np.sum(w0), 4.0 * np.pi)
    assert np.isclose(np.sum(w1), 4.0 * np.pi)
    assert not np.allclose(u0, u1)
    assert not np.allclose(v0, v1)


def test_diamond_difference_is_positive_and_reports_fixup() -> None:
    problem = make_hohlraum_problem(24, 24)
    result = solve_response_matrix(
        problem,
        n_mu=2,
        n_phi=8,
        max_iters=600,
        rtol=1.0e-8,
    )

    assert result.converged
    assert result.spatial_scheme == "diamond"
    assert result.fixup_fraction > 0.0
    assert result.double_fixup_fraction >= 0.0
    assert np.all(np.isfinite(result.flux))
    assert np.all(result.flux >= 0.0)


def test_diamond_difference_reduces_hohlraum_upwind_diffusion() -> None:
    problem = make_hohlraum_problem(32, 32)
    diamond = solve_response_matrix(
        problem,
        n_mu=2,
        n_phi=8,
        max_iters=600,
        rtol=1.0e-8,
        spatial_scheme="diamond",
    )
    upwind = solve_response_matrix(
        problem,
        n_mu=2,
        n_phi=8,
        max_iters=600,
        rtol=1.0e-8,
        spatial_scheme="upwind",
    )

    diamond_track = float(np.sum(diamond.flux) * problem.volume)
    upwind_track = float(np.sum(upwind.flux) * problem.volume)
    assert diamond.converged and upwind.converged
    assert diamond_track < 0.9 * upwind_track


def test_upwind_compatibility_and_invalid_scheme() -> None:
    problem = make_lattice_problem(16, 16)
    result = solve_response_matrix(
        problem,
        n_mu=2,
        n_phi=8,
        max_iters=300,
        rtol=1.0e-7,
        spatial_scheme="upwind",
    )

    assert result.converged
    assert result.spatial_scheme == "upwind"
    assert result.fixup_fraction == 0.0
    assert result.double_fixup_fraction == 0.0
    with pytest.raises(ValueError, match="spatial_scheme"):
        solve_response_matrix(problem, spatial_scheme="invalid")
