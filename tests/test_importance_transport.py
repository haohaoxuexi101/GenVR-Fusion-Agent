import numpy as np
import pytest

from gmc.benchmarks2d import Structured2DProblem, make_lattice_problem
from gmc.importance_transport import (
    BoundaryDetector,
    WeightWindowMap,
    combine_response_and_absorber_importance,
    make_adjoint_importance_map,
    make_all_boundary_importance_map,
    make_diffusion_adjoint_importance_map,
    make_inverse_flux_weight_windows,
    make_reciprocal_flux_importance,
    make_right_boundary_importance_map,
    prepare_coarse_flux_field,
    run_detector_mc,
    solve_diffusion_adjoint_importance,
)


def _transparent_problem() -> Structured2DProblem:
    nx = ny = 8
    return Structured2DProblem(
        name="transparent",
        nx=nx,
        ny=ny,
        width=1.0,
        height=1.0,
        sigma_a=np.zeros((ny, nx), dtype=np.float64),
        sigma_s=np.zeros((ny, nx), dtype=np.float64),
        source_kind="left_boundary",
        boundary_source_y_range=(0.4, 0.6),
    )


def test_importance_levels_are_monotone_toward_detector():
    problem = make_lattice_problem(28, 28)
    importance = make_right_boundary_importance_map(problem, n_levels=5, split_factor=2)
    assert importance.levels.shape == (problem.ny, problem.nx)
    assert np.all(np.diff(importance.levels[0]) >= 0)
    assert importance.levels[0, -1] == 5


def test_surface_splitting_preserves_transparent_detector_score():
    problem = _transparent_problem()
    detector = BoundaryDetector(face="right")
    importance = make_right_boundary_importance_map(problem, n_levels=3, split_factor=2)
    analog = run_detector_mc(problem, 400, detector=detector, seed=17, importance=None)
    vr = run_detector_mc(problem, 400, detector=detector, seed=17, importance=importance)
    assert np.array_equal(analog.root_scores, vr.root_scores)
    assert np.allclose(analog.flux, vr.flux)
    assert np.allclose(analog.flux_score_sum_sq, vr.flux_score_sum_sq)
    assert np.array_equal(
        analog.flux_contributing_roots,
        vr.flux_contributing_roots,
    )
    assert np.allclose(analog.flux_sample_variance, vr.flux_sample_variance)
    assert vr.split_events > 0
    assert vr.split_children > 0


def test_root_history_statistics_are_finite_on_lattice():
    problem = make_lattice_problem(14, 14)
    importance = make_right_boundary_importance_map(problem, n_levels=3, split_factor=2)
    far_mask = np.zeros((problem.ny, problem.nx), dtype=bool)
    far_mask[:, -2:] = True
    result = run_detector_mc(
        problem,
        200,
        importance=importance,
        seed=23,
        tally_masks={"far": far_mask},
    )
    assert np.isfinite(result.estimate)
    assert np.isfinite(result.sample_variance)
    assert result.histories == len(result.root_scores)
    assert result.flux_sample_variance.shape == result.flux.shape
    assert result.flux_standard_error.shape == result.flux.shape
    assert result.flux_relative_error.shape == result.flux.shape
    assert result.region_scores["far"].shape == (result.histories,)
    assert np.mean(result.region_scores["far"]) == pytest.approx(
        float(np.mean(result.flux[far_mask]))
    )
    assert np.all(result.flux_sample_variance >= 0.0)
    assert np.all(result.flux_standard_error >= 0.0)
    assert 0.0 <= result.root_ess <= result.histories
    assert result.transported_particles >= result.histories


def test_diffusion_adjoint_importance_is_positive_and_detector_directed():
    problem = make_lattice_problem(14, 14)
    potential = solve_diffusion_adjoint_importance(problem)
    importance = make_diffusion_adjoint_importance_map(problem, n_levels=5)
    assert potential.shape == (problem.ny, problem.nx)
    assert np.all(np.isfinite(potential))
    assert np.all(potential > 0.0)
    assert np.max(potential[:, -1]) > np.max(potential[:, 0])
    assert np.all(importance.levels[:, -1] == 5)
    assert np.max(importance.levels) == 5


def test_external_adjoint_field_quantizes_like_detector_importance():
    problem = _transparent_problem()
    potential = np.broadcast_to(
        np.geomspace(1.0, 64.0, problem.nx),
        (problem.ny, problem.nx),
    ).copy()
    importance = make_adjoint_importance_map(
        problem,
        potential,
        detector=BoundaryDetector(face="right"),
        n_levels=6,
    )
    assert np.all(np.diff(importance.levels, axis=1) >= 0)
    assert np.all(importance.levels[:, 0] == 0)
    assert np.all(importance.levels[:, -1] == 6)


def test_response_absorber_hybrid_raises_low_flux_absorber_importance():
    problem = make_lattice_problem(28, 28)
    detector = BoundaryDetector(face="all")
    response = solve_diffusion_adjoint_importance(problem, detector)
    coarse_flux = np.ones((problem.ny, problem.nx), dtype=np.float64)
    absorber_mask = problem.sigma_a > 0.0
    coarse_flux[absorber_mask] = 1.0e-4
    combined, reciprocal = combine_response_and_absorber_importance(
        problem,
        response,
        coarse_flux,
        flux_exponent=0.2,
        absorber_peak_fraction=0.25,
        strong_absorber_ratio=0.8,
    )
    baseline = make_adjoint_importance_map(problem, response, detector, n_levels=6)
    hybrid = make_adjoint_importance_map(problem, combined, detector, n_levels=6)
    assert np.mean(hybrid.levels[absorber_mask]) > np.mean(
        baseline.levels[absorber_mask]
    )
    assert np.mean(reciprocal[absorber_mask]) > np.mean(reciprocal[~absorber_mask])


def test_all_boundary_importance_is_low_at_center_and_high_on_perimeter():
    problem = make_lattice_problem(28, 28)
    detector = BoundaryDetector(face="all")
    importance = make_diffusion_adjoint_importance_map(
        problem,
        detector=detector,
        n_levels=5,
    )
    source_ix = int(3.5 / problem.dx)
    source_iy = int(3.5 / problem.dy)
    perimeter = np.concatenate(
        [
            importance.levels[0],
            importance.levels[-1],
            importance.levels[1:-1, 0],
            importance.levels[1:-1, -1],
        ]
    )
    assert importance.levels[source_iy, source_ix] == 0
    assert np.all(perimeter == 5)


def test_geometric_all_boundary_importance_is_center_outward():
    problem = make_lattice_problem(28, 28)
    importance = make_all_boundary_importance_map(problem, n_levels=5)
    source_mask = importance.levels[12:16, 12:16]
    perimeter = np.concatenate(
        [
            importance.levels[0],
            importance.levels[-1],
            importance.levels[1:-1, 0],
            importance.levels[1:-1, -1],
        ]
    )
    assert np.all(source_mask == 0)
    assert np.all(perimeter == 5)
    assert np.array_equal(importance.levels, np.fliplr(importance.levels))
    assert np.array_equal(importance.levels, np.flipud(importance.levels))


def test_weight_windows_preserve_transparent_detector_score():
    problem = _transparent_problem()
    targets = np.broadcast_to(
        np.geomspace(1.0, 0.125, problem.nx),
        (problem.ny, problem.nx),
    ).copy()
    levels = np.rint(-np.log2(targets)).astype(np.int64)
    windows = WeightWindowMap(
        target_weights=targets,
        levels=levels,
        upper_ratio=1.1,
    )
    detector = BoundaryDetector(face="right")
    analog = run_detector_mc(problem, 200, detector=detector, seed=31)
    vr = run_detector_mc(
        problem,
        200,
        detector=detector,
        weight_windows=windows,
        seed=31,
    )
    assert np.allclose(analog.root_scores, vr.root_scores)
    assert vr.split_events > 0
    assert vr.split_children > 0


def test_weight_window_reports_when_split_limit_is_active():
    problem = _transparent_problem()
    targets = np.full((problem.ny, problem.nx), 1.0 / 64.0)
    windows = WeightWindowMap(
        target_weights=targets,
        levels=np.full((problem.ny, problem.nx), 6, dtype=np.int64),
        upper_ratio=1.1,
        max_split=4,
    )
    result = run_detector_mc(
        problem,
        20,
        detector=BoundaryDetector(face="right"),
        weight_windows=windows,
        seed=41,
    )
    assert result.split_cap_hits > 0
    assert int(np.sum(result.split_cap_hits_map)) == result.split_cap_hits


def test_inverse_flux_windows_prioritize_low_flux_absorbers():
    problem = make_lattice_problem(28, 28)
    absorber_mask = problem.sigma_a > 0.0
    coarse_flux = np.ones((problem.ny, problem.nx), dtype=np.float64)
    coarse_flux[absorber_mask] = 0.1
    prepared = prepare_coarse_flux_field(
        problem,
        coarse_flux,
        smoothing_sigma=0.0,
        symmetrize_x=True,
    )
    reciprocal_importance = make_reciprocal_flux_importance(
        problem,
        prepared,
        flux_exponent=1.0,
    )
    windows = make_inverse_flux_weight_windows(
        problem,
        prepared,
        detector=BoundaryDetector(face="all"),
        n_levels=3,
        flux_exponent=1.0,
        absorber_min_level=3,
        absorber_halo_fraction=0.04,
    )
    source_slice = windows.target_weights[12:16, 12:16]
    perimeter = np.concatenate(
        [
            windows.levels[0],
            windows.levels[-1],
            windows.levels[1:-1, 0],
            windows.levels[1:-1, -1],
        ]
    )
    assert prepared.shape == (problem.ny, problem.nx)
    assert prepare_coarse_flux_field(
        problem,
        np.ones((14, 14), dtype=np.float64),
        smoothing_sigma=0.0,
    ).shape == (problem.ny, problem.nx)
    assert np.all(source_slice == 1.0)
    assert np.mean(reciprocal_importance[absorber_mask]) > np.mean(
        reciprocal_importance[~absorber_mask]
    )
    assert np.mean(windows.target_weights[absorber_mask]) < np.mean(
        windows.target_weights[~absorber_mask]
    )
    assert np.all(windows.levels[absorber_mask] == 3)
    assert np.all(perimeter == 3)


def test_absorber_halo_uses_graded_presplitting_levels():
    nx = ny = 9
    sigma_a = np.zeros((ny, nx), dtype=np.float64)
    sigma_s = np.ones((ny, nx), dtype=np.float64)
    sigma_a[4, 4] = 9.0
    sigma_s[4, 4] = 1.0
    problem = Structured2DProblem(
        name="single_absorber",
        nx=nx,
        ny=ny,
        width=9.0,
        height=9.0,
        sigma_a=sigma_a,
        sigma_s=sigma_s,
        source_kind="volume_box",
        source_box=(0.0, 1.0, 0.0, 1.0),
    )
    windows = make_inverse_flux_weight_windows(
        problem,
        np.ones((ny, nx), dtype=np.float64),
        n_levels=4,
        flux_exponent=1.0,
        absorber_min_level=4,
        absorber_halo_fraction=2.0 / 9.0,
    )
    assert windows.levels[4, 4] == 4
    assert windows.levels[4, 3] == 3
    assert windows.levels[4, 2] == 2
    assert windows.levels[4, 1] == 0


def test_absorber_boost_preserves_inverse_flux_gradient():
    nx = ny = 7
    sigma_a = np.zeros((ny, nx), dtype=np.float64)
    sigma_s = np.ones((ny, nx), dtype=np.float64)
    sigma_a[3, 2:5] = 9.0
    sigma_s[3, 2:5] = 1.0
    problem = Structured2DProblem(
        name="absorber_gradient",
        nx=nx,
        ny=ny,
        width=7.0,
        height=7.0,
        sigma_a=sigma_a,
        sigma_s=sigma_s,
        source_kind="volume_box",
        source_box=(0.0, 1.0, 0.0, 1.0),
    )
    coarse_flux = np.ones((ny, nx), dtype=np.float64)
    coarse_flux[3, 2:5] = np.array([0.5, 0.25, 0.125])
    baseline = make_inverse_flux_weight_windows(
        problem,
        coarse_flux,
        n_levels=6,
        flux_exponent=1.0,
    )
    boosted = make_inverse_flux_weight_windows(
        problem,
        coarse_flux,
        n_levels=6,
        flux_exponent=1.0,
        absorber_boost_levels=2.0,
    )
    absorber = np.s_[3, 2:5]
    assert np.allclose(
        boosted.target_weights[absorber],
        baseline.target_weights[absorber] / 4.0,
    )
    assert np.all(np.diff(boosted.levels[absorber]) > 0)
