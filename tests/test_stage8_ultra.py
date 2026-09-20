import numpy as np
from gmc.response_operator_ultra import (
    UltraPhaseSpace,build_boundary_response_mc_ultra,
    build_boundary_response_ballistic_split_mc_ultra,operator_metrics,
    build_library,build_internal_response_mc_ultra,solve_global_source_iteration,
)
from gmc.benchmarks2d import make_lattice_problem

def test_phase_space_projection_conserves_weight():
    d=UltraPhaseSpace(2,4,8)
    for face in ('left','right','bottom','top'):
        u,v=d.direction_for_entry(face,0)
        a=d.project_angle(face,u,v)
        assert abs(sum(w for _,w in a)-1.0)<1e-12
        p=d.project_position(face,0.31,0.47,1.0,1.0)
        assert abs(sum(w for _,w in p)-1.0)<1e-12

def test_phase_space_rotation_preserves_state_count_and_changes_nodes():
    baseline=UltraPhaseSpace(2,2,16,phi_offset_fraction=0.0)
    rotated=UltraPhaseSpace(2,2,16,phi_offset_fraction=0.25)
    assert baseline.n_state==rotated.n_state
    assert np.isclose(baseline.angle_weights.sum(),rotated.angle_weights.sum())
    assert not np.allclose(baseline.angle_nodes,rotated.angle_nodes)

def test_pure_scatter_operator_conservation():
    d=UltraPhaseSpace(1,2,8)
    op=build_boundary_response_mc_ultra(1.0,0.0,0.25,0.25,d,samples_per_state=128,seed=1)
    assert np.max(np.abs(op.P.sum(axis=0)-1.0))<1e-12
    assert np.max(np.abs(op.R.sum(axis=0)-1.0))<1e-12
    assert np.all(op.track>0)

def test_ballistic_split_matches_direct_mc_statistically():
    d=UltraPhaseSpace(1,2,8)
    a=build_boundary_response_mc_ultra(1.0,0.0,0.25,0.25,d,samples_per_state=512,seed=2)
    b=build_boundary_response_ballistic_split_mc_ultra(1.0,0.0,0.25,0.25,d,scattered_samples_per_state=512,seed=3)
    m=operator_metrics(b,a)
    assert m['prob_l1_mean_per_input']<0.12
    assert m['track_rel_l1']<0.08

def test_global_expected_flux_has_no_empty_cells():
    p=make_lattice_problem(14,14); d=UltraPhaseSpace(1,2,8)
    lib=build_library(p,d,'mc',samples_per_state=64,seed=4)
    src=build_internal_response_mc_ultra(1,0,p.dx,p.dy,d,samples=512,seed=5)
    r=solve_global_source_iteration(p,lib,d,src,rtol=1e-6,max_iters=1000)
    assert r.converged
    assert r.nonzero_fraction==1.0
    assert np.all(np.isfinite(r.flux))
    assert np.all(r.flux>0)

def test_analytic_streaming_operator():
    from gmc.response_operator_ultra import build_boundary_response_streaming_ultra
    d=UltraPhaseSpace(1,2,8)
    op=build_boundary_response_streaming_ultra(0.0,1.0,1.0,d)
    assert np.max(np.abs(op.P.sum(axis=0)-1.0))<1e-12
    assert np.max(np.abs(op.R.sum(axis=0)-1.0))<1e-12
