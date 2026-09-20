"""Minimal Generative Monte Carlo reproduction package."""

from .mc_cell import (
    generate_boundary_dataset,
    generate_internal_dataset,
    load_npz,
    load_internal_npz,
    save_npz,
)
from .model import BoundaryVelocityNet, ModelConfig, sample_rk4, cfm_loss
from .transforms import BoundaryTransform, InternalTransform
from .checkpoints import DEFAULT_BOUNDARY_CHECKPOINT, DEFAULT_INTERNAL_CHECKPOINT
from .method_taxonomy import get_method_taxonomy
from .benchmarks2d import make_hohlraum_problem, make_lattice_problem
from .transport2d import (
    BoundaryGMCSampler,
    InternalGMCSampler,
    run_standard_mc,
    run_gmc_boundary_transport,
    run_gmc_full_transport,
)
from .importance_transport import (
    BoundaryDetector,
    combine_response_and_absorber_importance,
    DetectorTransportResult,
    ImportanceMap,
    WeightWindowMap,
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
from .response_matrix2d import ResponseSolveResult, solve_response_matrix, product_s2_quadrature
