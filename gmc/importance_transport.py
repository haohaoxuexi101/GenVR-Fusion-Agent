"""Standalone importance and weight-window Monte Carlo for detector tallies.

The implementation deliberately keeps the physical transport kernel identical to
``run_standard_mc``.  Variance reduction can use discrete cell-importance
interfaces or continuous target-weight windows.  All descendants are accumulated
back to their original root history before estimating the variance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.ndimage import (
    distance_transform_cdt,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .benchmarks2d import Structured2DProblem
from .mc_cell import sample_isotropic_direction, sample_left_boundary_direction
from .transport2d import _distance_to_cell_boundary, _nudge_after_face, _track_and_weight


BoundaryFace = Literal["left", "right", "bottom", "top"]
DetectorFace = Literal["left", "right", "bottom", "top", "all"]


@dataclass(frozen=True)
class BoundaryDetector:
    """Weighted leakage detector on one or all outer boundary segments."""

    face: DetectorFace = "right"
    lower: float | None = None
    upper: float | None = None

    def accepts(self, face: BoundaryFace, x: float, y: float) -> bool:
        if self.face != "all" and face != self.face:
            return False
        coordinate = y if face in ("left", "right") else x
        if self.lower is not None and coordinate < self.lower:
            return False
        if self.upper is not None and coordinate > self.upper:
            return False
        return True


@dataclass(frozen=True)
class ImportanceMap:
    """Integer cell importance levels used for splitting and roulette."""

    levels: np.ndarray
    split_factor: int = 2

    def validate(self, problem: Structured2DProblem) -> None:
        if self.levels.shape != (problem.ny, problem.nx):
            raise ValueError(
                f"importance map shape {self.levels.shape} does not match "
                f"problem shape {(problem.ny, problem.nx)}"
            )
        if not np.issubdtype(self.levels.dtype, np.integer):
            raise ValueError("importance levels must be integers")
        if np.any(self.levels < 0):
            raise ValueError("importance levels must be non-negative")
        if self.split_factor < 2:
            raise ValueError("split_factor must be at least 2")


@dataclass(frozen=True)
class WeightWindowMap:
    """Cellwise target particle weights for unbiased population control."""

    target_weights: np.ndarray
    levels: np.ndarray
    lower_ratio: float = 0.5
    upper_ratio: float = 2.0
    max_split: int = 32

    @property
    def importance(self) -> np.ndarray:
        return 1.0 / self.target_weights

    def validate(self, problem: Structured2DProblem) -> None:
        expected_shape = (problem.ny, problem.nx)
        if self.target_weights.shape != expected_shape:
            raise ValueError(
                f"weight-window shape {self.target_weights.shape} does not match "
                f"problem shape {expected_shape}"
            )
        if self.levels.shape != expected_shape:
            raise ValueError(
                f"weight-window level shape {self.levels.shape} does not match "
                f"problem shape {expected_shape}"
            )
        if not np.all(np.isfinite(self.target_weights)):
            raise ValueError("target weights must be finite")
        if np.any(self.target_weights <= 0.0):
            raise ValueError("target weights must be positive")
        if not np.issubdtype(self.levels.dtype, np.integer):
            raise ValueError("weight-window levels must be integers")
        if not 0.0 < self.lower_ratio < 1.0:
            raise ValueError("lower_ratio must lie between zero and one")
        if self.upper_ratio <= 1.0:
            raise ValueError("upper_ratio must be greater than one")
        if self.max_split < 2:
            raise ValueError("max_split must be at least two")


@dataclass
class DetectorTransportResult:
    """Root-history statistics for a detector Monte Carlo calculation."""

    flux: np.ndarray
    flux_score_sum: np.ndarray
    flux_score_sum_sq: np.ndarray
    flux_contributing_roots: np.ndarray
    flux_sample_variance: np.ndarray
    flux_standard_error: np.ndarray
    flux_relative_error: np.ndarray
    cell_visits: np.ndarray
    split_events_map: np.ndarray
    split_children_map: np.ndarray
    split_cap_hits_map: np.ndarray
    roulette_kills_map: np.ndarray
    root_scores: np.ndarray
    histories: int
    steps: int
    transported_particles: int
    detector_crossings: int
    leakage_crossings: int
    weighted_leakage: float
    absorbed_weight: float
    track_length: float
    split_events: int
    split_children: int
    split_cap_hits: int
    roulette_survivals: int
    roulette_kills: int
    cutoff_survivals: int
    cutoff_kills: int
    max_bank_size: int
    region_scores: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def estimate(self) -> float:
        return float(np.mean(self.root_scores))

    @property
    def sample_variance(self) -> float:
        if self.histories < 2:
            return float("nan")
        return float(np.var(self.root_scores, ddof=1))

    @property
    def standard_error(self) -> float:
        if self.histories < 2:
            return float("nan")
        return float(np.sqrt(self.sample_variance / self.histories))

    @property
    def relative_error(self) -> float:
        mean = abs(self.estimate)
        if mean == 0.0:
            return float("inf")
        return self.standard_error / mean

    @property
    def contributing_roots(self) -> int:
        return int(np.count_nonzero(self.root_scores))

    @property
    def zero_score_fraction(self) -> float:
        return 1.0 - self.contributing_roots / float(self.histories)

    @property
    def root_ess(self) -> float:
        denominator = float(np.dot(self.root_scores, self.root_scores))
        if denominator == 0.0:
            return 0.0
        numerator = float(np.sum(self.root_scores)) ** 2
        return numerator / denominator

    @property
    def max_root_fraction(self) -> float:
        total = float(np.sum(self.root_scores))
        if total <= 0.0:
            return 0.0
        return float(np.max(self.root_scores) / total)


def _source_progress_start(problem: Structured2DProblem) -> float:
    if problem.source_kind == "left_boundary":
        return 0.0
    if problem.source_box is None:
        return 0.0
    return float(problem.source_box[1])


def make_right_boundary_importance_map(
    problem: Structured2DProblem,
    n_levels: int = 5,
    split_factor: int = 2,
) -> ImportanceMap:
    """Construct monotone importance levels toward the right boundary.

    The source region remains at level zero.  The final level is reached in the
    rightmost cell, so the last split occurs before a possible detector crossing.
    This intentionally simple map is a transparent baseline that a future GenVR
    agent can replace with learned spatial, angular, or pathway importance.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    start_x = _source_progress_start(problem)
    centers = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    denominator = max(problem.width - start_x, problem.dx)
    progress = np.clip((centers - start_x) / denominator, 0.0, 1.0)
    column_levels = np.floor(progress * (n_levels + 1)).astype(np.int64)
    column_levels = np.minimum(column_levels, n_levels)
    column_levels[centers <= start_x] = 0
    column_levels[-1] = n_levels
    levels = np.broadcast_to(column_levels, (problem.ny, problem.nx)).copy()
    result = ImportanceMap(levels=levels, split_factor=split_factor)
    result.validate(problem)
    return result


def make_all_boundary_importance_map(
    problem: Structured2DProblem,
    n_levels: int = 5,
    split_factor: int = 2,
) -> ImportanceMap:
    """Construct center-outward levels for total leakage on all boundaries.

    The source box is assigned level zero.  Outside it, importance increases
    monotonically toward the nearest outer boundary, and every perimeter cell
    is assigned the highest level.  This is a geometry-only baseline; the
    diffusion-adjoint map remains preferable when material effects matter.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    if problem.source_kind != "volume_box" or problem.source_box is None:
        raise ValueError("all-boundary importance requires a volume_box source")

    xmin, xmax, ymin, ymax = problem.source_box
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)

    left_progress = np.clip((xmin - x_grid) / max(xmin, problem.dx), 0.0, 1.0)
    right_progress = np.clip(
        (x_grid - xmax) / max(problem.width - xmax, problem.dx),
        0.0,
        1.0,
    )
    bottom_progress = np.clip((ymin - y_grid) / max(ymin, problem.dy), 0.0, 1.0)
    top_progress = np.clip(
        (y_grid - ymax) / max(problem.height - ymax, problem.dy),
        0.0,
        1.0,
    )
    progress = np.maximum.reduce(
        (left_progress, right_progress, bottom_progress, top_progress)
    )
    levels = np.floor(progress * (n_levels + 1)).astype(np.int64)
    levels = np.minimum(levels, n_levels)
    levels[_source_cell_mask(problem)] = 0
    levels[_detector_cell_mask(problem, BoundaryDetector(face="all"))] = n_levels
    result = ImportanceMap(levels=levels, split_factor=split_factor)
    result.validate(problem)
    return result


def _source_cell_mask(problem: Structured2DProblem) -> np.ndarray:
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)
    if problem.source_kind == "volume_box":
        if problem.source_box is None:
            raise ValueError("volume_box source requires problem.source_box")
        xmin, xmax, ymin, ymax = problem.source_box
        return (
            (x_grid >= xmin)
            & (x_grid < xmax)
            & (y_grid >= ymin)
            & (y_grid < ymax)
        )
    if problem.source_kind == "left_boundary":
        yrange = problem.boundary_source_y_range
        ymin, ymax = (0.0, problem.height) if yrange is None else yrange
        mask = np.zeros((problem.ny, problem.nx), dtype=bool)
        mask[:, 0] = (centers_y >= ymin) & (centers_y <= ymax)
        return mask
    raise ValueError(f"unknown source_kind={problem.source_kind}")


def _detector_cell_mask(
    problem: Structured2DProblem,
    detector: BoundaryDetector,
) -> np.ndarray:
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    mask = np.zeros((problem.ny, problem.nx), dtype=bool)
    for iy, y in enumerate(centers_y):
        if detector.accepts("left", 0.0, float(y)):
            mask[iy, 0] = True
        if detector.accepts("right", problem.width, float(y)):
            mask[iy, -1] = True
    for ix, x in enumerate(centers_x):
        if detector.accepts("bottom", float(x), 0.0):
            mask[0, ix] = True
        if detector.accepts("top", float(x), problem.height):
            mask[-1, ix] = True
    return mask


def prepare_coarse_flux_field(
    problem: Structured2DProblem,
    coarse_flux: np.ndarray,
    floor_quantile: float = 0.005,
    smoothing_sigma: float = 1.25,
    symmetrize_x: bool = False,
) -> np.ndarray:
    """Resample and regularize a coarse GenVR flux field on the MC mesh."""
    values = np.asarray(coarse_flux, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("coarse_flux must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("coarse_flux must contain only finite values")
    if not 0.0 <= floor_quantile < 1.0:
        raise ValueError("floor_quantile must lie in [0, 1)")
    if smoothing_sigma < 0.0:
        raise ValueError("smoothing_sigma must be non-negative")

    source_ny, source_nx = values.shape
    if values.shape != (problem.ny, problem.nx):
        source_x = (np.arange(source_nx, dtype=np.float64) + 0.5) / source_nx
        source_y = (np.arange(source_ny, dtype=np.float64) + 0.5) / source_ny
        target_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) / problem.nx
        target_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) / problem.ny
        x_resampled = np.vstack(
            [np.interp(target_x, source_x, row) for row in values]
        )
        values = np.column_stack(
            [np.interp(target_y, source_y, x_resampled[:, ix]) for ix in range(problem.nx)]
        )

    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("coarse_flux must contain positive values")
    flux_floor = max(float(np.quantile(positive, floor_quantile)), 1.0e-300)
    log_flux = np.log(np.maximum(values, flux_floor))
    if smoothing_sigma > 0.0:
        log_flux = gaussian_filter(log_flux, sigma=smoothing_sigma, mode="nearest")
    prepared = np.exp(log_flux)
    if symmetrize_x:
        prepared = 0.5 * (prepared + np.fliplr(prepared))
    return np.maximum(prepared, flux_floor)


def make_reciprocal_flux_importance(
    problem: Structured2DProblem,
    coarse_flux: np.ndarray,
    flux_exponent: float = 1.0,
) -> np.ndarray:
    """Return the auditable base importance ``I = (phi_source / phi)^alpha``."""
    values = np.asarray(coarse_flux, dtype=np.float64)
    if values.shape != (problem.ny, problem.nx):
        raise ValueError("coarse_flux must match the problem mesh")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("coarse_flux must be finite and positive")
    if flux_exponent <= 0.0:
        raise ValueError("flux_exponent must be positive")
    source_mask = _source_cell_mask(problem)
    if not np.any(source_mask):
        raise RuntimeError("source mask does not contain any cells")
    source_reference = float(np.median(values[source_mask]))
    if not np.isfinite(source_reference) or source_reference <= 0.0:
        raise RuntimeError("source-region coarse flux must be positive")
    return np.power(
        source_reference / np.maximum(values, np.finfo(np.float64).tiny),
        flux_exponent,
    )


def combine_response_and_absorber_importance(
    problem: Structured2DProblem,
    response_importance: np.ndarray,
    coarse_flux: np.ndarray,
    flux_exponent: float = 0.2,
    absorber_peak_fraction: float = 0.125,
    strong_absorber_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a GenVR low-flux objective inside strong absorbers.

    The detector-response adjoint remains the baseline everywhere.  Only cells
    classified as strong absorbers receive a second importance component, whose
    shape is taken from reciprocal forward flux and whose peak is expressed as a
    fraction of the normalized detector importance.
    """
    response = np.asarray(response_importance, dtype=np.float64)
    if response.shape != (problem.ny, problem.nx):
        raise ValueError("response_importance must match the problem mesh")
    if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
        raise ValueError("response_importance must be finite and strictly positive")
    if not 0.0 < absorber_peak_fraction <= 1.0:
        raise ValueError("absorber_peak_fraction must lie in (0, 1]")
    if not 0.0 <= strong_absorber_ratio <= 1.0:
        raise ValueError("strong_absorber_ratio must lie in [0, 1]")

    reciprocal = make_reciprocal_flux_importance(
        problem,
        coarse_flux,
        flux_exponent=flux_exponent,
    )
    sigma_total = problem.sigma_a + problem.sigma_s
    absorption_ratio = np.divide(
        problem.sigma_a,
        sigma_total,
        out=np.zeros_like(problem.sigma_a),
        where=sigma_total > 0.0,
    )
    strong_absorber_mask = absorption_ratio >= strong_absorber_ratio
    if not np.any(strong_absorber_mask):
        raise RuntimeError("no cells satisfy the strong-absorber criterion")

    normalized_response = response / float(np.max(response))
    absorber_shape = reciprocal / max(
        float(np.max(reciprocal[strong_absorber_mask])),
        np.finfo(np.float64).tiny,
    )
    combined = normalized_response.copy()
    combined[strong_absorber_mask] = np.maximum(
        combined[strong_absorber_mask],
        absorber_peak_fraction * absorber_shape[strong_absorber_mask],
    )
    return combined, reciprocal


def make_inverse_flux_weight_windows(
    problem: Structured2DProblem,
    coarse_flux: np.ndarray,
    detector: BoundaryDetector | None = None,
    response_importance: np.ndarray | None = None,
    n_levels: int = 4,
    split_factor: int = 2,
    flux_exponent: float = 0.5,
    response_exponent: float = 0.0,
    contrast_exponent: float = 0.0,
    contrast_scale_fraction: float = 0.05,
    halo_fraction: float = 0.0,
    strong_absorber_ratio: float = 0.8,
    absorber_boost_levels: float = 0.0,
    absorber_min_level: int = 0,
    absorber_halo_fraction: float = 0.0,
    enforce_monotone_to_detector: bool = False,
    lower_ratio: float = 0.5,
    upper_ratio: float = 2.0,
    max_split: int = 32,
) -> WeightWindowMap:
    """Build target weights from a coarse forward flux estimate.

    The target population weight follows ``w_target ∝ coarse_flux**alpha``;
    therefore the reciprocal importance follows ``I ∝ coarse_flux**(-alpha)``.
    The finite level cap limits population growth while preserving this ordering.
    Strong absorbers can be assigned a minimum level.  Their optional halo is
    graded down by one level per mesh layer so pre-splitting does not turn a
    large moderator region into a maximum-importance plateau.
    """
    if coarse_flux.shape != (problem.ny, problem.nx):
        raise ValueError("coarse_flux must already be resampled to the problem mesh")
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    if split_factor < 2:
        raise ValueError("split_factor must be at least two")
    if flux_exponent <= 0.0:
        raise ValueError("flux_exponent must be positive")
    if response_exponent < 0.0:
        raise ValueError("response_exponent must be non-negative")
    if contrast_exponent < 0.0:
        raise ValueError("contrast_exponent must be non-negative")
    if contrast_scale_fraction <= 0.0:
        raise ValueError("contrast_scale_fraction must be positive")
    if halo_fraction < 0.0:
        raise ValueError("halo_fraction must be non-negative")
    if not 0.0 <= strong_absorber_ratio <= 1.0:
        raise ValueError("strong_absorber_ratio must lie in [0, 1]")
    if absorber_boost_levels < 0.0:
        raise ValueError("absorber_boost_levels must be non-negative")
    if not 0 <= absorber_min_level <= n_levels:
        raise ValueError("absorber_min_level must lie between zero and n_levels")
    if absorber_halo_fraction < 0.0:
        raise ValueError("absorber_halo_fraction must be non-negative")

    source_mask = _source_cell_mask(problem)
    if not np.any(source_mask):
        raise RuntimeError("source mask does not contain any cells")
    minimum_weight = float(split_factor) ** (-n_levels)
    base_importance = make_reciprocal_flux_importance(
        problem,
        coarse_flux,
        flux_exponent=flux_exponent,
    )
    target_weights = 1.0 / base_importance
    if response_importance is not None and response_exponent > 0.0:
        response = np.asarray(response_importance, dtype=np.float64)
        if response.shape != coarse_flux.shape:
            raise ValueError("response_importance shape must match coarse_flux")
        if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
            raise ValueError("response_importance must be finite and positive")
        source_response = float(np.median(response[source_mask]))
        response_ratio = np.maximum(
            response
            / max(
                source_response,
                np.finfo(np.float64).tiny,
            ),
            1.0,
        )
        target_weights /= np.power(
            np.maximum(response_ratio, np.finfo(np.float64).tiny),
            response_exponent,
        )
    if contrast_exponent > 0.0:
        sigma = (
            max(0.5, contrast_scale_fraction * problem.ny),
            max(0.5, contrast_scale_fraction * problem.nx),
        )
        local_background = np.exp(
            gaussian_filter(np.log(coarse_flux), sigma=sigma, mode="nearest")
        )
        local_depression = np.clip(
            coarse_flux / np.maximum(local_background, np.finfo(np.float64).tiny),
            np.finfo(np.float64).tiny,
            1.0,
        )
        target_weights *= np.power(local_depression, contrast_exponent)
    target_weights = np.clip(target_weights, minimum_weight, 1.0)
    if detector is not None:
        target_weights[_detector_cell_mask(problem, detector)] = minimum_weight
    if halo_fraction > 0.0:
        halo_y = max(1, int(round(halo_fraction * problem.ny)))
        halo_x = max(1, int(round(halo_fraction * problem.nx)))
        target_weights = minimum_filter(
            target_weights,
            size=(2 * halo_y + 1, 2 * halo_x + 1),
            mode="nearest",
        )
    if absorber_boost_levels > 0.0 or absorber_min_level > 0:
        sigma_total = problem.sigma_a + problem.sigma_s
        absorption_ratio = problem.sigma_a / np.maximum(
            sigma_total,
            np.finfo(np.float64).tiny,
        )
        strong_absorber_mask = absorption_ratio >= strong_absorber_ratio
        if absorber_boost_levels > 0.0 and np.any(strong_absorber_mask):
            target_weights[strong_absorber_mask] *= float(split_factor) ** (
                -absorber_boost_levels
            )
            target_weights[strong_absorber_mask] = np.maximum(
                target_weights[strong_absorber_mask],
                minimum_weight,
            )
        if np.any(strong_absorber_mask):
            absorber_target_ceiling = float(split_factor) ** (-absorber_min_level)
            target_weights[strong_absorber_mask] = np.minimum(
                target_weights[strong_absorber_mask],
                absorber_target_ceiling,
            )
        if (
            absorber_min_level > 0
            and absorber_halo_fraction > 0.0
            and np.any(strong_absorber_mask)
        ):
            halo_y = max(1, int(round(absorber_halo_fraction * problem.ny)))
            halo_x = max(1, int(round(absorber_halo_fraction * problem.nx)))
            absorber_halo_mask = maximum_filter(
                strong_absorber_mask.astype(np.uint8),
                size=(2 * halo_y + 1, 2 * halo_x + 1),
                mode="nearest",
            ).astype(bool)
            absorber_halo_mask &= ~strong_absorber_mask
            distance_to_absorber = distance_transform_cdt(
                ~strong_absorber_mask,
                metric="chessboard",
            )
            halo_levels = np.maximum(
                absorber_min_level - distance_to_absorber,
                0,
            )
            active_halo = absorber_halo_mask & (halo_levels > 0)
            target_weights[active_halo] = np.minimum(
                target_weights[active_halo],
                np.power(
                    float(split_factor),
                    -halo_levels[active_halo].astype(np.float64),
                ),
            )
    if enforce_monotone_to_detector and detector is not None:
        if detector.face == "right":
            target_weights = np.minimum.accumulate(target_weights, axis=1)
        elif detector.face == "left":
            target_weights = np.minimum.accumulate(
                target_weights[:, ::-1],
                axis=1,
            )[:, ::-1]
        elif detector.face == "top":
            target_weights = np.minimum.accumulate(target_weights, axis=0)
        elif detector.face == "bottom":
            target_weights = np.minimum.accumulate(
                target_weights[::-1, :],
                axis=0,
            )[::-1, :]
    target_weights[source_mask] = 1.0

    continuous_levels = -np.log(target_weights) / np.log(float(split_factor))
    levels = np.rint(continuous_levels).astype(np.int64)
    levels = np.clip(levels, 0, n_levels)
    result = WeightWindowMap(
        target_weights=target_weights,
        levels=levels,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        max_split=max_split,
    )
    result.validate(problem)
    return result


def solve_diffusion_adjoint_importance(
    problem: Structured2DProblem,
    detector: BoundaryDetector | None = None,
) -> np.ndarray:
    """Solve a finite-volume diffusion surrogate for detector importance.

    The detector boundary is assigned unit adjoint value and every other outer
    boundary zero.  This is not an exact transport adjoint; it is a cheap,
    material-aware spatial surrogate used only to decide unbiased splitting and
    roulette.  The exact Monte Carlo kernel and detector tally remain unchanged.
    """
    detector = detector or BoundaryDetector(face="right")

    sigma_t = problem.sigma_a + problem.sigma_s
    diffusion = 1.0 / np.maximum(3.0 * sigma_t, 1.0e-12)
    n_cells = problem.nx * problem.ny
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.zeros(n_cells, dtype=np.float64)
    cell_volume = problem.volume

    def cell_id(ix: int, iy: int) -> int:
        return iy * problem.nx + ix

    for iy in range(problem.ny):
        center_y = (iy + 0.5) * problem.dy
        for ix in range(problem.nx):
            center_x = (ix + 0.5) * problem.dx
            row = cell_id(ix, iy)
            diagonal = float(problem.sigma_a[iy, ix] * cell_volume)
            local_diffusion = float(diffusion[iy, ix])

            for dix, diy, face_length, center_distance, boundary_face in (
                (-1, 0, problem.dy, problem.dx, "left"),
                (1, 0, problem.dy, problem.dx, "right"),
                (0, -1, problem.dx, problem.dy, "bottom"),
                (0, 1, problem.dx, problem.dy, "top"),
            ):
                neighbor_x = ix + dix
                neighbor_y = iy + diy
                if 0 <= neighbor_x < problem.nx and 0 <= neighbor_y < problem.ny:
                    neighbor_diffusion = float(diffusion[neighbor_y, neighbor_x])
                    harmonic_diffusion = (
                        2.0
                        * local_diffusion
                        * neighbor_diffusion
                        / max(local_diffusion + neighbor_diffusion, 1.0e-300)
                    )
                    conductance = harmonic_diffusion * face_length / center_distance
                    diagonal += conductance
                    rows.append(row)
                    columns.append(cell_id(neighbor_x, neighbor_y))
                    values.append(-conductance)
                    continue

                boundary_distance = 0.5 * center_distance
                conductance = local_diffusion * face_length / boundary_distance
                diagonal += conductance
                boundary_x = center_x
                boundary_y = center_y
                if boundary_face == "left":
                    boundary_x = 0.0
                elif boundary_face == "right":
                    boundary_x = problem.width
                elif boundary_face == "bottom":
                    boundary_y = 0.0
                else:
                    boundary_y = problem.height
                if detector.accepts(boundary_face, boundary_x, boundary_y):
                    rhs[row] += conductance

            rows.append(row)
            columns.append(row)
            values.append(diagonal)

    matrix = csr_matrix((values, (rows, columns)), shape=(n_cells, n_cells))
    potential = np.asarray(spsolve(matrix, rhs), dtype=np.float64).reshape(
        problem.ny, problem.nx
    )
    if not np.all(np.isfinite(potential)) or np.max(potential) <= 0.0:
        raise RuntimeError("diffusion adjoint solve produced an invalid importance field")
    return np.maximum(potential, np.finfo(np.float64).tiny)


def make_diffusion_adjoint_importance_map(
    problem: Structured2DProblem,
    detector: BoundaryDetector | None = None,
    n_levels: int = 8,
    split_factor: int = 2,
) -> ImportanceMap:
    """Quantize a material-aware diffusion adjoint into splitting levels."""
    detector = detector or BoundaryDetector(face="right")
    potential = solve_diffusion_adjoint_importance(problem, detector)
    return make_adjoint_importance_map(
        problem,
        potential,
        detector=detector,
        n_levels=n_levels,
        split_factor=split_factor,
    )


def make_adjoint_importance_map(
    problem: Structured2DProblem,
    potential: np.ndarray,
    detector: BoundaryDetector | None = None,
    n_levels: int = 8,
    split_factor: int = 2,
) -> ImportanceMap:
    """Quantize any positive detector-adjoint proxy into splitting levels."""
    if n_levels < 1:
        raise ValueError("n_levels must be positive")
    detector = detector or BoundaryDetector(face="right")
    potential = np.asarray(potential, dtype=np.float64)
    if potential.shape != (problem.ny, problem.nx):
        raise ValueError("adjoint potential must match the problem mesh")
    if not np.all(np.isfinite(potential)) or np.any(potential <= 0.0):
        raise ValueError("adjoint potential must be finite and strictly positive")
    source_mask = _source_cell_mask(problem)
    detector_mask = _detector_cell_mask(problem, detector)
    if not np.any(source_mask):
        raise RuntimeError("source mask does not contain any cells")
    if not np.any(detector_mask):
        raise RuntimeError("detector does not overlap any boundary cells")

    source_reference = float(np.max(potential[source_mask]))
    detector_reference = float(np.max(potential[detector_mask]))
    if detector_reference <= source_reference:
        raise RuntimeError("detector importance is not greater than source-region importance")

    log_potential = np.log(potential)
    log_source = np.log(max(source_reference, np.finfo(np.float64).tiny))
    log_detector = np.log(max(detector_reference, np.finfo(np.float64).tiny))
    progress = np.clip(
        (log_potential - log_source) / max(log_detector - log_source, 1.0e-12),
        0.0,
        1.0,
    )
    levels = np.floor(progress * (n_levels + 1)).astype(np.int64)
    levels = np.minimum(levels, n_levels)
    levels[source_mask] = 0
    levels[detector_mask] = n_levels
    result = ImportanceMap(levels=levels, split_factor=split_factor)
    result.validate(problem)
    return result


def _initial_particle(
    problem: Structured2DProblem,
    rng: np.random.Generator,
    direction_mode: str,
) -> tuple[float, float, float, float]:
    if problem.source_kind == "left_boundary":
        yrange = problem.boundary_source_y_range
        ymin, ymax = (0.0, problem.height) if yrange is None else yrange
        y = float(rng.uniform(ymin, ymax))
        direction = sample_left_boundary_direction(rng, 1, mode=direction_mode)[0]
        return 1.0e-12, y, float(direction[0]), float(direction[1])
    if problem.source_kind == "volume_box":
        if problem.source_box is None:
            raise ValueError("volume_box source requires problem.source_box")
        xmin, xmax, ymin, ymax = problem.source_box
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        direction = sample_isotropic_direction(rng, 1)[0]
        return x, y, float(direction[0]), float(direction[1])
    raise ValueError(f"unknown source_kind={problem.source_kind}")


def _apply_weight_cutoff(
    weight: float,
    cutoff: float,
    rng: np.random.Generator,
) -> tuple[float, bool, bool]:
    if cutoff <= 0.0 or weight >= cutoff:
        return weight, True, False
    survival_probability = weight / cutoff
    if rng.random() < survival_probability:
        return cutoff, True, True
    return 0.0, False, False


def _apply_weight_window(
    weight: float,
    target_weight: float,
    window: WeightWindowMap,
    rng: np.random.Generator,
) -> tuple[float, bool, int, bool, bool]:
    if weight > window.upper_ratio * target_weight:
        requested_multiplicity = max(2, int(np.ceil(weight / target_weight)))
        multiplicity = min(window.max_split, requested_multiplicity)
        return (
            weight / float(multiplicity),
            True,
            multiplicity - 1,
            False,
            requested_multiplicity > window.max_split,
        )
    if weight < window.lower_ratio * target_weight:
        survival_probability = min(1.0, weight / target_weight)
        if rng.random() < survival_probability:
            return target_weight, True, 0, True, False
        return 0.0, False, 0, False, False
    return weight, True, 0, False, False


def run_detector_mc(
    problem: Structured2DProblem,
    n_particles: int,
    detector: BoundaryDetector | None = None,
    importance: ImportanceMap | None = None,
    weight_windows: WeightWindowMap | None = None,
    seed: int = 1,
    direction_mode: str = "cosine",
    weight_cutoff: float = 1.0e-12,
    max_steps_per_particle: int = 200_000,
    max_particles_per_root: int = 1_000_000,
    tally_masks: dict[str, np.ndarray] | None = None,
) -> DetectorTransportResult:
    """Run analog, importance-splitting, or weight-window detector MC.

    Each source history starts with unit weight.  Descendant scores are summed
    into one root score, and only those root scores are treated as independent
    samples.  Passing both variance-reduction arguments as ``None`` gives the
    no-variance-reduction baseline with the same transport and tally kernel.
    """
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    if importance is not None and weight_windows is not None:
        raise ValueError("importance and weight_windows are mutually exclusive")
    detector = detector or BoundaryDetector()
    if importance is not None:
        importance.validate(problem)
    if weight_windows is not None:
        weight_windows.validate(problem)
    validated_tally_masks: dict[str, np.ndarray] = {}
    for name, raw_mask in (tally_masks or {}).items():
        if not name:
            raise ValueError("tally mask names must be non-empty")
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (problem.ny, problem.nx):
            raise ValueError(
                f"tally mask {name!r} shape {mask.shape} does not match "
                f"problem shape {(problem.ny, problem.nx)}"
            )
        if not np.any(mask):
            raise ValueError(f"tally mask {name!r} contains no cells")
        validated_tally_masks[name] = mask
    region_scores = {
        name: np.zeros(n_particles, dtype=np.float64)
        for name in validated_tally_masks
    }

    flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    flux_score_sum_sq = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    flux_contributing_roots = np.zeros((problem.ny, problem.nx), dtype=np.int64)
    root_flux_tl = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    cell_visits = np.zeros((problem.ny, problem.nx), dtype=np.int64)
    split_events_map = np.zeros((problem.ny, problem.nx), dtype=np.int64)
    split_children_map = np.zeros((problem.ny, problem.nx), dtype=np.int64)
    split_cap_hits_map = np.zeros((problem.ny, problem.nx), dtype=np.int64)
    roulette_kills_map = np.zeros((problem.ny, problem.nx), dtype=np.int64)
    root_scores = np.zeros(n_particles, dtype=np.float64)
    total_steps = 0
    transported_particles = 0
    detector_crossings = 0
    leakage_crossings = 0
    weighted_leakage = 0.0
    absorbed_weight = 0.0
    total_tl = 0.0
    split_events = 0
    split_children = 0
    split_cap_hits = 0
    roulette_survivals = 0
    roulette_kills = 0
    cutoff_survivals = 0
    cutoff_kills = 0
    maximum_bank_size = 1

    for root_id in range(n_particles):
        root_flux_tl.fill(0.0)
        rng = np.random.default_rng(np.random.SeedSequence([seed, root_id]))
        x, y, u, v = _initial_particle(problem, rng, direction_mode)
        initial_index = problem.cell_indices(x, y)
        if initial_index is None:
            raise RuntimeError("source particle was born outside the problem")
        initial_ix, initial_iy = initial_index
        initial_level = 0 if importance is None else int(importance.levels[initial_iy, initial_ix])
        bank: list[tuple[float, float, float, float, float, int]] = [
            (x, y, u, v, 1.0, initial_level)
        ]
        created_for_root = 1

        while bank:
            maximum_bank_size = max(maximum_bank_size, len(bank))
            x, y, u, v, weight, current_level = bank.pop()
            transported_particles += 1

            weight, alive, cutoff_survived = _apply_weight_cutoff(weight, weight_cutoff, rng)
            if not alive:
                cutoff_kills += 1
                continue
            if cutoff_survived:
                cutoff_survivals += 1

            for _step in range(max_steps_per_particle):
                total_steps += 1
                index = problem.cell_indices(x, y)
                if index is None:
                    raise RuntimeError("particle escaped without a recorded boundary crossing")
                ix, iy = index
                cell_visits[iy, ix] += 1
                x0 = ix * problem.dx
                y0 = iy * problem.dy
                xloc = min(problem.dx, max(0.0, x - x0))
                yloc = min(problem.dy, max(0.0, y - y0))
                sigma_s = float(problem.sigma_s[iy, ix])
                sigma_a = float(problem.sigma_a[iy, ix])
                collision_distance = float(rng.exponential(1.0 / sigma_s)) if sigma_s > 0.0 else np.inf
                boundary_distance, face = _distance_to_cell_boundary(
                    xloc, yloc, u, v, problem.dx, problem.dy
                )
                distance = min(collision_distance, boundary_distance)
                track, absorbed, weight = _track_and_weight(weight, sigma_a, distance)
                flux_tl[iy, ix] += track
                root_flux_tl[iy, ix] += track
                absorbed_weight += absorbed
                total_tl += track
                x += distance * u
                y += distance * v

                if collision_distance < boundary_distance:
                    direction = sample_isotropic_direction(rng, 1)[0]
                    u, v = float(direction[0]), float(direction[1])
                    if weight_windows is not None:
                        (
                            weight,
                            alive,
                            new_children,
                            roulette_survived,
                            split_was_capped,
                        ) = _apply_weight_window(
                            weight,
                            float(weight_windows.target_weights[iy, ix]),
                            weight_windows,
                            rng,
                        )
                        if split_was_capped:
                            split_cap_hits += 1
                            split_cap_hits_map[iy, ix] += 1
                        if new_children:
                            created_for_root += new_children
                            if created_for_root > max_particles_per_root:
                                raise RuntimeError(
                                    "weight-window splitting exceeded max_particles_per_root; "
                                    "reduce n_levels or max_split"
                                )
                            bank.extend(
                                (x, y, u, v, weight, current_level)
                                for _ in range(new_children)
                            )
                            split_events += 1
                            split_children += new_children
                            split_events_map[iy, ix] += 1
                            split_children_map[iy, ix] += new_children
                        if not alive:
                            roulette_kills += 1
                            roulette_kills_map[iy, ix] += 1
                            break
                        if roulette_survived:
                            roulette_survivals += 1
                    weight, alive, cutoff_survived = _apply_weight_cutoff(weight, weight_cutoff, rng)
                    if not alive:
                        cutoff_kills += 1
                        break
                    if cutoff_survived:
                        cutoff_survivals += 1
                    continue

                x, y = _nudge_after_face(x, y, face)
                next_index = problem.cell_indices(x, y)
                if next_index is None:
                    leakage_crossings += 1
                    weighted_leakage += weight
                    if detector.accepts(face, x, y):
                        root_scores[root_id] += weight
                        detector_crossings += 1
                    break

                next_ix, next_iy = next_index
                next_level = 0 if importance is None else int(importance.levels[next_iy, next_ix])
                level_delta = next_level - current_level

                if weight_windows is not None:
                    (
                        weight,
                        alive,
                        new_children,
                        roulette_survived,
                        split_was_capped,
                    ) = _apply_weight_window(
                        weight,
                        float(weight_windows.target_weights[next_iy, next_ix]),
                        weight_windows,
                        rng,
                    )
                    if split_was_capped:
                        split_cap_hits += 1
                        split_cap_hits_map[next_iy, next_ix] += 1
                    if new_children:
                        created_for_root += new_children
                        if created_for_root > max_particles_per_root:
                            raise RuntimeError(
                                "weight-window splitting exceeded max_particles_per_root; "
                                "reduce n_levels or max_split"
                            )
                        bank.extend(
                            (x, y, u, v, weight, next_level)
                            for _ in range(new_children)
                        )
                        split_events += 1
                        split_children += new_children
                        split_events_map[next_iy, next_ix] += 1
                        split_children_map[next_iy, next_ix] += new_children
                    if not alive:
                        roulette_kills += 1
                        roulette_kills_map[next_iy, next_ix] += 1
                        break
                    if roulette_survived:
                        roulette_survivals += 1
                elif importance is not None and level_delta > 0:
                    multiplicity = importance.split_factor ** level_delta
                    child_weight = weight / float(multiplicity)
                    new_children = multiplicity - 1
                    created_for_root += new_children
                    if created_for_root > max_particles_per_root:
                        raise RuntimeError(
                            "importance splitting exceeded max_particles_per_root; "
                            "reduce n_levels or split_factor"
                        )
                    bank.extend(
                        (x, y, u, v, child_weight, next_level)
                        for _ in range(new_children)
                    )
                    weight = child_weight
                    split_events += 1
                    split_children += new_children
                    split_events_map[next_iy, next_ix] += 1
                    split_children_map[next_iy, next_ix] += new_children
                elif importance is not None and level_delta < 0:
                    survival_probability = importance.split_factor ** level_delta
                    if rng.random() >= survival_probability:
                        roulette_kills += 1
                        roulette_kills_map[next_iy, next_ix] += 1
                        break
                    weight /= survival_probability
                    roulette_survivals += 1

                current_level = next_level
            else:
                raise RuntimeError(
                    "particle exceeded max_steps_per_particle; increase the limit or inspect the geometry"
                )

        root_flux_score = root_flux_tl / problem.volume
        flux_score_sum_sq += root_flux_score * root_flux_score
        flux_contributing_roots += root_flux_score > 0.0
        for name, mask in validated_tally_masks.items():
            region_scores[name][root_id] = float(np.mean(root_flux_score[mask]))

    normalization = float(n_particles)
    flux_score_sum = flux_tl / problem.volume
    flux = flux_score_sum / normalization
    if n_particles > 1:
        centered_sum_sq = flux_score_sum_sq - flux_score_sum * flux_score_sum / normalization
        flux_sample_variance = np.maximum(centered_sum_sq / (normalization - 1.0), 0.0)
        flux_standard_error = np.sqrt(flux_sample_variance / normalization)
    else:
        flux_sample_variance = np.full_like(flux, np.nan)
        flux_standard_error = np.full_like(flux, np.nan)
    flux_relative_error = np.full_like(flux, np.inf)
    np.divide(
        flux_standard_error,
        np.abs(flux),
        out=flux_relative_error,
        where=flux != 0.0,
    )
    return DetectorTransportResult(
        flux=flux,
        flux_score_sum=flux_score_sum,
        flux_score_sum_sq=flux_score_sum_sq,
        flux_contributing_roots=flux_contributing_roots,
        flux_sample_variance=flux_sample_variance,
        flux_standard_error=flux_standard_error,
        flux_relative_error=flux_relative_error,
        cell_visits=cell_visits,
        split_events_map=split_events_map,
        split_children_map=split_children_map,
        split_cap_hits_map=split_cap_hits_map,
        roulette_kills_map=roulette_kills_map,
        root_scores=root_scores,
        histories=n_particles,
        steps=total_steps,
        transported_particles=transported_particles,
        detector_crossings=detector_crossings,
        leakage_crossings=leakage_crossings,
        weighted_leakage=weighted_leakage / normalization,
        absorbed_weight=absorbed_weight / normalization,
        track_length=total_tl / normalization,
        split_events=split_events,
        split_children=split_children,
        split_cap_hits=split_cap_hits,
        roulette_survivals=roulette_survivals,
        roulette_kills=roulette_kills,
        cutoff_survivals=cutoff_survivals,
        cutoff_kills=cutoff_kills,
        max_bank_size=maximum_bank_size,
        region_scores=region_scores,
    )
