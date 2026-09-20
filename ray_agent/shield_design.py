from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol
import hashlib
import math

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from gmc.benchmarks2d import Structured2DProblem


MacroCell = tuple[int, int]

GRID_SIZE = 7
SOURCE_CELL: MacroCell = (3, 3)
BACKGROUND_SIGMA_S = 1.0
BACKGROUND_SIGMA_A = 0.0
ABSORBER_SIGMA_S = 0.5
ABSORBER_SIGMA_A = 9.5

OPERATORS = (
    "adjoint_sensitivity",
    "gmc_path_importance",
    "channel_block",
    "inner_barrier",
    "hotspot_cap",
    "symmetry_repair",
    "redistribute",
    "random_explore",
)


def _normalized_cells(cells: tuple[MacroCell, ...] | list[MacroCell]) -> tuple[MacroCell, ...]:
    return tuple(sorted((int(row), int(column)) for row, column in cells))


@dataclass(frozen=True)
class ShieldDesign:
    absorbers: tuple[MacroCell, ...]
    grid_size: int = GRID_SIZE
    source_cell: MacroCell = SOURCE_CELL

    def __post_init__(self) -> None:
        cells = _normalized_cells(self.absorbers)
        object.__setattr__(self, "absorbers", cells)
        if len(set(cells)) != len(cells):
            raise ValueError("absorber cells must be unique")
        if self.source_cell in cells:
            raise ValueError("the source cell cannot contain absorber material")
        for row, column in cells:
            if not 0 <= row < self.grid_size or not 0 <= column < self.grid_size:
                raise ValueError(f"absorber cell {(row, column)} lies outside the design grid")

    @property
    def absorber_count(self) -> int:
        return len(self.absorbers)

    @property
    def signature(self) -> str:
        payload = ";".join(f"{row},{column}" for row, column in self.absorbers)
        return hashlib.sha1(payload.encode("ascii")).hexdigest()[:12]

    def mask(self) -> np.ndarray:
        result = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for row, column in self.absorbers:
            result[row, column] = True
        return result

    def swap(self, remove: MacroCell, add: MacroCell) -> ShieldDesign:
        cells = set(self.absorbers)
        if remove not in cells:
            raise ValueError(f"cannot remove non-absorber cell {remove}")
        if add in cells or add == self.source_cell:
            raise ValueError(f"cannot add absorber at {add}")
        cells.remove(remove)
        cells.add(add)
        return ShieldDesign(tuple(cells), self.grid_size, self.source_cell)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "grid_size": self.grid_size,
            "source_cell": list(self.source_cell),
            "absorber_count": self.absorber_count,
            "absorbers": [list(cell) for cell in self.absorbers],
        }


def baseline_lattice_design() -> ShieldDesign:
    return ShieldDesign(
        (
            (5, 1),
            (5, 5),
            (4, 2),
            (4, 4),
            (3, 1),
            (3, 5),
            (2, 2),
            (2, 4),
            (1, 1),
            (1, 3),
            (1, 5),
        )
    )


def materialize_lattice_design(
    design: ShieldDesign,
    nx: int = 112,
    ny: int = 112,
) -> Structured2DProblem:
    if nx % design.grid_size or ny % design.grid_size:
        raise ValueError("nx and ny must be divisible by the macro design grid size")
    sigma_a = np.full((ny, nx), BACKGROUND_SIGMA_A, dtype=np.float64)
    sigma_s = np.full((ny, nx), BACKGROUND_SIGMA_S, dtype=np.float64)
    cells_y = ny // design.grid_size
    cells_x = nx // design.grid_size
    for row, column in design.absorbers:
        ys = slice(row * cells_y, (row + 1) * cells_y)
        xs = slice(column * cells_x, (column + 1) * cells_x)
        sigma_a[ys, xs] = ABSORBER_SIGMA_A
        sigma_s[ys, xs] = ABSORBER_SIGMA_S
    source_row, source_column = design.source_cell
    return Structured2DProblem(
        name=f"generated_shield_{design.signature}",
        nx=nx,
        ny=ny,
        width=float(design.grid_size),
        height=float(design.grid_size),
        sigma_a=sigma_a,
        sigma_s=sigma_s,
        source_kind="volume_box",
        source_box=(
            float(source_column),
            float(source_column + 1),
            float(source_row),
            float(source_row + 1),
        ),
    )


def source_distant_mask(problem: Structured2DProblem, quantile: float = 0.75) -> np.ndarray:
    if problem.source_kind != "volume_box" or problem.source_box is None:
        raise ValueError("source-distant masks require a volume-box source")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    xmin, xmax, ymin, ymax = problem.source_box
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)
    delta_x = np.maximum.reduce((xmin - x_grid, np.zeros_like(x_grid), x_grid - xmax))
    delta_y = np.maximum.reduce((ymin - y_grid, np.zeros_like(y_grid), y_grid - ymax))
    distance = np.hypot(delta_x, delta_y)
    non_source = distance > 0.0
    threshold = float(np.quantile(distance[non_source], quantile))
    return distance >= threshold


def source_distant_sector_masks(
    problem: Structured2DProblem,
    far_mask: np.ndarray | None = None,
    n_sectors: int = 8,
) -> tuple[np.ndarray, ...]:
    if problem.source_kind != "volume_box" or problem.source_box is None:
        raise ValueError("source-distant sectors require a volume-box source")
    if n_sectors < 2:
        raise ValueError("n_sectors must be at least two")
    selected = source_distant_mask(problem) if far_mask is None else np.asarray(far_mask)
    if selected.shape != (problem.ny, problem.nx):
        raise ValueError("far_mask shape does not match the problem mesh")
    xmin, xmax, ymin, ymax = problem.source_box
    source_x = 0.5 * (xmin + xmax)
    source_y = 0.5 * (ymin + ymax)
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)
    angles = np.mod(np.arctan2(y_grid - source_y, x_grid - source_x), 2.0 * math.pi)
    sector_ids = np.floor(angles * n_sectors / (2.0 * math.pi)).astype(np.int64)
    return tuple(selected & (sector_ids == index) for index in range(n_sectors))


def aggregate_macro_field(field: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    ny, nx = values.shape
    if nx % grid_size or ny % grid_size:
        raise ValueError("field shape must be divisible by grid_size")
    return values.reshape(
        grid_size,
        ny // grid_size,
        grid_size,
        nx // grid_size,
    ).mean(axis=(1, 3))


@dataclass(frozen=True)
class ShieldMetrics:
    far_mean: float
    far_p90: float
    far_cvar90: float
    far_max: float
    worst_sector_mean: float
    hotspot_ratio: float
    angular_imbalance: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_shield_metrics(
    flux: np.ndarray,
    far_mask: np.ndarray,
    tail_fraction: float = 0.10,
    sector_masks: tuple[np.ndarray, ...] | None = None,
) -> ShieldMetrics:
    values = np.asarray(flux, dtype=np.float64)[far_mask]
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size == 0:
        raise ValueError("far-field mask contains no finite non-negative flux values")
    ordered = np.sort(values)
    tail_count = max(1, int(math.ceil(tail_fraction * values.size)))
    far_mean = float(np.mean(values))
    far_max = float(ordered[-1])
    sector_means = []
    for sector_mask in sector_masks or ():
        mask = np.asarray(sector_mask, dtype=bool)
        if mask.shape != np.asarray(flux).shape:
            raise ValueError("sector mask shape does not match the flux field")
        sector_values = np.asarray(flux, dtype=np.float64)[mask]
        sector_values = sector_values[np.isfinite(sector_values) & (sector_values >= 0.0)]
        if sector_values.size:
            sector_means.append(float(np.mean(sector_values)))
    worst_sector_mean = max(sector_means, default=far_mean)
    mean_sector_mean = float(np.mean(sector_means)) if sector_means else far_mean
    return ShieldMetrics(
        far_mean=far_mean,
        far_p90=float(np.quantile(values, 0.90)),
        far_cvar90=float(np.mean(ordered[-tail_count:])),
        far_max=far_max,
        worst_sector_mean=worst_sector_mean,
        hotspot_ratio=far_max / max(far_mean, np.finfo(np.float64).tiny),
        angular_imbalance=(
            worst_sector_mean
            / max(mean_sector_mean, np.finfo(np.float64).tiny)
        ),
    )


DEFAULT_METRIC_WEIGHTS = {
    "far_cvar90": 0.40,
    "far_max": 0.20,
    "far_mean": 0.15,
    "worst_sector_mean": 0.20,
    "hotspot_ratio": 0.025,
    "angular_imbalance": 0.025,
}


def normalized_shield_risk(
    metrics: ShieldMetrics,
    baseline: ShieldMetrics,
    weights: dict[str, float] | None = None,
) -> float:
    selected = weights or DEFAULT_METRIC_WEIGHTS
    total_weight = float(sum(selected.values()))
    if total_weight <= 0.0:
        raise ValueError("metric weights must have a positive sum")
    risk = 0.0
    for name, weight in selected.items():
        candidate_value = float(getattr(metrics, name))
        baseline_value = max(
            float(getattr(baseline, name)),
            np.finfo(np.float64).tiny,
        )
        risk += float(weight) * candidate_value / baseline_value
    return risk / total_weight


@dataclass
class ShieldEvaluation:
    design: ShieldDesign
    flux: np.ndarray
    metrics: ShieldMetrics
    ensemble_metrics: tuple[ShieldMetrics, ...]
    runtime_s: float
    fidelity: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def robust_risk(
        self,
        baseline: ShieldEvaluation,
        weights: dict[str, float] | None = None,
        risk_aversion: float = 0.35,
    ) -> float:
        candidate_ensemble = self.ensemble_metrics or (self.metrics,)
        baseline_ensemble = baseline.ensemble_metrics or (baseline.metrics,)
        risks = []
        for index, candidate_metrics in enumerate(candidate_ensemble):
            baseline_metrics = baseline_ensemble[min(index, len(baseline_ensemble) - 1)]
            risks.append(normalized_shield_risk(candidate_metrics, baseline_metrics, weights))
        return float(np.mean(risks) + risk_aversion * np.std(risks))


class ShieldEvaluator(Protocol):
    fidelity: str

    def evaluate(self, design: ShieldDesign) -> ShieldEvaluation:
        ...


def _assemble_diffusion_operator(problem: Structured2DProblem) -> csr_matrix:
    ny, nx = problem.ny, problem.nx
    count = nx * ny
    dx, dy = problem.dx, problem.dy
    sigma_total = problem.sigma_a + problem.sigma_s
    diffusion = 1.0 / np.maximum(3.0 * sigma_total, 1.0e-12)
    matrix = lil_matrix((count, count), dtype=np.float64)

    def harmonic(left: float, right: float) -> float:
        return 2.0 * left * right / max(left + right, 1.0e-300)

    for row in range(ny):
        for column in range(nx):
            index = row * nx + column
            diagonal = float(problem.sigma_a[row, column] * dx * dy)
            for delta_row, delta_column, face_length, spacing in (
                (-1, 0, dx, dy),
                (1, 0, dx, dy),
                (0, -1, dy, dx),
                (0, 1, dy, dx),
            ):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if 0 <= neighbor_row < ny and 0 <= neighbor_column < nx:
                    conductance = (
                        harmonic(
                            float(diffusion[row, column]),
                            float(diffusion[neighbor_row, neighbor_column]),
                        )
                        * face_length
                        / spacing
                    )
                    neighbor_index = neighbor_row * nx + neighbor_column
                    matrix[index, neighbor_index] = -conductance
                    diagonal += conductance
                else:
                    diagonal += (
                        2.0
                        * float(diffusion[row, column])
                        * face_length
                        / spacing
                    )
            matrix[index, index] = diagonal
    return matrix.tocsr()


def _volume_source(problem: Structured2DProblem) -> np.ndarray:
    if problem.source_box is None:
        raise ValueError("diffusion screening requires a volume source")
    xmin, xmax, ymin, ymax = problem.source_box
    centers_x = (np.arange(problem.nx, dtype=np.float64) + 0.5) * problem.dx
    centers_y = (np.arange(problem.ny, dtype=np.float64) + 0.5) * problem.dy
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)
    source_mask = (
        (x_grid >= xmin)
        & (x_grid < xmax)
        & (y_grid >= ymin)
        & (y_grid < ymax)
    )
    source = np.zeros((problem.ny, problem.nx), dtype=np.float64)
    source[source_mask] = 1.0 / max(int(np.count_nonzero(source_mask)), 1)
    return source


def solve_far_field_adjoint(
    problem: Structured2DProblem,
    forward_flux: np.ndarray,
    tail_fraction: float = 0.10,
) -> np.ndarray:
    """Approximate the adjoint of a distributed far-field tail objective.

    The response source combines a uniform far-field term with extra weight on
    the currently hottest tail cells.  Multiplying this field by the forward
    flux gives the first-order absorption sensitivity used by the proposal model.
    """
    values = np.asarray(forward_flux, dtype=np.float64)
    if values.shape != (problem.ny, problem.nx):
        raise ValueError("forward_flux shape does not match the problem mesh")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")
    far_mask = source_distant_mask(problem)
    far_values = np.maximum(values[far_mask], 0.0)
    if far_values.size == 0:
        raise ValueError("far-field mask contains no cells")
    tail_threshold = float(np.quantile(far_values, 1.0 - tail_fraction))
    response = np.zeros_like(values)
    response[far_mask] = 0.25
    tail_mask = far_mask & (values >= tail_threshold)
    response[tail_mask] += 0.75
    response /= max(float(np.sum(response)), np.finfo(np.float64).tiny)
    adjoint = np.asarray(
        spsolve(_assemble_diffusion_operator(problem).T, response.ravel()),
        dtype=np.float64,
    ).reshape(problem.ny, problem.nx)
    return np.maximum(adjoint, 0.0)


class DiffusionShieldEvaluator:
    fidelity = "diffusion-ensemble"

    def __init__(
        self,
        nx: int = 56,
        ny: int = 56,
        absorber_uncertainty: tuple[float, ...] = (-0.05, 0.0, 0.05),
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.absorber_uncertainty = tuple(float(value) for value in absorber_uncertainty)

    @staticmethod
    def _solve(problem: Structured2DProblem) -> np.ndarray:
        matrix = _assemble_diffusion_operator(problem)
        source = _volume_source(problem)
        flux = np.asarray(
            spsolve(matrix, source.ravel()),
            dtype=np.float64,
        ).reshape(problem.ny, problem.nx)
        return np.maximum(flux, 0.0)

    def evaluate(self, design: ShieldDesign) -> ShieldEvaluation:
        start = perf_counter()
        fluxes: list[np.ndarray] = []
        metrics: list[ShieldMetrics] = []
        nominal_index = 0
        for index, uncertainty in enumerate(self.absorber_uncertainty):
            problem = materialize_lattice_design(design, self.nx, self.ny)
            absorber = problem.sigma_a > 0.0
            problem.sigma_a[absorber] *= 1.0 + uncertainty
            flux = self._solve(problem)
            far_mask = source_distant_mask(problem)
            sector_masks = source_distant_sector_masks(problem, far_mask)
            fluxes.append(flux)
            metrics.append(calculate_shield_metrics(flux, far_mask, sector_masks=sector_masks))
            if uncertainty == 0.0:
                nominal_index = index
        return ShieldEvaluation(
            design=design,
            flux=fluxes[nominal_index],
            metrics=metrics[nominal_index],
            ensemble_metrics=tuple(metrics),
            runtime_s=perf_counter() - start,
            fidelity=self.fidelity,
            metadata={"absorber_uncertainty": list(self.absorber_uncertainty)},
        )


class GenVRShieldEvaluator:
    def __init__(
        self,
        library_path: str | Path,
        internal_path: str | Path,
        nx: int = 112,
        ny: int = 112,
        n_pos: int = 2,
        n_mu: int = 4,
        n_phi: int = 32,
        device: str = "auto",
        dtype: str = "float64",
        rtol: float = 1.0e-8,
        max_iters: int = 1200,
        fidelity: str = "genvr-response-operator",
    ) -> None:
        from gmc.response_operator_ultra import (
            UltraPhaseSpace,
            load_ultra_internal,
            load_ultra_library,
        )
        import torch

        self.library_path = Path(library_path)
        self.internal_path = Path(internal_path)
        if not self.library_path.exists() or not self.internal_path.exists():
            raise FileNotFoundError("GenVR response-library cache is missing")
        self.library = load_ultra_library(str(self.library_path))
        self.internal = load_ultra_internal(str(self.internal_path))
        self.discretization = UltraPhaseSpace(n_pos=n_pos, n_mu=n_mu, n_phi=n_phi)
        self.nx = int(nx)
        self.ny = int(ny)
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self.device == "auto":
            self.device = "cpu"
        self.dtype = dtype
        self.rtol = float(rtol)
        self.max_iters = int(max_iters)
        self.fidelity = str(fidelity)

    def evaluate(self, design: ShieldDesign) -> ShieldEvaluation:
        from gmc.response_operator_ultra import solve_global_source_iteration

        problem = materialize_lattice_design(design, self.nx, self.ny)
        start = perf_counter()
        result = solve_global_source_iteration(
            problem,
            self.library,
            self.discretization,
            self.internal,
            rtol=self.rtol,
            max_iters=self.max_iters,
            device=self.device,
            dtype=self.dtype,
            check_every=8,
            solver="auto",
        )
        runtime = perf_counter() - start
        if not result.converged:
            raise RuntimeError(
                f"GenVR global solve failed for {design.signature}: "
                f"residual={result.relative_residual}"
            )
        far_mask = source_distant_mask(problem)
        sector_masks = source_distant_sector_masks(problem, far_mask)
        metrics = calculate_shield_metrics(
            result.flux,
            far_mask,
            sector_masks=sector_masks,
        )
        return ShieldEvaluation(
            design=design,
            flux=result.flux,
            metrics=metrics,
            ensemble_metrics=(metrics,),
            runtime_s=runtime,
            fidelity=self.fidelity,
            metadata={
                "iterations": result.iterations,
                "solver": result.method,
                "relative_residual": result.relative_residual,
                "leakage": result.leakage,
                "device": self.device,
                "library_path": str(self.library_path),
                "internal_path": str(self.internal_path),
            },
        )


@dataclass(frozen=True)
class ShieldMutation:
    operator: str
    remove: MacroCell
    add: MacroCell
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "remove": list(self.remove),
            "add": list(self.add),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ShieldProposal:
    parent_signature: str
    design: ShieldDesign
    mutation: ShieldMutation
    heuristic_score: float


@dataclass
class OperatorMemory:
    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in OPERATORS})
    mean_improvement: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in OPERATORS}
    )

    def update(self, operator: str, improvement: float) -> None:
        count = self.counts[operator] + 1
        old_mean = self.mean_improvement[operator]
        self.counts[operator] = count
        self.mean_improvement[operator] = old_mean + (improvement - old_mean) / count

    def bonus(self, operator: str) -> float:
        total = sum(self.counts.values()) + 1
        count = self.counts[operator]
        exploitation = max(self.mean_improvement[operator], 0.0)
        exploration = math.sqrt(math.log(total + 1.0) / (count + 1.0))
        return exploitation + 0.25 * exploration

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "mean_improvement": dict(self.mean_improvement),
        }


@dataclass(frozen=True)
class ShieldStrategy:
    operator_weights: dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in OPERATORS}
    )
    focus_sector: int | None = None
    exploration_fraction: float = 0.20
    hypothesis: str = "Block the strongest source-to-far-field streaming paths."
    decision_reason: str = "Default physics-guided strategy."

    def normalized_operator_weights(self) -> dict[str, float]:
        values = {name: max(float(self.operator_weights.get(name, 0.0)), 0.0) for name in OPERATORS}
        total = sum(values.values())
        if total <= 0.0:
            return {name: 1.0 / len(OPERATORS) for name in OPERATORS}
        return {name: value / total for name, value in values.items()}


class PhysicsGuidedProposalModel:
    def __init__(
        self,
        seed: int = 260918,
        importance_model: str = "diffusion_adjoint",
    ) -> None:
        if importance_model not in {"diffusion_adjoint", "gmc_flux_path"}:
            raise ValueError(
                "importance_model must be 'diffusion_adjoint' or 'gmc_flux_path'"
            )
        self.rng = np.random.default_rng(seed)
        self.importance_model = importance_model
        self._analysis_cache: dict[
            tuple[str, tuple[int, int], str],
            dict[str, Any],
        ] = {}

    @property
    def active_importance_operator(self) -> str:
        return (
            "gmc_path_importance"
            if self.importance_model == "gmc_flux_path"
            else "adjoint_sensitivity"
        )

    @staticmethod
    def _sector(row: int, column: int, source: MacroCell) -> int:
        delta_y = row - source[0]
        delta_x = column - source[1]
        angle = math.atan2(delta_y, delta_x)
        return int(np.floor(((angle + math.pi) / (2.0 * math.pi)) * 8.0)) % 8

    def _analyze(self, parent: ShieldEvaluation) -> dict[str, Any]:
        design = parent.design
        cache_key = (
            design.signature,
            tuple(parent.flux.shape),
            self.importance_model,
        )
        cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            return cached
        macro_flux = aggregate_macro_field(parent.flux, design.grid_size)
        scale = max(float(np.max(macro_flux)), np.finfo(np.float64).tiny)
        normalized_flux = macro_flux / scale
        source_row, source_column = design.source_cell
        rows, columns = np.indices((design.grid_size, design.grid_size))
        distance = np.hypot(rows - source_row, columns - source_column)
        maximum_distance = max(float(np.max(distance)), 1.0)
        sector_flux = np.zeros(8, dtype=np.float64)
        for row in range(design.grid_size):
            for column in range(design.grid_size):
                if (row, column) == design.source_cell:
                    continue
                sector = self._sector(row, column, design.source_cell)
                sector_flux[sector] += normalized_flux[row, column] * (
                    0.25 + distance[row, column] / maximum_distance
                )
        neighbor_flux = np.zeros_like(normalized_flux)
        for row in range(design.grid_size):
            for column in range(design.grid_size):
                local = []
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        if delta_row == 0 and delta_column == 0:
                            continue
                        neighbor_row = row + delta_row
                        neighbor_column = column + delta_column
                        if (
                            0 <= neighbor_row < design.grid_size
                            and 0 <= neighbor_column < design.grid_size
                        ):
                            local.append(normalized_flux[neighbor_row, neighbor_column])
                neighbor_flux[row, column] = float(np.mean(local)) if local else 0.0
        if self.importance_model == "gmc_flux_path":
            radial_weight = 0.25 + 0.75 * distance / maximum_distance
            path_continuity = 0.50 + 0.50 * neighbor_flux
            macro_sensitivity = normalized_flux * radial_weight * path_continuity
            importance_description = (
                "GMC flux weighted by source distance and neighboring-path continuity"
            )
        else:
            sensitivity_problem = materialize_lattice_design(
                design,
                parent.flux.shape[1],
                parent.flux.shape[0],
            )
            adjoint = solve_far_field_adjoint(sensitivity_problem, parent.flux)
            macro_sensitivity = aggregate_macro_field(
                np.maximum(parent.flux, 0.0) * adjoint,
                design.grid_size,
            )
            importance_description = "forward flux multiplied by diffusion adjoint"
        sensitivity_scale = max(
            float(np.max(macro_sensitivity)),
            np.finfo(np.float64).tiny,
        )
        normalized_sensitivity = macro_sensitivity / sensitivity_scale
        analysis = {
            "normalized_flux": normalized_flux,
            "normalized_sensitivity": normalized_sensitivity,
            "importance_model": self.importance_model,
            "importance_description": importance_description,
            "distance": distance,
            "maximum_distance": maximum_distance,
            "sector_flux": sector_flux,
            "hottest_sector": int(np.argmax(sector_flux)),
            "neighbor_flux": neighbor_flux,
        }
        self._analysis_cache[cache_key] = analysis
        return analysis

    def strategy_context(self, parent: ShieldEvaluation) -> dict[str, Any]:
        analysis = self._analyze(parent)
        design = parent.design
        absorber_set = set(design.absorbers)
        empty_cells = [
            (row, column)
            for row in range(design.grid_size)
            for column in range(design.grid_size)
            if (row, column) not in absorber_set and (row, column) != design.source_cell
        ]
        sensitivity = analysis["normalized_sensitivity"]
        ranked_additions = sorted(
            empty_cells,
            key=lambda cell: float(sensitivity[cell]),
            reverse=True,
        )[:6]
        ranked_removals = sorted(
            design.absorbers,
            key=lambda cell: float(sensitivity[cell]),
        )[:6]
        sector_flux = np.asarray(analysis["sector_flux"], dtype=np.float64)
        sector_scale = max(float(np.max(sector_flux)), np.finfo(np.float64).tiny)
        return {
            "importance_model": analysis["importance_model"],
            "importance_description": analysis["importance_description"],
            "hottest_sector": int(analysis["hottest_sector"]),
            "normalized_sector_leakage": (sector_flux / sector_scale).tolist(),
            "highest_sensitivity_empty_cells": [
                {
                    "cell": list(cell),
                    "normalized_importance": float(sensitivity[cell]),
                    "normalized_forward_adjoint": float(sensitivity[cell]),
                }
                for cell in ranked_additions
            ],
            "lowest_sensitivity_absorber_cells": [
                {
                    "cell": list(cell),
                    "normalized_importance": float(sensitivity[cell]),
                    "normalized_forward_adjoint": float(sensitivity[cell]),
                }
                for cell in ranked_removals
            ],
        }

    def propose(
        self,
        parent: ShieldEvaluation,
        count: int,
        memory: OperatorMemory,
        strategy: ShieldStrategy,
    ) -> list[ShieldProposal]:
        design = parent.design
        analysis = self._analyze(parent)
        normalized_flux = analysis["normalized_flux"]
        normalized_sensitivity = analysis["normalized_sensitivity"]
        source_row, source_column = design.source_cell
        distance = analysis["distance"]
        maximum_distance = float(analysis["maximum_distance"])
        sector_flux = analysis["sector_flux"]
        hottest_sector = int(analysis["hottest_sector"])
        if strategy.focus_sector is not None:
            hottest_sector = int(strategy.focus_sector) % 8

        absorber_set = set(design.absorbers)
        absorber_sector_counts = np.zeros(8, dtype=np.int64)
        for row, column in design.absorbers:
            absorber_sector_counts[self._sector(row, column, design.source_cell)] += 1
        empty_cells = [
            (row, column)
            for row in range(design.grid_size)
            for column in range(design.grid_size)
            if (row, column) not in absorber_set and (row, column) != design.source_cell
        ]

        neighbor_flux = analysis["neighbor_flux"]

        operator_weights = strategy.normalized_operator_weights()
        proposals: list[ShieldProposal] = []
        seen: set[str] = set()
        scored: list[tuple[float, ShieldProposal]] = []
        empty_sensitivity_threshold = float(
            np.quantile(
                [normalized_sensitivity[cell] for cell in empty_cells],
                0.75,
            )
        )
        for remove in design.absorbers:
            remove_distance = distance[remove]
            remove_utility = (
                0.45 * neighbor_flux[remove]
                + 0.55 * normalized_sensitivity[remove]
            ) * (
                0.5 + 0.5 * (maximum_distance - remove_distance) / maximum_distance
            )
            remove_sector = self._sector(remove[0], remove[1], design.source_cell)
            for add in empty_cells:
                add_distance = distance[add]
                sector = self._sector(add[0], add[1], design.source_cell)
                sector_need = sector_flux[sector] / max(float(np.max(sector_flux)), 1.0e-12)
                local_need = normalized_flux[add]
                sensitivity_need = normalized_sensitivity[add]
                radial_leverage = 1.0 - 0.55 * add_distance / maximum_distance
                add_need = (
                    0.30 * sector_need
                    + 0.25 * local_need
                    + 0.45 * sensitivity_need
                )
                if sector == hottest_sector:
                    add_need += 0.35

                opposite_sector = (sector + 4) % 8
                repairs_imbalance = (
                    sector == hottest_sector
                    and remove_sector == opposite_sector
                    and absorber_sector_counts[sector]
                    < absorber_sector_counts[opposite_sector]
                )
                if sensitivity_need >= empty_sensitivity_threshold:
                    operator = self.active_importance_operator
                    rationale = (
                        "Move absorber to a high GMC leakage-path importance cell."
                        if operator == "gmc_path_importance"
                        else "Move absorber to a high forward-times-adjoint sensitivity cell."
                    )
                elif repairs_imbalance:
                    operator = "symmetry_repair"
                    rationale = (
                        "Rebalance absorber coverage toward the leaking angular sector."
                    )
                elif sector == hottest_sector and add_distance <= 2.5:
                    operator = "channel_block"
                    rationale = "Place absorber inside the hottest radial streaming sector."
                elif add_distance <= 1.5:
                    operator = "inner_barrier"
                    rationale = "Strengthen the inner barrier before particles enter a long channel."
                elif add_distance >= 3.0 and local_need >= 0.25:
                    operator = "hotspot_cap"
                    rationale = "Suppress a measured source-distant flux hotspot."
                else:
                    operator = "redistribute"
                    rationale = "Move low-utility material toward a higher-risk transport path."

                movement = math.hypot(add[0] - remove[0], add[1] - remove[1])
                score = (
                    operator_weights[operator]
                    * memory.bonus(operator)
                    * (add_need * radial_leverage - 0.65 * remove_utility)
                    - 0.01 * movement
                )
                candidate = design.swap(remove, add)
                proposal = ShieldProposal(
                    parent_signature=design.signature,
                    design=candidate,
                    mutation=ShieldMutation(operator, remove, add, rationale),
                    heuristic_score=float(score),
                )
                scored.append((score, proposal))

        scored.sort(key=lambda item: item[0], reverse=True)
        directed_count = max(1, int(round(count * (1.0 - strategy.exploration_fraction))))
        for _, proposal in scored:
            if proposal.design.signature in seen:
                continue
            proposals.append(proposal)
            seen.add(proposal.design.signature)
            if len(proposals) >= directed_count:
                break

        random_pool = [proposal for _, proposal in scored if proposal.design.signature not in seen]
        self.rng.shuffle(random_pool)
        for proposal in random_pool:
            if len(proposals) >= count:
                break
            random_mutation = ShieldMutation(
                "random_explore",
                proposal.mutation.remove,
                proposal.mutation.add,
                "Explore a legal fixed-mass swap outside the current physics ranking.",
            )
            proposals.append(
                ShieldProposal(
                    proposal.parent_signature,
                    proposal.design,
                    random_mutation,
                    proposal.heuristic_score,
                )
            )
        return proposals


@dataclass
class SearchRecord:
    round_index: int
    proposal: ShieldProposal
    evaluation: ShieldEvaluation
    robust_risk: float
    acquisition: float
    parent_risk: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "parent_signature": self.proposal.parent_signature,
            "design": self.proposal.design.to_dict(),
            "mutation": self.proposal.mutation.to_dict(),
            "heuristic_score": self.proposal.heuristic_score,
            "robust_risk": self.robust_risk,
            "acquisition": self.acquisition,
            "parent_risk": self.parent_risk,
            "improvement": self.parent_risk - self.robust_risk,
            "metrics": self.evaluation.metrics.to_dict(),
            "runtime_s": self.evaluation.runtime_s,
            "fidelity": self.evaluation.fidelity,
        }


@dataclass
class VerificationStage:
    fidelity: str
    baseline: ShieldEvaluation
    candidates: tuple[ShieldEvaluation, ...]
    candidate_risks: tuple[float, ...]
    selected: ShieldEvaluation
    selected_risk: float
    accepted: bool
    rank_correlation_with_previous: float | None
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity": self.fidelity,
            "baseline": {
                "design": self.baseline.design.to_dict(),
                "metrics": self.baseline.metrics.to_dict(),
                "runtime_s": self.baseline.runtime_s,
                "metadata": self.baseline.metadata,
            },
            "candidates": [
                {
                    "design": evaluation.design.to_dict(),
                    "metrics": evaluation.metrics.to_dict(),
                    "risk": risk,
                    "runtime_s": evaluation.runtime_s,
                    "metadata": evaluation.metadata,
                }
                for evaluation, risk in zip(self.candidates, self.candidate_risks)
            ],
            "selected_design": self.selected.design.to_dict(),
            "selected_risk": self.selected_risk,
            "accepted": self.accepted,
            "rank_correlation_with_previous": self.rank_correlation_with_previous,
            "decision": self.decision,
        }


@dataclass
class ShieldSearchResult:
    baseline_fast: ShieldEvaluation
    best_fast: ShieldEvaluation
    best_fast_risk: float
    records: list[SearchRecord]
    memory: OperatorMemory
    strategy_trace: list[dict[str, Any]] = field(default_factory=list)
    high_fidelity_baseline: ShieldEvaluation | None = None
    high_fidelity_candidates: tuple[ShieldEvaluation, ...] = ()
    best_high_fidelity: ShieldEvaluation | None = None
    best_high_fidelity_risk: float | None = None
    verification_stages: tuple[VerificationStage, ...] = ()

    @property
    def best_verified_evaluation(self) -> ShieldEvaluation | None:
        if self.verification_stages:
            return self.verification_stages[-1].selected
        return self.best_high_fidelity

    @property
    def best_verified_risk(self) -> float | None:
        if self.verification_stages:
            return self.verification_stages[-1].selected_risk
        return self.best_high_fidelity_risk

    @property
    def best_design(self) -> ShieldDesign:
        verified = self.best_verified_evaluation
        if verified is not None:
            return verified.design
        return self.best_fast.design

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "baseline": {
                "design": self.baseline_fast.design.to_dict(),
                "metrics": self.baseline_fast.metrics.to_dict(),
                "fidelity": self.baseline_fast.fidelity,
            },
            "best_fast": {
                "design": self.best_fast.design.to_dict(),
                "metrics": self.best_fast.metrics.to_dict(),
                "robust_risk": self.best_fast_risk,
                "fidelity": self.best_fast.fidelity,
            },
            "operator_memory": self.memory.to_dict(),
            "strategy_trace": list(self.strategy_trace),
            "records": [record.to_dict() for record in self.records],
            "verification_stages": [
                stage.to_dict() for stage in self.verification_stages
            ],
        }
        if self.high_fidelity_baseline is not None:
            payload["high_fidelity_baseline"] = {
                "design": self.high_fidelity_baseline.design.to_dict(),
                "metrics": self.high_fidelity_baseline.metrics.to_dict(),
                "runtime_s": self.high_fidelity_baseline.runtime_s,
                "metadata": self.high_fidelity_baseline.metadata,
            }
        if self.best_high_fidelity is not None:
            payload["best_high_fidelity"] = {
                "design": self.best_high_fidelity.design.to_dict(),
                "metrics": self.best_high_fidelity.metrics.to_dict(),
                "robust_risk": self.best_high_fidelity_risk,
                "runtime_s": self.best_high_fidelity.runtime_s,
                "metadata": self.best_high_fidelity.metadata,
            }
            payload["high_fidelity_candidates"] = [
                {
                    "design": evaluation.design.to_dict(),
                    "metrics": evaluation.metrics.to_dict(),
                    "runtime_s": evaluation.runtime_s,
                    "metadata": evaluation.metadata,
                }
                for evaluation in self.high_fidelity_candidates
            ]
        verified = self.best_verified_evaluation
        if verified is not None:
            payload["final_verified"] = {
                "design": verified.design.to_dict(),
                "metrics": verified.metrics.to_dict(),
                "risk": self.best_verified_risk,
                "fidelity": verified.fidelity,
                "baseline_retained": (
                    verified.design.signature == self.baseline_fast.design.signature
                ),
            }
        return payload


class SmartShieldAgent:
    def __init__(
        self,
        fast_evaluator: ShieldEvaluator,
        high_fidelity_evaluator: ShieldEvaluator | None = None,
        verification_evaluators: tuple[ShieldEvaluator, ...] = (),
        proposal_model: PhysicsGuidedProposalModel | None = None,
        metric_weights: dict[str, float] | None = None,
        risk_aversion: float = 0.35,
        exploration_strength: float = 0.04,
        minimum_verified_improvement: float = 0.0,
    ) -> None:
        self.fast_evaluator = fast_evaluator
        self.high_fidelity_evaluator = high_fidelity_evaluator
        evaluators = []
        if high_fidelity_evaluator is not None:
            evaluators.append(high_fidelity_evaluator)
        evaluators.extend(verification_evaluators)
        self.verification_evaluators = tuple(evaluators)
        self.proposal_model = proposal_model or PhysicsGuidedProposalModel()
        self.metric_weights = metric_weights or dict(DEFAULT_METRIC_WEIGHTS)
        self.risk_aversion = float(risk_aversion)
        self.exploration_strength = float(exploration_strength)
        self.minimum_verified_improvement = float(minimum_verified_improvement)
        if not 0.0 <= self.minimum_verified_improvement < 1.0:
            raise ValueError("minimum_verified_improvement must lie in [0, 1)")

    @staticmethod
    def _design_distance(left: ShieldDesign, right: ShieldDesign) -> float:
        left_cells = set(left.absorbers)
        right_cells = set(right.absorbers)
        denominator = max(left.absorber_count + right.absorber_count, 1)
        return len(left_cells.symmetric_difference(right_cells)) / denominator

    @classmethod
    def _diverse_shortlist(
        cls,
        ranked: list[ShieldEvaluation],
        risks: dict[str, float],
        count: int,
    ) -> list[ShieldEvaluation]:
        if count <= 0 or not ranked:
            return []
        pool = ranked[: max(count * 4, count)]
        selected = [pool.pop(0)]
        best_risk = max(risks[selected[0].design.signature], np.finfo(np.float64).tiny)
        while pool and len(selected) < count:
            def selection_score(evaluation: ShieldEvaluation) -> float:
                relative_risk = risks[evaluation.design.signature] / best_risk
                diversity = min(
                    cls._design_distance(evaluation.design, item.design)
                    for item in selected
                )
                return relative_risk + 0.08 * (1.0 - diversity)

            next_index = min(range(len(pool)), key=lambda index: selection_score(pool[index]))
            selected.append(pool.pop(next_index))
        return selected

    @staticmethod
    def _rank_correlation(
        evaluations: tuple[ShieldEvaluation, ...],
        previous_risks: dict[str, float],
        current_risks: tuple[float, ...],
    ) -> float | None:
        if len(evaluations) < 2:
            return None
        previous = np.asarray(
            [previous_risks[evaluation.design.signature] for evaluation in evaluations],
            dtype=np.float64,
        )
        current = np.asarray(current_risks, dtype=np.float64)
        previous_ranks = np.argsort(np.argsort(previous)).astype(np.float64)
        current_ranks = np.argsort(np.argsort(current)).astype(np.float64)
        correlation = float(np.corrcoef(previous_ranks, current_ranks)[0, 1])
        return correlation if math.isfinite(correlation) else None

    @classmethod
    def _select_search_beam(
        cls,
        candidates: list[ShieldEvaluation],
        risks: dict[str, float],
        acquisitions: dict[str, float],
        beam_width: int,
    ) -> list[ShieldEvaluation]:
        if not candidates:
            return []
        unique = {candidate.design.signature: candidate for candidate in candidates}
        pool = list(unique.values())
        elite = min(pool, key=lambda item: risks[item.design.signature])
        selected = [elite]
        pool.remove(elite)
        best_risk = max(risks[elite.design.signature], np.finfo(np.float64).tiny)
        while pool and len(selected) < beam_width:
            def beam_score(evaluation: ShieldEvaluation) -> float:
                signature = evaluation.design.signature
                acquisition = acquisitions.get(signature, risks[signature]) / best_risk
                diversity = min(
                    cls._design_distance(evaluation.design, item.design)
                    for item in selected
                )
                return acquisition + 0.08 * (1.0 - diversity)

            next_index = min(range(len(pool)), key=lambda index: beam_score(pool[index]))
            selected.append(pool.pop(next_index))
        return selected

    def search(
        self,
        baseline_design: ShieldDesign | None = None,
        rounds: int = 4,
        beam_width: int = 4,
        proposals_per_parent: int = 24,
        high_fidelity_top_k: int = 2,
        verification_top_ks: tuple[int, ...] | None = None,
        strategy: ShieldStrategy | None = None,
        strategy_provider: Callable[[dict[str, Any]], ShieldStrategy] | None = None,
    ) -> ShieldSearchResult:
        if rounds < 1 or beam_width < 1 or proposals_per_parent < 1:
            raise ValueError("search budgets must be positive")
        strategy = strategy or ShieldStrategy()
        baseline_design = baseline_design or baseline_lattice_design()
        baseline = self.fast_evaluator.evaluate(baseline_design)
        archive: dict[str, ShieldEvaluation] = {baseline_design.signature: baseline}
        risks: dict[str, float] = {baseline_design.signature: 1.0}
        acquisitions: dict[str, float] = {baseline_design.signature: 1.0}
        beam = [baseline]
        memory = OperatorMemory()
        records: list[SearchRecord] = []
        strategy_trace: list[dict[str, Any]] = []

        for round_index in range(1, rounds + 1):
            if strategy_provider is not None:
                context = {
                    "round": round_index,
                    "fixed_objective": dict(self.metric_weights),
                    "beam": [
                        {
                            "signature": evaluation.design.signature,
                            "risk": risks[evaluation.design.signature],
                            "metrics": evaluation.metrics.to_dict(),
                            "absorbers": [list(cell) for cell in evaluation.design.absorbers],
                            "physics_diagnosis": self.proposal_model.strategy_context(
                                evaluation
                            ),
                        }
                        for evaluation in beam
                    ],
                    "operator_memory": memory.to_dict(),
                    "recent_outcomes": [
                        {
                            "operator": record.proposal.mutation.operator,
                            "improvement": record.parent_risk - record.robust_risk,
                            "risk": record.robust_risk,
                        }
                        for record in records[-12:]
                    ],
                }
                strategy = strategy_provider(context)
            strategy_trace.append(
                {
                    "round": round_index,
                    "operator_weights": strategy.normalized_operator_weights(),
                    "focus_sector": strategy.focus_sector,
                    "exploration_fraction": strategy.exploration_fraction,
                    "hypothesis": strategy.hypothesis,
                    "decision_reason": strategy.decision_reason,
                }
            )
            proposal_by_signature: dict[str, ShieldProposal] = {}
            parent_risks: dict[str, float] = {}
            for parent in beam:
                parent_risk = risks[parent.design.signature]
                parent_risks[parent.design.signature] = parent_risk
                for proposal in self.proposal_model.propose(
                    parent,
                    proposals_per_parent,
                    memory,
                    strategy,
                ):
                    if proposal.design.signature in archive:
                        continue
                    existing = proposal_by_signature.get(proposal.design.signature)
                    if existing is None or proposal.heuristic_score > existing.heuristic_score:
                        proposal_by_signature[proposal.design.signature] = proposal

            round_records: list[SearchRecord] = []
            for proposal in proposal_by_signature.values():
                evaluation = self.fast_evaluator.evaluate(proposal.design)
                archive[proposal.design.signature] = evaluation
                robust_risk = evaluation.robust_risk(
                    baseline,
                    self.metric_weights,
                    self.risk_aversion,
                )
                risks[proposal.design.signature] = robust_risk
                parent_risk = parent_risks[proposal.parent_signature]
                improvement = parent_risk - robust_risk
                memory.update(proposal.mutation.operator, improvement)
                acquisition = robust_risk - self.exploration_strength * memory.bonus(
                    proposal.mutation.operator
                )
                acquisitions[proposal.design.signature] = acquisition
                record = SearchRecord(
                    round_index,
                    proposal,
                    evaluation,
                    robust_risk,
                    acquisition,
                    parent_risk,
                )
                records.append(record)
                round_records.append(record)

            candidates = list(beam)
            candidates.extend(record.evaluation for record in round_records)
            beam = self._select_search_beam(
                candidates,
                risks,
                acquisitions,
                beam_width,
            )

        ranked = sorted(archive.values(), key=lambda item: risks[item.design.signature])
        best_fast = ranked[0]
        result = ShieldSearchResult(
            baseline_fast=baseline,
            best_fast=best_fast,
            best_fast_risk=risks[best_fast.design.signature],
            records=records,
            memory=memory,
            strategy_trace=strategy_trace,
        )

        evaluators = self.verification_evaluators
        if evaluators and high_fidelity_top_k > 0:
            if verification_top_ks is None:
                top_ks = (high_fidelity_top_k,) * len(evaluators)
            else:
                top_ks = tuple(int(value) for value in verification_top_ks)
                if len(top_ks) != len(evaluators):
                    raise ValueError(
                        "verification_top_ks must match the number of verification evaluators"
                    )
                if any(value < 1 for value in top_ks):
                    raise ValueError("verification top-k budgets must be positive")

            fast_candidates = [
                evaluation
                for evaluation in ranked
                if evaluation.design.signature != baseline_design.signature
            ]
            shortlisted = self._diverse_shortlist(
                fast_candidates,
                risks,
                top_ks[0],
            )
            previous_risks = dict(risks)
            verification_stages: list[VerificationStage] = []
            for stage_index, evaluator in enumerate(evaluators):
                if not shortlisted:
                    break
                stage_baseline = evaluator.evaluate(baseline_design)
                stage_candidates = tuple(
                    evaluator.evaluate(candidate.design) for candidate in shortlisted
                )
                stage_risks = tuple(
                    candidate.robust_risk(
                        stage_baseline,
                        self.metric_weights,
                        risk_aversion=0.0,
                    )
                    for candidate in stage_candidates
                )
                threshold = 1.0 - self.minimum_verified_improvement
                accepted_indices = [
                    index for index, risk in enumerate(stage_risks) if risk < threshold
                ]
                if accepted_indices:
                    best_index = min(accepted_indices, key=lambda index: stage_risks[index])
                    selected = stage_candidates[best_index]
                    selected_risk = float(stage_risks[best_index])
                    accepted = True
                    decision = (
                        f"accepted {selected.design.signature}: verified risk "
                        f"{selected_risk:.6g} < {threshold:.6g}"
                    )
                else:
                    selected = stage_baseline
                    selected_risk = 1.0
                    accepted = False
                    decision = (
                        "rejected all candidates: no verified improvement over baseline"
                    )
                rank_correlation = self._rank_correlation(
                    stage_candidates,
                    previous_risks,
                    stage_risks,
                )
                stage = VerificationStage(
                    fidelity=evaluator.fidelity,
                    baseline=stage_baseline,
                    candidates=stage_candidates,
                    candidate_risks=stage_risks,
                    selected=selected,
                    selected_risk=selected_risk,
                    accepted=accepted,
                    rank_correlation_with_previous=rank_correlation,
                    decision=decision,
                )
                verification_stages.append(stage)
                if stage_index == 0:
                    result.high_fidelity_baseline = stage_baseline
                    result.high_fidelity_candidates = stage_candidates
                    result.best_high_fidelity = selected
                    result.best_high_fidelity_risk = selected_risk
                if not accepted or stage_index + 1 >= len(evaluators):
                    break
                previous_risks = {
                    evaluation.design.signature: risk
                    for evaluation, risk in zip(stage_candidates, stage_risks)
                }
                improving = [
                    evaluation
                    for evaluation, risk in sorted(
                        zip(stage_candidates, stage_risks),
                        key=lambda item: item[1],
                    )
                    if risk < threshold
                ]
                shortlisted = improving[: top_ks[stage_index + 1]]
            result.verification_stages = tuple(verification_stages)
        return result
