from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import numpy as np

from gmc.method_taxonomy import FLUX_FIELD_ALIASES


def normalize_flux(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("flux array must not be empty")
    scale = max(float(np.max(np.abs(x))), 1.0e-300)
    return x / scale


def flux_shape_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    a = normalize_flux(candidate)
    b = normalize_flux(reference)
    if a.shape != b.shape:
        raise ValueError(f"flux shapes differ: candidate={a.shape}, reference={b.shape}")
    mask = (a > 0.0) | (b > 0.0)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        corr = 1.0 if np.allclose(a, b) else 0.0
    else:
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    rel_l1 = float(np.sum(np.abs(a - b)) / max(float(np.sum(np.abs(b))), 1.0e-300))
    log_rmse = 0.0
    if np.any(mask):
        log_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.log10(np.maximum(a[mask], 1.0e-15))
                        - np.log10(np.maximum(b[mask], 1.0e-15))
                    )
                    ** 2
                )
            )
        )
    return {"corr": corr, "normalized_rel_l1": rel_l1, "log10_rmse": log_rmse}


def flux_level_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"flux shapes differ: candidate={a.shape}, reference={b.shape}")
    reference_l1 = max(float(np.sum(np.abs(b))), 1.0e-300)
    reference_peak = max(float(np.max(np.abs(b))), 1.0e-300)
    reference_integral = float(np.sum(b))
    return {
        "relative_l1": float(np.sum(np.abs(a - b)) / reference_l1),
        "integral_ratio": float(np.sum(a) / max(abs(reference_integral), 1.0e-300)),
        "peak_ratio": float(np.max(np.abs(a)) / reference_peak),
    }


def spectral_ray_metrics(candidate: np.ndarray, reference: np.ndarray, angle_bins: int = 36) -> dict[str, float]:
    """Measure directional concentration in a pairwise residual.

    The residual is windowed before the 2-D FFT.  Power at the DC component and
    the Nyquist rim is excluded.  A spatial ray maps to concentrated spectral
    power in its perpendicular direction.  ``directional_concentration`` is the
    fraction of eligible power in the four strongest angular bins; an isotropic
    residual approaches 4/angle_bins.  ``ray_index`` multiplies that fraction by
    residual RMS so a tiny but directional residual is not over-penalized.
    """

    a = normalize_flux(candidate)
    b = normalize_flux(reference)
    if a.shape != b.shape:
        raise ValueError(f"flux shapes differ: candidate={a.shape}, reference={b.shape}")
    if a.ndim != 2:
        raise ValueError(f"spectral ray metrics require 2-D flux arrays, got {a.ndim}-D")
    residual = a - b
    ny, nx = residual.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    xx, yy = np.meshgrid(fx, fy)
    radius = np.sqrt(xx * xx + yy * yy)
    eligible = (radius >= 0.03) & (radius <= 0.45)
    theta = np.mod(np.arctan2(yy, xx), np.pi)
    edges = np.linspace(0.0, np.pi, angle_bins + 1)
    energy = np.zeros(angle_bins, dtype=np.float64)
    ids = np.clip(np.digitize(theta[eligible], edges) - 1, 0, angle_bins - 1)
    np.add.at(energy, ids, spectrum[eligible])
    total = max(float(np.sum(energy)), 1.0e-300)
    top = np.sort(energy)[-min(4, angle_bins) :]
    concentration = float(np.sum(top) / total)
    rms = float(np.sqrt(np.mean(residual * residual)))
    peak_to_mean = float(np.max(energy) / max(float(np.mean(energy)), 1.0e-300))
    return {
        "residual_rms": rms,
        "directional_concentration": concentration,
        "directional_peak_to_mean": peak_to_mean,
        "ray_index": rms * concentration,
    }


def comparison_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        "shape": flux_shape_metrics(candidate, reference),
        "level": flux_level_metrics(candidate, reference),
        "directional_residual": spectral_ray_metrics(candidate, reference),
    }


def load_fluxes(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load flux arrays while supporting both canonical and historical keys."""

    with np.load(npz_path) as data:
        fluxes = {name: np.asarray(data[name]) for name in data.files}

    nonempty_shapes: dict[str, tuple[int, ...]] = {}
    for canonical_name, aliases in FLUX_FIELD_ALIASES.items():
        present = [(name, fluxes[name]) for name in aliases if name in fluxes]
        if not present:
            empty = np.array([], dtype=np.float64)
            fluxes[canonical_name] = empty
            for alias in aliases:
                fluxes.setdefault(alias, empty)
            continue

        source_name, source = present[0]
        for other_name, other in present[1:]:
            if source.shape != other.shape or not np.array_equal(source, other, equal_nan=True):
                raise ValueError(
                    f"conflicting NPZ aliases for {canonical_name}: {source_name} and {other_name}"
                )
        if source.size:
            if source.ndim != 2:
                raise ValueError(f"flux field {source_name!r} must be 2-D, got shape {source.shape}")
            nonempty_shapes[canonical_name] = source.shape
        fluxes[canonical_name] = source
        for alias in aliases:
            fluxes.setdefault(alias, source)

    unique_shapes = set(nonempty_shapes.values())
    if len(unique_shapes) > 1:
        details = ", ".join(f"{name}={shape}" for name, shape in nonempty_shapes.items())
        raise ValueError(f"incompatible flux field shapes in {npz_path}: {details}")
    return fluxes


def select_reference_flux(
    fluxes: dict[str, np.ndarray],
    preference: tuple[str, ...] = (
        "global_history_mc",
        "projected_local_mc_response",
        "dense_sn_reference",
    ),
) -> tuple[str, np.ndarray]:
    """Select the strongest available reference and return its canonical ID."""

    for method_id in preference:
        values = fluxes.get(method_id, np.array([]))
        if values.size:
            return method_id, values
    raise ValueError(f"no usable reference flux found; checked {preference}")


def diagnose_experiment(npz_path: str | Path, metrics_path: str | Path, case: str) -> dict[str, Any]:
    fluxes = load_fluxes(npz_path)
    summary = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    case_metrics = summary["cases"][case]
    cfm = fluxes["genvr_cfm_response"]
    dense = fluxes["dense_sn_reference"]
    if dense.size == 0:
        raise ValueError("dense reference is required for ray diagnostics")
    cfm_vs_dense = flux_shape_metrics(cfm, dense)
    cfm_level_vs_dense = flux_level_metrics(cfm, dense)
    cfm_ray_vs_dense = spectral_ray_metrics(cfm, dense)
    config = summary.get("config", {})
    dense_order = case_metrics.get("dense_angular_order", [config.get("dense_n_mu"), config.get("dense_n_phi")])
    result: dict[str, Any] = {
        "n_state": int(case_metrics["n_state"]),
        "n_angle": int(case_metrics["n_angle"]),
        "interface_phi_offset_fraction": float(
            case_metrics.get("interface_phi_offset_fraction", config.get("phi_offset_fraction", 0.0))
        ),
        "dense_n_mu": int(dense_order[0]),
        "dense_n_phi": int(dense_order[1]),
        "dense_phi_offset_fraction": float(
            case_metrics.get("dense_phi_offset_fraction", config.get("dense_phi_offset_fraction", 0.0))
        ),
        "cfm_build_s": float(case_metrics["cfm_build_s"]),
        "cfm_solve_s": float(case_metrics["cfm_solve_s"]),
        "cfm_residual": float(case_metrics["cfm_residual"]),
        "cfm_vs_dense": cfm_vs_dense,
        "cfm_ray_vs_dense": cfm_ray_vs_dense,
        "cfm_level_vs_dense_sn": cfm_level_vs_dense,
        "cfm_directional_disagreement_vs_dense_sn": cfm_ray_vs_dense,
        "genvr_cfm_vs_dense_sn": cfm_vs_dense,
        "genvr_cfm_ray_vs_dense_sn": cfm_ray_vs_dense,
    }
    projected_mc = fluxes["projected_local_mc_response"]
    if projected_mc.size:
        projected_vs_dense = flux_shape_metrics(projected_mc, dense)
        projected_level_vs_dense = flux_level_metrics(projected_mc, dense)
        projected_ray_vs_dense = spectral_ray_metrics(projected_mc, dense)
        cfm_vs_projected = flux_shape_metrics(cfm, projected_mc)
        cfm_level_vs_projected = flux_level_metrics(cfm, projected_mc)
        cfm_directional_vs_projected = spectral_ray_metrics(cfm, projected_mc)
        result.update(
            {
                "oracle_vs_dense": projected_vs_dense,
                "oracle_ray_vs_dense": projected_ray_vs_dense,
                "cfm_vs_oracle": cfm_vs_projected,
                "cfm_learning_error": {
                    "shape": cfm_vs_projected,
                    "level": cfm_level_vs_projected,
                    "directional_residual": cfm_directional_vs_projected,
                },
                "projected_local_mc_level_vs_dense_sn": projected_level_vs_dense,
                "projected_directional_disagreement_vs_dense_sn": projected_ray_vs_dense,
                "projected_local_mc_vs_dense_sn": projected_vs_dense,
                "projected_local_mc_ray_vs_dense_sn": projected_ray_vs_dense,
                "genvr_cfm_vs_projected_local_mc": cfm_vs_projected,
            }
        )
    global_mc = fluxes["global_history_mc"]
    if global_mc.size:
        result["cfm_vs_global_mc"] = comparison_metrics(cfm, global_mc)
        result["dense_sn_vs_global_mc"] = comparison_metrics(dense, global_mc)
        if projected_mc.size:
            result["projected_local_mc_vs_global_mc"] = comparison_metrics(projected_mc, global_mc)
    result["wall_s"] = result["cfm_build_s"] + result["cfm_solve_s"]
    return result


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
