from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np

from .shield_design import ShieldDesign, ShieldEvaluation


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_gmc_candidate_snapshot(
    output_dir: str | Path,
    baseline_design: ShieldDesign,
    evaluation: ShieldEvaluation,
    *,
    risk: float,
    accepted: bool,
    screening_round: int,
    evaluation_index: int,
    generation_round: int,
    gmc_guided_generation: bool,
) -> dict[str, Any]:
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    root = Path(output_dir)
    screening_dir = root / "gmc_screening"
    screening_dir.mkdir(parents=True, exist_ok=True)

    signature = evaluation.design.signature
    stem = f"candidate_{evaluation_index:03d}_g{generation_round}_{signature}"
    image_path = screening_dir / f"{stem}.png"
    field_path = screening_dir / f"{stem}.npz"
    latest_path = screening_dir / "latest.png"

    flux = np.asarray(evaluation.flux, dtype=np.float64)
    positive = flux[np.isfinite(flux) & (flux > 0.0)]
    if positive.size == 0:
        raise ValueError("GMC candidate flux has no finite positive values")
    flux_ceiling = float(np.max(positive))
    flux_floor = max(float(np.quantile(positive, 0.005)), flux_ceiling * 1.0e-8)
    if flux_floor >= flux_ceiling:
        flux_floor = max(flux_ceiling * 0.5, np.finfo(np.float64).tiny)

    design = evaluation.design
    extent = [0.0, design.grid_size, 0.0, design.grid_size]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)

    axes[0].imshow(
        design.mask(),
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    source_row, source_column = design.source_cell
    axes[0].add_patch(
        Rectangle(
            (source_column, source_row),
            1.0,
            1.0,
            fill=False,
            edgecolor="#00bcd4",
            linewidth=2.4,
            linestyle="--",
        )
    )
    baseline_cells = set(baseline_design.absorbers)
    candidate_cells = set(design.absorbers)
    for row, column in sorted(baseline_cells - candidate_cells):
        axes[0].add_patch(
            Rectangle(
                (column, row),
                1.0,
                1.0,
                fill=False,
                edgecolor="#e45756",
                linewidth=2.5,
                linestyle="--",
            )
        )
    for row, column in sorted(candidate_cells - baseline_cells):
        axes[0].add_patch(
            Rectangle(
                (column, row),
                1.0,
                1.0,
                fill=False,
                edgecolor="#2ca02c",
                linewidth=2.8,
            )
        )
    axes[0].set_title("Candidate material layout")
    axes[0].set_xlabel("x macro-cell")
    axes[0].set_ylabel("y macro-cell")
    axes[0].set_xticks(np.arange(design.grid_size + 1))
    axes[0].set_yticks(np.arange(design.grid_size + 1))
    axes[0].grid(color="white", linewidth=0.6, alpha=0.7)

    flux_image = axes[1].imshow(
        np.maximum(flux, flux_floor),
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=flux_floor, vmax=flux_ceiling),
        interpolation="nearest",
    )
    axes[1].add_patch(
        Rectangle(
            (source_column, source_row),
            1.0,
            1.0,
            fill=False,
            edgecolor="#00e5ff",
            linewidth=1.8,
            linestyle="--",
        )
    )
    for row, column in design.absorbers:
        axes[1].add_patch(
            Rectangle(
                (column, row),
                1.0,
                1.0,
                fill=False,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.75,
            )
        )
    axes[1].set_title("GMC scalar-flux field")
    axes[1].set_xlabel("x macro-cell")
    axes[1].set_ylabel("y macro-cell")
    figure.colorbar(flux_image, ax=axes[1], label="GMC flux · logarithmic scale", shrink=0.86)

    status = "accepted" if accepted else "rejected"
    guidance = "GMC-guided child" if gmc_guided_generation else "generated candidate"
    figure.suptitle(
        f"GMC screen {evaluation_index:03d} · {status}\n"
        f"{signature} · generation {generation_round} · {guidance} · "
        f"risk={risk:.4f} · solve={evaluation.runtime_s:.2f} s",
        fontsize=13,
        weight="bold",
    )
    figure.savefig(image_path, dpi=120, bbox_inches="tight")
    plt.close(figure)

    np.savez_compressed(
        field_path,
        gmc_flux=flux,
        geometry_mask=design.mask(),
        source_cell=np.asarray(design.source_cell, dtype=np.int64),
    )
    shutil.copyfile(image_path, latest_path)

    entry = {
        "evaluation_index": int(evaluation_index),
        "screening_round": int(screening_round),
        "signature": signature,
        "generation_round": int(generation_round),
        "gmc_guided_generation": bool(gmc_guided_generation),
        "risk": float(risk),
        "accepted": bool(accepted),
        "runtime_s": float(evaluation.runtime_s),
        "geometry_and_gmc_field": str(image_path.relative_to(root)),
        "gmc_field": str(field_path.relative_to(root)),
        "latest": str(latest_path.relative_to(root)),
    }
    manifest_path = screening_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            manifest = {}
    else:
        manifest = {}
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        frames = []
    frames.append(entry)
    manifest = {
        "schema_version": "1.0",
        "description": "Live candidate geometry and GMC scalar-flux snapshots",
        "frames": frames,
        "latest": entry,
    }
    _write_json(manifest_path, manifest)
    _write_json(screening_dir / "latest.json", entry)
    return entry
