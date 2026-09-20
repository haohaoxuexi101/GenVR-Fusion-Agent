#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmc.benchmarks2d import make_hohlraum_problem  # noqa: E402
from gmc.response_operator_ultra import flux_shape_metrics  # noqa: E402


BACKGROUND = "#f7f8fb"
TEXT = "#182230"
MUTED = "#667085"
GRID = "#d9dee7"


def _material_tracks(problem, flux: np.ndarray) -> dict[str, float]:
    names = {
        (5.0, 0.0): "cavity",
        (5.0, 95.0): "central absorber",
        (50.0, 50.0): "outer wall",
        (90.0, 10.0): "frame",
        (95.0, 5.0): "source stripe",
    }
    result = {}
    for (sigma_s, sigma_a), label in names.items():
        mask = np.isclose(problem.sigma_s, sigma_s) & np.isclose(
            problem.sigma_a, sigma_a
        )
        result[label] = float(np.sum(flux[mask]) * problem.volume)
    return result


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(True, which="major", color=GRID, linewidth=0.7, alpha=0.65)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#cbd2dc")


def _backup_once(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _records(data: dict[str, object], key: str) -> list[dict[str, object]]:
    explicit = data.get(key)
    if isinstance(explicit, list):
        return explicit
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("dense convergence file does not contain records")
    if key == "spatial_records":
        return [
            record
            for record in records
            if record["n_mu"] == 4
            and record["n_phi"] == 32
            and record["nx"] in (65, 130, 260, 390, 520)
        ]
    return [record for record in records if record["nx"] == 150]


def _find_record(
    records: list[dict[str, object]], nx: int, n_mu: int, n_phi: int
) -> dict[str, object]:
    return next(
        record
        for record in records
        if record["nx"] == nx
        and record["ny"] == nx
        and record["n_mu"] == n_mu
        and record["n_phi"] == n_phi
    )


def build_report(run_dir: Path, dpi: int) -> dict[str, object]:
    dense_path = run_dir / "dense_convergence.json"
    legacy_dense_path = run_dir / "dense_convergence.json.positive_upwind"
    mc_path = run_dir / "mc_convergence.json"
    mc_fields_path = run_dir / "mc_convergence.npz"
    stage8_path = run_dir / "hohlraum_stage8_ultra.npz"
    metrics_path = run_dir / "metrics.json"
    for path in (dense_path, legacy_dense_path, mc_path, mc_fields_path, stage8_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    dense_data = json.loads(dense_path.read_text(encoding="utf-8"))
    legacy_dense_data = json.loads(legacy_dense_path.read_text(encoding="utf-8"))
    mc_data = json.loads(mc_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    spatial = sorted(_records(dense_data, "spatial_records"), key=lambda record: record["nx"])
    angular = sorted(
        _records(dense_data, "angular_records"),
        key=lambda record: record["n_mu"] * record["n_phi"],
    )
    legacy_spatial = sorted(
        _records(legacy_dense_data, "spatial_records"), key=lambda record: record["nx"]
    )
    legacy_angular = _records(legacy_dense_data, "angular_records")
    if len(spatial) < 4 or len(angular) < 3 or len(legacy_spatial) < 4:
        raise ValueError("the convergence inputs are incomplete")
    if dense_data.get("spatial_scheme") != "diamond":
        raise ValueError("dense_convergence.json must contain diamond-difference results")

    config = metrics["config"]
    if config.get("dense_spatial_scheme") != "diamond":
        raise ValueError("Stage-8 Hohlraum field is not marked as diamond difference")

    with np.load(stage8_path) as archive:
        cfm = np.asarray(archive["genvr_cfm_response"], dtype=np.float64)
        dense = np.asarray(archive["dense_sn_reference"], dtype=np.float64)
    with np.load(mc_fields_path) as archive:
        aggregate_mc = np.asarray(archive["aggregate_flux"], dtype=np.float64)

    problem = make_hohlraum_problem(cfm.shape[1], cfm.shape[0])
    cfm_track = float(np.sum(cfm) * problem.volume)
    dense_track = float(np.sum(dense) * problem.volume)
    mc_track = float(mc_data["track_length_mean"])
    mc_interval = [float(value) for value in mc_data["track_length_95ci"]]

    aligned_spatial = [
        record
        for record in spatial
        if bool(record.get("geometry_boundaries_cell_aligned", record["nx"] % 130 == 0))
    ]
    if len(aligned_spatial) < 3:
        raise ValueError("at least three geometry-aligned DD meshes are required")
    finest_dense_track = float(aligned_spatial[-1]["integrated_track"])
    aligned_refinement_change = (
        finest_dense_track / float(aligned_spatial[-2]["integrated_track"]) - 1.0
    )
    aligned_sequence_converged = abs(aligned_refinement_change) < 5.0e-3

    baseline_angular = _find_record(angular, 150, 4, 32)
    finest_angular = max(angular, key=lambda record: record["n_mu"] * record["n_phi"])
    angular_change = float(
        float(finest_angular["integrated_track"])
        / float(baseline_angular["integrated_track"])
        - 1.0
    )
    legacy_baseline = _find_record(legacy_angular, 150, 4, 32)
    legacy_track = float(legacy_baseline["integrated_track"])

    material_tracks = {
        "GenVR/CFM": _material_tracks(problem, cfm),
        "Dense DD 150²": _material_tracks(problem, dense),
        "Global MC 160k": _material_tracks(problem, aggregate_mc),
    }
    cfm_absorption = float(np.sum(problem.sigma_a * cfm) * problem.volume)
    dense_absorption = float(np.sum(problem.sigma_a * dense) * problem.volume)
    mc_absorption = float(np.sum(problem.sigma_a * aggregate_mc) * problem.volume)

    dense_bias = dense_track / mc_track - 1.0
    finest_dense_bias = finest_dense_track / mc_track - 1.0
    cfm_bias = cfm_track / mc_track - 1.0
    legacy_bias = legacy_track / mc_track - 1.0
    stage_to_aligned_offset = dense_track / finest_dense_track - 1.0
    diagnosis = {
        "classification": "positive_upwind_diffusion_removed_by_diamond_difference",
        "dense_spatially_converged": False,
        "aligned_dd_sequence_converged": aligned_sequence_converged,
        "aligned_390_to_520_change_fraction": aligned_refinement_change,
        "angular_refinement_change_fraction": angular_change,
        "legacy_upwind_150_integrated_track_bias_vs_mc": legacy_bias,
        "diamond_150_integrated_track_bias_vs_mc": dense_bias,
        "diamond_520_integrated_track_bias_vs_mc": finest_dense_bias,
        "cfm_integrated_track_bias_vs_mc": cfm_bias,
        "diamond_150_vs_aligned_520_offset": stage_to_aligned_offset,
        "display_note": f"finite-grid DD\ntrack bias vs MC  {100.0 * dense_bias:+.1f}%",
        "interpretation": (
            "Replacing positive upwind with diamond difference plus conservative zero-flux "
            "fixup removes the dominant artificial transverse diffusion: the 150² integrated-"
            "track bias changes from "
            f"{100.0 * legacy_bias:+.1f}% to {100.0 * dense_bias:+.1f}% versus aggregate MC. "
            "The 150² mesh does not align with every material boundary, so it remains a "
            "finite-grid diagnostic rather than continuum truth. The aligned 130/260/390/520 "
            "sequence is nearly stable at its finest two meshes."
        ),
    }
    _backup_once(mc_path, ".positive_upwind")
    mc_data["dense_spatial_scheme"] = "diamond"
    mc_data["dense_fixup"] = "conservative_zero_flux"
    mc_data["dense_fixup_fraction"] = float(baseline_angular["fixup_fraction"])
    mc_data["dense_double_fixup_fraction"] = float(
        baseline_angular["double_fixup_fraction"]
    )
    if "legacy_positive_upwind_dense" not in mc_data:
        mc_data["legacy_positive_upwind_dense"] = {
            "integrated_track": legacy_track,
            "vs_aggregate_mc": mc_data["dense_vs_aggregate_mc"],
        }
    mc_data["dense_integrated_track"] = dense_track
    mc_data["dense_vs_aggregate_mc"] = flux_shape_metrics(dense, aggregate_mc)
    mc_path.write_text(json.dumps(mc_data, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "schema_version": "2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": "hohlraum",
        "audited_stage8_mesh": [int(cfm.shape[1]), int(cfm.shape[0])],
        "audited_dense_angular_order": [4, 32],
        "audited_dense_spatial_scheme": "diamond",
        "dense_fixup": "conservative_zero_flux",
        "global_mc": {
            "histories": int(mc_data["total_histories"]),
            "integrated_track_mean": mc_track,
            "integrated_track_95ci": mc_interval,
            "positive_tally_fraction": float(mc_data["positive_tally_fraction"]),
        },
        "genvr_cfm": {
            "integrated_track": cfm_track,
            "absorption": cfm_absorption,
            "vs_aggregate_mc": flux_shape_metrics(cfm, aggregate_mc),
        },
        "dense_sn_150": {
            "integrated_track": dense_track,
            "absorption": dense_absorption,
            "fixup_fraction": float(baseline_angular["fixup_fraction"]),
            "double_fixup_fraction": float(baseline_angular["double_fixup_fraction"]),
            "geometry_boundaries_cell_aligned": False,
            "vs_aggregate_mc": flux_shape_metrics(dense, aggregate_mc),
        },
        "legacy_positive_upwind_150": {
            "integrated_track": legacy_track,
            "bias_vs_aggregate_mc": legacy_bias,
        },
        "dense_aligned_finest": aligned_spatial[-1],
        "aggregate_mc_absorption": mc_absorption,
        "dense_spatial_convergence": spatial,
        "dense_angular_convergence": angular,
        "legacy_positive_upwind_spatial_convergence": legacy_spatial,
        "material_integrated_tracks": material_tracks,
        "diagnosis": diagnosis,
    }

    audit_path = run_dir / "reference_audit.json"
    figure_path = run_dir / "reference_audit.png"
    _backup_once(audit_path, ".positive_upwind")
    _backup_once(figure_path, ".positive_upwind")
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), facecolor=BACKGROUND)
    for axis in axes.flat:
        _style_axis(axis)

    axis = axes[0, 0]
    mesh_sizes = np.asarray([record["nx"] for record in spatial])
    dense_tracks = np.asarray([record["integrated_track"] for record in spatial])
    legacy_mesh_sizes = np.asarray([record["nx"] for record in legacy_spatial])
    legacy_tracks = np.asarray([record["integrated_track"] for record in legacy_spatial])
    axis.plot(
        legacy_mesh_sizes,
        legacy_tracks,
        "o--",
        color="#d97706",
        linewidth=1.4,
        alpha=0.75,
        label="Legacy positive upwind",
    )
    axis.plot(
        mesh_sizes,
        dense_tracks,
        "o-",
        color="#7f56d9",
        linewidth=1.8,
        label="DD + zero-flux fixup",
    )
    axis.scatter(
        [150],
        [dense_track],
        marker="*",
        s=90,
        color="#4169a1",
        zorder=4,
        label="Stage-8 DD 150²",
    )
    axis.axhspan(mc_interval[0], mc_interval[1], color="#12b76a", alpha=0.18, label="MC 95% CI")
    axis.axhline(mc_track, color="#12b76a", linewidth=1.4)
    axis.axhline(cfm_track, color="#4169a1", linewidth=1.2, linestyle=":", label="GenVR/CFM")
    axis.set_xscale("log", base=2)
    axis.set_xticks(mesh_sizes, [str(value) for value in mesh_sizes])
    axis.set_xlabel("Cells per side")
    axis.set_ylabel("Integrated track length")
    axis.set_title("(a) Spatial scheme and mesh convergence", loc="left", color=TEXT, fontweight="semibold")
    axis.legend(frameon=False, fontsize=7.5)

    axis = axes[0, 1]
    directions = np.asarray([record["n_mu"] * record["n_phi"] for record in angular])
    angular_tracks = np.asarray([record["integrated_track"] for record in angular])
    labels = [f"{record['n_mu']}×{record['n_phi']}" for record in angular]
    axis.plot(directions, angular_tracks, "o-", color="#7f56d9", linewidth=1.8)
    axis.axhspan(mc_interval[0], mc_interval[1], color="#12b76a", alpha=0.18)
    for x_value, y_value, label in zip(directions, angular_tracks, labels):
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    axis.set_xlabel("Angular directions before z-symmetry merging")
    axis.set_ylabel("Integrated track length")
    axis.set_title("(b) DD angular convergence at 150²", loc="left", color=TEXT, fontweight="semibold")

    axis = axes[1, 0]
    material_order = ["cavity", "source stripe", "frame", "outer wall", "central absorber"]
    x_positions = np.arange(len(material_order), dtype=float)
    width = 0.24
    colors = ("#4169a1", "#7f56d9", "#12b76a")
    for index, (method, values) in enumerate(material_tracks.items()):
        axis.bar(
            x_positions + (index - 1) * width,
            [values[name] for name in material_order],
            width=width,
            color=colors[index],
            alpha=0.88,
            label=method,
        )
    axis.set_yscale("log")
    axis.set_xticks(x_positions, material_order, rotation=18, ha="right")
    axis.set_ylabel("Integrated track by material")
    axis.set_title("(c) Material-resolved track length", loc="left", color=TEXT, fontweight="semibold")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    methods = ["GenVR/CFM", "Upwind 150²", "DD 150²", "DD 520²"]
    biases = 100.0 * np.asarray([cfm_bias, legacy_bias, dense_bias, finest_dense_bias])
    bars = axis.bar(
        methods,
        biases,
        color=("#4169a1", "#d97706", "#7f56d9", "#6941c6"),
        width=0.62,
    )
    axis.axhline(0.0, color="#344054", linewidth=1.0)
    axis.set_ylabel("Integrated-track bias vs aggregate MC [%]")
    axis.set_title("(d) Independent-reference bias", loc="left", color=TEXT, fontweight="semibold")
    axis.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, biases):
        vertical_alignment = "bottom" if value >= 0.0 else "top"
        offset = 0.8 if value >= 0.0 else -0.8
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.1f}%",
            ha="center",
            va=vertical_alignment,
            color=TEXT,
            fontsize=8,
        )

    figure.suptitle(
        "Hohlraum reference audit — diamond difference removes the upwind diffusion bias",
        x=0.06,
        y=0.985,
        ha="left",
        color=TEXT,
        fontsize=15,
        fontweight="semibold",
    )
    figure.text(
        0.06,
        0.025,
        f"160,000-history MC track = {mc_track:.5f} "
        f"(95% CI {mc_interval[0]:.5f}–{mc_interval[1]:.5f}); "
        f"DD 150² = {dense_track:.5f}; aligned DD 520² = {finest_dense_track:.5f}. "
        "Dense S_N remains a finite-grid diagnostic, not continuum truth.",
        color=MUTED,
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.12, wspace=0.22, hspace=0.34)
    figure.savefig(
        figure_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the Hohlraum Dense-S_N reference audit")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/hohlraum")
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    report = build_report(run_dir, args.dpi)
    print(json.dumps(report["diagnosis"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
