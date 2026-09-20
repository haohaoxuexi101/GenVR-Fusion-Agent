#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmc.benchmarks2d import (  # noqa: E402
    make_hohlraum_problem,
    make_lattice_problem,
    make_tokamak_coils_problem,
    make_tokamak_square_discrete_problem,
)
from gmc.stage8_reporting import plot_stage8_case  # noqa: E402


FIELD_ALIASES = {
    "genvr_cfm_response": ("genvr_cfm_response", "cfm"),
    "projected_local_mc_response": ("projected_local_mc_response", "oracle"),
    "dense_sn_reference": ("dense_sn_reference", "dense"),
    "global_history_mc": ("global_history_mc", "mc"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _problem(case: str, nx: int, ny: int):
    if case == "lattice":
        return make_lattice_problem(nx, ny)
    if case == "hohlraum":
        return make_hohlraum_problem(nx, ny)
    if case == "iter_r":
        return make_tokamak_square_discrete_problem(nx, ny)
    if case == "iter_a":
        return make_tokamak_coils_problem(nx, ny)
    raise ValueError(f"unsupported Stage-8 case {case!r}")


def _load_field(archive: np.lib.npyio.NpzFile, canonical_name: str) -> np.ndarray | None:
    for name in FIELD_ALIASES[canonical_name]:
        if name in archive.files:
            values = np.asarray(archive[name], dtype=np.float64)
            return values if values.size else None
    return None


def _dense_audit_note(
    run_dir: Path,
    case: str,
    nx: int,
    ny: int,
    spatial_scheme: str,
) -> str | None:
    audit_path = run_dir / "reference_audit.json"
    if not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("case") != case or audit.get("audited_stage8_mesh") != [nx, ny]:
        return None
    if audit.get("audited_dense_spatial_scheme") != spatial_scheme:
        return None
    diagnosis = audit.get("diagnosis", {})
    if diagnosis.get("dense_spatially_converged", True):
        return None
    note = diagnosis.get("display_note")
    return str(note) if note else "not grid-converged"


def _replot(run_dir: Path, dpi: int) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing Stage-8 metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics.get("config", {})
    case = str(config["case"])
    nx = int(config["nx"])
    ny = int(config["ny"])
    npz_path = run_dir / f"{case}_stage8_ultra.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing Stage-8 field archive: {npz_path}")
    image_path = run_dir / f"{case}_stage8_ultra.png"
    legacy_image = run_dir / f"{case}_stage8_ultra.legacy.png"
    if image_path.exists() and not legacy_image.exists():
        shutil.copy2(image_path, legacy_image)

    with np.load(npz_path) as archive:
        cfm = _load_field(archive, "genvr_cfm_response")
        dense = _load_field(archive, "dense_sn_reference")
        global_mc = _load_field(archive, "global_history_mc")
    if cfm is None:
        raise ValueError(f"{npz_path} does not contain a GenVR/CFM field")

    problem = _problem(case, nx, ny)
    dense_spatial_scheme = str(config.get("dense_spatial_scheme", "upwind"))
    dense_scheme_label = {
        "diamond": "diamond-difference",
        "upwind": "positive-upwind",
    }[dense_spatial_scheme]
    dense_note = _dense_audit_note(run_dir, case, nx, ny, dense_spatial_scheme)
    plot_stage8_case(
        problem,
        cfm,
        None,
        dense,
        global_mc,
        run_dir,
        case,
        n_pos=int(config["n_pos"]),
        n_mu=int(config["n_mu"]),
        n_phi=int(config["n_phi"]),
        mc_histories=int(config.get("mc_histories", 0)) or None,
        dense_note=dense_note,
        dense_spatial_scheme=dense_spatial_scheme,
        dpi=dpi,
    )
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "derived_without_transport_recalculation": True,
        "source_metrics": metrics_path.name,
        "source_metrics_sha256": _sha256(metrics_path),
        "source_fields": npz_path.name,
        "source_fields_sha256": _sha256(npz_path),
        "legacy_figure": legacy_image.name if legacy_image.exists() else None,
        "figure": image_path.name,
        "case": case,
        "phase_space": {
            "n_pos": int(config["n_pos"]),
            "n_mu": int(config["n_mu"]),
            "n_phi": int(config["n_phi"]),
        },
        "global_history_mc_histories": int(config.get("mc_histories", 0)),
        "display_changes": [
            "Uses canonical method names instead of Exact-MC/oracle display labels.",
            "Uses one common absolute normalization for all flux panels.",
            "Treats finite-history global MC as a qualitative statistical reference.",
            f"Uses dense {dense_scheme_label} S_N as the deterministic visual diagnostic for GenVR/CFM.",
            "Shows a signed normalized shape ratio between GenVR/CFM and dense finite-angle S_N.",
            "Adds a cellwise CFM-versus-dense shape-agreement panel.",
            "Mutes zero-tally global-MC cells and below-scale ratio cells in light gray.",
            "Uses a compact publication-style layout with shared color scales and comparison metrics.",
        ],
        "dense_spatial_scheme": dense_spatial_scheme,
        "dense_reference_audit": "reference_audit.json" if dense_note else None,
    }
    (run_dir / "figure_postprocess.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw saved Stage-8 NPZ artifacts without rerunning transport."
    )
    parser.add_argument("run_dirs", nargs="+", help="one or more Stage-8 output directories")
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args()

    manifests = []
    for raw_path in args.run_dirs:
        run_dir = Path(raw_path)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        manifests.append(_replot(run_dir, args.dpi))
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
