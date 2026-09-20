#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ray_agent.shield_agent_reporting import write_autonomous_shield_figures
from ray_agent.shield_design import GenVRShieldEvaluator, ShieldDesign, baseline_lattice_design


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _phase_space_from_path(path: str) -> tuple[int, int, int]:
    match = re.search(r"_p(\d+)_m(\d+)_f(\d+)(?:_|\.)", Path(path).name)
    if match is None:
        raise ValueError(f"cannot infer n_pos/n_mu/n_phi from response cache {path!r}")
    return tuple(int(value) for value in match.groups())


def _load_verified_design(run_dir: Path) -> ShieldDesign:
    path = run_dir / "best_design.json"
    if not path.exists():
        raise FileNotFoundError(f"missing verified design: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return ShieldDesign(
        absorbers=tuple(tuple(int(item) for item in cell) for cell in value["absorbers"]),
        grid_size=int(value.get("grid_size", 7)),
        source_cell=tuple(int(item) for item in value.get("source_cell", [3, 3])),
    )


def _recompute_response_fields(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    response_library = str(config["response_library"])
    internal_response = str(config["internal_response"])
    projected_library = str(config["exact_response_library"])
    projected_internal = str(config["exact_internal_response"])
    n_pos, n_mu, n_phi = _phase_space_from_path(response_library)
    device = str(config.get("device", "auto"))
    baseline = baseline_lattice_design()
    candidate = _load_verified_design(run_dir)
    genvr = GenVRShieldEvaluator(
        _resolve(response_library),
        _resolve(internal_response),
        n_pos=n_pos,
        n_mu=n_mu,
        n_phi=n_phi,
        device=device,
        fidelity="genvr-cfm-response",
    )
    projected = GenVRShieldEvaluator(
        _resolve(projected_library),
        _resolve(projected_internal),
        n_pos=n_pos,
        n_mu=n_mu,
        n_phi=n_phi,
        device=device,
        fidelity="exact-local-mc-response",
    )
    print(
        "Recomputing deterministic response fields from cached local operators: "
        f"n_pos={n_pos}, n_mu={n_mu}, n_phi={n_phi}",
        flush=True,
    )
    baseline_genvr = genvr.evaluate(baseline)
    candidate_genvr = genvr.evaluate(candidate)
    baseline_projected = projected.evaluate(baseline)
    candidate_projected = projected.evaluate(candidate)
    discretization = genvr.discretization
    return {
        "baseline_genvr_cfm_response": baseline_genvr.flux,
        "candidate_genvr_cfm_response": candidate_genvr.flux,
        "baseline_projected_local_mc_response": baseline_projected.flux,
        "candidate_projected_local_mc_response": candidate_projected.flux,
        "metadata": {
            "schema_version": "1.0",
            "candidate_signature": candidate.signature,
            "discretization": {
                "n_pos": int(discretization.n_pos),
                "n_mu": int(discretization.n_mu),
                "n_phi": int(discretization.n_phi),
                "n_angle": int(discretization.n_angle),
                "n_state": int(discretization.n_state),
            },
            "genvr_method": "GenVR/CFM projected response",
            "diagnostic_method": "Projected local-MC response",
            "diagnostic_scope": (
                "Both fields use the same finite interface projection; the local-MC field "
                "is a kernel diagnostic, not an exact continuum truth."
            ),
            "recomputed_from_cached_local_operators": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the visual report for an autonomous GenShield run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--recompute-responses",
        action="store_true",
        help=(
            "Recompute baseline/candidate GenVR-CFM and projected local-MC fields from "
            "the cached local response operators. No LLM or Monte Carlo certification is rerun."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary_path = run_dir / "summary.json"
    trajectory_path = run_dir / "trajectory.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing autonomous-agent summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    response_fields = _recompute_response_fields(run_dir) if args.recompute_responses else None
    figures = write_autonomous_shield_figures(
        summary,
        run_dir,
        trajectory_path=trajectory_path,
        response_fields=response_fields,
        dpi=args.dpi,
    )
    relative_figures = {
        name: str(Path(path).relative_to(run_dir))
        for name, path in figures.items()
    }
    response_metadata_path = run_dir / "response_fields.json"
    response_metadata = (
        json.loads(response_metadata_path.read_text(encoding="utf-8"))
        if response_metadata_path.exists()
        else {}
    )
    if response_metadata.get("recomputed_from_cached_local_operators"):
        response_fields_source = "recomputed_from_cached_local_operators"
    elif response_metadata:
        response_fields_source = "saved_during_original_agent_run"
    else:
        response_fields_source = "not_available"
    report = {
        "source_summary": str(summary_path.name),
        "source_trajectory": str(trajectory_path.name),
        "derived_without_new_llm_calls": True,
        "derived_without_new_transport_calls": not args.recompute_responses,
        "transport_solve_performed_in_this_redraw": bool(args.recompute_responses),
        "response_fields_source": response_fields_source,
        "response_fields_recomputed_from_cached_local_operators": bool(
            response_metadata.get("recomputed_from_cached_local_operators")
        ),
        "figures": relative_figures,
    }
    (run_dir / "visual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Autonomous GenShield visual report",
        "",
        (
            "Derived from saved run artifacts. No new LLM or Monte Carlo certification call was made; "
            + (
                "the deterministic response fields were recomputed from the original cached local operators."
                if args.recompute_responses
                else (
                    "this redraw reused deterministic response fields previously recomputed from the "
                    "original cached local operators."
                    if response_fields_source == "recomputed_from_cached_local_operators"
                    else "no new transport solve was made."
                )
            )
        ),
        "",
    ]
    lines.extend(f"- `{name}`: `{path}`" for name, path in relative_figures.items())
    (run_dir / "visual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(relative_figures, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
