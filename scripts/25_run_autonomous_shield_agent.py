#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ray_agent.metrics import finite_json
from ray_agent.shield_agent_protocol import DeepSeekShieldToolAgent
from ray_agent.shield_agent_reporting import write_autonomous_shield_figures
from ray_agent.shield_agent_tools import ShieldAgentBudgets, ShieldToolEnvironment
from ray_agent.shield_agent_verifier import (
    ShieldAgentVerifier,
    ShieldVerificationCriteria,
)
from ray_agent.shield_autonomous_agent import AutonomousShieldAgent
from ray_agent.shield_design import (
    DiffusionShieldEvaluator,
    GenVRShieldEvaluator,
)


DEFAULT_LIBRARY = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_cfm_112x112_p4_m4_f32_s1000_rk12_5f816a4032cd_direct.npz"
)
DEFAULT_INTERNAL = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_cfm_internal_112x112_p4_m4_f32_s1000_rk12_2e0c41f7562b.npz"
)
DEFAULT_EXACT_LIBRARY = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_oracle_112x112_p4_m4_f32_s1000_rk12_exact_direct.npz"
)
DEFAULT_EXACT_INTERNAL = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_oracle_internal_112x112_p4_m4_f32_s1000_rk12_exact.npz"
)


def gmc_state_count(n_pos: int, n_mu: int, n_phi: int) -> int:
    return int(n_pos) * int(n_mu) * int(n_phi)


class TrajectoryLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **finite_json(payload),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default="configs/gmc_material_discovery_agent.json",
    )
    pre_args, _ = pre_parser.parse_known_args()
    config_path = Path(pre_args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    defaults: dict[str, Any] = {}
    if config_path.exists():
        defaults = json.loads(config_path.read_text(encoding="utf-8"))
    if "max_generated_structures" not in defaults and "max_diffusion_evaluations" in defaults:
        defaults["max_generated_structures"] = defaults["max_diffusion_evaluations"]
    if "max_structures_per_call" not in defaults and "max_diffusion_per_call" in defaults:
        defaults["max_structures_per_call"] = defaults["max_diffusion_per_call"]
    defaults.pop("max_diffusion_evaluations", None)
    defaults.pop("max_diffusion_per_call", None)

    parser = argparse.ArgumentParser(
        description=(
            "Run the autonomous GMC-guided material-layout discovery agent. "
            "A real API policy is mandatory and no deterministic fallback exists."
        ),
        parents=[pre_parser],
    )
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--memory-window", type=int, default=4)
    parser.add_argument("--max-response-failures", type=int, default=3)
    parser.add_argument("--max-visible-candidates", type=int, default=8)
    parser.add_argument("--max-visible-beam", type=int, default=4)
    parser.add_argument(
        "--discovery-mode",
        choices=("gmc_only", "diffusion_prefilter"),
        default="gmc_only",
    )
    parser.add_argument("--fast-nx", type=int, default=56)
    parser.add_argument("--fast-ny", type=int, default=56)
    parser.add_argument("--risk-aversion", type=float, default=0.35)
    parser.add_argument("--exploration-strength", type=float, default=0.04)
    parser.add_argument("--minimum-verified-improvement", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=260918)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--response-library", default=str(DEFAULT_LIBRARY))
    parser.add_argument("--internal-response", default=str(DEFAULT_INTERNAL))
    parser.add_argument("--gmc-nx", type=int, default=112)
    parser.add_argument("--gmc-ny", type=int, default=112)
    parser.add_argument("--gmc-n-pos", type=int, default=4)
    parser.add_argument("--gmc-n-mu", type=int, default=4)
    parser.add_argument("--gmc-n-phi", type=int, default=32)
    parser.add_argument("--exact-response-library", default=str(DEFAULT_EXACT_LIBRARY))
    parser.add_argument("--exact-internal-response", default=str(DEFAULT_EXACT_INTERNAL))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--thinking", default="enabled")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout-s", type=int, default=120)
    parser.add_argument("--response-retries", type=int, default=2)
    parser.add_argument(
        "--max-generated-structures",
        "--max-diffusion-evaluations",
        dest="max_generated_structures",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--max-structures-per-call",
        "--max-diffusion-per-call",
        dest="max_structures_per_call",
        type=int,
        default=96,
    )
    parser.add_argument("--max-genvr-candidates", type=int, default=4)
    parser.add_argument("--max-gmc-per-call", type=int, default=8)
    parser.add_argument("--max-exact-candidates", type=int, default=2)
    parser.add_argument("--max-mc-root-histories", type=int, default=20_000)
    parser.add_argument("--max-mc-histories-per-call", type=int, default=2_500)
    parser.add_argument("--max-mc-replicates-per-call", type=int, default=4)
    parser.add_argument("--max-mc-grid", type=int, default=56)
    parser.add_argument("--maximum-consistency-z", type=float, default=2.0)
    parser.add_argument("--maximum-far-difference-z", type=float, default=-1.96)
    parser.add_argument("--minimum-mc-histories-per-method", type=int, default=1000)
    parser.add_argument("--minimum-mc-replicates", type=int, default=2)
    parser.add_argument("--minimum-mc-grid", type=int, default=7)
    parser.add_argument("--minimum-gmc-candidates-before-mc", type=int, default=1)
    parser.add_argument(
        "--minimum-gmc-guided-candidates-before-mc",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--require-projected-mc-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-projected-mc-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-far-field-ci-separation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--outdir", default=None)
    parser.set_defaults(**defaults)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_dir: Path,
    config_path: Path,
    policy: DeepSeekShieldToolAgent,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tracked = [
        config_path,
        ROOT / "scripts/25_run_autonomous_shield_agent.py",
        ROOT / "scripts/run_gmc_material_discovery_agent.sh",
        ROOT / "ray_agent/shield_agent_protocol.py",
        ROOT / "ray_agent/shield_agent_tools.py",
        ROOT / "ray_agent/shield_live_reporting.py",
        ROOT / "ray_agent/shield_agent_verifier.py",
        ROOT / "ray_agent/shield_autonomous_agent.py",
        ROOT / "ray_agent/shield_agent_reporting.py",
        ROOT / "ray_agent/shield_design.py",
        ROOT / "ray_agent/shield_certification.py",
        _resolve(args.response_library),
        _resolve(args.internal_response),
    ]
    if args.enable_projected_mc_audit or args.require_projected_mc_audit:
        tracked.extend(
            [
                _resolve(args.exact_response_library),
                _resolve(args.exact_internal_response),
            ]
        )
    files = {}
    for path in tracked:
        if path.exists() and path.is_file():
            key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            files[key] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "agent_policy": type(policy).__name__,
        "provider": "DeepSeek",
        "model": policy.model,
        "llm_required": True,
        "deterministic_policy_fallback": False,
        "human_intervention": "None after launch.",
        "discovery_mode": args.discovery_mode,
        "diffusion_model_used": args.discovery_mode == "diffusion_prefilter",
        "gmc_discretization": {
            "spatial_grid": [args.gmc_nx, args.gmc_ny],
            "n_pos": args.gmc_n_pos,
            "n_mu": args.gmc_n_mu,
            "n_phi": args.gmc_n_phi,
            "n_state": gmc_state_count(args.gmc_n_pos, args.gmc_n_mu, args.gmc_n_phi),
        },
        "files": files,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _write_summary_markdown(
    summary: dict[str, Any],
    path: Path,
    figures: dict[str, str] | None = None,
) -> None:
    decision = summary["verifier_decision"]
    lines = [
        "# Autonomous GenShield Agent result",
        "",
        f"- Final claim: `{summary['final_claim']}`",
        f"- Candidate: `{summary['candidate_signature']}`",
        f"- Termination: `{summary['termination']}`",
        f"- Agent steps: `{summary['steps_completed']}`",
        f"- Invalid LLM responses recovered/rejected: `{summary['response_failures']}`",
        f"- Verifier termination accepted: `{decision['accepted']}`",
        f"- Scientifically verified design: `{summary['final_claim'] == 'verified_improvement'}`",
        f"- Discovery mode: `{summary.get('discovery_mode', 'unknown')}`",
        f"- Diffusion model used: `{summary.get('discovery_mode') == 'diffusion_prefilter'}`",
        f"- GMC-guided structures screened: `{summary.get('discovery_progress', {}).get('gmc_guided_and_screened', 0)}`",
        "- Deterministic policy fallback: `false`",
        "",
        "The LLM selected one scientific tool per step. Geometry remained compiler-",
        "constrained, and the independent verifier alone controlled the final claim.",
    ]
    if figures:
        lines.extend(["", "## Figures", ""])
        for name, figure_path in figures.items():
            relative = Path(figure_path)
            try:
                relative = relative.relative_to(path.parent)
            except ValueError:
                pass
            lines.append(f"- `{name}`: `{relative}`")
    if summary.get("report_error"):
        lines.extend(["", f"- Visualization report error: `{summary['report_error']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    run_id = f"autoshield-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = (
        _resolve(args.outdir)
        if args.outdir
        else ROOT / "outputs" / "autonomous_shield_agent" / run_id
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = DeepSeekShieldToolAgent(
        model=args.model,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.llm_max_tokens,
        timeout_s=args.llm_timeout_s,
        response_retries=args.response_retries,
    )
    budgets = ShieldAgentBudgets(
        max_diffusion_evaluations=args.max_generated_structures,
        max_diffusion_per_call=args.max_structures_per_call,
        max_genvr_candidates=args.max_genvr_candidates,
        max_gmc_per_call=args.max_gmc_per_call,
        max_exact_candidates=args.max_exact_candidates,
        max_mc_root_histories=args.max_mc_root_histories,
        max_mc_histories_per_call=args.max_mc_histories_per_call,
        max_mc_replicates_per_call=args.max_mc_replicates_per_call,
        max_mc_grid=args.max_mc_grid,
    )
    criteria = ShieldVerificationCriteria(
        maximum_abs_unbiased_consistency_z=args.maximum_consistency_z,
        maximum_far_field_difference_z=args.maximum_far_difference_z,
        minimum_histories_per_method=args.minimum_mc_histories_per_method,
        minimum_replicates=args.minimum_mc_replicates,
        minimum_mc_grid=args.minimum_mc_grid,
        require_far_field_ci_separation=args.require_far_field_ci_separation,
        require_projected_mc_audit=args.require_projected_mc_audit,
    )
    gmc_evaluator = GenVRShieldEvaluator(
        _resolve(args.response_library),
        _resolve(args.internal_response),
        nx=args.gmc_nx,
        ny=args.gmc_ny,
        n_pos=args.gmc_n_pos,
        n_mu=args.gmc_n_mu,
        n_phi=args.gmc_n_phi,
        device=args.device,
        fidelity="genvr-cfm-response",
    )
    exact_evaluator = None
    if args.enable_projected_mc_audit or args.require_projected_mc_audit:
        exact_evaluator = GenVRShieldEvaluator(
            _resolve(args.exact_response_library),
            _resolve(args.exact_internal_response),
            nx=args.gmc_nx,
            ny=args.gmc_ny,
            n_pos=args.gmc_n_pos,
            n_mu=args.gmc_n_mu,
            n_phi=args.gmc_n_phi,
            device=args.device,
            fidelity="exact-local-mc-response",
        )
    environment = ShieldToolEnvironment(
        fast_evaluator=(
            DiffusionShieldEvaluator(args.fast_nx, args.fast_ny)
            if args.discovery_mode == "diffusion_prefilter"
            else None
        ),
        genvr_evaluator=gmc_evaluator,
        exact_evaluator=exact_evaluator,
        output_dir=output_dir,
        budgets=budgets,
        seed=args.seed,
        risk_aversion=args.risk_aversion,
        exploration_strength=args.exploration_strength,
        minimum_verified_improvement=args.minimum_verified_improvement,
        require_projected_mc_audit=args.require_projected_mc_audit,
        minimum_gmc_candidates_before_mc=args.minimum_gmc_candidates_before_mc,
        minimum_gmc_guided_candidates_before_mc=(
            args.minimum_gmc_guided_candidates_before_mc
        ),
        discovery_mode=args.discovery_mode,
        minimum_mc_histories_per_method=args.minimum_mc_histories_per_method,
        minimum_mc_replicates=args.minimum_mc_replicates,
        minimum_mc_grid=args.minimum_mc_grid,
    )
    verifier = ShieldAgentVerifier(criteria)
    logger = TrajectoryLogger(output_dir / "trajectory.jsonl", run_id)

    config_path = _resolve(args.config)
    effective_config = vars(args).copy()
    effective_config.update(
        {
            "config": str(config_path),
            "outdir": str(output_dir),
            "effective_model": policy.model,
            "llm_required": True,
            "deterministic_policy_fallback": False,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(effective_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = _write_manifest(output_dir, config_path, policy, args)
    logger.write(
        "run_started",
        {
            "config": effective_config,
            "manifest": manifest,
            "verifier_criteria": criteria.__dict__,
            "agent_contract": {
                "one_tool_per_step": True,
                "geometry_compiler_controlled": True,
                "primary_design_evaluator": "cached GenVR/GMC response operator",
                "diffusion_model_used": args.discovery_mode == "diffusion_prefilter",
                "physical_certifier": "unbiased global Monte Carlo",
                "live_gmc_candidate_visualization": (
                    "geometry plus GMC scalar-flux field after every new GMC evaluation"
                ),
                "required_discovery_cycle": (
                    "generate -> GMC screen -> GMC-guided regenerate -> GMC rescreen"
                ),
                "minimum_gmc_candidates_before_mc": (
                    args.minimum_gmc_candidates_before_mc
                ),
                "minimum_gmc_guided_candidates_before_mc": (
                    args.minimum_gmc_guided_candidates_before_mc
                ),
                "minimum_mc_histories_per_design_and_method": (
                    args.minimum_mc_histories_per_method
                ),
                "minimum_mc_replicates": args.minimum_mc_replicates,
                "minimum_mc_grid": [args.minimum_mc_grid, args.minimum_mc_grid],
                "projected_local_mc_role": (
                    "required learned-kernel audit"
                    if args.require_projected_mc_audit
                    else "optional learned-kernel audit"
                ),
                "llm_may_change_verifier_thresholds": False,
                "api_failure_policy": "abort run",
                "invalid_response_policy": (
                    "retry with the same LLM and no deterministic fallback"
                ),
            },
        },
    )

    agent = AutonomousShieldAgent(
        policy=policy,
        environment=environment,
        verifier=verifier,
        max_steps=args.max_steps,
        memory_window=args.memory_window,
        max_response_failures=args.max_response_failures,
        max_visible_candidates=args.max_visible_candidates,
        max_visible_beam=args.max_visible_beam,
        logger=logger,
    )
    result = agent.run()
    summary = result.to_dict(environment)
    summary["campaign_configuration"] = {
        "gmc_grid": [args.gmc_nx, args.gmc_ny],
        "gmc_phase_space": {
            "n_pos": args.gmc_n_pos,
            "n_mu": args.gmc_n_mu,
            "n_phi": args.gmc_n_phi,
            "n_state": gmc_state_count(
                args.gmc_n_pos,
                args.gmc_n_mu,
                args.gmc_n_phi,
            ),
        },
        "mc_precision_floor": {
            "grid": [args.minimum_mc_grid, args.minimum_mc_grid],
            "histories_per_design_and_method": (
                args.minimum_mc_histories_per_method
            ),
            "replicates": args.minimum_mc_replicates,
            "weight_window_guide": "candidate-specific GMC flux",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(finite_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result.candidate_signature is not None:
        evidence = environment.candidate(result.candidate_signature)
        if evidence is not None:
            (output_dir / "best_design.json").write_text(
                json.dumps(evidence.design.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    report_error = None
    try:
        figures = write_autonomous_shield_figures(
            summary,
            output_dir,
            trajectory_path=output_dir / "trajectory.jsonl",
            environment=environment,
            dpi=240,
        )
    except Exception as exc:
        figures = {}
        report_error = f"{type(exc).__name__}: {exc}"
        logger.write("report_failure", {"error": report_error})
    figure_manifest = {
        name: str(Path(path).relative_to(output_dir))
        for name, path in figures.items()
    }
    summary["figures"] = figure_manifest
    summary["report_error"] = report_error
    (output_dir / "summary.json").write_text(
        json.dumps(finite_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.write(
        "report_generated",
        {"figures": figure_manifest, "report_error": report_error},
    )
    _write_summary_markdown(summary, output_dir / "summary.md", figure_manifest)
    print(
        json.dumps(
            {
                "run_dir": str(output_dir),
                "final_claim": result.final_claim,
                "candidate_signature": result.candidate_signature,
                "figures": figures,
                "report_error": report_error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
