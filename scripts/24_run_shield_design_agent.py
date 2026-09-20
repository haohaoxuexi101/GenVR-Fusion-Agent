#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ray_agent.shield_design import (
    DiffusionShieldEvaluator,
    GenVRShieldEvaluator,
    PhysicsGuidedProposalModel,
    ShieldSearchResult,
    ShieldStrategy,
    SmartShieldAgent,
    baseline_lattice_design,
    materialize_lattice_design,
    source_distant_mask,
)
from ray_agent.shield_certification import certify_shield_pair
from ray_agent.shield_policy import EvidenceDrivenShieldStrategist, ShieldLLMStrategist


DEFAULT_LIBRARY = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_cfm_112x112_p2_m4_f32_s1000_rk12_5f816a4032cd_direct.npz"
)
DEFAULT_INTERNAL = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_cfm_internal_112x112_p2_m4_f32_s1000_rk12_2e0c41f7562b.npz"
)
DEFAULT_EXACT_LIBRARY = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_oracle_112x112_p2_m4_f32_s1000_rk12_exact_direct.npz"
)
DEFAULT_EXACT_INTERNAL = (
    ROOT
    / "outputs/response_cache_genvr_open/"
    "lattice_oracle_internal_112x112_p2_m4_f32_s1000_rk12_exact.npz"
)


class JsonlLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default="configs/shield_design_agent.json",
    )
    pre_args, _ = pre_parser.parse_known_args()
    defaults: dict[str, Any] = {}
    config_path = Path(pre_args.config)
    if config_path.exists():
        defaults = json.loads(config_path.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="Run the multi-fidelity GenShield intelligent shielding-design agent.",
        parents=[pre_parser],
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--proposals-per-parent", type=int, default=24)
    parser.add_argument("--fast-nx", type=int, default=56)
    parser.add_argument("--fast-ny", type=int, default=56)
    parser.add_argument("--risk-aversion", type=float, default=0.35)
    parser.add_argument("--exploration-strength", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=260918)
    parser.add_argument("--high-fidelity-top-k", type=int, default=2)
    parser.add_argument("--fast-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--response-library", default=str(DEFAULT_LIBRARY))
    parser.add_argument("--internal-response", default=str(DEFAULT_INTERNAL))
    parser.add_argument(
        "--use-exact-response",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exact-response-library", default=str(DEFAULT_EXACT_LIBRARY))
    parser.add_argument("--exact-internal-response", default=str(DEFAULT_EXACT_INTERNAL))
    parser.add_argument("--exact-top-k", type=int, default=1)
    parser.add_argument("--minimum-verified-improvement", type=float, default=0.0)
    parser.add_argument(
        "--mc-certify",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--mc-nx", type=int, default=28)
    parser.add_argument("--mc-ny", type=int, default=28)
    parser.add_argument("--mc-histories", type=int, default=1000)
    parser.add_argument("--mc-replicates", type=int, default=2)
    parser.add_argument("--mc-levels", type=int, default=None)
    parser.add_argument("--mc-maximum-levels", type=int, default=12)
    parser.add_argument("--mc-flux-exponent", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--thinking", default="disabled")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--outdir",
        default="outputs/shield_design_agent/latest",
    )
    return parser.parse_args()


def _plot_result(result: ShieldSearchResult, output_path: Path) -> None:
    final_stage = result.verification_stages[-1] if result.verification_stages else None
    baseline = final_stage.baseline if final_stage is not None else result.baseline_fast
    best = final_stage.selected if final_stage is not None else result.best_fast
    problem = materialize_lattice_design(best.design, best.flux.shape[1], best.flux.shape[0])
    far_mask = source_distant_mask(problem)
    common_scale = max(
        float(np.max(baseline.flux)),
        float(np.max(best.flux)),
        1.0e-300,
    )
    normalized_baseline = baseline.flux / common_scale
    normalized_best = best.flux / common_scale
    positive = np.concatenate(
        [
            normalized_baseline[normalized_baseline > 0.0],
            normalized_best[normalized_best > 0.0],
        ]
    )
    floor = max(float(np.quantile(positive, 0.005)), 1.0e-12)
    extent = [0.0, problem.width, 0.0, problem.height]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for axis, design, title in (
        (axes[0, 0], baseline.design, "Baseline material layout"),
        (axes[0, 1], best.design, "Agent-designed material layout"),
    ):
        image = axis.imshow(
            design.mask(),
            origin="lower",
            extent=extent,
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        source_row, source_column = design.source_cell
        axis.add_patch(
            plt.Rectangle(
                (source_column, source_row),
                1.0,
                1.0,
                fill=False,
                edgecolor="cyan",
                linewidth=2.0,
                linestyle="--",
            )
        )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label="absorber occupancy")

    delta_mask = best.design.mask().astype(int) - baseline.design.mask().astype(int)
    delta = axes[0, 2].imshow(
        delta_mask,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[0, 2].set_title("Agent edits: red added, blue removed")
    fig.colorbar(delta, ax=axes[0, 2], label="material edit")

    for axis, values, title in (
        (axes[1, 0], normalized_baseline, "Baseline flux, common normalization"),
        (axes[1, 1], normalized_best, "Agent flux, common normalization"),
    ):
        image = axis.imshow(
            np.log10(np.maximum(values, floor)),
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=np.log10(floor),
            vmax=0.0,
            interpolation="nearest",
        )
        axis.contour(
            far_mask,
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="lime",
            linewidths=1.2,
        )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label="log10 normalized flux")

    round_best: list[float] = [1.0]
    current = 1.0
    maximum_round = max((record.round_index for record in result.records), default=0)
    for round_index in range(1, maximum_round + 1):
        current = min(
            [current]
            + [
                record.robust_risk
                for record in result.records
                if record.round_index == round_index
            ]
        )
        round_best.append(current)
    axes[1, 2].plot(range(len(round_best)), round_best, marker="o")
    axes[1, 2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 2].set_xlabel("agent round")
    axes[1, 2].set_ylabel("robust normalized shielding risk")
    axes[1, 2].set_title("Closed-loop design improvement")
    axes[1, 2].grid(alpha=0.25)

    for axis in axes.ravel()[:5]:
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    verified_risk = final_stage.selected_risk if final_stage is not None else None
    subtitle = (
        f"{final_stage.fidelity} risk={verified_risk:.3f}; {final_stage.decision}"
        if final_stage is not None and verified_risk is not None
        else f"diffusion-screened risk={result.best_fast_risk:.3f}"
    )
    fig.suptitle(
        "GenShield-Agent: fixed-mass generative shielding inverse design\n"
        f"{subtitle}; design={best.design.signature}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_mc_certification(
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    baseline_flux = arrays["baseline_variance_reduced_flux"]
    candidate_flux = arrays["candidate_variance_reduced_flux"]
    common_scale = max(
        float(np.max(baseline_flux)),
        float(np.max(candidate_flux)),
        1.0e-300,
    )
    positive = np.concatenate(
        [baseline_flux[baseline_flux > 0.0], candidate_flux[candidate_flux > 0.0]]
    )
    floor = max(float(np.quantile(positive / common_scale, 0.005)), 1.0e-12)
    extent = [0.0, 7.0, 0.0, 7.0]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for axis, values, title in (
        (axes[0, 0], baseline_flux / common_scale, "Baseline unbiased VR flux"),
        (axes[0, 1], candidate_flux / common_scale, "Candidate unbiased VR flux"),
    ):
        image = axis.imshow(
            np.log10(np.maximum(values, floor)),
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=np.log10(floor),
            vmax=0.0,
        )
        fig.colorbar(image, ax=axis, label="log10 flux, common normalization")
    ratio = candidate_flux / np.maximum(baseline_flux, 1.0e-300)
    ratio_image = axes[0, 2].imshow(
        np.log10(np.clip(ratio, 1.0e-3, 1.0e3)),
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[0, 2].set_title("Candidate / baseline flux")
    fig.colorbar(ratio_image, ax=axes[0, 2], label="log10 flux ratio")

    levels = arrays["candidate_importance_levels"]
    level_image = axes[1, 0].imshow(
        levels,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
    )
    axes[1, 0].set_title("Candidate GenVR-derived importance levels")
    fig.colorbar(level_image, ax=axes[1, 0], label="importance level")

    analog_error = arrays["candidate_analog_relative_error"]
    vr_error = arrays["candidate_variance_reduced_relative_error"]
    error_cap = 10.0
    error_image = axes[1, 1].imshow(
        np.log10(
            np.maximum(
                np.clip(analog_error, 0.0, error_cap)
                / np.maximum(np.clip(vr_error, 0.0, error_cap), 1.0e-6),
                1.0e-3,
            )
        ),
        origin="lower",
        extent=extent,
        cmap="PiYG",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[1, 1].set_title("Candidate cellwise error gain")
    fig.colorbar(error_image, ax=axes[1, 1], label="log10(R analog / R VR)")

    labels = ["baseline", "candidate"]
    estimates = []
    errors = []
    for name in labels:
        region = summary["designs"][name]["variance_reduced"]["regions"]["far_field"]
        estimates.append(float(region["estimate"]))
        errors.append(1.96 * float(region["standard_error"] or 0.0))
    axes[1, 2].bar(labels, estimates, yerr=errors, capsize=5)
    axes[1, 2].set_title("Far-field mean flux, 95% CI")
    axes[1, 2].set_ylabel("track-length flux")

    for axis in axes.ravel()[:5]:
        axis.set_xlabel("x [cm]")
        axis.set_ylabel("y [cm]")
    comparison = summary["design_comparison"]["far_field"]
    reduction = comparison["estimated_reduction_fraction"]
    reduction_text = "n/a" if reduction is None else f"{100.0 * float(reduction):.1f}%"
    fig.suptitle(
        "Unbiased Monte Carlo certification\n"
        f"estimated far-field reduction={reduction_text}; "
        f"separated 95% intervals={comparison['candidate_upper_below_baseline_lower']}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    final_verified = summary.get("final_verified")
    lines = [
        "# GenShield-Agent 运行摘要",
        "",
        f"- 运行 ID：`{summary['run_id']}`",
        f"- 扩散筛选最优风险：`{summary['best_fast']['robust_risk']:.6f}`",
    ]
    if final_verified is not None:
        lines.extend(
            [
                f"- 最终验证级别：`{final_verified['fidelity']}`",
                f"- 最终验证风险：`{final_verified['risk']:.6f}`",
                f"- 最终设计签名：`{final_verified['design']['signature']}`",
                f"- 是否保留基准：`{final_verified['baseline_retained']}`",
            ]
        )
    lines.extend(["", "## 验证链", ""])
    for stage in summary.get("verification_stages", []):
        lines.append(
            f"- `{stage['fidelity']}`：risk=`{stage['selected_risk']:.6f}`，"
            f"accepted=`{stage['accepted']}`；{stage['decision']}"
        )
    certification = summary.get("mc_certification")
    if certification is not None:
        comparison = certification["design_comparison"]["far_field"]
        reduction = comparison["estimated_reduction_fraction"]
        lines.extend(
            [
                "",
                "## 无偏 MC 认证",
                "",
                f"- 远端平均通量估计下降：`{100.0 * float(reduction):.2f}%`",
                "- 候选 95% 上界低于基准 95% 下界："
                f"`{comparison['candidate_upper_below_baseline_lower']}`",
                "- FOM 只用于评价验证效率，不是屏蔽优化目标。",
            ]
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `summary.json`：完整机器可读结果",
            "- `best_design.json`：最终材料布局",
            "- `trajectory.jsonl`：搜索与验证轨迹",
            "- `shield_design_comparison.png`：共同色标设计对比",
            "- `mc_certification.json` / `mc_certification.png`：无偏统计证据",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"shield-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    trajectory_path = output_dir / "trajectory.jsonl"
    trajectory_path.unlink(missing_ok=True)
    logger = JsonlLogger(trajectory_path, run_id)
    config = vars(args).copy()
    logger.write("run_started", {"config": config})

    fast_evaluator = DiffusionShieldEvaluator(args.fast_nx, args.fast_ny)
    verification_evaluators = []
    verification_top_ks = []
    if not args.fast_only and args.high_fidelity_top_k > 0:
        verification_evaluators.append(
            GenVRShieldEvaluator(
                args.response_library,
                args.internal_response,
                device=args.device,
                fidelity="genvr-cfm-response",
            )
        )
        verification_top_ks.append(args.high_fidelity_top_k)
    if (
        not args.fast_only
        and args.use_exact_response
        and args.exact_top_k > 0
    ):
        verification_evaluators.append(
            GenVRShieldEvaluator(
                args.exact_response_library,
                args.exact_internal_response,
                device=args.device,
                fidelity="exact-local-mc-response",
            )
        )
        verification_top_ks.append(args.exact_top_k)
    agent = SmartShieldAgent(
        fast_evaluator,
        verification_evaluators=tuple(verification_evaluators),
        proposal_model=PhysicsGuidedProposalModel(args.seed),
        risk_aversion=args.risk_aversion,
        exploration_strength=args.exploration_strength,
        minimum_verified_improvement=args.minimum_verified_improvement,
    )

    strategist = None
    llm_exchanges: list[dict[str, Any]] = []
    if args.use_llm:
        strategist = ShieldLLMStrategist(
            model=args.model,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            max_tokens=640,
        )

    fallback_strategist = EvidenceDrivenShieldStrategist()
    fallback_strategy = ShieldStrategy()

    def strategy_provider(context: dict[str, Any]) -> ShieldStrategy:
        if strategist is None:
            return fallback_strategist.advise(context)
        try:
            advice = strategist.advise(context)
            if strategist.last_trace is not None:
                llm_exchanges.append(strategist.last_trace)
                logger.write("llm_exchange", strategist.last_trace)
            logger.write(
                "strategy_decision",
                {
                    "round": context["round"],
                    "operator_weights": advice.normalized_operator_weights(),
                    "focus_sector": advice.focus_sector,
                    "exploration_fraction": advice.exploration_fraction,
                    "hypothesis": advice.hypothesis,
                    "decision_reason": advice.decision_reason,
                },
            )
            return advice
        except Exception as exc:
            logger.write(
                "strategy_fallback",
                {"round": context["round"], "error": str(exc)},
            )
            return fallback_strategist.advise(context)

    result = agent.search(
        baseline_lattice_design(),
        rounds=args.rounds,
        beam_width=args.beam_width,
        proposals_per_parent=args.proposals_per_parent,
        high_fidelity_top_k=(
            0 if args.fast_only or not verification_top_ks else verification_top_ks[0]
        ),
        verification_top_ks=tuple(verification_top_ks) or None,
        strategy=fallback_strategy,
        strategy_provider=strategy_provider,
    )
    for strategy in result.strategy_trace:
        logger.write("strategy_applied", strategy)
    for record in result.records:
        logger.write("candidate_evaluated", record.to_dict())

    summary = result.to_dict()
    summary["run_id"] = run_id
    summary["config"] = config
    summary["llm_exchange_count"] = len(llm_exchanges)
    summary["claim_scope"] = (
        "Fixed 7x7 macro-grid, fixed central source, exactly 11 absorber blocks, "
        "existing two-material library, diffusion ensemble for search, cached GenVR "
        "response reranking, and an independent projected local-MC response veto."
    )
    mc_artifacts: dict[str, str] = {}
    if args.mc_certify:
        final_stage = result.verification_stages[-1] if result.verification_stages else None
        certification_baseline = (
            final_stage.baseline if final_stage is not None else result.baseline_fast
        )
        certification_candidate = (
            final_stage.selected if final_stage is not None else result.best_fast
        )
        logger.write(
            "mc_certification_started",
            {
                "baseline_design": certification_baseline.design.to_dict(),
                "candidate_design": certification_candidate.design.to_dict(),
                "histories_per_replicate": args.mc_histories,
                "replicates": args.mc_replicates,
                "grid": [args.mc_nx, args.mc_ny],
            },
        )
        certification, certification_arrays = certify_shield_pair(
            certification_baseline.design,
            certification_candidate.design,
            certification_baseline.flux,
            certification_candidate.flux,
            nx=args.mc_nx,
            ny=args.mc_ny,
            histories=args.mc_histories,
            replicates=args.mc_replicates,
            n_levels=args.mc_levels,
            maximum_levels=args.mc_maximum_levels,
            flux_exponent=args.mc_flux_exponent,
            seed=args.seed + 700_001,
        )
        certification_path = output_dir / "mc_certification.json"
        arrays_path = output_dir / "mc_certification_fields.npz"
        figure_path = output_dir / "mc_certification.png"
        certification_path.write_text(
            json.dumps(certification, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(arrays_path, **certification_arrays)
        _plot_mc_certification(certification, certification_arrays, figure_path)
        mc_artifacts = {
            "summary": str(certification_path),
            "fields": str(arrays_path),
            "figure": str(figure_path),
        }
        summary["mc_certification"] = certification
        logger.write(
            "mc_certification_completed",
            {
                "comparison": certification["design_comparison"],
                "artifacts": mc_artifacts,
            },
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_markdown(summary, output_dir / "summary.md")
    (output_dir / "best_design.json").write_text(
        json.dumps(result.best_design.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_result(result, output_dir / "shield_design_comparison.png")
    logger.write(
        "run_completed",
        {
            "best_design": result.best_design.to_dict(),
            "best_fast_risk": result.best_fast_risk,
            "best_high_fidelity_risk": result.best_high_fidelity_risk,
            "best_verified_risk": result.best_verified_risk,
            "final_verification": (
                result.verification_stages[-1].to_dict()
                if result.verification_stages
                else None
            ),
            "artifacts": {
                "summary": str(output_dir / "summary.json"),
                "summary_markdown": str(output_dir / "summary.md"),
                "best_design": str(output_dir / "best_design.json"),
                "figure": str(output_dir / "shield_design_comparison.png"),
                **({"mc_certification": mc_artifacts} if mc_artifacts else {}),
            },
        },
    )
    print(json.dumps({
        "best_design": result.best_design.to_dict(),
        "best_fast_risk": result.best_fast_risk,
        "best_high_fidelity_risk": result.best_high_fidelity_risk,
        "best_verified_risk": result.best_verified_risk,
        "final_verification_fidelity": (
            result.verification_stages[-1].fidelity
            if result.verification_stages
            else None
        ),
        "mc_certification": mc_artifacts or None,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
