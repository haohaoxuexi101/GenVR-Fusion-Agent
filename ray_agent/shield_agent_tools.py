from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import math

import numpy as np

from .metrics import finite_json
from .shield_agent_protocol import ShieldAgentAction, canonical_tool_name
from .shield_certification import certify_shield_pair
from .shield_design import (
    DEFAULT_METRIC_WEIGHTS,
    OPERATORS,
    OperatorMemory,
    PhysicsGuidedProposalModel,
    ShieldDesign,
    ShieldEvaluation,
    ShieldEvaluator,
    ShieldStrategy,
    SmartShieldAgent,
    baseline_lattice_design,
)
from .shield_live_reporting import write_gmc_candidate_snapshot
from .shield_policy import validate_shield_strategy


class ShieldToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShieldAgentBudgets:
    max_diffusion_evaluations: int = 320
    max_diffusion_per_call: int = 96
    max_genvr_candidates: int = 4
    max_gmc_per_call: int = 8
    max_exact_candidates: int = 2
    max_mc_root_histories: int = 20_000
    max_mc_histories_per_call: int = 2_500
    max_mc_replicates_per_call: int = 4
    max_mc_grid: int = 56

    @property
    def max_generated_structures(self) -> int:
        return self.max_diffusion_evaluations

    @property
    def max_structures_per_call(self) -> int:
        return self.max_diffusion_per_call

    def __post_init__(self) -> None:
        values = {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        if any(value < 1 for value in values.values()):
            raise ValueError("all autonomous-agent budgets must be positive")


@dataclass
class ShieldAgentUsage:
    diffusion_evaluations: int = 0
    genvr_candidates: int = 0
    exact_candidates: int = 0
    mc_root_histories: int = 0

    @property
    def generated_structures(self) -> int:
        return self.diffusion_evaluations

    @generated_structures.setter
    def generated_structures(self, value: int) -> None:
        self.diffusion_evaluations = int(value)

    def to_dict(
        self,
        budgets: ShieldAgentBudgets,
        discovery_mode: str = "diffusion_prefilter",
    ) -> dict[str, Any]:
        structure_name = (
            "generated_structures"
            if discovery_mode == "gmc_only"
            else "diffusion_evaluations"
        )
        limits = {
            structure_name: budgets.max_generated_structures,
            "genvr_candidates": budgets.max_genvr_candidates,
            "exact_candidates": budgets.max_exact_candidates,
            "mc_root_histories": budgets.max_mc_root_histories,
        }
        used = {
            structure_name: self.generated_structures,
            "genvr_candidates": self.genvr_candidates,
            "exact_candidates": self.exact_candidates,
            "mc_root_histories": self.mc_root_histories,
        }
        return {
            name: {
                "used": used[name],
                "limit": limit,
                "remaining": max(limit - used[name], 0),
            }
            for name, limit in limits.items()
        }


@dataclass
class CandidateEvidence:
    design: ShieldDesign
    parent_signature: str | None = None
    mutation: dict[str, Any] | None = None
    generation_round: int = 0
    generation_guidance_fidelity: str | None = None
    gmc_guided_generation: bool = False
    proposal_score: float | None = None
    diffusion: ShieldEvaluation | None = None
    diffusion_risk: float | None = None
    genvr: ShieldEvaluation | None = None
    genvr_risk: float | None = None
    exact: ShieldEvaluation | None = None
    exact_risk: float | None = None
    mc_reports: list[dict[str, Any]] = field(default_factory=list)

    def highest_fidelity(self) -> str:
        if self.mc_reports:
            return "unbiased-mc"
        if self.exact is not None:
            return "exact-local-mc-response"
        if self.genvr is not None:
            return "genvr-cfm-response"
        if self.diffusion is not None:
            return "diffusion-ensemble"
        return "unevaluated"


def _evaluation_payload(
    evaluation: ShieldEvaluation | None,
    risk: float | None,
) -> dict[str, Any] | None:
    if evaluation is None or risk is None:
        return None
    return {
        "risk": float(risk),
        "metrics": evaluation.metrics.to_dict(),
        "runtime_s": float(evaluation.runtime_s),
        "fidelity": evaluation.fidelity,
    }


def _rank_correlation(previous: list[float], current: list[float]) -> float | None:
    if len(previous) < 2 or len(previous) != len(current):
        return None
    left = np.argsort(np.argsort(np.asarray(previous))).astype(np.float64)
    right = np.argsort(np.argsort(np.asarray(current))).astype(np.float64)
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


class ShieldToolEnvironment:
    """Stateful scientific instruments exposed to the autonomous LLM agent."""

    def __init__(
        self,
        fast_evaluator: ShieldEvaluator | None,
        genvr_evaluator: ShieldEvaluator | None,
        exact_evaluator: ShieldEvaluator | None,
        output_dir: str | Path,
        budgets: ShieldAgentBudgets | None = None,
        seed: int = 260918,
        metric_weights: dict[str, float] | None = None,
        risk_aversion: float = 0.35,
        exploration_strength: float = 0.04,
        minimum_verified_improvement: float = 0.0,
        require_projected_mc_audit: bool = False,
        minimum_gmc_candidates_before_mc: int = 1,
        minimum_gmc_guided_candidates_before_mc: int = 1,
        discovery_mode: str = "diffusion_prefilter",
        minimum_mc_histories_per_method: int = 2,
        minimum_mc_replicates: int = 1,
        minimum_mc_grid: int = 7,
        render_gmc_candidates: bool = True,
        mc_runner: Callable[..., tuple[dict[str, Any], dict[str, np.ndarray]]] = certify_shield_pair,
    ) -> None:
        if discovery_mode not in {"diffusion_prefilter", "gmc_only"}:
            raise ValueError(
                "discovery_mode must be 'diffusion_prefilter' or 'gmc_only'"
            )
        if discovery_mode == "diffusion_prefilter" and fast_evaluator is None:
            raise ValueError("diffusion_prefilter mode requires a fast evaluator")
        if discovery_mode == "gmc_only" and genvr_evaluator is None:
            raise ValueError("gmc_only mode requires a GMC evaluator")
        self.discovery_mode = discovery_mode
        self.fast_evaluator = fast_evaluator
        self.genvr_evaluator = genvr_evaluator
        self.exact_evaluator = exact_evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.budgets = budgets or ShieldAgentBudgets()
        self.usage = ShieldAgentUsage()
        self.metric_weights = metric_weights or dict(DEFAULT_METRIC_WEIGHTS)
        self.risk_aversion = float(risk_aversion)
        self.exploration_strength = float(exploration_strength)
        self.minimum_verified_improvement = float(minimum_verified_improvement)
        if not 0.0 <= self.minimum_verified_improvement < 1.0:
            raise ValueError("minimum_verified_improvement must lie in [0, 1)")
        self.require_projected_mc_audit = bool(require_projected_mc_audit)
        self.minimum_gmc_candidates_before_mc = int(minimum_gmc_candidates_before_mc)
        if self.minimum_gmc_candidates_before_mc < 1:
            raise ValueError("minimum_gmc_candidates_before_mc must be positive")
        self.minimum_gmc_guided_candidates_before_mc = int(
            minimum_gmc_guided_candidates_before_mc
        )
        if self.minimum_gmc_guided_candidates_before_mc < 0:
            raise ValueError(
                "minimum_gmc_guided_candidates_before_mc must be non-negative"
            )
        if self.minimum_gmc_candidates_before_mc > self.budgets.max_genvr_candidates:
            raise ValueError(
                "minimum_gmc_candidates_before_mc exceeds the GMC candidate budget"
            )
        if (
            self.minimum_gmc_guided_candidates_before_mc
            > self.budgets.max_genvr_candidates - 1
        ):
            raise ValueError(
                "minimum_gmc_guided_candidates_before_mc leaves no GMC budget for a parent"
            )
        self.minimum_mc_histories_per_method = int(
            minimum_mc_histories_per_method
        )
        self.minimum_mc_replicates = int(minimum_mc_replicates)
        self.minimum_mc_grid = int(minimum_mc_grid)
        self.render_gmc_candidates = bool(render_gmc_candidates)
        if self.minimum_mc_histories_per_method < 2:
            raise ValueError("minimum_mc_histories_per_method must be at least two")
        if self.minimum_mc_replicates < 1:
            raise ValueError("minimum_mc_replicates must be positive")
        if self.minimum_mc_grid < 7 or self.minimum_mc_grid % 7:
            raise ValueError("minimum_mc_grid must be divisible by seven and at least seven")
        if self.minimum_mc_grid > self.budgets.max_mc_grid:
            raise ValueError("minimum_mc_grid exceeds the MC grid budget")
        minimum_histories_per_replicate = math.ceil(
            self.minimum_mc_histories_per_method / self.minimum_mc_replicates
        )
        if minimum_histories_per_replicate > self.budgets.max_mc_histories_per_call:
            raise ValueError(
                "minimum MC histories cannot be met within the per-call history budget"
            )
        if self.minimum_mc_replicates > self.budgets.max_mc_replicates_per_call:
            raise ValueError(
                "minimum MC replicates exceed the per-call replicate budget"
            )
        minimum_mc_charge = (
            4
            * minimum_histories_per_replicate
            * self.minimum_mc_replicates
        )
        if minimum_mc_charge > self.budgets.max_mc_root_histories:
            raise ValueError(
                "minimum MC precision cannot be met within the total root-history budget"
            )
        self.seed = int(seed)
        self.proposal_model = PhysicsGuidedProposalModel(
            self.seed,
            importance_model=(
                "gmc_flux_path"
                if self.discovery_mode == "gmc_only"
                else "diffusion_adjoint"
            ),
        )
        self.operator_memory = OperatorMemory()
        self.gmc_operator_memory = OperatorMemory()
        self.mc_runner = mc_runner
        self.mc_run_index = 0
        self.structure_generation_rounds = 0
        self.gmc_screening_rounds = 0
        self.requested_beam_width = 4
        self.search_records: list[dict[str, Any]] = []
        self.baseline_design = baseline_lattice_design()
        self._genvr_baseline: ShieldEvaluation | None = None
        self._exact_baseline: ShieldEvaluation | None = None
        if self.discovery_mode == "gmc_only":
            if self.genvr_evaluator is None:
                raise RuntimeError("GMC evaluator is missing")
            self._genvr_baseline = self.genvr_evaluator.evaluate(
                self.baseline_design
            )
            baseline_evidence = CandidateEvidence(
                design=self.baseline_design,
                genvr=self._genvr_baseline,
                genvr_risk=1.0,
            )
        else:
            if self.fast_evaluator is None:
                raise RuntimeError("diffusion evaluator is missing")
            baseline = self.fast_evaluator.evaluate(self.baseline_design)
            self.usage.generated_structures = 1
            baseline_evidence = CandidateEvidence(
                design=self.baseline_design,
                diffusion=baseline,
                diffusion_risk=1.0,
            )
        self.candidates: dict[str, CandidateEvidence] = {
            self.baseline_design.signature: baseline_evidence
        }
        self.beam_signatures = [self.baseline_design.signature]

    def budget_usage(self) -> dict[str, Any]:
        return self.usage.to_dict(self.budgets, self.discovery_mode)

    @property
    def verification_threshold(self) -> float:
        return 1.0 - self.minimum_verified_improvement

    @property
    def genvr_baseline(self) -> ShieldEvaluation | None:
        return self._genvr_baseline

    @property
    def exact_baseline(self) -> ShieldEvaluation | None:
        return self._exact_baseline

    def discovery_progress(self) -> dict[str, Any]:
        non_baseline = [
            evidence
            for evidence in self.candidates.values()
            if evidence.design.signature != self.baseline_design.signature
        ]
        gmc_guided_structures = sum(
            evidence.gmc_guided_generation for evidence in non_baseline
        )
        gmc_guided_screened = sum(
            evidence.gmc_guided_generation and evidence.genvr is not None
            for evidence in non_baseline
        )
        total_gmc_screened = sum(
            evidence.genvr is not None for evidence in non_baseline
        )
        minimum_batch_passed = (
            total_gmc_screened >= self.minimum_gmc_candidates_before_mc
        )
        guided_cycle_passed = (
            gmc_guided_screened >= self.minimum_gmc_guided_candidates_before_mc
        )
        return {
            "discovery_mode": self.discovery_mode,
            "structure_generation_rounds": self.structure_generation_rounds,
            "gmc_screening_rounds": self.gmc_screening_rounds,
            "generated_structures": len(non_baseline),
            "gmc_screened_structures": total_gmc_screened,
            "minimum_gmc_screened_required": self.minimum_gmc_candidates_before_mc,
            "minimum_gmc_batch_passed": minimum_batch_passed,
            "gmc_guided_structures": gmc_guided_structures,
            "gmc_guided_and_screened": gmc_guided_screened,
            "minimum_gmc_guided_screened_required": (
                self.minimum_gmc_guided_candidates_before_mc
            ),
            "gmc_guided_cycle_passed": guided_cycle_passed,
            "mc_discovery_gate_passed": minimum_batch_passed and guided_cycle_passed,
        }

    def tool_specs(self) -> list[dict[str, Any]]:
        if self.discovery_mode == "gmc_only":
            generation_description = (
                "Generate new legal fixed-inventory material structures from GMC-screened "
                "parents. The compiler uses the parent GMC flux, leakage sectors, path "
                "continuity, hotspot control, symmetry repair, learned operator outcomes, "
                "and explicit exploration. No neutron-diffusion solve or diffusion prefilter "
                "is used; generated structures remain unevaluated until screen_gmc is called."
            )
            parent_description = (
                "optional list of currently observed GMC-screened parent signatures"
            )
        else:
            generation_description = (
                "Generate new legal fixed-inventory material structures from selected "
                "parents. The compiler proposes one-for-one material moves using adjoint "
                "sensitivity, leakage channels, hotspot control, symmetry repair, and "
                "exploration; GMC-screened parent fields guide later generations. A cheap "
                "diffusion ensemble prefilters the generated structures."
            )
            parent_description = (
                "optional list of currently observed diffusion candidate signatures"
            )
        return [
            {
                "name": "generate_structures",
                "description": (
                    generation_description
                    + " Monte Carlo stays "
                    "locked until the configured GMC-guided regeneration cycle is complete."
                ),
                "arguments": {
                    "operator_weights": (
                        f"object with only {list(OPERATORS)} and values in [0,1]"
                    ),
                    "focus_sector": "null or integer 0..7",
                    "exploration_fraction": "float in [0.05,0.50]",
                    "proposals_per_parent": f"integer 1..{self.budgets.max_structures_per_call}",
                    "beam_width": "integer 1..8",
                    "parent_signatures": parent_description,
                },
            },
            {
                "name": "screen_gmc",
                "description": (
                    "Evaluate a batch of generated structures with the cached GenVR/GMC "
                    "response operator. GMC is the primary design-ranking model and its flux "
                    "field can guide the next structure-generation round and MC weight windows."
                ),
                "arguments": {"design_signatures": "one or more observed candidate signatures"},
            },
            {
                "name": "audit_projected_mc",
                "description": (
                    "Optionally audit a GMC-screened structure with an independently sampled "
                    "projected local-MC response operator using the same interface phase space. "
                    "This diagnoses learned-kernel error; it is not physical certification and "
                    "is not required unless the frozen campaign config says so."
                ),
                "arguments": {"design_signatures": "one or more CFM-approved signatures"},
            },
            {
                "name": "certify_mc",
                "description": (
                    "Run analog and GMC-weight-window Monte Carlo for one GMC-screened design, "
                    "using root-history far-field confidence intervals. This is the physical "
                    "certification stage. Each candidate can be certified once, so allocate "
                    "enough histories to meet the frozen verifier contract."
                ),
                "arguments": {
                    "design_signature": "one GMC-approved candidate signature",
                    "histories": (
                        f"integer per replicate; histories*replicates >= "
                        f"{self.minimum_mc_histories_per_method} and value <= "
                        f"{self.budgets.max_mc_histories_per_call}"
                    ),
                    "replicates": (
                        f"integer {self.minimum_mc_replicates}.."
                        f"{self.budgets.max_mc_replicates_per_call}"
                    ),
                    "nx": (
                        f"mesh width divisible by 7 and in "
                        f"[{self.minimum_mc_grid}, {self.budgets.max_mc_grid}]"
                    ),
                    "ny": (
                        f"mesh height divisible by 7 and in "
                        f"[{self.minimum_mc_grid}, {self.budgets.max_mc_grid}]"
                    ),
                    "n_levels": "null or integer 1..20",
                    "flux_exponent": "float in (0,1]",
                },
            },
            {
                "name": "finish",
                "description": (
                    "Request termination. The independent verifier accepts or rejects the "
                    "request; the model cannot waive missing evidence."
                ),
                "arguments": {
                    "candidate_signature": "candidate signature or null",
                    "claim": "verified_improvement, no_verified_improvement, or inconclusive",
                    "summary": "short evidence-grounded conclusion",
                },
            },
        ]

    def _candidate_payload(self, evidence: CandidateEvidence) -> dict[str, Any]:
        baseline_cells = set(self.baseline_design.absorbers)
        candidate_cells = set(evidence.design.absorbers)
        latest_mc = evidence.mc_reports[-1] if evidence.mc_reports else None
        return {
            "signature": evidence.design.signature,
            "is_baseline": evidence.design.signature == self.baseline_design.signature,
            "lineage": {
                "parent_signature": evidence.parent_signature,
                "mutation": evidence.mutation,
                "generation_round": evidence.generation_round,
                "generation_guidance_fidelity": (
                    evidence.generation_guidance_fidelity
                ),
                "gmc_guided_generation": evidence.gmc_guided_generation,
            },
            "removed_from_baseline": [
                list(cell) for cell in sorted(baseline_cells - candidate_cells)
            ],
            "added_to_baseline": [
                list(cell) for cell in sorted(candidate_cells - baseline_cells)
            ],
            "absorber_count": evidence.design.absorber_count,
            "proposal_score": evidence.proposal_score,
            "highest_fidelity": evidence.highest_fidelity(),
            "proposal_score": evidence.proposal_score,
            "diffusion": _evaluation_payload(evidence.diffusion, evidence.diffusion_risk),
            "genvr": _evaluation_payload(evidence.genvr, evidence.genvr_risk),
            "gmc": _evaluation_payload(evidence.genvr, evidence.genvr_risk),
            "exact": _evaluation_payload(evidence.exact, evidence.exact_risk),
            "projected_mc_audit": _evaluation_payload(
                evidence.exact,
                evidence.exact_risk,
            ),
            "latest_mc": (
                {
                    "design_comparison": latest_mc["summary"]["design_comparison"],
                    "artifact": latest_mc["artifact"],
                }
                if latest_mc is not None
                else None
            ),
        }

    @staticmethod
    def _agent_evaluation_payload(
        evaluation: ShieldEvaluation | None,
        risk: float | None,
    ) -> dict[str, Any] | None:
        if evaluation is None or risk is None:
            return None
        metrics = evaluation.metrics
        return {
            "risk": float(risk),
            "far_mean": float(metrics.far_mean),
            "far_cvar90": float(metrics.far_cvar90),
            "far_max": float(metrics.far_max),
            "worst_sector_mean": float(metrics.worst_sector_mean),
            "fidelity": evaluation.fidelity,
        }

    def _agent_candidate_payload(
        self,
        evidence: CandidateEvidence,
    ) -> dict[str, Any]:
        baseline_cells = set(self.baseline_design.absorbers)
        candidate_cells = set(evidence.design.absorbers)
        latest_mc = evidence.mc_reports[-1] if evidence.mc_reports else None
        mc_summary = None
        if latest_mc is not None:
            far_field = latest_mc["summary"]["design_comparison"]["far_field"]
            mc_summary = {
                "candidate_over_baseline": far_field.get(
                    "candidate_over_baseline"
                ),
                "difference_z": far_field.get("difference_z"),
                "ci95_separated": far_field.get(
                    "candidate_upper_below_baseline_lower"
                ),
                "artifact": latest_mc["artifact"],
            }
        return {
            "signature": evidence.design.signature,
            "is_baseline": evidence.design.signature == self.baseline_design.signature,
            "lineage": {
                "parent_signature": evidence.parent_signature,
                "mutation": evidence.mutation,
                "generation_round": evidence.generation_round,
                "generation_guidance_fidelity": (
                    evidence.generation_guidance_fidelity
                ),
                "gmc_guided_generation": evidence.gmc_guided_generation,
            },
            "removed_from_baseline": [
                list(cell) for cell in sorted(baseline_cells - candidate_cells)
            ],
            "added_to_baseline": [
                list(cell) for cell in sorted(candidate_cells - baseline_cells)
            ],
            "highest_fidelity": evidence.highest_fidelity(),
            "diffusion": self._agent_evaluation_payload(
                evidence.diffusion,
                evidence.diffusion_risk,
            ),
            "genvr": self._agent_evaluation_payload(
                evidence.genvr,
                evidence.genvr_risk,
            ),
            "gmc": self._agent_evaluation_payload(
                evidence.genvr,
                evidence.genvr_risk,
            ),
            "exact": self._agent_evaluation_payload(
                evidence.exact,
                evidence.exact_risk,
            ),
            "projected_mc_audit": self._agent_evaluation_payload(
                evidence.exact,
                evidence.exact_risk,
            ),
            "latest_mc": mc_summary,
        }

    @staticmethod
    def _discovery_risk(evidence: CandidateEvidence) -> float:
        if evidence.genvr_risk is not None:
            return float(evidence.genvr_risk)
        if evidence.diffusion_risk is not None:
            return float(evidence.diffusion_risk)
        if evidence.proposal_score is not None:
            return 1.0e6 - float(evidence.proposal_score)
        return math.inf

    @staticmethod
    def _guidance_evaluation(evidence: CandidateEvidence) -> ShieldEvaluation | None:
        return evidence.genvr or evidence.diffusion

    def observe_for_agent(
        self,
        maximum_candidates: int = 8,
        maximum_beam: int = 4,
    ) -> dict[str, Any]:
        if maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        if maximum_beam < 1:
            raise ValueError("maximum_beam must be positive")
        ranked = sorted(
            self.candidates.values(),
            key=self._discovery_risk,
        )
        promoted = [
            item.design.signature
            for item in ranked
            if item.genvr is not None or item.exact is not None or item.mc_reports
        ]
        pending = sorted(
            (
                item
                for item in ranked
                if item.design.signature != self.baseline_design.signature
                and item.genvr is None
            ),
            key=lambda item: float(item.proposal_score or -math.inf),
            reverse=True,
        )
        if self.discovery_mode == "gmc_only":
            priority_signatures = [
                *self.beam_signatures,
                *(item.design.signature for item in pending),
                *promoted,
                self.baseline_design.signature,
                *(item.design.signature for item in ranked),
            ]
        else:
            priority_signatures = [
                *promoted,
                *self.beam_signatures,
                self.baseline_design.signature,
                *(item.design.signature for item in ranked),
            ]
        visible = []
        seen = set()
        for signature in priority_signatures:
            if signature in seen:
                continue
            visible.append(self.candidates[signature])
            seen.add(signature)
            if len(visible) >= maximum_candidates:
                break
        beam = []
        for signature in self.beam_signatures[:maximum_beam]:
            evidence = self.candidates[signature]
            guidance = self._guidance_evaluation(evidence)
            if guidance is None:
                continue
            diagnosis = self.proposal_model.strategy_context(guidance)
            beam.append(
                {
                    **self._agent_candidate_payload(evidence),
                    "generation_guidance_fidelity": guidance.fidelity,
                    "physics_diagnosis": {
                        "hottest_sector": diagnosis["hottest_sector"],
                        "normalized_sector_leakage": diagnosis[
                            "normalized_sector_leakage"
                        ],
                        "highest_sensitivity_empty_cells": diagnosis[
                            "highest_sensitivity_empty_cells"
                        ][:4],
                        "lowest_sensitivity_absorber_cells": diagnosis[
                            "lowest_sensitivity_absorber_cells"
                        ][:4],
                    },
                }
            )
        non_baseline = [
            item
            for item in ranked
            if item.design.signature != self.baseline_design.signature
        ]
        discovery_progress = self.discovery_progress()
        evidence_queue = {
            "awaiting_gmc_screening": [
                item.design.signature
                for item in non_baseline
                if item.genvr is None
            ][:6],
            "optional_projected_mc_audit": [
                item.design.signature
                for item in non_baseline
                if item.genvr_risk is not None
                and item.genvr_risk < self.verification_threshold
                and item.exact is None
            ][:6],
            "awaiting_mc_certification": [
                item.design.signature
                for item in non_baseline
                if item.genvr_risk is not None
                and item.genvr_risk < self.verification_threshold
                and (
                    not self.require_projected_mc_audit
                    or (
                        item.exact_risk is not None
                        and item.exact_risk < self.verification_threshold
                    )
                )
                and discovery_progress["mc_discovery_gate_passed"]
                and not item.mc_reports
            ][:6],
            "mc_evaluated": [
                item.design.signature for item in non_baseline if item.mc_reports
            ][:6],
        }
        evidence_queue["gmc_guided_awaiting_screening"] = [
            item.design.signature
            for item in non_baseline
            if item.gmc_guided_generation and item.genvr is None
        ][:6]
        evidence_queue["gmc_guided_generation_needed"] = not discovery_progress[
            "gmc_guided_cycle_passed"
        ]
        return finite_json(
            {
                "objective": {
                    "discovery_mode": self.discovery_mode,
                    "primary": "discover material structures that reduce source-distant full-field neutron flux",
                    "tail": "reduce far-field CVaR90, maximum, and worst angular sector",
                    "gmc_role": "primary high-throughput structure evaluator and MC weight-window guide",
                    "mc_role": "unbiased physical certification for selected structures",
                    "fom_role": "verification efficiency only; never the shielding objective",
                },
                "protected_constraints": {
                    "grid": [7, 7],
                    "source_cell": list(self.baseline_design.source_cell),
                    "absorber_count": self.baseline_design.absorber_count,
                    "geometry_actions": "only compiler-generated one-for-one material swaps",
                    "verification_threshold": self.verification_threshold,
                    "projected_mc_audit_required": self.require_projected_mc_audit,
                    "minimum_gmc_candidates_before_mc": self.minimum_gmc_candidates_before_mc,
                    "minimum_gmc_guided_candidates_before_mc": (
                        self.minimum_gmc_guided_candidates_before_mc
                    ),
                    "minimum_mc_histories_per_design_and_method": (
                        self.minimum_mc_histories_per_method
                    ),
                    "minimum_mc_replicates": self.minimum_mc_replicates,
                    "minimum_mc_grid": [
                        self.minimum_mc_grid,
                        self.minimum_mc_grid,
                    ],
                },
                "budgets": self.budget_usage(),
                "currently_executable_scientific_tools": (
                    self.executable_scientific_tools()
                ),
                "beam": beam,
                "visible_candidates": [
                    self._agent_candidate_payload(item) for item in visible
                ],
                "evidence_queue": evidence_queue,
                "discovery_progress": discovery_progress,
                "evidence_counts": {
                    "candidate_designs": len(self.candidates) - 1,
                    "gmc_screened": sum(
                        item.genvr is not None for item in non_baseline
                    ),
                    "genvr_evaluated": sum(
                        item.genvr is not None for item in non_baseline
                    ),
                    "exact_evaluated": sum(
                        item.exact is not None for item in self.candidates.values()
                    ),
                    "mc_certified": sum(
                        bool(item.mc_reports) for item in self.candidates.values()
                    ),
                    "gmc_guided_structures": discovery_progress[
                        "gmc_guided_structures"
                    ],
                    "gmc_guided_and_screened": discovery_progress[
                        "gmc_guided_and_screened"
                    ],
                },
                "operator_memory": {
                    "proposal_prefilter": (
                        None
                        if self.discovery_mode == "gmc_only"
                        else self.operator_memory.to_dict()
                    ),
                    "gmc_validated": self.gmc_operator_memory.to_dict(),
                },
            }
        )

    def observe(self, maximum_candidates: int = 12) -> dict[str, Any]:
        ranked = sorted(
            self.candidates.values(),
            key=self._discovery_risk,
        )
        priority_signatures = list(self.beam_signatures)
        priority_signatures.extend(
            item.design.signature
            for item in ranked
            if item.genvr is not None or item.exact is not None or item.mc_reports
        )
        priority_signatures.extend(item.design.signature for item in ranked)
        visible = []
        seen = set()
        for signature in priority_signatures:
            if signature in seen:
                continue
            visible.append(self.candidates[signature])
            seen.add(signature)
            if len(visible) >= maximum_candidates:
                break
        beam = []
        for signature in self.beam_signatures:
            evidence = self.candidates[signature]
            guidance = self._guidance_evaluation(evidence)
            if guidance is None:
                continue
            beam.append(
                {
                    **self._candidate_payload(evidence),
                    "generation_guidance_fidelity": guidance.fidelity,
                    "physics_diagnosis": self.proposal_model.strategy_context(
                        guidance
                    ),
                }
            )
        discovery_progress = self.discovery_progress()
        return finite_json(
            {
                "objective": {
                    "discovery_mode": self.discovery_mode,
                    "primary": "discover material structures that reduce source-distant full-field neutron flux",
                    "tail": "reduce far-field CVaR90, maximum, and worst angular sector",
                    "gmc_role": "primary high-throughput structure evaluator and MC weight-window guide",
                    "mc_role": "unbiased physical certification for selected structures",
                    "fom_role": "verification efficiency only; never the shielding objective",
                },
                "protected_constraints": {
                    "grid": [7, 7],
                    "source_cell": list(self.baseline_design.source_cell),
                    "absorber_count": self.baseline_design.absorber_count,
                    "geometry_actions": "only compiler-generated one-for-one material swaps",
                    "verification_threshold": self.verification_threshold,
                    "projected_mc_audit_required": self.require_projected_mc_audit,
                    "minimum_gmc_candidates_before_mc": self.minimum_gmc_candidates_before_mc,
                    "minimum_gmc_guided_candidates_before_mc": (
                        self.minimum_gmc_guided_candidates_before_mc
                    ),
                    "minimum_mc_histories_per_design_and_method": (
                        self.minimum_mc_histories_per_method
                    ),
                    "minimum_mc_replicates": self.minimum_mc_replicates,
                    "minimum_mc_grid": [
                        self.minimum_mc_grid,
                        self.minimum_mc_grid,
                    ],
                },
                "budgets": self.budget_usage(),
                "currently_executable_scientific_tools": (
                    self.executable_scientific_tools()
                ),
                "beam": beam,
                "visible_candidates": [
                    self._candidate_payload(item) for item in visible
                ],
                "discovery_progress": discovery_progress,
                "evidence_counts": {
                    "candidate_designs": len(self.candidates) - 1,
                    "gmc_screened": sum(
                        item.genvr is not None
                        and item.design.signature != self.baseline_design.signature
                        for item in self.candidates.values()
                    ),
                    "genvr_evaluated": sum(
                        item.genvr is not None
                        and item.design.signature != self.baseline_design.signature
                        for item in self.candidates.values()
                    ),
                    "exact_evaluated": sum(
                        item.exact is not None for item in self.candidates.values()
                    ),
                    "mc_certified": sum(
                        bool(item.mc_reports) for item in self.candidates.values()
                    ),
                    "gmc_guided_structures": discovery_progress[
                        "gmc_guided_structures"
                    ],
                    "gmc_guided_and_screened": discovery_progress[
                        "gmc_guided_and_screened"
                    ],
                },
                "operator_memory": {
                    "proposal_prefilter": (
                        None
                        if self.discovery_mode == "gmc_only"
                        else self.operator_memory.to_dict()
                    ),
                    "gmc_validated": self.gmc_operator_memory.to_dict(),
                },
            }
        )

    def execute(self, action: ShieldAgentAction) -> dict[str, Any]:
        tool = canonical_tool_name(action.tool)
        if tool == "generate_structures":
            if self.discovery_mode == "gmc_only":
                return self._generate_gmc_candidates(action)
            return self._run_diffusion_batch(action)
        if tool == "screen_gmc":
            return self._promote(action, "genvr")
        if tool == "audit_projected_mc":
            return self._promote(action, "exact")
        if tool == "certify_mc":
            return self._run_unbiased_mc(action)
        if tool == "finish":
            return {
                "finish_requested": True,
                "arguments": dict(action.arguments),
            }
        raise ShieldToolError(f"unsupported tool {action.tool!r}")

    def executable_scientific_tools(
        self,
        minimum_mc_histories: int = 2,
        minimum_mc_replicates: int = 1,
    ) -> list[str]:
        tools = []
        gmc_budget_available = (
            self.genvr_evaluator is not None
            and self.usage.genvr_candidates < self.budgets.max_genvr_candidates
        )
        if self.discovery_mode == "gmc_only":
            parent_available = any(
                evidence.genvr is not None
                for evidence in self.candidates.values()
            )
        else:
            parent_available = any(
                evidence.diffusion is not None
                for evidence in self.candidates.values()
            )
        if (
            self.usage.generated_structures < self.budgets.max_generated_structures
            and gmc_budget_available
            and parent_available
        ):
            tools.append("generate_structures")
        if (
            gmc_budget_available
            and any(
                evidence.design.signature != self.baseline_design.signature
                and evidence.genvr is None
                and (
                    self.discovery_mode == "gmc_only"
                    or evidence.diffusion is not None
                )
                for evidence in self.candidates.values()
            )
        ):
            tools.append("screen_gmc")
        if (
            self.exact_evaluator is not None
            and self.usage.exact_candidates < self.budgets.max_exact_candidates
            and any(
                evidence.genvr_risk is not None
                and evidence.genvr_risk < self.verification_threshold
                and evidence.exact is None
                for evidence in self.candidates.values()
            )
        ):
            tools.append("audit_projected_mc")
        required_histories = max(
            int(minimum_mc_histories),
            self.minimum_mc_histories_per_method,
        )
        required_replicates = max(
            int(minimum_mc_replicates),
            self.minimum_mc_replicates,
        )
        histories_per_replicate = math.ceil(
            required_histories / required_replicates
        )
        minimum_mc_charge = (
            4 * histories_per_replicate * required_replicates
        )
        if (
            self.usage.mc_root_histories + minimum_mc_charge
            <= self.budgets.max_mc_root_histories
            and histories_per_replicate
            <= self.budgets.max_mc_histories_per_call
            and required_replicates
            <= self.budgets.max_mc_replicates_per_call
            and self.discovery_progress()["mc_discovery_gate_passed"]
            and any(
                evidence.genvr_risk is not None
                and evidence.genvr_risk < self.verification_threshold
                and (
                    not self.require_projected_mc_audit
                    or (
                        evidence.exact_risk is not None
                        and evidence.exact_risk < self.verification_threshold
                    )
                )
                and not evidence.mc_reports
                for evidence in self.candidates.values()
            )
        ):
            tools.append("certify_mc")
        return tools

    def _generate_gmc_candidates(
        self,
        action: ShieldAgentAction,
    ) -> dict[str, Any]:
        remaining = (
            self.budgets.max_generated_structures
            - self.usage.generated_structures
        )
        if remaining <= 0:
            raise ShieldToolError("structure-generation budget is exhausted")
        arguments = action.arguments
        proposals_per_parent = int(arguments.get("proposals_per_parent", 12))
        if not 1 <= proposals_per_parent <= self.budgets.max_structures_per_call:
            raise ShieldToolError(
                "proposals_per_parent lies outside the declared budget"
            )
        beam_width = int(arguments.get("beam_width", 4))
        if not 1 <= beam_width <= 8:
            raise ShieldToolError("beam_width must lie in [1, 8]")
        parent_signatures = arguments.get("parent_signatures") or self.beam_signatures
        if not isinstance(parent_signatures, list) or not parent_signatures:
            raise ShieldToolError("parent_signatures must be a non-empty list")
        if len(parent_signatures) > 8:
            raise ShieldToolError("at most eight parents may be selected per call")
        parents = []
        for signature in parent_signatures:
            evidence = self.candidates.get(str(signature))
            if evidence is None or evidence.genvr is None:
                raise ShieldToolError(
                    f"parent signature {signature!r} has no GMC evidence"
                )
            parents.append(evidence)
        try:
            strategy = validate_shield_strategy(
                {
                    "operator_weights": arguments.get(
                        "operator_weights",
                        {name: 1.0 for name in OPERATORS},
                    ),
                    "focus_sector": arguments.get("focus_sector"),
                    "exploration_fraction": arguments.get(
                        "exploration_fraction",
                        0.20,
                    ),
                    "hypothesis": action.hypothesis,
                    "decision_reason": action.decision_reason,
                }
            )
        except (TypeError, ValueError) as exc:
            raise ShieldToolError(str(exc)) from exc
        proposals = {}
        for parent in parents:
            for proposal in self.proposal_model.propose(
                parent.genvr,
                proposals_per_parent,
                self.gmc_operator_memory,
                strategy,
            ):
                if proposal.design.signature in self.candidates:
                    continue
                previous = proposals.get(proposal.design.signature)
                if (
                    previous is None
                    or proposal.heuristic_score > previous.heuristic_score
                ):
                    proposals[proposal.design.signature] = proposal
        if not proposals:
            raise ShieldToolError(
                "the selected GMC parents and strategy produced no unseen legal candidates"
            )
        selected_proposals = sorted(
            proposals.values(),
            key=lambda proposal: proposal.heuristic_score,
            reverse=True,
        )[: min(remaining, self.budgets.max_structures_per_call)]
        generation_round = self.structure_generation_rounds + 1
        generated = []
        for proposal in selected_proposals:
            parent_evidence = self.candidates[proposal.parent_signature]
            gmc_guided_generation = (
                proposal.parent_signature != self.baseline_design.signature
                and parent_evidence.genvr is not None
            )
            evidence = CandidateEvidence(
                design=proposal.design,
                parent_signature=proposal.parent_signature,
                mutation=proposal.mutation.to_dict(),
                generation_round=generation_round,
                generation_guidance_fidelity=parent_evidence.genvr.fidelity,
                gmc_guided_generation=gmc_guided_generation,
                proposal_score=float(proposal.heuristic_score),
            )
            self.candidates[proposal.design.signature] = evidence
            record = {
                "stage": "gmc_guided_structure_generation",
                "parent_signature": proposal.parent_signature,
                "parent_guidance_fidelity": parent_evidence.genvr.fidelity,
                "generation_round": generation_round,
                "gmc_guided_generation": gmc_guided_generation,
                "operator_memory_source": "gmc_validated",
                "design_signature": proposal.design.signature,
                "mutation": proposal.mutation.to_dict(),
                "proposal_score": float(proposal.heuristic_score),
                "gmc_risk": None,
            }
            self.search_records.append(record)
            generated.append(record)
        self.usage.generated_structures += len(generated)
        self.structure_generation_rounds = generation_round
        self.requested_beam_width = beam_width
        return finite_json(
            {
                "tool": canonical_tool_name(action.tool),
                "stage": "gmc_guided_structure_generation",
                "evaluated_candidates": 0,
                "generated_candidates": len(generated),
                "generation_round": generation_round,
                "gmc_guided_candidates_generated": sum(
                    bool(record["gmc_guided_generation"])
                    for record in generated
                ),
                "strategy": {
                    "operator_weights": strategy.normalized_operator_weights(),
                    "focus_sector": strategy.focus_sector,
                    "exploration_fraction": strategy.exploration_fraction,
                    "importance_model": self.proposal_model.importance_model,
                },
                "best_new_candidates": generated[:8],
                "new_beam_signatures": list(self.beam_signatures),
                "discovery_progress": self.discovery_progress(),
                "budget": self.budget_usage()["generated_structures"],
            }
        )

    def _run_diffusion_batch(self, action: ShieldAgentAction) -> dict[str, Any]:
        remaining = (
            self.budgets.max_diffusion_evaluations
            - self.usage.diffusion_evaluations
        )
        if remaining <= 0:
            raise ShieldToolError("diffusion evaluation budget is exhausted")
        arguments = action.arguments
        proposals_per_parent = int(arguments.get("proposals_per_parent", 12))
        if not 1 <= proposals_per_parent <= self.budgets.max_diffusion_per_call:
            raise ShieldToolError("proposals_per_parent lies outside the declared budget")
        beam_width = int(arguments.get("beam_width", 4))
        if not 1 <= beam_width <= 8:
            raise ShieldToolError("beam_width must lie in [1, 8]")
        parent_signatures = arguments.get("parent_signatures") or self.beam_signatures
        if not isinstance(parent_signatures, list) or not parent_signatures:
            raise ShieldToolError("parent_signatures must be a non-empty list")
        if len(parent_signatures) > 8:
            raise ShieldToolError("at most eight parents may be selected per call")
        parents = []
        for signature in parent_signatures:
            evidence = self.candidates.get(str(signature))
            if evidence is None or evidence.diffusion is None:
                raise ShieldToolError(
                    f"parent signature {signature!r} has no diffusion evidence"
                )
            parents.append(evidence)
        try:
            strategy = validate_shield_strategy(
                {
                    "operator_weights": arguments.get(
                        "operator_weights",
                        {name: 1.0 for name in OPERATORS},
                    ),
                    "focus_sector": arguments.get("focus_sector"),
                    "exploration_fraction": arguments.get(
                        "exploration_fraction",
                        0.20,
                    ),
                    "hypothesis": action.hypothesis,
                    "decision_reason": action.decision_reason,
                }
            )
        except (TypeError, ValueError) as exc:
            raise ShieldToolError(str(exc)) from exc
        proposals = {}
        parent_risks = {}
        guidance_fidelities = {}
        proposal_memories = {}
        for parent in parents:
            parent_risks[parent.design.signature] = float(parent.diffusion_risk)
            guidance = self._guidance_evaluation(parent)
            if guidance is None:
                raise ShieldToolError(
                    f"parent signature {parent.design.signature!r} has no usable field"
                )
            guidance_fidelities[parent.design.signature] = guidance.fidelity
            proposal_memory = (
                self.gmc_operator_memory
                if parent.genvr is not None
                else self.operator_memory
            )
            proposal_memories[parent.design.signature] = proposal_memory
            for proposal in self.proposal_model.propose(
                guidance,
                proposals_per_parent,
                proposal_memory,
                strategy,
            ):
                if proposal.design.signature in self.candidates:
                    continue
                previous = proposals.get(proposal.design.signature)
                if previous is None or proposal.heuristic_score > previous.heuristic_score:
                    proposals[proposal.design.signature] = proposal
        if not proposals:
            raise ShieldToolError(
                "the selected parents and strategy produced no unseen legal candidates"
            )
        call_limit = min(remaining, self.budgets.max_diffusion_per_call)
        selected_proposals = list(proposals.values())[:call_limit]
        generation_round = self.structure_generation_rounds + 1
        evaluated = []
        acquisitions = {
            signature: float(evidence.diffusion_risk)
            for signature, evidence in self.candidates.items()
            if evidence.diffusion_risk is not None
        }
        baseline = self.candidates[self.baseline_design.signature].diffusion
        if baseline is None:
            raise RuntimeError("baseline diffusion evaluation is missing")
        for proposal in selected_proposals:
            evaluation = self.fast_evaluator.evaluate(proposal.design)
            risk = evaluation.robust_risk(
                baseline,
                self.metric_weights,
                self.risk_aversion,
            )
            parent_risk = parent_risks[proposal.parent_signature]
            improvement = parent_risk - risk
            self.operator_memory.update(proposal.mutation.operator, improvement)
            proposal_memory = proposal_memories[proposal.parent_signature]
            acquisition = risk - self.exploration_strength * proposal_memory.bonus(
                proposal.mutation.operator
            )
            parent_evidence = self.candidates[proposal.parent_signature]
            gmc_guided_generation = parent_evidence.genvr is not None
            evidence = CandidateEvidence(
                design=proposal.design,
                parent_signature=proposal.parent_signature,
                mutation=proposal.mutation.to_dict(),
                generation_round=generation_round,
                generation_guidance_fidelity=guidance_fidelities[
                    proposal.parent_signature
                ],
                gmc_guided_generation=gmc_guided_generation,
                diffusion=evaluation,
                diffusion_risk=float(risk),
            )
            self.candidates[proposal.design.signature] = evidence
            acquisitions[proposal.design.signature] = float(acquisition)
            record = {
                "parent_signature": proposal.parent_signature,
                "parent_guidance_fidelity": guidance_fidelities[
                    proposal.parent_signature
                ],
                "generation_round": generation_round,
                "gmc_guided_generation": gmc_guided_generation,
                "operator_memory_source": (
                    "gmc_validated" if gmc_guided_generation else "proposal_prefilter"
                ),
                "design_signature": proposal.design.signature,
                "mutation": proposal.mutation.to_dict(),
                "heuristic_score": float(proposal.heuristic_score),
                "diffusion_risk": float(risk),
                "acquisition": float(acquisition),
                "improvement_over_parent": float(improvement),
            }
            self.search_records.append(record)
            evaluated.append(record)
        self.usage.diffusion_evaluations += len(evaluated)
        self.structure_generation_rounds = generation_round
        pool = [
            evidence.diffusion
            for evidence in self.candidates.values()
            if evidence.diffusion is not None
        ]
        risks = {
            signature: float(evidence.diffusion_risk)
            for signature, evidence in self.candidates.items()
            if evidence.diffusion_risk is not None
        }
        selected_beam = SmartShieldAgent._select_search_beam(
            pool,
            risks,
            acquisitions,
            beam_width,
        )
        self.beam_signatures = [item.design.signature for item in selected_beam]
        best_records = sorted(evaluated, key=lambda item: item["diffusion_risk"])[:8]
        return finite_json(
            {
                "tool": canonical_tool_name(action.tool),
                "stage": "structure_generation",
                "evaluated_candidates": len(evaluated),
                "generation_round": generation_round,
                "gmc_guided_candidates_generated": sum(
                    bool(record["gmc_guided_generation"]) for record in evaluated
                ),
                "strategy": {
                    "operator_weights": strategy.normalized_operator_weights(),
                    "focus_sector": strategy.focus_sector,
                    "exploration_fraction": strategy.exploration_fraction,
                },
                "best_new_candidates": best_records,
                "new_beam_signatures": list(self.beam_signatures),
                "discovery_progress": self.discovery_progress(),
                "budget": self.budget_usage()["diffusion_evaluations"],
            }
        )

    def _selected_signatures(self, action: ShieldAgentAction) -> list[str]:
        raw = action.arguments.get("design_signatures")
        if not isinstance(raw, list) or not raw:
            raise ShieldToolError("design_signatures must be a non-empty list")
        signatures = []
        for value in raw:
            signature = str(value)
            if signature == self.baseline_design.signature:
                raise ShieldToolError("the baseline is evaluated automatically")
            if signature not in self.candidates:
                raise ShieldToolError(f"unknown candidate signature {signature!r}")
            if signature not in signatures:
                signatures.append(signature)
        return signatures

    def _promote(self, action: ShieldAgentAction, stage: str) -> dict[str, Any]:
        if stage == "genvr":
            evaluator = self.genvr_evaluator
            limit = self.budgets.max_genvr_candidates
            used = self.usage.genvr_candidates
            evidence_attribute = "genvr"
            risk_attribute = "genvr_risk"
        else:
            evaluator = self.exact_evaluator
            limit = self.budgets.max_exact_candidates
            used = self.usage.exact_candidates
            evidence_attribute = "exact"
            risk_attribute = "exact_risk"
        if evaluator is None:
            raise ShieldToolError(f"{stage} evaluator is not configured")
        signatures = self._selected_signatures(action)
        new_signatures = [
            signature
            for signature in signatures
            if getattr(self.candidates[signature], evidence_attribute) is None
        ]
        per_call_limit = (
            self.budgets.max_gmc_per_call if stage == "genvr" else limit
        )
        available = min(limit - used, per_call_limit)
        if len(new_signatures) > available:
            raise ShieldToolError(
                f"{stage} candidate budget permits only {max(available, 0)} new evaluations in this call"
            )
        if stage == "exact":
            for signature in new_signatures:
                evidence = self.candidates[signature]
                if (
                    evidence.genvr_risk is None
                    or evidence.genvr_risk >= self.verification_threshold
                ):
                    raise ShieldToolError(
                        f"candidate {signature} has not passed the GenVR/CFM gate"
                    )
        if stage == "genvr":
            if self._genvr_baseline is None:
                self._genvr_baseline = evaluator.evaluate(self.baseline_design)
            baseline = self._genvr_baseline
        else:
            if self._exact_baseline is None:
                self._exact_baseline = evaluator.evaluate(self.baseline_design)
            baseline = self._exact_baseline
        results = []
        previous_risks = []
        current_risks = []
        new_evaluations_completed = 0
        screening_round = self.gmc_screening_rounds + 1 if stage == "genvr" else None
        for signature in signatures:
            evidence = self.candidates[signature]
            evaluation = getattr(evidence, evidence_attribute)
            risk = getattr(evidence, risk_attribute)
            newly_evaluated = evaluation is None
            if evaluation is None:
                evaluation = evaluator.evaluate(evidence.design)
                risk = evaluation.robust_risk(
                    baseline,
                    self.metric_weights,
                    risk_aversion=0.0,
                )
                setattr(evidence, evidence_attribute, evaluation)
                setattr(evidence, risk_attribute, float(risk))
                if stage == "genvr" and evidence.mutation is not None:
                    parent = self.candidates.get(str(evidence.parent_signature))
                    parent_risk = (
                        parent.genvr_risk
                        if parent is not None and parent.genvr_risk is not None
                        else 1.0
                    )
                    self.gmc_operator_memory.update(
                        str(evidence.mutation["operator"]),
                        float(parent_risk) - float(risk),
                    )
            accepted = float(risk) < self.verification_threshold
            visualization = None
            visualization_error = None
            if stage == "genvr" and newly_evaluated:
                new_evaluations_completed += 1
                if self.render_gmc_candidates:
                    try:
                        visualization = write_gmc_candidate_snapshot(
                            self.output_dir,
                            self.baseline_design,
                            evaluation,
                            risk=float(risk),
                            accepted=accepted,
                            screening_round=int(screening_round or 0),
                            evaluation_index=used + new_evaluations_completed,
                            generation_round=evidence.generation_round,
                            gmc_guided_generation=evidence.gmc_guided_generation,
                        )
                    except Exception as exc:
                        visualization_error = f"{type(exc).__name__}: {exc}"
            if stage == "genvr":
                previous_risk = evidence.diffusion_risk
                if previous_risk is None and evidence.proposal_score is not None:
                    previous_risk = -float(evidence.proposal_score)
            else:
                previous_risk = evidence.genvr_risk
            if previous_risk is not None:
                previous_risks.append(float(previous_risk))
                current_risks.append(float(risk))
            if stage == "genvr":
                for record in reversed(self.search_records):
                    if record.get("design_signature") == signature:
                        record["gmc_screening_round"] = screening_round
                        record["gmc_risk"] = float(risk)
                        record["gmc_accepted"] = accepted
                        record["gmc_metrics"] = evaluation.metrics.to_dict()
                        record["gmc_runtime_s"] = float(evaluation.runtime_s)
                        if visualization is not None:
                            record["gmc_visualization"] = visualization
                        if visualization_error is not None:
                            record["gmc_visualization_error"] = visualization_error
                        parent = self.candidates.get(str(evidence.parent_signature))
                        parent_risk = (
                            parent.genvr_risk
                            if parent is not None and parent.genvr_risk is not None
                            else 1.0
                        )
                        record["gmc_improvement_over_parent"] = (
                            float(parent_risk) - float(risk)
                        )
                        break
            result = {
                "signature": signature,
                "risk": float(risk),
                "accepted": accepted,
                "generation_round": evidence.generation_round,
                "gmc_guided_generation": evidence.gmc_guided_generation,
                "mutation": evidence.mutation,
                "metrics": evaluation.metrics.to_dict(),
                "runtime_s": float(evaluation.runtime_s),
            }
            if visualization is not None:
                result["visualization"] = visualization
            if visualization_error is not None:
                result["visualization_error"] = visualization_error
            results.append(result)
        if stage == "genvr":
            self.usage.genvr_candidates += len(new_signatures)
            if new_signatures:
                self.gmc_screening_rounds = int(screening_round or 0)
            screened = sorted(
                (
                    evidence
                    for evidence in self.candidates.values()
                    if evidence.genvr_risk is not None
                ),
                key=lambda item: float(item.genvr_risk),
            )
            merged = [item.design.signature for item in screened]
            merged.extend(self.beam_signatures)
            self.beam_signatures = list(dict.fromkeys(merged))[
                : self.requested_beam_width
            ]
        else:
            self.usage.exact_candidates += len(new_signatures)
        return finite_json(
            {
                "tool": canonical_tool_name(action.tool),
                "stage": "gmc_screening" if stage == "genvr" else "projected_mc_audit",
                "baseline_metrics": baseline.metrics.to_dict(),
                "verification_threshold": self.verification_threshold,
                "results": results,
                "rank_correlation_with_previous": _rank_correlation(
                    previous_risks,
                    current_risks,
                ),
                "gmc_operator_memory": self.gmc_operator_memory.to_dict(),
                "discovery_progress": self.discovery_progress(),
                "budget": self.budget_usage()[
                    "genvr_candidates" if stage == "genvr" else "exact_candidates"
                ],
            }
        )

    def _run_unbiased_mc(self, action: ShieldAgentAction) -> dict[str, Any]:
        arguments = action.arguments
        signature = str(arguments.get("design_signature") or "")
        evidence = self.candidates.get(signature)
        if evidence is None or signature == self.baseline_design.signature:
            raise ShieldToolError("MC requires a known non-baseline candidate signature")
        if evidence.genvr is None or evidence.genvr_risk is None:
            raise ShieldToolError("MC requires prior GMC screening evidence")
        if evidence.genvr_risk >= self.verification_threshold:
            raise ShieldToolError(
                "MC is reserved for candidates that pass the GMC screening gate"
            )
        discovery_progress = self.discovery_progress()
        if not discovery_progress["minimum_gmc_batch_passed"]:
            raise ShieldToolError(
                "MC certification is locked until the minimum GMC discovery batch is complete"
            )
        if not discovery_progress["gmc_guided_cycle_passed"]:
            raise ShieldToolError(
                "MC certification is locked until GMC-guided structures are generated and screened"
            )
        if self.require_projected_mc_audit and (
            evidence.exact is None
            or evidence.exact_risk is None
            or evidence.exact_risk >= self.verification_threshold
        ):
            raise ShieldToolError(
                "this campaign requires a passed projected local-MC audit before MC"
            )
        if evidence.mc_reports:
            raise ShieldToolError(
                "a candidate may be MC-certified only once to prevent optional stopping"
            )
        if self._genvr_baseline is None:
            raise RuntimeError("GMC baseline is missing")
        replicates = int(arguments.get("replicates", self.minimum_mc_replicates))
        if not 1 <= replicates <= self.budgets.max_mc_replicates_per_call:
            raise ShieldToolError("replicates lies outside the declared per-call budget")
        default_histories = math.ceil(
            self.minimum_mc_histories_per_method / replicates
        )
        histories = int(arguments.get("histories", default_histories))
        nx = int(arguments.get("nx", self.minimum_mc_grid))
        ny = int(arguments.get("ny", self.minimum_mc_grid))
        n_levels_raw = arguments.get("n_levels")
        n_levels = None if n_levels_raw is None else int(n_levels_raw)
        flux_exponent = float(arguments.get("flux_exponent", 0.75))
        if not 2 <= histories <= self.budgets.max_mc_histories_per_call:
            raise ShieldToolError("histories lies outside the declared per-call budget")
        total_histories_per_method = histories * replicates
        if total_histories_per_method < self.minimum_mc_histories_per_method:
            raise ShieldToolError(
                "MC certification requires at least "
                f"{self.minimum_mc_histories_per_method} root histories per design and method"
            )
        if replicates < self.minimum_mc_replicates:
            raise ShieldToolError(
                "MC certification requires at least "
                f"{self.minimum_mc_replicates} independent replicates"
            )
        if (
            nx < self.minimum_mc_grid
            or ny < self.minimum_mc_grid
            or nx > self.budgets.max_mc_grid
            or ny > self.budgets.max_mc_grid
            or nx % 7
            or ny % 7
        ):
            raise ShieldToolError(
                "MC mesh dimensions must be divisible by seven and satisfy the frozen "
                f"range [{self.minimum_mc_grid}, {self.budgets.max_mc_grid}]"
            )
        if n_levels is not None and not 1 <= n_levels <= 20:
            raise ShieldToolError("n_levels must be null or lie in [1, 20]")
        if not 0.0 < flux_exponent <= 1.0:
            raise ShieldToolError("flux_exponent must lie in (0, 1]")
        requested_root_histories = 4 * histories * replicates
        remaining = self.budgets.max_mc_root_histories - self.usage.mc_root_histories
        if requested_root_histories > remaining:
            raise ShieldToolError(
                f"MC request needs {requested_root_histories} root histories but only {remaining} remain"
            )
        summary, arrays = self.mc_runner(
            self.baseline_design,
            evidence.design,
            self._genvr_baseline.flux,
            evidence.genvr.flux,
            nx=nx,
            ny=ny,
            histories=histories,
            replicates=replicates,
            n_levels=n_levels,
            flux_exponent=flux_exponent,
            seed=self.seed + self.mc_run_index * 1_000_003,
        )
        self.usage.mc_root_histories += requested_root_histories
        self.mc_run_index += 1
        run_dir = self.output_dir / "mc" / f"{self.mc_run_index:02d}_{signature}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "certification.json"
        arrays_path = run_dir / "fields.npz"
        summary_path.write_text(
            json.dumps(finite_json(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(arrays_path, **arrays)
        report = {
            "summary": finite_json(summary),
            "artifact": {
                "summary": str(summary_path),
                "fields": str(arrays_path),
            },
            "root_histories_charged": requested_root_histories,
        }
        evidence.mc_reports.append(report)
        return finite_json(
            {
                "tool": canonical_tool_name(action.tool),
                "stage": "unbiased_mc_certification",
                "signature": signature,
                "weight_window_guide": "genvr-cfm-response",
                "design_comparison": summary["design_comparison"],
                "baseline_consistency": summary["designs"]["baseline"][
                    "unbiased_consistency"
                ],
                "candidate_consistency": summary["designs"]["candidate"][
                    "unbiased_consistency"
                ],
                "population_control": {
                    "baseline": {
                        key: summary["designs"]["baseline"]["variance_reduced"][key]
                        for key in (
                            "transported_particles_per_root",
                            "split_children",
                            "split_cap_hits",
                            "max_bank_size",
                        )
                    },
                    "candidate": {
                        key: summary["designs"]["candidate"]["variance_reduced"][key]
                        for key in (
                            "transported_particles_per_root",
                            "split_children",
                            "split_cap_hits",
                            "max_bank_size",
                        )
                    },
                },
                "artifact": report["artifact"],
                "budget": self.budget_usage()["mc_root_histories"],
            }
        )

    def candidate(self, signature: str | None) -> CandidateEvidence | None:
        if signature is None:
            return None
        return self.candidates.get(signature)

    def best_exact_candidate(self) -> CandidateEvidence | None:
        accepted = [
            evidence
            for evidence in self.candidates.values()
            if evidence.design.signature != self.baseline_design.signature
            and evidence.exact_risk is not None
            and evidence.exact_risk < self.verification_threshold
        ]
        return min(accepted, key=lambda item: item.exact_risk) if accepted else None

    def best_gmc_candidate(self) -> CandidateEvidence | None:
        accepted = [
            evidence
            for evidence in self.candidates.values()
            if evidence.design.signature != self.baseline_design.signature
            and evidence.genvr_risk is not None
            and evidence.genvr_risk < self.verification_threshold
        ]
        return min(accepted, key=lambda item: item.genvr_risk) if accepted else None

    def public_candidates(self) -> list[dict[str, Any]]:
        return [
            self._candidate_payload(evidence)
            for evidence in sorted(
                self.candidates.values(),
                key=self._discovery_risk,
            )
        ]
