from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

from .metrics import finite_json
from .shield_agent_protocol import ShieldAgentAction
from .shield_agent_tools import CandidateEvidence, ShieldToolEnvironment


@dataclass(frozen=True)
class ShieldVerificationCriteria:
    maximum_abs_unbiased_consistency_z: float = 2.0
    maximum_far_field_difference_z: float = -1.96
    minimum_histories_per_method: int = 1000
    minimum_replicates: int = 2
    minimum_mc_grid: int = 7
    require_far_field_ci_separation: bool = True
    require_projected_mc_audit: bool = False

    def __post_init__(self) -> None:
        if self.maximum_abs_unbiased_consistency_z <= 0.0:
            raise ValueError("maximum consistency z-score must be positive")
        if self.maximum_far_field_difference_z >= 0.0:
            raise ValueError("maximum far-field difference z-score must be negative")
        if self.minimum_histories_per_method < 2:
            raise ValueError("minimum MC histories per method must be at least two")
        if self.minimum_replicates < 1:
            raise ValueError("minimum MC replicates must be positive")
        if self.minimum_mc_grid < 7 or self.minimum_mc_grid % 7:
            raise ValueError(
                "minimum MC grid must be divisible by seven and at least seven"
            )


@dataclass(frozen=True)
class ShieldVerificationDecision:
    accepted: bool
    terminal: bool
    requested_claim: str
    final_claim: str | None
    candidate_signature: str | None
    checks: dict[str, Any]
    reasons: tuple[str, ...]
    hard_stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return finite_json(
            {
                "accepted": self.accepted,
                "terminal": self.terminal,
                "requested_claim": self.requested_claim,
                "final_claim": self.final_claim,
                "candidate_signature": self.candidate_signature,
                "checks": self.checks,
                "reasons": list(self.reasons),
                "hard_stop_reason": self.hard_stop_reason,
            }
        )


def _check(
    passed: bool,
    observed: Any,
    required: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


class ShieldAgentVerifier:
    """Independent, threshold-locked reviewer for autonomous shielding claims."""

    def __init__(
        self,
        criteria: ShieldVerificationCriteria | None = None,
    ) -> None:
        self.criteria = criteria or ShieldVerificationCriteria()

    def _projected_mc_audit_required(
        self,
        environment: ShieldToolEnvironment,
    ) -> bool:
        return bool(
            self.criteria.require_projected_mc_audit
            or environment.require_projected_mc_audit
        )

    def public_contract(
        self,
        environment: ShieldToolEnvironment,
    ) -> dict[str, Any]:
        return finite_json(
            {
                "authority": "independent verifier; the agent cannot modify these values",
                "required_fidelity_chain": [
                    "genvr-cfm-response",
                    "unbiased-mc",
                ],
                "optional_diagnostic": "projected-local-mc-response",
                "fidelity_display_names": {
                    "genvr-cfm-response": "GenVR/CFM projected response",
                    "exact-local-mc-response": "Projected local-MC response (legacy machine id)",
                    "unbiased-mc": "Unbiased global Monte Carlo",
                },
                "relative_risk_threshold": environment.verification_threshold,
                "maximum_abs_far_field_analog_vr_consistency_z": (
                    self.criteria.maximum_abs_unbiased_consistency_z
                ),
                "maximum_far_field_candidate_minus_baseline_z": (
                    self.criteria.maximum_far_field_difference_z
                ),
                "minimum_histories_per_design_and_method": (
                    self.criteria.minimum_histories_per_method
                ),
                "minimum_replicates": self.criteria.minimum_replicates,
                "minimum_mc_grid": [
                    self.criteria.minimum_mc_grid,
                    self.criteria.minimum_mc_grid,
                ],
                "require_far_field_95pct_interval_separation": (
                    self.criteria.require_far_field_ci_separation
                ),
                "mc_certifications_per_candidate": 1,
                "minimum_gmc_candidates_before_mc": (
                    environment.minimum_gmc_candidates_before_mc
                ),
                "minimum_gmc_guided_candidates_before_mc": (
                    environment.minimum_gmc_guided_candidates_before_mc
                ),
                "required_discovery_cycle": (
                    "generate -> GMC screen -> regenerate from GMC field -> GMC rescreen"
                ),
                "projected_mc_audit_required": (
                    self._projected_mc_audit_required(environment)
                ),
                "fom_is_acceptance_criterion": False,
            }
        )

    def _candidate_checks(
        self,
        environment: ShieldToolEnvironment,
        evidence: CandidateEvidence | None,
    ) -> tuple[dict[str, Any], list[str]]:
        checks: dict[str, Any] = {}
        reasons: list[str] = []
        threshold = environment.verification_threshold
        baseline_signature = environment.baseline_design.signature
        if evidence is None:
            checks["candidate_exists"] = _check(False, None, "known candidate")
            return checks, ["the requested candidate is not present in the evidence archive"]

        signature = evidence.design.signature
        non_baseline = signature != baseline_signature
        checks["non_baseline_candidate"] = _check(
            non_baseline,
            signature,
            f"signature different from baseline {baseline_signature}",
        )
        if not non_baseline:
            reasons.append("the baseline cannot be certified as an improvement over itself")

        genvr_risk = evidence.genvr_risk
        genvr_passed = (
            genvr_risk is not None
            and math.isfinite(float(genvr_risk))
            and float(genvr_risk) < threshold
        )
        checks["genvr_cfm_gate"] = _check(
            genvr_passed,
            genvr_risk,
            f"risk < {threshold:.12g}",
        )
        if not genvr_passed:
            reasons.append("candidate has not passed the GenVR/CFM gate")

        discovery_progress = environment.discovery_progress()
        minimum_gmc_batch_passed = bool(
            discovery_progress["minimum_gmc_batch_passed"]
        )
        checks["minimum_gmc_discovery_batch"] = _check(
            minimum_gmc_batch_passed,
            discovery_progress["gmc_screened_structures"],
            f">= {environment.minimum_gmc_candidates_before_mc}",
        )
        if not minimum_gmc_batch_passed:
            reasons.append("the minimum GMC discovery batch is incomplete")

        gmc_guided_cycle_passed = bool(
            discovery_progress["gmc_guided_cycle_passed"]
        )
        checks["gmc_guided_regeneration"] = _check(
            gmc_guided_cycle_passed,
            discovery_progress["gmc_guided_and_screened"],
            f">= {environment.minimum_gmc_guided_candidates_before_mc}",
        )
        if not gmc_guided_cycle_passed:
            reasons.append(
                "GMC evidence has not yet guided and screened the required new generation"
            )

        exact_risk = evidence.exact_risk
        exact_passed = (
            exact_risk is not None
            and math.isfinite(float(exact_risk))
            and float(exact_risk) < threshold
        )
        projected_mc_audit_required = self._projected_mc_audit_required(environment)
        checks["projected_local_mc_audit"] = _check(
            exact_passed if projected_mc_audit_required else True,
            exact_risk,
            (
                f"risk < {threshold:.12g}"
                if projected_mc_audit_required
                else "optional diagnostic"
            ),
        )
        if projected_mc_audit_required and not exact_passed:
            reasons.append(
                "candidate has not passed the required projected local-MC audit"
            )

        report = evidence.mc_reports[-1] if evidence.mc_reports else None
        checks["unbiased_mc_present"] = _check(
            report is not None and len(evidence.mc_reports) == 1,
            len(evidence.mc_reports),
            "exactly one completed certification",
        )
        if report is None or len(evidence.mc_reports) != 1:
            reasons.append("candidate has no unbiased Monte Carlo certification")
            return checks, reasons

        summary = report.get("summary")
        if not isinstance(summary, dict):
            checks["mc_schema"] = _check(False, type(summary).__name__, "object")
            reasons.append("Monte Carlo certification has an invalid summary schema")
            return checks, reasons

        try:
            reported_baseline = summary["designs"]["baseline"]["design"]["signature"]
            reported_candidate = summary["designs"]["candidate"]["design"]["signature"]
            replicates = int(summary["replicates"])
            grid = [int(value) for value in summary["grid"]]
            method_histories = {
                f"{design_name}.{method_name}": int(
                    summary["designs"][design_name][method_name]["histories"]
                )
                for design_name in ("baseline", "candidate")
                for method_name in ("analog", "variance_reduced")
            }
            baseline_consistency = float(
                summary["designs"]["baseline"]["unbiased_consistency"]
                ["far_field"]["unpaired_consistency_z"]
            )
            candidate_consistency = float(
                summary["designs"]["candidate"]["unbiased_consistency"]
                ["far_field"]["unpaired_consistency_z"]
            )
            comparison = summary["design_comparison"]["far_field"]
            candidate_over_baseline = float(comparison["candidate_over_baseline"])
            difference_z = float(comparison["difference_z"])
            ci_separated = bool(
                comparison["candidate_upper_below_baseline_lower"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            checks["mc_schema"] = _check(False, str(exc), "complete certification schema")
            reasons.append("Monte Carlo certification is missing required audit fields")
            return checks, reasons

        identity_passed = (
            reported_baseline == baseline_signature
            and reported_candidate == signature
        )
        checks["mc_design_identity"] = _check(
            identity_passed,
            {
                "baseline": reported_baseline,
                "candidate": reported_candidate,
            },
            {
                "baseline": baseline_signature,
                "candidate": signature,
            },
        )
        if not identity_passed:
            reasons.append("Monte Carlo report does not identify the requested design pair")

        histories_passed = all(
            histories >= self.criteria.minimum_histories_per_method
            for histories in method_histories.values()
        )
        checks["mc_histories"] = _check(
            histories_passed,
            method_histories,
            f"each >= {self.criteria.minimum_histories_per_method}",
        )
        if not histories_passed:
            reasons.append("Monte Carlo root-history count is below the frozen verifier minimum")

        replicates_passed = replicates >= self.criteria.minimum_replicates
        checks["mc_replicates"] = _check(
            replicates_passed,
            replicates,
            f">= {self.criteria.minimum_replicates}",
        )
        if not replicates_passed:
            reasons.append("Monte Carlo replicate count is below the frozen verifier minimum")

        grid_passed = (
            len(grid) == 2
            and all(value >= self.criteria.minimum_mc_grid for value in grid)
            and all(value % 7 == 0 for value in grid)
        )
        checks["mc_grid"] = _check(
            grid_passed,
            grid,
            (
                "both dimensions divisible by seven and >= "
                f"{self.criteria.minimum_mc_grid}"
            ),
        )
        if not grid_passed:
            reasons.append("Monte Carlo tally grid is below the frozen verifier minimum")

        consistency_limit = self.criteria.maximum_abs_unbiased_consistency_z
        consistency_passed = (
            math.isfinite(baseline_consistency)
            and math.isfinite(candidate_consistency)
            and abs(baseline_consistency) <= consistency_limit
            and abs(candidate_consistency) <= consistency_limit
        )
        checks["analog_weight_window_consistency"] = _check(
            consistency_passed,
            {
                "baseline_far_field_z": baseline_consistency,
                "candidate_far_field_z": candidate_consistency,
            },
            f"both absolute z-scores <= {consistency_limit:.12g}",
        )
        if not consistency_passed:
            reasons.append("analog and weight-window far-field estimates are inconsistent")

        ratio_passed = (
            math.isfinite(candidate_over_baseline)
            and candidate_over_baseline < threshold
        )
        checks["mc_improvement_threshold"] = _check(
            ratio_passed,
            candidate_over_baseline,
            f"candidate/baseline < {threshold:.12g}",
        )
        if not ratio_passed:
            reasons.append("unbiased MC does not meet the frozen improvement threshold")

        difference_passed = (
            math.isfinite(difference_z)
            and difference_z <= self.criteria.maximum_far_field_difference_z
        )
        checks["mc_far_field_difference_z"] = _check(
            difference_passed,
            difference_z,
            f"<= {self.criteria.maximum_far_field_difference_z:.12g}",
        )
        if not difference_passed:
            reasons.append("far-field reduction is not statistically resolved by the z test")

        interval_passed = (
            ci_separated if self.criteria.require_far_field_ci_separation else True
        )
        checks["mc_far_field_ci_separation"] = _check(
            interval_passed,
            ci_separated,
            self.criteria.require_far_field_ci_separation,
        )
        if not interval_passed:
            reasons.append("candidate and baseline far-field 95% intervals overlap")

        checks["deep_far_field_diagnostic"] = summary["design_comparison"].get(
            "deep_far_field"
        )
        checks["fom_is_not_acceptance_criterion"] = {
            "passed": True,
            "observed": {
                design_name: summary["designs"][design_name][
                    "unbiased_consistency"
                ]["far_field"].get("fom_gain")
                for design_name in ("baseline", "candidate")
            },
            "required": "reported for efficiency only",
        }
        return checks, reasons

    def _verified_candidates(
        self,
        environment: ShieldToolEnvironment,
    ) -> list[str]:
        verified = []
        for evidence in environment.candidates.values():
            checks, _ = self._candidate_checks(environment, evidence)
            if checks and all(
                item.get("passed", True)
                for item in checks.values()
                if isinstance(item, dict) and "passed" in item
            ):
                verified.append(evidence.design.signature)
        return verified

    def review_finish(
        self,
        action: ShieldAgentAction,
        environment: ShieldToolEnvironment,
    ) -> ShieldVerificationDecision:
        if action.tool != "finish":
            raise ValueError("the verifier only reviews finish actions")
        claim = str(action.arguments.get("claim") or "")
        raw_signature = action.arguments.get("candidate_signature")
        signature = None if raw_signature is None else str(raw_signature)

        if claim not in {
            "verified_improvement",
            "no_verified_improvement",
            "inconclusive",
        }:
            return ShieldVerificationDecision(
                accepted=False,
                terminal=False,
                requested_claim=claim,
                final_claim=None,
                candidate_signature=signature,
                checks={"recognized_claim": _check(False, claim, "declared claim")},
                reasons=("the requested finish claim is not recognized",),
            )

        if claim == "verified_improvement":
            evidence = environment.candidate(signature)
            checks, reasons = self._candidate_checks(environment, evidence)
            accepted = not reasons and all(
                item.get("passed", True)
                for item in checks.values()
                if isinstance(item, dict) and "passed" in item
            )
            return ShieldVerificationDecision(
                accepted=accepted,
                terminal=accepted,
                requested_claim=claim,
                final_claim=claim if accepted else None,
                candidate_signature=signature,
                checks=checks,
                reasons=tuple(reasons),
            )

        executable = environment.executable_scientific_tools(
            self.criteria.minimum_histories_per_method,
            self.criteria.minimum_replicates,
        )
        completion_blockers = [
            tool
            for tool in executable
            if not (
                tool == "audit_projected_mc"
                and not self._projected_mc_audit_required(environment)
            )
        ]
        verified = self._verified_candidates(environment)
        non_baseline_count = len(environment.candidates) - 1
        gmc_pass_without_mc = [
            evidence.design.signature
            for evidence in environment.candidates.values()
            if evidence.genvr_risk is not None
            and evidence.genvr_risk < environment.verification_threshold
            and (
                not self._projected_mc_audit_required(environment)
                or (
                    evidence.exact_risk is not None
                    and evidence.exact_risk < environment.verification_threshold
                )
            )
            and not evidence.mc_reports
        ]
        common_checks = {
            "candidate_signature_is_null": _check(signature is None, signature, None),
            "non_baseline_search_executed": _check(
                non_baseline_count > 0,
                non_baseline_count,
                "> 0",
            ),
            "no_verified_candidate_exists": _check(
                not verified,
                verified,
                [],
            ),
            "no_scientific_tool_remains_executable": _check(
                not completion_blockers,
                completion_blockers,
                [],
            ),
        }

        if claim == "no_verified_improvement":
            common_checks["all_gmc_passes_mc_tested"] = _check(
                not gmc_pass_without_mc,
                gmc_pass_without_mc,
                [],
            )
            reasons = []
            if signature is not None:
                reasons.append("no_verified_improvement must not name a candidate")
            if non_baseline_count <= 0:
                reasons.append("no non-baseline design has been evaluated")
            if verified:
                reasons.append("at least one archived candidate already satisfies the verifier")
            if completion_blockers:
                reasons.append("scientific tools and budget remain available")
            if gmc_pass_without_mc:
                reasons.append(
                    "a GMC-approved candidate still lacks unbiased MC evidence"
                )
            accepted = not reasons
            return ShieldVerificationDecision(
                accepted=accepted,
                terminal=accepted,
                requested_claim=claim,
                final_claim=claim if accepted else None,
                candidate_signature=None,
                checks=common_checks,
                reasons=tuple(reasons),
            )

        reasons = []
        if signature is not None:
            reasons.append("inconclusive must not name a preferred candidate")
        if completion_blockers:
            reasons.append("inconclusive is premature while scientific tools remain executable")
        if verified:
            reasons.append("a verified candidate exists and must be reported explicitly")
        accepted = not reasons
        return ShieldVerificationDecision(
            accepted=accepted,
            terminal=accepted,
            requested_claim=claim,
            final_claim=claim if accepted else None,
            candidate_signature=None,
            checks=common_checks,
            reasons=tuple(reasons),
        )

    def review_hard_stop(
        self,
        environment: ShieldToolEnvironment,
        reason: str,
    ) -> ShieldVerificationDecision:
        verified = self._verified_candidates(environment)
        return ShieldVerificationDecision(
            accepted=True,
            terminal=True,
            requested_claim="inconclusive",
            final_claim="inconclusive",
            candidate_signature=None,
            checks={
                "hard_stop_enforced": _check(True, reason, "non-agent stop condition"),
                "verified_candidates_not_selected_by_agent": verified,
            },
            reasons=(
                "the autonomous loop reached a non-agent hard stop; no unrequested "
                "scientific claim was synthesized",
            ),
            hard_stop_reason=reason,
        )
