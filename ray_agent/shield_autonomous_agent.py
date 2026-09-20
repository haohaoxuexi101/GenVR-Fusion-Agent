from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .metrics import finite_json
from .shield_agent_protocol import (
    ShieldAgentAction,
    ShieldAgentPolicy,
    ShieldAgentResponseError,
    validate_agent_action,
)
from .shield_agent_tools import ShieldToolEnvironment, ShieldToolError
from .shield_agent_verifier import (
    ShieldAgentVerifier,
    ShieldVerificationDecision,
)


class ShieldEventLogger(Protocol):
    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class AutonomousShieldRunResult:
    final_claim: str
    candidate_signature: str | None
    termination: str
    steps_completed: int
    verifier_decision: ShieldVerificationDecision
    memory: tuple[dict[str, Any], ...]
    final_observation: dict[str, Any]
    policy_class: str
    response_failures: int = 0

    def to_dict(self, environment: ShieldToolEnvironment) -> dict[str, Any]:
        selected = environment.candidate(self.candidate_signature)
        return finite_json(
            {
                "schema_version": "1.0",
                "architecture": (
                    "LLM observation-hypothesis-tool-evidence loop with an "
                    "independent threshold-locked verifier"
                ),
                "policy_class": self.policy_class,
                "discovery_mode": environment.discovery_mode,
                "deterministic_policy_fallback": False,
                "final_claim": self.final_claim,
                "candidate_signature": self.candidate_signature,
                "final_design": selected.design.to_dict() if selected is not None else None,
                "termination": self.termination,
                "steps_completed": self.steps_completed,
                "response_failures": self.response_failures,
                "verifier_decision": self.verifier_decision.to_dict(),
                "budget_usage": environment.budget_usage(),
                "operator_memory": environment.operator_memory.to_dict(),
                "gmc_operator_memory": environment.gmc_operator_memory.to_dict(),
                "discovery_progress": environment.discovery_progress(),
                "agent_completeness": {
                    "autonomous_tool_selection": True,
                    "constraint_compiled_structures": True,
                    "gmc_primary_discovery_evaluator": True,
                    "gmc_feedback_changes_later_generations": bool(
                        environment.discovery_progress()[
                            "gmc_guided_cycle_passed"
                        ]
                    ),
                    "gmc_weight_windows_used_for_mc": any(
                        evidence.mc_reports
                        for evidence in environment.candidates.values()
                    ),
                    "independent_threshold_locked_verifier": True,
                    "full_trajectory_logged": True,
                    "scope": (
                        "complete closed loop for this frozen benchmark; not a claim of "
                        "universal engineering autonomy"
                    ),
                },
                "candidate_archive": environment.public_candidates(),
                "search_records": environment.search_records,
                "agent_memory": list(self.memory),
                "final_observation": self.final_observation,
            }
        )


class AutonomousShieldAgent:
    """Runs a real tool-using policy until an independent verifier terminates it."""

    def __init__(
        self,
        policy: ShieldAgentPolicy,
        environment: ShieldToolEnvironment,
        verifier: ShieldAgentVerifier,
        max_steps: int = 12,
        memory_window: int = 8,
        max_response_failures: int = 3,
        max_visible_candidates: int = 8,
        max_visible_beam: int = 4,
        logger: ShieldEventLogger | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if memory_window < 1:
            raise ValueError("memory_window must be positive")
        if max_response_failures < 1:
            raise ValueError("max_response_failures must be positive")
        if max_visible_candidates < 1:
            raise ValueError("max_visible_candidates must be positive")
        if max_visible_beam < 1:
            raise ValueError("max_visible_beam must be positive")
        if (
            verifier.criteria.require_projected_mc_audit
            != environment.require_projected_mc_audit
        ):
            raise ValueError(
                "verifier and environment disagree on the projected-MC audit requirement"
            )
        if (
            verifier.criteria.minimum_histories_per_method
            != environment.minimum_mc_histories_per_method
            or verifier.criteria.minimum_replicates
            != environment.minimum_mc_replicates
            or verifier.criteria.minimum_mc_grid
            != environment.minimum_mc_grid
        ):
            raise ValueError(
                "verifier and environment disagree on frozen Monte Carlo precision minima"
            )
        self.policy = policy
        self.environment = environment
        self.verifier = verifier
        self.max_steps = int(max_steps)
        self.memory_window = int(memory_window)
        self.max_response_failures = int(max_response_failures)
        self.max_visible_candidates = int(max_visible_candidates)
        self.max_visible_beam = int(max_visible_beam)
        self.logger = logger

    def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.write(event_type, finite_json(payload))

    def _record_trace(self, step: int) -> None:
        traces = getattr(self.policy, "last_traces", None)
        if traces:
            for trace in traces:
                self._log("llm_exchange", {"step": step, **trace})
            return
        trace = getattr(self.policy, "last_trace", None)
        if trace is not None:
            self._log("llm_exchange", {"step": step, **trace})

    @staticmethod
    def _compact_metrics(metrics: Any) -> Any:
        if not isinstance(metrics, dict):
            return metrics
        names = ("far_mean", "far_cvar90", "far_max", "worst_sector_mean")
        return {name: metrics[name] for name in names if name in metrics}

    @classmethod
    def _compact_tool_result(cls, result: dict[str, Any]) -> dict[str, Any]:
        compact = {
            key: result[key]
            for key in (
                "tool",
                "stage",
                "signature",
                "evaluated_candidates",
                "generated_candidates",
                "generation_round",
                "strategy",
                "new_beam_signatures",
                "verification_threshold",
                "rank_correlation_with_previous",
                "budget",
                "artifact",
                "population_control",
                "finish_requested",
                "arguments",
                "verifier_decision",
            )
            if key in result
        }
        if "baseline_metrics" in result:
            compact["baseline_metrics"] = cls._compact_metrics(
                result["baseline_metrics"]
            )
        if "best_new_candidates" in result:
            compact["best_new_candidates"] = [
                {
                    key: item[key]
                    for key in (
                        "parent_signature",
                        "design_signature",
                        "mutation",
                        "parent_guidance_fidelity",
                        "generation_round",
                        "proposal_score",
                        "diffusion_risk",
                        "gmc_risk",
                        "improvement_over_parent",
                        "gmc_improvement_over_parent",
                    )
                    if key in item
                }
                for item in result["best_new_candidates"][:4]
            ]
        if "results" in result:
            compact["results"] = [
                {
                    **{
                        key: item[key]
                        for key in ("signature", "risk", "accepted")
                        if key in item
                    },
                    "metrics": cls._compact_metrics(item.get("metrics")),
                }
                for item in result["results"]
            ]
        for key in (
            "design_comparison",
            "baseline_consistency",
            "candidate_consistency",
        ):
            if key in result:
                compact[key] = result[key]
        return finite_json(compact)

    def _result(
        self,
        decision: ShieldVerificationDecision,
        termination: str,
        steps_completed: int,
        memory: list[dict[str, Any]],
        response_failures: int,
    ) -> AutonomousShieldRunResult:
        result = AutonomousShieldRunResult(
            final_claim=str(decision.final_claim or "inconclusive"),
            candidate_signature=decision.candidate_signature,
            termination=termination,
            steps_completed=steps_completed,
            verifier_decision=decision,
            memory=tuple(memory),
            final_observation=self.environment.observe(),
            policy_class=type(self.policy).__name__,
            response_failures=response_failures,
        )
        self._log(
            "run_completed",
            {
                "final_claim": result.final_claim,
                "candidate_signature": result.candidate_signature,
                "termination": result.termination,
                "steps_completed": result.steps_completed,
                "response_failures": result.response_failures,
                "verifier_decision": decision.to_dict(),
            },
        )
        return result

    def run(self) -> AutonomousShieldRunResult:
        memory: list[dict[str, Any]] = []
        last_feedback: dict[str, Any] | None = None
        scientific_step = 1
        total_response_failures = 0
        consecutive_response_failures = 0

        while scientific_step <= self.max_steps:
            step = scientific_step
            observation = self.environment.observe_for_agent(
                maximum_candidates=self.max_visible_candidates,
                maximum_beam=self.max_visible_beam,
            )
            observation["currently_executable_scientific_tools"] = (
                self.environment.executable_scientific_tools(
                    self.verifier.criteria.minimum_histories_per_method,
                    self.verifier.criteria.minimum_replicates,
                )
            )
            agent_observation = {
                **observation,
                "verifier_contract": self.verifier.public_contract(
                    self.environment
                ),
                "agent_control": {
                    "step": step,
                    "maximum_steps": self.max_steps,
                    "remaining_steps_after_this_decision": self.max_steps - step,
                    "decision_attempt": consecutive_response_failures + 1,
                    "maximum_consecutive_response_failures": (
                        self.max_response_failures
                    ),
                    "last_environment_feedback": last_feedback,
                },
            }
            visible_memory = memory[-self.memory_window :]
            executable_tools = set(
                observation["currently_executable_scientific_tools"]
            )
            executable_tools.add("finish")
            tools = [
                tool
                for tool in self.environment.tool_specs()
                if tool["name"] in executable_tools
            ]
            self._log(
                "agent_input",
                {
                    "step": step,
                    "observation": agent_observation,
                    "tools": tools,
                    "memory": visible_memory,
                },
            )

            try:
                self.policy.last_trace = None
            except (AttributeError, TypeError):
                pass
            try:
                self.policy.last_traces = []
            except (AttributeError, TypeError):
                pass
            try:
                proposed_action = self.policy.decide(
                    agent_observation,
                    tools,
                    visible_memory,
                )
                action = validate_agent_action(proposed_action.to_dict())
            except ShieldAgentResponseError as exc:
                self._record_trace(step)
                traces = getattr(self.policy, "last_traces", None) or []
                failed_attempts = sum(
                    trace.get("accepted") is False for trace in traces
                ) or 1
                total_response_failures += failed_attempts
                consecutive_response_failures += 1
                feedback = {
                    "type": "agent_response_rejected",
                    "step": step,
                    "decision_attempt": consecutive_response_failures,
                    "api_response_failures": failed_attempts,
                    "error": str(exc),
                    "instruction": (
                        "Return one valid allowlisted JSON tool action; no fallback action "
                        "will be substituted."
                    ),
                }
                self._log("agent_output", {"step": step, "accepted": False, **feedback})
                self._log("environment_feedback", feedback)
                memory.append(feedback)
                last_feedback = feedback
                if consecutive_response_failures >= self.max_response_failures:
                    reason = (
                        "maximum consecutive agent response failures reached "
                        f"({self.max_response_failures})"
                    )
                    decision = self.verifier.review_hard_stop(
                        self.environment,
                        reason=reason,
                    )
                    self._log(
                        "verifier_decision",
                        {"step": step, **decision.to_dict()},
                    )
                    return self._result(
                        decision,
                        "maximum_response_failures_reached",
                        step - 1,
                        memory,
                        total_response_failures,
                    )
                continue
            except ValueError as exc:
                self._record_trace(step)
                traces = getattr(self.policy, "last_traces", None) or []
                failed_attempts = sum(
                    trace.get("accepted") is False for trace in traces
                ) or 1
                total_response_failures += failed_attempts
                consecutive_response_failures += 1
                feedback = {
                    "type": "agent_action_rejected",
                    "step": step,
                    "decision_attempt": consecutive_response_failures,
                    "api_response_failures": failed_attempts,
                    "error": str(exc),
                    "instruction": "Choose one tool using only its declared argument schema.",
                }
                self._log("agent_output", {"step": step, "accepted": False, **feedback})
                self._log("environment_feedback", feedback)
                memory.append(feedback)
                last_feedback = feedback
                if consecutive_response_failures >= self.max_response_failures:
                    reason = (
                        "maximum consecutive agent response failures reached "
                        f"({self.max_response_failures})"
                    )
                    decision = self.verifier.review_hard_stop(
                        self.environment,
                        reason=reason,
                    )
                    self._log(
                        "verifier_decision",
                        {"step": step, **decision.to_dict()},
                    )
                    return self._result(
                        decision,
                        "maximum_response_failures_reached",
                        step - 1,
                        memory,
                        total_response_failures,
                    )
                continue
            except Exception as exc:
                self._record_trace(step)
                self._log(
                    "agent_failure",
                    {
                        "step": step,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fallback_used": False,
                    },
                )
                raise

            self._record_trace(step)
            traces = getattr(self.policy, "last_traces", None) or []
            total_response_failures += sum(
                trace.get("accepted") is False for trace in traces
            )
            consecutive_response_failures = 0
            self._log(
                "agent_output",
                {"step": step, "accepted": True, "action": action.to_dict()},
            )
            self._log(
                "decision",
                {
                    "step": step,
                    "policy": type(self.policy).__name__,
                    "tool": action.tool,
                    "hypothesis": action.hypothesis,
                    "expected_observation": action.expected_observation,
                    "decision_reason": action.decision_reason,
                },
            )
            self._log(
                "tool_call",
                {
                    "step": step,
                    "tool": action.tool,
                    "arguments": action.arguments,
                },
            )

            try:
                tool_result = self.environment.execute(action)
            except ShieldToolError as exc:
                feedback = {
                    "type": "tool_call_rejected",
                    "step": step,
                    "tool": action.tool,
                    "error": str(exc),
                }
                self._log(
                    "tool_result",
                    {
                        "step": step,
                        "tool": action.tool,
                        "accepted": False,
                        "error": str(exc),
                    },
                )
                self._log("environment_feedback", feedback)
                memory.append(
                    {
                        "step": step,
                        "action": action.to_dict(),
                        "tool_result": feedback,
                    }
                )
                last_feedback = feedback
                scientific_step += 1
                continue
            except Exception as exc:
                self._log(
                    "tool_failure",
                    {
                        "step": step,
                        "tool": action.tool,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise

            if action.tool == "finish":
                decision = self.verifier.review_finish(action, self.environment)
                combined_result = {
                    **tool_result,
                    "verifier_decision": decision.to_dict(),
                }
                self._log(
                    "tool_result",
                    {
                        "step": step,
                        "tool": action.tool,
                        "accepted": decision.accepted,
                        "result": combined_result,
                    },
                )
                self._log(
                    "verifier_decision",
                    {"step": step, **decision.to_dict()},
                )
                compact_result = self._compact_tool_result(combined_result)
                memory.append(
                    {
                        "step": step,
                        "action": action.to_dict(),
                        "tool_result": compact_result,
                    }
                )
                if decision.accepted:
                    return self._result(
                        decision,
                        "verifier_accepted_finish",
                        step,
                        memory,
                        total_response_failures,
                    )
                feedback = {
                    "type": "finish_rejected",
                    "step": step,
                    "reasons": list(decision.reasons),
                    "failed_checks": {
                        name: check
                        for name, check in decision.checks.items()
                        if isinstance(check, dict) and check.get("passed") is False
                    },
                    "instruction": (
                        "Continue gathering evidence with an executable scientific tool; "
                        "the verifier thresholds cannot be changed."
                    ),
                }
                self._log("environment_feedback", feedback)
                last_feedback = feedback
                scientific_step += 1
                continue

            compact_result = self._compact_tool_result(tool_result)
            self._log(
                "tool_result",
                {
                    "step": step,
                    "tool": action.tool,
                    "accepted": True,
                    "result": tool_result,
                },
            )
            feedback = {
                "type": "scientific_observation",
                "step": step,
                "tool": action.tool,
                "tool_result": compact_result,
            }
            self._log("environment_feedback", feedback)
            memory.append(
                {
                    "step": step,
                    "action": action.to_dict(),
                    "tool_result": compact_result,
                }
            )
            last_feedback = feedback
            scientific_step += 1

        decision = self.verifier.review_hard_stop(
            self.environment,
            reason=f"maximum agent steps reached ({self.max_steps})",
        )
        self._log("verifier_decision", {"step": self.max_steps, **decision.to_dict()})
        return self._result(
            decision,
            "maximum_steps_reached",
            self.max_steps,
            memory,
            total_response_failures,
        )
