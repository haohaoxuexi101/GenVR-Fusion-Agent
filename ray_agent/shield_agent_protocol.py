from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import json
import math
import urllib.request

from .policies import DeepSeekPolicy


CANONICAL_AGENT_TOOLS = (
    "generate_structures",
    "screen_gmc",
    "audit_projected_mc",
    "certify_mc",
    "finish",
)

TOOL_ALIASES = {
    "generate_structures": "generate_structures",
    "run_diffusion_batch": "generate_structures",
    "screen_gmc": "screen_gmc",
    "promote_genvr": "screen_gmc",
    "audit_projected_mc": "audit_projected_mc",
    "promote_exact": "audit_projected_mc",
    "certify_mc": "certify_mc",
    "run_unbiased_mc": "certify_mc",
    "finish": "finish",
}

AGENT_TOOLS = tuple(TOOL_ALIASES)

TOOL_ARGUMENTS = {
    "generate_structures": {
        "operator_weights",
        "focus_sector",
        "exploration_fraction",
        "proposals_per_parent",
        "beam_width",
        "parent_signatures",
    },
    "screen_gmc": {"design_signatures"},
    "audit_projected_mc": {"design_signatures"},
    "certify_mc": {
        "design_signature",
        "histories",
        "replicates",
        "nx",
        "ny",
        "n_levels",
        "flux_exponent",
    },
    "finish": {"candidate_signature", "claim", "summary"},
}


def canonical_tool_name(tool: str) -> str:
    try:
        return TOOL_ALIASES[tool]
    except KeyError as exc:
        raise ValueError(f"unknown agent tool {tool!r}") from exc

BANNED_ARGUMENT_KEYS = {
    "absorbers",
    "geometry",
    "material_budget",
    "materials",
    "metric_weights",
    "objective",
    "source",
    "source_cell",
    "threshold",
    "verification_threshold",
}

SHIELD_OPERATOR_NAMES = {
    "adjoint_sensitivity",
    "gmc_path_importance",
    "channel_block",
    "inner_barrier",
    "hotspot_cap",
    "symmetry_repair",
    "redistribute",
    "random_explore",
}


class ShieldAgentResponseError(RuntimeError):
    """Raised when the LLM responds but does not produce a valid tool action."""


@dataclass(frozen=True)
class ShieldAgentAction:
    tool: str
    arguments: dict[str, Any]
    hypothesis: str
    expected_observation: str
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "hypothesis": self.hypothesis,
            "expected_observation": self.expected_observation,
            "decision_reason": self.decision_reason,
        }


class ShieldAgentPolicy(Protocol):
    last_trace: dict[str, Any] | None

    def decide(
        self,
        observation: dict[str, Any],
        tools: list[dict[str, Any]],
        memory: list[dict[str, Any]],
    ) -> ShieldAgentAction:
        ...


def _check_banned_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in BANNED_ARGUMENT_KEYS:
                raise ValueError(
                    f"agent arguments cannot modify protected field {key!r}"
                )
            _check_banned_keys(item)
    elif isinstance(value, list):
        for item in value:
            _check_banned_keys(item)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_tool_arguments(tool: str, arguments: dict[str, Any]) -> None:
    tool = canonical_tool_name(tool)
    if tool == "generate_structures":
        weights = arguments.get("operator_weights")
        if weights is not None:
            if not isinstance(weights, dict):
                raise ValueError("operator_weights must be an object")
            unknown = set(weights) - SHIELD_OPERATOR_NAMES
            if unknown:
                raise ValueError(f"unknown shield operators: {sorted(unknown)}")
            if any(not _is_number(value) for value in weights.values()):
                raise ValueError("operator weights must be finite numbers")
            if any(not 0.0 <= float(value) <= 1.0 for value in weights.values()):
                raise ValueError("operator weights must lie in [0, 1]")
            if weights and sum(float(value) for value in weights.values()) <= 0.0:
                raise ValueError("at least one operator weight must be positive")
        focus_sector = arguments.get("focus_sector")
        if focus_sector is not None and not _is_integer(focus_sector):
            raise ValueError("focus_sector must be an integer or null")
        if focus_sector is not None and not 0 <= focus_sector <= 7:
            raise ValueError("focus_sector must lie in [0, 7]")
        exploration_fraction = arguments.get("exploration_fraction")
        if exploration_fraction is not None and not _is_number(exploration_fraction):
            raise ValueError("exploration_fraction must be a finite number")
        if exploration_fraction is not None and not 0.05 <= float(
            exploration_fraction
        ) <= 0.50:
            raise ValueError("exploration_fraction must lie in [0.05, 0.50]")
        for name in ("proposals_per_parent", "beam_width"):
            value = arguments.get(name)
            if value is not None and not _is_integer(value):
                raise ValueError(f"{name} must be an integer")
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        parents = arguments.get("parent_signatures")
        if parents is not None and (
            not isinstance(parents, list)
            or not parents
            or not all(isinstance(item, str) and item for item in parents)
        ):
            raise ValueError("parent_signatures must be a non-empty list of signatures")
        return

    if tool in {"screen_gmc", "audit_projected_mc"}:
        signatures = arguments.get("design_signatures")
        if (
            not isinstance(signatures, list)
            or not signatures
            or not all(isinstance(item, str) and item for item in signatures)
        ):
            raise ValueError("design_signatures must be a non-empty list of signatures")
        return

    if tool == "certify_mc":
        signature = arguments.get("design_signature")
        if not isinstance(signature, str) or not signature:
            raise ValueError("design_signature must be a non-empty string")
        for name in ("histories", "replicates", "nx", "ny"):
            value = arguments.get(name)
            if value is not None and not _is_integer(value):
                raise ValueError(f"{name} must be an integer")
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        n_levels = arguments.get("n_levels")
        if n_levels is not None and not _is_integer(n_levels):
            raise ValueError("n_levels must be an integer or null")
        if n_levels is not None and n_levels < 1:
            raise ValueError("n_levels must be positive")
        flux_exponent = arguments.get("flux_exponent")
        if flux_exponent is not None and not _is_number(flux_exponent):
            raise ValueError("flux_exponent must be a finite number")
        if flux_exponent is not None and float(flux_exponent) <= 0.0:
            raise ValueError("flux_exponent must be positive")
        return

    candidate_signature = arguments.get("candidate_signature")
    if candidate_signature is not None and not isinstance(candidate_signature, str):
        raise ValueError("finish candidate_signature must be a string or null")
    summary = arguments.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("finish summary must be a non-empty string")


def validate_agent_action(payload: dict[str, Any]) -> ShieldAgentAction:
    if not isinstance(payload, dict):
        raise ValueError("agent action must be a JSON object")
    allowed_top_level = {
        "tool",
        "arguments",
        "hypothesis",
        "expected_observation",
        "decision_reason",
    }
    unknown_top_level = set(payload) - allowed_top_level
    if unknown_top_level:
        raise ValueError(f"unknown agent action fields: {sorted(unknown_top_level)}")
    raw_tool = str(payload.get("tool") or "")
    tool = canonical_tool_name(raw_tool)
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("agent action arguments must be an object")
    unknown_arguments = set(arguments) - TOOL_ARGUMENTS[tool]
    if unknown_arguments:
        raise ValueError(
            f"tool {raw_tool!r} received unknown arguments: {sorted(unknown_arguments)}"
        )
    _check_banned_keys(arguments)
    _validate_tool_arguments(tool, arguments)
    hypothesis = str(payload.get("hypothesis") or "").strip()
    expected_observation = str(payload.get("expected_observation") or "").strip()
    decision_reason = str(payload.get("decision_reason") or "").strip()
    if not hypothesis:
        raise ValueError("agent action requires a falsifiable hypothesis")
    if not expected_observation:
        raise ValueError("agent action requires an expected observation")
    if not decision_reason:
        raise ValueError("agent action requires a decision reason")
    if tool == "finish":
        claim = arguments.get("claim")
        if claim not in {
            "verified_improvement",
            "no_verified_improvement",
            "inconclusive",
        }:
            raise ValueError("finish claim is not recognized")
        candidate_signature = arguments.get("candidate_signature")
        if claim == "verified_improvement" and not candidate_signature:
            raise ValueError("verified_improvement requires a candidate signature")
        if claim != "verified_improvement" and candidate_signature is not None:
            raise ValueError(f"{claim} must not name a candidate signature")
    return ShieldAgentAction(
        tool=tool,
        arguments=arguments,
        hypothesis=hypothesis[:320],
        expected_observation=expected_observation[:320],
        decision_reason=decision_reason[:480],
    )


class DeepSeekShieldToolAgent(DeepSeekPolicy):
    """LLM agent that autonomously selects one scientific tool per step."""

    SYSTEM = (
        "You are the autonomous principal investigator for a constrained computational-"
        "materials discovery campaign for neutron shielding. At every turn, choose exactly "
        "one listed scientific tool. Use measured evidence only. You control structure-"
        "generation strategy, parent selection, exploration versus exploitation, GMC "
        "screening allocation, Monte Carlo certification resources, and stopping. Learn "
        "from operator outcomes, use GMC-screened flux fields to breed later generations, "
        "and learn from GMC-to-MC disagreement. You may not type geometry "
        "coordinates, change the source, material inventory, objective, evidence thresholds, "
        "or bypass a fidelity gate. The geometry compiler creates legal structures and the "
        "separate verifier—not you—decides whether a final claim is permitted. Return JSON only."
    )

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        thinking: str = "disabled",
        reasoning_effort: str = "low",
        max_tokens: int = 2048,
        timeout_s: int = 90,
        response_retries: int = 2,
    ) -> None:
        super().__init__(
            model=model,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        if response_retries < 0:
            raise ValueError("response_retries must be non-negative")
        self.response_retries = int(response_retries)
        self.last_traces: list[dict[str, Any]] = []

    @staticmethod
    def _decode_action(content: str) -> ShieldAgentAction:
        try:
            raw_action = json.loads(content)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            raw_action = None
            for start, character in enumerate(content):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(content[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    raw_action = candidate
                    break
            if raw_action is None:
                if not content.strip():
                    raise ShieldAgentResponseError(
                        "shield tool agent did not return a JSON action"
                    )
                raise ShieldAgentResponseError(
                    "shield tool agent returned malformed JSON"
                )
        if not isinstance(raw_action, dict):
            raise ShieldAgentResponseError("shield tool action must be a JSON object")
        try:
            return validate_agent_action(raw_action)
        except ValueError as exc:
            raise ShieldAgentResponseError(str(exc)) from exc

    def decide(
        self,
        observation: dict[str, Any],
        tools: list[dict[str, Any]],
        memory: list[dict[str, Any]],
    ) -> ShieldAgentAction:
        prompt = {
            "scientific_goal": (
                "Autonomously discover a fixed-inventory absorber structure that reduces "
                "source-distant full-field neutron flux and tail hotspots. Use cached GMC "
                "as the primary high-throughput evaluator and reserve unbiased Monte Carlo "
                "for precise certification of a small number of selected designs."
            ),
            "current_observation": observation,
            "recent_scientific_memory": memory,
            "available_tools": tools,
            "instruction": (
                "Return ONLY one JSON object with tool, arguments, hypothesis, "
                "expected_observation, and decision_reason. Select exactly one tool. "
                "Use only candidate signatures present in the observation. Do not "
                "submit absorber coordinates or protected configuration fields. A "
                "finish request can be rejected by the independent verifier. Structure "
                "generation must be repeated from GMC-screened parents until the observation's "
                "GMC-guided discovery gate is satisfied; do not rush from the first proposal "
                "batch to MC. If discovery_mode is gmc_only, do not request or infer any "
                "diffusion-model evidence: use GMC fields, GMC risks, lineage, and operator "
                "memory only. Keep the "
                "three explanatory strings concise so the complete JSON fits comfortably "
                "within the output limit. Projected local-MC is an optional learned-kernel "
                "audit, not physical certification."
            ),
        }
        base_messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        self.last_trace = None
        self.last_traces = []
        last_error: ShieldAgentResponseError | None = None
        for attempt in range(1, self.response_retries + 2):
            recovery_mode = attempt > 1
            messages = list(base_messages)
            if recovery_mode:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous API response could not be parsed as a complete "
                            f"tool action: {last_error}. Retry now with one compact JSON "
                            "object only. Do not include markdown or prose outside JSON."
                        ),
                    }
                )
            request_thinking = "disabled" if recovery_mode else self.thinking
            request_reasoning_effort = (
                "low" if recovery_mode else self.reasoning_effort
            )
            request_max_tokens = (
                max(self.max_tokens, 2048) if recovery_mode else self.max_tokens
            )
            body = json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": request_thinking},
                    "reasoning_effort": request_reasoning_effort,
                    "max_tokens": request_max_tokens,
                    "stream": False,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choice = payload["choices"][0]
            message = choice.get("message") or {}
            content_value = message.get("content")
            content = content_value if isinstance(content_value, str) else ""
            trace = {
                "provider": "DeepSeek",
                "model": self.model,
                "base_url": self.base_url,
                "attempt": attempt,
                "recovery_mode": recovery_mode,
                "thinking": request_thinking,
                "reasoning_effort": request_reasoning_effort,
                "max_tokens": request_max_tokens,
                "messages": messages,
                "assistant_content": content,
                "finish_reason": choice.get("finish_reason"),
                "response_id": payload.get("id"),
                "system_fingerprint": payload.get("system_fingerprint"),
                "usage": payload.get("usage"),
                "secret_recorded": False,
            }
            try:
                action = self._decode_action(content)
                available_tool_names = {
                    canonical_tool_name(str(tool.get("name")))
                    for tool in tools
                    if isinstance(tool, dict) and tool.get("name")
                }
                if action.tool not in available_tool_names:
                    raise ShieldAgentResponseError(
                        f"tool {action.tool!r} is not currently executable"
                    )
            except ShieldAgentResponseError as exc:
                last_error = exc
                trace["accepted"] = False
                trace["response_error"] = str(exc)
                self.last_trace = trace
                self.last_traces.append(trace)
                continue
            trace["accepted"] = True
            self.last_trace = trace
            self.last_traces.append(trace)
            return action
        raise ShieldAgentResponseError(
            f"{last_error} after {self.response_retries + 1} API response attempts"
        )
