from __future__ import annotations

from typing import Any
import json
import math
import re
import urllib.request

from .policies import DeepSeekPolicy
from .shield_design import OPERATORS, ShieldStrategy


class EvidenceDrivenShieldStrategist:
    """Deterministic scientist policy used with or without an external LLM."""

    def advise(self, context: dict[str, Any]) -> ShieldStrategy:
        round_index = max(int(context.get("round", 1)), 1)
        memory = context.get("operator_memory", {})
        counts = memory.get("counts", {})
        improvements = memory.get("mean_improvement", {})
        total_trials = sum(max(int(counts.get(name, 0)), 0) for name in OPERATORS)
        priors = {
            "adjoint_sensitivity": 1.0,
            "channel_block": 0.85,
            "inner_barrier": 0.70,
            "hotspot_cap": 0.75,
            "symmetry_repair": 0.65,
            "redistribute": 0.45,
            "random_explore": 0.25,
        }
        raw_scores = {}
        for operator in OPERATORS:
            count = max(int(counts.get(operator, 0)), 0)
            empirical = max(float(improvements.get(operator, 0.0)), 0.0)
            uncertainty = math.sqrt(math.log(total_trials + 2.0) / (count + 1.0))
            raw_scores[operator] = priors[operator] * (0.35 + empirical + 0.20 * uncertainty)
        scale = max(max(raw_scores.values()), 1.0e-12)
        weights = {name: min(value / scale, 1.0) for name, value in raw_scores.items()}

        beam = context.get("beam", [])
        best = min(beam, key=lambda item: float(item.get("risk", math.inf))) if beam else {}
        diagnosis = best.get("physics_diagnosis", {})
        hottest_sector = diagnosis.get("hottest_sector")
        focus_sector = int(hottest_sector) if hottest_sector is not None else None
        recent = context.get("recent_outcomes", [])
        recent_improvements = [float(item.get("improvement", 0.0)) for item in recent]
        improving = [value for value in recent_improvements if value > 0.0]
        stagnating = bool(recent_improvements) and not improving
        exploration_fraction = min(0.45, 0.16 + 0.04 * math.sqrt(round_index))
        if stagnating:
            exploration_fraction = min(0.50, exploration_fraction + 0.15)
        elif improving:
            exploration_fraction = max(0.08, exploration_fraction - 0.05)

        additions = diagnosis.get("highest_sensitivity_empty_cells", [])
        target = additions[0].get("cell") if additions else None
        hypothesis = (
            f"Far-field tail sensitivity is concentrated in sector {focus_sector}; "
            f"test fixed-mass moves toward cell {target}."
            if focus_sector is not None
            else "Use forward-adjoint sensitivity while retaining exploratory swaps."
        )
        return ShieldStrategy(
            operator_weights=weights,
            focus_sector=focus_sector,
            exploration_fraction=exploration_fraction,
            hypothesis=hypothesis,
            decision_reason=(
                "Operator weights combine physics priors, measured mean improvement, "
                "and an uncertainty bonus; exploration rises when recent trials stall."
            ),
        )


class ShieldLLMStrategist(DeepSeekPolicy):
    """LLM strategist whose output is compiled into safe geometry mutations.

    The model cannot directly submit a material map or change the fixed scientific
    objective. It may only reweight predeclared mutation operators, select one of
    eight angular sectors, and adjust the exploration fraction.
    """

    SYSTEM = (
        "You are a neutron-shielding design scientist supervising a constrained "
        "multi-fidelity optimizer. Use forward flux, distributed-adjoint sensitivity, "
        "angular leakage, uncertainty, and measured operator outcomes to diagnose "
        "transport mechanisms. You cannot change the material budget, source, "
        "objective, or directly output geometry. Prefer falsifiable one-round "
        "hypotheses and return only a safe search strategy."
    )

    def advise(self, context: dict[str, Any]) -> ShieldStrategy:
        prompt = {
            "scientific_goal": (
                "Reduce source-distant far-field neutron flux, especially the "
                "worst 10 percent of cells, at fixed absorber mass."
            ),
            "immutable_objective": context.get("fixed_objective"),
            "allowed_operators": list(OPERATORS),
            "measured_context": context,
            "instruction": (
                "Return ONLY JSON with operator_weights, focus_sector, "
                "exploration_fraction, hypothesis, decision_reason. "
                "operator_weights must contain only allowed operators and values "
                "in [0,1]. focus_sector must be null or an integer 0..7. "
                "exploration_fraction must lie in [0.05,0.50]. Do not invent flux "
                "values and do not propose a geometry."
            ),
        }
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "thinking": {"type": self.thinking},
                "reasoning_effort": self.reasoning_effort,
                "max_tokens": self.max_tokens,
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
        content = payload["choices"][0]["message"]["content"]
        self.last_trace = {
            "provider": "DeepSeek",
            "model": self.model,
            "base_url": self.base_url,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
            "messages": messages,
            "assistant_content": content,
            "response_id": payload.get("id"),
            "system_fingerprint": payload.get("system_fingerprint"),
            "usage": payload.get("usage"),
            "secret_recorded": False,
        }
        try:
            advice = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise RuntimeError("shield strategist did not return JSON")
            advice = json.loads(match.group(0))
        return validate_shield_strategy(advice)


def validate_shield_strategy(payload: dict[str, Any]) -> ShieldStrategy:
    raw_weights = payload.get("operator_weights", {})
    if not isinstance(raw_weights, dict):
        raise ValueError("operator_weights must be an object")
    unknown = set(raw_weights) - set(OPERATORS)
    if unknown:
        raise ValueError(f"unknown shield operators: {sorted(unknown)}")
    weights = {name: float(raw_weights.get(name, 0.0)) for name in OPERATORS}
    if any(not 0.0 <= value <= 1.0 for value in weights.values()):
        raise ValueError("operator weights must lie in [0, 1]")
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one operator weight must be positive")

    raw_sector = payload.get("focus_sector")
    focus_sector = None if raw_sector is None else int(raw_sector)
    if focus_sector is not None and not 0 <= focus_sector <= 7:
        raise ValueError("focus_sector must be null or an integer in [0, 7]")
    exploration_fraction = float(payload.get("exploration_fraction", 0.20))
    if not 0.05 <= exploration_fraction <= 0.50:
        raise ValueError("exploration_fraction must lie in [0.05, 0.50]")
    hypothesis = str(payload.get("hypothesis") or "LLM-adjusted shielding search strategy.")
    decision_reason = str(
        payload.get("decision_reason")
        or "Validated LLM strategy restricted to compiled search controls."
    )
    return ShieldStrategy(
        operator_weights=weights,
        focus_sector=focus_sector,
        exploration_fraction=exploration_fraction,
        hypothesis=hypothesis[:240],
        decision_reason=decision_reason[:320],
    )
