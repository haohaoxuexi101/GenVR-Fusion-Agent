from __future__ import annotations

from dataclasses import replace
from typing import Any
import json
import os
import random
import re
import urllib.request

from .schema import ExperimentAction


def _scientific_feedback(observation: dict[str, Any]) -> dict[str, Any]:
    learning = observation.get("cfm_learning_error", {}).get("shape", {}).get(
        "normalized_rel_l1"
    )
    if learning is None:
        learning = observation.get("cfm_vs_oracle", {}).get("normalized_rel_l1")
    disagreement = observation.get(
        "cfm_directional_disagreement_vs_dense_sn",
        observation.get("cfm_ray_vs_dense", {}),
    ).get("ray_index")
    physical_ratio = observation.get("cfm_vs_global_mc", {}).get("level", {}).get(
        "integral_ratio"
    )
    return {
        "experiment_id": observation["experiment_id"],
        "action": observation.get("action", {}),
        "n_state": observation.get("n_state"),
        "learning_shape_l1": learning,
        "directional_disagreement_vs_dense_sn": disagreement,
        "global_mc_integral_ratio": physical_ratio,
        "controlled_comparisons": observation.get("controlled_comparisons", []),
        "wall_s": observation.get("process_elapsed_s", observation.get("wall_s")),
    }


def decision_text(action: ExperimentAction, history: list[dict[str, Any]]) -> str:
    if not history:
        return "Establish the predeclared baseline before applying one controlled numerical intervention."
    feedback = _scientific_feedback(history[-1])
    learning = feedback["learning_shape_l1"]
    learning_text = "unavailable" if learning is None else f"{learning:.6g}"
    return (
        f"The previous CFM-versus-projected-MC shape error was {learning_text}. Run "
        f"{action.experiment_id} as the next controlled intervention; CFM-versus-S_N directional "
        "disagreement is diagnostic only and must not be minimized as a ray score."
    )


def _action_controls(
    action: ExperimentAction,
    default_dense_n_mu: int | None = None,
    default_dense_n_phi: int | None = None,
) -> dict[str, Any]:
    return {
        "n_pos": action.n_pos,
        "n_mu": action.n_mu,
        "n_phi": action.n_phi,
        "interface_phi_offset_fraction": action.interface_phi_offset_fraction,
        "dense_n_mu": action.dense_n_mu or default_dense_n_mu,
        "dense_n_phi": action.dense_n_phi or default_dense_n_phi,
        "dense_phi_offset_fraction": action.dense_phi_offset_fraction,
        "cfm_samples": action.cfm_samples,
        "oracle_samples": action.oracle_samples,
        "internal_samples": action.internal_samples,
        "oracle_internal_samples": action.oracle_internal_samples,
        "seed": action.seed,
        "cfm_mode": action.cfm_mode,
        "mc_histories": action.mc_histories or 0,
    }


def _observation_controls(observation: dict[str, Any]) -> dict[str, Any]:
    action = ExperimentAction.from_dict(observation["action"])
    controls = _action_controls(action)
    controls["dense_n_mu"] = observation.get("dense_n_mu", controls["dense_n_mu"])
    controls["dense_n_phi"] = observation.get("dense_n_phi", controls["dense_n_phi"])
    return controls


def _single_control_contrast(
    action: ExperimentAction,
    observation: dict[str, Any],
    default_dense_n_mu: int | None = None,
    default_dense_n_phi: int | None = None,
) -> str | None:
    candidate = _action_controls(action, default_dense_n_mu, default_dense_n_phi)
    completed = _observation_controls(observation)
    if candidate["dense_n_mu"] is None:
        candidate["dense_n_mu"] = completed["dense_n_mu"]
    if candidate["dense_n_phi"] is None:
        candidate["dense_n_phi"] = completed["dense_n_phi"]
    differences = []
    for name, candidate_value in candidate.items():
        completed_value = completed[name]
        if isinstance(candidate_value, float) or isinstance(completed_value, float):
            if abs(float(candidate_value or 0.0) - float(completed_value or 0.0)) > 1.0e-12:
                differences.append(name)
        elif candidate_value != completed_value:
            differences.append(name)
    return differences[0] if len(differences) == 1 else None


class CausalPolicy:
    """Deterministic scientific policy over a predeclared causal ablation sequence."""

    def choose(self, candidates: list[ExperimentAction], history: list[dict[str, Any]]) -> tuple[ExperimentAction, str]:
        if not candidates:
            raise StopIteration
        action = candidates[0]
        return action, decision_text(action, history)


class ScientificPolicy:
    """Choose the next action that closes the strongest missing causal contrast."""

    PRIORITY = {
        "seed": 100,
        "dense_phi_offset_fraction": 95,
        "interface_phi_offset_fraction": 90,
        "dense_n_phi": 80,
        "dense_n_mu": 75,
        "n_phi": 70,
        "n_mu": 65,
        "n_pos": 60,
        "cfm_samples": 50,
        "oracle_samples": 50,
        "internal_samples": 45,
        "oracle_internal_samples": 45,
        "mc_histories": 20,
    }

    def __init__(
        self,
        dense_n_mu: int | None = None,
        dense_n_phi: int | None = None,
        learning_shape_l1_limit: float = 0.08,
        rotation_shape_l1_limit: float = 0.03,
        angular_convergence_shape_l1_limit: float = 0.03,
        repeatability_shape_l1_limit: float = 0.03,
        global_mc_integral_rel_limit: float = 0.10,
    ) -> None:
        self.dense_n_mu = dense_n_mu
        self.dense_n_phi = dense_n_phi
        self.learning_shape_l1_limit = learning_shape_l1_limit
        self.rotation_shape_l1_limit = rotation_shape_l1_limit
        self.angular_convergence_shape_l1_limit = angular_convergence_shape_l1_limit
        self.repeatability_shape_l1_limit = repeatability_shape_l1_limit
        self.global_mc_integral_rel_limit = global_mc_integral_rel_limit

    @staticmethod
    def _candidate_signature(action: ExperimentAction) -> tuple[Any, ...]:
        return (
            action.n_pos,
            action.n_mu,
            action.n_phi,
            action.cfm_mode,
            action.cfm_samples,
            action.oracle_samples,
            action.internal_samples,
            action.oracle_internal_samples,
        )

    def _screening_metrics(
        self,
        action: ExperimentAction,
        history: list[dict[str, Any]],
    ) -> dict[str, float | None]:
        signature = self._candidate_signature(action)
        matching = [
            observation
            for observation in history
            if self._candidate_signature(ExperimentAction.from_dict(observation["action"])) == signature
        ]
        baseline = next(
            (
                observation
                for observation in matching
                if float(observation["action"].get("interface_phi_offset_fraction", 0.0)) == 0.0
                and int(observation["action"].get("mc_histories") or 0) == 0
            ),
            None,
        )
        learning = None
        if baseline is not None:
            learning = baseline.get("cfm_learning_error", {}).get("shape", {}).get(
                "normalized_rel_l1"
            )

        experiment_ids = {observation["experiment_id"] for observation in matching}
        comparisons: dict[tuple[str, str, str], dict[str, Any]] = {}
        for observation in history:
            for comparison in observation.get("controlled_comparisons", []):
                key = (
                    comparison["comparison_type"],
                    comparison["baseline_experiment"],
                    comparison["intervention_experiment"],
                )
                comparisons[key] = comparison

        def values(comparison_type: str, both_in_candidate: bool) -> list[float]:
            result = []
            for comparison in comparisons.values():
                if comparison["comparison_type"] != comparison_type:
                    continue
                endpoints = {
                    comparison["baseline_experiment"],
                    comparison["intervention_experiment"],
                }
                if both_in_candidate and not endpoints <= experiment_ids:
                    continue
                if not both_in_candidate and endpoints.isdisjoint(experiment_ids):
                    continue
                metrics = comparison["method_sensitivity"].get("cfm")
                if metrics is not None:
                    result.append(float(metrics["shape"]["normalized_rel_l1"]))
            return result

        rotations = values("interface_rotation", True)
        repeats = values("repeatability", True)
        angular = values("interface_angular_refinement", False)
        return {
            "learning": None if learning is None else float(learning),
            "rotation": max(rotations, default=None),
            "repeatability": max(repeats, default=None),
            "angular_convergence": min(angular, default=None),
        }

    def _certification_rank(
        self,
        action: ExperimentAction,
        history: list[dict[str, Any]],
    ) -> tuple[int, float]:
        metrics = self._screening_metrics(action, history)
        complete = all(value is not None for value in metrics.values())
        score = sum(float(value) for value in metrics.values() if value is not None)
        return (0 if complete else 1, score)

    def should_stop(self, history: list[dict[str, Any]]) -> bool:
        for observation in history:
            action = ExperimentAction.from_dict(observation["action"])
            if (action.mc_histories or 0) <= 0:
                continue
            metrics = self._screening_metrics(action, history)
            if any(value is None for value in metrics.values()):
                continue
            physical_ratio = observation.get("cfm_vs_global_mc", {}).get("level", {}).get(
                "integral_ratio"
            )
            if physical_ratio is None:
                continue
            if (
                float(metrics["learning"]) <= self.learning_shape_l1_limit
                and float(metrics["rotation"]) <= self.rotation_shape_l1_limit
                and float(metrics["angular_convergence"]) <= self.angular_convergence_shape_l1_limit
                and float(metrics["repeatability"]) <= self.repeatability_shape_l1_limit
                and abs(float(physical_ratio) - 1.0) <= self.global_mc_integral_rel_limit
            ):
                return True
        return False

    def choose(
        self,
        candidates: list[ExperimentAction],
        history: list[dict[str, Any]],
    ) -> tuple[ExperimentAction, str]:
        if not candidates:
            raise StopIteration
        ranked = []
        for index, action in enumerate(candidates):
            contrasts = [
                factor
                for observation in history
                if (
                    factor := _single_control_contrast(
                        action,
                        observation,
                        self.dense_n_mu,
                        self.dense_n_phi,
                    )
                )
                is not None
            ]
            priority = max((self.PRIORITY.get(factor, 0) for factor in contrasts), default=-1)
            certification_penalty = 1 if (action.mc_histories or 0) > 0 else 0
            certification_rank = (
                self._certification_rank(action, history)
                if certification_penalty
                else (0, 0.0)
            )
            ranked.append(
                (
                    -priority,
                    certification_penalty,
                    certification_rank[0],
                    certification_rank[1],
                    action.n_pos * action.n_mu * action.n_phi,
                    index,
                    action,
                    contrasts,
                )
            )
        _, _, _, _, _, _, selected, contrasts = min(ranked)
        if contrasts:
            factor = max(contrasts, key=lambda name: self.PRIORITY.get(name, 0))
            reason = (
                f"Select {selected.experiment_id} because it closes a one-factor {factor} contrast. "
                "Selection is based on missing causal evidence, not CFM-versus-S_N disagreement."
            )
        else:
            reason = (
                f"Select {selected.experiment_id} as the lowest-state admissible experiment needed "
                "to open a new controlled comparison."
            )
        return selected, reason


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def choose(self, candidates: list[ExperimentAction], history: list[dict[str, Any]]) -> tuple[ExperimentAction, str]:
        if not candidates:
            raise StopIteration
        action = self.rng.choice(candidates)
        return action, "Random reference policy sampled this admissible action without using environment feedback."


class DeepSeekPolicy:
    """Allowlisted DeepSeek policy; the secret is read only from DEEPSEEK_API_KEY."""

    SYSTEM = (
        "You are a cautious numerical transport scientist performing causal artifact attribution. "
        "Treat CFM-versus-S_N residual directionality only as disagreement between two finite methods, "
        "never as the intrinsic ray effect of either method. Prefer paired rotation, refinement, and "
        "repeatability controls, and do not invent measurements."
    )

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        thinking: str = "disabled",
        reasoning_effort: str = "low",
        max_tokens: int = 384,
        timeout_s: int = 90,
    ) -> None:
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("DEEPSEEK_MODEL", model)
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.last_trace: dict[str, Any] | None = None

    def list_models(self) -> list[str]:
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return sorted(str(item["id"]) for item in payload.get("data", []) if item.get("id"))

    def choose(self, candidates: list[ExperimentAction], history: list[dict[str, Any]]) -> tuple[ExperimentAction, str]:
        public_history = [_scientific_feedback(observation) for observation in history]
        prompt = {
            "research_question": (
                "Which numerical layer causes direction-locked artifacts, and what is the cheapest "
                "configuration that is rotation-robust, angularly converged, and faithful to projected local-MC?"
            ),
            "history": public_history,
            "allowed_actions": [a.to_dict() for a in candidates],
            "instruction": (
                "Return JSON with experiment_id, hypothesis, and decision_reason. Select exactly one "
                "allowed experiment_id. Prefer an action that changes one control variable and closes "
                "a missing rotation, refinement, or repeatability comparison."
            ),
        }
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.thinking},
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
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
            "messages": [
                {"role": "system", "content": "You are a cautious numerical transport scientist. Do not invent measurements. Select one safe predeclared experiment."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "assistant_content": content,
            "response_id": payload.get("id"),
            "system_fingerprint": payload.get("system_fingerprint"),
            "usage": payload.get("usage"),
            "secret_recorded": False,
        }
        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise RuntimeError("DeepSeek did not return a JSON decision")
            decision = json.loads(match.group(0))
        by_id = {a.experiment_id: a for a in candidates}
        selected = str(decision.get("experiment_id", ""))
        if selected not in by_id:
            raise RuntimeError(f"DeepSeek selected an action outside the allowlist: {selected!r}")
        action = replace(by_id[selected], hypothesis=str(decision.get("hypothesis") or by_id[selected].hypothesis))
        return action, str(decision.get("decision_reason") or "DeepSeek selected this predeclared causal probe.")
