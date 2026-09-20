from __future__ import annotations

from typing import Any
import json
import re
import urllib.request

from .policies import DeepSeekPolicy


class OpenProposer(DeepSeekPolicy):
    """LLM policy that freely proposes (n_pos, n_mu, n_phi) within declared bounds.

    Unlike the allowlisted DeepSeekPolicy used by the official ablation, this
    policy proposes new discretizations. Proposals are validated against the
    declared bounds by the driver before any environment execution.
    """

    SYSTEM = (
        "You are a cautious numerical transport scientist exploring a GMC response environment. "
        "You may propose new phase-space discretizations, but only within the declared bounds. "
        "Treat CFM-versus-S_N directional disagreement as a diagnostic, not an intrinsic ray score. "
        "Use projected local-MC to assess learned-kernel fidelity and do not invent measurements."
    )

    def propose(
        self,
        bounds: dict[str, Any],
        history: list[dict[str, Any]],
        rejection_note: str = "",
    ) -> dict[str, Any]:
        public_history = []
        for item in history:
            action = item.get("action", {})
            public_history.append(
                {
                    "experiment_id": item["experiment_id"],
                    "n_pos": action.get("n_pos"),
                    "n_mu": action.get("n_mu"),
                    "n_phi": action.get("n_phi"),
                    "n_state": item.get("n_state"),
                    "learning_shape_l1": item.get("cfm_learning_error", {})
                    .get("shape", {})
                    .get("normalized_rel_l1"),
                    "directional_disagreement_vs_dense_sn": item.get(
                        "cfm_directional_disagreement_vs_dense_sn",
                        item.get("cfm_ray_vs_dense", {}),
                    ).get("ray_index"),
                    "controlled_comparisons": item.get("controlled_comparisons", []),
                    "wall_s": item.get("wall_s"),
                }
            )
        prompt = {
            "research_question": (
                "Which interface phase-space discretization most improves learned-kernel fidelity "
                "per unit cost and should next receive a matched rotation control?"
            ),
            "declared_bounds": bounds,
            "history": public_history,
            "rejection_note": rejection_note or None,
            "instruction": (
                "Return ONLY a JSON object with keys n_pos, n_mu, n_phi, hypothesis, decision_reason. "
                "n_pos must be in n_pos_options, n_mu in n_mu_options, n_phi in n_phi_options, "
                "and n_pos*n_mu*n_phi must not exceed max_total_states. "
                "Keep hypothesis under 40 words and decision_reason under 80 words. "
                "Do not include markdown, code fences, or trailing text. "
                "Propose the experiment you expect to be most informative given the history; do "
                "not optimize the CFM-versus-S_N directional disagreement."
            ),
        }
        body = json.dumps(
            {
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
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        self.last_raw = content
        self.last_trace = {
            "provider": "DeepSeek",
            "model": self.model,
            "base_url": self.base_url,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "assistant_content": content,
            "response_id": payload.get("id"),
            "system_fingerprint": payload.get("system_fingerprint"),
            "usage": payload.get("usage"),
            "secret_recorded": False,
        }
        try:
            proposal = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise RuntimeError("DeepSeek did not return a JSON proposal")
            proposal = json.loads(match.group(0))
        return proposal
