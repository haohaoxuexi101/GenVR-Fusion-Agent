#!/usr/bin/env python3
"""Open-exploration driver.

Unlike run_ray_agent.py (predeclared causal ablation, where the LLM only
orders a fixed action set), this driver lets the LLM freely propose
(n_pos, n_mu, n_phi) configurations inside bounds declared in the config's
"open_search" section. Proposals are validated before execution; invalid or
duplicate proposals are fed back as rejection notes. Results, proposals and
API exchanges are all recorded in the run directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import random
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ray_agent.environment import RayEffectEnvironment
from ray_agent.open_policy import OpenProposer
from ray_agent.reporting import finalize_run
from ray_agent.schema import AgentConfig, ExperimentAction


def load_open_config(path: Path) -> tuple[AgentConfig, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    open_search = data.pop("open_search", None)
    if not open_search:
        raise ValueError("config must contain an 'open_search' bounds section")
    data["actions"] = tuple(ExperimentAction.from_dict(x) for x in data["actions"])
    # Deliberately no AgentConfig.validate(): open mode is not the predeclared
    # ablation, and the validation constraint max_causal_trials<=len(actions)
    # does not apply to free proposals.
    return AgentConfig(**data), open_search


def salvage_proposal(text: str, bounds: dict[str, Any]) -> dict[str, Any] | None:
    """Recover n_pos/n_mu/n_phi from a truncated or malformed LLM response."""
    def grab(name: str) -> int | None:
        match = re.search(r'["\']?' + name + r'["\']?\s*[:=]\s*(\d+)', text)
        return int(match.group(1)) if match else None
    n_pos, n_mu, n_phi = grab("n_pos"), grab("n_mu"), grab("n_phi")
    if n_pos is None or n_mu is None or n_phi is None:
        return None
    return {"n_pos": n_pos, "n_mu": n_mu, "n_phi": n_phi, "hypothesis": "", "decision_reason": ""}


def validate_proposal(proposal: dict[str, Any], bounds: dict[str, Any], seen: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int] | None, str]:
    try:
        n_pos = int(proposal["n_pos"])
        n_mu = int(proposal["n_mu"])
        n_phi = int(proposal["n_phi"])
    except (KeyError, TypeError, ValueError):
        return None, "proposal must contain integer n_pos, n_mu, n_phi"
    errors: list[str] = []
    if n_pos not in bounds["n_pos_options"]:
        errors.append(f"n_pos={n_pos} not in {bounds['n_pos_options']}")
    if n_mu not in bounds["n_mu_options"]:
        errors.append(f"n_mu={n_mu} not in {bounds['n_mu_options']}")
    if n_phi not in bounds["n_phi_options"]:
        errors.append(f"n_phi={n_phi} not in {bounds['n_phi_options']}")
    states = n_pos * n_mu * n_phi
    if states > bounds["max_total_states"]:
        errors.append(f"state budget {states} exceeds max_total_states={bounds['max_total_states']}")
    if (n_pos, n_mu, n_phi) in seen:
        errors.append(f"configuration (n_pos={n_pos}, n_mu={n_mu}, n_phi={n_phi}) was already run")
    if errors:
        return None, "; ".join(errors)
    return (n_pos, n_mu, n_phi), ""


def make_action(prefix: str, n_pos: int, n_mu: int, n_phi: int, bounds: dict[str, Any], hypothesis: str, seed: int) -> ExperimentAction:
    return ExperimentAction(
        experiment_id=f"{prefix}_p{n_pos}_m{n_mu}_f{n_phi}",
        hypothesis=hypothesis,
        n_pos=n_pos,
        n_mu=n_mu,
        n_phi=n_phi,
        cfm_samples=bounds["cfm_samples"],
        oracle_samples=bounds["oracle_samples"],
        internal_samples=bounds["internal_samples"],
        oracle_internal_samples=bounds["oracle_internal_samples"],
        seed=seed,
        cfm_mode="direct",
    )


def finalize_open(
    run_dir: Path,
    case: str,
    observations: list[dict[str, Any]],
    random_observations: list[dict[str, Any]],
    bounds: dict[str, Any],
    config: AgentConfig,
) -> dict[str, Any]:
    summary = finalize_run(
        run_dir,
        case,
        observations,
        random_observations,
        config=config,
    )
    summary["open_search_bounds"] = bounds
    summary["open_search_role"] = (
        "hypothesis_generation_only; a configuration is not certified unless matched rotation "
        "controls and the declared acceptance gates are present"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recommendation = summary.get("recommendation")
    if recommendation is None:
        text = "No open-search proposal is scientifically certified without matched rotation controls."
    else:
        text = (
            f"Certified candidate: {recommendation['candidate_id']} with configuration "
            f"{recommendation['configuration']}."
        )
    (run_dir / "recommendation.md").write_text(
        f"# Open exploration recommendation\n\n{text}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GenVR-Fusion open phase-space exploration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg, bounds = load_open_config(config_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{cfg.run_name}"
    run_dir = Path(args.run_dir).resolve() if args.run_dir else ROOT / "submission" / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")

    env = RayEffectEnvironment(ROOT, cfg, run_dir, run_id)
    shutil.copy2(config_path, run_dir / "config.json")
    env.write_manifest(config_path)

    baseline = cfg.actions[0]
    print(f"[OpenExplore] bootstrap={baseline.experiment_id}", flush=True)
    observations = [env.run_experiment(baseline, role="open_agent", history=[])]

    proposer = OpenProposer(
        model=cfg.llm_model,
        thinking=cfg.llm_thinking,
        reasoning_effort=cfg.llm_reasoning_effort,
        max_tokens=cfg.llm_max_tokens,
        timeout_s=cfg.llm_timeout_s,
    )
    seen: list[tuple[int, int, int]] = [(baseline.n_pos, baseline.n_mu, baseline.n_phi)]
    proposal_index = 1
    while len(observations) < cfg.max_causal_trials:
        rejection = ""
        params = None
        proposal: dict[str, Any] = {}
        for _ in range(3):
            try:
                proposal = proposer.propose(bounds, observations, rejection)
            except Exception as exc:
                if getattr(proposer, "last_trace", None):
                    env.logger.write("llm_exchange", proposer.last_trace)
                salvaged = salvage_proposal(getattr(proposer, "last_raw", ""), bounds)
                params, error = (validate_proposal(salvaged, bounds, seen) if salvaged else (None, ""))
                if params:
                    proposal = salvaged
                    env.logger.write("environment_feedback", {"type": "proposal_salvaged", "message": f"JSON parse failed ({exc}); recovered numeric fields by regex."})
                    break
                rejection = (
                    "Your previous response was truncated or not valid JSON. Return ONLY a JSON object "
                    "with keys n_pos, n_mu, n_phi, hypothesis, decision_reason. Keep hypothesis under 40 "
                    "words and decision_reason under 80 words. Do not include markdown or trailing text."
                )
                env.logger.write("environment_feedback", {"type": "proposal_rejected", "message": f"unparseable proposal: {exc}"})
                continue
            env.logger.write("llm_exchange", proposer.last_trace)
            params, error = validate_proposal(proposal, bounds, seen)
            if not error:
                break
            rejection = error
            env.logger.write("environment_feedback", {"type": "proposal_rejected", "message": error})
        if params is None:
            print("[OpenExplore] no valid proposal after retries; stopping.", flush=True)
            break
        n_pos, n_mu, n_phi = params
        action = make_action("proposal", n_pos, n_mu, n_phi, bounds, str(proposal.get("hypothesis") or ""), bounds["seed"])
        proposal_index += 1
        seen.append(params)
        env.logger.write("decision", {
            "policy": "OpenProposer",
            "selected": action.experiment_id,
            "hypothesis": action.hypothesis,
            "reason": str(proposal.get("decision_reason") or ""),
        })
        print(f"[OpenExplore] decision=OpenProposer action={action.experiment_id}", flush=True)
        observations.append(env.run_experiment(action, role="open_agent", history=observations))

    rng = random.Random(bounds["seed"] + 991)
    random_observations: list[dict[str, Any]] = []
    for index in range(cfg.random_baseline_trials):
        n_pos = rng.choice(bounds["n_pos_options"])
        n_mu = rng.choice(bounds["n_mu_options"])
        n_phi = rng.choice(bounds["n_phi_options"])
        action = make_action(f"random_{index + 1}", n_pos, n_mu, n_phi, bounds, "Random reference configuration.", bounds["seed"] + 50000 + index)
        env.logger.write("decision", {"policy": "RandomPolicy", "selected": action.experiment_id, "reason": "Random reference policy sampled an admissible configuration without environment feedback."})
        print(f"[OpenExplore] random_reference action={action.experiment_id}", flush=True)
        random_observations.append(
            env.run_experiment(
                action,
                role="random_reference",
                history=observations + random_observations,
            )
        )

    summary = finalize_open(run_dir, cfg.case, observations, random_observations, bounds, cfg)
    env.logger.write("run_completed", {
        "scientific_claim": summary["scientific_claim"],
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
        "ablation_figure": "ablation_summary.png",
    })
    print(json.dumps({"run_dir": str(run_dir), "scientific_claim": summary["scientific_claim"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
