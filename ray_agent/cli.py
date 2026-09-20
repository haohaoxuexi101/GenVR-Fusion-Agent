from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import random
import shutil
import sys

from .environment import RayEffectEnvironment
from .policies import CausalPolicy, DeepSeekPolicy, RandomPolicy, ScientificPolicy
from .reporting import finalize_run
from .schema import AgentConfig, ExperimentAction


def main() -> None:
    parser = argparse.ArgumentParser(description="GenVR-Fusion autonomous rare-event transport exploration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--policy", choices=["causal", "scientific", "deepseek"], default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = AgentConfig.load(config_path)
    if args.policy and args.policy != cfg.policy:
        data = cfg.to_dict()
        data["policy"] = args.policy
        data["actions"] = tuple(ExperimentAction.from_dict(x) for x in data["actions"])
        cfg = AgentConfig(**data)
        cfg.validate()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{cfg.run_name}"
    run_dir = Path(args.run_dir).resolve() if args.run_dir else root / "submission" / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")

    env = RayEffectEnvironment(root, cfg, run_dir, run_id)
    shutil.copy2(config_path, run_dir / "config.json")
    env.write_manifest(config_path)
    if cfg.policy == "deepseek":
        policy = DeepSeekPolicy(
            model=cfg.llm_model,
            thinking=cfg.llm_thinking,
            reasoning_effort=cfg.llm_reasoning_effort,
            max_tokens=cfg.llm_max_tokens,
            timeout_s=cfg.llm_timeout_s,
        )
    elif cfg.policy == "scientific":
        policy = ScientificPolicy(
            dense_n_mu=cfg.dense_n_mu,
            dense_n_phi=cfg.dense_n_phi,
            learning_shape_l1_limit=cfg.learning_shape_l1_limit,
            rotation_shape_l1_limit=cfg.rotation_shape_l1_limit,
            angular_convergence_shape_l1_limit=cfg.angular_convergence_shape_l1_limit,
            repeatability_shape_l1_limit=cfg.repeatability_shape_l1_limit,
            global_mc_integral_rel_limit=cfg.global_mc_integral_rel_limit,
        )
    else:
        policy = CausalPolicy()
    remaining = list(cfg.actions)
    observations = []
    while remaining and len(observations) < cfg.max_causal_trials:
        try:
            if not observations:
                action = remaining[0]
                reason = "Run the frozen coarse baseline before exposing environment feedback to the LLM."
                decision_policy = "BootstrapPolicy"
            else:
                action, reason = policy.choose(remaining, observations)
                decision_policy = policy.__class__.__name__
        except Exception as exc:
            if cfg.policy != "deepseek":
                raise
            retry = "stop_required_llm_run" if cfg.llm_required else "fallback_to_causal"
            env.logger.write("error", {"stage": "deepseek_policy", "message": str(exc), "retry": retry})
            if cfg.llm_required:
                raise
            policy = CausalPolicy()
            action, reason = policy.choose(remaining, observations)
            decision_policy = "CausalPolicyFallback"
        if isinstance(policy, DeepSeekPolicy) and policy.last_trace is not None:
            env.logger.write("llm_exchange", policy.last_trace)
        env.logger.write("decision", {"policy": decision_policy, "selected": action.experiment_id, "reason": reason})
        print(f"[GenVR-Fusion] decision={decision_policy} action={action.experiment_id}", flush=True)
        obs = env.run_experiment(action, role="causal_agent", history=observations)
        learning_error = obs.get("cfm_learning_error", {}).get("shape", {}).get(
            "normalized_rel_l1"
        )
        disagreement = obs["cfm_directional_disagreement_vs_dense_sn"]["ray_index"]
        print(
            f"[GenVR-Fusion] feedback action={action.experiment_id} "
            f"learning_l1={learning_error if learning_error is not None else 'n/a'} "
            f"sn_disagreement={disagreement:.6g} "
            f"controlled_pairs={len(obs['controlled_comparisons'])}",
            flush=True,
        )
        observations.append(obs)
        remaining = [x for x in remaining if x.experiment_id != action.experiment_id]
        if isinstance(policy, ScientificPolicy) and policy.should_stop(observations):
            env.logger.write(
                "stopping_rule",
                {
                    "reason": "A candidate passed learning, rotation, angular-convergence, repeatability, and global-MC gates.",
                    "experiment_id": action.experiment_id,
                },
            )
            break

    random_observations = []
    random_policy = RandomPolicy(seed=cfg.actions[0].seed + 991)
    pool = list(cfg.actions[1:])
    for index in range(min(cfg.random_baseline_trials, len(pool))):
        selected, reason = random_policy.choose(pool, random_observations)
        action = ExperimentAction(
            **{**selected.to_dict(), "experiment_id": f"random_{index + 1}_{selected.experiment_id}", "seed": selected.seed + 50000 + index}
        )
        env.logger.write("decision", {"policy": "RandomPolicy", "selected": action.experiment_id, "reason": reason})
        random_observations.append(
            env.run_experiment(
                action,
                role="random_reference",
                history=observations + random_observations,
            )
        )
        pool = [x for x in pool if x.experiment_id != selected.experiment_id]

    summary = finalize_run(run_dir, cfg.case, observations, random_observations, config=cfg)
    env.logger.write("run_completed", {
        "scientific_claim": summary["scientific_claim"],
        "summary_json": "summary.json",
        "summary_markdown": "summary.md",
        "ablation_figure": "ablation_summary.png",
    })
    print(json.dumps({"run_dir": str(run_dir), "scientific_claim": summary["scientific_claim"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
