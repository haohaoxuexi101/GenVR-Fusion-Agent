from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ray_agent.policies import DeepSeekPolicy
from ray_agent.schema import AgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DeepSeek model availability and allowlisted feedback decisions.")
    parser.add_argument("--models", action="append", default=None,
                        help="model name; repeat the flag to validate several (default: deepseek-v4-flash and deepseek-v4-pro)")
    parser.add_argument("--output", default="submission/api_validation/deepseek_model_matrix.json")
    args = parser.parse_args()
    models = args.models or ["deepseek-v4-flash", "deepseek-v4-pro"]
    models = [item.strip() for item in models if item.strip()]
    cfg = AgentConfig.load(ROOT / "configs" / "ray_agent_core_llm.json")
    baseline = json.loads(
        (ROOT / "submission" / "runs" / "official_final" / "experiments" / "coarse" / "diagnostics.json").read_text(encoding="utf-8")
    )
    candidates = list(cfg.actions[1:])
    result = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "key_source": "DEEPSEEK_API_KEY environment variable",
        "secret_recorded": False,
        "input_feedback": {
            "experiment_id": baseline["experiment_id"],
            "n_state": baseline["n_state"],
            "ray_index": baseline["cfm_ray_vs_dense"]["ray_index"],
            "rel_l1": baseline["cfm_vs_dense"]["normalized_rel_l1"],
            "wall_s": baseline["wall_s"],
        },
        "models": [],
    }
    probe = DeepSeekPolicy(model=models[0], thinking="disabled", reasoning_effort="low", max_tokens=384, timeout_s=90)
    available = probe.list_models()
    result["available_models"] = available
    for model in models:
        if model not in available:
            raise RuntimeError(f"required model not returned by /models: {model}")
        policy = DeepSeekPolicy(model=model, thinking="disabled", reasoning_effort="low", max_tokens=384, timeout_s=90)
        start = time.perf_counter()
        action, reason = policy.choose(candidates, [baseline])
        elapsed = time.perf_counter() - start
        trace = policy.last_trace or {}
        result["models"].append({
            "model": model,
            "status": "PASS",
            "latency_s": elapsed,
            "selected_experiment_id": action.experiment_id,
            "selected_from_allowlist": action.experiment_id in {x.experiment_id for x in candidates},
            "decision_reason": reason,
            "usage": trace.get("usage"),
            "response_id": trace.get("response_id"),
            "system_fingerprint": trace.get("system_fingerprint"),
        })
    result["cross_model_same_action"] = len({item["selected_experiment_id"] for item in result["models"]}) == 1
    result["status"] = "PASS"
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "available_models": available,
        "decisions": [
            {"model": item["model"], "selected": item["selected_experiment_id"], "latency_s": round(item["latency_s"], 3)}
            for item in result["models"]
        ],
        "output": str(output),
        "secret_recorded": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
