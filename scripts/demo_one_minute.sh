#!/usr/bin/env bash
# One-minute LLM demo: DeepSeek drives the frozen GMC environment.
# Model selection: DEEPSEEK_MODEL (env or .env) overrides the config default
# (deepseek-v4-flash). The API key is resolved automatically (environment
# variable -> .env at repo root -> hidden prompt) and is never written to
# disk or logs.
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name
load_api_key

MODEL="${DEEPSEEK_MODEL:-}"
SRC_CONFIG="$ROOT/configs/ray_agent_demo.json"
RUN_CONFIG="$(make_run_config "$SRC_CONFIG" "$MODEL")"
trap 'rm -f "$RUN_CONFIG"' EXIT
EFFECTIVE_MODEL="$MODEL"
if [ -z "$EFFECTIVE_MODEL" ]; then
    EFFECTIVE_MODEL="$(grep -o '"llm_model"[[:space:]]*:[[:space:]]*"[^"]*"' "$SRC_CONFIG" | head -n1 | cut -d'"' -f4)"
fi
echo "[Demo] Model: $EFFECTIVE_MODEL ($([ -n "$MODEL" ] && echo explicit || echo 'config default'))"

cd "$ROOT"
echo "[Demo] DeepSeek -> frozen GMC environment -> verifier -> JSONL"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/run_ray_agent.py --config "$RUN_CONFIG"
