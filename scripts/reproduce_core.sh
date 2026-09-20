#!/usr/bin/env bash
# Reproduces the official DeepSeek-driven core run. Results are written to a
# new UTC-timestamped directory under submission/runs/, so existing evidence
# is never overwritten. Model selection: DEEPSEEK_MODEL (env or .env)
# overrides the config default (deepseek-v4-flash). The API key is resolved
# automatically (environment variable -> .env at repo root -> hidden prompt).
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name
load_api_key

MODEL="${DEEPSEEK_MODEL:-}"
SRC_CONFIG="$ROOT/configs/ray_agent_core_llm.json"
RUN_CONFIG="$(make_run_config "$SRC_CONFIG" "$MODEL")"
trap 'rm -f "$RUN_CONFIG"' EXIT
EFFECTIVE_MODEL="$MODEL"
if [ -z "$EFFECTIVE_MODEL" ]; then
    EFFECTIVE_MODEL="$(grep -o '"llm_model"[[:space:]]*:[[:space:]]*"[^"]*"' "$SRC_CONFIG" | head -n1 | cut -d'"' -f4)"
fi
echo "[Reproduce] Model: $EFFECTIVE_MODEL ($([ -n "$MODEL" ] && echo explicit || echo 'config default'))"

cd "$ROOT"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/run_ray_agent.py --config "$RUN_CONFIG"
