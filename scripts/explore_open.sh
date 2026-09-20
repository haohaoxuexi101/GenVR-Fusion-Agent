#!/usr/bin/env bash
# Open exploration: the LLM freely proposes (n_pos, n_mu, n_phi) inside the
# bounds declared in the config's open_search section. Model selection:
# DEEPSEEK_MODEL (env or .env) overrides the config default.
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

CONFIG="${EXPLORE_CONFIG:-$ROOT/configs/ray_agent_open_explore.json}"

find_conda
resolve_env_name
load_api_key

MODEL="${DEEPSEEK_MODEL:-}"
RUN_CONFIG="$(make_run_config "$CONFIG" "$MODEL")"
trap 'rm -f "$RUN_CONFIG"' EXIT
EFFECTIVE_MODEL="$MODEL"
if [ -z "$EFFECTIVE_MODEL" ]; then
    EFFECTIVE_MODEL="$(grep -o '"llm_model"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG" | head -n1 | cut -d'"' -f4)"
fi
echo "[OpenExplore] Model: $EFFECTIVE_MODEL ($([ -n "$MODEL" ] && echo explicit || echo 'config default'))"

cd "$ROOT"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/run_open_exploration.py --config "$RUN_CONFIG"
