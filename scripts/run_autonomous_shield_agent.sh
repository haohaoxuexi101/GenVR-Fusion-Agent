#!/usr/bin/env bash
# Real DeepSeek tool-agent entry point. No deterministic policy fallback exists.
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name
load_api_key

cd "$ROOT"
echo "[AutoShield] GMC-guided structure discovery -> unbiased MC verifier"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/25_run_autonomous_shield_agent.py \
    --config "${AUTONOMOUS_SHIELD_CONFIG:-configs/autonomous_shield_agent.json}" \
    "$@"
