#!/usr/bin/env bash
# Offline smoke test: causal policy, small grid, no DeepSeek API key required.
# Validates the closed loop, JSONL logging and artifact layout only.
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name

cd "$ROOT"
echo "[Smoke] Offline causal-policy smoke run (no API key required)."
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/run_ray_agent.py --config configs/ray_agent_smoke.json
