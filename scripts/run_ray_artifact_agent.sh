#!/usr/bin/env bash
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name

POLICY="${RAY_ARTIFACT_POLICY:-scientific}"
if [ "$POLICY" = "deepseek" ]; then
    load_api_key
fi

cd "$ROOT"
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python scripts/run_ray_agent.py \
    --config "${RAY_ARTIFACT_CONFIG:-configs/ray_artifact_agent.json}" \
    --policy "$POLICY" \
    "$@"
