#!/usr/bin/env bash
# GMC-guided material-layout discovery with unbiased MC certification.
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name
load_api_key

cd "$ROOT"
echo "[GMC-Discovery] GMC-only structure discovery -> GMC weight windows -> high-statistics MC certification"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/25_run_autonomous_shield_agent.py \
    --config "${GMC_DISCOVERY_CONFIG:-configs/gmc_material_discovery_agent.json}" \
    "$@"
