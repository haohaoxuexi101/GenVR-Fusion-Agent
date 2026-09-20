#!/usr/bin/env bash
# Validates the DeepSeek model matrix against the live API. Model selection:
# DEEPSEEK_MODELS="m1,m2" (default: deepseek-v4-flash,deepseek-v4-pro). The
# API key is resolved automatically (environment variable -> .env at repo
# root -> hidden prompt).
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name
load_api_key

MODELS="${DEEPSEEK_MODELS:-}"
ARGS=()
if [ -n "$MODELS" ]; then
    IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
    for model in "${MODEL_LIST[@]}"; do
        [ -n "$model" ] && ARGS+=(--models "$model")
    done
fi

cd "$ROOT"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/validate_deepseek_models.py "${ARGS[@]}"
