#!/usr/bin/env bash
# One-step environment setup: detects conda, picks the CUDA or CPU spec
# automatically, then creates/updates the environment and runs a self-check.
. "$(cd "$(dirname "${BASH_SOURCE[0]:-$PWD/scripts}")" && pwd)/common.sh"

find_conda
resolve_env_name

if has_nvidia_gpu; then
    ENV_FILE="$ROOT/environment.yml"
else
    ENV_FILE="$ROOT/environment_cpu.yml"
fi
echo "[setup] conda: $CONDA_BIN"
echo "[setup] environment: $ENV_NAME; spec: $(basename "$ENV_FILE")"

if "$CONDA_BIN" run -n "$ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
    echo "[setup] Environment '$ENV_NAME' exists; updating..."
    "$CONDA_BIN" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "[setup] Creating environment '$ENV_NAME'..."
    "$CONDA_BIN" env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

cd "$ROOT"
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/00_check_environment.py
echo "[setup] Done."
echo "[setup] Next: bash scripts/smoke_test.sh (offline, no API key) or bash scripts/reproduce_core.sh (LLM)."
