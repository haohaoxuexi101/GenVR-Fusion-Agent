#!/usr/bin/env bash
# Shared helpers for GenVR-Fusion entry-point bash scripts.
# Source it from other scripts in this directory:
#   . "$(dirname "${BASH_SOURCE[0]}")/common.sh"

set -euo pipefail

# Repository root, resolved from this file's location when possible, with a
# cwd fallback for exotic invocations where BASH_SOURCE may be empty.
_SOURCE="${BASH_SOURCE[0]:-}"
if [ -n "$_SOURCE" ] && [ "$_SOURCE" != "${_SOURCE##*/}" ]; then
    ROOT="$(cd "$(dirname "$_SOURCE")/.." && pwd)"
else
    ROOT="$(pwd)"
fi

# Overridable: CONDA_EXE points to a conda/mamba/micromamba executable,
# RAY_AGENT_ENV overrides the environment name.
CONDA_BIN=""
ENV_NAME=""

find_conda() {
    if [ -n "${CONDA_EXE:-}" ]; then
        if [ -x "$CONDA_EXE" ]; then
            CONDA_BIN="$CONDA_EXE"
            return 0
        fi
        echo "CONDA_EXE is set but not executable: $CONDA_EXE" >&2
        return 1
    fi
    local candidate
    for candidate in conda micromamba mamba; do
        if command -v "$candidate" >/dev/null 2>&1; then
            CONDA_BIN="$(command -v "$candidate")"
            return 0
        fi
    done
    for candidate in \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/usr/local/miniconda3/bin/conda"; do
        if [ -x "$candidate" ]; then
            CONDA_BIN="$candidate"
            return 0
        fi
    done
    echo "conda not found. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html), rerun scripts/setup_env.sh, or set CONDA_EXE." >&2
    return 1
}

resolve_env_name() {
    if [ -n "${RAY_AGENT_ENV:-}" ]; then
        ENV_NAME="$RAY_AGENT_ENV"
        return 0
    fi
    local name
    name="$(sed -n 's/^[[:space:]]*name:[[:space:]]*//p' "$ROOT/environment.yml" | head -n 1)"
    ENV_NAME="${name:-ssn}"
}

has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

# Resolve DEEPSEEK_API_KEY: environment variable -> .env at repo root ->
# interactive hidden prompt. Exports the key when resolved.
load_api_key() {
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
        export DEEPSEEK_API_KEY
        return 0
    fi
    if [ -f "$ROOT/.env" ]; then
        set -a
        # shellcheck disable=SC1091
        . "$ROOT/.env"
        set +a
    fi
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
        export DEEPSEEK_API_KEY
        return 0
    fi
    if [ -t 0 ]; then
        printf 'DeepSeek API key (input is hidden): ' >&2
        IFS= read -r -s DEEPSEEK_API_KEY || true
        printf '\n' >&2
    fi
    if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
        echo "DEEPSEEK_API_KEY is required. Set the DEEPSEEK_API_KEY environment variable, create a .env file at the repository root (DEEPSEEK_API_KEY=sk-...), or run interactively to answer the hidden prompt." >&2
        return 1
    fi
    export DEEPSEEK_API_KEY
    return 0
}

# Return a runnable config path on stdout. When a model override is given, a
# temporary copy (under tmp/, gitignored) sets llm_model; otherwise the
# original config path is echoed unchanged. Callers should delete the
# temporary copy when it differs from the source. The official results were
# produced with deepseek-v4-flash; other models create independent runs whose
# config.json and trajectory record the chosen model verbatim.
make_run_config() {
    local src="$1" model="${2:-}" out
    if [ -z "$model" ]; then
        printf '%s\n' "$src"
        return 0
    fi
    mkdir -p "$ROOT/tmp"
    out="$(mktemp "$ROOT/tmp/ray_agent_cfg.XXXXXX.json")"
    # Also tag run_name with the model so different-model runs never mix.
    "$CONDA_BIN" run -n "$ENV_NAME" python -c "import json,sys,re;c=json.load(open(sys.argv[1],encoding='utf-8'));c['llm_model']=sys.argv[2];c['run_name']=str(c['run_name'])+'_'+re.sub(r'[^A-Za-z0-9_.-]','_',sys.argv[2]);json.dump(c,open(sys.argv[3],'w',encoding='utf-8'),ensure_ascii=False,indent=2)" "$src" "$model" "$out"
    printf '%s\n' "$out"
}
