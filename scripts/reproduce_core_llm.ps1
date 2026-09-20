param(
    [string]$CondaExe = "",
    [string]$EnvName = "",
    [string]$Model = ""
)
# Reproduces the official DeepSeek-driven core run. Results are written to a
# new UTC-timestamped directory under submission/runs/, so existing evidence
# is never overwritten. Model selection: -Model <name> > DEEPSEEK_MODEL
# (env/.env) > config default (deepseek-v4-flash). The API key is resolved
# automatically (environment variable -> .env at repo root -> hidden prompt).
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\conda_env.ps1"

$conda = Find-CondaExe -Explicit $CondaExe
$envName = Get-RayEnvName -Explicit $EnvName
$root = (Resolve-Path "$PSScriptRoot\..").Path
$resolved = Resolve-ApiKey
$model = Resolve-Model -Explicit $Model
$sourceConfig = Join-Path $root "configs\ray_agent_core_llm.json"
$runConfig = New-PatchedConfig -Root $root -Source $sourceConfig -Model $model

try {
    $effectiveModel = if ($model) { $model } else { Get-ConfigModel -Source $sourceConfig }
    Write-Host ("[Reproduce] Model: " + $effectiveModel + $(if ($model) { " (explicit)" } else { " (config default)" }))
    Push-Location $root
    try {
        & $conda run -n $envName python scripts/run_ray_agent.py --config $runConfig
        if ($LASTEXITCODE -ne 0) { throw "Official LLM exploration failed." }
    } finally {
        Pop-Location
    }
} finally {
    if ($resolved.Ephemeral) { Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue }
    if ($runConfig -and ($runConfig -ne $sourceConfig)) {
        Remove-Item -LiteralPath $runConfig -Force -ErrorAction SilentlyContinue
    }
}
