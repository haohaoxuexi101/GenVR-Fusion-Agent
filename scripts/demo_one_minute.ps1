param(
    [string]$CondaExe = "",
    [string]$EnvName = "",
    [string]$Model = ""
)
# One-minute LLM demo: DeepSeek drives the frozen GMC environment.
# Model selection: -Model <name> > DEEPSEEK_MODEL (env/.env) > config default
# (deepseek-v4-flash). The API key is resolved automatically (environment
# variable -> .env at repo root -> hidden prompt) and is never written to
# disk or logs.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\conda_env.ps1"

$conda = Find-CondaExe -Explicit $CondaExe
$envName = Get-RayEnvName -Explicit $EnvName
$root = (Resolve-Path "$PSScriptRoot\..").Path
$resolved = Resolve-ApiKey
$model = Resolve-Model -Explicit $Model
$sourceConfig = Join-Path $root "configs\ray_agent_demo.json"
$runConfig = New-PatchedConfig -Root $root -Source $sourceConfig -Model $model

try {
    $effectiveModel = if ($model) { $model } else { Get-ConfigModel -Source $sourceConfig }
    Write-Host ("[Demo] Model: " + $effectiveModel + $(if ($model) { " (explicit)" } else { " (config default)" }))
    Write-Host "[Demo] DeepSeek -> frozen GMC environment -> verifier -> JSONL"
    Push-Location $root
    try {
        & $conda run -n $envName python scripts/run_ray_agent.py --config $runConfig
        if ($LASTEXITCODE -ne 0) { throw "One-minute demo failed." }
    } finally {
        Pop-Location
    }
} finally {
    if ($resolved.Ephemeral) { Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue }
    if ($runConfig -and ($runConfig -ne (Join-Path $root "configs\ray_agent_demo.json"))) {
        Remove-Item -LiteralPath $runConfig -Force -ErrorAction SilentlyContinue
    }
}
