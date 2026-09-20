param(
    [string]$CondaExe = "",
    [string]$EnvName = "",
    [string]$Model = "",
    [string]$Config = "configs/ray_agent_open_explore.json"
)
# Open exploration: the LLM freely proposes (n_pos, n_mu, n_phi) inside the
# bounds declared in the config's open_search section. Model selection:
# -Model <name> > DEEPSEEK_MODEL (env/.env) > config default.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\conda_env.ps1"

$conda = Find-CondaExe -Explicit $CondaExe
$envName = Get-RayEnvName -Explicit $EnvName
$root = (Resolve-Path "$PSScriptRoot\..").Path
$resolved = Resolve-ApiKey
$model = Resolve-Model -Explicit $Model
$sourceConfig = (Resolve-Path (Join-Path $root $Config)).Path
$runConfig = New-PatchedConfig -Root $root -Source $sourceConfig -Model $model

try {
    $effectiveModel = if ($model) { $model } else { Get-ConfigModel -Source $sourceConfig }
    Write-Host ("[OpenExplore] Model: " + $effectiveModel + $(if ($model) { " (explicit)" } else { " (config default)" }))
    Push-Location $root
    try {
        & $conda run -n $envName python scripts/run_open_exploration.py --config $runConfig
        if ($LASTEXITCODE -ne 0) { throw "Open exploration failed." }
    } finally {
        Pop-Location
    }
} finally {
    if ($resolved.Ephemeral) { Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue }
    if ($runConfig -and ($runConfig -ne $sourceConfig)) {
        Remove-Item -LiteralPath $runConfig -Force -ErrorAction SilentlyContinue
    }
}
