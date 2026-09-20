param(
    [string]$CondaExe = "",
    [string]$EnvName = "",
    [string]$Model = "",
    [string]$Config = "configs/autonomous_shield_agent.json"
)
# Real DeepSeek tool-agent entry point. No deterministic policy fallback exists.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\conda_env.ps1"

$conda = Find-CondaExe -Explicit $CondaExe
$envName = Get-RayEnvName -Explicit $EnvName
$root = (Resolve-Path "$PSScriptRoot\..").Path
$resolved = Resolve-ApiKey
$model = Resolve-Model -Explicit $Model
$previousModel = $env:DEEPSEEK_MODEL

try {
    if ($model) { $env:DEEPSEEK_MODEL = $model }
    $arguments = @("scripts/25_run_autonomous_shield_agent.py", "--config", $Config)
    if ($model) { $arguments += @("--model", $model) }
    Write-Host "[AutoShield] GMC-guided structure discovery -> unbiased MC verifier"
    Push-Location $root
    try {
        & $conda run -n $envName python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Autonomous shielding agent failed." }
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $previousModel) {
        Remove-Item Env:DEEPSEEK_MODEL -ErrorAction SilentlyContinue
    } else {
        $env:DEEPSEEK_MODEL = $previousModel
    }
    if ($resolved.Ephemeral) {
        Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    }
}
