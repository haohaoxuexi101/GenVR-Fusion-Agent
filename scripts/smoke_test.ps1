param(
    [string]$CondaExe = "",
    [string]$EnvName = ""
)
# Offline smoke test: causal policy, small grid, no DeepSeek API key required.
# Validates the closed loop, JSONL logging and artifact layout only.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\conda_env.ps1"

$conda = Find-CondaExe -Explicit $CondaExe
$envName = Get-RayEnvName -Explicit $EnvName
$root = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "[Smoke] Offline causal-policy smoke run (no API key required)."
Push-Location $root
try {
    & $conda run -n $envName python scripts/run_ray_agent.py --config configs/ray_agent_smoke.json
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed." }
} finally {
    Pop-Location
}
