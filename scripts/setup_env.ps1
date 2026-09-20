param(
    [string]$CondaExe = "",
    [string]$EnvName = ""
)
# One-step environment setup: detects conda, picks the CUDA or CPU spec
# automatically, then creates/updates the environment and runs a self-check.
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\conda_env.ps1"

$conda = Find-CondaExe -Explicit $CondaExe
$envName = Get-RayEnvName -Explicit $EnvName
$root = (Resolve-Path "$PSScriptRoot\..").Path

$useCpu = -not (Test-NvidiaGpu)
$envFile = if ($useCpu) { Join-Path $root "environment_cpu.yml" } else { Join-Path $root "environment.yml" }
Write-Host "[setup] conda: $conda"
Write-Host "[setup] GPU detected: $(-not $useCpu); spec: $(Split-Path $envFile -Leaf); environment: $envName"

$envList = (& $conda env list --json 2>$null | Out-String | ConvertFrom-Json)
$exists = @($envList.envs | Where-Object { (Split-Path $_ -Leaf) -eq $envName }).Count -gt 0
if ($exists) {
    Write-Host "[setup] Environment '$envName' exists; updating..."
    & $conda env update -n $envName -f $envFile --prune
} else {
    Write-Host "[setup] Creating environment '$envName'..."
    & $conda env create -n $envName -f $envFile
}
if ($LASTEXITCODE -ne 0) { throw "conda environment create/update failed." }

Push-Location $root
try {
    & $conda run -n $envName python scripts/00_check_environment.py
    if ($LASTEXITCODE -ne 0) { throw "Environment self-check failed." }
} finally {
    Pop-Location
}
Write-Host "[setup] Done."
Write-Host "[setup] Next: .\scripts\smoke_test.ps1 (offline, no API key) or .\scripts\demo_one_minute.ps1 (LLM demo)."
