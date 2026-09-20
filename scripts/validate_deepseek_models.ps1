param(
    [string]$CondaExe = "",
    [string]$EnvName = "",
    [string]$Models = ""
)
# Validates the DeepSeek model matrix against the live API. Model selection:
# -Models "m1,m2" (default: deepseek-v4-flash,deepseek-v4-pro). The API key
# is resolved automatically (environment variable -> .env at repo root ->
# hidden prompt).
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\conda_env.ps1"

$conda = Find-CondaExe -Explicit $CondaExe
$envName = Get-RayEnvName -Explicit $EnvName
$root = (Resolve-Path "$PSScriptRoot\..").Path
$resolved = Resolve-ApiKey
if (-not $Models) {
    Import-RootEnv
    if (-not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_MODELS)) { $Models = $env:DEEPSEEK_MODELS }
}
# Normalize: an unquoted comma-separated parameter may arrive as a string array.
$Models = (@($Models) -join ',')
$modelsArg = @()
if ($Models) {
    foreach ($item in ($Models -split ',')) {
        $name = $item.Trim()
        if ($name) { $modelsArg += @('--models', $name) }
    }
}

try {
    Push-Location $root
    try {
        & $conda run -n $envName python scripts/validate_deepseek_models.py @modelsArg
        if ($LASTEXITCODE -ne 0) { throw "DeepSeek model validation failed." }
    } finally {
        Pop-Location
    }
} finally {
    if ($resolved.Ephemeral) { Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue }
}
