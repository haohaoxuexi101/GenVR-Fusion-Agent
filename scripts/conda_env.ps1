# Shared helpers for GenVR-Fusion entry-point PowerShell scripts.
# Dot-source this file from the scripts directory:
#   . "$PSScriptRoot\conda_env.ps1"

function Find-CondaExe {
    <#
    Resolve the conda executable in this order:
      1. explicit path passed by the caller (-CondaExe)
      2. 'conda' found on PATH
      3. common install locations
    #>
    param([string]$Explicit)
    if ($Explicit) {
        if (Test-Path -LiteralPath $Explicit) { return (Resolve-Path -LiteralPath $Explicit).Path }
        throw "Conda executable not found at: $Explicit"
    }
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    $candidates = @(
        "$env:ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "G:\ProgramData\anaconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    ) | Select-Object -Unique
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    throw "conda not found. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html), rerun scripts/setup_env.ps1, or pass -CondaExe."
}

function Get-RayEnvName {
    <#
    Environment name resolution:
      1. explicit parameter (-EnvName)
      2. the 'name' field of environment.yml
      3. fallback 'ssn'
    #>
    param([string]$Explicit)
    if ($Explicit) { return $Explicit }
    $yml = Join-Path $PSScriptRoot "..\environment.yml"
    if (Test-Path -LiteralPath $yml) {
        $match = Select-String -Path $yml -Pattern '^\s*name:\s*(\S+)' | Select-Object -First 1
        if ($match -and $match.Matches.Count -gt 0) { return $match.Matches[0].Groups[1].Value }
    }
    return "ssn"
}

function Test-NvidiaGpu {
    <# True when an NVIDIA GPU with a working driver is visible. #>
    $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvsmi) { return $false }
    & $nvsmi.Source -L *> $null
    return ($LASTEXITCODE -eq 0)
}

function Import-RootEnv {
    <# Load KEY=VALUE lines from the repository-root .env into the process
       environment (idempotent; values are never printed). #>
    $root = (Resolve-Path "$PSScriptRoot\..").Path
    $envPath = Join-Path $root ".env"
    if (Test-Path -LiteralPath $envPath) {
        foreach ($line in Get-Content -LiteralPath $envPath) {
            if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*?)\s*$') {
                Set-Item -Path ("Env:" + $Matches[1]) -Value ($Matches[2].Trim('"').Trim("'"))
            }
        }
    }
}

function Resolve-ApiKey {
    <#
    Resolve DEEPSEEK_API_KEY in this order:
      1. process environment variable
      2. .env file at the repository root (auto-loaded; gitignored)
      3. interactive hidden prompt (not echoed, never written to disk)
    Returns @{ Key = ...; Ephemeral = $bool }. Ephemeral=true means this
    process acquired the key from the prompt; callers should remove the
    environment variable afterwards.
    #>
    if (-not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
        return @{ Key = $env:DEEPSEEK_API_KEY; Ephemeral = $false }
    }
    Import-RootEnv
    if (-not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
        return @{ Key = $env:DEEPSEEK_API_KEY; Ephemeral = $false }
    }
    $secure = Read-Host "DeepSeek API key (input is hidden)" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
        throw "DEEPSEEK_API_KEY is required. Set the environment variable, create a .env file at the repository root (DEEPSEEK_API_KEY=sk-...), or answer the interactive prompt."
    }
    return @{ Key = $env:DEEPSEEK_API_KEY; Ephemeral = $true }
}

function Resolve-Model {
    <#
    Model selection for LLM-driven runs:
      1. explicit parameter (-Model)
      2. DEEPSEEK_MODEL (environment variable or .env)
      3. '' -> the config's own default (deepseek-v4-flash)
    The official results were produced with deepseek-v4-flash; choosing
    another model creates an independent run whose config.json and
    trajectory record the chosen model verbatim.
    #>
    param([string]$Explicit)
    if ($Explicit) { return $Explicit }
    Import-RootEnv
    if (-not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_MODEL)) { return $env:DEEPSEEK_MODEL }
    return ""
}

function Get-ConfigModel {
    <# Read the llm_model field of a config file (the effective default). #>
    param([string]$Source)
    return (Get-Content -LiteralPath $Source -Raw | ConvertFrom-Json).llm_model
}

function New-PatchedConfig {
    <#
    Return a runnable config path. When Model is non-empty, a temporary copy
    (under tmp/, gitignored) overrides llm_model; otherwise the original
    config is returned untouched. Callers should delete the temporary file
    when it differs from the source.
    #>
    param([string]$Root, [string]$Source, [string]$Model)
    if (-not $Model) { return $Source }
    $cfg = Get-Content -LiteralPath $Source -Raw | ConvertFrom-Json
    $cfg.llm_model = $Model
    # Tag the run directory with the model so different-model runs never mix.
    $suffix = ($Model -replace '[^A-Za-z0-9_.-]', '_')
    $cfg.run_name = ([string]$cfg.run_name) + "_" + $suffix
    $tmpDir = Join-Path $Root "tmp"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
    $out = Join-Path $tmpDir ("ray_agent_" + [Guid]::NewGuid().ToString("N") + ".json")
    $json = $cfg | ConvertTo-Json -Depth 20
    # UTF-8 without BOM: AgentConfig.load rejects a BOM header.
    [System.IO.File]::WriteAllText($out, $json, (New-Object System.Text.UTF8Encoding($false)))
    return $out
}
