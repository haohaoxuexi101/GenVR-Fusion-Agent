param(
    [string]$CondaExe = "",
    [string]$EnvName = "",
    [string]$Model = ""
)
$ErrorActionPreference = "Stop"
& "$PSScriptRoot\reproduce_core_llm.ps1" -CondaExe $CondaExe -EnvName $EnvName -Model $Model
