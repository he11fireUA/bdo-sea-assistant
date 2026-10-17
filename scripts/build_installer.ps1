$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Candidates = @(
    @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
)
if ($Candidates.Count -eq 0) {
    throw 'Inno Setup 6 is required. Install it, then rerun this script.'
}

& $Candidates[0] (Join-Path $ProjectRoot 'packaging\installer.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE." }
$Installer = Join-Path $ProjectRoot 'release\installer\BDOSeaAssistant-0.2.0-win-x64-Setup.exe'
if (-not (Test-Path $Installer)) { throw 'Inno Setup did not produce the installer.' }
Get-FileHash -Algorithm SHA256 $Installer | Format-Table -AutoSize
Write-Host "Installer: $Installer"
