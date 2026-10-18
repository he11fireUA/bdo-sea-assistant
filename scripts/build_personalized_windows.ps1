param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^[A-Za-z0-9_-]{1,64}$')]
  [string]$Recipient,
  [ValidatePattern('^[A-Za-z0-9_-]{0,96}$')]
  [string]$BuildId = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IdentityPath = Join-Path $ProjectRoot 'bdo_assistant\build_identity.py'
$ReleaseZip = Join-Path $ProjectRoot 'release\BDOSeaAssistant-0.2.3-win-x64-portable.zip'
if (-not $BuildId) { $BuildId = "bsa-alpha-$Recipient-$([guid]::NewGuid().ToString('N').Substring(0, 12))" }
$Destination = Join-Path $env:USERPROFILE "Downloads\BDO-Sea-Assistant-0.2.3-alpha-$Recipient-$BuildId-portable.zip"
$OriginalBytes = [System.IO.File]::ReadAllBytes($IdentityPath)

# This file is compiled into the PyInstaller bytecode, then the generic source
# is restored even if packaging fails. The recipient token is deliberately not
# stored in AppData or settings, where a recipient could trivially replace it.
$PersonalizedIdentity = @"
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildIdentity:
    """Immutable identity compiled into a frozen distribution."""

    recipient: str
    build_id: str
    window_title: str
    credit_line: str | None = None


DEFAULT_BUILD_IDENTITY = BuildIdentity(
    recipient="$Recipient",
    build_id="$BuildId",
    window_title="BDO Sea Assistant - Alpha for $Recipient test version",
    credit_line="Coded by he11fireUA",
)
"@

try {
  [System.IO.File]::WriteAllText($IdentityPath, $PersonalizedIdentity, [System.Text.UTF8Encoding]::new($false))
  & (Join-Path $PSScriptRoot 'build_windows.ps1')
  Copy-Item -LiteralPath $ReleaseZip -Destination $Destination -Force
  Get-FileHash -Algorithm SHA256 -LiteralPath $Destination | Format-Table -AutoSize
  Write-Host "Personalized recipient: $Recipient"
  Write-Host "Embedded build ID: $BuildId"
  Write-Host "Portable ZIP: $Destination"
}
finally {
  [System.IO.File]::WriteAllBytes($IdentityPath, $OriginalBytes)
}
