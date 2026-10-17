$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$WorkPath = Join-Path $ReleaseRoot 'build'
$DistPath = Join-Path $ReleaseRoot 'dist'
$Bundle = Join-Path $DistPath 'BDOSeaAssistant'
$ZipPath = Join-Path $ReleaseRoot 'BDOSeaAssistant-0.2.1-win-x64-portable.zip'

Set-Location $ProjectRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = '.'
$env:QT_QPA_PLATFORM = 'offscreen'
uv sync --frozen --group build
if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE." }
uv run --no-sync pytest -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw "Test suite failed with exit code $LASTEXITCODE." }

# Only release output owned by this script may be replaced; source, AppData and captures are never touched.
if (Test-Path $WorkPath) { Remove-Item -LiteralPath $WorkPath -Recurse -Force }
if (Test-Path $DistPath) { Remove-Item -LiteralPath $DistPath -Recurse -Force }
if (Test-Path $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }

uv run --no-sync python -m PyInstaller --noconfirm --clean --workpath $WorkPath --distpath $DistPath packaging/bdo_sea_assistant.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }
if (-not (Test-Path (Join-Path $Bundle 'BDOSeaAssistant.exe'))) { throw 'PyInstaller did not produce BDOSeaAssistant.exe.' }

$requiredAssets = @(
  '_internal\assets\app_icon.ico',
  '_internal\assets\minimap\minimap_2d_anchor.png',
  '_internal\assets\minimap\manifest.json',
  '_internal\assets\voice\notification_bell.wav',
  '_internal\assets\voice\uk\manifest.json',
  '_internal\assets\voice\en\manifest.json'
)
foreach ($asset in $requiredAssets) {
  if (-not (Test-Path (Join-Path $Bundle $asset))) { throw "Bundled asset missing: $asset" }
}

Compress-Archive -Path $Bundle -DestinationPath $ZipPath -CompressionLevel Optimal
Get-FileHash -Algorithm SHA256 (Join-Path $Bundle 'BDOSeaAssistant.exe'), $ZipPath | Format-Table -AutoSize
Write-Host "Bundle: $Bundle"
Write-Host "Portable ZIP: $ZipPath"
