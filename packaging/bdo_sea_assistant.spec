from __future__ import annotations

from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ

PROJECT_ROOT = Path(SPECPATH).parent
ASSETS_ROOT = PROJECT_ROOT / "assets"
ASSET_FILES = (
    ASSETS_ROOT / "app_icon.ico",
    ASSETS_ROOT / "minimap" / "minimap_2d_anchor.png",
    ASSETS_ROOT / "minimap" / "manifest.json",
    ASSETS_ROOT / "voice" / "notification_bell.wav",
    *(sorted((ASSETS_ROOT / "voice" / "uk").glob("*.wav"))),
    ASSETS_ROOT / "voice" / "uk" / "manifest.json",
    *(sorted((ASSETS_ROOT / "voice" / "en").glob("*.wav"))),
    ASSETS_ROOT / "voice" / "en" / "manifest.json",
)
for asset in ASSET_FILES:
    if not asset.is_file():
        raise SystemExit(f"Required packaged asset is missing: {asset}")

DATAS = [(str(asset), str(Path("assets") / asset.relative_to(ASSETS_ROOT).parent)) for asset in ASSET_FILES]

a = Analysis(
    [str(PROJECT_ROOT / "run_modern_app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=DATAS,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BDOSeaAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ASSETS_ROOT / "app_icon.ico"),
    version=str(PROJECT_ROOT / "packaging" / "windows-version-info.txt"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BDOSeaAssistant",
)
