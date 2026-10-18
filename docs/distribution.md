# BDO Sea Assistant — Windows distribution

## Supported build target

- Windows 11 x64.
- Per-user installation; no administrator rights, service, driver, scheduled task, auto-start, updater, game input, memory access or injection.
- The recipient calibrates the app for their own BDO UI geometry. Their settings and recordings live under `%LOCALAPPDATA%\BDOSeaAssistant` and are not included in the package.

## Build

Run from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

The script runs the source suite, builds a PyInstaller onedir bundle in `release\dist\BDOSeaAssistant`, validates required assets, and produces `release\BDOSeaAssistant-0.2.4-win-x64-portable.zip`. It only replaces `release\build`, `release\dist`, and that ZIP; it never deletes source, captures, recordings, settings, or AppData.

To build the installer after a successful bundle, install Inno Setup 6 and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

This creates `release\installer\BDOSeaAssistant-0.2.4-win-x64-Setup.exe`. The installer is per-user and uninstall leaves `%LOCALAPPDATA%\BDOSeaAssistant` intact.

## Tester instructions

1. Install the Setup EXE or unpack the portable ZIP completely.
2. Start `BDOSeaAssistant.exe`; do not move its `_internal` directory.
3. Open BDO, select its window, and complete your own calibration.
4. Start monitoring only when ready. The game, capture, and voice test do not start automatically.
5. To update, close the app and run the newer installer. To remove it, uninstall; settings, calibration, logs, and recordings are retained.

## Public release channel

A stable tag matching `vMAJOR.MINOR.PATCH` triggers the reviewed Windows release workflow. The tagged commit must be contained in `main`, and every release-facing version field must match the tag.

After tests, portable and installer smoke checks, privacy scanning, and uninstall-preservation verification, GitHub Releases receives exactly:

- `BDOSeaAssistant-0.2.4-win-x64-portable.zip`;
- `BDOSeaAssistant-0.2.4-win-x64-Setup.exe`;
- `SHA256SUMS.txt`.

The release is created as a draft first. Uploaded assets are downloaded and compared byte-for-byte before the workflow publishes the release. Ordinary branch pushes and pull requests cannot publish releases.

## Current release limits

Public alpha binaries are unsigned until Authenticode signing is added. SHA-256 verifies file integrity but is not a substitute for a cryptographic publisher signature. The in-app updater currently checks the latest published release and opens its GitHub page; it does not silently download or install anything.

This is not a claim of Windows 10 support or anti-cheat approval. Public packages contain only the reviewed application assets described in [ASSET_NOTICES.md](../ASSET_NOTICES.md); local settings, calibration, captures, recordings, logs, and credentials are excluded.
