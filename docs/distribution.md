# BDO Sea Assistant — private test distribution

## Supported build target

- Windows 11 x64.
- Per-user installation; no administrator rights, service, driver, scheduled task, auto-start, updater, game input, memory access or injection.
- The recipient calibrates the app for their own BDO UI geometry. Their settings and recordings live under `%LOCALAPPDATA%\BDOSeaAssistant` and are not included in the package.

## Build

Run from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

The script runs the source suite, builds a PyInstaller onedir bundle in `release\dist\BDOSeaAssistant`, validates required assets, and produces `release\BDOSeaAssistant-0.1.0-win-x64-portable.zip`. It only replaces `release\build`, `release\dist`, and that ZIP; it never deletes source, captures, recordings, settings, or AppData.

To build the installer after a successful bundle, install Inno Setup 6 and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

This creates `release\installer\BDOSeaAssistant-0.1.0-win-x64-Setup.exe`. The installer is per-user and uninstall leaves `%LOCALAPPDATA%\BDOSeaAssistant` intact.

## Tester instructions

1. Install the Setup EXE or unpack the portable ZIP completely.
2. Start `BDOSeaAssistant.exe`; do not move its `_internal` directory.
3. Open BDO, select its window, and complete your own calibration.
4. Start monitoring only when ready. The game, capture, and voice test do not start automatically.
5. To update, close the app and run the newer installer. To remove it, uninstall; settings, calibration, logs, and recordings are retained.

## Release limits

This is an unsigned private-test build until separately code-signed. It is not a claim of Windows 10 support, anti-cheat approval, or public redistributability of bundled assets. Before sharing outside the approved private test circle, verify asset/licence rights, test a clean Windows machine and a physical BDO machine, then generate checksums after any signing step.
