# Contributing to BDO Sea Assistant

Thank you for helping improve the project. BDO Sea Assistant is a Windows-only, read-only screen assistant for Black Desert Online.

## Development environment

- Windows 11 x64
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)

Install the locked development environment from PowerShell:

```powershell
uv sync --frozen --group dev
```

Run the required test suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "."
$env:QT_QPA_PLATFORM = "offscreen"
uv run --no-sync pytest -q -p no:cacheprovider
```

## Pull requests

1. Open an issue for large behavior or architecture changes before implementation.
2. Work in a focused branch and keep each pull request limited to one concern.
3. Add deterministic tests before or alongside behavior changes.
4. Update documentation when behavior, configuration, packaging, or user-visible text changes.
5. Keep detection logic out of the Qt UI layer.
6. If dependencies change, commit both `pyproject.toml` and `uv.lock`.
7. Ensure the complete suite passes on Python 3.11 and 3.12.

Tests must not require a running BDO client, physical capture hardware, a real game window, an audio device, a visible Qt window, administrator rights, tray interaction, or network access. Mock capture, audio, popup, tray, and Win32 boundaries instead of skipping coverage.

## Privacy and test evidence

Never commit or attach:

- raw captures or journeys;
- private chat, player names, guild names, or personal information;
- local settings, calibration, logs, AppData, `.env`, tokens, or keys;
- generated `release/` output;
- images or audio without redistribution rights.

Image fixtures must be the smallest useful crop, scrubbed of identifying information, and suitable for redistribution. Use synthetic fixtures whenever possible.

Security reports and private evidence must follow [SECURITY.md](SECURITY.md), not public issues.

## Licence

By submitting a contribution, you agree that it will be licensed under the GNU General Public License v3.0 only (`GPL-3.0-only`).
