# BDO Sea Assistant

[![CI](https://github.com/he11fireUA/bdo-sea-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/he11fireUA/bdo-sea-assistant/actions/workflows/ci.yml)

A read-only Windows screen assistant for Black Desert Online.

The source code is licensed under the GNU General Public License v3.0 only.
See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.
Asset provenance and third-party notices are documented in [ASSET_NOTICES.md](ASSET_NOTICES.md).

The project captures the BDO window through Windows Graphics Capture and analyzes visible UI pixels. It does not send input, read game memory, inject code, hook DirectX, or modify game files.

## Working baseline

The first user-accepted working version is preserved at **`first-working-v1`**
(commit `7457770`). See [snapshot scope](docs/first-working-version.md).

The current desktop entry point is `run_modern_app.py` (PySide6). Its production
navigation path uses direct minimap texture change plus route length, confirmed
for five seconds. Local voice/popup alerts and the in-app recorder are available;
Telegram and distribution/update work below remain roadmap items, not shipped
features. Legacy replay/diagnostic modules must not be mistaken for the active
detector.

Run automated tests without starting capture or audio:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -p no:cacheprovider
```

## Product goal

The final deliverable is a polished Windows desktop application with:

- a modern user interface;
- a system-tray icon and background monitoring controls;
- UI calibration through **Find UI Positions**;
- independent Telegram, local voice, and desktop popup notification channels for arrival, likely grounding/stuck states, ship HP below 60%, and provisions below 10%;
- Ukrainian and English application/notification localization, including bundled offline voice packs;
- an installer if distribution and update requirements justify one.

## Development order

### Phase 1 — Functional core (current)

- deterministic and replayable tests;
- Windows Graphics Capture validation;
- automatic minimap and ship-bar calibration;
- HP and provisions measurement;
- minimap-based movement/arrival/stuck classification;
- debounce, hysteresis, cooldown, and channel-independent notification routing;
- configuration schema and failure handling.

### Phase 2 — Application shell

- modern desktop UI;
- calibration preview with Confirm, Retry, and Manual correction;
- settings and diagnostic views;
- Ukrainian and English UI localization;
- independent Telegram/voice/popup toggles, voice language, volume, playback-device selection, and test controls;
- tray menu, start/stop monitoring, and status indicators;
- startup behavior and clean shutdown.

### Phase 3 — Distribution

- packaged executable;
- installer only if needed;
- verified application updates delivered through tagged GitHub Releases;
- upgrades, configuration migration, logs, and uninstallation verification.

The UI layer must call the tested core rather than contain detection logic itself. This keeps calibration and monitoring usable from automated tests, replay tools, CLI diagnostics, and the final application.
