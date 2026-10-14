# Product Backlog

## Notification channels and localization

**Priority:** After the functional detectors and event-state logic are stable.

### User value

Users must be able to receive important sea-assistant alerts without creating or configuring a Telegram bot. Telegram, local voice, and desktop popup alerts can be enabled independently or in any parallel combination.

### Functional requirements

- Supported channels:
  - Telegram;
  - local voice playback;
  - desktop popup notifications.
- Channel combinations:
  - every channel can be enabled or disabled independently;
  - any enabled channels receive the same event in parallel;
  - if all channels are disabled, show an explicit UI warning that no notification channel is active.
- Initial locales:
  - Ukrainian (`uk`);
  - English (`en`).
- The application UI, notification text, and voice assets use localization keys rather than hardcoded sentences.
- Bundle two offline voice packs, one per supported locale. Normal operation must not depend on cloud TTS or internet access.
- Voice settings:
  - enable/disable;
  - language;
  - volume;
  - playback device where supported;
  - **Test voice notification** button.
- Popup settings:
  - enable/disable;
  - display/monitor selection where supported;
  - screen position, duration, and severity behavior;
  - **Test popup notification** button;
  - must not steal keyboard focus, minimize BDO, or generate game input.
- Telegram settings:
  - enable/disable;
  - credentials and target validation;
  - **Test Telegram notification** button;
  - setup remains optional.
- A failure in one channel must not suppress delivery through another enabled channel.
- Shared event debounce/cooldown prevents each channel from repeating the same alert independently.

### Initial localized events

- arrival at destination;
- likely stuck/grounded;
- ship HP below configured threshold;
- provisions below configured threshold;
- capture lost or game window unavailable;
- minimap visibility lost/unusable (`MINIMAP_VISIBILITY_LOST`);
- minimap visibility restored (`MINIMAP_VISIBILITY_RESTORED`);
- UI calibration lost/invalid.

Dynamic values such as destination and measured percentage should be inserted through locale-aware templates.

### Minimap visibility and observability loss

The minimap is a key navigation sensor. A healthy WGC stream does not mean navigation remains observable: opening the global/world map, hiding the minimap, covering it with another game UI panel, changing minimap mode/scale, or moving the UI outside the confirmed calibration can make minimap motion and route evidence unavailable.

- Detect minimap availability independently from full-window capture health and from generic calibration validity.
- Require the confirmed minimap frame/anchor geometry plus enough visible map texture/route-layer evidence; a static overlay inside the ROI must not count as a valid minimap merely because pixels are present.
- Debounce loss across several consecutive frames or a short time window so momentary UI animations do not alert.
- On confirmed loss, enter an explicit navigation observability state such as `MINIMAP_UNAVAILABLE`/`OBSERVABILITY_LOST`.
- While unavailable:
  - suspend `ARRIVED`, `STUCK`, navigation-motion, route-completion, and related persistence timers;
  - do not interpret the global map or another static overlay as stationarity;
  - keep independent detectors such as HP/provisions running if their own ROIs remain valid;
  - expose degraded monitoring status in the UI/tray.
- Emit one `MINIMAP_VISIBILITY_LOST` transition event through the shared `NotificationRouter`; do not repeat it continuously while the global map remains open.
- Localized user guidance should say that the assistant cannot see the minimap and suggest closing the global map or restoring/checking the game UI layout.
- Automatically recover only after the minimap geometry and content remain valid for several consecutive frames; emit one `MINIMAP_VISIBILITY_RESTORED` event and restart navigation persistence windows from a clean state.
- If geometry does not recover, offer **Find UI positions** / recalibration rather than silently saving new coordinates.
- Keep this condition distinct from:
  - `CAPTURE_LOST`: the BDO window/frame stream itself is unavailable;
  - `UI_CALIBRATION_LOST`: expected UI geometry moved or no longer matches;
  - ordinary low motion: the minimap is visible and valid, but the ship may be stationary.

Acceptance fixtures/tests:

- global map opened long enough to cover the minimap -> one loss alert, no false `ARRIVED`/`STUCK`;
- global map closed -> one restored alert after recovery debounce;
- very brief global-map/UI overlay -> no noisy alert if shorter than the configured loss debounce;
- unrelated game panel not covering the minimap -> monitoring stays healthy;
- minimap hidden or moved -> loss/degraded state followed by recalibration guidance;
- WGC capture failure -> `CAPTURE_LOST`, not minimap-only loss;
- HP/provisions remain independently detectable while navigation is suspended, when their ROIs are still visible;
- application restart while the global map is open -> degraded state is established without replaying stale navigation events.

### Architecture

```text
Detector/state machine
        |
        v
 NotificationEvent
        |
        v
 NotificationRouter
    |          |          |
    v          v          v
Telegram     Voice      Popup
Notifier     Notifier   Notifier
```

`NotificationEvent` should contain an event key, severity, timestamp, cooldown identity, and typed parameters. Each notifier renders the event in its configured locale. Detection code must not contain Telegram API calls, audio playback calls, or user-facing sentences.

### Acceptance-test matrix

- Telegram on, voice off;
- Telegram off, voice on;
- both on;
- popup only;
- popup together with either or both other channels;
- all channels off;
- Ukrainian text and Ukrainian voice pack;
- English text and English voice pack;
- Telegram failure while voice succeeds;
- missing/corrupt voice asset while Telegram succeeds;
- popup rendering failure while other enabled channels succeed;
- one event crossing produces one delivery per enabled channel;
- reset/re-arm allows the next legitimate event;
- application restart does not immediately replay stale alerts.

### Deferred decisions

- Final voice talent and tone;
- final popup visual design, animation, placement, and severity styles;
- native Windows toast versus a custom non-focus-stealing application overlay;
- whether UI locale and notification locale can be configured separately;
- quiet hours and per-event channel selection;
- installer format and whether voice packs can be optional components.

## Motion and stationarity robustness

**Priority:** Core detector/replay phase.

- Distinguish coherent directional minimap-layer translation from bounded back-and-forth oscillation around the same position.
- Add a `stationary_under_attack` fixture from Tinberra Island, where aggressive NPC fire makes the hull and minimap features jitter while the ship remains navigationally stationary.
- Do not transition from `STATIONARY` to `MOVING` on isolated spikes or alternating impact jitter; require persistent net displacement over a temporal window.
- Preserve arrival state through combat jitter unless coherent route progress resumes.
- Latch `ARRIVED` once sustained stationarity is confirmed with endpoint/route/port evidence; brief bounded post-arrival docking repositioning must not retract the event or generate a duplicate arrival.
- Leave latched `ARRIVED` only after persistent coherent departure or a new route/task state, not from a few seconds of local movement near the same endpoint.
- Add `STUCK_WITH_COLLISION_JITTER`: an active unfinished autoroute, no persistent net displacement, and bounded oscillation caused by contact with parked player ships.
- Distinguish `ARRIVED` from `STUCK` using route completion/endpoint state: a route line continuing beyond the current position is evidence against arrival.
- Detect recovery only after coherent persistent net displacement resumes; one high-diff collision frame is insufficient.
- Treat barter destinations and ordering as round-specific observations: available points are generated/randomized per round, so one recorded route is neither an exhaustive location catalog nor a universal next-destination model.
- Keep route-strategy metadata separate from arrival/STUCK/resource detection; unvisited locations in one round must not be interpreted as unavailable.
- Evaluate robust estimators such as median optical flow, feature-track consensus, cumulative net displacement, and oscillation-return tests rather than relying on scalar frame-to-frame difference alone.
- Treat HP loss/combat effects only as explanatory context; HP changes must not be required for stationarity classification.
- Add replay acceptance tests for:
  - stationary ship under repeated NPC fire;
  - stationary ship with one or several minimap-diff spikes;
  - genuine departure after stationary combat;
  - slow manoeuvring with persistent net displacement;
  - arrival while combat effects continue.

## Application updates through GitHub

**Priority:** Design alongside packaging; implement after the application shell and release format are stable.

### User value

Users should be able to discover and install trusted BDO Sea Assistant updates without manually replacing files or losing configuration. The public project repository and CI/CD source of releases will be hosted on GitHub.

### Release source

- Publish immutable, semantically versioned tags and GitHub Releases, for example `v1.2.0`.
- Build distributable artifacts in GitHub Actions rather than uploading untracked local binaries.
- The updater consumes GitHub Release assets; it must never update from mutable raw files on `main`.
- Keep release metadata and assets publicly accessible without embedding a GitHub token in the client.

### Update manifest

Each release should publish a machine-readable manifest containing at least:

- schema version;
- application version;
- release channel;
- minimum supported version;
- download URL;
- artifact size;
- SHA-256 digest;
- cryptographic signature;
- release-notes URL;
- whether the update is mandatory for compatibility/security.

The application embeds only the public verification key. Release-signing private keys belong in protected GitHub Actions secrets and must never be committed to the repository or shipped with the application.

### Client behavior

- Support **Check for updates** from the application and tray menu.
- Optionally check in the background at a bounded interval; do not block monitoring startup on a network failure.
- Display current version, available version, release notes, download size, and verification status.
- Initial policy: ask for confirmation before downloading/installing; no surprise silent updates.
- Download to a temporary location and verify both manifest signature and SHA-256 before execution.
- Use a separate updater process after the main application exits; do not overwrite a running executable in place.
- Install atomically where possible, retain a rollback path, and preserve user configuration/logs.
- Run configuration-schema migrations explicitly and back up the previous config before migration.
- Report update failures without disabling the existing installed version.
- Never send GitHub credentials, Telegram credentials, config files, or local logs as part of an update check.

### Release channels

- `stable` for normal users;
- optionally `beta` later for testers;
- never downgrade across channels without explicit confirmation.

### Security and Windows distribution

- Evaluate Authenticode code signing before public distribution to reduce SmartScreen warnings.
- Protect release tags, CI release jobs, signing secrets, and production environments in GitHub.
- Record provenance for the commit/tag used to build every artifact.
- Treat HTTPS as transport security, not artifact authenticity; signature verification remains mandatory.

### Acceptance tests

- no update available;
- valid newer update;
- network/GitHub unavailable;
- malformed or unsupported manifest;
- incorrect SHA-256;
- invalid/missing signature;
- interrupted download and resume/retry;
- insufficient disk space;
- user declines update;
- successful atomic install and restart;
- failed install followed by rollback;
- config migration succeeds and preserves settings;
- stale/beta/downgrade release is rejected according to policy.

### Deferred decisions

- portable archive, traditional installer, MSIX, or a combination;
- signing-key technology and secure CI integration;
- update-check cadence and stable/beta UX;
- whether critical updates may become mandatory after a grace period.
