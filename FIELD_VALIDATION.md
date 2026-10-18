# Field validation backlog

This document collects questions and fixtures that cannot be resolved honestly from the distributed offline dataset. None of these block location-agnostic detector, state-machine, or notification-core development.

Four operator-labeled local sessions covering general movement, sustained horizontal movement, arrival, and grounding informed the current 3D detector work. They are not published as automated replay fixtures, so they do not establish reproducible field-validation coverage for this repository.

## Fixtures needed in a future BDO session

### Minimap observability

Record the calibrated minimap ROI at full capture resolution while:

1. the minimap is normally visible for at least 10 seconds;
2. the world map is opened for 1–2 seconds and closed;
3. the world map is opened for at least 10 seconds and closed;
4. an unrelated panel is opened without covering the minimap;
5. if practical, the minimap is hidden or moved and then restored.

This is needed to tune `content_valid` separately from capture and calibration health. Navigation timers must be suspended while the minimap is unavailable and restarted with empty persistence windows after recovery.

Offline implementation status (2026-08-02):

- the saved minimap anchor remains a visibility diagnostic while navigation uses the calibrated 3D minimap ROI;
- detector configuration and evidence are fail-closed for malformed/non-finite inputs;
- startup remains `UNKNOWN` until persistent visible evidence;
- every non-visible or incomplete-evidence frame suspends navigation immediately, including the loss-debounce interval;
- confirmed `ARRIVED`/`STUCK` latches survive sensor suspension, while unconfirmed and recovery/departure counters reset;
- an end-to-end synthetic occlusion replay verifies one LOST/RESTORED event and fresh navigation persistence after restore.

The synthetic occluder proves orchestration semantics only. It does **not** prove that the production threshold recognizes every real BDO world map, menu, or hidden-HUD variant; the field sequence above remains required. Version 0.2.1 uses the fixed-orientation 3D minimap, recognizes axis-aligned horizontal route geometry, bridges brief `MOVING → UNKNOWN` ambiguity for up to five seconds, and gates all minimap analysis on the paired ship HP and rations HUD.

### Alpha runtime threat boundary

Alpha fail-closed guarantees cover malformed/missing WGC frames and ROIs, invalid public constructor arguments, detector/OpenCV failures, incomplete observations, ordinary public-API misuse, and notification adapter failures. These cases must suspend dependent navigation and must not reuse stale persistence.

The alpha does not claim isolation against arbitrary Python code already executing inside the assistant process. Deliberate `object.__setattr__` mutation of private slots, manual re-entry into `__init__`, monkeypatching class methods, or fabrication of internal detector objects are out of scope: that capability is equivalent to full process compromise. The alpha has no third-party in-process plugin API. Signed distribution and process-integrity controls belong to the later packaging/updater phase.

### Minimap motion quality

A local operator-labeled 60-second horizontal-motion session recorded 280 minimap frames. Mechanical replay produced valid coherent texture motion for all 279 frame pairs; the failure was route-presence geometry, not Lucas–Kanade motion tracking. After the route fix, the authoritative analyzer/state-machine replay produced one startup `UNKNOWN` followed by 279 consecutive `MOVING` states.

The same replay did **not** support discarding the upper 50% of the minimap. Feature counts by vertical quarter were `11219 / 20033 / 11549 / 3070` from top to bottom: the upper half supplied most available points, while the bottom quarter was sparsest. Full-ROI, lower-half, and lower-two-thirds motion estimates all remained valid, so production retains the full masked ROI rather than hard-cropping useful geometry.

A later 60-second trace recorded 277 frames around islands where terrain repeatedly occluded and fragmented the auto-route. Texture motion remained trusted on 270 frames, but route evidence was `UNKNOWN` on 228 frames, causing a 26.7-second visible `UNKNOWN` interval even though the ship kept moving. Version 0.2.2 therefore treats trusted coherent texture translation as sufficient for `MOVING` and evaluates route presence only after physical stationarity. Replay of that trace produced `MOVING` for all 270 trusted-moving observations with zero movement dropouts; the startup frame and six physically stationary samples remained conservatively `UNKNOWN`.

A 60-second stationary post-arrival session then captured wave-driven minimap rocking: 60 of 271 valid frame pairs exceeded the old `0.04 px` pair threshold even though the ship made no progress. The false vectors were predominantly vertical and oscillatory; their summed path was 57 times larger than their final net displacement. Version 0.2.4 evaluates the last 20 frame-pair vectors (four seconds at the production cadence), requiring at least 75% moving pairs to enter `MOVING`, retaining it down to 50%, and requiring a net-progress ratio of at least 30%. Authoritative replay classified all 252 post-warm-up wave windows as `STATIONARY`, all 260 horizontal-travel windows as `MOVING`, and all 257 island-travel windows as `MOVING`. Terminal confirmation requires both three seconds and 21 accepted stationary samples, so neither cadence jitter nor tolerated quiet onset pairs can let a verdict beat the complete 20-pair movement-entry opportunity: a clean synthetic stop confirms after roughly six seconds. The captured moving-to-wave transition settled to `DEPARTED` only after the real oscillation had remained below movement criteria plus the terminal floor, without returning to `MOVING`.

The current Git fixtures are intentionally compact `110×100` JPEG crops. They are suitable for deterministic regression logic but can erase subpixel motion. A future recording should retain the full calibrated minimap ROI at approximately 1 Hz for selected short windows:

- normal approach;
- five-second false pause followed by movement;
- collision jitter;
- combat jitter;
- confirmed recovery;
- arrival plus local docking repositioning.

The production estimator must be validated without reading filenames, manifest labels, source sequence numbers, or localized text as features.

### Resource transitions

If safely available during normal play, record:

- ship HP crossing below 60%, remaining low for at least five samples, then recovering above 65%;
- rations crossing below 10%, remaining low for at least five samples, then a manual refill above 15%;
- one temporary invalid/covered resource ROI.

Do not intentionally endanger the ship merely to manufacture a fixture.

### UI compatibility

Capture calibration samples for configurations actually used or planned:

- another BDO UI scale;
- moved ship HUD;
- another resolution/aspect ratio;
- multi-monitor placement;
- minimized BDO window;
- locked Windows session.

## Operator decisions

1. Resolved: version 0.2.x targets the fixed-orientation 3D minimap; the former 2D-only decision is obsolete.
2. Should `MINIMAP_VISIBILITY_RESTORED` be sent through every enabled channel, or only popup/log to avoid unnecessary voice and Telegram noise?
3. Should UI language and notification-channel language be one global setting, or may Telegram, voice, and popup each use a different locale?
4. For offline voice, should destination names be spoken when known, or should the first voice pack use generic phrases such as “Прибули” to avoid pronunciation and localization errors?
5. Preferred default persistence values to validate in game:
   - stationarity candidate: 6 seconds;
   - arrival/stuck confirmation: longer and dependent on route evidence;
   - minimap loss/recovery debounce: 3 valid/invalid samples.
6. Should a restored-minimap event be silent when visibility was lost for only a short period but long enough to cross debounce?
7. Is changing BDO UI scale or moving the ship HUD part of normal use, or mainly an installation/calibration scenario?
8. For popup UX, is a custom non-focus-stealing overlay preferred over a native Windows toast?

## Offline conclusions already supported

- Barter destination identity is not required for motion, stationarity, arrival, stuck, HP/rations, or observability core logic.
- Five seconds of low motion is insufficient for arrival.
- Combat/collision pixel activity without coherent cumulative displacement is not route movement.
- Recovery onset may be backdated after persistence: current fixture supports onset at sequence 1076 with confirmation around 1079.
- Ship resource track geometry should be calibrated/frozen using the shared UI scale; colored fill is measured separately.
- Notification channels remain independent; one adapter failure must not block the others.
