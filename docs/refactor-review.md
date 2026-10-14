# Cleanup after first-working-v1

## Scope completed

- Removed hidden Navigation and Diagnostics page definitions, widgets, vector icons, candidate tester rendering and unused status matrix builder.
- Preserved technical runtime messages in a bounded in-memory deque (200 entries), rather than constructing an invisible diagnostics QWidget or dropping error details.
- Preserved live dashboard, event history, recorder lifecycle and simple navigation wiring.
- Fixed repeat scheduling after settings Apply and valid capture/calibration reacquisition; missing resource readings cannot create stale repeats. Removed unread last-navigation state.
- Added live adapter replacement and minimap-loss-delay configuration through session/bridge under the existing lifecycle lock.
- Routed Test Voice through the same serial worker as notification clips; removed obsolete sleeper arguments.
- Retained short HP/Rations display deadband but accept sustained small changes, including 98 -> 100. Raw resource warning inputs unchanged.
- Saved feedback is shown only after runtime application succeeds.

## Validation

Parent executed full suite after integration: **687 passed, 1 skipped**.
Baseline: 619 passed, 1 skipped. The intermediate UI-only cleanup had 615 passed,
1 skipped because four tests for removed legacy UI widgets were removed; retained
runtime/recorder/locale tests still passed. Added regression cases cover the new
fixes. Native capture/audio were not started for automated validation.

## Intentionally retained

- `route_evidence.py`: route geometry is used by the live simple analyzer.
- `desktop_ui.py`: modern UI still imports shared controller construction; do not delete the whole legacy Tk module without extracting that factory.
- Conservative analyzer, candidate/texture/temporal/surface/shadow and replay modules: standalone replay/testing dependencies remain. No candidate authority was added to the live factory.
- Local recordings, calibration, settings and installed assets remain untouched.

## Limits

Audio crackling has not been reproduced or proven fixed. This change fixes the
confirmed Test Voice serialization defect, not an assumed CPU/audio-driver issue.
A source snapshot is not a backup of local application data. No new native
reliability claim follows from the automated suite.
