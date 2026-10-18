# Changelog

## 0.2.3

- Fixed stationary completed routes remaining `UNKNOWN` when a tiny isolated blue-violet ship/node marker fragment touched the center ambiguity radius.
- Preserve `UNKNOWN` for route-like continuations across the center mask, and `PRESENT` for a genuine route; isolated fragments now permit `ABSENT` route evidence.
- Replayed two private operator-labeled post-arrival sessions mechanically through the production analyzer: every frame classified route `ABSENT`. The sessions begin after arrival and are not whole-route arrival validation.
- Updated Windows package metadata and artifact names to `0.2.3`.

## 0.2.2

- Made trusted coherent minimap translation sufficient for `MOVING`, even when islands or terrain occlude the auto-route.
- Deferred route classification until physical stationarity; route `PRESENT`, `ABSENT`, and `UNKNOWN` now decide only `STUCK`, `DEPARTED`, or continued uncertainty after the ship stops.
- Verified the new truth table on a 60-second 277-frame field trace: all 270 trusted moving observations produced `MOVING`, with no route-dependent dropout; six physically stationary samples remained conservatively `UNKNOWN`.
- Added the runtime version to the default application window title.
- Updated Windows package metadata and artifact names to `0.2.2`.

## 0.2.1

- Fixed sustained horizontal travel falling from `MOVING` to `UNKNOWN`: the motion estimator was already tracking all replayed frame pairs, while the horizontal blue-violet route was rejected because its axis-aligned strip filled its bounding box and began just outside the old ship-center radius.
- Made route-line thickness validation orientation-independent and widened the center-adjacency allowance only enough to cover the ship-marker occlusion.
- Added left-to-right and right-to-left horizontal-motion regressions.
- Kept displayed `MOVING` for up to five continuous seconds of ambiguous evidence; explicit `NO_SHIP` remains immediate and terminal states are not created from ambiguity.
- Disarmed `DEPARTED` after `NO_SHIP` until a new trusted `MOVING` observation, preventing a repeated arrival when the operator returns to the helm while already stationary.
- Documented the required fixed-orientation 3D minimap with a privacy-reviewed reference image and recalibration guidance.
- Updated Windows package metadata and artifact names to `0.2.1`.
