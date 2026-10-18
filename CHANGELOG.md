# Changelog

## 0.2.4

- Replaced pair-to-pair `MOVING` decisions with a rolling four-second progress window at the production 0.2-second analysis cadence.
- Require sustained displacement in at least 75% of the window to enter `MOVING`, retain it down to 50%, and require at least 30% net progress relative to total point travel; this hysteresis rejects wave-driven back-and-forth rocking without flapping on isolated quiet pairs.
- Combine the motion-window transition with at least three seconds and 21 accepted samples of terminal route confirmation, preventing a terminal verdict before the complete 20-pair movement-entry opportunity has elapsed even with tolerated quiet pairs or capture-cadence jitter; the earliest clean stop decision is approximately six seconds.
- Replayed the 272-frame stationary wave session with zero false `MOVING`; the 280-frame horizontal session and 277-frame island-travel session remained continuously `MOVING` after the initial window warm-up. A moving-to-wave replay settled to `DEPARTED` after eight seconds without returning to `MOVING`.
- Updated Windows package metadata and artifact names to `0.2.4`.

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
