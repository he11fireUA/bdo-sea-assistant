# Changelog

## 0.2.1

- Fixed sustained horizontal travel falling from `MOVING` to `UNKNOWN`: the motion estimator was already tracking all replayed frame pairs, while the horizontal blue-violet route was rejected because its axis-aligned strip filled its bounding box and began just outside the old ship-center radius.
- Made route-line thickness validation orientation-independent and widened the center-adjacency allowance only enough to cover the ship-marker occlusion.
- Added left-to-right and right-to-left horizontal-motion regressions.
- Kept displayed `MOVING` for up to five continuous seconds of ambiguous evidence; explicit `NO_SHIP` remains immediate and terminal states are not created from ambiguity.
- Disarmed `DEPARTED` after `NO_SHIP` until a new trusted `MOVING` observation, preventing a repeated arrival when the operator returns to the helm while already stationary.
- Documented the required fixed-orientation 3D minimap with a privacy-reviewed reference image and recalibration guidance.
- Updated Windows package metadata and artifact names to `0.2.1`.
