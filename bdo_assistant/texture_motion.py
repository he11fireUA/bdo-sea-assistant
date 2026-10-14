"""Conservative geometric translation evidence for prepared minimap surfaces.

The estimator is deliberately evidence-only: an invalid result is frozen at zero
and carries a reason rather than being interpreted as stationary movement.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

import cv2
import numpy as np

from bdo_assistant.motion_surface import MotionSurface, prepare_motion_surface


class TextureMotionReason(str, Enum):
    OK = "ok"
    INSUFFICIENT_TEXTURE = "insufficient_texture"
    INSUFFICIENT_TRACKS = "insufficient_tracks"
    INSUFFICIENT_DISTRIBUTION = "insufficient_distribution"
    INCONSISTENT_MOTION = "inconsistent_motion"


@dataclass(frozen=True)
class TextureMotionEvidence:
    """A previous-to-current image displacement, or frozen invalid evidence."""

    dx: float
    dy: float
    dt: float
    residual: float
    track_count: int
    inlier_count: int
    tile_coverage: int
    valid: bool
    reason: TextureMotionReason


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an exact positive integer")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _frozen(
    reason: TextureMotionReason, dt: float, tracks: int = 0, inliers: int = 0, tiles: int = 0
) -> TextureMotionEvidence:
    return TextureMotionEvidence(0.0, 0.0, dt, 0.0, tracks, inliers, tiles, False, reason)


def _tiled_corners(surface: MotionSurface, rows: int, columns: int, per_tile: int) -> np.ndarray | None:
    height, width = surface.previous_gray.shape
    corners: list[np.ndarray] = []
    mask = surface.previous_feature_mask.astype(np.uint8)
    for row in range(rows):
        y1, y2 = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            x1, x2 = column * width // columns, (column + 1) * width // columns
            tile_mask = np.zeros_like(mask)
            tile_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            found = cv2.goodFeaturesToTrack(
                surface.previous_gray, maxCorners=per_tile, qualityLevel=0.01,
                minDistance=5, mask=tile_mask, blockSize=5,
            )
            if found is not None:
                corners.append(found)
    return np.concatenate(corners, axis=0) if corners else None


def _endpoint_mask(surface: MotionSurface, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    height, width = surface.previous_gray.shape
    source = np.rint(starts).astype(int)
    destination = np.rint(ends).astype(int)
    in_bounds = (
        (source[:, 0] >= 0) & (source[:, 0] < width) & (source[:, 1] >= 0) & (source[:, 1] < height)
        & (destination[:, 0] >= 0) & (destination[:, 0] < width) & (destination[:, 1] >= 0) & (destination[:, 1] < height)
    )
    accepted = np.zeros(len(starts), dtype=bool)
    indices = np.flatnonzero(in_bounds)
    accepted[indices] = (
        surface.previous_valid_mask[source[indices, 1], source[indices, 0]]
        & surface.current_valid_mask[destination[indices, 1], destination[indices, 0]]
    )
    return accepted


def _tile_coverage(points: np.ndarray, inliers: np.ndarray, shape: tuple[int, int], rows: int, columns: int) -> int:
    height, width = shape
    occupied: set[tuple[int, int]] = set()
    for x, y in points[inliers]:
        occupied.add((min(rows - 1, int(y) * rows // height), min(columns - 1, int(x) * columns // width)))
    return len(occupied)


def estimate_texture_motion(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    dt: float = 1.0,
) -> TextureMotionEvidence:
    """Return conservative translation evidence directly from two minimap crops.

    Crop-pair discontinuities, including resizes and dimensions unsupported by
    the surface preparer, are unavailable motion evidence rather than errors.
    Configuration errors remain programmer errors on the lower-level APIs.
    """
    elapsed = _positive_finite(dt, "dt")
    try:
        surface = prepare_motion_surface(previous, current)
    except ValueError:
        return _frozen(TextureMotionReason.INSUFFICIENT_TEXTURE, elapsed)
    return estimate_texture_translation(surface, dt=elapsed)


def estimate_texture_translation(
    surface: MotionSurface,
    *,
    tile_rows: int = 3,
    tile_columns: int = 3,
    max_corners_per_tile: int = 24,
    min_track_count: int = 12,
    min_inlier_count: int = 9,
    min_tile_coverage: int = 3,
    dt: float = 1.0,
    forward_backward_error: float = 0.75,
    consensus_residual: float = 0.8,
) -> TextureMotionEvidence:
    """Estimate previous-to-current translation using distributed LK tracks.

    ``dx > 0`` means texture moved right in the current image; ``dy > 0``
    means it moved down.  All rejected cases return a frozen zero displacement.
    """
    if not isinstance(surface, MotionSurface):
        raise ValueError("surface must be MotionSurface")
    rows, columns = _positive_int(tile_rows, "tile_rows"), _positive_int(tile_columns, "tile_columns")
    per_tile = _positive_int(max_corners_per_tile, "max_corners_per_tile")
    min_tracks, min_inliers, min_tiles = (
        _positive_int(min_track_count, "min_track_count"), _positive_int(min_inlier_count, "min_inlier_count"), _positive_int(min_tile_coverage, "min_tile_coverage")
    )
    if min_inliers > min_tracks or min_tiles > rows * columns:
        raise ValueError("minimum requirements exceed estimator capacity")
    elapsed = _positive_finite(dt, "dt")
    fb_limit, residual_limit = _positive_finite(forward_backward_error, "forward_backward_error"), _positive_finite(consensus_residual, "consensus_residual")
    if not surface.observable:
        return _frozen(TextureMotionReason.INSUFFICIENT_TEXTURE, elapsed)
    points = _tiled_corners(surface, rows, columns, per_tile)
    if points is None:
        return _frozen(TextureMotionReason.INSUFFICIENT_TRACKS, elapsed)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(surface.previous_gray, surface.current_gray, points, None, **lk)
    if forward is None or forward_status is None:
        return _frozen(TextureMotionReason.INSUFFICIENT_TRACKS, elapsed)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(surface.current_gray, surface.previous_gray, forward, None, **lk)
    if backward is None or backward_status is None:
        return _frozen(TextureMotionReason.INSUFFICIENT_TRACKS, elapsed)
    starts, ends, returns = points.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2)
    tracked = (forward_status.ravel() == 1) & (backward_status.ravel() == 1)
    tracked &= np.linalg.norm(starts - returns, axis=1) <= fb_limit
    tracked &= _endpoint_mask(surface, starts, ends)
    starts, ends = starts[tracked], ends[tracked]
    track_count = len(starts)
    if track_count < min_tracks:
        return _frozen(TextureMotionReason.INSUFFICIENT_TRACKS, elapsed, track_count)
    displacements = ends - starts
    median = np.median(displacements, axis=0)
    deviations = np.linalg.norm(displacements - median, axis=1)
    inliers = deviations <= residual_limit
    inlier_count = int(inliers.sum())
    coverage = _tile_coverage(starts, inliers, surface.previous_gray.shape, rows, columns)
    if inlier_count < min_inliers:
        return _frozen(TextureMotionReason.INCONSISTENT_MOTION, elapsed, track_count, inlier_count, coverage)
    if coverage < min_tiles:
        return _frozen(TextureMotionReason.INSUFFICIENT_DISTRIBUTION, elapsed, track_count, inlier_count, coverage)
    translation = np.median(displacements[inliers], axis=0)
    residual = float(np.median(np.linalg.norm(displacements[inliers] - translation, axis=1)))
    return TextureMotionEvidence(float(translation[0]), float(translation[1]), elapsed, residual, track_count, inlier_count, coverage, True, TextureMotionReason.OK)
