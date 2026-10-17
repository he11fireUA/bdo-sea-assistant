"""Prepare conservative, overlay-free minimap texture surfaces.

This module only describes whether two minimap crops contain usable geometry.  It
makes no motion estimate and emits no navigation event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

import cv2
import numpy as np


class SurfaceReason(str, Enum):
    OK = "ok"
    INSUFFICIENT_TEXTURE = "insufficient_texture"


@dataclass(frozen=True)
class MotionSurface:
    """Gradient-supported image regions safe to offer to a geometric estimator."""

    previous_gray: np.ndarray
    current_gray: np.ndarray
    previous_valid_mask: np.ndarray
    current_valid_mask: np.ndarray
    previous_feature_mask: np.ndarray
    current_feature_mask: np.ndarray
    correspondence_mask: np.ndarray
    accepted_feature_count: int
    supported_tile_count: int
    observable: bool
    reason: SurfaceReason


def _validate_crop(value: object, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 3
        or value.shape[2] != 3
        or value.dtype != np.uint8
        or value.size == 0
    ):
        raise ValueError(f"{name} must be a non-empty BGR uint8 crop")
    if value.shape[0] < 16 or value.shape[1] < 16:
        raise ValueError(f"{name} must be at least 16x16 pixels")
    return value


def _finite_fraction(value: object, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite fraction")
    result = float(value)
    if not isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an exact positive integer")
    return value


def _validate_configuration(
    *,
    border_fraction: object,
    center_radius_fraction: object,
    tile_rows: object,
    tile_columns: object,
    gradient_threshold: object,
    min_feature_count: object,
    min_tile_support: object,
) -> tuple[float, float, int, int, float, int, int]:
    border = _finite_fraction(border_fraction, "border_fraction", lower=0.0, upper=0.25)
    center = _finite_fraction(
        center_radius_fraction, "center_radius_fraction", lower=0.0, upper=0.40
    )
    rows = _positive_int(tile_rows, "tile_rows")
    columns = _positive_int(tile_columns, "tile_columns")
    threshold = _finite_fraction(gradient_threshold, "gradient_threshold", lower=0.0, upper=255.0)
    features = _positive_int(min_feature_count, "min_feature_count")
    tiles = _positive_int(min_tile_support, "min_tile_support")
    if tiles > rows * columns:
        raise ValueError("min_tile_support cannot exceed tile count")
    return border, center, rows, columns, threshold, features, tiles


def _route_overlay_mask(image: np.ndarray) -> np.ndarray:
    """Recognize legacy red/yellow and fixed-3D blue-violet route palettes."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = ((hue <= 10) | (hue >= 170)) & (saturation >= 110) & (value >= 110)
    yellow = (hue >= 15) & (hue <= 40) & (saturation >= 110) & (value >= 110)
    violet = (hue >= 105) & (hue <= 145) & (saturation >= 25) & (value >= 35)
    return red | yellow | violet


def _geometry_mask(shape: tuple[int, int], border_fraction: float, center_fraction: float) -> np.ndarray:
    height, width = shape
    mask = np.ones(shape, dtype=bool)
    border = int(np.ceil(min(height, width) * border_fraction))
    if border:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    radius = int(np.ceil(min(height, width) * center_fraction))
    if radius:
        # Include a two-pixel halo so a central marker's gradient boundary is excluded too.
        center = np.zeros(shape, dtype=np.uint8)
        cv2.circle(center, (width // 2, height // 2), radius + 2, 1, -1)
        mask &= ~center.astype(bool)
    return mask


def _gradient_feature_mask(gray: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    return valid & (magnitude >= threshold)


def _tile_support(mask: np.ndarray, rows: int, columns: int) -> int:
    height, width = mask.shape
    count = 0
    for row in range(rows):
        y1, y2 = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            x1, x2 = column * width // columns, (column + 1) * width // columns
            if mask[y1:y2, x1:x2].any():
                count += 1
    return count


def prepare_motion_surface(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    border_fraction: float = 0.08,
    center_radius_fraction: float = 0.13,
    tile_rows: int = 3,
    tile_columns: int = 3,
    gradient_threshold: float = 18.0,
    min_feature_count: int = 48,
    min_tile_support: int = 3,
) -> MotionSurface:
    """Return only mutually observable, non-overlay texture support.

    Overlay exclusion is the union from both frames, dilated to remove route
    edges.  Thus a moving overlay cannot manufacture features in either image.
    """
    previous = _validate_crop(previous, "previous")
    current = _validate_crop(current, "current")
    if previous.shape != current.shape:
        raise ValueError("previous and current crops must have identical dimensions")
    border, center, rows, columns, threshold, required_features, required_tiles = _validate_configuration(
        border_fraction=border_fraction,
        center_radius_fraction=center_radius_fraction,
        tile_rows=tile_rows,
        tile_columns=tile_columns,
        gradient_threshold=gradient_threshold,
        min_feature_count=min_feature_count,
        min_tile_support=min_tile_support,
    )
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    geometry = _geometry_mask(previous_gray.shape, border, center)
    overlays = _route_overlay_mask(previous) | _route_overlay_mask(current)
    # Sobel features straddle a line's boundary; mask a two-pixel halo as well.
    overlays = cv2.dilate(overlays.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)).astype(bool)
    valid = geometry & ~overlays
    previous_features = _gradient_feature_mask(previous_gray, valid, threshold)
    current_features = _gradient_feature_mask(current_gray, valid, threshold)
    correspondence = previous_features & current_features
    feature_count = int(np.count_nonzero(correspondence))
    tile_count = _tile_support(correspondence, rows, columns)
    observable = feature_count >= required_features and tile_count >= required_tiles
    return MotionSurface(
        previous_gray=previous_gray,
        current_gray=current_gray,
        previous_valid_mask=valid.copy(),
        current_valid_mask=valid.copy(),
        previous_feature_mask=previous_features,
        current_feature_mask=current_features,
        correspondence_mask=correspondence,
        accepted_feature_count=feature_count,
        supported_tile_count=tile_count,
        observable=observable,
        reason=SurfaceReason.OK if observable else SurfaceReason.INSUFFICIENT_TEXTURE,
    )


def track_endpoints_are_valid(
    surface: MotionSurface,
    source: tuple[int, int],
    destination: tuple[int, int],
) -> bool:
    """Check source and destination against their own frame's feature masks."""
    if not isinstance(surface, MotionSurface):
        raise ValueError("surface must be MotionSurface")
    for point, name in ((source, "source"), (destination, "destination")):
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or type(point[0]) is not int
            or type(point[1]) is not int
        ):
            raise ValueError(f"{name} endpoint must be an exact integer (x, y) tuple")
    source_x, source_y = source
    destination_x, destination_y = destination
    height, width = surface.previous_feature_mask.shape
    if not (0 <= source_x < width and 0 <= source_y < height):
        return False
    if not (0 <= destination_x < width and 0 <= destination_y < height):
        return False
    return bool(
        surface.previous_feature_mask[source_y, source_x]
        and surface.current_feature_mask[destination_y, destination_x]
    )
