from __future__ import annotations

from math import isfinite
from numbers import Real
from enum import Enum

import cv2
import numpy as np


class RouteGeometryClass(str, Enum):
    AHEAD = "ahead"
    NEAR = "near"
    AMBIGUOUS = "ambiguous"


class RoutePresence(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


def _positive_seconds(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number or None")
    return float(value)


def _observed_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0:
        raise ValueError("observed_monotonic must be a finite non-negative number")
    return float(value)


class RouteEvidenceTracker:
    def __init__(
        self,
        *,
        confirmations: int = 3,
        ahead_seconds: float | None = None,
        near_seconds: float | None = None,
    ) -> None:
        # confirmations is retained for sequence-only callers, not converted to time.
        if type(confirmations) is not int or confirmations < 3:
            raise ValueError("confirmations must be an integer of at least 3")
        self._confirmations = confirmations
        self._ahead_seconds = _positive_seconds(ahead_seconds, "ahead_seconds")
        self._near_seconds = _positive_seconds(near_seconds, "near_seconds")
        self._ahead_count = 0
        self._near_count = 0
        self._ahead_since: float | None = None
        self._near_since: float | None = None
        self._last_observed_monotonic: float | None = None
        self._route_seen_ahead = False

    def reset(self) -> None:
        self._ahead_count = 0
        self._near_count = 0
        self._ahead_since = None
        self._near_since = None
        self._last_observed_monotonic = None
        self._route_seen_ahead = False

    def __call__(
        self,
        sequence: int,
        minimap_roi: np.ndarray,
        observed_monotonic: float | None = None,
    ) -> tuple[bool | None, bool | None]:
        return self.update(assess_route_geometry(minimap_roi), observed_monotonic=observed_monotonic)

    def _validate_time(self, observed_monotonic: float | None) -> float | None:
        if observed_monotonic is None:
            return None
        value = _observed_time(observed_monotonic)
        if self._last_observed_monotonic is not None and value <= self._last_observed_monotonic:
            self.reset()
            raise ValueError("observed_monotonic must strictly increase")
        self._last_observed_monotonic = value
        return value

    def _confirmed(self, count: int, since: float | None, seconds: float | None, now: float | None) -> tuple[bool, float | None]:
        if seconds is None:
            return count >= self._confirmations, since
        if now is None:
            return False, None
        if since is None:
            return False, now
        return now - since >= seconds, since

    def update(
        self,
        geometry: RouteGeometryClass,
        observed_monotonic: float | None = None,
    ) -> tuple[bool | None, bool | None]:
        if type(geometry) is not RouteGeometryClass:
            raise ValueError("geometry must be RouteGeometryClass")
        now = self._validate_time(observed_monotonic)
        if geometry is RouteGeometryClass.AHEAD:
            self._near_count = 0
            self._near_since = None
            self._ahead_count += 1
            confirmed, self._ahead_since = self._confirmed(
                self._ahead_count, self._ahead_since, self._ahead_seconds, now
            )
            if confirmed:
                self._route_seen_ahead = True
                return False, True
            return None, None
        if geometry is RouteGeometryClass.NEAR and self._route_seen_ahead:
            self._ahead_count = 0
            self._ahead_since = None
            self._near_count += 1
            confirmed, self._near_since = self._confirmed(
                self._near_count, self._near_since, self._near_seconds, now
            )
            if confirmed:
                return True, False
            return None, None
        self._ahead_count = 0
        self._near_count = 0
        self._ahead_since = None
        self._near_since = None
        return None, None


def assess_route_geometry(minimap_bgr: np.ndarray) -> RouteGeometryClass:
    """Classify route extent around the fixed center of a 2D minimap.

    This is geometry-only evidence. NEAR must not independently prove arrival;
    a temporal tracker must first observe AHEAD.
    """
    height, width = minimap_bgr.shape[:2]
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    route_mask = (
        ((hue <= 15) | (hue >= 165))
        & (saturation >= 45)
        & (value >= 20)
    ).astype(np.uint8)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(route_mask, 8)
    center_x = width / 2.0
    center_y = height / 2.0
    center_radius = max(6.0, min(height, width) * 0.025)
    max_distance = 0.0

    max_route_area = height * width * 0.03
    for label, (_, _, _, _, area) in enumerate(stats[1:], 1):
        # The route is a narrow line. Large filled red/orange terrain regions
        # can cross the player marker but are not evidence that a route remains.
        if area < 3 or area > max_route_area:
            continue
        ys, xs = np.nonzero(labels == label)
        if not xs.size:
            continue
        center_distances = np.hypot(xs - center_x, ys - center_y)
        if float(center_distances.min()) > center_radius:
            continue
        max_distance = max(max_distance, float(center_distances.max()))

    if max_distance == 0.0:
        return RouteGeometryClass.AMBIGUOUS

    normalized_extent = max_distance / (min(height, width) / 2.0)
    if normalized_extent >= 0.40:
        return RouteGeometryClass.AHEAD
    if normalized_extent <= 0.30:
        return RouteGeometryClass.NEAR
    return RouteGeometryClass.AMBIGUOUS


def assess_target_marker_geometry(minimap_bgr: np.ndarray) -> RouteGeometryClass | None:
    """Classify the cyan navigation-target pin relative to the fixed ship center.

    ``None`` means the distinctive pin is not observable; callers must then rely
    on weaker route-line evidence rather than inventing target proximity.
    """
    height, width = minimap_bgr.shape[:2]
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    marker_mask = (
        (hue >= 75)
        & (hue <= 110)
        & (saturation >= 100)
        & (value >= 120)
    ).astype(np.uint8)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(marker_mask, 8)
    center_x = width / 2.0
    center_y = height / 2.0
    distances: list[float] = []
    for (_, _, component_width, component_height, area), (x, y) in zip(stats[1:], centroids[1:]):
        aspect_ratio = component_width / component_height if component_height else 0.0
        if not (50 <= area <= 240 and 6 <= component_width <= 14 and 12 <= component_height <= 26):
            continue
        if not 0.3 <= aspect_ratio <= 0.8:
            continue
        distances.append(float(np.hypot(x - center_x, y - center_y)))
    if not distances:
        return None
    return (
        RouteGeometryClass.NEAR
        if min(distances) <= min(height, width) * 0.18
        else RouteGeometryClass.AHEAD
    )


def assess_route_presence(minimap_bgr: np.ndarray) -> RoutePresence:
    """Classify conservative evidence for the blue-violet 3D auto-route."""
    if not isinstance(minimap_bgr, np.ndarray) or minimap_bgr.ndim != 3 or minimap_bgr.shape[2] != 3:
        raise ValueError("minimap_bgr must be a BGR color image")
    if minimap_bgr.size == 0 or minimap_bgr.dtype != np.uint8:
        raise ValueError("minimap_bgr must be a non-empty uint8 image")

    height, width = minimap_bgr.shape[:2]
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    route_mask = (
        (hue >= 110)
        & (hue <= 145)
        & (saturation >= 25)
        & (value >= 35)
    ).astype(np.uint8)
    route_mask = cv2.morphologyEx(
        route_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    if not np.any(route_mask):
        return RoutePresence.ABSENT

    _, labels, stats, _ = cv2.connectedComponentsWithStats(route_mask, 8)
    center_x = width / 2.0
    center_y = height / 2.0
    center_radius = max(6.0, min(height, width) * 0.06)
    ambiguity_radius = max(center_radius, min(height, width) * 0.09)
    required_extent = min(height, width) * 0.23
    max_area = height * width * 0.02

    ambiguous_near_center = False
    for label, (_, _, component_width, component_height, area) in enumerate(stats[1:], 1):
        if area < 8 or component_width <= 0 or component_height <= 0:
            continue
        ys, xs = np.nonzero(labels == label)
        if not xs.size:
            continue
        distances = np.hypot(xs - center_x, ys - center_y)
        if float(distances.min()) <= ambiguity_radius:
            ambiguous_near_center = True
        if area > max_area:
            continue
        fill_ratio = area / (component_width * component_height)
        points = np.column_stack((xs, ys)).astype(np.float32)
        covariance = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(eigenvalues[-1] / max(eigenvalues[0], 1e-6))
        if (
            float(distances.min()) <= center_radius
            and float(distances.max()) >= required_extent
            and fill_ratio <= 0.35
            and elongation >= 6.0
        ):
            return RoutePresence.PRESENT

    return RoutePresence.UNKNOWN if ambiguous_near_center else RoutePresence.ABSENT
