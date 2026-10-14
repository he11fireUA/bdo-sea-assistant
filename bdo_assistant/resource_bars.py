from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import cv2
import numpy as np

from .ui_locator import Rect


class ResourceKind(str, Enum):
    HP = "hp"
    RATIONS = "rations"


@dataclass(frozen=True)
class ResourceReading:
    observable: bool
    percent: float | None
    support: float
    candidate_percent: float | None = None


def estimate_fill_percent(image: np.ndarray, track: Rect, kind: ResourceKind) -> float:
    if image is None or image.ndim != 3:
        raise ValueError("image must be a BGR color frame")
    if track.width <= 0 or track.height <= 0:
        raise ValueError("track must have positive dimensions")
    x1 = max(0, track.x)
    y1 = max(0, track.y)
    x2 = min(image.shape[1], track.x + track.width)
    y2 = min(image.shape[0], track.y + track.height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("track does not intersect the frame")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if kind is ResourceKind.HP:
        mask = ((hue <= 10) | (hue >= 170)) & (saturation >= 95) & (value >= 75)
    elif kind is ResourceKind.RATIONS:
        mask = (hue >= 15) & (hue <= 40) & (saturation >= 95) & (value >= 75)
    else:
        raise ValueError(f"unsupported resource kind: {kind}")

    required_pixels = max(1, int(np.ceil(crop.shape[0] * 0.50)))
    filled = mask.sum(axis=0) >= required_pixels

    # Bridge tiny JPEG/antialias gaps without joining unrelated distant text.
    active = np.flatnonzero(filled)
    if active.size == 0:
        return 0.0
    for left, right in zip(active, active[1:]):
        if 1 < right - left <= 3:
            filled[left : right + 1] = True

    first = int(np.flatnonzero(filled)[0])
    endpoint = crop.shape[1]
    empty_run = 0
    for column in range(first, crop.shape[1]):
        if filled[column]:
            empty_run = 0
        else:
            empty_run += 1
            if empty_run >= 4:
                endpoint = column - empty_run + 1
                break
    filled_width = max(0, endpoint - first)
    if endpoint == crop.shape[1] and filled_width / crop.shape[1] >= 0.97:
        return 100.0
    return float(np.clip(filled_width / crop.shape[1] * 100.0, 0.0, 100.0))


def measure_resource_bar(image: np.ndarray, track: Rect, kind: ResourceKind) -> ResourceReading:
    """Measure a calibrated bar only when its full track is visually present."""
    if image is None or image.ndim != 3:
        raise ValueError("image must be a BGR color frame")
    if track.width <= 0 or track.height <= 0:
        raise ValueError("track must have positive dimensions")
    if (
        track.x < 0
        or track.y < 0
        or track.x + track.width > image.shape[1]
        or track.y + track.height > image.shape[0]
    ):
        return ResourceReading(False, None, 0.0)
    x1 = max(0, track.x)
    y1 = max(0, track.y)
    x2 = min(image.shape[1], track.x + track.width)
    y2 = min(image.shape[0], track.y + track.height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("track does not intersect the frame")
    boundary_differences = []
    if y1 > 0:
        boundary_differences.append(
            float(np.mean(np.abs(crop[0].astype(np.int16) - image[y1 - 1, x1:x2].astype(np.int16))))
        )
    if y2 < image.shape[0]:
        boundary_differences.append(
            float(np.mean(np.abs(crop[-1].astype(np.int16) - image[y2, x1:x2].astype(np.int16))))
        )
    if x1 > 0:
        boundary_differences.append(
            float(np.mean(np.abs(crop[:, 0].astype(np.int16) - image[y1:y2, x1 - 1].astype(np.int16))))
        )
    if x2 < image.shape[1]:
        boundary_differences.append(
            float(np.mean(np.abs(crop[:, -1].astype(np.int16) - image[y1:y2, x2].astype(np.int16))))
        )
    has_track_boundary = bool(boundary_differences) and max(boundary_differences) >= 8.0

    # The paired ship HUD has a pale, neutral rail immediately above each
    # calibrated track. A dark outlined scene panel can satisfy generic contrast
    # and neutral-coverage checks, but it does not have this track-specific frame.
    def has_pale_rail(row: np.ndarray) -> bool:
        rail_hsv = cv2.cvtColor(row[np.newaxis, :, :], cv2.COLOR_BGR2HSV)[0]
        return bool(
            np.mean(
                (rail_hsv[:, 1] <= 50)
                & (rail_hsv[:, 2] >= 90)
                & (rail_hsv[:, 2] <= 180)
            )
            >= 0.55
        )

    has_track_frame = y1 > 0 and has_pale_rail(image[y1 - 1, x1:x2])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if kind is ResourceKind.HP:
        colored = ((hue <= 10) | (hue >= 170)) & (saturation >= 95) & (value >= 75)
    elif kind is ResourceKind.RATIONS:
        colored = (hue >= 15) & (hue <= 40) & (saturation >= 95) & (value >= 75)
    else:
        raise ValueError(f"unsupported resource kind: {kind}")
    neutral_track = (saturation <= 50) & (value >= 15) & (value <= 100)
    supporting_columns = (colored | neutral_track).sum(axis=0) >= max(
        1, int(np.ceil(crop.shape[0] * 0.30))
    )
    support = float(np.mean(supporting_columns))
    strong_colored_fill = float(
        np.mean(colored.sum(axis=0) >= max(1, int(np.ceil(crop.shape[0] * 0.50))))
    ) >= 0.95
    candidate_percent = (
        estimate_fill_percent(image, track, kind)
        if support >= 0.60 and has_track_boundary
        else None
    )
    # A nearly solid calibrated color fill is independently strong evidence of a
    # ship bar. It preserves the pale-rail requirement for low/empty tracks,
    # while allowing field-confirmed full bars whose rail is not pale.
    if candidate_percent is None or not (has_track_frame or strong_colored_fill):
        return ResourceReading(False, None, support, candidate_percent)
    return ResourceReading(True, candidate_percent, support, candidate_percent)


class ThresholdWarningTracker:
    def __init__(
        self,
        *,
        warning_below: float,
        reset_at: float,
        confirmation_frames: int = 3,
    ) -> None:
        if not isfinite(warning_below) or not isfinite(reset_at):
            raise ValueError("resource thresholds must be finite")
        if not 0 <= warning_below <= 100 or not 0 <= reset_at <= 100:
            raise ValueError("resource thresholds must be between 0 and 100")
        if warning_below >= reset_at:
            raise ValueError("reset threshold must be above warning threshold")
        if type(confirmation_frames) is not int or confirmation_frames < 1:
            raise ValueError("confirmation_frames must be a positive integer")
        self.warning_below = warning_below
        self.reset_at = reset_at
        self.confirmation_frames = confirmation_frames
        self.alerted = False
        self._low_count = 0

    def reset_pending(self) -> None:
        self._low_count = 0

    def reset(self) -> None:
        self.alerted = False
        self._low_count = 0

    def update(self, percent: float) -> bool:
        if not 0 <= percent <= 100:
            raise ValueError("resource percent must be between 0 and 100")
        if self.alerted:
            if percent >= self.reset_at:
                self.alerted = False
                self._low_count = 0
            return False

        if percent <= self.warning_below:
            self._low_count += 1
            if self._low_count >= self.confirmation_frames:
                self.alerted = True
                self._low_count = 0
                return True
        else:
            self._low_count = 0
        return False
