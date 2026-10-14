from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, ulp
from numbers import Real
from time import monotonic
from typing import Callable

import cv2
import numpy as np

from .notifications import EventType


class MinimapVisibilityState(str, Enum):
    UNKNOWN = "unknown"
    VISIBLE = "visible"
    LOST = "lost"


@dataclass(frozen=True)
class MinimapVisibilityEvidence:
    is_visible: bool
    confidence: float
    minimum_confidence: float = 0.50

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "minimum_confidence", float(self.minimum_confidence))

    def validate(self) -> None:
        if type(self.is_visible) is not bool:
            raise ValueError("visibility evidence is_visible must be boolean")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, Real):
            raise ValueError("visibility evidence confidence must be finite and between 0 and 1")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("visibility evidence confidence must be finite and between 0 and 1")
        if isinstance(self.minimum_confidence, bool) or not isinstance(
            self.minimum_confidence, Real
        ):
            raise ValueError("visibility evidence minimum confidence is invalid")
        minimum_confidence = float(self.minimum_confidence)
        if not isfinite(minimum_confidence) or not 0.50 <= minimum_confidence <= 1:
            raise ValueError("visibility evidence minimum confidence is invalid")
        if self.is_visible is not (confidence >= minimum_confidence):
            raise ValueError("visibility evidence decision must match confidence threshold")


def _validate_bgr_uint8(value: object, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 3
        or value.shape[2] != 3
        or value.size == 0
        or value.dtype != np.uint8
    ):
        raise ValueError(f"{name} must be a non-empty BGR color uint8 array")
    return value


def _normalize_detector_config(
    expected_scale: object,
    minimum_confidence: object,
) -> tuple[float, float]:
    if isinstance(expected_scale, bool) or not isinstance(expected_scale, Real):
        raise ValueError("expected_scale must be a finite number")
    scale = float(expected_scale)
    lower_tolerance = 8 * ulp(0.60)
    upper_tolerance = 8 * ulp(1.80)
    if not isfinite(scale) or not 0.60 - lower_tolerance <= scale <= 1.80 + upper_tolerance:
        raise ValueError("expected_scale must be between 0.60 and 1.80")
    scale = min(1.80, max(0.60, scale))
    if isinstance(minimum_confidence, bool) or not isinstance(minimum_confidence, Real):
        raise ValueError("minimum_confidence must be a finite number")
    confidence = float(minimum_confidence)
    if not isfinite(confidence) or not 0.50 <= confidence <= 1:
        raise ValueError("minimum_confidence must be between 0.50 and 1")
    return scale, confidence


def _prepare_scaled_anchor(
    anchor_template: object,
    expected_scale: float,
    roi_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    anchor = _validate_bgr_uint8(anchor_template, "anchor_template")
    template = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
    if float(template.std()) < 1.0:
        raise ValueError("anchor_template must contain visible texture")
    width = round(template.shape[1] * expected_scale)
    height = round(template.shape[0] * expected_scale)
    if width < 8 or height < 8:
        raise ValueError("scaled anchor must retain at least 8x8 pixels")
    if roi_shape is not None and (width > roi_shape[1] or height > roi_shape[0]):
        raise ValueError("scaled anchor must fit inside the minimap ROI")
    interpolation = cv2.INTER_AREA if expected_scale < 1.0 else cv2.INTER_CUBIC
    candidate = cv2.resize(template, (width, height), interpolation=interpolation)
    if float(candidate.std()) < 1.0:
        raise ValueError("scaled anchor must retain visible texture")
    return candidate


def assess_minimap_visibility(
    image: np.ndarray,
    anchor_template: np.ndarray,
    *,
    expected_scale: float,
    minimum_confidence: float = 0.60,
) -> MinimapVisibilityEvidence:
    image = _validate_bgr_uint8(image, "image")
    expected_scale, minimum_confidence = _normalize_detector_config(
        expected_scale,
        minimum_confidence,
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidate = _prepare_scaled_anchor(anchor_template, expected_scale, gray.shape)
    response = cv2.matchTemplate(gray, candidate, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, _ = cv2.minMaxLoc(response)
    confidence = float(confidence)
    if not isfinite(confidence):
        raise ValueError("anchor match must produce finite confidence")
    if not 0 <= confidence <= 1:
        raise ValueError("anchor match confidence must be between 0 and 1")
    return MinimapVisibilityEvidence(
        is_visible=confidence >= minimum_confidence,
        confidence=confidence,
        minimum_confidence=minimum_confidence,
    )


class MinimapVisibilityTracker:
    __slots__ = (
        "_lost_frames",
        "_restored_frames",
        "_state",
        "_lost_count",
        "_restored_count",
        "_lost_after_seconds",
        "_clock",
        "_lost_since",
    )

    def __init__(
        self,
        *,
        lost_frames: int = 3,
        restored_frames: int = 3,
        lost_after_seconds: float | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._lost_frames = lost_frames
        self._restored_frames = restored_frames
        self._lost_after_seconds = lost_after_seconds
        self._clock = clock
        MinimapVisibilityTracker._validate_configuration(self)
        self._state = MinimapVisibilityState.UNKNOWN
        self._lost_count = 0
        self._restored_count = 0
        self._lost_since: float | None = None

    def _validate_configuration(self) -> None:
        thresholds = (self._lost_frames, self._restored_frames)
        if any(type(value) is not int or value < 2 for value in thresholds):
            raise ValueError(
                "visibility confirmation thresholds must be positive integers of at least 2"
            )
        if self._lost_after_seconds is not None and (
            isinstance(self._lost_after_seconds, bool)
            or not isinstance(self._lost_after_seconds, Real)
            or not isfinite(float(self._lost_after_seconds))
            or float(self._lost_after_seconds) <= 0
        ):
            raise ValueError("lost_after_seconds must be a finite positive number or None")
        if not callable(self._clock):
            raise ValueError("clock must be callable")

    @property
    def lost_frames(self) -> int:
        return self._lost_frames

    @property
    def restored_frames(self) -> int:
        return self._restored_frames

    @property
    def lost_after_seconds(self) -> float | None:
        return (
            None
            if self._lost_after_seconds is None
            else float(self._lost_after_seconds)
        )

    @property
    def state(self) -> MinimapVisibilityState:
        return self._state

    @property
    def navigation_enabled(self) -> bool:
        return self._state is MinimapVisibilityState.VISIBLE

    def configure_loss_delay(self, lost_after_seconds: float | None) -> None:
        """Change loss timing without discarding visibility or recovery evidence."""
        # Validate a candidate before touching any live state.
        MinimapVisibilityTracker(lost_after_seconds=lost_after_seconds)
        if lost_after_seconds != self._lost_after_seconds:
            self._lost_after_seconds = lost_after_seconds
            self._lost_count = 0
            self._lost_since = None

    def reset_pending(self) -> None:
        MinimapVisibilityTracker._validate_configuration(self)
        self._lost_count = 0
        self._restored_count = 0
        self._lost_since = None

    def _confirm_loss(self) -> bool:
        if self._lost_after_seconds is None:
            self._lost_count += 1
            return self._lost_count >= self._lost_frames
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, Real) or not isfinite(float(now)):
            raise ValueError("clock must return a finite number")
        now = float(now)
        if self._lost_since is None:
            self._lost_since = now
            return False
        if now < self._lost_since:
            raise ValueError("clock must not move backwards")
        return now - self._lost_since >= float(self._lost_after_seconds)

    def update(self, is_visible: bool) -> EventType | None:
        MinimapVisibilityTracker._validate_configuration(self)
        if type(is_visible) is not bool:
            raise ValueError("visibility observation must be boolean")
        if self._state is MinimapVisibilityState.UNKNOWN:
            if is_visible:
                self._lost_count = 0
                self._lost_since = None
                self._restored_count += 1
                if self._restored_count >= self._restored_frames:
                    self._state = MinimapVisibilityState.VISIBLE
                    self._restored_count = 0
                return None
            self._restored_count = 0
            if self._confirm_loss():
                self._state = MinimapVisibilityState.LOST
                self._lost_count = 0
                self._lost_since = None
                return EventType.MINIMAP_VISIBILITY_LOST
            return None

        if self._state is MinimapVisibilityState.VISIBLE:
            self._restored_count = 0
            if is_visible:
                self._lost_count = 0
                self._lost_since = None
                return None
            if self._confirm_loss():
                self._state = MinimapVisibilityState.LOST
                self._lost_count = 0
                self._lost_since = None
                return EventType.MINIMAP_VISIBILITY_LOST
            return None

        self._lost_count = 0
        if not is_visible:
            self._restored_count = 0
            return None
        self._lost_since = None
        self._restored_count += 1
        if self._restored_count >= self._restored_frames:
            self._state = MinimapVisibilityState.VISIBLE
            self._restored_count = 0
            return EventType.MINIMAP_VISIBILITY_RESTORED
        return None
