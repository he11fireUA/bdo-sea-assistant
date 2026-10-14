from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import hypot, isfinite
from typing import Iterable

import cv2
import numpy as np


class MotionClass(str, Enum):
    MOVING = "moving"
    STATIONARY = "stationary"
    JITTER = "jitter"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MotionVector:
    dx: float
    dy: float
    confidence: float

    @property
    def magnitude(self) -> float:
        return hypot(self.dx, self.dy)


@dataclass(frozen=True)
class ActivityEvidence:
    mae: float
    changed_fraction: float


@dataclass(frozen=True)
class ActivityDecision:
    soft_activity_persistent: bool
    broad_activity_persistent: bool
    onset_sequence: int | None = None
    confirmed_sequence: int | None = None


class ActivityMotionTracker:
    """Temporal activity veto when compact JPEGs cannot yield stable geometry.

    Soft or broad activity may veto a stationarity candidate.  It is not
    geometric evidence and never confirms movement, recovery, arrival, or stuck.
    """

    def __init__(
        self,
        *,
        window: int = 4,
        required_soft_frames: int = 3,
        required_broad_frames: int = 3,
        soft_mae: float = 0.35,
        broad_mae: float = 3.0,
        broad_changed_fraction: float = 0.04,
    ) -> None:
        if type(window) is not int or window < 3:
            raise ValueError("activity window must be an integer of at least 3 frames")
        if type(required_soft_frames) is not int or not 3 <= required_soft_frames <= window:
            raise ValueError(
                "required_soft_frames must be an integer from 3 through the window size"
            )
        if type(required_broad_frames) is not int or not 3 <= required_broad_frames <= window:
            raise ValueError(
                "required_broad_frames must be an integer from 3 through the window size"
            )
        if not all(isfinite(value) for value in (soft_mae, broad_mae, broad_changed_fraction)):
            raise ValueError("activity thresholds must be finite")
        if soft_mae < 0 or broad_mae < 0 or not 0 <= broad_changed_fraction <= 1:
            raise ValueError("activity thresholds are out of range")
        if soft_mae > broad_mae:
            raise ValueError("soft_mae must not exceed broad_mae")
        self.window = window
        self.required_soft_frames = required_soft_frames
        self.required_broad_frames = required_broad_frames
        self.soft_mae = soft_mae
        self.broad_mae = broad_mae
        self.broad_changed_fraction = broad_changed_fraction
        self._samples: deque[tuple[int, ActivityEvidence]] = deque(maxlen=window)

    def update(self, sequence: int, evidence: ActivityEvidence) -> ActivityDecision:
        if not isfinite(evidence.mae) or not isfinite(evidence.changed_fraction):
            raise ValueError("activity evidence must be finite")
        if evidence.mae < 0 or not 0 <= evidence.changed_fraction <= 1:
            raise ValueError("activity evidence is out of range")
        self._samples.append((sequence, evidence))
        soft = [item.mae >= self.soft_mae for _, item in self._samples]
        broad = [
            item.mae > self.broad_mae
            and item.changed_fraction > self.broad_changed_fraction
            for _, item in self._samples
        ]
        soft_persistent = sum(soft) >= self.required_soft_frames
        broad_activity_persistent = sum(broad) >= self.required_broad_frames
        onset = None
        confirmed = None
        if broad_activity_persistent:
            onset = next(
                sample_sequence
                for (sample_sequence, _), is_soft in zip(self._samples, soft)
                if is_soft
            )
            confirmed = sequence
        return ActivityDecision(soft_persistent, broad_activity_persistent, onset, confirmed)


def _as_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError("motion image must be grayscale or BGR")
    return image.astype(np.float32) / 255.0


def estimate_activity(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    border: int = 5,
    center_radius_ratio: float = 0.13,
    changed_pixel_threshold: int = 10,
) -> ActivityEvidence:
    if previous.shape != current.shape:
        raise ValueError("activity frames must have identical dimensions")
    previous_gray = (
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if previous.ndim == 3 else previous
    )
    current_gray = (
        cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
    )
    if previous_gray.ndim != 2 or current_gray.ndim != 2:
        raise ValueError("activity image must be grayscale or BGR")

    height, width = previous_gray.shape
    valid = np.ones((height, width), dtype=np.uint8)
    if border > 0:
        valid[:border, :] = 0
        valid[-border:, :] = 0
        valid[:, :border] = 0
        valid[:, -border:] = 0
    radius = round(min(height, width) * center_radius_ratio)
    cv2.circle(valid, (width // 2, height // 2), radius, 0, -1)
    selected = valid.astype(bool)
    if not np.any(selected):
        raise ValueError("activity mask contains no pixels")

    difference = cv2.absdiff(previous_gray, current_gray)
    values = difference[selected]
    return ActivityEvidence(
        mae=float(np.mean(values)),
        changed_fraction=float(np.mean(values > changed_pixel_threshold)),
    )


def estimate_translation(previous: np.ndarray, current: np.ndarray) -> MotionVector:
    previous_gray = _as_gray_float(previous)
    current_gray = _as_gray_float(current)
    if previous_gray.shape != current_gray.shape:
        raise ValueError("motion frames must have identical dimensions")

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        confidence, warp = cv2.findTransformECC(
            previous_gray,
            current_gray,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            1,
        )
    except cv2.error:
        return MotionVector(0.0, 0.0, 0.0)
    confidence_value = float(confidence)
    bounded_confidence = (
        max(0.0, min(1.0, confidence_value)) if isfinite(confidence_value) else 0.0
    )
    return MotionVector(float(warp[0, 2]), float(warp[1, 2]), bounded_confidence)


def classify_motion_window(
    vectors: Iterable[MotionVector],
    *,
    minimum_confidence: float = 0.5,
    minimum_valid_samples: int = 3,
    minimum_valid_fraction: float = 0.60,
    stationary_mean_step: float = 0.08,
    stationary_net_displacement: float = 0.25,
    moving_net_displacement: float = 2.0,
    moving_mean_step: float = 0.25,
    moving_coherence: float = 0.65,
    jitter_minimum_path: float = 2.0,
    jitter_maximum_coherence: float = 0.35,
) -> MotionClass:
    thresholds = (
        minimum_confidence,
        minimum_valid_fraction,
        stationary_mean_step,
        stationary_net_displacement,
        moving_net_displacement,
        moving_mean_step,
        moving_coherence,
        jitter_minimum_path,
        jitter_maximum_coherence,
    )
    if not all(isfinite(value) for value in thresholds):
        raise ValueError("motion thresholds must be finite")
    if (
        not 0 <= minimum_confidence <= 1
        or stationary_mean_step < 0
        or stationary_net_displacement < 0
        or moving_net_displacement < 0
        or moving_mean_step < 0
        or not 0 <= moving_coherence <= 1
        or jitter_minimum_path < 0
        or not 0 <= jitter_maximum_coherence <= 1
    ):
        raise ValueError("motion thresholds are out of range")
    if (
        type(minimum_valid_samples) is not int
        or minimum_valid_samples < 3
        or not 0.60 <= minimum_valid_fraction <= 1.0
    ):
        raise ValueError("motion evidence guards must require at least 3 samples and 60% validity")
    all_samples = tuple(vectors)
    samples = tuple(
        vector
        for vector in all_samples
        if isfinite(vector.dx)
        and isfinite(vector.dy)
        and isfinite(vector.confidence)
        and 0 <= vector.confidence <= 1
        and vector.confidence >= minimum_confidence
    )
    if (
        len(samples) < minimum_valid_samples
        or not all_samples
        or len(samples) / len(all_samples) < minimum_valid_fraction
    ):
        return MotionClass.UNKNOWN

    path_length = sum(vector.magnitude for vector in samples)
    net_dx = sum(vector.dx for vector in samples)
    net_dy = sum(vector.dy for vector in samples)
    net_displacement = hypot(net_dx, net_dy)
    mean_step = path_length / len(samples)
    coherence = net_displacement / path_length if path_length else 1.0
    moving_steps = sum(vector.magnitude > stationary_mean_step for vector in samples)
    moving_step_fraction = moving_steps / len(samples)

    if mean_step <= stationary_mean_step and net_displacement <= stationary_net_displacement:
        return MotionClass.STATIONARY
    if (
        net_displacement >= moving_net_displacement
        and mean_step >= moving_mean_step
        and coherence >= moving_coherence
        and moving_steps >= 3
        and moving_step_fraction >= 0.60
    ):
        return MotionClass.MOVING
    if path_length >= jitter_minimum_path and coherence <= jitter_maximum_coherence:
        return MotionClass.JITTER
    return MotionClass.UNKNOWN
