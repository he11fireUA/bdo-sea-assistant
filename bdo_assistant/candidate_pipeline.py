"""Offline-derived candidate motion metrics used only through live shadow mode."""
from __future__ import annotations

from dataclasses import replace

from .monitoring import MonitoringSample
from .motion import MotionClass
from .navigation_state import NavigationObservation
from .temporal_motion import CandidateMotionBridge, TemporalMotionTracker
from .texture_motion import estimate_texture_motion


class CandidateTextureMotionPipeline:
    """Produce scalar texture-motion diagnostics without a navigation verdict."""

    def __init__(self) -> None:
        self._previous_roi = None
        self._previous_time: float | None = None
        self._tracker = TemporalMotionTracker()
        self._bridge = CandidateMotionBridge()

    def process(self, sample: MonitoringSample) -> dict[str, str | int | float | bool | None]:
        if sample.minimap_roi is None:
            self.reset()
            return {"reason": "no_minimap_roi", "motion": "unknown", "valid": False}
        current = sample.minimap_roi
        timestamp = sample.received_monotonic
        if self._previous_roi is None or self._previous_time is None:
            self._previous_roi = current.copy()
            self._previous_time = timestamp
            return {"reason": "first_sample", "motion": "unknown", "valid": False}
        elapsed = timestamp - self._previous_time
        if elapsed <= 0:
            self.reset()
            self._previous_roi = current.copy()
            self._previous_time = timestamp
            return {"reason": "timing_discontinuity", "motion": "unknown", "valid": False}
        evidence = estimate_texture_motion(self._previous_roi, current, dt=elapsed)
        raw_decision = self._tracker.update(timestamp, evidence)
        decision = self._bridge.update(timestamp, raw_decision)
        self._previous_roi = current.copy()
        self._previous_time = timestamp
        return {
            "motion": decision.motion_class.value,
            "reason": decision.reason,
            "raw_motion": raw_decision.motion_class.value,
            "raw_reason": raw_decision.reason,
            "decision_lag_samples": 1,
            "valid": evidence.valid,
            "dx": evidence.dx,
            "dy": evidence.dy,
            "dt": evidence.dt,
            "tracks": evidence.track_count,
            "inliers": evidence.inlier_count,
            "tiles": evidence.tile_coverage,
        }

    def reset(self) -> None:
        self._previous_roi = None
        self._previous_time = None
        self._tracker = TemporalMotionTracker()
        self._bridge = CandidateMotionBridge()


class CandidateMotionAuthorityLoop:
    """Promote verified MOVING only; terminal states remain fail-closed."""

    def __init__(self, legacy_loop: object, *, motion_pipeline: object | None = None) -> None:
        if not callable(getattr(legacy_loop, "process", None)):
            raise ValueError("legacy_loop must provide process")
        self._legacy_loop = legacy_loop
        self._motion_pipeline = CandidateTextureMotionPipeline() if motion_pipeline is None else motion_pipeline
        if not all(callable(getattr(self._motion_pipeline, name, None)) for name in ("process", "reset")):
            raise ValueError("motion_pipeline must provide process and reset")

    def __getattr__(self, name: str):
        return getattr(self._legacy_loop, name)

    def process(self, sample: MonitoringSample):
        metrics = self._motion_pipeline.process(sample)
        try:
            candidate_motion = MotionClass(metrics["motion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("motion_pipeline returned invalid motion") from exc
        observation = sample.navigation_observation
        if observation is not None:
            promoted = MotionClass.MOVING if candidate_motion is MotionClass.MOVING else MotionClass.UNKNOWN
            sample = replace(sample, navigation_observation=NavigationObservation(
                promoted, observation.route_complete, observation.endpoint_ahead
            ))
        return self._legacy_loop.process(sample)

    def capture_closed(self, sequence: int):
        self._motion_pipeline.reset()
        return self._legacy_loop.capture_closed(sequence)
