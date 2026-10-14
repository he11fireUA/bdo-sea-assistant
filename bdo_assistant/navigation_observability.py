from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .minimap_observability import (
    MinimapVisibilityEvidence,
    MinimapVisibilityTracker,
    _normalize_detector_config,
    _prepare_scaled_anchor,
    _validate_bgr_uint8,
    assess_minimap_visibility,
)
from .motion import MotionClass
from .navigation_state import NavigationObservation, NavigationState, NavigationStateMachine
from .notifications import EventType


@dataclass(frozen=True)
class NavigationGateResult:
    event: EventType | None
    navigation_state: NavigationState
    navigation_updated: bool


@dataclass(frozen=True)
class MinimapNavigationResult:
    evidence: MinimapVisibilityEvidence
    event: EventType | None
    navigation_state: NavigationState
    navigation_updated: bool


class NavigationObservabilityGate:
    __slots__ = ("_visibility", "_navigation")

    def __init__(
        self,
        visibility: MinimapVisibilityTracker,
        navigation: NavigationStateMachine,
    ) -> None:
        NavigationObservabilityGate._validate_dependencies(visibility, navigation)
        self._visibility = visibility
        self._navigation = navigation

    @staticmethod
    def _validate_dependencies(
        visibility: object,
        navigation: object,
    ) -> None:
        if (
            type(visibility) is not MinimapVisibilityTracker
            or type(navigation) is not NavigationStateMachine
        ):
            raise ValueError("observability dependencies have invalid types")
        MinimapVisibilityTracker._validate_configuration(visibility)
        NavigationStateMachine._validate_configuration(navigation)

    @property
    def visibility(self) -> MinimapVisibilityTracker:
        return self._visibility

    @property
    def navigation(self) -> NavigationStateMachine:
        return self._navigation

    def update(
        self,
        evidence: MinimapVisibilityEvidence,
        observation: NavigationObservation | None = None,
        observed_monotonic: float | None = None,
    ) -> NavigationGateResult:
        NavigationObservabilityGate._validate_dependencies(
            self._visibility,
            self._navigation,
        )
        try:
            if type(evidence) is not MinimapVisibilityEvidence:
                raise ValueError("evidence must be exact MinimapVisibilityEvidence")
            MinimapVisibilityEvidence.validate(evidence)
        except ValueError:
            MinimapVisibilityTracker.reset_pending(self._visibility)
            NavigationStateMachine.suspend(self._navigation)
            raise
        observation_complete = False
        if type(observation) is NavigationObservation:
            try:
                NavigationObservation.validate(observation)
            except ValueError:
                pass
            else:
                route_context_complete = (
                    type(observation.route_complete) is bool
                    and type(observation.endpoint_ahead) is bool
                )
                independent_moving = observation.motion is MotionClass.MOVING
                trusted_recovery_motion = (
                    observation.motion is MotionClass.MOVING
                    and self._navigation.state
                    in (NavigationState.STUCK, NavigationState.RECOVERING)
                )
                observation_complete = (
                    observation.motion is not MotionClass.UNKNOWN
                    and (route_context_complete or independent_moving or trusted_recovery_motion)
                )
        event = MinimapVisibilityTracker.update(self._visibility, evidence.is_visible)
        # A missing minimap is not stationary ship evidence.  Keep visibility
        # diagnostic, but never feed its stale pixels into simple terminals.
        if not evidence.is_visible:
            return NavigationGateResult(
                event=event,
                navigation_state=NavigationStateMachine.suspend(self._navigation),
                navigation_updated=False,
            )
        if self._navigation.simple and type(observation) is NavigationObservation:
            return NavigationGateResult(
                event=event,
                navigation_state=NavigationStateMachine.update(
                    self._navigation,
                    observation,
                    observed_monotonic=observed_monotonic,
                ),
                navigation_updated=True,
            )
        if (
            not evidence.is_visible
            or not self._visibility.navigation_enabled
            or not observation_complete
        ):
            return NavigationGateResult(
                event=event,
                navigation_state=NavigationStateMachine.suspend(self._navigation),
                navigation_updated=False,
            )
        return NavigationGateResult(
            event=event,
            navigation_state=NavigationStateMachine.update(
                self._navigation,
                observation,
                observed_monotonic=observed_monotonic,
            ),
            navigation_updated=True,
        )


class MinimapNavigationMonitor:
    __slots__ = (
        "_anchor_shape",
        "_anchor_bytes",
        "_expected_scale",
        "_minimum_confidence",
        "_gate",
    )

    def __init__(
        self,
        *,
        anchor_template: np.ndarray,
        expected_scale: float,
        visibility: MinimapVisibilityTracker,
        navigation: NavigationStateMachine,
        minimum_confidence: float = 0.60,
    ) -> None:
        NavigationObservabilityGate._validate_dependencies(visibility, navigation)
        validated_anchor = _validate_bgr_uint8(anchor_template, "anchor_template")
        expected_scale, minimum_confidence = _normalize_detector_config(
            expected_scale,
            minimum_confidence,
        )
        _prepare_scaled_anchor(validated_anchor, expected_scale)
        self._anchor_shape = validated_anchor.shape
        self._anchor_bytes = bytes(validated_anchor.tobytes())
        self._expected_scale = expected_scale
        self._minimum_confidence = minimum_confidence
        self._gate = NavigationObservabilityGate(visibility, navigation)

    @property
    def expected_scale(self) -> float:
        return self._expected_scale

    @property
    def minimum_confidence(self) -> float:
        return self._minimum_confidence

    @property
    def gate(self) -> NavigationObservabilityGate:
        return self._gate

    def update(
        self,
        minimap_roi: np.ndarray,
        observation: NavigationObservation | None = None,
        observed_monotonic: float | None = None,
    ) -> MinimapNavigationResult:
        if type(self._gate) is not NavigationObservabilityGate:
            raise ValueError("monitor gate has invalid type")
        NavigationObservabilityGate._validate_dependencies(
            self._gate.visibility,
            self._gate.navigation,
        )
        expected_scale, minimum_confidence = _normalize_detector_config(
            self._expected_scale,
            self._minimum_confidence,
        )
        anchor_template = np.frombuffer(self._anchor_bytes, dtype=np.uint8).reshape(
            self._anchor_shape
        )
        try:
            evidence = assess_minimap_visibility(
                minimap_roi,
                anchor_template,
                expected_scale=expected_scale,
                minimum_confidence=minimum_confidence,
            )
        except Exception:
            NavigationStateMachine.suspend(self._gate.navigation)
            raise
        decision = NavigationObservabilityGate.update(
            self._gate,
            evidence,
            observation,
            observed_monotonic=observed_monotonic,
        )
        return MinimapNavigationResult(
            evidence=evidence,
            event=decision.event,
            navigation_state=decision.navigation_state,
            navigation_updated=decision.navigation_updated,
        )
