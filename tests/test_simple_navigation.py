from __future__ import annotations

import cv2
import numpy as np

from bdo_assistant.motion import MotionClass
from bdo_assistant.navigation_state import NavigationObservation, NavigationState, NavigationStateMachine
from bdo_assistant.route_evidence import RouteGeometryClass
from bdo_assistant.runtime import SimpleNavigationAnalyzer


def _moving_texture(offset: int, *, shape: tuple[int, int, int] = (80, 80, 3)) -> np.ndarray:
    """A textured patch that moves within one LK tile but changes many pixels."""
    image = np.zeros(shape, dtype=np.uint8)
    patch = np.indices((20, 20)).sum(axis=0) % 2 * 255
    image[3:23, 3 + offset:23 + offset] = patch[..., None]
    return image


def _target_marker(*, center_x: int, center_y: int) -> np.ndarray:
    minimap = np.zeros((260, 300, 3), dtype=np.uint8)
    cv2.rectangle(minimap, (center_x - 5, center_y - 10), (center_x + 4, center_y + 9), (255, 255, 0), -1)
    return minimap


def test_simple_detector_prefers_a_far_cyan_target_marker_over_misleading_near_route(monkeypatch) -> None:
    monkeypatch.setattr("bdo_assistant.runtime.assess_route_geometry", lambda roi: RouteGeometryClass.NEAR)
    analyzer = SimpleNavigationAnalyzer()
    far_target = _target_marker(center_x=210, center_y=65)

    observation = analyzer(0, far_target)

    assert observation.route_complete is False
    assert observation.endpoint_ahead is True


def test_simple_detector_marks_a_near_cyan_target_marker_as_route_complete(monkeypatch) -> None:
    monkeypatch.setattr("bdo_assistant.runtime.assess_route_geometry", lambda roi: RouteGeometryClass.AHEAD)
    analyzer = SimpleNavigationAnalyzer()
    near_target = _target_marker(center_x=130, center_y=106)

    observation = analyzer(0, near_target)

    assert observation.route_complete is True
    assert observation.endpoint_ahead is False


def test_simple_detector_confirms_moving_from_direct_texture_and_long_route(monkeypatch) -> None:
    monkeypatch.setattr("bdo_assistant.runtime.assess_route_geometry", lambda roi: RouteGeometryClass.AHEAD)
    analyzer = SimpleNavigationAnalyzer(moving_pixels=0.05)
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)

    assert analyzer(0, _moving_texture(0)).motion is MotionClass.UNKNOWN
    for timestamp in range(1, 6):
        observation = analyzer(timestamp, _moving_texture(timestamp))
        assert observation.motion is MotionClass.MOVING
        assert machine.update(observation, observed_monotonic=float(timestamp)) is NavigationState.UNKNOWN
    assert machine.update(analyzer(6, _moving_texture(6)), observed_monotonic=6.0) is NavigationState.MOVING


def test_simple_detector_confirms_arrived_from_static_texture_and_short_route(monkeypatch) -> None:
    monkeypatch.setattr("bdo_assistant.runtime.assess_route_geometry", lambda roi: RouteGeometryClass.NEAR)
    analyzer = SimpleNavigationAnalyzer()
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    roi = _moving_texture(0)

    assert analyzer(0, roi).motion is MotionClass.UNKNOWN
    assert machine.update(analyzer(1, roi), observed_monotonic=10.0) is NavigationState.UNKNOWN
    assert machine.update(analyzer(2, roi), observed_monotonic=14.99) is NavigationState.UNKNOWN
    assert machine.update(analyzer(3, roi), observed_monotonic=15.0) is NavigationState.ARRIVED


def test_simple_detector_confirms_stuck_from_static_texture_and_long_route(monkeypatch) -> None:
    monkeypatch.setattr("bdo_assistant.runtime.assess_route_geometry", lambda roi: RouteGeometryClass.AHEAD)
    analyzer = SimpleNavigationAnalyzer()
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    roi = _moving_texture(0)

    assert analyzer(0, roi).motion is MotionClass.UNKNOWN
    assert machine.update(analyzer(1, roi), observed_monotonic=20.0) is NavigationState.UNKNOWN
    assert machine.update(analyzer(2, roi), observed_monotonic=25.0) is NavigationState.STUCK


def test_simple_detector_measures_texture_when_route_geometry_is_ambiguous(monkeypatch) -> None:
    analyzer = SimpleNavigationAnalyzer(moving_pixels=0.05)
    monkeypatch.setattr("bdo_assistant.runtime.assess_route_geometry", lambda roi: RouteGeometryClass.AMBIGUOUS)

    assert analyzer(0, _moving_texture(0)).motion is MotionClass.UNKNOWN
    ambiguous = analyzer(1, _moving_texture(1))
    assert ambiguous.motion is MotionClass.MOVING
    assert ambiguous.route_complete is ambiguous.endpoint_ahead is None

    monkeypatch.setattr("bdo_assistant.runtime.assess_route_geometry", lambda roi: RouteGeometryClass.AHEAD)
    mismatched = analyzer(2, _moving_texture(2, shape=(81, 80, 3)))
    assert mismatched.motion is MotionClass.UNKNOWN


def test_simple_state_machine_surfaces_sustained_motion_with_ambiguous_route() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    moving_unknown_route = NavigationObservation(MotionClass.MOVING, None, None)

    assert machine.update(moving_unknown_route, observed_monotonic=1.0) is NavigationState.UNKNOWN
    assert machine.update(moving_unknown_route, observed_monotonic=5.99) is NavigationState.UNKNOWN
    assert machine.update(moving_unknown_route, observed_monotonic=6.0) is NavigationState.MOVING


def test_simple_state_machine_confirms_arrival_when_a_near_pulse_is_obscured_after_approach() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    ahead_motion = NavigationObservation(MotionClass.MOVING, False, True)
    near_motion = NavigationObservation(MotionClass.MOVING, True, False)
    stationary_ambiguous = NavigationObservation(MotionClass.STATIONARY, None, None)

    assert machine.update(ahead_motion, observed_monotonic=1.0) is NavigationState.UNKNOWN
    assert machine.update(ahead_motion, observed_monotonic=6.0) is NavigationState.MOVING
    assert machine.update(near_motion, observed_monotonic=7.0) is NavigationState.MOVING
    assert machine.update(stationary_ambiguous, observed_monotonic=8.0) is NavigationState.MOVING
    assert machine.update(stationary_ambiguous, observed_monotonic=13.0) is NavigationState.ARRIVED


def test_simple_state_machine_expires_moving_after_static_ambiguous_route_without_near() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    ahead_motion = NavigationObservation(MotionClass.MOVING, False, True)
    stationary_ambiguous = NavigationObservation(MotionClass.STATIONARY, None, None)

    assert machine.update(ahead_motion, observed_monotonic=1.0) is NavigationState.UNKNOWN
    assert machine.update(ahead_motion, observed_monotonic=6.0) is NavigationState.MOVING
    assert machine.update(stationary_ambiguous, observed_monotonic=7.0) is NavigationState.MOVING
    assert machine.update(stationary_ambiguous, observed_monotonic=12.0) is NavigationState.UNKNOWN


def test_simple_state_machine_keeps_confirmed_state_during_pending_or_ambiguous_evidence() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    arrived_evidence = NavigationObservation(MotionClass.STATIONARY, True, False)
    moving_evidence = NavigationObservation(MotionClass.MOVING, False, True)

    assert machine.update(arrived_evidence, observed_monotonic=1.0) is NavigationState.UNKNOWN
    assert machine.update(arrived_evidence, observed_monotonic=6.0) is NavigationState.ARRIVED
    assert machine.update(moving_evidence, observed_monotonic=7.0) is NavigationState.ARRIVED
    assert machine.update(NavigationObservation(MotionClass.MOVING, None, None), observed_monotonic=8.0) is NavigationState.ARRIVED
    assert machine.update(moving_evidence, observed_monotonic=12.0) is NavigationState.ARRIVED
    assert machine.update(moving_evidence, observed_monotonic=17.0) is NavigationState.MOVING

    assert machine.suspend() is NavigationState.UNKNOWN
