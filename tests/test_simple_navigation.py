from __future__ import annotations

import cv2
import numpy as np
import pytest

from bdo_assistant.motion import MotionClass
from bdo_assistant.navigation_state import NavigationObservation, NavigationState, NavigationStateMachine
from bdo_assistant import route_evidence
from bdo_assistant.runtime import SimpleNavigationAnalyzer


def _map_texture(*, dx: float = 0.0, dy: float = 0.0, route: bool = True, marker_angle: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    gray = rng.integers(35, 220, (260, 300), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(image, transform, (300, 260), flags=cv2.INTER_LINEAR)
    center = (150, 130)
    if route:
        cv2.line(image, center, (85, 250), (115, 65, 60), 2, cv2.LINE_AA)
    marker = np.array([[150, 112], [141, 141], [150, 136], [159, 141]], dtype=np.float32)
    rotation = cv2.getRotationMatrix2D(center, marker_angle, 1.0)
    rotated = cv2.transform(marker[None, :, :], rotation)[0].astype(np.int32)
    cv2.polylines(image, [rotated], True, (255, 255, 210), 3, cv2.LINE_AA)
    return image


def test_route_presence_accepts_thin_purple_line_from_ship_center() -> None:
    assert route_evidence.assess_route_presence(_map_texture(route=True)) is route_evidence.RoutePresence.PRESENT


def test_route_presence_accepts_horizontal_3d_route_occluded_by_ship_marker() -> None:
    image = _map_texture(route=False)
    # The live 3D route is a bright, nearly filled horizontal strip whose first
    # visible pixel starts just beyond the centered ship-marker silhouette.
    cv2.rectangle(image, (167, 135), (299, 141), (115, 65, 60), -1)

    assert route_evidence.assess_route_presence(image) is route_evidence.RoutePresence.PRESENT


@pytest.mark.parametrize("dx", [-2.0, 2.0])
def test_horizontal_3d_route_reports_motion_in_both_directions(dx: float) -> None:
    analyzer = SimpleNavigationAnalyzer(minimum_displacement=0.04)
    previous = _map_texture(route=False)
    current = _map_texture(dx=dx, route=False)
    for image in (previous, current):
        cv2.rectangle(image, (167, 135), (299, 141), (115, 65, 60), -1)

    assert analyzer(0, previous).motion is MotionClass.UNKNOWN
    observation = analyzer(1, current)

    assert observation.motion is MotionClass.MOVING
    assert observation.route_complete is False
    assert observation.endpoint_ahead is True


def test_route_presence_rejects_missing_route_and_large_blue_terrain() -> None:
    image = _map_texture(route=False)
    cv2.rectangle(image, (158, 35), (250, 120), (180, 85, 55), -1)

    assert route_evidence.assess_route_presence(image) is route_evidence.RoutePresence.UNKNOWN


def test_route_presence_marks_fragmented_or_occluded_route_unknown() -> None:
    image = _map_texture(route=False)
    cv2.line(image, (150, 130), (140, 148), (115, 65, 60), 2, cv2.LINE_AA)
    cv2.line(image, (132, 164), (85, 250), (115, 65, 60), 2, cv2.LINE_AA)

    assert route_evidence.assess_route_presence(image) is route_evidence.RoutePresence.UNKNOWN


def test_route_presence_rejects_offset_thin_blue_violet_terrain_component() -> None:
    image = _map_texture(route=False)
    contour = np.array([[166, 116], [178, 98], [195, 82], [220, 69], [247, 64]], dtype=np.int32)
    cv2.polylines(image, [contour], False, (115, 65, 60), 2, cv2.LINE_AA)

    assert route_evidence.assess_route_presence(image) is route_evidence.RoutePresence.UNKNOWN


def test_route_presence_reports_clear_absence_only_without_route_like_pixels() -> None:
    assert route_evidence.assess_route_presence(_map_texture(route=False)) is route_evidence.RoutePresence.ABSENT


def test_route_presence_ignores_unrelated_route_colored_pixels_far_from_ship() -> None:
    image = _map_texture(route=False)
    cv2.line(image, (235, 28), (280, 62), (115, 65, 60), 2, cv2.LINE_AA)

    assert route_evidence.assess_route_presence(image) is route_evidence.RoutePresence.ABSENT


def test_detector_reports_motion_for_displacement_in_any_direction() -> None:
    for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        analyzer = SimpleNavigationAnalyzer(minimum_displacement=0.04)
        assert analyzer(0, _map_texture(route=True)).motion is MotionClass.UNKNOWN

        observation = analyzer(1, _map_texture(dx=dx, dy=dy, route=True))

        assert observation.motion is MotionClass.MOVING
        assert observation.route_complete is False
        assert observation.endpoint_ahead is True


def test_detector_detects_slow_subpixel_motion_without_using_direction() -> None:
    analyzer = SimpleNavigationAnalyzer(minimum_displacement=0.04)
    analyzer(0, _map_texture(route=True))

    observation = analyzer(1, _map_texture(dx=0.20, dy=-0.15, route=True))

    assert observation.motion is MotionClass.MOVING


def test_detector_ignores_rotation_inside_central_ship_marker_zone() -> None:
    analyzer = SimpleNavigationAnalyzer(minimum_displacement=0.04)
    analyzer(0, _map_texture(route=True, marker_angle=0))

    observation = analyzer(1, _map_texture(route=True, marker_angle=90))

    assert observation.motion is MotionClass.STATIONARY


def test_detector_maps_missing_route_to_departed_evidence() -> None:
    analyzer = SimpleNavigationAnalyzer(minimum_displacement=0.04)
    roi = _map_texture(route=False)

    assert analyzer(0, roi).motion is MotionClass.UNKNOWN
    observation = analyzer(1, roi)

    assert observation.motion is MotionClass.STATIONARY
    assert observation.route_complete is True
    assert observation.endpoint_ahead is False


def test_detector_maps_fragmented_route_to_unknown_and_resets_departure_confirmation() -> None:
    analyzer = SimpleNavigationAnalyzer(minimum_displacement=0.04)
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    absent = _map_texture(route=False)
    fragmented = absent.copy()
    cv2.line(fragmented, (150, 130), (140, 148), (115, 65, 60), 2, cv2.LINE_AA)
    cv2.line(fragmented, (132, 164), (85, 250), (115, 65, 60), 2, cv2.LINE_AA)

    analyzer(0, absent)
    assert machine.update(analyzer(1, absent), observed_monotonic=1.0) is NavigationState.UNKNOWN
    unknown = analyzer(2, fragmented)
    assert unknown.route_complete is None and unknown.endpoint_ahead is None
    assert machine.update(unknown, observed_monotonic=5.0) is NavigationState.UNKNOWN
    assert machine.update(analyzer(3, absent), observed_monotonic=6.0) is NavigationState.UNKNOWN
    assert machine.update(analyzer(4, absent), observed_monotonic=11.0) is NavigationState.DEPARTED


def test_simple_state_machine_reports_moving_immediately_only_with_route() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    moving_with_route = NavigationObservation(MotionClass.MOVING, False, True)
    moving_without_route = NavigationObservation(MotionClass.MOVING, True, False)

    assert machine.update(moving_with_route, observed_monotonic=1.0) is NavigationState.MOVING
    assert machine.update(moving_without_route, observed_monotonic=2.0) is NavigationState.MOVING


def test_simple_state_machine_expires_continuous_unknown_after_five_seconds() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    unknown = NavigationObservation(MotionClass.UNKNOWN, None, None)

    assert machine.update(moving, observed_monotonic=10.0) is NavigationState.MOVING
    assert machine.update(unknown, observed_monotonic=10.1) is NavigationState.MOVING
    assert machine.update(unknown, observed_monotonic=15.09) is NavigationState.MOVING
    assert machine.update(unknown, observed_monotonic=15.1) is NavigationState.UNKNOWN


def test_trusted_moving_resets_the_unknown_grace_timer() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    unknown = NavigationObservation(MotionClass.UNKNOWN, None, None)

    assert machine.update(moving, observed_monotonic=1.0) is NavigationState.MOVING
    assert machine.update(unknown, observed_monotonic=2.0) is NavigationState.MOVING
    assert machine.update(moving, observed_monotonic=4.0) is NavigationState.MOVING
    assert machine.update(unknown, observed_monotonic=5.0) is NavigationState.MOVING
    assert machine.update(unknown, observed_monotonic=9.9) is NavigationState.MOVING
    assert machine.update(unknown, observed_monotonic=10.0) is NavigationState.UNKNOWN


def test_no_ship_disarms_departed_until_new_trusted_movement() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    stationary_without_route = NavigationObservation(MotionClass.STATIONARY, True, False)

    assert machine.update(moving, observed_monotonic=1.0) is NavigationState.MOVING
    assert machine.no_ship() is NavigationState.NO_SHIP
    assert machine.update(stationary_without_route, observed_monotonic=2.0) is NavigationState.UNKNOWN
    assert machine.update(stationary_without_route, observed_monotonic=8.0) is NavigationState.UNKNOWN

    assert machine.update(moving, observed_monotonic=9.0) is NavigationState.MOVING
    assert machine.update(stationary_without_route, observed_monotonic=10.0) is NavigationState.MOVING
    assert machine.update(stationary_without_route, observed_monotonic=15.0) is NavigationState.DEPARTED


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), True])
def test_simple_state_machine_rejects_invalid_unknown_grace_seconds(value: object) -> None:
    with pytest.raises(ValueError, match="unknown_grace_seconds"):
        NavigationStateMachine(simple=True, unknown_grace_seconds=value)


def test_simple_state_machine_confirms_departed_after_five_static_seconds_without_route() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    stationary_without_route = NavigationObservation(MotionClass.STATIONARY, True, False)

    assert machine.update(stationary_without_route, observed_monotonic=10.0) is NavigationState.UNKNOWN
    assert machine.update(stationary_without_route, observed_monotonic=14.99) is NavigationState.UNKNOWN
    assert machine.update(stationary_without_route, observed_monotonic=15.0) is NavigationState.DEPARTED


def test_simple_state_machine_confirms_stuck_after_five_static_seconds_with_route() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    stationary_with_route = NavigationObservation(MotionClass.STATIONARY, False, True)

    assert machine.update(stationary_with_route, observed_monotonic=20.0) is NavigationState.UNKNOWN
    assert machine.update(stationary_with_route, observed_monotonic=24.99) is NavigationState.UNKNOWN
    assert machine.update(stationary_with_route, observed_monotonic=25.0) is NavigationState.STUCK


def test_unknown_evidence_resets_terminal_confirmation_window() -> None:
    machine = NavigationStateMachine(simple=True, confirmation_seconds=5.0)
    stationary_with_route = NavigationObservation(MotionClass.STATIONARY, False, True)
    unknown = NavigationObservation(MotionClass.UNKNOWN, None, None)

    assert machine.update(stationary_with_route, observed_monotonic=1.0) is NavigationState.UNKNOWN
    assert machine.update(unknown, observed_monotonic=5.0) is NavigationState.UNKNOWN
    assert machine.update(stationary_with_route, observed_monotonic=6.0) is NavigationState.UNKNOWN
    assert machine.update(stationary_with_route, observed_monotonic=11.0) is NavigationState.STUCK