from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.minimap_observability import (
    MinimapVisibilityEvidence,
    MinimapVisibilityTracker,
)
from bdo_assistant.motion import MotionClass
from bdo_assistant.navigation_observability import (
    MinimapNavigationMonitor,
    NavigationObservabilityGate,
)
from bdo_assistant.navigation_state import (
    NavigationObservation,
    NavigationState,
    NavigationStateMachine,
)
from bdo_assistant.notifications import EventType


FIXTURES = Path(__file__).parent / "fixtures"


def _arrival_observation() -> NavigationObservation:
    return NavigationObservation(
        motion=MotionClass.STATIONARY,
        route_complete=True,
        endpoint_ahead=False,
    )


def _evidence(is_visible: bool) -> MinimapVisibilityEvidence:
    return MinimapVisibilityEvidence(
        is_visible=is_visible,
        confidence=0.95 if is_visible else 0.10,
    )


def test_visibility_gate_suspends_navigation_and_restarts_fresh_after_restore() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )
    arrival = _arrival_observation()

    first = gate.update(_evidence(True), arrival)
    confirmed = gate.update(_evidence(True), arrival)
    pending_loss = gate.update(_evidence(False), arrival)
    lost = gate.update(_evidence(False), arrival)
    pending_restore = gate.update(_evidence(True), arrival)
    restored = gate.update(_evidence(True), arrival)
    arrived = gate.update(_evidence(True), arrival)

    assert first.navigation_state is NavigationState.UNKNOWN
    assert not first.navigation_updated
    assert confirmed.navigation_state is NavigationState.STATIONARY_CANDIDATE
    assert confirmed.navigation_updated
    assert pending_loss.navigation_state is NavigationState.UNKNOWN
    assert not pending_loss.navigation_updated
    assert pending_loss.event is None
    assert lost.event is EventType.MINIMAP_VISIBILITY_LOST
    assert lost.navigation_state is NavigationState.UNKNOWN
    assert pending_restore.navigation_state is NavigationState.UNKNOWN
    assert not pending_restore.navigation_updated
    assert restored.event is EventType.MINIMAP_VISIBILITY_RESTORED
    assert restored.navigation_state is NavigationState.STATIONARY_CANDIDATE
    assert restored.navigation_updated
    assert arrived.navigation_state is NavigationState.ARRIVED


def test_simple_navigation_suspends_before_terminal_verdict_when_minimap_is_not_visible() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(simple=True, confirmation_seconds=5.0),
    )
    arrival = _arrival_observation()

    pending = gate.update(_evidence(False), arrival, observed_monotonic=1.0)
    arrived = gate.update(_evidence(False), arrival, observed_monotonic=6.0)

    assert not pending.navigation_updated
    assert pending.navigation_state is NavigationState.UNKNOWN
    assert not arrived.navigation_updated
    assert arrived.navigation_state is NavigationState.UNKNOWN
    assert arrived.event is EventType.MINIMAP_VISIBILITY_LOST


def test_visibility_gate_allows_stuck_recovery_with_unknown_route() -> None:
    navigation = NavigationStateMachine(stuck_frames=2, recovery_frames=2)
    blocked = NavigationObservation(MotionClass.STATIONARY, False, True)
    navigation.update(blocked)
    assert navigation.update(blocked) is NavigationState.STUCK
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2), navigation
    )
    moving_unknown_route = NavigationObservation(MotionClass.MOVING, None, None)

    gate.update(_evidence(True), moving_unknown_route)
    recovering = gate.update(_evidence(True), moving_unknown_route)
    moved = gate.update(_evidence(True), moving_unknown_route)

    assert recovering.navigation_state is NavigationState.RECOVERING
    assert recovering.navigation_updated
    assert moved.navigation_state is NavigationState.MOVING
    assert moved.navigation_updated


def test_visibility_gate_surfaces_moving_with_unknown_route_without_terminal_evidence() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_seconds=1.0, stuck_seconds=1.0),
    )
    moving_unknown_route = NavigationObservation(MotionClass.MOVING, None, None)

    gate.update(_evidence(True), moving_unknown_route, observed_monotonic=1.0)
    result = gate.update(_evidence(True), moving_unknown_route, observed_monotonic=3.0)

    assert result.navigation_updated
    assert result.navigation_state is NavigationState.MOVING


def test_visible_frame_without_navigation_observation_stays_suspended() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )

    first_missing = gate.update(_evidence(True), None)
    missing = gate.update(_evidence(True), None)
    first_complete = gate.update(_evidence(True), _arrival_observation())

    assert first_missing.navigation_state is NavigationState.UNKNOWN
    assert not first_missing.navigation_updated
    assert missing.navigation_state is NavigationState.UNKNOWN
    assert not missing.navigation_updated
    assert first_complete.navigation_state is NavigationState.STATIONARY_CANDIDATE
    assert first_complete.navigation_updated


def test_visibility_gate_rejects_forged_duck_typed_evidence() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )
    forged = type("ForgedEvidence", (), {"is_visible": True})()

    with pytest.raises(ValueError, match="MinimapVisibilityEvidence"):
        gate.update(forged, _arrival_observation())

    assert gate.visibility.state.value == "unknown"
    assert gate.navigation.state is NavigationState.UNKNOWN


def test_invalid_evidence_clears_pending_navigation_persistence() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )
    arrival = _arrival_observation()
    gate.update(_evidence(True), arrival)
    assert gate.update(_evidence(True), arrival).navigation_state is NavigationState.STATIONARY_CANDIDATE

    with pytest.raises(ValueError, match="MinimapVisibilityEvidence"):
        gate.update(object(), arrival)

    resumed = gate.update(_evidence(True), arrival)
    assert resumed.navigation_state is NavigationState.STATIONARY_CANDIDATE


def test_invalid_evidence_resets_pending_visibility_restoration() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )
    hidden = _evidence(False)
    visible = _evidence(True)
    gate.update(hidden)
    assert gate.update(hidden).event is EventType.MINIMAP_VISIBILITY_LOST
    assert gate.update(visible).event is None

    with pytest.raises(ValueError, match="MinimapVisibilityEvidence"):
        gate.update(object())

    assert gate.update(visible).event is None
    restored = gate.update(visible)
    assert restored.event is EventType.MINIMAP_VISIBILITY_RESTORED


def test_visibility_gate_rejects_subclass_forged_evidence() -> None:
    class ForgedEvidence(MinimapVisibilityEvidence):
        pass

    forged = object.__new__(ForgedEvidence)
    object.__setattr__(forged, "is_visible", True)
    object.__setattr__(forged, "confidence", float("nan"))
    object.__setattr__(forged, "minimum_confidence", 0.50)
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )

    with pytest.raises(ValueError, match="exact MinimapVisibilityEvidence"):
        gate.update(forged, _arrival_observation())

    assert gate.visibility.state.value == "unknown"
    assert gate.navigation.state is NavigationState.UNKNOWN


def test_visibility_gate_revalidates_exact_evidence_before_state_mutation() -> None:
    corrupted = MinimapVisibilityEvidence(
        is_visible=True,
        confidence=0.95,
        minimum_confidence=0.60,
    )
    object.__setattr__(corrupted, "confidence", float("nan"))
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )

    with pytest.raises(ValueError, match="visibility evidence"):
        gate.update(corrupted, _arrival_observation())

    assert gate.visibility.state.value == "unknown"
    assert gate.navigation.state is NavigationState.UNKNOWN


def test_visibility_gate_cannot_shadow_evidence_validator() -> None:
    corrupted = MinimapVisibilityEvidence(
        is_visible=True,
        confidence=0.95,
        minimum_confidence=0.60,
    )
    object.__setattr__(corrupted, "confidence", float("nan"))
    object.__setattr__(corrupted, "validate", lambda: None)
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )

    with pytest.raises(ValueError, match="visibility evidence"):
        gate.update(corrupted, _arrival_observation())

    assert gate.visibility.state.value == "unknown"


def test_visibility_gate_suspends_incomplete_navigation_bundle() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )
    incomplete = NavigationObservation(
        motion=MotionClass.STATIONARY,
        route_complete=None,
        endpoint_ahead=False,
    )

    startup = gate.update(_evidence(True), _arrival_observation())
    missing_route = gate.update(_evidence(True), incomplete)
    malformed = gate.update(_evidence(True), object())
    first_complete = gate.update(_evidence(True), _arrival_observation())

    assert startup.navigation_state is NavigationState.UNKNOWN
    assert not missing_route.navigation_updated
    assert missing_route.navigation_state is NavigationState.UNKNOWN
    assert not malformed.navigation_updated
    assert malformed.navigation_state is NavigationState.UNKNOWN
    assert first_complete.navigation_state is NavigationState.STATIONARY_CANDIDATE


def test_visibility_gate_suspends_corrupted_exact_navigation_observation() -> None:
    gate = NavigationObservabilityGate(
        MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        NavigationStateMachine(arrival_frames=2),
    )
    corrupted = _arrival_observation()
    object.__setattr__(corrupted, "motion", "stationary")

    first = gate.update(_evidence(True), corrupted)
    second = gate.update(_evidence(True), corrupted)
    complete = gate.update(_evidence(True), _arrival_observation())

    assert not first.navigation_updated
    assert not second.navigation_updated
    assert second.navigation_state is NavigationState.UNKNOWN
    assert complete.navigation_state is NavigationState.STATIONARY_CANDIDATE


def test_pixel_monitor_replays_visibility_loss_and_fresh_navigation_restore() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    hidden = np.zeros_like(minimap)
    monitor = MinimapNavigationMonitor(
        anchor_template=anchor,
        expected_scale=1.0,
        visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        navigation=NavigationStateMachine(arrival_frames=2),
    )
    arrival = _arrival_observation()

    results = [
        monitor.update(frame, arrival)
        for frame in (minimap, minimap, hidden, hidden, minimap, minimap, minimap)
    ]

    assert [result.event for result in results] == [
        None,
        None,
        None,
        EventType.MINIMAP_VISIBILITY_LOST,
        None,
        EventType.MINIMAP_VISIBILITY_RESTORED,
        None,
    ]
    assert results[3].navigation_state is NavigationState.UNKNOWN
    assert results[5].navigation_state is NavigationState.STATIONARY_CANDIDATE
    assert results[6].navigation_state is NavigationState.ARRIVED
    assert results[0].evidence.is_visible
    assert not results[2].evidence.is_visible


def test_pixel_monitor_suspends_before_rejecting_malformed_frame() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    monitor = MinimapNavigationMonitor(
        anchor_template=anchor,
        expected_scale=1.0,
        visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        navigation=NavigationStateMachine(arrival_frames=2),
    )
    arrival = _arrival_observation()
    monitor.update(minimap, arrival)
    assert monitor.update(minimap, arrival).navigation_state is NavigationState.STATIONARY_CANDIDATE

    with pytest.raises(ValueError, match="non-empty BGR"):
        monitor.update(np.zeros((0, 10, 3), dtype=np.uint8), arrival)

    resumed = monitor.update(minimap, arrival)
    assert resumed.navigation_state is NavigationState.STATIONARY_CANDIDATE


def test_pixel_monitor_owns_private_immutable_anchor_copy() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    monitor = MinimapNavigationMonitor(
        anchor_template=anchor,
        expected_scale=1.0,
        visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
        navigation=NavigationStateMachine(arrival_frames=2),
    )
    anchor[:] = 0

    assert not hasattr(monitor, "anchor_template")
    assert monitor.update(minimap, _arrival_observation()).evidence.is_visible


@pytest.mark.parametrize(
    ("anchor_factory", "expected_scale", "minimum_confidence"),
    [
        (lambda: "not-an-array", 1.0, 0.60),
        (lambda: np.zeros((1, 2, 3), dtype=np.uint8), 1.0, 0.60),
        (lambda: cv2.imread(str(FIXTURES / "minimap_2d_anchor.png")), float("nan"), 0.60),
        (lambda: cv2.imread(str(FIXTURES / "minimap_2d_anchor.png")), 1.0, 0.49),
    ],
)
def test_pixel_monitor_rejects_invalid_constructor_inputs(
    anchor_factory,
    expected_scale: float,
    minimum_confidence: float,
) -> None:
    with pytest.raises(ValueError):
        MinimapNavigationMonitor(
            anchor_template=anchor_factory(),
            expected_scale=expected_scale,
            minimum_confidence=minimum_confidence,
            visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
            navigation=NavigationStateMachine(arrival_frames=2),
        )


@pytest.mark.parametrize(
    ("visibility", "navigation"),
    [
        (object(), NavigationStateMachine()),
        (MinimapVisibilityTracker(), object()),
    ],
)
def test_monitor_and_gate_reject_invalid_dependencies(visibility, navigation) -> None:
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    with pytest.raises(ValueError, match="dependencies"):
        NavigationObservabilityGate(visibility, navigation)
    with pytest.raises(ValueError, match="dependencies"):
        MinimapNavigationMonitor(
            anchor_template=anchor,
            expected_scale=1.0,
            visibility=visibility,
            navigation=navigation,
        )


def test_monitor_and_gate_configuration_is_read_only() -> None:
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    visibility = MinimapVisibilityTracker()
    navigation = NavigationStateMachine()
    gate = NavigationObservabilityGate(visibility, navigation)
    monitor = MinimapNavigationMonitor(
        anchor_template=anchor,
        expected_scale=1.0,
        visibility=visibility,
        navigation=navigation,
    )

    for target, name, value in (
        (visibility, "lost_frames", 1),
        (navigation, "arrival_frames", 1),
        (gate, "visibility", object()),
        (gate, "navigation", object()),
        (monitor, "expected_scale", 0.60),
        (monitor, "minimum_confidence", 0.50),
        (monitor, "gate", object()),
    ):
        with pytest.raises(AttributeError):
            setattr(target, name, value)
