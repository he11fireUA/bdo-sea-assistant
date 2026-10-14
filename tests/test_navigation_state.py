import pytest

from bdo_assistant.navigation_state import (
    NavigationObservation,
    NavigationState,
    NavigationStateMachine,
)
from bdo_assistant.motion import MotionClass


def _observation(motion, *, route_complete=False, endpoint_ahead=True):
    return NavigationObservation(
        motion=motion,
        route_complete=route_complete,
        endpoint_ahead=endpoint_ahead,
    )


def test_five_second_false_pause_never_becomes_arrived_or_stuck() -> None:
    machine = NavigationStateMachine(arrival_frames=3, stuck_frames=15)

    states = [
        machine.update(_observation(MotionClass.STATIONARY))
        for _ in range(5)
    ]
    resumed = machine.update(_observation(MotionClass.MOVING))

    assert NavigationState.ARRIVED not in states
    assert NavigationState.STUCK not in states
    assert resumed is NavigationState.MOVING


def test_arrival_requires_route_completion_evidence() -> None:
    machine = NavigationStateMachine(arrival_frames=3)
    observation = _observation(
        MotionClass.STATIONARY,
        route_complete=True,
        endpoint_ahead=False,
    )

    assert machine.update(observation) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(observation) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(observation) is NavigationState.ARRIVED


def test_contradictory_complete_and_ahead_route_evidence_cannot_arrive() -> None:
    machine = NavigationStateMachine(arrival_frames=3)
    contradictory = _observation(
        MotionClass.STATIONARY,
        route_complete=True,
        endpoint_ahead=True,
    )

    states = [machine.update(contradictory) for _ in range(5)]

    assert NavigationState.ARRIVED not in states
    assert states[-1] is NavigationState.STATIONARY_CANDIDATE


def test_elapsed_arrival_uses_observed_time_not_the_number_of_updates() -> None:
    machine = NavigationStateMachine(arrival_seconds=2.0)
    arrival = _observation(
        MotionClass.STATIONARY,
        route_complete=True,
        endpoint_ahead=False,
    )

    assert machine.update(arrival, observed_monotonic=10.0) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(arrival, observed_monotonic=10.1) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(arrival, observed_monotonic=11.9) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(arrival, observed_monotonic=12.0) is NavigationState.ARRIVED


def test_moving_without_route_context_is_surfaced_but_cannot_confirm_terminal_state() -> None:
    machine = NavigationStateMachine(arrival_seconds=1.0, stuck_seconds=1.0)
    moving_unknown_route = _observation(
        MotionClass.MOVING,
        route_complete=None,
        endpoint_ahead=None,
    )

    assert machine.update(moving_unknown_route, observed_monotonic=1.0) is NavigationState.MOVING
    assert machine.update(moving_unknown_route, observed_monotonic=10.0) is NavigationState.MOVING
    assert machine.state not in (NavigationState.ARRIVED, NavigationState.STUCK)


def test_arrived_latches_through_bounded_repositioning() -> None:
    machine = NavigationStateMachine(arrival_frames=2, departure_frames=4)
    arrived = _observation(MotionClass.STATIONARY, route_complete=True, endpoint_ahead=False)
    machine.update(arrived)
    assert machine.update(arrived) is NavigationState.ARRIVED

    for motion in (MotionClass.JITTER, MotionClass.UNKNOWN, MotionClass.MOVING):
        state = machine.update(
            _observation(motion, route_complete=True, endpoint_ahead=False)
        )
        assert state is NavigationState.ARRIVED


def test_stuck_requires_persistence_and_sustained_recovery() -> None:
    machine = NavigationStateMachine(stuck_frames=4, recovery_frames=3)
    blocked = _observation(MotionClass.STATIONARY)
    for _ in range(3):
        assert machine.update(blocked) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(blocked) is NavigationState.STUCK

    moving = _observation(MotionClass.MOVING)
    assert machine.update(moving) is NavigationState.RECOVERING
    assert machine.update(_observation(MotionClass.JITTER)) is NavigationState.STUCK
    assert machine.update(moving) is NavigationState.RECOVERING
    assert machine.update(moving) is NavigationState.RECOVERING
    assert machine.update(moving) is NavigationState.MOVING


def test_stuck_recovers_from_trusted_motion_without_route_context() -> None:
    machine = NavigationStateMachine(stuck_frames=2, recovery_frames=2)
    blocked = _observation(MotionClass.STATIONARY, route_complete=False, endpoint_ahead=True)
    machine.update(blocked)
    assert machine.update(blocked) is NavigationState.STUCK

    route_unknown_motion = _observation(
        MotionClass.MOVING,
        route_complete=None,
        endpoint_ahead=None,
    )
    assert machine.update(route_unknown_motion) is NavigationState.RECOVERING
    assert machine.update(route_unknown_motion) is NavigationState.MOVING


@pytest.mark.parametrize("value", [0, 1, -1, 1.5, float("nan"), True])
@pytest.mark.parametrize(
    "field",
    ["arrival_frames", "stuck_frames", "recovery_frames", "departure_frames"],
)
def test_navigation_rejects_invalid_frame_thresholds(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        NavigationStateMachine(**{field: value})


def test_suspend_discards_unconfirmed_arrival_evidence() -> None:
    machine = NavigationStateMachine(arrival_frames=3)
    arrived = _observation(
        MotionClass.STATIONARY,
        route_complete=True,
        endpoint_ahead=False,
    )
    assert machine.update(arrived) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(arrived) is NavigationState.STATIONARY_CANDIDATE

    assert machine.suspend() is NavigationState.UNKNOWN
    assert machine.update(arrived) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(arrived) is NavigationState.STATIONARY_CANDIDATE
    assert machine.update(arrived) is NavigationState.ARRIVED


@pytest.mark.parametrize(
    ("latched", "route_complete", "endpoint_ahead"),
    [
        (NavigationState.ARRIVED, True, False),
        (NavigationState.STUCK, False, True),
    ],
)
def test_suspend_preserves_confirmed_latched_state(
    latched: NavigationState,
    route_complete: bool,
    endpoint_ahead: bool,
) -> None:
    machine = NavigationStateMachine(arrival_frames=2, stuck_frames=2)
    observation = _observation(
        MotionClass.STATIONARY,
        route_complete=route_complete,
        endpoint_ahead=endpoint_ahead,
    )
    machine.update(observation)
    assert machine.update(observation) is latched

    assert machine.suspend() is latched


@pytest.mark.parametrize(
    ("motion", "route_complete", "endpoint_ahead"),
    [
        ("stationary", True, False),
        (MotionClass.STATIONARY, 1, False),
        (MotionClass.STATIONARY, True, 0),
    ],
)
def test_navigation_observation_rejects_malformed_fields(
    motion: object,
    route_complete: object,
    endpoint_ahead: object,
) -> None:
    with pytest.raises(ValueError, match="NavigationObservation"):
        NavigationObservation(
            motion=motion,
            route_complete=route_complete,
            endpoint_ahead=endpoint_ahead,
        )


def test_navigation_state_machine_rejects_non_observation_input() -> None:
    machine = NavigationStateMachine()

    with pytest.raises(ValueError, match="NavigationObservation"):
        machine.update(object())


def test_invalid_direct_input_clears_pending_persistence() -> None:
    machine = NavigationStateMachine(arrival_frames=2)
    arrival = _observation(MotionClass.STATIONARY, route_complete=True, endpoint_ahead=False)
    assert machine.update(arrival) is NavigationState.STATIONARY_CANDIDATE

    with pytest.raises(ValueError, match="NavigationObservation"):
        machine.update(object())

    assert machine.update(arrival) is NavigationState.STATIONARY_CANDIDATE


def test_incomplete_motion_observation_begins_stuck_recovery() -> None:
    machine = NavigationStateMachine(stuck_frames=2, recovery_frames=2)
    blocked = _observation(MotionClass.STATIONARY)
    machine.update(blocked)
    assert machine.update(blocked) is NavigationState.STUCK
    incomplete = NavigationObservation(
        motion=MotionClass.MOVING,
        route_complete=None,
        endpoint_ahead=None,
    )

    assert machine.update(incomplete) is NavigationState.RECOVERING


def test_navigation_state_machine_revalidates_corrupted_exact_observation() -> None:
    machine = NavigationStateMachine(arrival_frames=2)
    corrupted = NavigationObservation(
        motion=MotionClass.STATIONARY,
        route_complete=True,
        endpoint_ahead=False,
    )
    object.__setattr__(corrupted, "route_complete", 1)

    with pytest.raises(ValueError, match="NavigationObservation"):
        machine.update(corrupted)

    assert machine.state is NavigationState.UNKNOWN
