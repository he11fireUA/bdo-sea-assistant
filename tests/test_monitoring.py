from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.minimap_observability import MinimapVisibilityTracker
from bdo_assistant.monitoring import AlphaMonitoringLoop, MonitoringSample
from bdo_assistant.motion import MotionClass
from bdo_assistant.navigation_observability import MinimapNavigationMonitor
from bdo_assistant.navigation_state import NavigationObservation, NavigationState, NavigationStateMachine
from bdo_assistant.notifications import EventType
from bdo_assistant.resource_bars import ThresholdWarningTracker
from bdo_assistant.runtime import CalibratedFrameSampleBuilder


FIXTURES = Path(__file__).parent / "fixtures"


def _arrival_observation() -> NavigationObservation:
    return NavigationObservation(
        motion=MotionClass.STATIONARY,
        route_complete=True,
        endpoint_ahead=False,
    )


def _loop() -> AlphaMonitoringLoop:
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    return AlphaMonitoringLoop(
        minimap=MinimapNavigationMonitor(
            anchor_template=anchor,
            expected_scale=1.0,
            visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
            navigation=NavigationStateMachine(arrival_frames=2),
        ),
        capture_lost_frames=2,
        capture_restored_frames=2,
    )


def _simple_loop() -> AlphaMonitoringLoop:
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    return AlphaMonitoringLoop(
        minimap=MinimapNavigationMonitor(
            anchor_template=anchor,
            expected_scale=1.0,
            visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
            navigation=NavigationStateMachine(simple=True, confirmation_seconds=5.0),
        ),
        capture_lost_frames=2,
        capture_restored_frames=2,
    )


def test_sequence_only_monitoring_sample_uses_documented_synthetic_time() -> None:
    sample = MonitoringSample(7, True, True)

    assert sample.source_sequence == 7
    assert sample.processing_ordinal == 7
    assert sample.received_monotonic == 7.0
    assert sample.native_source_timestamp is None
    assert sample.calibration_epoch == 0


def test_simple_navigation_uses_first_valid_wgc_sample_before_capture_health_confirms() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _simple_loop()
    arrival = _arrival_observation()

    pending = loop.process(
        MonitoringSample(1, True, True, minimap, arrival, received_monotonic=1.0)
    )
    arrived = loop.process(
        MonitoringSample(2, True, True, minimap, arrival, received_monotonic=6.0)
    )

    assert pending.navigation_state is NavigationState.UNKNOWN
    assert arrived.navigation_state is NavigationState.ARRIVED


def test_repeat_arrival_alert_repeats_only_after_configured_interval() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = AlphaMonitoringLoop(
        minimap=_simple_loop().minimap,
        repeat_alerts_enabled=True,
        repeat_alert_interval_seconds=30,
    )
    arrival = _arrival_observation()

    pending = loop.process(MonitoringSample(1, True, True, minimap, arrival, received_monotonic=1.0))
    initial = loop.process(MonitoringSample(2, True, True, minimap, arrival, received_monotonic=6.0))
    early = loop.process(MonitoringSample(3, True, True, minimap, arrival, received_monotonic=35.0))
    repeated = loop.process(MonitoringSample(4, True, True, minimap, arrival, received_monotonic=36.0))

    assert pending.events == ()
    assert [event.event_type for event in initial.events] == [EventType.ARRIVED]
    assert early.events == ()
    assert [event.event_type for event in repeated.events] == [EventType.ARRIVED]


REPEAT_ALERTS = (EventType.ARRIVED, EventType.STUCK, EventType.SHIP_HP_LOW, EventType.RATIONS_LOW)


def _repeat_scenario(event_type):
    loop = _simple_loop()
    loop.configure_repeat_alerts(True, 30)
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    observation = {
        EventType.ARRIVED: _arrival_observation(),
        EventType.STUCK: NavigationObservation(MotionClass.STATIONARY, False, True),
    }.get(event_type, NavigationObservation(MotionClass.MOVING, False, True))
    sequence = 0

    def sample(now, **overrides):
        nonlocal sequence
        sequence += 1
        values = dict(capture_ok=True, calibration_ok=True, minimap_roi=minimap,
                      navigation_observation=observation,
                      hp_percent=0 if event_type is EventType.SHIP_HP_LOW else 100,
                      rations_percent=0 if event_type is EventType.RATIONS_LOW else 100,
                      resource_bars_visible=True, received_monotonic=float(now))
        values.update(overrides)
        return loop.process(MonitoringSample(sequence, **values))

    initial = [sample(now) for now in (1, 2, 6)]
    assert sum(event.event_type is event_type for result in initial for event in result.events) == 1
    return loop, sample, minimap


@pytest.mark.parametrize("event_type", REPEAT_ALERTS)
@pytest.mark.parametrize("change", ["apply", "enable", "disable", "interval"])
def test_repeat_configuration_rearms_active_condition_on_next_sample(event_type, change):
    loop, sample, _ = _repeat_scenario(event_type)
    if change == "enable":
        loop.configure_repeat_alerts(False, 30)
        assert event_type not in [event.event_type for event in sample(40).events]
    interval = 60 if change == "interval" else 30
    loop.configure_repeat_alerts(change != "disable", interval)
    assert event_type not in [event.event_type for event in sample(100).events]
    assert event_type not in [event.event_type for event in sample(100 + interval - 1).events]
    due = sample(100 + interval)
    assert (event_type in [event.event_type for event in due.events]) is (change != "disable")


@pytest.mark.parametrize("event_type", REPEAT_ALERTS)
@pytest.mark.parametrize("loss", ["capture", "closed", "calibration", "visibility", "detector"])
def test_active_alert_reacquisition_requires_fresh_confirmation(event_type, loss):
    loop, sample, minimap = _repeat_scenario(event_type)
    overrides = {
        "capture": dict(capture_ok=False),
        "calibration": dict(calibration_ok=False),
        "visibility": dict(minimap_roi=np.zeros_like(minimap)),
        "detector": dict(minimap_roi=np.zeros((0, 10, 3), dtype=np.uint8)),
    }
    if loss == "closed":
        lost = loop.capture_closed(4)
        # Advance the scenario's sequence past capture_closed.
        with pytest.raises(ValueError, match="strictly increase"):
            sample(40)
    else:
        lost = sample(40, **overrides[loss])
        again = sample(70, **overrides[loss])
        assert not any(event.event_type in REPEAT_ALERTS for event in again.events)
    assert not any(event.event_type in REPEAT_ALERTS for event in lost.events)
    assert lost.navigation_state is NavigationState.UNKNOWN
    assert event_type not in [event.event_type for event in sample(100).events]
    assert event_type not in [event.event_type for event in sample(104).events]
    confirmed = sample(105)
    assert [event.event_type for event in confirmed.events].count(event_type) == 1
    assert event_type not in [event.event_type for event in sample(134).events]
    assert [event.event_type for event in sample(135).events].count(event_type) == 1


@pytest.mark.parametrize("event_type,field", [(EventType.SHIP_HP_LOW, "hp_percent"),
                                             (EventType.RATIONS_LOW, "rations_percent")])
@pytest.mark.parametrize("loss", ["missing", "hud"])
def test_resource_repeat_requires_fresh_readings_after_loss(event_type, field, loss):
    _, sample, _ = _repeat_scenario(event_type)
    overrides = {field: None} if loss == "missing" else {"resource_bars_visible": False}
    for now in (36, 70):
        assert event_type not in [event.event_type for event in sample(now, **overrides).events]
    assert event_type not in [event.event_type for event in sample(100).events]
    assert event_type not in [event.event_type for event in sample(101).events]
    assert [event.event_type for event in sample(102).events].count(event_type) == 1
    assert event_type not in [event.event_type for event in sample(131).events]
    assert [event.event_type for event in sample(132).events].count(event_type) == 1


def test_capture_loss_suspends_navigation_and_restores_with_fresh_persistence() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    arrival = _arrival_observation()

    startup = [
        loop.process(MonitoringSample(i, True, True, minimap, arrival))
        for i in range(1, 4)
    ]
    pending_loss = loop.process(MonitoringSample(4, False, False))
    lost = loop.process(MonitoringSample(5, False, False))
    repeated_loss = loop.process(MonitoringSample(6, False, False))
    pending_restore = loop.process(MonitoringSample(7, True, True, minimap, arrival))
    restored = loop.process(MonitoringSample(8, True, True, minimap, arrival))
    arrived = loop.process(MonitoringSample(9, True, True, minimap, arrival))

    assert startup[-1].navigation_state is NavigationState.STATIONARY_CANDIDATE
    assert pending_loss.navigation_state is NavigationState.UNKNOWN
    assert pending_loss.events == ()
    assert [event.event_type for event in lost.events] == [EventType.CAPTURE_LOST]
    assert repeated_loss.events == ()
    assert pending_restore.events == ()
    assert [event.event_type for event in restored.events] == [EventType.CAPTURE_RESTORED]
    assert restored.navigation_state is NavigationState.STATIONARY_CANDIDATE
    assert [event.event_type for event in arrived.events] == [EventType.ARRIVED]
    assert arrived.navigation_state is NavigationState.ARRIVED


def test_calibration_loss_emits_once_and_restarts_navigation_persistence() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    arrival = _arrival_observation()
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, arrival))

    pending = loop.process(MonitoringSample(4, True, False))
    lost = loop.process(MonitoringSample(5, True, False))
    repeated = loop.process(MonitoringSample(6, True, False))
    recovered = loop.process(MonitoringSample(7, True, True, minimap, arrival))
    arrived = loop.process(MonitoringSample(8, True, True, minimap, arrival))

    assert pending.events == ()
    assert [event.event_type for event in lost.events] == [EventType.UI_CALIBRATION_LOST]
    assert repeated.events == ()
    assert recovered.navigation_state is NavigationState.STATIONARY_CANDIDATE
    assert [event.event_type for event in arrived.events] == [EventType.ARRIVED]


def test_resource_warnings_confirm_once_and_rearm_after_refill() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    loop = AlphaMonitoringLoop(
        minimap=MinimapNavigationMonitor(
            anchor_template=anchor,
            expected_scale=1.0,
            visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
            navigation=NavigationStateMachine(arrival_frames=2),
        ),
        hp_warning=ThresholdWarningTracker(
            warning_below=60,
            reset_at=65,
            confirmation_frames=3,
        ),
        rations_warning=ThresholdWarningTracker(
            warning_below=10,
            reset_at=15,
            confirmation_frames=3,
        ),
    )
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    sequence = 0

    def cycle(hp: float, rations: float):
        nonlocal sequence
        sequence += 1
        return loop.process(
            MonitoringSample(sequence, True, True, minimap, moving, hp, rations)
        )

    cycle(80, 20)
    cycle(80, 20)
    cycle(80, 20)
    low = [cycle(59, 9), cycle(58, 8), cycle(57, 7)]
    repeated = cycle(56, 6)
    cycle(80, 20)
    rearmed = [cycle(59, 9), cycle(58, 8), cycle(57, 7)]

    assert [event.event_type for event in low[-1].events] == [
        EventType.SHIP_HP_LOW,
        EventType.RATIONS_LOW,
    ]
    assert repeated.events == ()
    assert [event.event_type for event in rearmed[-1].events] == [
        EventType.SHIP_HP_LOW,
        EventType.RATIONS_LOW,
    ]


def test_structured_outlined_non_ship_background_is_not_an_observable_empty_hud_or_alert() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    frame = np.full((340, 400, 3), (25, 25, 25), dtype=np.uint8)
    frame[: minimap.shape[0], : minimap.shape[1]] = minimap
    layout = {
        "schema_version": 1,
        "regions": {
            "minimap": {"normalized": {"x": 0.0, "y": 0.0, "width": 0.815, "height": 0.882}},
            "ship_hp_bar": {"normalized": {"x": 0.025, "y": 0.912, "width": 0.5, "height": 0.029}},
            "ship_rations_bar": {"normalized": {"x": 0.025, "y": 0.956, "width": 0.5, "height": 0.029}},
        },
    }
    # A shore/menu-like outlined panel at the calibrated locations used to satisfy
    # generic neutral coverage plus contrast-boundary checks as two empty bars.
    for y in (310, 325):
        frame[y - 1 : y + 11, 9] = 0
        frame[y - 1 : y + 11, 210] = 0
        frame[y - 1, 9:211] = 0
        frame[y + 10, 9:211] = 0
    builder = CalibratedFrameSampleBuilder(
        layout,
        lambda sequence, roi: NavigationObservation(MotionClass.MOVING, False, True),
    )
    loop = _loop()

    samples = [builder.build(sequence, frame) for sequence in range(1, 4)]
    results = [loop.process(sample) for sample in samples]

    assert all(sample.resource_bars_visible is False for sample in samples)
    assert all(sample.hp_percent is None and sample.rations_percent is None for sample in samples)
    assert all(
        event.event_type not in (EventType.SHIP_HP_LOW, EventType.RATIONS_LOW)
        for result in results
        for event in result.events
    )


def test_rations_only_candidate_stays_absent_and_never_alerts() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    frame = np.zeros((340, 400, 3), dtype=np.uint8)
    frame[: minimap.shape[0], : minimap.shape[1]] = minimap
    frame[310:320, 10:210] = (25, 25, 25)
    frame[325:335, 10:210] = (25, 25, 25)
    frame[327:333, 10:26] = (0, 210, 220)  # 8% Rations candidate; no rail or HP anchor
    layout = {
        "schema_version": 1,
        "regions": {
            "minimap": {"normalized": {"x": 0.0, "y": 0.0, "width": 0.815, "height": 0.882}},
            "ship_hp_bar": {"normalized": {"x": 0.025, "y": 0.912, "width": 0.5, "height": 0.029}},
            "ship_rations_bar": {"normalized": {"x": 0.025, "y": 0.956, "width": 0.5, "height": 0.029}},
        },
    }
    builder = CalibratedFrameSampleBuilder(
        layout,
        lambda sequence, roi: NavigationObservation(MotionClass.MOVING, False, True),
    )
    loop = _loop()

    samples = [builder.build(sequence, frame) for sequence in range(1, 4)]
    results = [loop.process(sample) for sample in samples]

    assert all(sample.resource_bars_visible is False for sample in samples)
    assert all(sample.hp_percent is None and sample.rations_percent is None for sample in samples)
    assert all(
        event.event_type not in (EventType.SHIP_HP_LOW, EventType.RATIONS_LOW)
        for result in results
        for event in result.events
    )


def test_absent_ship_hud_suppresses_low_resource_alerts_with_visible_minimap() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    for sequence in range(1, 4):
        loop.process(
            MonitoringSample(sequence, True, True, minimap, moving, 80, 20, True)
        )

    results = [
        loop.process(
            MonitoringSample(sequence, True, True, minimap, moving, 0, 0, False)
        )
        for sequence in range(4, 8)
    ]

    assert all(result.minimap_visible is True for result in results)
    assert all(
        event.event_type not in (EventType.SHIP_HP_LOW, EventType.RATIONS_LOW)
        for result in results
        for event in result.events
    )

    reboarded = [
        loop.process(
            MonitoringSample(sequence, True, True, minimap, moving, 0, 0, True)
        )
        for sequence in range(8, 11)
    ]
    assert [event.event_type for event in reboarded[-1].events] == [
        EventType.SHIP_HP_LOW,
        EventType.RATIONS_LOW,
    ]


def test_absent_ship_hud_rearms_prior_resource_alerts_after_reboarding() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 80, 20, True))
    low = [
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 0, 0, True))
        for sequence in range(4, 7)
    ]
    assert [event.event_type for event in low[-1].events] == [
        EventType.SHIP_HP_LOW,
        EventType.RATIONS_LOW,
    ]

    loop.process(MonitoringSample(7, True, True, minimap, moving, 0, 0, False))
    reboarded = [
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 0, 0, True))
        for sequence in range(8, 11)
    ]

    assert [event.event_type for event in reboarded[-1].events] == [
        EventType.SHIP_HP_LOW,
        EventType.RATIONS_LOW,
    ]


def test_live_warning_zone_change_applies_without_restarting_monitoring() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 80, 49, True))

    before = [
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 80, 49, True))
        for sequence in range(4, 7)
    ]
    assert all(
        EventType.RATIONS_LOW not in [event.event_type for event in result.events]
        for result in before
    )

    loop.configure_resource_warning_zones(60, 50)
    after = [
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 80, 49, True))
        for sequence in range(7, 10)
    ]

    assert [event.event_type for event in after[-1].events] == [EventType.RATIONS_LOW]


def test_hidden_minimap_suppresses_false_resource_warnings() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    hidden = np.zeros_like(minimap)
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 80, 20))

    hidden_results = [
        loop.process(MonitoringSample(sequence, True, True, hidden, moving, 0, 0))
        for sequence in range(4, 8)
    ]

    event_types = [
        event.event_type
        for result in hidden_results
        for event in result.events
    ]
    assert event_types == [EventType.MINIMAP_VISIBILITY_LOST]


def test_detector_failure_isolated_into_diagnostics_and_calibration_loss() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    arrival = _arrival_observation()
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, arrival))
    malformed = np.zeros((0, 10, 3), dtype=np.uint8)

    first = loop.process(MonitoringSample(4, True, True, malformed, arrival))
    second = loop.process(MonitoringSample(5, True, True, malformed, arrival))
    recovered = loop.process(MonitoringSample(6, True, True, minimap, arrival))

    assert first.navigation_state is NavigationState.UNKNOWN
    assert first.events == ()
    assert first.diagnostics and "minimap detector" in first.diagnostics[0]
    assert [event.event_type for event in second.events] == [
        EventType.UI_CALIBRATION_LOST
    ]
    assert recovered.navigation_state is NavigationState.STATIONARY_CANDIDATE


def test_sensor_gap_resets_pending_resource_confirmation() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 80, 20))
    loop.process(MonitoringSample(4, True, True, minimap, moving, 59, 20))
    loop.process(MonitoringSample(5, True, True, minimap, moving, 58, 20))
    loop.process(MonitoringSample(6, True, False))

    first = loop.process(MonitoringSample(7, True, True, minimap, moving, 57, 20))
    second = loop.process(MonitoringSample(8, True, True, minimap, moving, 56, 20))
    third = loop.process(MonitoringSample(9, True, True, minimap, moving, 55, 20))

    assert EventType.SHIP_HP_LOW not in [event.event_type for event in first.events]
    assert EventType.SHIP_HP_LOW not in [event.event_type for event in second.events]
    assert EventType.SHIP_HP_LOW in [event.event_type for event in third.events]


def test_duplicate_sequence_is_rejected_before_state_mutation() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    arrival = _arrival_observation()
    loop.process(MonitoringSample(1, True, True, minimap, arrival))

    with pytest.raises(ValueError, match="strictly increase"):
        loop.process(MonitoringSample(1, True, True, minimap, arrival))

    second = loop.process(MonitoringSample(2, True, True, minimap, arrival))
    assert second.navigation_state is NavigationState.UNKNOWN


def test_missing_resource_readings_reset_pending_confirmations() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, moving, 80, 20))
    loop.process(MonitoringSample(4, True, True, minimap, moving, 59, 9))
    loop.process(MonitoringSample(5, True, True, minimap, moving, 58, 8))
    loop.process(MonitoringSample(6, True, True, minimap, moving, None, None))

    first = loop.process(MonitoringSample(7, True, True, minimap, moving, 57, 7))
    second = loop.process(MonitoringSample(8, True, True, minimap, moving, 56, 6))
    third = loop.process(MonitoringSample(9, True, True, minimap, moving, 55, 5))

    assert first.events == ()
    assert second.events == ()
    assert [event.event_type for event in third.events] == [
        EventType.SHIP_HP_LOW,
        EventType.RATIONS_LOW,
    ]


def test_capture_gap_resets_pending_calibration_loss() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    loop.process(MonitoringSample(1, True, True, minimap, moving))
    loop.process(MonitoringSample(2, True, True, minimap, moving))
    assert loop.process(MonitoringSample(3, True, False)).events == ()
    loop.process(MonitoringSample(4, False, False))

    first = loop.process(MonitoringSample(5, True, False))
    second = loop.process(MonitoringSample(6, True, False))

    assert first.events == ()
    assert [event.event_type for event in second.events] == [
        EventType.UI_CALIBRATION_LOST
    ]


@pytest.mark.parametrize("field", ["hp_warning", "rations_warning"])
@pytest.mark.parametrize("value", [False, 0])
def test_loop_rejects_falsy_invalid_resource_tracker(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match="resource warning trackers"):
        AlphaMonitoringLoop(minimap=_loop().minimap, **kwargs)


@pytest.mark.parametrize("field", ["hp_percent", "rations_percent"])
def test_boolean_resource_percent_rejected_before_sequence_mutation(field: str) -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    values = {field: False}

    with pytest.raises(ValueError, match="resource percentages"):
        loop.process(MonitoringSample(1, True, True, minimap, _arrival_observation(), **values))

    accepted = loop.process(MonitoringSample(1, True, True, minimap, _arrival_observation()))
    assert accepted.sequence == 1


def test_detector_exception_breaks_pending_minimap_loss_debounce() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    hidden = np.zeros_like(minimap)
    malformed = np.zeros((0, 10, 3), dtype=np.uint8)
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    for sequence in range(1, 4):
        loop.process(MonitoringSample(sequence, True, True, minimap, moving))
    assert loop.process(MonitoringSample(4, True, True, hidden, moving)).events == ()
    loop.process(MonitoringSample(5, True, True, malformed, moving))

    first = loop.process(MonitoringSample(6, True, True, hidden, moving))
    second = loop.process(MonitoringSample(7, True, True, hidden, moving))

    assert first.events == ()
    assert [event.event_type for event in second.events] == [
        EventType.MINIMAP_VISIBILITY_LOST
    ]


def test_failed_stuck_recovery_does_not_duplicate_stuck_event() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    loop = AlphaMonitoringLoop(
        minimap=MinimapNavigationMonitor(
            anchor_template=anchor,
            expected_scale=1.0,
            visibility=MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
            navigation=NavigationStateMachine(stuck_frames=2, recovery_frames=2),
        )
    )
    blocked = NavigationObservation(MotionClass.STATIONARY, False, True)
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    loop.process(MonitoringSample(1, True, True, minimap, blocked))
    loop.process(MonitoringSample(2, True, True, minimap, blocked))
    loop.process(MonitoringSample(3, True, True, minimap, blocked))
    stuck = loop.process(MonitoringSample(4, True, True, minimap, blocked))
    recovering = loop.process(MonitoringSample(5, True, True, minimap, moving))
    stuck_again = loop.process(MonitoringSample(6, True, True, minimap, blocked))

    assert [event.event_type for event in stuck.events] == [EventType.STUCK]
    assert recovering.events == ()
    assert stuck_again.events == ()


def test_capture_closed_forces_consistent_lost_state_and_restoration() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    loop = _loop()
    moving = NavigationObservation(MotionClass.MOVING, False, True)
    loop.process(MonitoringSample(1, True, True, minimap, moving))
    loop.process(MonitoringSample(2, True, True, minimap, moving))

    closed = loop.capture_closed(3)
    pending = loop.process(MonitoringSample(4, True, True, minimap, moving))
    restored = loop.process(MonitoringSample(5, True, True, minimap, moving))

    assert [event.event_type for event in closed.events] == [EventType.CAPTURE_LOST]
    assert pending.events == ()
    assert [event.event_type for event in restored.events] == [EventType.CAPTURE_RESTORED]
