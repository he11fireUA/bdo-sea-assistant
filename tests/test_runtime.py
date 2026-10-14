from dataclasses import dataclass, field
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

import bdo_assistant.runtime as runtime_module
from bdo_assistant.minimap_observability import MinimapVisibilityEvidence, MinimapVisibilityTracker
from bdo_assistant.monitoring import MonitoringResult, MonitoringSample
from bdo_assistant.navigation_observability import NavigationObservabilityGate
from bdo_assistant.motion import MotionClass, MotionVector
from bdo_assistant.navigation_state import (
    NavigationObservation,
    NavigationState,
    NavigationStateMachine,
)
from bdo_assistant.notifications import EventType, NotificationEvent, Severity
from bdo_assistant.resource_bars import ResourceKind, measure_resource_bar
from bdo_assistant.ui_locator import Rect
from bdo_assistant.runtime import (
    CalibratedFrameSampleBuilder,
    ConservativeNavigationAnalyzer,
    SimpleNavigationAnalyzer,
    RuntimeBridge,
    ShadowMetrics,
    WgcWindowFrameSource,
)
from bdo_assistant.route_evidence import RouteEvidenceTracker, RouteGeometryClass
from bdo_assistant.capture_freshness import CaptureFreshness


@dataclass
class FakeSource:
    started: bool = False
    stopped: bool = False

    def start(self, on_frame, on_closed) -> None:
        self.started = True
        self.on_frame = on_frame
        self.on_closed = on_closed

    def stop(self) -> None:
        self.stopped = True


@dataclass
class FakeBuilder:
    sequences: list[int] = field(default_factory=list)

    def build(self, sequence: int, frame: np.ndarray) -> MonitoringSample:
        self.sequences.append(sequence)
        return MonitoringSample(sequence, True, True, frame)


@dataclass
class FakeLoop:
    samples: list[MonitoringSample] = field(default_factory=list)

    def process(self, sample: MonitoringSample) -> MonitoringResult:
        self.samples.append(sample)
        events = (
            (NotificationEvent(EventType.CAPTURE_RESTORED, Severity.INFO),)
            if sample.capture_ok
            else (NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL),)
        )
        return MonitoringResult(sample.sequence, events, NavigationState.UNKNOWN)


@dataclass
class FakeRouter:
    events: list[NotificationEvent] = field(default_factory=list)

    def route(self, event: NotificationEvent):
        self.events.append(event)
        return ()


def test_runtime_stop_waits_for_startup_before_stopping_source() -> None:
    startup_entered = threading.Event()
    release_startup = threading.Event()
    stop_attempted = threading.Event()

    class BlockingWatchdog:
        def start(self, _on_silence):
            startup_entered.set()
            assert release_startup.wait(1)
            return 1

        def frame_arrived(self, _generation):
            return CaptureFreshness.FRESH

        def stop(self):
            pass

    class OrderedSource(FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def start(self, on_frame, on_closed) -> None:
            self.calls.append("start")
            super().start(on_frame, on_closed)

        def stop(self) -> None:
            self.calls.append("stop")
            super().stop()

    source = OrderedSource()
    bridge = RuntimeBridge(
        source,
        FakeBuilder(),
        FakeLoop(),
        FakeRouter(),
        lambda status: None,
        capture_watchdog=BlockingWatchdog(),
    )

    starter = threading.Thread(target=bridge.start)
    starter.start()
    assert startup_entered.wait(1)

    def stop() -> None:
        stop_attempted.set()
        bridge.stop()

    stopper = threading.Thread(target=stop)
    stopper.start()
    assert stop_attempted.wait(1)
    release_startup.set()
    starter.join(1)
    stopper.join(1)

    assert not starter.is_alive()
    assert not stopper.is_alive()
    assert source.calls == ["start", "stop"]
    assert bridge.running is False


def test_runtime_stop_allows_source_stop_to_wait_for_inflight_frame_callback() -> None:
    frame_started = threading.Event()
    stop_finished = threading.Event()

    class CallbackWaitingSource(FakeSource):
        def stop(self) -> None:
            worker = threading.Thread(
                target=lambda: self.on_frame(np.zeros((20, 30, 3), dtype=np.uint8)),
                daemon=True,
            )
            worker.start()
            assert frame_started.wait(1)
            worker.join(1)
            assert not worker.is_alive()
            super().stop()

        def start(self, on_frame, on_closed) -> None:
            def tracked_frame(frame) -> None:
                frame_started.set()
                on_frame(frame)

            super().start(tracked_frame, on_closed)

    source = CallbackWaitingSource()
    bridge = RuntimeBridge(source, FakeBuilder(), FakeLoop(), FakeRouter(), lambda status: None)
    bridge.start()
    stopper = threading.Thread(target=lambda: (bridge.stop(), stop_finished.set()), daemon=True)
    stopper.start()

    assert stop_finished.wait(1)
    assert source.stopped
    assert bridge.running is False


def test_runtime_shadow_preserves_legacy_verdict_and_routes_no_candidate_events() -> None:
    class Candidate:
        def __init__(self) -> None:
            self.samples = []
            self.done = threading.Event()

        def process(self, sample):
            self.samples.append(sample)
            self.done.set()
            return {"motion": "moving", "candidate_event": "arrived"}

        def reset(self):
            pass

    class LegacyLoop(FakeLoop):
        def process(self, sample):
            self.samples.append(sample)
            return MonitoringResult(
                sample.sequence,
                (NotificationEvent(EventType.STUCK, Severity.WARNING),),
                NavigationState.STUCK,
                minimap_visible=True,
            )

    candidate = Candidate()
    router = FakeRouter()
    statuses = []
    bridge = RuntimeBridge(
        FakeSource(), FakeBuilder(), LegacyLoop(), router, statuses.append, shadow_processor=candidate
    )
    bridge.start()
    bridge.process_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    assert candidate.done.wait(1)

    assert bridge.navigation_state is NavigationState.STUCK
    assert [event.event_type for event in router.events] == [EventType.STUCK]
    assert candidate.samples[0].received_monotonic is not None
    metrics = statuses[-1].shadow_metrics
    assert type(metrics) is ShadowMetrics
    assert metrics.received_monotonic == candidate.samples[0].received_monotonic
    assert metrics.native_source_timestamp == candidate.samples[0].native_source_timestamp
    assert metrics.values == {"motion": "moving", "candidate_event": "arrived"}
    assert not any(isinstance(value, np.ndarray) for value in metrics.values.values())


def test_runtime_shadow_bounds_pending_work_and_counts_drops() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingCandidate:
        def process(self, sample):
            entered.set()
            release.wait(2)
            return {"motion": "unknown"}

        def reset(self):
            pass

    statuses = []
    bridge = RuntimeBridge(
        FakeSource(),
        FakeBuilder(),
        FakeLoop(),
        FakeRouter(),
        statuses.append,
        shadow_processor=BlockingCandidate(),
        shadow_queue_capacity=1,
    )
    bridge.start()
    bridge.process_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    assert entered.wait(1)
    bridge.process_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    bridge.process_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    release.set()
    bridge.stop()

    accounting = bridge.shadow_accounting
    assert accounting.submitted == 2
    assert accounting.dropped == 1
    assert accounting.completed <= accounting.submitted


def test_runtime_shadow_resets_with_session_and_cadence_discontinuity() -> None:
    class Candidate:
        def __init__(self) -> None:
            self.resets = 0

        def process(self, sample):
            return {"sequence": sample.sequence}

        def reset(self):
            self.resets += 1

    candidate = Candidate()
    source = FakeSource()
    cadence = runtime_module.AnalysisCadence(interval_seconds=1.0, discontinuity_seconds=2.0)
    clock = iter((1.0, 4.0))
    bridge = RuntimeBridge(
        source,
        FakeBuilder(),
        FakeLoop(),
        FakeRouter(),
        lambda status: None,
        shadow_processor=candidate,
        analysis_cadence=cadence,
        clock=lambda: next(clock),
    )

    bridge.start()
    source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    bridge.stop()
    bridge.start()

    assert candidate.resets == 3


def test_runtime_watchdog_publishes_capture_loss_during_callback_silence() -> None:
    class FakeWatchdog:
        def start(self, on_silence):
            self.on_silence = on_silence
            return 1

        def frame_arrived(self, _generation):
            return CaptureFreshness.FRESH

        def stop(self):
            self.stopped = True

    source = FakeSource()
    loop = FakeLoop()
    router = FakeRouter()
    statuses = []
    watchdog = FakeWatchdog()
    bridge = RuntimeBridge(
        source, FakeBuilder(), loop, router, statuses.append, capture_watchdog=watchdog
    )

    bridge.start()
    watchdog.on_silence(CaptureFreshness.SILENT)

    assert bridge.running is True
    assert source.stopped is False
    assert [sample.capture_ok for sample in loop.samples] == [False]
    assert [event.event_type for event in router.events] == [EventType.CAPTURE_LOST]
    assert statuses[-1].diagnostic == "capture silence"


def test_runtime_bridge_injects_real_monotonic_metadata_and_rejects_stale_generation() -> None:
    source = FakeSource()
    builder = FakeBuilder()
    loop = FakeLoop()
    clock_values = iter((10.0, 10.25))
    bridge = RuntimeBridge(
        source,
        builder,
        loop,
        FakeRouter(),
        lambda status: None,
        clock=lambda: next(clock_values),
    )

    bridge.start()
    stale_callback = source.on_frame
    source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    bridge.stop()
    bridge.start()
    stale_callback(np.zeros((20, 30, 3), dtype=np.uint8))
    source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))

    assert [(sample.sequence, sample.source_sequence, sample.processing_ordinal) for sample in loop.samples] == [
        (1, 1, 1),
        (2, 2, 2),
    ]
    assert [sample.received_monotonic for sample in loop.samples] == [10.0, 10.25]
    assert all(sample.native_source_timestamp is None for sample in loop.samples)
    assert all(sample.calibration_epoch == 0 for sample in loop.samples)


def test_runtime_bridge_recovers_unknown_route_then_rearms_stuck_episode() -> None:
    class ScriptedBuilder:
        def __init__(self) -> None:
            self.observations = iter(
                (
                    NavigationObservation(MotionClass.MOVING, False, True),
                    NavigationObservation(MotionClass.MOVING, False, True),
                    NavigationObservation(MotionClass.STATIONARY, False, True),
                    NavigationObservation(MotionClass.STATIONARY, False, True),
                    NavigationObservation(MotionClass.MOVING, None, None),
                    NavigationObservation(MotionClass.MOVING, None, None),
                    NavigationObservation(MotionClass.STATIONARY, False, True),
                    NavigationObservation(MotionClass.STATIONARY, False, True),
                )
            )

        def build(self, sequence: int, frame: np.ndarray) -> MonitoringSample:
            return MonitoringSample(sequence, True, True, frame, next(self.observations))

    class NavigationMonitoringFake:
        def __init__(self) -> None:
            self.gate = NavigationObservabilityGate(
                MinimapVisibilityTracker(lost_frames=2, restored_frames=2),
                NavigationStateMachine(stuck_frames=2, recovery_frames=2),
            )
            self.announced_stuck = False

        def process(self, sample: MonitoringSample) -> MonitoringResult:
            decision = self.gate.update(
                MinimapVisibilityEvidence(True, 1.0), sample.navigation_observation
            )
            events = ()
            if decision.navigation_state is NavigationState.STUCK and not self.announced_stuck:
                events = (NotificationEvent(EventType.STUCK, Severity.WARNING),)
                self.announced_stuck = True
            if decision.navigation_state is NavigationState.MOVING:
                self.announced_stuck = False
            return MonitoringResult(
                sample.sequence, events, decision.navigation_state, minimap_visible=True
            )

    source = FakeSource()
    router = FakeRouter()
    statuses = []
    bridge = RuntimeBridge(source, ScriptedBuilder(), NavigationMonitoringFake(), router, statuses.append)
    bridge.start()
    for _ in range(8):
        source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))

    frame_statuses = [status for status in statuses if status.result is not None]
    assert [status.navigation_state for status in frame_statuses] == [
        NavigationState.UNKNOWN,
        NavigationState.MOVING,
        NavigationState.STATIONARY_CANDIDATE,
        NavigationState.STUCK,
        NavigationState.RECOVERING,
        NavigationState.MOVING,
        NavigationState.STATIONARY_CANDIDATE,
        NavigationState.STUCK,
    ]
    assert [status.result.navigation_state for status in frame_statuses] == [
        status.navigation_state for status in frame_statuses
    ]
    assert [event.event_type for event in router.events] == [EventType.STUCK, EventType.STUCK]


def test_runtime_bridge_routes_frames_events_status_and_capture_close() -> None:
    source = FakeSource()
    builder = FakeBuilder()
    loop = FakeLoop()
    router = FakeRouter()
    statuses = []
    bridge = RuntimeBridge(source, builder, loop, router, statuses.append)

    bridge.start()
    source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
    source.on_closed()
    bridge.stop()

    assert source.started and source.stopped
    assert builder.sequences == [1]
    assert [(sample.sequence, sample.capture_ok) for sample in loop.samples] == [
        (1, True),
        (2, False),
    ]
    assert [event.event_type for event in router.events] == [
        EventType.CAPTURE_RESTORED,
        EventType.CAPTURE_LOST,
    ]
    assert statuses[-1].running is False
    assert statuses[-1].sequence == 2


def test_runtime_bridge_applies_live_resource_warning_zone_update() -> None:
    class TuningLoop(FakeLoop):
        updated = None

        def configure_resource_warning_zones(self, hp: int, rations: int) -> None:
            self.updated = (hp, rations)

    loop = TuningLoop()
    bridge = RuntimeBridge(FakeSource(), FakeBuilder(), loop, FakeRouter(), lambda status: None)

    bridge.configure_resource_warning_zones(60, 50)

    assert loop.updated == (60, 50)


def test_runtime_bridge_converts_builder_failure_to_diagnostics_without_crash() -> None:
    class FailingBuilder:
        def build(self, sequence, frame):
            raise ValueError("bad frame")

    statuses = []
    bridge = RuntimeBridge(FakeSource(), FailingBuilder(), FakeLoop(), FakeRouter(), statuses.append)

    bridge.start()
    bridge.process_frame(np.zeros((1, 1, 3), dtype=np.uint8))

    assert statuses[-1].running is True
    assert "bad frame" in statuses[-1].diagnostic


def test_runtime_start_failure_rolls_back_running_state() -> None:
    class FailingSource(FakeSource):
        def start(self, on_frame, on_closed):
            raise RuntimeError("start failed")

    statuses = []
    bridge = RuntimeBridge(FailingSource(), FakeBuilder(), FakeLoop(), FakeRouter(), statuses.append)

    with pytest.raises(RuntimeError, match="start failed"):
        bridge.start()

    assert not bridge.running
    assert "start failed" in statuses[-1].diagnostic


def test_runtime_resets_builder_before_each_capture_session() -> None:
    class ResettableBuilder(FakeBuilder):
        resets = 0

        def reset(self):
            self.resets += 1

    source = FakeSource()
    builder = ResettableBuilder()
    bridge = RuntimeBridge(source, builder, FakeLoop(), FakeRouter(), lambda status: None)

    bridge.start()
    bridge.stop()
    bridge.start()

    assert builder.resets == 2


def test_capture_close_survives_source_stop_failure_and_publishes_loss() -> None:
    class StopFailingSource(FakeSource):
        def stop(self):
            raise RuntimeError("stop failed")

    source = StopFailingSource()
    router = FakeRouter()
    statuses = []
    bridge = RuntimeBridge(source, FakeBuilder(), FakeLoop(), router, statuses.append)
    bridge.start()

    source.on_closed()

    assert not bridge.running
    assert EventType.CAPTURE_LOST in [event.event_type for event in router.events]
    assert "stop failed" in statuses[-1].diagnostic


def test_runtime_blocks_restart_until_capture_close_processing_finishes() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingCloseLoop(FakeLoop):
        def capture_closed(self, sequence):
            entered.set()
            release.wait(2)
            return MonitoringResult(
                sequence,
                (NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL),),
                NavigationState.UNKNOWN,
            )

    source = FakeSource()
    bridge = RuntimeBridge(source, FakeBuilder(), BlockingCloseLoop(), FakeRouter(), lambda status: None)
    bridge.start()
    closer = threading.Thread(target=source.on_closed)
    closer.start()
    assert entered.wait(1)

    with pytest.raises(RuntimeError, match="successful stop"):
        bridge.start()

    release.set()
    closer.join(2)
    bridge.start()
    assert bridge.running


def test_native_capture_close_synthesizes_capture_lost_event() -> None:
    class SilentLoop(FakeLoop):
        def process(self, sample):
            self.samples.append(sample)
            return MonitoringResult(sample.sequence, (), NavigationState.UNKNOWN)

    source = FakeSource()
    router = FakeRouter()
    statuses = []
    bridge = RuntimeBridge(source, FakeBuilder(), SilentLoop(), router, statuses.append)
    bridge.start()

    source.on_closed()

    assert [event.event_type for event in router.events] == [EventType.CAPTURE_LOST]
    assert [event.event_type for event in statuses[-1].result.events] == [
        EventType.CAPTURE_LOST
    ]


def test_wgc_source_targets_hwnd_converts_bgr_and_stops_control() -> None:
    created = []

    class Control:
        stopped = False

        def stop(self):
            self.stopped = True

    class Capture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.events = {}
            self.control = Control()
            created.append(self)

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            return self.control

    class Converted:
        def __init__(self, array):
            self.frame_buffer = array

    class Frame:
        def __init__(self, array):
            self.array = array

        def convert_to_bgr(self):
            return Converted(self.array)

    frames = []
    closed = []
    source = WgcWindowFrameSource(123, capture_factory=Capture)
    source.start(frames.append, lambda: closed.append(True))
    image = np.ones((3, 4, 3), dtype=np.uint8)
    created[0].events["on_frame_arrived"](Frame(image), object())
    image[:] = 0
    created[0].events["on_closed"]()
    source.start(frames.append, lambda: closed.append(True))
    source.stop()

    assert created[0].kwargs == {
        "cursor_capture": False,
        "draw_border": False,
        "window_hwnd": 123,
    }
    assert frames[0].shape == (3, 4, 3)
    assert frames[0].min() == 1
    assert closed == [True]
    assert len(created) == 2
    assert created[1].control.stopped


def test_wgc_stop_does_not_deadlock_when_native_stop_waits_for_frame_callback() -> None:
    callback_entered = threading.Event()
    callback_released = threading.Event()

    class Control:
        def __init__(self) -> None:
            self.frame_thread: threading.Thread | None = None

        def stop(self) -> None:
            self.frame_thread = threading.Thread(target=emit_frame)
            self.frame_thread.start()
            assert callback_entered.wait(1)
            callback_released.set()
            self.frame_thread.join(1)
            assert not self.frame_thread.is_alive()

    class Capture:
        def __init__(self, **kwargs):
            self.events = {}
            self.control = Control()

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            return self.control

    capture = Capture()
    source = WgcWindowFrameSource(123, capture_factory=lambda **kwargs: capture)
    source.start(lambda frame: None, lambda: None)

    def emit_frame() -> None:
        callback_entered.set()
        capture.events["on_frame_arrived"](
            type("Frame", (), {"convert_to_bgr": lambda self: type("C", (), {"frame_buffer": np.ones((2, 2, 3), dtype=np.uint8)})()})(),
            capture.control,
        )

    stop_thread = threading.Thread(target=source.stop)
    stop_thread.start()
    stop_thread.join(1)

    assert not stop_thread.is_alive()
    assert not source.is_running


def test_wgc_stop_failure_retains_control_for_retry() -> None:
    class Control:
        attempts = 0

        def stop(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("stop failed")

    class Capture:
        def __init__(self, **kwargs):
            self.events = {}
            self.control = Control()

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            return self.control

    source = WgcWindowFrameSource(123, capture_factory=Capture)
    source.start(lambda frame: None, lambda: None)

    with pytest.raises(RuntimeError, match="stop failed"):
        source.stop()
    source.stop()


def test_wgc_synchronous_close_during_start_does_not_leave_dead_control() -> None:
    created = []

    class Control:
        stopped = False

        def stop(self):
            self.stopped = True

    class Capture:
        def __init__(self, **kwargs):
            self.events = {}
            self.control = Control()
            created.append(self)

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            self.events["on_closed"]()
            return self.control

    closed = []
    source = WgcWindowFrameSource(123, capture_factory=Capture)
    source.start(lambda frame: None, lambda: closed.append(True))
    source.start(lambda frame: None, lambda: closed.append(True))

    assert closed == [True, True]
    assert created[0].control.stopped


def test_wgc_ignores_delayed_close_callback_from_previous_session() -> None:
    created = []

    class Control:
        stopped = False

        def stop(self):
            self.stopped = True

    class Capture:
        def __init__(self, **kwargs):
            self.events = {}
            self.control = Control()
            created.append(self)

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            return self.control

    closed = []
    source = WgcWindowFrameSource(123, capture_factory=Capture)
    source.start(lambda frame: None, lambda: closed.append("first"))
    old_close = created[0].events["on_closed"]
    old_close()
    source.start(lambda frame: None, lambda: closed.append("second"))

    old_close()
    source.stop()

    assert closed == ["first"]
    assert created[1].control.stopped


def test_wgc_native_callbacks_contain_stop_and_client_callback_errors() -> None:
    class Control:
        def stop(self):
            raise RuntimeError("native stop failed")

    class Capture:
        def __init__(self, **kwargs):
            self.events = {}
            self.control = Control()

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            return self.control

    capture = Capture()
    source = WgcWindowFrameSource(123, capture_factory=lambda **kwargs: capture)
    source.start(
        lambda frame: (_ for _ in ()).throw(RuntimeError("frame callback failed")),
        lambda: (_ for _ in ()).throw(RuntimeError("close callback failed")),
    )

    capture.events["on_frame_arrived"](
        type("Frame", (), {"convert_to_bgr": lambda self: type("C", (), {"frame_buffer": np.ones((2, 2, 3), dtype=np.uint8)})()})(),
        capture.control,
    )
    capture.events["on_closed"]()

    assert "native stop failed" in source.last_error
    assert "close callback failed" in source.last_error


def test_status_callback_failure_does_not_escape_runtime_lifecycle() -> None:
    def failing_status(status):
        raise RuntimeError("status sink failed")

    bridge = RuntimeBridge(FakeSource(), FakeBuilder(), FakeLoop(), FakeRouter(), failing_status)

    bridge.start()
    bridge.stop()

    assert not bridge.running
    assert bridge.last_status is not None
    assert "status sink failed" in bridge.last_status_error


def test_wgc_concurrent_stop_waits_for_start_and_stops_returned_control() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Control:
        stopped = False

        def stop(self):
            self.stopped = True

    class Capture:
        def __init__(self, **kwargs):
            self.events = {}
            self.control = Control()

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            entered.set()
            release.wait(2)
            return self.control

    capture = Capture()
    source = WgcWindowFrameSource(123, capture_factory=lambda **kwargs: capture)
    starter = threading.Thread(target=lambda: source.start(lambda frame: None, lambda: None))
    stopper = threading.Thread(target=source.stop)
    starter.start()
    assert entered.wait(1)
    stopper.start()
    release.set()
    starter.join(2)
    stopper.join(2)

    assert capture.control.stopped
    assert not source.is_running


def test_wgc_inflight_old_frame_is_dropped_after_restart() -> None:
    created = []

    class Control:
        def stop(self):
            pass

    class Capture:
        def __init__(self, **kwargs):
            self.events = {}
            self.control = Control()
            created.append(self)

        def event(self, callback):
            self.events[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            return self.control

    entered = threading.Event()
    release = threading.Event()
    delivered = []
    source = WgcWindowFrameSource(123, capture_factory=Capture)
    source.start(delivered.append, lambda: None)

    class Frame:
        def convert_to_bgr(self):
            entered.set()
            release.wait(2)
            return type("Converted", (), {"frame_buffer": np.ones((2, 2, 3), dtype=np.uint8)})()

    old_callback = created[0].events["on_frame_arrived"]
    worker = threading.Thread(target=lambda: old_callback(Frame(), created[0].control))
    worker.start()
    assert entered.wait(1)
    source.stop()
    source.start(delivered.append, lambda: None)
    release.set()
    worker.join(2)

    assert delivered == []


def test_calibrated_builder_extracts_normalized_rois_resources_and_navigation() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[0:50, 100:200] = 80
    frame[59, 10:110] = (120, 120, 120)
    frame[60:70, 10:110] = (25, 25, 25)
    frame[62:68, 10:60] = (0, 0, 230)
    frame[74, 10:110] = (120, 120, 120)
    frame[75:85, 10:110] = (25, 25, 25)
    frame[77:83, 10:30] = (0, 210, 220)
    layout = {
        "schema_version": 1,
        "regions": {
            "minimap": {"normalized": {"x": 0.5, "y": 0.0, "width": 0.5, "height": 0.5}},
            "ship_hp_bar": {"normalized": {"x": 0.05, "y": 0.6, "width": 0.5, "height": 0.1}},
            "ship_rations_bar": {"normalized": {"x": 0.05, "y": 0.75, "width": 0.5, "height": 0.1}},
        },
    }
    observation = NavigationObservation(MotionClass.MOVING, False, True)
    seen = []

    def navigation(sequence, minimap_roi):
        seen.append((sequence, minimap_roi.shape))
        return observation

    sample = CalibratedFrameSampleBuilder(layout, navigation).build(7, frame)

    assert sample.sequence == 7
    assert sample.capture_ok and sample.calibration_ok
    assert sample.minimap_roi.shape == (50, 100, 3)
    assert sample.navigation_observation is observation
    assert sample.hp_percent == pytest.approx(50, abs=3)
    assert sample.rations_percent == pytest.approx(20, abs=3)
    assert seen == [(7, (50, 100, 3))]


def test_calibrated_builder_promotes_rations_candidate_when_hp_is_strictly_observable() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[0:50, 100:200] = 80
    frame[60:70, 10:110] = (0, 0, 230)  # full red HP is the strict HUD anchor
    frame[75:85, 10:110] = (25, 25, 25)
    frame[77:83, 10:97] = (0, 210, 220)  # valid partial Rations, deliberately no pale rail
    layout = {
        "schema_version": 1,
        "regions": {
            "minimap": {"normalized": {"x": 0.5, "y": 0.0, "width": 0.5, "height": 0.5}},
            "ship_hp_bar": {"normalized": {"x": 0.05, "y": 0.6, "width": 0.5, "height": 0.1}},
            "ship_rations_bar": {"normalized": {"x": 0.05, "y": 0.75, "width": 0.5, "height": 0.1}},
        },
    }
    builder = CalibratedFrameSampleBuilder(
        layout,
        lambda sequence, roi: NavigationObservation(MotionClass.MOVING, False, True),
    )

    sample = builder.build(7, frame)

    assert sample.resource_bars_visible is True
    assert sample.hp_percent == 100.0
    assert sample.rations_percent == pytest.approx(87, abs=3)


def test_calibrated_builder_does_not_promote_rations_candidate_without_hp_anchor() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[0:50, 100:200] = 80
    frame[60:70, 10:110] = (25, 25, 25)
    frame[75:85, 10:110] = (25, 25, 25)
    frame[77:83, 10:97] = (0, 210, 220)  # candidate has no pale rail or HP partner
    layout = {
        "schema_version": 1,
        "regions": {
            "minimap": {"normalized": {"x": 0.5, "y": 0.0, "width": 0.5, "height": 0.5}},
            "ship_hp_bar": {"normalized": {"x": 0.05, "y": 0.6, "width": 0.5, "height": 0.1}},
            "ship_rations_bar": {"normalized": {"x": 0.05, "y": 0.75, "width": 0.5, "height": 0.1}},
        },
    }
    builder = CalibratedFrameSampleBuilder(
        layout,
        lambda sequence, roi: NavigationObservation(MotionClass.MOVING, False, True),
    )

    candidate = measure_resource_bar(frame, Rect(10, 75, 100, 10), ResourceKind.RATIONS)
    sample = builder.build(7, frame)

    assert candidate.candidate_percent == pytest.approx(87, abs=3)
    assert sample.resource_bars_visible is False
    assert sample.hp_percent is None
    assert sample.rations_percent is None


def test_calibrated_builder_marks_both_missing_ship_bars_as_unobservable() -> None:
    frame = np.full((100, 200, 3), (130, 80, 25), dtype=np.uint8)
    layout = {
        "schema_version": 1,
        "regions": {
            "minimap": {"normalized": {"x": 0.5, "y": 0.0, "width": 0.5, "height": 0.5}},
            "ship_hp_bar": {"normalized": {"x": 0.05, "y": 0.6, "width": 0.5, "height": 0.1}},
            "ship_rations_bar": {"normalized": {"x": 0.05, "y": 0.75, "width": 0.5, "height": 0.1}},
        },
    }
    builder = CalibratedFrameSampleBuilder(
        layout,
        lambda sequence, roi: NavigationObservation(MotionClass.UNKNOWN, None, None),
    )

    sample = builder.build(7, frame)

    assert sample.resource_bars_visible is False
    assert sample.hp_percent is None
    assert sample.rations_percent is None


def test_calibrated_builder_rejects_non_finite_normalized_region() -> None:
    layout = {
        "schema_version": 1,
        "regions": {
            name: {"normalized": {"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5}}
            for name in ("minimap", "ship_hp_bar", "ship_rations_bar")
        },
    }
    layout["regions"]["minimap"]["normalized"]["x"] = float("nan")
    builder = CalibratedFrameSampleBuilder(
        layout,
        lambda sequence, roi: NavigationObservation(MotionClass.UNKNOWN, None, None),
    )

    with pytest.raises(ValueError, match="invalid normalized region"):
        builder.build(1, np.zeros((100, 100, 3), dtype=np.uint8))


def test_conservative_navigation_analyzer_requires_temporal_coherent_motion() -> None:
    rng = np.random.default_rng(42)
    base = cv2.GaussianBlur(
        rng.integers(0, 256, size=(80, 80), dtype=np.uint8),
        (3, 3),
        0,
    )
    analyzer = ConservativeNavigationAnalyzer(
        route_evidence=lambda sequence, roi: (False, True),
        window_size=6,
    )
    observations = []
    for sequence, shift in enumerate(range(5), start=1):
        image = cv2.warpAffine(
            base,
            np.float32([[1, 0, shift], [0, 1, 0]]),
            (80, 80),
            borderMode=cv2.BORDER_REFLECT,
        )
        observations.append(analyzer(sequence, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)))

    assert observations[0].motion is MotionClass.UNKNOWN
    assert observations[-1].motion is MotionClass.MOVING
    assert observations[-1].route_complete is False
    assert observations[-1].endpoint_ahead is True


def test_persistent_minimap_texture_activity_vetoes_stationary_classification(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "estimate_translation",
        lambda previous, current: MotionVector(0.0, 0.0, 1.0),
    )
    analyzer = ConservativeNavigationAnalyzer(route_evidence=lambda sequence, roi: (False, True))
    frames = [
        np.full((80, 80, 3), value, dtype=np.uint8)
        for value in (0, 255, 0, 255)
    ]

    observations = [analyzer(sequence, frame) for sequence, frame in enumerate(frames, 1)]

    assert observations[-1].motion is MotionClass.UNKNOWN


def test_navigation_analyzer_resets_temporal_window_on_sequence_gap() -> None:
    base = cv2.GaussianBlur(
        np.random.default_rng(5).integers(0, 256, size=(80, 80), dtype=np.uint8),
        (3, 3),
        0,
    )
    analyzer = ConservativeNavigationAnalyzer(
        route_evidence=lambda sequence, roi: (False, True),
        window_size=6,
    )

    def image(shift):
        gray = cv2.warpAffine(
            base,
            np.float32([[1, 0, shift], [0, 1, 0]]),
            (80, 80),
            borderMode=cv2.BORDER_REFLECT,
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    analyzer(1, image(0))
    analyzer(2, image(1))
    analyzer(3, image(2))
    after_gap = analyzer(5, image(3))

    assert after_gap.motion is MotionClass.UNKNOWN


def test_navigation_analyzer_reset_propagates_to_route_evidence() -> None:
    class ResettableRouteEvidence:
        resets = 0

        def __call__(self, sequence, roi):
            return None, None

        def reset(self):
            self.resets += 1

    route_evidence = ResettableRouteEvidence()
    analyzer = ConservativeNavigationAnalyzer(route_evidence=route_evidence)

    analyzer.reset()

    assert route_evidence.resets == 1


def test_navigation_shape_change_resets_route_evidence_before_new_frame() -> None:
    calls = []

    class ResettableRouteEvidence:
        def __call__(self, sequence, roi):
            calls.append(("frame", sequence, roi.shape[:2]))
            return None, None

        def reset(self):
            calls.append(("reset", None, None))

    analyzer = ConservativeNavigationAnalyzer(route_evidence=ResettableRouteEvidence())
    analyzer(1, np.zeros((100, 100, 3), dtype=np.uint8))
    after_resize = analyzer(2, np.zeros((120, 120, 3), dtype=np.uint8))

    assert after_resize.motion is MotionClass.UNKNOWN
    assert calls == [
        ("frame", 1, (100, 100)),
        ("reset", None, None),
        ("frame", 2, (120, 120)),
    ]


def test_navigation_shape_change_discards_prior_ahead_route_provenance() -> None:
    class ShapeRouteEvidence:
        def __init__(self):
            self.tracker = RouteEvidenceTracker(confirmations=3)

        def __call__(self, sequence, roi):
            geometry = (
                RouteGeometryClass.AHEAD
                if roi.shape[:2] == (100, 100)
                else RouteGeometryClass.NEAR
            )
            return self.tracker.update(geometry)

        def reset(self):
            self.tracker.reset()

    analyzer = ConservativeNavigationAnalyzer(route_evidence=ShapeRouteEvidence())
    old_shape = np.zeros((100, 100, 3), dtype=np.uint8)
    new_shape = np.zeros((120, 120, 3), dtype=np.uint8)
    for sequence in range(1, 4):
        analyzer(sequence, old_shape)
    observations = [analyzer(sequence, new_shape) for sequence in range(4, 7)]

    assert [(item.route_complete, item.endpoint_ahead) for item in observations] == [
        (None, None),
        (None, None),
        (None, None),
    ]


def test_navigation_shape_reset_allows_fresh_ahead_to_near_completion() -> None:
    class ScriptedRouteEvidence:
        def __init__(self):
            self.tracker = RouteEvidenceTracker(confirmations=3)
            self.geometry = {
                1: RouteGeometryClass.AHEAD,
                2: RouteGeometryClass.AHEAD,
                3: RouteGeometryClass.AHEAD,
                4: RouteGeometryClass.NEAR,
                5: RouteGeometryClass.AHEAD,
                6: RouteGeometryClass.AHEAD,
                7: RouteGeometryClass.AHEAD,
                8: RouteGeometryClass.NEAR,
                9: RouteGeometryClass.NEAR,
                10: RouteGeometryClass.NEAR,
            }

        def __call__(self, sequence, roi):
            return self.tracker.update(self.geometry[sequence])

        def reset(self):
            self.tracker.reset()

    analyzer = ConservativeNavigationAnalyzer(route_evidence=ScriptedRouteEvidence())
    old_shape = np.zeros((100, 100, 3), dtype=np.uint8)
    new_shape = np.zeros((120, 120, 3), dtype=np.uint8)
    for sequence in range(1, 4):
        analyzer(sequence, old_shape)
    analyzer(4, new_shape)
    observations = [analyzer(sequence, new_shape) for sequence in range(5, 11)]

    assert [(item.route_complete, item.endpoint_ahead) for item in observations] == [
        (None, None),
        (None, None),
        (False, True),
        (None, None),
        (None, None),
        (True, False),
    ]


def test_navigation_sequence_gap_resets_route_evidence_before_new_frame() -> None:
    calls = []

    class ResettableRouteEvidence:
        def __call__(self, sequence, roi):
            calls.append(("frame", sequence))
            return None, None

        def reset(self):
            calls.append(("reset", None))

    analyzer = ConservativeNavigationAnalyzer(route_evidence=ResettableRouteEvidence())
    roi = np.zeros((80, 80, 3), dtype=np.uint8)

    analyzer(1, roi)
    analyzer(3, roi)

    assert calls == [("frame", 1), ("reset", None), ("frame", 3)]


def test_runtime_minimap_visibility_loss_does_not_reset_simple_navigation_baseline() -> None:
    fixtures = Path(__file__).parent / "fixtures" / "route_evidence"
    ahead = cv2.imread(str(fixtures / "stuck_route_ahead.jpg"), cv2.IMREAD_COLOR)
    near = cv2.imread(str(fixtures / "arrival_near_endpoint.png"), cv2.IMREAD_COLOR)
    assert ahead is not None and near is not None

    class RouteBuilder:
        def __init__(self):
            self._navigation_analyzer = SimpleNavigationAnalyzer()
            self.tracker = RouteEvidenceTracker(confirmations=3)
            self.evidence = []
            self.reset_calls = 0

        def build(self, sequence, frame):
            route_complete, endpoint_ahead = self.tracker(sequence, frame)
            self.evidence.append((route_complete, endpoint_ahead))
            observation = NavigationObservation(
                MotionClass.UNKNOWN,
                route_complete,
                endpoint_ahead,
            )
            return MonitoringSample(sequence, True, True, frame, observation)

        def reset(self):
            self.reset_calls += 1
            self.tracker.reset()

    class VisibilityLoop:
        def process(self, sample):
            return MonitoringResult(
                sample.sequence,
                (),
                NavigationState.UNKNOWN,
                minimap_visible=sample.sequence != 4,
            )

    builder = RouteBuilder()
    bridge = RuntimeBridge(FakeSource(), builder, VisibilityLoop(), FakeRouter(), lambda status: None)
    bridge.start()
    builder.reset_calls = 0
    for frame in (ahead, ahead, ahead, near, near, near, near):
        bridge.process_frame(frame)

    assert builder.evidence[2] == (False, True)
    assert builder.reset_calls == 0
