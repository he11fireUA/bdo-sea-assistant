"""Live delivery configuration through the session, without native capture/audio."""
from types import SimpleNamespace

import numpy as np
import pytest

from bdo_assistant.minimap_observability import MinimapVisibilityTracker
from bdo_assistant.monitoring import MonitoringResult, MonitoringSample
from bdo_assistant.navigation_state import NavigationState
from bdo_assistant.notifications import EventType, NotificationEvent, NotificationRouter, Severity
from bdo_assistant.runtime import RuntimeBridge
from bdo_assistant.runtime_session import RuntimeSessionController


class Adapter:
    def __init__(self, name):
        self.name = name
        self.events = []

    def send(self, event):
        self.events.append(event)


class Source:
    def start(self, on_frame, on_closed):
        self.on_frame = on_frame

    def stop(self):
        pass


def make_session():
    tracker = MinimapVisibilityTracker(lost_after_seconds=10)
    source = Source()
    original = Adapter("old")
    event = NotificationEvent(EventType.RATIONS_LOW, Severity.WARNING)
    loop = SimpleNamespace(
        minimap=SimpleNamespace(gate=SimpleNamespace(visibility=tracker)),
        process=lambda sample: MonitoringResult(sample.sequence, (event,), NavigationState.UNKNOWN),
    )
    builder = SimpleNamespace(build=lambda sequence, frame: MonitoringSample(sequence, True, True, frame))
    bridge = RuntimeBridge(source, builder, loop, NotificationRouter((original,)), lambda status: None)
    session = RuntimeSessionController(lambda: bridge, lambda snapshot: None)
    return session, bridge, source, tracker, original


@pytest.mark.parametrize("delay", [True, None, "30", 10.0, 9, 301, float("nan")])
def test_invalid_delay_leaves_router_and_pending_loss_unchanged(delay):
    session, bridge, source, tracker, original = make_session()
    now = [0.0]
    tracker._clock = lambda: now[0]
    tracker.update(False)
    old_router = bridge.router
    with pytest.raises(ValueError, match="minimap_loss_delay_seconds"):
        bridge.configure_delivery((Adapter("new"),), delay)
    assert bridge.router is old_router
    assert tracker.lost_after_seconds == 10
    now[0] = 10.0
    assert tracker.update(False) is EventType.MINIMAP_VISIBILITY_LOST


@pytest.mark.parametrize("adapters", [None, [], (object(),)])
def test_invalid_adapters_leave_router_and_delay_unchanged(adapters):
    session, bridge, source, tracker, original = make_session()
    old_router = bridge.router
    with pytest.raises(ValueError, match="adapter"):
        bridge.configure_delivery(adapters, 20)
    assert bridge.router is old_router
    assert tracker.lost_after_seconds == 10


def test_unsupported_loop_does_not_swap_router():
    session, bridge, source, tracker, original = make_session()
    bridge.loop = SimpleNamespace(process=lambda sample: None)
    old_router = bridge.router
    with pytest.raises(RuntimeError, match="live minimap loss delay"):
        bridge.configure_delivery((), 20)
    assert bridge.router is old_router


def test_session_delivery_requires_running_supported_bridge():
    session, bridge, source, tracker, original = make_session()
    with pytest.raises(RuntimeError, match="not running"):
        session.configure_delivery((), 20)
    session.start()
    try:
        bridge.configure_delivery = None
        with pytest.raises(RuntimeError, match="does not support"):
            session.configure_delivery((), 20)
    finally:
        session.stop()


def test_delivery_swap_waits_for_complete_inflight_fanout():
    import threading

    session, bridge, source, tracker, original = make_session()
    attempted = threading.Event()
    completed = threading.Event()
    failures = []
    replacement = Adapter("new")
    lock = bridge._lifecycle_lock

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread() is worker:
                attempted.set()
            return lock.__enter__()

        def __exit__(self, *args):
            return lock.__exit__(*args)

    def configure():
        try:
            session.configure_delivery((replacement,), 20)
        except BaseException as error:
            failures.append(error)
        finally:
            completed.set()

    worker = threading.Thread(target=configure, daemon=True)
    bridge._lifecycle_lock = ObservedLock()
    observed = []

    class First(Adapter):
        def send(self, event):
            worker.start()
            assert attempted.wait(2)
            observed.append((self.name, tracker.lost_after_seconds, completed.is_set()))

    class Last(Adapter):
        def send(self, event):
            observed.append((self.name, tracker.lost_after_seconds, completed.is_set()))

    bridge.router = NotificationRouter((First("first"), Last("last")))
    session.start()
    try:
        source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        worker.join(2)
        assert not worker.is_alive()
        assert not failures
        assert observed == [("first", 10, False), ("last", 10, False)]
        assert tracker.lost_after_seconds == 20
        source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))
        assert len(replacement.events) == 1
    finally:
        session.stop()


def test_session_configure_delivery_replaces_active_adapters_and_delay():
    session, bridge, source, tracker, original = make_session()
    replacement = Adapter("new")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    session.start()
    try:
        source.on_frame(frame)
        session.configure_delivery((replacement,), 30)
        source.on_frame(frame)
        assert len(original.events) == 1
        assert len(replacement.events) == 1
        assert tracker.lost_after_seconds == 30
        assert bridge.last_status.deliveries[0].adapter == "new"
        session.configure_delivery((), 300)
        source.on_frame(frame)
        assert len(replacement.events) == 1
        assert tracker.lost_after_seconds == 300
        assert bridge.last_status.deliveries == ()
    finally:
        session.stop()
