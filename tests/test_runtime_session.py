import pytest

from bdo_assistant.navigation_state import NavigationState
from bdo_assistant.runtime import RuntimeBridge, RuntimeStatus
from bdo_assistant.runtime_session import (
    RuntimeSessionController,
    RuntimeSessionState,
)


class FakeBridge:
    def __init__(self, *, fail_start: bool = False, fail_stop_once: bool = False) -> None:
        self.running = False
        self.fail_start = fail_start
        self.fail_stop_once = fail_stop_once
        self.start_calls = 0
        self.stop_calls = 0
        self.cleanup_required = False

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("start failed")
        self.running = True

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop_once and self.stop_calls == 1:
            raise RuntimeError("stop failed")
        self.running = False
        self.cleanup_required = False


def test_session_controller_builds_and_starts_bridge_once() -> None:
    bridge = FakeBridge()
    states = []
    controller = RuntimeSessionController(lambda: bridge, states.append)

    controller.start()

    assert controller.state is RuntimeSessionState.RUNNING
    assert controller.bridge is bridge
    assert bridge.start_calls == 1
    assert [state.state for state in states] == [
        RuntimeSessionState.STARTING,
        RuntimeSessionState.RUNNING,
    ]
    with pytest.raises(RuntimeError, match="already active"):
        controller.start()


def test_session_forwards_live_resource_warning_zones_to_active_bridge() -> None:
    class TuningBridge(FakeBridge):
        updated = None

        def configure_resource_warning_zones(self, hp: int, rations: int) -> None:
            self.updated = (hp, rations)

    bridge = TuningBridge()
    controller = RuntimeSessionController(lambda: bridge, lambda snapshot: None)
    controller.start()

    controller.configure_resource_warning_zones(60, 50)

    assert bridge.updated == (60, 50)


def test_session_controller_start_failure_is_visible_and_retryable() -> None:
    failing = FakeBridge(fail_start=True)
    healthy = FakeBridge()
    bridges = iter((failing, healthy))
    states = []
    controller = RuntimeSessionController(lambda: next(bridges), states.append)

    with pytest.raises(RuntimeError, match="start failed"):
        controller.start()

    assert controller.state is RuntimeSessionState.ERROR
    assert controller.bridge is None
    assert "start failed" in controller.last_error
    assert failing.stop_calls == 1

    controller.start()
    assert controller.state is RuntimeSessionState.RUNNING
    assert controller.bridge is healthy


def test_session_controller_preserves_bridge_for_stop_retry() -> None:
    bridge = FakeBridge(fail_stop_once=True)
    controller = RuntimeSessionController(lambda: bridge, lambda state: None)
    controller.start()

    with pytest.raises(RuntimeError, match="stop failed"):
        controller.stop()

    assert controller.state is RuntimeSessionState.ERROR
    assert controller.bridge is bridge
    assert "stop failed" in controller.last_error

    controller.stop()
    assert controller.state is RuntimeSessionState.STOPPED
    assert controller.bridge is None
    assert bridge.stop_calls == 2


def test_session_controller_stop_is_idempotent_before_start() -> None:
    controller = RuntimeSessionController(lambda: FakeBridge(), lambda state: None)

    controller.stop()

    assert controller.state is RuntimeSessionState.STOPPED


def test_session_controller_reacts_to_asynchronous_bridge_closure() -> None:
    bridge = FakeBridge()
    states = []
    controller = RuntimeSessionController(lambda: bridge, states.append)
    controller.start()

    controller.handle_runtime_status(
        RuntimeStatus(
            running=False,
            sequence=12,
            navigation_state=NavigationState.UNKNOWN,
            diagnostic="capture closed",
        )
    )

    assert controller.state is RuntimeSessionState.ERROR
    assert controller.bridge is None
    assert "capture closed" in controller.last_error
    assert states[-1].bridge_active is False


def test_async_close_retains_bridge_when_native_cleanup_requires_retry() -> None:
    bridge = FakeBridge()
    states = []
    controller = RuntimeSessionController(lambda: bridge, states.append)
    controller.start()
    bridge.running = False
    bridge.cleanup_required = True

    controller.handle_runtime_status(
        RuntimeStatus(
            running=False,
            sequence=13,
            navigation_state=NavigationState.UNKNOWN,
            diagnostic="capture closed; source stop failed",
        )
    )

    assert controller.state is RuntimeSessionState.ERROR
    assert controller.bridge is bridge
    assert states[-1].bridge_active is True

    controller.stop()
    assert bridge.stop_calls == 1
    assert controller.state is RuntimeSessionState.STOPPED
    assert controller.bridge is None


def test_real_bridge_async_close_retries_failed_source_cleanup() -> None:
    class Source:
        def __init__(self):
            self.stop_calls = 0
            self.on_closed = None

        def start(self, on_frame, on_closed):
            self.on_closed = on_closed

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("source stop failed")

    class Builder:
        def build(self, sequence, frame):
            raise AssertionError("no frame expected")

    class Loop:
        def process(self, sample):
            raise AssertionError("no sample expected")

        def capture_closed(self, sequence):
            from bdo_assistant.monitoring import MonitoringResult
            return MonitoringResult(sequence, (), NavigationState.UNKNOWN)

    class Router:
        def route(self, event):
            return ()

    source = Source()
    holder = {}

    def factory():
        return RuntimeBridge(
            source, Builder(), Loop(), Router(),
            lambda status: holder["controller"].handle_runtime_status(status),
        )

    controller = RuntimeSessionController(factory, lambda snapshot: None)
    holder["controller"] = controller
    controller.start()
    source.on_closed()

    assert controller.state is RuntimeSessionState.ERROR
    assert controller.bridge.cleanup_required

    controller.stop()

    assert source.stop_calls == 2
    assert controller.state is RuntimeSessionState.STOPPED
    assert controller.bridge is None


def test_real_bridge_partial_start_failure_retains_cleanup_retry() -> None:
    class Source:
        def __init__(self):
            self.stop_calls = 0

        def start(self, on_frame, on_closed):
            raise RuntimeError("partial source start failed")

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("first cleanup failed")

    class Builder:
        def build(self, sequence, frame):
            raise AssertionError("no frame expected")

    class Loop:
        def process(self, sample):
            raise AssertionError("no sample expected")

    class Router:
        def route(self, event):
            return ()

    source = Source()
    bridge = RuntimeBridge(source, Builder(), Loop(), Router(), lambda status: None)
    controller = RuntimeSessionController(lambda: bridge, lambda snapshot: None)

    with pytest.raises(RuntimeError, match="partial source start failed"):
        controller.start()

    assert controller.state is RuntimeSessionState.ERROR
    assert controller.bridge is bridge
    assert bridge.cleanup_required
    assert source.stop_calls == 1

    controller.stop()

    assert source.stop_calls == 2
    assert controller.state is RuntimeSessionState.STOPPED
    assert controller.bridge is None


def test_real_bridge_successful_async_cleanup_releases_session_bridge() -> None:
    class Source:
        def start(self, on_frame, on_closed): self.on_closed = on_closed
        def stop(self): pass
    class Builder:
        def build(self, sequence, frame): raise AssertionError
    class Loop:
        def process(self, sample): raise AssertionError
        def capture_closed(self, sequence):
            from bdo_assistant.monitoring import MonitoringResult
            return MonitoringResult(sequence, (), NavigationState.UNKNOWN)
    class Router:
        def route(self, event): return ()

    source = Source()
    holder = {}
    bridge = RuntimeBridge(
        source, Builder(), Loop(), Router(),
        lambda status: holder["controller"].handle_runtime_status(status),
    )
    controller = RuntimeSessionController(lambda: bridge, lambda snapshot: None)
    holder["controller"] = controller
    controller.start()
    source.on_closed()

    assert controller.state is RuntimeSessionState.ERROR
    assert controller.bridge is None
    assert not bridge.cleanup_required
