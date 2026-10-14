from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from bdo_assistant.notifications import (
    EventType,
    NotificationEvent,
    NotificationRouter,
    Severity,
)


@dataclass
class RecordingAdapter:
    name: str
    fail: bool = False
    received: list[NotificationEvent] = field(default_factory=list)

    def send(self, event: NotificationEvent) -> None:
        self.received.append(event)
        if self.fail:
            raise RuntimeError(f"{self.name} offline")


@dataclass
class NestedPayloadMutatingAdapter:
    name: str = "mutator"

    def send(self, event: NotificationEvent) -> None:
        event.parameters["route"]["points"].append("poisoned")


def test_router_isolates_adapter_failure_and_continues_fanout() -> None:
    telegram = RecordingAdapter("telegram", fail=True)
    voice = RecordingAdapter("voice")
    popup = RecordingAdapter("popup")
    router = NotificationRouter([telegram, voice, popup])
    event = NotificationEvent(EventType.STUCK, Severity.WARNING, {"seconds": 40})

    results = router.route(event)

    assert [result.adapter for result in results] == ["telegram", "voice", "popup"]
    assert [result.success for result in results] == [False, True, True]
    assert "telegram offline" in results[0].error
    assert voice.received == [event]
    assert popup.received == [event]


def test_event_is_typed_and_payload_is_immutable() -> None:
    event = NotificationEvent(
        EventType.ARRIVED,
        Severity.INFO,
        {"destination": "Iliya Island"},
    )

    assert event.event_type is EventType.ARRIVED
    assert event.parameters["destination"] == "Iliya Island"

    try:
        event.parameters["destination"] = "Elsewhere"
    except TypeError:
        pass
    else:
        raise AssertionError("event payload must be immutable")


def test_nested_payload_is_immutable_between_adapters() -> None:
    mutator = NestedPayloadMutatingAdapter()
    recorder = RecordingAdapter("recorder")
    router = NotificationRouter([mutator, recorder])
    event = NotificationEvent(
        EventType.STUCK,
        Severity.WARNING,
        {"route": {"points": ["A", "B"]}},
    )

    results = router.route(event)

    assert [result.success for result in results] == [False, True]
    assert tuple(recorder.received[0].parameters["route"]["points"]) == ("A", "B")


@pytest.mark.parametrize(
    "adapter",
    [object(), type("NoSendAdapter", (), {"name": "broken"})()],
)
def test_router_rejects_invalid_adapter_dependencies(adapter: object) -> None:
    with pytest.raises(ValueError, match="adapter"):
        NotificationRouter([adapter])


def test_router_uses_cached_name_and_continues_when_adapter_fails() -> None:
    class FailingNameAdapter:
        def __init__(self) -> None:
            self.name_reads = 0

        @property
        def name(self) -> str:
            self.name_reads += 1
            if self.name_reads > 1:
                raise RuntimeError("name unavailable")
            return "flaky"

        def send(self, event: NotificationEvent) -> None:
            raise RuntimeError("send failed")

    flaky = FailingNameAdapter()
    healthy = RecordingAdapter("healthy")
    router = NotificationRouter([flaky, healthy])
    event = NotificationEvent(EventType.STUCK, Severity.WARNING)

    results = router.route(event)

    assert [result.adapter for result in results] == ["flaky", "healthy"]
    assert [result.success for result in results] == [False, True]
    assert healthy.received == [event]
