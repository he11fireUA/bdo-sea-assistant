from dataclasses import dataclass, field

import pytest

from bdo_assistant.notifications import (
    EventType,
    NotificationEvent,
    NotificationRouter,
    Severity,
)
from bdo_assistant.popup import (
    PopupAdapter,
    PopupMessage,
    PopupSettings,
    Win32PopupPresenter,
)


@dataclass
class RecordingPresenter:
    messages: list[PopupMessage] = field(default_factory=list)
    fail: bool = False

    def show(self, message: PopupMessage) -> None:
        if self.fail:
            raise RuntimeError("popup unavailable")
        self.messages.append(message)


@dataclass
class RecordingAdapter:
    name: str = "voice"
    received: list[NotificationEvent] = field(default_factory=list)

    def send(self, event: NotificationEvent) -> None:
        self.received.append(event)


def test_popup_adapter_localizes_event_and_test_action() -> None:
    presenter = RecordingPresenter()
    adapter = PopupAdapter(presenter, PopupSettings(locale="uk", duration_ms=4000))

    adapter.send(NotificationEvent(EventType.ARRIVED, Severity.INFO))
    adapter.test_popup()

    assert presenter.messages[0] == PopupMessage(
        title="BDO Sea Assistant",
        body="Прибули.",
        severity=Severity.INFO,
        duration_ms=4000,
    )
    assert presenter.messages[1].body == "Перевірка спливаючих сповіщень."


def test_disabled_popup_is_silent() -> None:
    presenter = RecordingPresenter()
    adapter = PopupAdapter(presenter, PopupSettings(enabled=False))

    adapter.send(NotificationEvent(EventType.STUCK, Severity.WARNING))
    adapter.test_popup()

    assert presenter.messages == []


def test_popup_failure_does_not_block_later_adapter() -> None:
    popup = PopupAdapter(RecordingPresenter(fail=True), PopupSettings())
    voice = RecordingAdapter()
    event = NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL)

    results = NotificationRouter([popup, voice]).route(event)

    assert [result.success for result in results] == [False, True]
    assert voice.received == [event]


@pytest.mark.parametrize(
    "settings",
    [
        PopupSettings(locale="de"),
        PopupSettings(duration_ms=499),
        PopupSettings(duration_ms=60_001),
    ],
)
def test_popup_rejects_invalid_settings(settings: PopupSettings) -> None:
    with pytest.raises(ValueError, match="popup settings"):
        PopupAdapter(RecordingPresenter(), settings)


def test_win32_presenter_applies_no_activate_styles_and_flags() -> None:
    class User32:
        def __init__(self) -> None:
            self.calls = []

        def GetWindowLongW(self, hwnd, index):
            self.calls.append(("get", hwnd, index))
            return 0x10

        def SetWindowLongW(self, hwnd, index, style):
            self.calls.append(("style", hwnd, index, style))

        def ShowWindow(self, hwnd, command):
            self.calls.append(("show", hwnd, command))

        def SetWindowPos(self, hwnd, after, x, y, width, height, flags):
            self.calls.append(("pos", hwnd, after, x, y, width, height, flags))

    user32 = User32()
    Win32PopupPresenter._apply_no_activate(user32, 123, 10, 20, 380, 100)

    style = next(call[3] for call in user32.calls if call[0] == "style")
    show = next(call for call in user32.calls if call[0] == "show")
    position = next(call for call in user32.calls if call[0] == "pos")
    assert style & 0x08000000  # WS_EX_NOACTIVATE
    assert style & 0x00000080  # WS_EX_TOOLWINDOW
    assert show == ("show", 123, 4)  # SW_SHOWNOACTIVATE
    assert position[-1] & 0x0010  # SWP_NOACTIVATE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "", "body": "body", "severity": Severity.INFO, "duration_ms": 1000},
        {"title": "title", "body": "", "severity": Severity.INFO, "duration_ms": 1000},
        {"title": "title", "body": "body", "severity": "info", "duration_ms": 1000},
        {"title": "title", "body": "body", "severity": Severity.INFO, "duration_ms": 1},
    ],
)
def test_popup_message_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="popup message"):
        PopupMessage(**kwargs)
