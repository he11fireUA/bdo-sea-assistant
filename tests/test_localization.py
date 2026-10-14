import pytest

from bdo_assistant.localization import Localizer
from bdo_assistant.notifications import EventType, NotificationEvent, Severity


def test_localizer_formats_arrival_in_ukrainian_and_english() -> None:
    event = NotificationEvent(
        EventType.ARRIVED,
        Severity.INFO,
        {"destination": "Iliya Island"},
    )

    assert Localizer("uk").format(event) == "Прибули: Iliya Island."
    assert Localizer("en").format(event) == "Arrived: Iliya Island."


def test_minimap_visibility_messages_have_actionable_recovery_text() -> None:
    lost = NotificationEvent(EventType.MINIMAP_VISIBILITY_LOST, Severity.WARNING)
    restored = NotificationEvent(EventType.MINIMAP_VISIBILITY_RESTORED, Severity.INFO)

    assert "глобальну карту" in Localizer("uk").format(lost)
    assert "Global Map" in Localizer("en").format(lost)
    assert Localizer("uk").format(restored) == "Мінікарту знову видно. Моніторинг навігації відновлено."


def test_localizer_rejects_unsupported_locale() -> None:
    with pytest.raises(ValueError, match="unsupported locale"):
        Localizer("de")
