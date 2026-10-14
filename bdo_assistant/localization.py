from __future__ import annotations

from .notifications import EventType, NotificationEvent


_TEMPLATES = {
    "uk": {
        EventType.ARRIVED: "Прибули: {destination}.",
        EventType.STUCK: "Схоже, корабель застряг. Немає прогресу маршрутом уже {seconds} с.",
        EventType.SHIP_HP_LOW: "Міцність корабля низька: {percent}%.",
        EventType.RATIONS_LOW: "Запаси провізії низькі: {percent}%.",
        EventType.CAPTURE_LOST: "Втрачено захоплення вікна Black Desert. Перевірте, чи гра відкрита.",
        EventType.CAPTURE_RESTORED: "Захоплення вікна Black Desert відновлено.",
        EventType.MINIMAP_VISIBILITY_LOST: "Асистент не бачить мінікарту. Закрийте глобальну карту або перевірте розташування ігрового UI.",
        EventType.MINIMAP_VISIBILITY_RESTORED: "Мінікарту знову видно. Моніторинг навігації відновлено.",
        EventType.UI_CALIBRATION_LOST: "Втрачено калібрування інтерфейсу. Перевірте масштаб і розташування HUD.",
    },
    "en": {
        EventType.ARRIVED: "Arrived: {destination}.",
        EventType.STUCK: "The ship may be stuck. No route progress for {seconds} s.",
        EventType.SHIP_HP_LOW: "Ship durability is low: {percent}%.",
        EventType.RATIONS_LOW: "Rations are low: {percent}%.",
        EventType.CAPTURE_LOST: "Black Desert window capture was lost. Check whether the game is open.",
        EventType.CAPTURE_RESTORED: "Black Desert window capture was restored.",
        EventType.MINIMAP_VISIBILITY_LOST: "The assistant cannot see the minimap. Close the Global Map or check the game UI layout.",
        EventType.MINIMAP_VISIBILITY_RESTORED: "The minimap is visible again. Navigation monitoring resumed.",
        EventType.UI_CALIBRATION_LOST: "UI calibration was lost. Check the HUD scale and layout.",
    },
}


class Localizer:
    def __init__(self, locale: str) -> None:
        if locale not in _TEMPLATES:
            raise ValueError(f"unsupported locale: {locale}")
        self.locale = locale

    def format(self, event: NotificationEvent) -> str:
        template = _TEMPLATES[self.locale][event.event_type]
        return template.format_map(event.parameters)
