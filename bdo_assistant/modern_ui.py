from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real


class PageId(str, Enum):
    DASHBOARD = "dashboard"
    CAPTURE = "capture"
    CALIBRATION = "calibration"
    ALERTS = "alerts"
    SETTINGS = "settings"
    RECORDER = "recorder"


@dataclass(frozen=True)
class PageDefinition:
    page_id: PageId
    title_uk: str
    title_en: str
    icon: str


APP_PAGES = (
    PageDefinition(PageId.DASHBOARD, "Огляд", "Dashboard", "⌂"),
    PageDefinition(PageId.CAPTURE, "Захоплення", "Capture", "◉"),
    PageDefinition(PageId.CALIBRATION, "Калібрування", "Calibration", "⌗"),
    PageDefinition(PageId.ALERTS, "Сповіщення", "Alerts", "◌"),
    PageDefinition(PageId.SETTINGS, "Налаштування", "Settings", "⚙"),
)

RECORDER_PAGE = PageDefinition(PageId.RECORDER, "Запис", "Recorder", "●")


_ALLOWED_STATES = {
    "monitoring": {"stopped", "starting", "running", "stopping", "error"},
    "capture": {"unknown", "active", "lost"},
    "calibration": {"unknown", "valid", "lost"},
    "minimap": {"unknown", "visible", "lost"},
    "resources": {"unknown", "visible", "absent"},
    "navigation": {
        "unknown",
        "no_ship",
        "moving",
        "stationary",
        "arrived",
        "departed",
        "stuck",
        "recovering",
    },
}


@dataclass(frozen=True)
class DashboardSnapshot:
    monitoring: str = "stopped"
    capture: str = "unknown"
    calibration: str = "unknown"
    minimap: str = "unknown"
    navigation: str = "unknown"
    hp_percent: float | None = None
    rations_percent: float | None = None
    resources: str = "unknown"
    window_title: str | None = None
    diagnostic: str = ""

    def validate(self) -> None:
        for field_name, allowed in _ALLOWED_STATES.items():
            if getattr(self, field_name) not in allowed:
                raise ValueError(f"invalid dashboard state: {field_name}")
        for value in (self.hp_percent, self.rations_percent):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or not 0 <= float(value) <= 100
            ):
                raise ValueError("resource percentages must be finite values from 0 to 100")
        if self.window_title is not None and (
            type(self.window_title) is not str or not self.window_title.strip()
        ):
            raise ValueError("window title must be a non-empty string or None")
        if type(self.diagnostic) is not str:
            raise ValueError("diagnostic must be a string")


@dataclass(frozen=True)
class NavigationItem:
    page_id: PageId
    title: str
    icon: str


class ModernUiModel:
    def __init__(self, *, locale: str = "uk", developer_recorder: bool = False) -> None:
        self.pages = APP_PAGES + ((RECORDER_PAGE,) if developer_recorder else ())
        self._locale = "uk"
        self._dashboard = DashboardSnapshot()
        self.set_locale(locale)

    @property
    def locale(self) -> str:
        return self._locale

    @property
    def dashboard(self) -> DashboardSnapshot:
        return self._dashboard

    def set_locale(self, locale: str) -> None:
        if type(locale) is not str or locale not in ("uk", "en"):
            raise ValueError("locale must be uk or en")
        self._locale = locale

    def navigation_items(self) -> tuple[NavigationItem, ...]:
        return tuple(
            NavigationItem(
                page.page_id,
                page.title_uk if self._locale == "uk" else page.title_en,
                page.icon,
            )
            for page in self.pages
        )

    def update_dashboard(self, snapshot: DashboardSnapshot) -> None:
        if type(snapshot) is not DashboardSnapshot:
            raise ValueError("snapshot must be DashboardSnapshot")
        snapshot.validate()
        self._dashboard = snapshot
