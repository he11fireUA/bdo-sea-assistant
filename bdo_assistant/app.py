from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class AlphaAppSettings:
    locale: str = "uk"
    voice_enabled: bool = True
    popup_enabled: bool = True
    voice_volume: float = 0.8
    popup_duration_ms: int = 5000
    minimap_loss_delay_seconds: int = 10
    ship_hp_warning_percent: int = 60
    ship_rations_warning_percent: int = 10
    repeat_alerts_enabled: bool = False
    repeat_alert_interval_seconds: int = 60

    def validate(self) -> None:
        if (
            self.locale not in ("uk", "en")
            or type(self.voice_enabled) is not bool
            or type(self.popup_enabled) is not bool
            or isinstance(self.voice_volume, bool)
            or not isinstance(self.voice_volume, Real)
            or not isfinite(float(self.voice_volume))
            or not 0 <= float(self.voice_volume) <= 1
            or type(self.popup_duration_ms) is not int
            or not 500 <= self.popup_duration_ms <= 60_000
            or type(self.minimap_loss_delay_seconds) is not int
            or not 10 <= self.minimap_loss_delay_seconds <= 300
            or type(self.ship_hp_warning_percent) is not int
            or not 1 <= self.ship_hp_warning_percent <= 99
            or type(self.ship_rations_warning_percent) is not int
            or not 1 <= self.ship_rations_warning_percent <= 99
            or type(self.repeat_alerts_enabled) is not bool
            or type(self.repeat_alert_interval_seconds) is not int
            or not 30 <= self.repeat_alert_interval_seconds <= 300
        ):
            raise ValueError("app settings are invalid")


class JsonSettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> AlphaAppSettings:
        if not self.path.exists():
            return AlphaAppSettings()
        try:
            values = json.loads(self.path.read_text(encoding="utf-8"))
            settings = AlphaAppSettings(**values)
            settings.validate()
        except (OSError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"app settings file is invalid: {self.path}") from exc
        return settings

    def save(self, settings: AlphaAppSettings) -> None:
        if type(settings) is not AlphaAppSettings:
            raise ValueError("app settings must be AlphaAppSettings")
        settings.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class AlphaAppController:
    def __init__(
        self,
        store: JsonSettingsStore,
        *,
        voice_factory: Callable[[AlphaAppSettings], object],
        popup_factory: Callable[[AlphaAppSettings], object],
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self._voice_factory = voice_factory
        self._popup_factory = popup_factory
        if close_callback is not None and not callable(close_callback):
            raise ValueError("close_callback must be callable or None")
        self._close_callback = close_callback
        self.settings = store.load()
        self._rebuild_adapters()

    def _rebuild_adapters(self) -> None:
        self._voice = self._voice_factory(self.settings)
        self._popup = self._popup_factory(self.settings)

    def apply(self, settings: AlphaAppSettings) -> None:
        self.store.save(settings)
        self.settings = settings
        self._rebuild_adapters()

    def test_voice(self) -> None:
        self._voice.test_voice()

    def test_popup(self) -> None:
        self._popup.test_popup()

    def enabled_adapters(self) -> tuple[object, ...]:
        adapters = []
        if self.settings.voice_enabled:
            adapters.append(self._voice)
        if self.settings.popup_enabled:
            adapters.append(self._popup)
        return tuple(adapters)

    def close(self) -> None:
        if self._close_callback is not None:
            self._close_callback()
