from dataclasses import dataclass, field
from pathlib import Path
import threading

import pytest

from bdo_assistant.app import AlphaAppController, AlphaAppSettings, JsonSettingsStore
from bdo_assistant import desktop_ui
from bdo_assistant.desktop_ui import create_tray_image, default_settings_path
from bdo_assistant.notifications import EventType, NotificationEvent, Severity


@dataclass
class FakeAdapter:
    name: str
    tests: list[str] = field(default_factory=list)

    def send(self, event) -> None:
        pass

    def test_voice(self) -> None:
        self.tests.append("voice")

    def test_popup(self) -> None:
        self.tests.append("popup")


def test_settings_round_trip_and_atomic_save(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = JsonSettingsStore(path)
    settings = AlphaAppSettings(
        locale="en",
        voice_enabled=False,
        popup_enabled=True,
        voice_volume=0.65,
        popup_duration_ms=4500,
        minimap_loss_delay_seconds=75,
        ship_hp_warning_percent=73,
        ship_rations_warning_percent=27,
        repeat_alerts_enabled=True,
        repeat_alert_interval_seconds=90,
    )

    store.save(settings)

    assert store.load() == settings
    assert not (tmp_path / "settings.json.tmp").exists()


def test_missing_settings_file_returns_defaults(tmp_path: Path) -> None:
    assert JsonSettingsStore(tmp_path / "missing.json").load() == AlphaAppSettings()


def test_default_controller_voices_minimap_visibility_restored(monkeypatch, tmp_path) -> None:
    class AudioBackend:
        def __init__(self) -> None:
            self.calls = []
            self.phrase_played = threading.Event()

        def play(self, path, volume) -> None:
            self.calls.append((path, volume))
            if path.name == "minimap_visibility_restored.wav":
                self.phrase_played.set()

    class PopupPresenter:
        def show(self, message) -> None:
            pass

    audio = AudioBackend()
    monkeypatch.setattr(desktop_ui, "SoundDevicePlaybackBackend", lambda: audio)
    monkeypatch.setattr(desktop_ui, "Win32PopupPresenter", lambda **kwargs: PopupPresenter())
    controller = desktop_ui.build_default_controller(
        project_root=Path(__file__).parents[1],
        settings_path=tmp_path / "settings.json",
    )
    voice = controller.enabled_adapters()[0]

    voice.send(
        NotificationEvent(EventType.MINIMAP_VISIBILITY_RESTORED, Severity.INFO)
    )

    assert audio.phrase_played.wait(2)
    assert [path.name for path, _ in audio.calls] == [
        "notification_bell.wav",
        "minimap_visibility_restored.wav",
    ]


@pytest.mark.parametrize(
    "settings",
    [
        AlphaAppSettings(locale="de"),
        AlphaAppSettings(voice_volume=-0.1),
        AlphaAppSettings(voice_volume=1.1),
        AlphaAppSettings(popup_duration_ms=100),
        AlphaAppSettings(minimap_loss_delay_seconds=9),
        AlphaAppSettings(minimap_loss_delay_seconds=301),
        AlphaAppSettings(minimap_loss_delay_seconds=10.5),
        AlphaAppSettings(ship_hp_warning_percent=0),
        AlphaAppSettings(ship_hp_warning_percent=100),
        AlphaAppSettings(ship_hp_warning_percent=True),
        AlphaAppSettings(ship_rations_warning_percent=0),
        AlphaAppSettings(ship_rations_warning_percent=100),
        AlphaAppSettings(ship_rations_warning_percent=12.5),
    ],
)
def test_settings_store_rejects_invalid_values(tmp_path: Path, settings: AlphaAppSettings) -> None:
    with pytest.raises(ValueError, match="app settings"):
        JsonSettingsStore(tmp_path / "settings.json").save(settings)


def test_controller_saves_settings_builds_enabled_adapters_and_runs_tests(tmp_path: Path) -> None:
    voices = []
    popups = []

    def voice_factory(settings):
        adapter = FakeAdapter("voice")
        voices.append((settings, adapter))
        return adapter

    def popup_factory(settings):
        adapter = FakeAdapter("popup")
        popups.append((settings, adapter))
        return adapter

    controller = AlphaAppController(
        JsonSettingsStore(tmp_path / "settings.json"),
        voice_factory=voice_factory,
        popup_factory=popup_factory,
    )
    settings = AlphaAppSettings(locale="en", voice_volume=0.5)

    controller.apply(settings)
    controller.test_voice()
    controller.test_popup()
    adapters = controller.enabled_adapters()

    assert JsonSettingsStore(tmp_path / "settings.json").load() == settings
    assert voices[-1][1].tests == ["voice"]
    assert popups[-1][1].tests == ["popup"]
    assert [adapter.name for adapter in adapters] == ["voice", "popup"]


def test_controller_closes_registered_notification_worker(tmp_path: Path) -> None:
    closed = []
    controller = AlphaAppController(
        JsonSettingsStore(tmp_path / "settings.json"),
        voice_factory=lambda settings: FakeAdapter("voice"),
        popup_factory=lambda settings: FakeAdapter("popup"),
        close_callback=lambda: closed.append("voice"),
    )

    controller.close()

    assert closed == ["voice"]


def test_default_settings_path_uses_local_app_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert default_settings_path() == tmp_path / "BDOSeaAssistant" / "settings.json"


def test_tray_image_has_expected_rgba_size() -> None:
    image = create_tray_image()

    assert image.mode == "RGBA"
    assert image.size == (64, 64)
