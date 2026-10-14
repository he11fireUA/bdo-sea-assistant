from __future__ import annotations

from pathlib import Path
import threading

from .app import AlphaAppController, AlphaAppSettings, JsonSettingsStore
from .app_paths import app_data_dir, bundle_root
from .popup import PopupAdapter, PopupSettings, Win32PopupPresenter
from .voice import (
    OfflineVoiceAdapter,
    QueuedVoicePlaybackScheduler,
    SoundDevicePlaybackBackend,
    VoiceSettings,
)


def default_settings_path() -> Path:
    return app_data_dir() / "settings.json"


def build_default_controller(
    *,
    project_root: Path | None = None,
    settings_path: Path | None = None,
) -> AlphaAppController:
    project_root = project_root or bundle_root()
    voice_root = project_root / "assets" / "voice"
    audio_backend = SoundDevicePlaybackBackend()
    voice_scheduler = QueuedVoicePlaybackScheduler(audio_backend)
    popup_presenter = Win32PopupPresenter(width=380)

    def voice_factory(settings: AlphaAppSettings) -> OfflineVoiceAdapter:
        return OfflineVoiceAdapter(
            voice_root,
            audio_backend,
            VoiceSettings(
                enabled=settings.voice_enabled,
                locale=settings.locale,
                volume=settings.voice_volume,
                announce_restored=True,
            ),
            scheduler=voice_scheduler,
        )

    def popup_factory(settings: AlphaAppSettings) -> PopupAdapter:
        return PopupAdapter(
            popup_presenter,
            PopupSettings(
                enabled=settings.popup_enabled,
                locale=settings.locale,
                duration_ms=settings.popup_duration_ms,
            ),
        )

    return AlphaAppController(
        JsonSettingsStore(settings_path or default_settings_path()),
        voice_factory=voice_factory,
        popup_factory=popup_factory,
        close_callback=lambda: voice_scheduler.close(timeout=0),
    )


def create_tray_image():
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), "#0f172a")
    draw = ImageDraw.Draw(image)
    draw.polygon([(10, 39), (54, 39), (46, 51), (18, 51)], fill="#38bdf8")
    draw.polygon([(31, 9), (31, 37), (50, 37)], fill="#f8fafc")
    draw.line([(31, 8), (31, 40)], fill="#f8fafc", width=3)
    return image


class AlphaDesktopWindow:
    def __init__(self, controller: AlphaAppController) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.controller = controller
        self.root = tk.Tk()
        self.root.title("BDO Sea Assistant — Alpha")
        self.root.geometry("520x420")
        self.root.minsize(480, 380)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        settings = controller.settings
        self.locale = tk.StringVar(value=settings.locale)
        self.voice_enabled = tk.BooleanVar(value=settings.voice_enabled)
        self.popup_enabled = tk.BooleanVar(value=settings.popup_enabled)
        self.voice_volume = tk.DoubleVar(value=settings.voice_volume)
        self.popup_duration = tk.IntVar(value=settings.popup_duration_ms)
        self.status = tk.StringVar(value="Моніторинг: не запущено")

        main = ttk.Frame(self.root, padding=18)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="BDO Sea Assistant", font=("Segoe UI Semibold", 18)).pack(
            anchor="w"
        )
        ttk.Label(
            main,
            text="Пасивний WGC-only морський асистент · 2D minimap",
        ).pack(anchor="w", pady=(2, 16))
        status_frame = ttk.LabelFrame(main, text="Стан", padding=10)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status).pack(anchor="w")

        settings_frame = ttk.LabelFrame(main, text="Сповіщення", padding=10)
        settings_frame.pack(fill="x", pady=12)
        row = ttk.Frame(settings_frame)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Мова", width=18).pack(side="left")
        ttk.Combobox(
            row,
            textvariable=self.locale,
            values=("uk", "en"),
            state="readonly",
            width=8,
        ).pack(side="left")
        ttk.Checkbutton(
            settings_frame,
            text="Offline Voice",
            variable=self.voice_enabled,
        ).pack(anchor="w", pady=3)
        ttk.Checkbutton(
            settings_frame,
            text="Non-focus Popup",
            variable=self.popup_enabled,
        ).pack(anchor="w", pady=3)
        volume_row = ttk.Frame(settings_frame)
        volume_row.pack(fill="x", pady=3)
        ttk.Label(volume_row, text="Гучність Voice", width=18).pack(side="left")
        ttk.Scale(
            volume_row,
            from_=0,
            to=1,
            variable=self.voice_volume,
            orient="horizontal",
        ).pack(side="left", fill="x", expand=True)
        duration_row = ttk.Frame(settings_frame)
        duration_row.pack(fill="x", pady=3)
        ttk.Label(duration_row, text="Popup, мс", width=18).pack(side="left")
        ttk.Spinbox(
            duration_row,
            from_=500,
            to=60_000,
            increment=500,
            textvariable=self.popup_duration,
            width=10,
        ).pack(side="left")

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="Застосувати", command=self.apply).pack(side="left")
        ttk.Button(buttons, text="Test Voice", command=self.test_voice).pack(
            side="left", padx=6
        )
        ttk.Button(buttons, text="Test Popup", command=self.test_popup).pack(side="left")
        ttk.Button(buttons, text="У трей", command=self.hide).pack(side="right")
        ttk.Label(
            main,
            text="Telegram перенесено на наступну версію. BDO не запускається автоматично.",
            foreground="#64748b",
            wraplength=470,
        ).pack(anchor="w", pady=(14, 0))
        threading.Thread(
            target=self._start_tray,
            name="bdo-tray",
            daemon=True,
        ).start()

    def _settings_from_form(self) -> AlphaAppSettings:
        return AlphaAppSettings(
            locale=self.locale.get(),
            voice_enabled=bool(self.voice_enabled.get()),
            popup_enabled=bool(self.popup_enabled.get()),
            voice_volume=float(self.voice_volume.get()),
            popup_duration_ms=int(self.popup_duration.get()),
        )

    def _action(self, label: str, callback) -> None:
        try:
            callback()
        except Exception as exc:
            self.status.set(f"{label}: помилка — {exc}")
        else:
            self.status.set(f"{label}: виконано")

    def apply(self) -> None:
        self._action("Налаштування", lambda: self.controller.apply(self._settings_from_form()))

    def test_voice(self) -> None:
        self.apply()
        self._action("Test Voice", self.controller.test_voice)

    def test_popup(self) -> None:
        self.apply()
        self._action("Test Popup", self.controller.test_popup)

    def show(self) -> None:
        self.root.after(0, self.root.deiconify)

    def hide(self) -> None:
        self.root.withdraw()

    def exit(self) -> None:
        if hasattr(self, "tray"):
            self.tray.stop()
        self.root.after(0, self.root.destroy)

    def _start_tray(self) -> None:
        import pystray

        self.tray = pystray.Icon(
            "bdo-sea-assistant",
            create_tray_image(),
            "BDO Sea Assistant",
            menu=pystray.Menu(
                pystray.MenuItem("Показати", lambda icon, item: self.show(), default=True),
                pystray.MenuItem("Вийти", lambda icon, item: self.exit()),
            ),
        )
        self.tray.run()

    def run(self) -> None:
        self.root.mainloop()


def run() -> None:
    AlphaDesktopWindow(build_default_controller()).run()
