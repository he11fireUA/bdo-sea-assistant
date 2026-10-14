from __future__ import annotations

from dataclasses import dataclass
import queue
import sys
import threading
from typing import Protocol

from .localization import Localizer
from .notifications import EventType, NotificationEvent, Severity


_GENERIC_TEXT = {
    "uk": {
        EventType.ARRIVED: "Прибули.",
        EventType.STUCK: "Схоже, корабель застряг.",
        EventType.SHIP_HP_LOW: "Міцність корабля низька.",
        EventType.RATIONS_LOW: "Запаси провізії низькі.",
        EventType.CAPTURE_LOST: "Втрачено захоплення гри.",
        EventType.CAPTURE_RESTORED: "Захоплення гри відновлено.",
        EventType.MINIMAP_VISIBILITY_LOST: "Мінікарту не видно.",
        EventType.MINIMAP_VISIBILITY_RESTORED: "Мінікарту знову видно.",
        EventType.UI_CALIBRATION_LOST: "Втрачено калібрування інтерфейсу.",
    },
    "en": {
        EventType.ARRIVED: "Arrived.",
        EventType.STUCK: "The ship appears to be stuck.",
        EventType.SHIP_HP_LOW: "Ship durability is low.",
        EventType.RATIONS_LOW: "Rations are low.",
        EventType.CAPTURE_LOST: "Game capture has been lost.",
        EventType.CAPTURE_RESTORED: "Game capture has been restored.",
        EventType.MINIMAP_VISIBILITY_LOST: "The minimap is not visible.",
        EventType.MINIMAP_VISIBILITY_RESTORED: "The minimap is visible again.",
        EventType.UI_CALIBRATION_LOST: "Interface calibration has been lost.",
    },
}


@dataclass(frozen=True)
class PopupSettings:
    enabled: bool = True
    locale: str = "uk"
    duration_ms: int = 5000


@dataclass(frozen=True)
class PopupMessage:
    title: str
    body: str
    severity: Severity
    duration_ms: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.title, str)
            or not self.title.strip()
            or not isinstance(self.body, str)
            or not self.body.strip()
            or not isinstance(self.severity, Severity)
            or type(self.duration_ms) is not int
            or not 500 <= self.duration_ms <= 60_000
        ):
            raise ValueError("popup message fields are invalid")


class PopupPresenter(Protocol):
    def show(self, message: PopupMessage) -> None: ...


class Win32PopupPresenter:
    """Queued topmost popup that uses the Win32 no-activate window style."""

    def __init__(self, *, width: int = 380) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Win32PopupPresenter requires Windows")
        if type(width) is not int or not 200 <= width <= 1000:
            raise ValueError("popup width must be between 200 and 1000")
        self._width = width
        self._messages: queue.Queue[PopupMessage] = queue.Queue()
        self._last_error: str | None = None
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="bdo-popup-presenter",
            daemon=True,
        )
        self._worker.start()

    def show(self, message: PopupMessage) -> None:
        if type(message) is not PopupMessage:
            raise ValueError("message must be PopupMessage")
        self._messages.put_nowait(message)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _worker_loop(self) -> None:
        while True:
            message = self._messages.get()
            try:
                self._display(message)
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._messages.task_done()

    @staticmethod
    def _apply_no_activate(user32, hwnd: int, x: int, y: int, width: int, height: int) -> None:
        gwl_exstyle = -20
        ws_ex_toolwindow = 0x00000080
        ws_ex_noactivate = 0x08000000
        style = user32.GetWindowLongW(hwnd, gwl_exstyle)
        user32.SetWindowLongW(
            hwnd,
            gwl_exstyle,
            style | ws_ex_toolwindow | ws_ex_noactivate,
        )
        sw_shownoactivate = 4
        swp_noactivate = 0x0010
        swp_showwindow = 0x0040
        hwnd_topmost = -1
        user32.ShowWindow(hwnd, sw_shownoactivate)
        user32.SetWindowPos(
            hwnd,
            hwnd_topmost,
            x,
            y,
            width,
            height,
            swp_noactivate | swp_showwindow,
        )

    def _display(self, message: PopupMessage) -> None:
        import ctypes
        import tkinter as tk

        colors = {
            Severity.INFO: ("#182433", "#7dd3fc"),
            Severity.WARNING: ("#33260f", "#fbbf24"),
            Severity.CRITICAL: ("#351417", "#fb7185"),
        }
        background, accent = colors[message.severity]
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.configure(background=background)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.96)
        frame = tk.Frame(root, background=background, padx=16, pady=12)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text=message.title,
            foreground=accent,
            background=background,
            font=("Segoe UI Semibold", 11),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            frame,
            text=message.body,
            foreground="#f8fafc",
            background=background,
            font=("Segoe UI", 10),
            justify="left",
            anchor="w",
            wraplength=self._width - 32,
        ).pack(fill="x", pady=(6, 0))
        root.update_idletasks()
        height = root.winfo_reqheight()
        x = root.winfo_screenwidth() - self._width - 24
        y = root.winfo_screenheight() - height - 64
        root.geometry(f"{self._width}x{height}+{x}+{y}")
        root.update_idletasks()

        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        self._apply_no_activate(user32, hwnd, x, y, self._width, height)
        root.after(message.duration_ms, root.destroy)
        root.mainloop()


class PopupAdapter:
    name = "popup"

    def __init__(self, presenter: PopupPresenter, settings: PopupSettings) -> None:
        if (
            type(settings) is not PopupSettings
            or type(settings.enabled) is not bool
            or settings.locale not in _GENERIC_TEXT
            or type(settings.duration_ms) is not int
            or not 500 <= settings.duration_ms <= 60_000
        ):
            raise ValueError("popup settings are invalid")
        if not callable(getattr(presenter, "show", None)):
            raise ValueError("popup presenter must provide callable show")
        self._presenter = presenter
        self.settings = settings
        self._localizer = Localizer(settings.locale)

    def _body(self, event: NotificationEvent) -> str:
        try:
            return self._localizer.format(event)
        except KeyError:
            return _GENERIC_TEXT[self.settings.locale][event.event_type]

    def send(self, event: NotificationEvent) -> None:
        if not self.settings.enabled:
            return
        self._presenter.show(
            PopupMessage(
                title="BDO Sea Assistant",
                body=self._body(event),
                severity=event.severity,
                duration_ms=self.settings.duration_ms,
            )
        )

    def test_popup(self) -> None:
        if not self.settings.enabled:
            return
        body = (
            "Перевірка спливаючих сповіщень."
            if self.settings.locale == "uk"
            else "Popup alerts test."
        )
        self._presenter.show(
            PopupMessage(
                title="BDO Sea Assistant",
                body=body,
                severity=Severity.INFO,
                duration_ms=self.settings.duration_ms,
            )
        )
