from __future__ import annotations

import sys
import threading
from collections import deque
from datetime import datetime
from statistics import median
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, Signal, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .app import AlphaAppController, AlphaAppSettings
from .app_paths import app_data_dir, asset_path
from .build_identity import BuildIdentity, DEFAULT_BUILD_IDENTITY
from .calibration_store import (
    CalibrationReadiness,
    CalibrationStatus,
    JsonCalibrationStore,
    default_calibration_path,
)
from .calibration_capture import OneShotCalibrationCapture
from .calibration_flow import build_default_calibration_flow
from .dashboard_projection import project_runtime_status
from .modern_ui import DashboardSnapshot, ModernUiModel, PageId
from .runtime_readiness import RuntimeDependencyStatus, load_verified_minimap_anchor
from .runtime_factory import RuntimeFactoryInputs, build_runtime_bridge
from .runtime import WgcWindowFrameSource
from .minimap_replay_recorder import NativeMinimapReplayRecorder, NativeMinimapReplaySession

from .runtime_session import (
    RuntimeSessionController,
    RuntimeSessionSnapshot,
    RuntimeSessionState,
)
from .window_discovery import WindowCandidate, discover_bdo_windows
from .ui_locator import Rect


def _default_calibration_inspector() -> CalibrationReadiness:
    return JsonCalibrationStore(default_calibration_path()).inspect()


class _RuntimeSessionSignals(QObject):
    snapshot = Signal(object)
    runtime_status = Signal(object)


class _RecorderSignals(QObject):
    started = Signal(object)
    preview = Signal(object)
    terminal = Signal(object, object)


def application_icon_path() -> Path:
    """Return the packaged Windows icon used for application branding."""
    return asset_path("app_icon.ico")


def application_icon() -> QIcon:
    """Load the packaged Windows icon for windows and the Qt application."""
    return QIcon(str(application_icon_path()))


def configure_application(application: QApplication | None = None) -> QApplication:
    """Apply shared Qt application metadata and its taskbar/window icon."""
    application = application or QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("BDO Sea Assistant")
    application.setOrganizationName("BDO Sea Assistant")
    application.setWindowIcon(application_icon())
    return application


APP_STYLESHEET = """
QMainWindow, QWidget#appSurface {
    background: #07111f;
    color: #e8f1f8;
    font-family: "Segoe UI";
    font-size: 13px;
}
QWidget#pageBody, QScrollArea QWidget#qt_scrollarea_viewport {
    background: #07111f;
}
QFrame#sidebar {
    background: #091827;
    border-right: 1px solid #173047;
}
QLabel#brandTitle { font-size: 18px; font-weight: 700; color: #f2f8fc; }
QLabel#brandMeta { color: #6f8ca5; font-size: 11px; }
QPushButton[nav="true"] {
    background: transparent;
    color: #8da7bc;
    border: 0;
    border-radius: 9px;
    padding: 12px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 14px;
}
QPushButton[nav="true"]:hover { background: #10263a; color: #d9ecf7; }
QPushButton[nav="true"][active="true"] {
    background: #103149;
    color: #63d8ee;
    border-left: 3px solid #32c8df;
}
QFrame#topbar { background: #081522; border-bottom: 1px solid #173047; }
QLabel#pageTitle { font-size: 22px; font-weight: 700; color: #f5f9fc; }
QLabel#pageSubtitle, QLabel[muted="true"] { color: #7290a8; }
QLabel#eyebrow { color: #42cee4; font-weight: 700; font-size: 11px; }
QFrame[card="true"] {
    background: #0b1b2b;
    border: 1px solid #17354c;
    border-radius: 13px;
}
QFrame[card="true"]:hover { border-color: #23617a; }
QLabel[cardTitle="true"] { color: #86a2b7; font-weight: 600; }
QLabel[metric="true"] { color: #f3f8fb; font-size: 23px; font-weight: 700; }
QLabel[status="ok"] { color: #4ee0a0; font-weight: 700; }
QLabel[status="warn"] { color: #ffc857; font-weight: 700; }
QLabel[status="idle"] { color: #8ca4b8; font-weight: 700; }
QPushButton {
    background: #14354a;
    color: #dff4fa;
    border: 1px solid #25536b;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #18465f; border-color: #35bdd1; }
QPushButton#primaryButton { background: #22a9bd; color: #04131c; border-color: #39d3e8; }
QPushButton#primaryButton[settingsDirty="true"] { background: #76e8f4; color: #04131c; border-color: #b4f7ff; }
QPushButton#primaryButton:hover { background: #35c4d6; }
QPushButton[recorderMarker="true"][active="true"] { background: #22a9bd; color: #04131c; border-color: #39d3e8; }
QPushButton:disabled { background: #101e2a; color: #53697a; border-color: #1a2e3c; }
QComboBox, QSpinBox {
    background: #091725;
    color: #e4f0f6;
    border: 1px solid #244256;
    border-radius: 7px;
    padding: 7px 9px;
}
QComboBox QAbstractItemView { background: #0b1b2b; color: #e4f0f6; selection-background-color: #17465d; }
QCheckBox { color: #d6e4ec; spacing: 9px; }
QCheckBox::indicator { width: 17px; height: 17px; }
QProgressBar {
    background: #07141f;
    border: 1px solid #17374b;
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk { background: #32c8df; border-radius: 5px; }
QListWidget {
    background: #081725;
    color: #c9dae5;
    border: 1px solid #17354c;
    border-radius: 9px;
    padding: 6px;
}
QListWidget::item { padding: 9px; border-bottom: 1px solid #12283a; }
QScrollArea { border: 0; background: transparent; }
QSlider::groove:horizontal { height: 5px; background: #173448; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; margin: -6px 0; border-radius: 8px; background: #42cee4; }
"""


_UI_TEXT: dict[str, str] = {
    "Активна сторінка": "Active page",
    "Перейти до сторінки: {title}": "Go to page: {title}",
    "Стан системи та останні сигнали": "System status and recent signals",
    "Переносима геометрія HUD": "Portable HUD geometry",
    "Локальні незалежні канали": "Local independent channels",
    "Мова, канали й поведінка": "Language, channels, and behavior",
    "Локальний native minimap ROI запис": "Local native minimap ROI recording",
    "Морський моніторинг": "Sea monitoring",
    "Система не робить припущень до отримання підтверджених WGC-кадрів.": "The system makes no assumptions before receiving verified WGC frames.",
    "▶  Почати моніторинг": "▶  Start monitoring",
    "■  Зупинити": "■  Stop",
    "Runtime factory ще не підключений до modern UI": "Runtime factory is not connected to the modern UI yet",
    "ЗАХОПЛЕННЯ ВІКНА · WGC": "WINDOW CAPTURE · WGC",
    "Експортувати .txt": "Export .txt",
    "Сесія ще не запускалася — подій немає": "The session has not started yet — no events",
    "Запис маршруту": "Route recording",
    "Локальний запис minimap ROI з живими мітками сценарію.": "Local minimap ROI recording with live scenario markers.",
    "●  Старт": "●  Start",
    "■  Стоп": "■  Stop",
    "Прибув": "Arrived",
    "Застряг": "Stuck",
    "Рух": "Moving",
    "ПРЕВ’Ю ЗАПИСУ": "RECORDING PREVIEW",
    "Запис ще не розпочато": "Recording has not started yet",
    "Перегляд папки записів": "Open recordings folder",
    "Захоплення вікна": "Window capture",
    "Прив’язка єдиного вікна BDO через Windows Graphics Capture. Гра ніколи не запускається автоматично.": "Binding one BDO window through Windows Graphics Capture. The game is never launched automatically.",
    "Вікно BDO не вибрано": "No BDO window selected",
    "Натисни «Захопити вікно» для пасивної прив’язки HWND": "Click ‘Capture window’ to passively bind an HWND",
    "Захопити вікно": "Capture window",
    "Не перевірено": "Not checked",
    "Калібрування HUD": "HUD calibration",
    "Оновити стан": "Refresh status",
    "Почати калібрування": "Start calibration",
    "Підтвердити й зберегти": "Confirm and save",
    "Ручна корекція": "Manual correction",
    "Повторити": "Retry",
    "Сповіщення": "Alerts",
    "Offline Voice та non-focus Popup — незалежні failure domains. Telegram перенесено після alpha.": "Offline Voice and non-focus Popup are independent failure domains. Telegram is deferred until after alpha.",
    "Готово до явного тесту. Автоматичне програвання вимкнено.": "Ready for an explicit test. Automatic playback is disabled.",
    "Налаштування": "Settings",
    "Локальна конфігурація, канали та майбутній immutable release updater.": "Local configuration, channels, and a future immutable-release updater.",
    "Мова інтерфейсу": "Interface language",
    "Затримка втрати мінікарти": "Minimap loss delay",
    "Зона сповіщення HP корабля": "Ship HP warning zone",
    "Зона сповіщення раціонів": "Rations warning zone",
    "Повторне сповіщення": "Repeat alerts",
    "Інтервал повтору": "Repeat interval",
    "Застосувати": "Apply",
    "Збережено": "Saved",
    "Перевірити оновлення": "Check for updates",
    "с": "s",
    "●  OFFLINE / НЕ ЗАПУЩЕНО": "●  OFFLINE / NOT STARTED",
    "Пошук за executable identity + title · border/cursor capture disabled": "Search by executable identity + title · border/cursor capture disabled",
    "Відкрите вікно BlackDesert64.exe не знайдено": "No open BlackDesert64.exe window was found",
    "Усі runtime dependencies готові": "All runtime dependencies are ready",
    "Start заблоковано: {details}": "Start blocked: {details}",
    "Monitoring готовий з обмеженням: {details}": "Monitoring is ready with limitations: {details}",
    "Запис недоступний: {error}": "Recording unavailable: {error}",
    "Запуск WGC запису…": "Starting WGC recording…",
    "Очікування minimap кадру…": "Waiting for a minimap frame…",
    "Мітка не збережена: {error}": "Marker was not saved: {error}",
    "Фіналізація manifest…": "Finalizing manifest…",
    "Запис завершився з помилкою: {error}": "Recording ended with an error: {error}",
    "Папка записів недоступна: {error}": "Recordings folder unavailable: {error}",
    "Експорт подій: помилка — {error}": "Event export failed — {error}",
    "Експортовано події: {path}": "Events exported: {path}",
}


def _navigation_icon(page_id: PageId, color: str) -> QIcon:
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(color), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    if page_id is PageId.DASHBOARD:
        painter.drawPolyline(QPolygonF([QPointF(4, 11), QPointF(12, 4), QPointF(20, 11)]))
        painter.drawRoundedRect(QRectF(6, 10, 12, 10), 1.5, 1.5)
    elif page_id is PageId.CAPTURE:
        painter.drawRoundedRect(QRectF(3, 5, 18, 14), 2.5, 2.5)
        painter.drawEllipse(QPointF(12, 12), 3.2, 3.2)
    elif page_id is PageId.CALIBRATION:
        painter.drawEllipse(QPointF(12, 12), 5.5, 5.5)
        for a, b in ((QPointF(12, 2), QPointF(12, 7)), (QPointF(12, 17), QPointF(12, 22)), (QPointF(2, 12), QPointF(7, 12)), (QPointF(17, 12), QPointF(22, 12))):
            painter.drawLine(a, b)
    elif page_id is PageId.ALERTS:
        path = QPainterPath(QPointF(6, 17))
        path.quadTo(QPointF(7, 15), QPointF(7, 10))
        path.quadTo(QPointF(7, 5), QPointF(12, 5))
        path.quadTo(QPointF(17, 5), QPointF(17, 10))
        path.quadTo(QPointF(17, 15), QPointF(19, 17))
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawArc(QRectF(10, 17, 4, 4), 180 * 16, 180 * 16)
    elif page_id is PageId.RECORDER:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#ff5d6c"))
        painter.drawEllipse(QPointF(12, 12), 6.0, 6.0)
        painter.setPen(QPen(QColor(color), 1.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    else:
        painter.drawEllipse(QPointF(12, 12), 4.0, 4.0)
        painter.drawEllipse(QPointF(12, 12), 8.0, 8.0)
        painter.drawLine(QPointF(12, 2), QPointF(12, 5))
        painter.drawLine(QPointF(12, 19), QPointF(12, 22))
        painter.drawLine(QPointF(2, 12), QPointF(5, 12))
        painter.drawLine(QPointF(19, 12), QPointF(22, 12))
    painter.end()
    return QIcon(pixmap)


class ResourceGauge(QWidget):
    def __init__(self, title: str, color: str) -> None:
        super().__init__()
        self.title = title
        self.color = QColor(color)
        self.percent: float | None = None
        self._recent_percents: deque[float] = deque(maxlen=3)
        self._stable_readings: deque[int] = deque(maxlen=5)
        self.setMinimumHeight(116)

    def set_percent(self, percent: float | None) -> None:
        if percent is None:
            self._recent_percents.clear()
            self._stable_readings.clear()
            self.percent = None
        else:
            self._recent_percents.append(float(percent))
            self._stable_readings.append(round(percent))
            filtered = float(median(self._recent_percents))
            stable = len(self._stable_readings) == 5 and len(set(self._stable_readings)) == 1
            if self.percent is None or abs(filtered - self.percent) > 2.0 or stable:
                self.percent = filtered
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#081725"))
        painter.drawRoundedRect(rect, 12, 12)
        painter.setPen(QColor("#86a2b7"))
        painter.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        painter.drawText(rect.adjusted(16, 13, -16, 0), Qt.AlignTop | Qt.AlignLeft, self.title)
        value = "—" if self.percent is None else f"{self.percent:.0f}%"
        painter.setPen(QColor("#f3f8fb"))
        painter.setFont(QFont("Segoe UI", 24, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, value)
        track = rect.adjusted(16, rect.height() - 32, -16, -17)
        painter.setBrush(QColor("#142b3c"))
        painter.drawRoundedRect(track, 5, 5)
        if self.percent is not None:
            fill = track.adjusted(0, 0, -int(track.width() * (1 - self.percent / 100)), 0)
            painter.setBrush(self.color)
            painter.drawRoundedRect(fill, 5, 5)


class AspectFitPreviewLabel(QLabel):
    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._source_pixmap = QPixmap()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 180)

    def setPixmap(self, pixmap: QPixmap) -> None:  # noqa: N802 - Qt API
        self._source_pixmap = QPixmap(pixmap)
        self._rescale()

    def clear(self) -> None:
        self._source_pixmap = QPixmap()
        super().clear()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        super().setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class ModernMainWindow(QMainWindow):
    def __init__(
        self,
        controller: AlphaAppController,
        *,
        window_discovery: Callable[[], tuple[WindowCandidate, ...]] = discover_bdo_windows,
        calibration_inspector: Callable[[], CalibrationReadiness] = _default_calibration_inspector,
        runtime_session: object | None = None,
        route_evidence_available: bool = False,
        calibration_flow: object | None = None,
        calibration_frame_provider: Callable[[], np.ndarray] | None = None,
        calibration_destination: str | Path | None = None,
        manual_region_provider: Callable[[object], object] | None = None,
        calibration_requires_selected_window: bool = False,
        event_time_provider: Callable[[], str] | None = None,
        event_export_path_provider: Callable[[], str | Path | None] | None = None,
        developer_recorder: bool = False,
        recorder_source_factory: Callable[[int], object] = WgcWindowFrameSource,
        recordings_directory: str | Path | None = None,
        recordings_folder_opener: Callable[[Path], bool] | None = None,
        build_identity: BuildIdentity | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        if build_identity is None:
            build_identity = DEFAULT_BUILD_IDENTITY
        if not isinstance(build_identity, BuildIdentity):
            raise ValueError("build_identity must be BuildIdentity or None")
        self._build_identity = build_identity
        self.runtime_diagnostics: deque[str] = deque(maxlen=200)
        if not callable(recorder_source_factory):
            raise ValueError("recorder_source_factory must be callable")
        self._recorder_source_factory = recorder_source_factory
        self._recordings_directory = Path(recordings_directory or app_data_dir() / "recordings")
        if recordings_folder_opener is None:
            recordings_folder_opener = lambda directory: QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
        if not callable(recordings_folder_opener):
            raise ValueError("recordings_folder_opener must be callable")
        self._recordings_folder_opener = recordings_folder_opener
        self._recorder_session: NativeMinimapReplaySession | None = None
        self._recorder_rect: Rect | None = None
        self._recorder_signals = _RecorderSignals(self)
        self._recorder_signals.started.connect(self._on_recorder_started)
        self._recorder_signals.preview.connect(self._on_recorder_preview)
        self._recorder_signals.terminal.connect(self._on_recorder_terminal)
        if not callable(window_discovery):
            raise ValueError("window_discovery must be callable")
        self._window_discovery = window_discovery
        if event_time_provider is None:
            event_time_provider = lambda: datetime.now().strftime("%H:%M:%S")
        if not callable(event_time_provider):
            raise ValueError("event_time_provider must be callable")
        self._event_time_provider = event_time_provider
        if event_export_path_provider is None:
            def event_export_path_provider() -> str | None:
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Експортувати події",
                    "bdo-sea-assistant-events.txt",
                    "Text files (*.txt)",
                )
                return path or None
        if not callable(event_export_path_provider):
            raise ValueError("event_export_path_provider must be callable")
        self._event_export_path_provider = event_export_path_provider
        if not callable(calibration_inspector):
            raise ValueError("calibration_inspector must be callable")
        self._calibration_inspector = calibration_inspector
        if runtime_session is not None and (
            not callable(getattr(runtime_session, "start", None))
            or not callable(getattr(runtime_session, "stop", None))
        ):
            raise ValueError("runtime_session must provide start and stop")
        if type(route_evidence_available) is not bool:
            raise ValueError("route_evidence_available must be boolean")
        self._runtime_session = runtime_session
        self._session_running = False
        self._runtime_signals = _RuntimeSessionSignals(self)
        self._runtime_signals.snapshot.connect(self._on_runtime_session_snapshot)
        self._runtime_signals.runtime_status.connect(self._on_runtime_status)
        set_state_callback = getattr(runtime_session, "set_state_callback", None)
        if callable(set_state_callback):
            set_state_callback(self._runtime_signals.snapshot.emit)
        self.calibration_readiness = CalibrationReadiness(
            CalibrationStatus.MISSING,
            "calibration not inspected",
        )
        self.selected_window: WindowCandidate | None = None
        try:
            self._runtime_anchor = load_verified_minimap_anchor(
                asset_path("minimap", "manifest.json").parent
            )
        except ValueError:
            self._runtime_anchor = None
        self._route_evidence_available = route_evidence_available
        self._runtime_factory_available = runtime_session is not None
        self._calibration_capture: OneShotCalibrationCapture | None = None
        if calibration_flow is None and calibration_frame_provider is None:
            self._calibration_capture = OneShotCalibrationCapture()
            calibration_flow = build_default_calibration_flow()
            calibration_frame_provider = self._capture_selected_calibration_frame
            manual_region_provider = self._show_manual_region_dialog
            calibration_requires_selected_window = True
        self._calibration_flow = calibration_flow
        self._calibration_frame_provider = calibration_frame_provider
        self._calibration_destination = Path(
            calibration_destination or default_calibration_path()
        )
        self._manual_region_provider = manual_region_provider
        if type(calibration_requires_selected_window) is not bool:
            raise ValueError("calibration_requires_selected_window must be boolean")
        self._calibration_requires_selected_window = calibration_requires_selected_window
        if calibration_flow is not None and any(
            not callable(getattr(calibration_flow, method, None))
            for method in ("capture", "detect", "confirm_and_save", "retry")
        ):
            raise ValueError("calibration_flow has invalid interface")
        if calibration_frame_provider is not None and not callable(calibration_frame_provider):
            raise ValueError("calibration_frame_provider must be callable")
        if manual_region_provider is not None and not callable(manual_region_provider):
            raise ValueError("manual_region_provider must be callable")
        self.model = ModernUiModel(locale=controller.settings.locale, developer_recorder=developer_recorder)
        self.navigation_buttons: list[QPushButton] = []
        self._page_titles: list[QLabel] = []
        self.setObjectName("modernMainWindow")
        self.setWindowTitle(self._build_identity.window_title)
        self.setWindowIcon(application_icon())
        self.setMinimumSize(1120, 720)
        self.resize(1380, 860)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self._register_static_texts()
        self._set_locale(self.model.locale)
        self.refresh_calibration_readiness()
        self._render_dashboard()

    def _build_ui(self) -> None:
        surface = QWidget(objectName="appSurface")
        root = QHBoxLayout(surface)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._build_topbar())
        self.page_stack = QStackedWidget(objectName="pageStack")
        for page in self.model.pages:
            self.page_stack.addWidget(self._build_page(page.page_id))
        content_layout.addWidget(self.page_stack, 1)
        root.addWidget(content, 1)
        self.setCentralWidget(surface)
        self._activate_page(0)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(252)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(7)
        brand = QLabel("BDO Sea Assistant", objectName="brandTitle")
        meta = QLabel("PASSIVE · WGC · 2D", objectName="brandMeta")
        layout.addWidget(brand)
        layout.addWidget(meta)
        layout.addSpacing(22)
        for page_index, item in enumerate(self.model.navigation_items()):
            button = QPushButton(item.title)
            button.setIcon(_navigation_icon(item.page_id, "#8da7bc"))
            button.setIconSize(QSize(21, 21))
            button.setMinimumHeight(46)
            button.setAccessibleName(item.title)
            button.setAccessibleDescription(f"Перейти до сторінки: {item.title}")
            button.setCheckable(True)
            button.setProperty("nav", True)
            button.setProperty("pageId", item.page_id.value)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, i=page_index: self._activate_page(i))
            self.navigation_buttons.append(button)
            layout.addWidget(button)
        layout.addStretch(1)
        boundary = self._card()
        boundary_layout = QVBoxLayout(boundary)
        boundary_layout.addWidget(self._small_label("READ-ONLY BOUNDARY", "#42cee4"))
        boundary_layout.addWidget(self._muted("Pixels only\nNo input · No memory access"))
        layout.addWidget(boundary)
        return sidebar

    def _build_topbar(self) -> QFrame:
        topbar = QFrame(objectName="topbar")
        topbar.setFixedHeight(76)
        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(28, 0, 28, 0)
        title_block = QVBoxLayout()
        self.top_title = QLabel("Огляд", objectName="pageTitle")
        self.top_subtitle = QLabel("Стан системи та останні сигнали", objectName="pageSubtitle")
        title_block.addWidget(self.top_title)
        title_block.addWidget(self.top_subtitle)
        layout.addLayout(title_block)
        layout.addStretch(1)
        self.connection_badge = QLabel("●  OFFLINE / НЕ ЗАПУЩЕНО")
        self.connection_badge.setProperty("status", "idle")
        layout.addWidget(self.connection_badge)
        return topbar

    def _build_page(self, page_id: PageId) -> QWidget:
        builders: dict[PageId, Callable[[], QWidget]] = {
            PageId.DASHBOARD: self._dashboard_page,
            PageId.CAPTURE: self._capture_page,
            PageId.CALIBRATION: self._calibration_page,
            PageId.ALERTS: self._alerts_page,
            PageId.SETTINGS: self._settings_page,
            PageId.RECORDER: self._recorder_page,
        }
        return self._scroll_page(builders[page_id](), page_id)

    def _scroll_page(self, body: QWidget, page_id: PageId) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName(f"{page_id.value}Scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        return scroll

    def _page_body(self, eyebrow: str, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        body = QWidget(objectName="pageBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(30, 28, 30, 30)
        layout.setSpacing(16)
        layout.addWidget(self._small_label(eyebrow, "#42cee4"))
        heading = QLabel(title)
        heading.setProperty("metric", True)
        layout.addWidget(heading)
        if subtitle:
            layout.addWidget(self._muted(subtitle))
        return body, layout

    def _dashboard_page(self) -> QWidget:
        body, layout = self._page_body(
            "LIVE OVERVIEW",
            "Морський моніторинг",
            "",
        )
        layout.setContentsMargins(30, 18, 30, 18)
        layout.setSpacing(10)
        actions = QHBoxLayout()
        self.start_button = QPushButton("▶  Почати моніторинг", objectName="startMonitoringButton")
        self.start_button.setObjectName("startMonitoringButton")
        self.start_button.setEnabled(False)
        self.start_button.setToolTip("Runtime factory ще не підключений до modern UI")
        self.start_button.clicked.connect(self._start_runtime_session)
        self.stop_button = QPushButton("■  Зупинити", objectName="stopMonitoringButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_runtime_session)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.runtime_blockers = self._muted("")
        self.runtime_blockers.setVisible(False)
        layout.addWidget(self.runtime_blockers)
        grid = QGridLayout()
        grid.setSpacing(14)
        self.metric_labels = {}
        metrics = (
            ("capture", "ЗАХОПЛЕННЯ ВІКНА · WGC", "UNKNOWN"),
            ("calibration", "CALIBRATION", "UNKNOWN"),
            ("minimap", "2D MINIMAP", "UNKNOWN"),
            ("navigation", "NAVIGATION", "UNKNOWN"),
        )
        for index, (key, title, value) in enumerate(metrics):
            card = self._metric_card(title, value)
            self.metric_labels[key] = card.findChild(QLabel, "metricValue")
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)
        resources = QHBoxLayout()
        self.hp_gauge = ResourceGauge("SHIP HP", "#ff5d6c")
        self.rations_gauge = ResourceGauge("RATIONS", "#e5b84b")
        resources.addWidget(self.hp_gauge)
        resources.addWidget(self.rations_gauge)
        layout.addLayout(resources)
        event_card = self._card()
        event_card.setMinimumHeight(250)
        event_layout = QVBoxLayout(event_card)
        event_header = QHBoxLayout()
        event_header.addWidget(self._small_label("RECENT EVENTS", "#86a2b7"))
        event_header.addStretch(1)
        export_events = QPushButton("Експортувати .txt", objectName="exportEventsButton")
        export_events.clicked.connect(self._export_recent_events)
        event_header.addWidget(export_events)
        event_layout.addLayout(event_header)
        self.recent_events = QListWidget()
        self.recent_events.setFixedHeight(205)
        self.recent_events.addItem("Сесія ще не запускалася — подій немає")
        event_layout.addWidget(self.recent_events)
        layout.addWidget(event_card)
        layout.addStretch(1)
        return body

    def _recorder_page(self) -> QWidget:
        body, layout = self._page_body(
            "NATIVE MINIMAP RECORDER",
            "Запис маршруту",
            "Локальний запис minimap ROI з живими мітками сценарію.",
        )
        controls = self._card()
        grid = QGridLayout(controls)
        self.recorder_start_button = QPushButton("●  Старт", objectName="recorderStartButton")
        self.recorder_stop_button = QPushButton("■  Стоп", objectName="recorderStopButton")
        self.recorder_stop_button.setEnabled(False)
        self.recorder_arrived_button = QPushButton("Прибув", objectName="recorderArrivedButton")
        self.recorder_stuck_button = QPushButton("Застряг", objectName="recorderStuckButton")
        self.recorder_moving_button = QPushButton("Рух", objectName="recorderMovingButton")
        for button in (self.recorder_arrived_button, self.recorder_stuck_button, self.recorder_moving_button):
            button.setCheckable(True)
            button.setProperty("recorderMarker", True)
            button.setProperty("active", False)
            button.setEnabled(False)
        self.recorder_start_button.clicked.connect(self._start_recorder)
        self.recorder_stop_button.clicked.connect(self._stop_recorder)
        self.recorder_arrived_button.clicked.connect(lambda: self._label_recorder("ARRIVED"))
        self.recorder_stuck_button.clicked.connect(lambda: self._label_recorder("STUCK"))
        self.recorder_moving_button.clicked.connect(lambda: self._label_recorder("MOVING"))
        grid.addWidget(self.recorder_start_button, 0, 0)
        grid.addWidget(self.recorder_stop_button, 0, 1)
        grid.addWidget(self.recorder_arrived_button, 1, 0)
        grid.addWidget(self.recorder_stuck_button, 1, 1)
        grid.addWidget(self.recorder_moving_button, 2, 0, 1, 2)
        layout.addWidget(controls)
        preview = self._card()
        preview_layout = QVBoxLayout(preview)
        preview_layout.addWidget(self._small_label("ПРЕВ’Ю ЗАПИСУ", "#86a2b7"))
        self.recorder_preview = AspectFitPreviewLabel("Запис ще не розпочато")
        self.recorder_preview.setProperty("muted", True)
        self.recorder_preview.setMinimumHeight(180)
        self.recorder_preview.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.recorder_preview)
        layout.addWidget(preview)
        self.recorder_open_folder_button = QPushButton("Перегляд папки записів", objectName="recorderOpenFolderButton")
        self.recorder_open_folder_button.clicked.connect(self._open_recordings_folder)
        layout.addWidget(self.recorder_open_folder_button)
        layout.addStretch(1)
        return body

    def _start_recorder(self) -> None:
        if self._recorder_session is not None or self.selected_window is None:
            return
        readiness = self.calibration_readiness
        try:
            if not readiness.ready or readiness.layout is None:
                raise ValueError("confirmed calibration is required")
            absolute = readiness.layout["regions"]["minimap"]["absolute"]
            capture_size = readiness.layout["capture_size"]
            rect = Rect(absolute["x"], absolute["y"], absolute["width"], absolute["height"])
            expected_size = (capture_size["width"], capture_size["height"])
            if (
                any(type(value) is not int or value <= 0 for value in (*expected_size, rect.width, rect.height))
                or type(rect.x) is not int or type(rect.y) is not int or rect.x < 0 or rect.y < 0
            ):
                raise ValueError("calibration geometry is invalid")
            self._recordings_directory.mkdir(parents=True, exist_ok=True)
            destination = self._recordings_directory / datetime.now().strftime("recording-%Y%m%d-%H%M%S-%f")
            source = self._recorder_source_factory(self.selected_window.hwnd)
            recorder = NativeMinimapReplayRecorder(
                destination,
                minimap=rect,
                expected_capture_size=expected_size,
                max_frames=None,
            )
            session = NativeMinimapReplaySession(source, recorder, on_frame_recorded=self._recorder_signals.preview.emit)
        except Exception as exc:
            self.recorder_preview.setText(f"Запис недоступний: {exc}")
            return
        self._recorder_rect = rect
        self.recorder_start_button.setEnabled(False)
        self.recorder_preview.setText("Запуск WGC запису…")

        def start_in_background() -> None:
            try:
                session.start()
            except Exception as exc:
                self._recorder_signals.terminal.emit(None, exc)
            else:
                self._recorder_signals.started.emit(session)

        threading.Thread(target=start_in_background, name="minimap-recorder-start", daemon=True).start()

    def _on_recorder_started(self, session: object) -> None:
        if type(session) is not NativeMinimapReplaySession:
            return
        self._recorder_session = session
        self.recorder_stop_button.setEnabled(True)
        for button in (self.recorder_arrived_button, self.recorder_stuck_button, self.recorder_moving_button):
            button.setEnabled(True)
        self.recorder_preview.setText("Очікування minimap кадру…")

    def _on_recorder_preview(self, frame: object) -> None:
        rect = self._recorder_rect
        if not isinstance(frame, np.ndarray) or rect is None:
            return
        crop = frame[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]
        if crop.size == 0:
            return
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        if not image.isNull():
            self.recorder_preview.setPixmap(QPixmap.fromImage(image))

    def _label_recorder(self, label: str) -> None:
        session = self._recorder_session
        if session is None:
            return
        try:
            session.label(label)
        except Exception as exc:
            self.recorder_preview.setText(f"Мітка не збережена: {exc}")
            return
        buttons = {
            "ARRIVED": self.recorder_arrived_button,
            "STUCK": self.recorder_stuck_button,
            "MOVING": self.recorder_moving_button,
        }
        for name, button in buttons.items():
            active = name == label
            button.setChecked(active)
            button.setProperty("active", active)
            button.style().unpolish(button)
            button.style().polish(button)

    def _stop_recorder(self) -> None:
        session = self._recorder_session
        if session is None:
            return
        self.recorder_stop_button.setEnabled(False)
        for button in (self.recorder_arrived_button, self.recorder_stuck_button, self.recorder_moving_button):
            button.setEnabled(False)
        self.recorder_preview.setText("Фіналізація manifest…")

        def stop_in_background() -> None:
            try:
                manifest = session.stop()
            except Exception as exc:
                self._recorder_signals.terminal.emit(None, exc)
            else:
                self._recorder_signals.terminal.emit(manifest, None)

        threading.Thread(target=stop_in_background, name="minimap-recorder-stop", daemon=True).start()

    def _on_recorder_terminal(self, manifest: object, error: object) -> None:
        self._recorder_session = None
        self.recorder_start_button.setEnabled(True)
        if isinstance(error, Exception):
            self.recorder_preview.setText(f"Запис завершився з помилкою: {error}")
        elif isinstance(manifest, Path):
            self.recorder_preview.setText(f"Manifest: {manifest}")
        for button in (self.recorder_arrived_button, self.recorder_stuck_button, self.recorder_moving_button):
            button.setEnabled(False)

    def _open_recordings_folder(self) -> None:
        try:
            self._recordings_directory.mkdir(parents=True, exist_ok=True)
            if not self._recordings_folder_opener(self._recordings_directory):
                raise RuntimeError("desktop service refused recordings folder")
        except Exception as exc:
            self.recorder_preview.setText(f"Папка записів недоступна: {exc}")

    def _capture_page(self) -> QWidget:
        body, layout = self._page_body(
            "WINDOW CAPTURE",
            "Захоплення вікна",
            "Прив’язка єдиного вікна BDO через Windows Graphics Capture. Гра ніколи не запускається автоматично.",
        )
        card = self._card()
        form = QVBoxLayout(card)
        form.addWidget(self._small_label("TARGET WINDOW", "#86a2b7"))
        self.window_combo = QComboBox(objectName="windowCandidateCombo")
        self.window_combo.setObjectName("windowCandidateCombo")
        self.window_combo.setPlaceholderText("Вікно BDO не вибрано")
        self.window_combo.currentIndexChanged.connect(self._window_selection_changed)
        form.addWidget(self.window_combo)
        self.window_detail = self._muted("Натисни «Захопити вікно» для пасивної прив’язки HWND")
        form.addWidget(self.window_detail)
        form.addWidget(self._muted("Пошук за executable identity + title · border/cursor capture disabled"))
        row = QHBoxLayout()
        find_button = QPushButton("Захопити вікно", objectName="refreshWindowsButton")
        find_button.setObjectName("refreshWindowsButton")
        find_button.clicked.connect(self.refresh_window_candidates)
        row.addWidget(find_button)
        row.addStretch(1)
        form.addLayout(row)
        layout.addWidget(card)
        layout.addStretch(1)
        return body

    def refresh_window_candidates(self) -> None:
        try:
            candidates = self._window_discovery()
        except Exception as exc:
            self.window_combo.clear()
            self.selected_window = None
            self.window_detail.setText(f"Помилка пошуку вікон: {exc}")
            self.start_button.setEnabled(False)
            self._refresh_runtime_readiness()
            return
        if not isinstance(candidates, tuple) or any(
            type(candidate) is not WindowCandidate for candidate in candidates
        ):
            self.window_combo.clear()
            self.selected_window = None
            self.window_detail.setText("Window discovery повернув некоректні дані")
            self.start_button.setEnabled(False)
            self._refresh_runtime_readiness()
            return
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for candidate in candidates:
            suffix = " · minimized" if candidate.minimized else ""
            self.window_combo.addItem(
                f"{candidate.title} · {candidate.width}×{candidate.height}{suffix}",
                candidate,
            )
        self.window_combo.blockSignals(False)
        if candidates:
            self.window_combo.setCurrentIndex(0)
            self._window_selection_changed(0)
        else:
            self.selected_window = None
            self.window_detail.setText("Відкрите вікно BlackDesert64.exe не знайдено")
            self.start_button.setEnabled(False)
            self._refresh_runtime_readiness()

    def _window_selection_changed(self, index: int) -> None:
        candidate = self.window_combo.itemData(index) if index >= 0 else None
        if type(candidate) is not WindowCandidate:
            self.selected_window = None
            self.start_button.setEnabled(False)
            self._refresh_runtime_readiness()
            return
        self.selected_window = candidate
        state = "minimized" if candidate.minimized else "visible"
        self.window_detail.setText(
            f"HWND {candidate.hwnd} · PID {candidate.pid} · "
            f"{candidate.width}×{candidate.height} · {state}"
        )
        self.start_button.setEnabled(False)
        self.start_button.setToolTip(
            "Runtime factory та підтверджене калібрування ще не підключені"
        )
        self._refresh_runtime_readiness()

    def _calibration_page(self) -> QWidget:
        body, layout = self._page_body(
            "UI GEOMETRY",
            "Калібрування HUD",
            "Capture → Detect → Preview → Confirm або Manual correction → Save.",
        )
        steps = QHBoxLayout()
        for number, title in enumerate(("Capture", "Detect", "Preview", "Confirm"), 1):
            card = self._card()
            card_layout = QVBoxLayout(card)
            card_layout.addWidget(self._small_label(f"0{number}", "#42cee4"))
            card_layout.addWidget(QLabel(title))
            steps.addWidget(card)
        layout.addLayout(steps)
        readiness = self._card()
        readiness_layout = QHBoxLayout(readiness)
        readiness_text = QVBoxLayout()
        readiness_text.addWidget(self._small_label("CONFIRMED LAYOUT", "#86a2b7"))
        self.calibration_status = QLabel("NOT INSPECTED")
        self.calibration_status.setProperty("metric", True)
        readiness_text.addWidget(self.calibration_status)
        self.calibration_diagnostic = self._muted("Calibration has not been inspected")
        readiness_text.addWidget(self.calibration_diagnostic)
        readiness_layout.addLayout(readiness_text)
        readiness_layout.addStretch(1)
        refresh = QPushButton("Оновити стан")
        refresh.clicked.connect(self.refresh_calibration_readiness)
        readiness_layout.addWidget(refresh)
        layout.addWidget(readiness)
        preview = self._card()
        preview.setMinimumHeight(260)
        preview_layout = QVBoxLayout(preview)
        preview_layout.addStretch(1)
        self.calibration_preview = AspectFitPreviewLabel("PREVIEW AREA\nLive frame required")
        self.calibration_preview.setAlignment(Qt.AlignCenter)
        self.calibration_preview.setProperty("muted", True)
        preview_layout.addWidget(self.calibration_preview)
        preview_layout.addStretch(1)
        layout.addWidget(preview)
        calibration_actions = QHBoxLayout()
        self.begin_calibration_button = QPushButton(
            "Почати калібрування", objectName="beginCalibrationButton"
        )
        self.begin_calibration_button.setEnabled(
            self._calibration_flow is not None
            and self._calibration_frame_provider is not None
            and (
                not self._calibration_requires_selected_window
                or self.selected_window is not None
            )
        )
        self.begin_calibration_button.setToolTip("Live WGC frame required for calibration")
        self.begin_calibration_button.clicked.connect(self._begin_calibration)
        self.confirm_calibration_button = QPushButton("Підтвердити й зберегти")
        self.confirm_calibration_button.setEnabled(False)
        self.confirm_calibration_button.clicked.connect(self._confirm_calibration)
        self.manual_calibration_button = QPushButton("Ручна корекція")
        self.manual_calibration_button.setEnabled(False)
        self.manual_calibration_button.clicked.connect(self._apply_manual_calibration)
        self.retry_calibration_button = QPushButton("Повторити")
        self.retry_calibration_button.setEnabled(False)
        self.retry_calibration_button.clicked.connect(self._retry_calibration)
        calibration_actions.addWidget(self.begin_calibration_button)
        calibration_actions.addWidget(self.manual_calibration_button)
        calibration_actions.addWidget(self.confirm_calibration_button)
        calibration_actions.addWidget(self.retry_calibration_button)
        calibration_actions.addStretch(1)
        layout.addLayout(calibration_actions)
        return body

    def _begin_calibration(self) -> None:
        if self._calibration_flow is None or self._calibration_frame_provider is None:
            return
        self._clear_calibration_review()
        try:
            frame = self._calibration_frame_provider()
            self._calibration_flow.capture(frame)
            self._calibration_flow.detect(self._runtime_anchor)
            preview = self._calibration_flow.preview
            if (
                not isinstance(preview, np.ndarray)
                or preview.ndim != 3
                or preview.shape[2] != 3
                or preview.size == 0
                or preview.dtype != np.uint8
            ):
                raise ValueError("calibration preview is invalid")
            rgb = np.ascontiguousarray(preview[:, :, ::-1])
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()
            if image.isNull():
                raise ValueError("calibration preview image is null")
            pixmap = QPixmap.fromImage(image)
            if pixmap.isNull():
                raise ValueError("calibration preview pixmap is null")
            self.calibration_preview.setPixmap(pixmap)
        except Exception as exc:
            self.calibration_diagnostic.setText(f"Calibration failed: {exc}")
            self.confirm_calibration_button.setEnabled(False)
            return
        self.calibration_diagnostic.setText("Preview ready — review before confirmation")
        self.confirm_calibration_button.setEnabled(True)
        self.manual_calibration_button.setEnabled(self._manual_region_provider is not None)
        self.retry_calibration_button.setEnabled(True)

    def _capture_selected_calibration_frame(self) -> np.ndarray:
        if self.selected_window is None or self._calibration_capture is None:
            raise RuntimeError("select a BDO window before calibration capture")
        return self._calibration_capture.capture(self.selected_window)

    def _show_manual_region_dialog(self, flow: object) -> Mapping[str, Rect]:
        frame = getattr(flow, "frame", None)
        layout = getattr(flow, "layout", None)
        if not isinstance(frame, np.ndarray) or not isinstance(layout, Mapping):
            raise RuntimeError("reviewed calibration preview is required")
        height, width = frame.shape[:2]
        dialog = QDialog(self)
        dialog.setWindowTitle("Ручна корекція HUD")
        form = QGridLayout(dialog)
        fields: dict[str, tuple[QSpinBox, QSpinBox, QSpinBox, QSpinBox]] = {}
        names = ("minimap", "ship_hp_bar", "ship_rations_bar")
        for row, name in enumerate(names):
            absolute = layout["regions"][name]["absolute"]
            form.addWidget(QLabel(name), row, 0)
            spins = tuple(QSpinBox() for _ in range(4))
            maxima = (width - 1, height - 1, width, height)
            values = (
                absolute["x"],
                absolute["y"],
                absolute["width"],
                absolute["height"],
            )
            for column, (spin, maximum, value) in enumerate(
                zip(spins, maxima, values), 1
            ):
                spin.setRange(0 if column < 3 else 1, maximum)
                spin.setValue(value)
                form.addWidget(spin, row, column)
            fields[name] = spins
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addWidget(buttons, len(names), 0, 1, 5)
        if dialog.exec() != QDialog.Accepted:
            raise RuntimeError("manual correction cancelled")
        return {
            name: Rect(
                spins[0].value(),
                spins[1].value(),
                spins[2].value(),
                spins[3].value(),
            )
            for name, spins in fields.items()
        }

    def _apply_manual_calibration(self) -> None:
        if (
            self._calibration_flow is None
            or self._manual_region_provider is None
            or not self.manual_calibration_button.isEnabled()
        ):
            return
        self._clear_calibration_review()
        try:
            regions = self._manual_region_provider(self._calibration_flow)
            self._calibration_flow.apply_manual_regions(regions)
            preview = self._calibration_flow.preview
            if (
                not isinstance(preview, np.ndarray)
                or preview.ndim != 3
                or preview.shape[2] != 3
                or preview.size == 0
                or preview.dtype != np.uint8
            ):
                raise ValueError("manual calibration preview is invalid")
            rgb = np.ascontiguousarray(preview[:, :, ::-1])
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()
            if image.isNull():
                raise ValueError("calibration preview image is null")
            pixmap = QPixmap.fromImage(image)
            if pixmap.isNull():
                raise ValueError("calibration preview pixmap is null")
            self.calibration_preview.setPixmap(pixmap)
        except Exception as exc:
            self.calibration_diagnostic.setText(f"Manual correction failed: {exc}")
            self.confirm_calibration_button.setEnabled(False)
            return
        self.calibration_diagnostic.setText(
            "Manual preview ready — review before confirmation"
        )
        self.confirm_calibration_button.setEnabled(True)

    def _confirm_calibration(self) -> None:
        if self._calibration_flow is None or not self.confirm_calibration_button.isEnabled():
            return
        try:
            self._calibration_flow.confirm_and_save(self._calibration_destination)
        except Exception as exc:
            self.calibration_diagnostic.setText(f"Save failed: {exc}")
            return
        self.confirm_calibration_button.setEnabled(False)
        self.calibration_diagnostic.setText("Calibration saved")
        self.refresh_calibration_readiness()

    def _retry_calibration(self) -> None:
        if self._calibration_flow is not None:
            self._calibration_flow.retry()
        self._clear_calibration_review()
        self.calibration_diagnostic.setText("Calibration reset")

    def _clear_calibration_review(self) -> None:
        self.calibration_preview.clear()
        self.calibration_preview.setText("PREVIEW AREA\nNo reviewed preview")
        self.confirm_calibration_button.setEnabled(False)
        self.manual_calibration_button.setEnabled(False)
        self.retry_calibration_button.setEnabled(False)

    def refresh_calibration_readiness(self) -> None:
        try:
            readiness = self._calibration_inspector()
        except Exception as exc:
            readiness = CalibrationReadiness(
                CalibrationStatus.INVALID,
                f"calibration inspection failed: {exc}",
            )
        if type(readiness) is not CalibrationReadiness:
            readiness = CalibrationReadiness(
                CalibrationStatus.INVALID,
                "calibration inspector returned invalid data",
            )
        self.calibration_readiness = readiness
        self.calibration_status.setText(readiness.status.value.upper())
        self.calibration_diagnostic.setText(readiness.diagnostic)
        status_color = {
            CalibrationStatus.VALID: "#4ee0a0",
            CalibrationStatus.MISSING: "#ffc857",
            CalibrationStatus.INVALID: "#ff6b78",
        }[readiness.status]
        self.calibration_status.setStyleSheet(f"color: {status_color};")
        self._refresh_runtime_readiness()

    def _refresh_runtime_readiness(self) -> None:
        calibration_button = getattr(self, "begin_calibration_button", None)
        if calibration_button is not None:
            calibration_button.setEnabled(
                self._calibration_flow is not None
                and self._calibration_frame_provider is not None
                and (
                    not self._calibration_requires_selected_window
                    or self.selected_window is not None
                )
            )
        status = RuntimeDependencyStatus.assess(
            candidate=self.selected_window,
            calibration=self.calibration_readiness,
            anchor=self._runtime_anchor,
            route_evidence_available=self._route_evidence_available,
            runtime_factory_available=self._runtime_factory_available,
        )
        self.start_button.setEnabled(status.ready and not self._session_running)
        self.stop_button.setEnabled(self._session_running)
        if status.blockers:
            self.runtime_blockers.setVisible(True)
            self.runtime_blockers.setText(
                self._translate("Start заблоковано: {details}").format(
                    details=" · ".join(status.blockers)
                )
            )
            self.start_button.setToolTip("; ".join(status.blockers))
        elif status.limitations:
            self.runtime_blockers.setVisible(True)
            self.runtime_blockers.setText(
                self._translate("Monitoring готовий з обмеженням: {details}").format(
                    details=" · ".join(status.limitations)
                )
            )
            self.start_button.setToolTip("; ".join(status.limitations))
        else:
            self.runtime_blockers.setVisible(False)
            self.runtime_blockers.clear()
            self.start_button.setToolTip("Почати passive WGC monitoring")

    def _start_runtime_session(self) -> None:
        if self._runtime_session is None or not self.start_button.isEnabled():
            return
        try:
            self._runtime_session.start()
        except Exception as exc:
            self.connection_badge.setText(f"●  ERROR · {exc}")
            self.connection_badge.setProperty("status", "warn")
            return
        self._session_running = True
        self.connection_badge.setText("●  RUNNING")
        self.connection_badge.setProperty("status", "ok")
        self._refresh_runtime_readiness()

    def _stop_runtime_session(self) -> None:
        if self._runtime_session is None or not self._session_running:
            return
        session = self._runtime_session
        self.stop_button.setEnabled(False)
        self.connection_badge.setText("●  STOPPING")
        self.connection_badge.setProperty("status", "idle")

        def stop_in_background() -> None:
            try:
                session.stop()
            except Exception as exc:
                self._runtime_signals.snapshot.emit(
                    RuntimeSessionSnapshot(RuntimeSessionState.ERROR, str(exc), bridge_active=True)
                )
            else:
                self._runtime_signals.snapshot.emit(
                    RuntimeSessionSnapshot(RuntimeSessionState.STOPPED, bridge_active=False)
                )

        threading.Thread(target=stop_in_background, name="runtime-stop", daemon=True).start()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._runtime_session is not None and self._session_running:
            try:
                self._runtime_session.stop()
            except Exception as exc:
                diagnostic = f"runtime shutdown failed: {exc}"
                self.runtime_diagnostics.appendleft(diagnostic)
                self.connection_badge.setText(f"●  ERROR · {diagnostic}")
                self.connection_badge.setProperty("status", "warn")
                event.ignore()
                return
            self._session_running = False
        close_controller = getattr(self.controller, "close", None)
        if callable(close_controller):
            try:
                close_controller()
            except Exception as exc:
                diagnostic = f"notification shutdown failed: {exc}"
                self.runtime_diagnostics.appendleft(diagnostic)
                self.connection_badge.setText(f"●  ERROR · {diagnostic}")
                self.connection_badge.setProperty("status", "warn")
                event.ignore()
                return
        super().closeEvent(event)

    def _on_runtime_session_snapshot(self, snapshot: RuntimeSessionSnapshot) -> None:
        if type(snapshot) is not RuntimeSessionSnapshot:
            return
        self._session_running = snapshot.bridge_active
        if snapshot.state is RuntimeSessionState.RUNNING:
            self.connection_badge.setText("●  RUNNING")
            self.connection_badge.setProperty("status", "ok")
        elif snapshot.state is RuntimeSessionState.ERROR:
            detail = f" · {snapshot.error}" if snapshot.error else ""
            self.connection_badge.setText(f"●  ERROR{detail}")
            self.connection_badge.setProperty("status", "warn")
        elif snapshot.state is RuntimeSessionState.STARTING:
            self.connection_badge.setText("●  STARTING")
            self.connection_badge.setProperty("status", "idle")
        elif snapshot.state is RuntimeSessionState.STOPPING:
            self.connection_badge.setText("●  STOPPING")
            self.connection_badge.setProperty("status", "idle")
        else:
            self.connection_badge.setText("●  STOPPED")
            self.connection_badge.setProperty("status", "idle")
        self._refresh_runtime_readiness()

    def attach_runtime_session(
        self,
        session: RuntimeSessionController,
        *,
        route_evidence_available: bool = False,
    ) -> None:
        if type(session) is not RuntimeSessionController:
            raise ValueError("session must be RuntimeSessionController")
        if type(route_evidence_available) is not bool:
            raise ValueError("route_evidence_available must be boolean")
        self._runtime_session = session
        self._runtime_factory_available = True
        self._route_evidence_available = route_evidence_available
        session.set_state_callback(self._runtime_signals.snapshot.emit)
        self._refresh_runtime_readiness()

    def _on_runtime_status(self, status: object) -> None:
        if type(self._runtime_session) is RuntimeSessionController:
            try:
                self._runtime_session.handle_runtime_status(status)
            except ValueError:
                return
        if (
            getattr(status, "diagnostic", "") == "shadow metrics"
            and getattr(status, "result", None) is None
            and getattr(status, "sample", None) is None
        ):
            # Standalone shadow callbacks must never overwrite the live dashboard.
            return
        try:
            snapshot = project_runtime_status(
                status,
                window_title=(
                    self.selected_window.title if self.selected_window is not None else None
                ),
            )
        except ValueError:
            return
        self.update_dashboard(snapshot)
        diagnostic = getattr(status, "diagnostic", "")
        if diagnostic:
            self.runtime_diagnostics.appendleft(diagnostic)
        for delivery in getattr(status, "deliveries", ()):
            if not delivery.success:
                self.runtime_diagnostics.appendleft(f"{delivery.adapter}: {delivery.error}")
        result = getattr(status, "result", None)
        for diagnostic in getattr(result, "diagnostics", ()):
            self.runtime_diagnostics.appendleft(diagnostic)
        events = getattr(result, "events", ()) if result is not None else ()
        if events:
            if self.recent_events.count() == 1 and "подій немає" in self.recent_events.item(0).text():
                self.recent_events.clear()
            for event in events:
                timestamp = self._event_time_provider()
                self.recent_events.insertItem(
                    0,
                    f"[{timestamp}] {event.event_type.value.upper()}",
                )

    def _alerts_page(self) -> QWidget:
        body, layout = self._page_body(
            "LOCAL CHANNELS",
            "Сповіщення",
            "",
        )
        settings = self.controller.settings
        channels = QHBoxLayout()
        for title, detail, enabled in (
            ("OFFLINE VOICE", "Vika UK / Vika EN\nPCM16 · 48 kHz · local", settings.voice_enabled),
            ("NON-FOCUS POPUP", "WS_EX_NOACTIVATE\nserial queue · local", settings.popup_enabled),
        ):
            card = self._card()
            box = QVBoxLayout(card)
            box.addWidget(self._small_label(title, "#42cee4"))
            status = QLabel("ENABLED" if enabled else "DISABLED")
            status.setProperty("status", "ok" if enabled else "idle")
            box.addWidget(status)
            box.addWidget(self._muted(detail))
            channels.addWidget(card)
        layout.addLayout(channels)
        actions = QHBoxLayout()
        test_voice = QPushButton("Test Voice", objectName="testVoiceButton")
        test_voice.setObjectName("testVoiceButton")
        test_voice.clicked.connect(lambda: self._run_action("Test Voice", self.controller.test_voice))
        test_popup = QPushButton("Test Popup", objectName="testPopupButton")
        test_popup.setObjectName("testPopupButton")
        test_popup.clicked.connect(lambda: self._run_action("Test Popup", self.controller.test_popup))
        actions.addWidget(test_voice)
        actions.addWidget(test_popup)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.action_status = self._muted("")
        self.action_status.setVisible(False)
        layout.addWidget(self.action_status)
        layout.addStretch(1)
        return body

    def _settings_page(self) -> QWidget:
        body, layout = self._page_body(
            "APPLICATION",
            "Налаштування",
            "",
        )
        card = self._card()
        form = QGridLayout(card)
        settings = self.controller.settings
        form.addWidget(QLabel("Мова інтерфейсу"), 0, 0)
        self.locale_combo = QComboBox()
        self.locale_combo.addItem("Українська", "uk")
        self.locale_combo.addItem("English", "en")
        self.locale_combo.setCurrentIndex(0 if settings.locale == "uk" else 1)
        form.addWidget(self.locale_combo, 0, 1)
        self.voice_check = QCheckBox("Offline Voice")
        self.voice_check.setChecked(settings.voice_enabled)
        form.addWidget(self.voice_check, 1, 0, 1, 2)
        self.popup_check = QCheckBox("Non-focus Popup")
        self.popup_check.setChecked(settings.popup_enabled)
        form.addWidget(self.popup_check, 2, 0, 1, 2)
        form.addWidget(QLabel("Voice volume"), 3, 0)
        volume_row = QWidget()
        volume_layout = QHBoxLayout(volume_row)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(1, 100)
        self.volume_slider.setValue(round(settings.voice_volume * 100))
        self.volume_value = QLabel(
            f"{self.volume_slider.value()}%",
            objectName="voiceVolumeValue",
        )
        self.volume_value.setMinimumWidth(44)
        self.volume_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.volume_slider.valueChanged.connect(
            lambda value: self.volume_value.setText(f"{value}%")
        )
        volume_layout.addWidget(self.volume_slider, 1)
        volume_layout.addWidget(self.volume_value)
        form.addWidget(volume_row, 3, 1)
        form.addWidget(QLabel("Затримка втрати мінікарти"), 4, 0)
        minimap_delay_row = QWidget()
        minimap_delay_layout = QHBoxLayout(minimap_delay_row)
        minimap_delay_layout.setContentsMargins(0, 0, 0, 0)
        self.minimap_loss_delay_slider = QSlider(Qt.Horizontal)
        self.minimap_loss_delay_slider.setRange(10, 300)
        self.minimap_loss_delay_slider.setValue(settings.minimap_loss_delay_seconds)
        self.minimap_loss_delay_value = QLabel(
            f"{self.minimap_loss_delay_slider.value()} с",
            objectName="minimapLossDelayValue",
        )
        self.minimap_loss_delay_value.setMinimumWidth(52)
        self.minimap_loss_delay_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.minimap_loss_delay_slider.valueChanged.connect(
            lambda value: self.minimap_loss_delay_value.setText(
                f"{value} {self._translate('с')}"
            )
        )
        minimap_delay_layout.addWidget(self.minimap_loss_delay_slider, 1)
        minimap_delay_layout.addWidget(self.minimap_loss_delay_value)
        form.addWidget(minimap_delay_row, 4, 1)
        form.addWidget(QLabel("Зона сповіщення HP корабля"), 5, 0)
        hp_warning_row = QWidget()
        hp_warning_layout = QHBoxLayout(hp_warning_row)
        hp_warning_layout.setContentsMargins(0, 0, 0, 0)
        self.hp_warning_slider = QSlider(Qt.Horizontal)
        self.hp_warning_slider.setRange(1, 99)
        self.hp_warning_slider.setValue(settings.ship_hp_warning_percent)
        self.hp_warning_value = QLabel(
            f"{self.hp_warning_slider.value()}%",
            objectName="shipHpWarningValue",
        )
        self.hp_warning_value.setMinimumWidth(44)
        self.hp_warning_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.hp_warning_slider.valueChanged.connect(
            lambda value: self.hp_warning_value.setText(f"{value}%")
        )
        hp_warning_layout.addWidget(self.hp_warning_slider, 1)
        hp_warning_layout.addWidget(self.hp_warning_value)
        form.addWidget(hp_warning_row, 5, 1)
        form.addWidget(QLabel("Зона сповіщення раціонів"), 6, 0)
        rations_warning_row = QWidget()
        rations_warning_layout = QHBoxLayout(rations_warning_row)
        rations_warning_layout.setContentsMargins(0, 0, 0, 0)
        self.rations_warning_slider = QSlider(Qt.Horizontal)
        self.rations_warning_slider.setRange(1, 99)
        self.rations_warning_slider.setValue(settings.ship_rations_warning_percent)
        self.rations_warning_value = QLabel(
            f"{self.rations_warning_slider.value()}%",
            objectName="shipRationsWarningValue",
        )
        self.rations_warning_value.setMinimumWidth(44)
        self.rations_warning_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rations_warning_slider.valueChanged.connect(
            lambda value: self.rations_warning_value.setText(f"{value}%")
        )
        rations_warning_layout.addWidget(self.rations_warning_slider, 1)
        rations_warning_layout.addWidget(self.rations_warning_value)
        form.addWidget(rations_warning_row, 6, 1)
        form.addWidget(QLabel("Popup duration, ms"), 7, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(500, 60_000)
        self.duration_spin.setSingleStep(500)
        self.duration_spin.setValue(settings.popup_duration_ms)
        form.addWidget(self.duration_spin, 7, 1)
        self.repeat_alerts_check = QCheckBox("Повторне сповіщення")
        self.repeat_alerts_check.setChecked(settings.repeat_alerts_enabled)
        form.addWidget(self.repeat_alerts_check, 8, 0, 1, 2)
        form.addWidget(QLabel("Інтервал повтору"), 9, 0)
        repeat_row = QWidget()
        repeat_layout = QHBoxLayout(repeat_row)
        repeat_layout.setContentsMargins(0, 0, 0, 0)
        self.repeat_alert_interval_slider = QSlider(Qt.Horizontal)
        self.repeat_alert_interval_slider.setRange(30, 300)
        self.repeat_alert_interval_slider.setValue(settings.repeat_alert_interval_seconds)
        self.repeat_alert_interval_value = QLabel(
            f"{self.repeat_alert_interval_slider.value()} с",
            objectName="repeatAlertIntervalValue",
        )
        self.repeat_alert_interval_value.setMinimumWidth(52)
        self.repeat_alert_interval_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.repeat_alert_interval_slider.valueChanged.connect(
            lambda value: self.repeat_alert_interval_value.setText(
                f"{value} {self._translate('с')}"
            )
        )
        repeat_layout.addWidget(self.repeat_alert_interval_slider, 1)
        repeat_layout.addWidget(self.repeat_alert_interval_value)
        form.addWidget(repeat_row, 9, 1)
        self.apply_button = QPushButton("Застосувати", objectName="primaryButton")
        self.apply_button.setProperty("settingsDirty", False)
        self.apply_button.clicked.connect(self._apply_settings)
        for control, signal in (
            (self.locale_combo, self.locale_combo.currentIndexChanged),
            (self.voice_check, self.voice_check.toggled),
            (self.popup_check, self.popup_check.toggled),
            (self.volume_slider, self.volume_slider.valueChanged),
            (self.minimap_loss_delay_slider, self.minimap_loss_delay_slider.valueChanged),
            (self.hp_warning_slider, self.hp_warning_slider.valueChanged),
            (self.rations_warning_slider, self.rations_warning_slider.valueChanged),
            (self.duration_spin, self.duration_spin.valueChanged),
            (self.repeat_alerts_check, self.repeat_alerts_check.toggled),
            (self.repeat_alert_interval_slider, self.repeat_alert_interval_slider.valueChanged),
        ):
            signal.connect(self._mark_settings_dirty)
        form.addWidget(self.apply_button, 10, 0, 1, 2, Qt.AlignLeft)
        layout.addWidget(card)
        update = self._card()
        update_layout = QHBoxLayout(update)
        update_text = "UPDATE CHANNEL\nNot configured · immutable releases only"
        if self._build_identity.credit_line:
            update_text += f"\n{self._build_identity.credit_line}"
        update_layout.addWidget(self._muted(update_text, object_name="updateChannelInfo"))
        update_layout.addStretch(1)
        check = QPushButton("Перевірити оновлення")
        check.setEnabled(False)
        check.setToolTip("Signed manifest/updater перенесено в distribution checkpoint")
        update_layout.addWidget(check)
        layout.addWidget(update)
        layout.addStretch(1)
        return body

    def _activate_page(self, index: int) -> None:
        self.page_stack.setCurrentIndex(index)
        page = self.model.pages[index]
        self.top_title.setText(page.title_uk if self.model.locale == "uk" else page.title_en)
        subtitles = {
            PageId.DASHBOARD: "Стан системи та останні сигнали",
            PageId.CAPTURE: "HWND-scoped Windows Graphics Capture",
            PageId.CALIBRATION: "Переносима геометрія HUD",
            PageId.ALERTS: "Локальні незалежні канали",
            PageId.SETTINGS: "Мова, канали й поведінка",
            PageId.RECORDER: "Локальний native minimap ROI запис",
        }
        self.top_subtitle.setText(self._translate(subtitles[page.page_id]))
        for button in self.navigation_buttons:
            active = button.property("pageId") == page.page_id.value
            button.setProperty("active", active)
            button.setChecked(active)
            button.setAccessibleDescription(
                self._translate("Активна сторінка") if active
                else self._translate("Перейти до сторінки: {title}").format(title=button.text())
            )
            button.setIcon(
                _navigation_icon(
                    PageId(button.property("pageId")),
                    "#63d8ee" if active else "#8da7bc",
                )
            )
            button.style().unpolish(button)
            button.style().polish(button)

    def _render_dashboard(self) -> None:
        snapshot = self.model.dashboard
        mapping = {
            "capture": snapshot.capture,
            "calibration": snapshot.calibration,
            "minimap": snapshot.minimap,
            "navigation": snapshot.navigation,
        }
        for key, value in mapping.items():
            label = self.metric_labels[key]
            label.setText(value.upper())
            if key == "navigation":
                navigation_colors = {
                    "moving": "#4ee0a0",
                    "arrived": "#32c8df",
                    "stuck": "#ff6b78",
                    "recovering": "#8aeaf7",
                    "unknown": "#ffc857",
                }
                label.setStyleSheet(
                    f"color: {navigation_colors.get(value.lower(), '#f3f8fb')};"
                )
        if snapshot.minimap == "visible" and snapshot.resources == "absent":
            return
        self.hp_gauge.set_percent(snapshot.hp_percent)
        self.rations_gauge.set_percent(snapshot.rations_percent)

    def update_dashboard(self, snapshot: DashboardSnapshot) -> None:
        self.model.update_dashboard(snapshot)
        self._render_dashboard()

    def _export_recent_events(self) -> None:
        destination = self._event_export_path_provider()
        if destination is None or str(destination).strip() == "":
            return
        path = Path(destination)
        if path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")
        lines = [
            self.recent_events.item(index).text()
            for index in range(self.recent_events.count())
            if "подій немає" not in self.recent_events.item(index).text()
        ]
        try:
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except OSError as exc:
            self.action_status.setText(f"Експорт подій: помилка — {exc}")
            self.action_status.setStyleSheet("color: #ff6b78;")
            return
        self.action_status.setText(f"Експортовано події: {path}")
        self.action_status.setStyleSheet("color: #4ee0a0;")

    def _mark_settings_dirty(self, *_args) -> None:
        self.apply_button.setProperty("settingsDirty", True)
        self.apply_button.setText(self._translate("Застосувати"))
        self.apply_button.style().unpolish(self.apply_button)
        self.apply_button.style().polish(self.apply_button)

    def _mark_settings_saved(self) -> None:
        self.apply_button.setProperty("settingsDirty", False)
        self.apply_button.setText(self._translate("Збережено"))
        self.apply_button.style().unpolish(self.apply_button)
        self.apply_button.style().polish(self.apply_button)

    def _apply_settings(self) -> None:
        settings = AlphaAppSettings(
            locale=self.locale_combo.currentData(),
            voice_enabled=self.voice_check.isChecked(),
            popup_enabled=self.popup_check.isChecked(),
            voice_volume=self.volume_slider.value() / 100,
            popup_duration_ms=self.duration_spin.value(),
            minimap_loss_delay_seconds=self.minimap_loss_delay_slider.value(),
            ship_hp_warning_percent=self.hp_warning_slider.value(),
            ship_rations_warning_percent=self.rations_warning_slider.value(),
            repeat_alerts_enabled=self.repeat_alerts_check.isChecked(),
            repeat_alert_interval_seconds=self.repeat_alert_interval_slider.value(),
        )
        def apply_and_refresh_locale() -> None:
            self.controller.apply(settings)
            self._set_locale(settings.locale)
            if self._session_running:
                configure_delivery = getattr(self._runtime_session, "configure_delivery", None)
                if callable(configure_delivery):
                    configure_delivery(
                        self.controller.enabled_adapters(), settings.minimap_loss_delay_seconds
                    )
                configure = getattr(self._runtime_session, "configure_resource_warning_zones", None)
                if callable(configure):
                    configure(
                        settings.ship_hp_warning_percent,
                        settings.ship_rations_warning_percent,
                    )
                configure_repeats = getattr(self._runtime_session, "configure_repeat_alerts", None)
                if callable(configure_repeats):
                    configure_repeats(
                        settings.repeat_alerts_enabled,
                        settings.repeat_alert_interval_seconds,
                    )
            self._mark_settings_saved()

        self._run_action("Налаштування", apply_and_refresh_locale)

    def _register_static_texts(self) -> None:
        for widget_type in (QLabel, QPushButton, QCheckBox):
            for widget in self.findChildren(widget_type):
                if widget.text() in _UI_TEXT:
                    widget.setProperty("localeSource", widget.text())
        for list_widget in self.findChildren(QListWidget):
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if item.text() in _UI_TEXT:
                    item.setData(Qt.UserRole, item.text())

    def _translate(self, text: str) -> str:
        return _UI_TEXT.get(text, text) if self.model.locale == "en" else text

    def _refresh_static_texts(self) -> None:
        for widget_type in (QLabel, QPushButton, QCheckBox):
            for widget in self.findChildren(widget_type):
                source = widget.property("localeSource")
                if widget is self.connection_badge and widget.property("status") != "idle":
                    continue
                if isinstance(source, str):
                    widget.setText(self._translate(source))
        for list_widget in self.findChildren(QListWidget):
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                source = item.data(Qt.UserRole)
                if isinstance(source, str):
                    item.setText(self._translate(source))
        self.minimap_loss_delay_value.setText(
            f"{self.minimap_loss_delay_slider.value()} {self._translate('с')}"
        )
        self.locale_combo.setItemText(
            0,
            "Ukrainian" if self.model.locale == "en" else "Українська",
        )

    def _set_locale(self, locale: str) -> None:
        self.model.set_locale(locale)
        titles_by_page = {
            item.page_id.value: item.title for item in self.model.navigation_items()
        }
        for button in self.navigation_buttons:
            title = titles_by_page[button.property("pageId")]
            button.setText(title)
            button.setAccessibleName(title)
        self._refresh_static_texts()
        self._activate_page(self.page_stack.currentIndex())

    def _run_action(self, label: str, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:
            self.action_status.setVisible(True)
            self.action_status.setText(f"{label}: помилка — {exc}")
            self.action_status.setStyleSheet("color: #ff6b78;")
        else:
            self.action_status.setVisible(True)
            self.action_status.setText(f"{label}: виконано")
            self.action_status.setStyleSheet("color: #4ee0a0;")

    @staticmethod
    def _card() -> QFrame:
        card = QFrame()
        card.setProperty("card", True)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return card

    @staticmethod
    def _muted(text: str, *, object_name: str = "") -> QLabel:
        label = QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        label.setProperty("muted", True)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _small_label(text: str, color: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("eyebrow")
        label.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 11px;")
        return label

    def _metric_card(self, title: str, value: str) -> QFrame:
        card = self._card()
        card.setMinimumHeight(92)
        layout = QVBoxLayout(card)
        heading = QLabel(title)
        heading.setProperty("cardTitle", True)
        metric = QLabel(value, objectName="metricValue")
        metric.setObjectName("metricValue")
        metric.setProperty("metric", True)
        layout.addWidget(heading)
        layout.addStretch(1)
        layout.addWidget(metric)
        return card


def wire_production_runtime(
    window: ModernMainWindow,
    controller: AlphaAppController,
) -> RuntimeSessionController:
    def on_runtime_status(status) -> None:
        window._runtime_signals.runtime_status.emit(status)

    def bridge_factory():
        if window.selected_window is None:
            raise RuntimeError("BDO window not selected")
        inputs = RuntimeFactoryInputs(
            candidate=window.selected_window,
            calibration=window.calibration_readiness,
            anchor=window._runtime_anchor,

            adapters=controller.enabled_adapters(),
            status_callback=on_runtime_status,
            minimap_loss_delay_seconds=controller.settings.minimap_loss_delay_seconds,
            ship_hp_warning_percent=controller.settings.ship_hp_warning_percent,
            ship_rations_warning_percent=controller.settings.ship_rations_warning_percent,
            repeat_alerts_enabled=controller.settings.repeat_alerts_enabled,
            repeat_alert_interval_seconds=controller.settings.repeat_alert_interval_seconds,
        )
        return build_runtime_bridge(inputs)

    session = RuntimeSessionController(bridge_factory, lambda snapshot: None)
    window.attach_runtime_session(session, route_evidence_available=True)
    return session


def run() -> int:
    from .desktop_ui import build_default_controller

    application = configure_application()
    controller = build_default_controller()
    window = ModernMainWindow(controller)
    wire_production_runtime(window, controller)
    window.show()
    return application.exec()
