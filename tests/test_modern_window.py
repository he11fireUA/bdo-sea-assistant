import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QStackedWidget

from bdo_assistant.app import AlphaAppSettings
from bdo_assistant.calibration_flow import build_default_calibration_flow
from bdo_assistant.calibration_store import CalibrationReadiness, CalibrationStatus, JsonCalibrationStore
from bdo_assistant.modern_ui import DashboardSnapshot, PageId
from bdo_assistant import modern_window
from bdo_assistant.modern_window import ModernMainWindow, ResourceGauge, wire_production_runtime
from bdo_assistant.monitoring import MonitoringResult, MonitoringSample
from bdo_assistant.navigation_state import NavigationState
from bdo_assistant.notifications import DeliveryResult, EventType, NotificationEvent, Severity
from bdo_assistant.runtime import RuntimeBridge, RuntimeStatus
from bdo_assistant.runtime_session import RuntimeSessionSnapshot, RuntimeSessionState
from bdo_assistant.window_discovery import WindowCandidate


class FakeController:
    def __init__(self) -> None:
        self.settings = AlphaAppSettings()
        self.voice_tests = 0
        self.popup_tests = 0
        self.applied = []

    def apply(self, settings: AlphaAppSettings) -> None:
        self.settings = settings
        self.applied.append(settings)

    def test_voice(self) -> None:
        self.voice_tests += 1

    def test_popup(self) -> None:
        self.popup_tests += 1

    def enabled_adapters(self) -> tuple:
        return ()


class FakeRuntimeSession:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.state_callback = None

    def set_state_callback(self, callback) -> None:
        self.state_callback = callback

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def emit(self, snapshot) -> None:
        self.state_callback(snapshot)


class FakeCalibrationFlow:
    def __init__(self) -> None:
        self.captured = 0
        self.detected = 0
        self.saved = []
        self.preview = None
        self.manual_regions = None

    def capture(self, frame) -> None:
        self.captured += 1

    def detect(self, anchor) -> None:
        self.detected += 1
        import numpy as np
        self.preview = np.full((80, 160, 3), 90, dtype=np.uint8)

    def confirm_and_save(self, path) -> None:
        self.saved.append(path)

    def retry(self) -> None:
        self.preview = None

    def apply_manual_regions(self, regions) -> None:
        self.manual_regions = regions


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_user_interface_has_no_route_recorder() -> None:
    _app()
    window = ModernMainWindow(FakeController())
    for locale in ("uk", "en"):
        window._set_locale(locale)
        assert all(button.property("pageId") != "recorder" for button in window.navigation_buttons)
        assert window.findChild(QScrollArea, "recorderScroll") is None
        assert window.findChild(QPushButton, "recorderStartButton") is None
    window.close()


def test_personalized_build_identity_is_visible_in_title_and_update_card() -> None:
    from bdo_assistant.build_identity import BuildIdentity

    _app()
    identity = BuildIdentity(
        recipient="Zedorovich",
        build_id="test-zedorovich",
        window_title="BDO Sea Assistant - Alpha for Zedorovich test version",
        credit_line="Coded by he11fireUA",
    )
    window = ModernMainWindow(FakeController(), build_identity=identity)

    assert window.windowTitle() == identity.window_title
    update = window.findChild(QLabel, "updateChannelInfo")
    assert update is not None
    assert update.text().endswith(identity.credit_line)
    window.close()


def test_modern_window_builds_complete_navigation_and_pages() -> None:
    _app()
    window = ModernMainWindow(FakeController())

    stack = window.findChild(QStackedWidget, "pageStack")
    assert stack is not None
    assert stack.count() == len(window.navigation_buttons)
    assert [stack.widget(index).objectName() for index in range(stack.count())] == [
        f"{button.property('pageId')}Scroll" for button in window.navigation_buttons
    ]
    assert window.findChild(QScrollArea, "navigationScroll") is None
    assert window.findChild(QScrollArea, "diagnosticsScroll") is None
    assert window.findChild(QLabel, "candidateMotionValue") is None
    assert [button.property("pageId") for button in window.navigation_buttons] == [
        PageId.DASHBOARD.value,
        PageId.CAPTURE.value,
        PageId.CALIBRATION.value,
        PageId.ALERTS.value,
        PageId.SETTINGS.value,
    ]
    assert window.objectName() == "modernMainWindow"
    assert window.minimumWidth() >= 1120
    assert window.minimumHeight() >= 720
    window.close()


def test_sidebar_uses_large_consistent_vector_icons_and_readable_rows() -> None:
    _app()
    window = ModernMainWindow(FakeController())

    visible_items = window.model.navigation_items()
    for button, item in zip(window.navigation_buttons, visible_items):
        assert button.text() == item.title
        assert not button.icon().isNull()
        assert button.iconSize().width() >= 20
        assert button.iconSize().height() >= 20
        assert button.minimumHeight() >= 44
        assert button.accessibleName() == item.title
        assert button.isCheckable()
        assert bool(button.accessibleDescription())
    window.close()


def test_unavailable_runtime_actions_are_honestly_disabled() -> None:
    _app()
    window = ModernMainWindow(FakeController())

    start = window.findChild(QPushButton, "startMonitoringButton")
    calibration = window.findChild(QPushButton, "beginCalibrationButton")

    assert start is not None and not start.isEnabled()
    assert "runtime" in start.toolTip().lower()
    assert calibration is not None and not calibration.isEnabled()
    assert "live" in calibration.toolTip().lower()
    window.close()


def test_default_gui_wires_calibration_after_window_selection_without_capture() -> None:
    _app()
    candidate = WindowCandidate(
        777, 88, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False
    )
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
    )

    assert not window.begin_calibration_button.isEnabled()
    window.refresh_window_candidates()

    assert window.begin_calibration_button.isEnabled()
    assert window._calibration_flow is not None
    assert callable(window._calibration_frame_provider)
    assert callable(window._manual_region_provider)
    window.close()


def test_alert_test_buttons_use_existing_controller() -> None:
    _app()
    controller = FakeController()
    window = ModernMainWindow(controller)

    window.findChild(QPushButton, "testVoiceButton").click()
    window.findChild(QPushButton, "testPopupButton").click()

    assert controller.voice_tests == 1
    assert controller.popup_tests == 1
    window.close()


def test_sidebar_switches_page_and_marks_active_navigation() -> None:
    _app()
    window = ModernMainWindow(FakeController())
    stack = window.findChild(QStackedWidget, "pageStack")

    alerts = next(button for button in window.navigation_buttons if button.property("pageId") == PageId.ALERTS.value)
    alerts.click()

    assert stack.currentWidget().objectName() == "alertsScroll"
    assert alerts.property("active") is True
    assert window.navigation_buttons[0].property("active") is False
    window.close()


def test_settings_page_applies_validated_values_to_controller() -> None:
    _app()
    controller = FakeController()
    window = ModernMainWindow(controller)
    window.locale_combo.setCurrentIndex(1)
    window.voice_check.setChecked(False)
    window.popup_check.setChecked(True)
    window.volume_slider.setValue(35)
    window.hp_warning_slider.setValue(73)
    window.rations_warning_slider.setValue(27)
    window.duration_spin.setValue(6500)

    window._apply_settings()

    assert window.model.locale == "en"
    assert window.navigation_buttons[0].text() == "Dashboard"
    assert window.top_title.text() == "Dashboard"
    assert controller.applied[-1] == AlphaAppSettings(
        locale="en",
        voice_enabled=False,
        popup_enabled=True,
        voice_volume=0.35,
        popup_duration_ms=6500,
        ship_hp_warning_percent=73,
        ship_rations_warning_percent=27,
    )
    window.close()


def test_settings_apply_button_tracks_unsaved_changes_and_confirms_save() -> None:
    _app()
    window = ModernMainWindow(FakeController())
    button = window.findChild(QPushButton, "primaryButton")

    assert button is not None
    assert button.text() == "Застосувати"
    assert button.property("settingsDirty") is False

    window.volume_slider.setValue(35)

    assert button.property("settingsDirty") is True
    assert button.text() == "Застосувати"

    window._apply_settings()

    assert button.property("settingsDirty") is False
    assert button.text() == "Збережено"

    window.popup_check.setChecked(False)

    assert button.property("settingsDirty") is True
    assert button.text() == "Застосувати"
    window.close()


def test_locale_switch_translates_visible_pages_without_changing_runtime_state() -> None:
    _app()
    window = ModernMainWindow(FakeController())
    next(button for button in window.navigation_buttons if button.property("pageId") == "alerts").click()
    window.volume_slider.setValue(35)
    window.start_button.setEnabled(True)
    window.connection_badge.setText("●  RUNNING")
    window.connection_badge.setProperty("status", "ok")

    window._set_locale("en")

    assert {
        button.property("pageId"): button.text()
        for button in window.navigation_buttons
    } == {
        PageId.DASHBOARD.value: "Dashboard",
        PageId.CAPTURE.value: "Capture",
        PageId.CALIBRATION.value: "Calibration",
        PageId.ALERTS.value: "Alerts",
        PageId.SETTINGS.value: "Settings",
    }
    assert window.top_title.text() == "Alerts"
    assert window.top_subtitle.text() == "Local independent channels"
    assert window.start_button.text() == "▶  Start monitoring"
    assert window.findChild(QPushButton, "refreshWindowsButton").text() == "Capture window"
    assert window.findChild(QPushButton, "beginCalibrationButton").text() == "Start calibration"
    assert window.findChild(QPushButton, "testVoiceButton").text() == "Test Voice"
    assert window.findChild(QPushButton, "primaryButton").text() == "Apply"
    assert window.findChild(QPushButton, "recorderStartButton") is None
    assert window.recent_events.item(0).text() == "The session has not started yet — no events"
    assert window.page_stack.currentWidget().objectName() == "alertsScroll"
    assert window.start_button.isEnabled()
    assert window.connection_badge.text() == "●  RUNNING"
    assert window.volume_slider.value() == 35

    window._set_locale("uk")

    assert window.navigation_buttons[0].text() == "Огляд"
    assert window.start_button.text() == "▶  Почати моніторинг"
    assert window.page_stack.currentWidget().objectName() == "alertsScroll"
    assert window.start_button.isEnabled()
    assert window.volume_slider.value() == 35
    window.close()


def test_apply_settings_updates_warning_zones_in_active_runtime_session() -> None:
    class RuntimeSession:
        def __init__(self) -> None:
            self.updated = []

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def configure_resource_warning_zones(self, hp: int, rations: int) -> None:
            self.updated.append((hp, rations))

    _app()
    session = RuntimeSession()
    window = ModernMainWindow(FakeController(), runtime_session=session)
    window._session_running = True
    window.hp_warning_slider.setValue(60)
    window.rations_warning_slider.setValue(50)

    window._apply_settings()

    assert session.updated == [(60, 50)]
    window.close()


def test_voice_volume_slider_shows_live_percent_from_1_to_100() -> None:
    _app()
    window = ModernMainWindow(FakeController())
    value = window.findChild(QLabel, "voiceVolumeValue")

    assert window.volume_slider.minimum() == 1
    assert window.volume_slider.maximum() == 100
    assert value is not None
    assert value.text() == "80%"

    window.volume_slider.setValue(35)

    assert value.text() == "35%"
    window.close()


def test_resource_warning_sliders_show_live_percent_from_1_to_99() -> None:
    _app()
    window = ModernMainWindow(FakeController())

    window.hp_warning_slider.setValue(1)
    window.rations_warning_slider.setValue(99)

    assert window.hp_warning_slider.minimum() == 1
    assert window.hp_warning_slider.maximum() == 99
    assert window.hp_warning_value.text() == "1%"
    assert window.rations_warning_slider.minimum() == 1
    assert window.rations_warning_slider.maximum() == 99
    assert window.rations_warning_value.text() == "99%"
    window.close()


def test_minimap_loss_delay_slider_shows_live_seconds_from_10_to_300() -> None:
    _app()
    window = ModernMainWindow(FakeController())
    value = window.findChild(QLabel, "minimapLossDelayValue")

    assert window.minimap_loss_delay_slider.minimum() == 10
    assert window.minimap_loss_delay_slider.maximum() == 300
    assert value is not None
    assert value.text() == "10 с"

    window.minimap_loss_delay_slider.setValue(75)

    assert value.text() == "75 с"
    window._apply_settings()
    assert window.controller.applied[-1].minimap_loss_delay_seconds == 75
    window.close()


def test_resource_gauge_holds_two_percent_measurement_jitter() -> None:
    _app()
    gauge = ResourceGauge("SHIP HP", "#ff5d6c")

    for percent in (87, 89, 87, 89, 87, 89):
        gauge.set_percent(percent)

    assert gauge.percent == 87
    gauge.set_percent(83)
    gauge.set_percent(83)
    assert gauge.percent == 83
    gauge.close()


def test_resource_gauge_eventually_accepts_stable_small_change() -> None:
    _app()
    gauge = ResourceGauge("SHIP HP", "#ff5d6c")
    gauge.set_percent(98)
    for _ in range(5):
        gauge.set_percent(100)
    assert gauge.percent == 100
    gauge.set_percent(None)
    gauge.set_percent(1)
    for _ in range(5):
        gauge.set_percent(0)
    assert gauge.percent == 0
    gauge.close()


def test_settings_apply_updates_active_delivery_and_delay() -> None:
    class Session(FakeRuntimeSession):
        def configure_delivery(self, adapters, delay):
            self.delivery = (adapters, delay)
    _app()
    session = Session()
    controller = FakeController()
    window = ModernMainWindow(controller, runtime_session=session)
    window._session_running = True
    window.minimap_loss_delay_slider.setValue(75)
    window._apply_settings()
    assert session.delivery == (controller.enabled_adapters(), 75)
    assert window.apply_button.text() == "Збережено"
    window.close()


def test_settings_save_waits_for_runtime_apply_success() -> None:
    class Session:
        def start(self):
            pass
        def configure_resource_warning_zones(self, *args):
            raise RuntimeError("configuration failed")
        def stop(self):
            pass
    _app()
    window = ModernMainWindow(FakeController(), runtime_session=Session())
    window._session_running = True
    window.volume_slider.setValue(37)
    window._apply_settings()
    assert window.apply_button.text() == "Застосувати"
    assert window.apply_button.property("settingsDirty") is True
    assert "configuration failed" in window.action_status.text()
    window.close()


def test_dashboard_snapshot_updates_metrics_and_resource_gauges() -> None:
    _app()
    window = ModernMainWindow(FakeController())

    window.update_dashboard(
        DashboardSnapshot(
            monitoring="running",
            capture="active",
            calibration="valid",
            minimap="visible",
            navigation="moving",
            hp_percent=82,
            rations_percent=44,
        )
    )

    assert window.metric_labels["capture"].text() == "ACTIVE"
    assert window.metric_labels["navigation"].text() == "MOVING"
    assert window.metric_labels["navigation"].styleSheet() == "color: #4ee0a0;"
    assert window.hp_gauge.percent == 82
    assert window.rations_gauge.percent == 44
    window.close()


@pytest.mark.parametrize(
    ("navigation", "color"),
    [
        ("arrived", "#32c8df"),
        ("stuck", "#ff6b78"),
        ("recovering", "#8aeaf7"),
        ("unknown", "#ffc857"),
    ],
)
def test_dashboard_colors_each_navigation_state(navigation: str, color: str) -> None:
    _app()
    window = ModernMainWindow(FakeController())

    window.update_dashboard(DashboardSnapshot(navigation=navigation))

    assert window.metric_labels["navigation"].styleSheet() == f"color: {color};"
    window.close()


def test_dashboard_fits_default_window_without_vertical_scrolling() -> None:
    app = _app()
    window = ModernMainWindow(FakeController())
    window.resize(1380, 860)
    window.show()
    app.processEvents()

    scroll = window.findChild(QScrollArea, "dashboardScroll")

    assert scroll is not None
    assert scroll.verticalScrollBar().maximum() == 0
    assert window.recent_events.isVisible()
    window.close()


def test_dashboard_retains_last_resources_only_when_ship_hud_is_absent() -> None:
    _app()
    window = ModernMainWindow(FakeController())
    window.update_dashboard(
        DashboardSnapshot(
            monitoring="running",
            capture="active",
            calibration="valid",
            minimap="visible",
            navigation="moving",
            hp_percent=82,
            rations_percent=44,
            resources="visible",
        )
    )
    window.update_dashboard(
        DashboardSnapshot(
            monitoring="running",
            capture="active",
            calibration="valid",
            minimap="visible",
            navigation="moving",
            resources="absent",
        )
    )

    assert window.hp_gauge.percent == 82
    assert window.rations_gauge.percent == 44

    window.update_dashboard(
        DashboardSnapshot(
            monitoring="running",
            capture="active",
            calibration="valid",
            minimap="lost",
            navigation="unknown",
        )
    )
    assert window.hp_gauge.percent is None
    assert window.rations_gauge.percent is None
    window.close()


def test_modern_window_exposes_desktop_entrypoint() -> None:
    assert callable(modern_window.run)


def test_capture_page_refreshes_real_window_candidates_without_starting_capture() -> None:
    _app()
    candidate = WindowCandidate(
        hwnd=777,
        pid=88,
        title="BLACK DESERT - 528853",
        executable="BlackDesert64.exe",
        width=5120,
        height=1440,
        minimized=False,
    )
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
    )

    window.refresh_window_candidates()

    assert window.window_combo.count() == 1
    assert window.window_combo.currentData() == candidate
    assert "5120×1440" in window.window_detail.text()
    assert window.findChild(QPushButton, "refreshWindowsButton").text() == "Захопити вікно"
    assert window.selected_window == candidate
    assert not window.start_button.isEnabled()
    assert "runtime factory" in window.start_button.toolTip().lower()
    window.close()


def test_capture_page_reports_no_open_bdo_window() -> None:
    _app()
    window = ModernMainWindow(FakeController(), window_discovery=lambda: ())

    window.refresh_window_candidates()

    assert window.window_combo.count() == 0
    assert "не знайдено" in window.window_detail.text().lower()
    assert window.selected_window is None
    window.close()


def test_calibration_page_reports_missing_layout_fail_closed() -> None:
    _app()
    window = ModernMainWindow(
        FakeController(),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.MISSING,
            "calibration file not found",
        ),
    )

    assert "MISSING" in window.calibration_status.text()
    assert "calibration file not found" in window.calibration_diagnostic.text()
    assert not window.start_button.isEnabled()
    window.close()


def test_calibration_page_reports_valid_confirmed_layout() -> None:
    _app()
    window = ModernMainWindow(
        FakeController(),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
    )

    assert "VALID" in window.calibration_status.text()
    assert "confirmed calibration loaded" in window.calibration_diagnostic.text()
    window.close()


def test_runtime_blockers_shrink_after_window_and_calibration_become_ready() -> None:
    _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
    )

    window.refresh_window_candidates()

    blocker_text = window.runtime_blockers.text()
    assert "BDO window not selected" not in blocker_text
    assert "confirmed calibration missing" not in blocker_text
    assert "verified minimap anchor missing" not in blocker_text
    assert "runtime factory unavailable" in blocker_text
    assert not window.start_button.isEnabled()
    window.close()


def test_runtime_blockers_restore_window_missing_after_discovery_loses_candidate() -> None:
    _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
    )
    window.refresh_window_candidates()
    assert "BDO window not selected" not in window.runtime_blockers.text()

    window._window_discovery = lambda: ()
    window.refresh_window_candidates()

    assert "BDO window not selected" in window.runtime_blockers.text()
    assert not window.start_button.isEnabled()
    window.close()


def test_ready_gui_start_and_stop_use_injected_runtime_session() -> None:
    _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)
    session = FakeRuntimeSession()
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
        runtime_session=session,
        route_evidence_available=True,
    )
    window.refresh_window_candidates()
    assert window.start_button.isEnabled()

    window.start_button.click()

    assert session.start_calls == 1
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()
    assert "RUNNING" in window.connection_badge.text()

    window.stop_button.click()

    deadline = time.monotonic() + 1.0
    while not window.start_button.isEnabled() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert session.stop_calls == 1
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()
    assert "STOPPED" in window.connection_badge.text()
    window.close()


def test_gui_reacts_to_asynchronous_runtime_session_error() -> None:
    app = _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)
    session = FakeRuntimeSession()
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
        runtime_session=session,
        route_evidence_available=True,
    )
    window.refresh_window_candidates()
    window.start_button.click()

    session.emit(
        RuntimeSessionSnapshot(
            RuntimeSessionState.ERROR,
            "capture closed",
            bridge_active=False,
        )
    )
    app.processEvents()

    assert "ERROR" in window.connection_badge.text()
    assert "capture closed" in window.connection_badge.text()
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()
    window.close()


def test_gui_calibration_shows_preview_and_saves_only_after_confirm(tmp_path) -> None:
    import numpy as np

    _app()
    flow = FakeCalibrationFlow()
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=flow,
        calibration_frame_provider=lambda: np.zeros((80, 160, 3), dtype=np.uint8),
        calibration_destination=tmp_path / "ui_layout.json",
    )

    assert flow.saved == []
    window.begin_calibration_button.click()

    assert flow.captured == 1
    assert flow.detected == 1
    assert window.calibration_preview.pixmap() is not None
    assert window.confirm_calibration_button.isEnabled()
    assert flow.saved == []

    window.confirm_calibration_button.click()

    assert flow.saved == [tmp_path / "ui_layout.json"]
    window.close()


def test_gui_manual_correction_updates_preview_without_saving(tmp_path) -> None:
    import numpy as np

    _app()
    flow = FakeCalibrationFlow()
    regions = {"minimap": object(), "ship_hp_bar": object(), "ship_rations_bar": object()}
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=flow,
        calibration_frame_provider=lambda: np.zeros((80, 160, 3), dtype=np.uint8),
        calibration_destination=tmp_path / "ui_layout.json",
        manual_region_provider=lambda flow: regions,
    )
    window.begin_calibration_button.click()
    window.manual_calibration_button.click()

    assert flow.manual_regions == regions
    assert flow.saved == []
    assert window.confirm_calibration_button.isEnabled()
    window.close()


def test_real_calibration_preview_scales_inside_visible_label() -> None:
    import cv2

    app = _app()
    frame = cv2.imread("tests/fixtures/full_frame_arehaza.jpg", cv2.IMREAD_COLOR)
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=build_default_calibration_flow(),
        calibration_frame_provider=lambda: frame,
    )
    window.resize(1380, 860)
    window.show()
    window.begin_calibration_button.click()
    app.processEvents()

    pixmap = window.calibration_preview.pixmap()
    assert pixmap.width() <= window.calibration_preview.width()
    assert pixmap.height() <= window.calibration_preview.height()
    window.close()


def test_confirmed_real_calibration_refreshes_gui_to_valid(tmp_path) -> None:
    import cv2

    _app()
    destination = tmp_path / "ui_layout.json"
    frame = cv2.imread("tests/fixtures/full_frame_arehaza.jpg", cv2.IMREAD_COLOR)
    store = JsonCalibrationStore(destination)
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=build_default_calibration_flow(),
        calibration_frame_provider=lambda: frame,
        calibration_destination=destination,
        calibration_inspector=store.inspect,
    )
    window.begin_calibration_button.click()
    window.confirm_calibration_button.click()

    assert window.calibration_readiness.status is CalibrationStatus.VALID
    assert window.calibration_status.text() == "VALID"
    assert "confirmed calibration missing" not in window.runtime_blockers.text()
    window.close()


def test_production_runtime_wiring_is_lazy_and_enables_route_monitoring() -> None:
    _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
    )

    session = wire_production_runtime(window, window.controller)
    window.refresh_window_candidates()

    assert window._runtime_session is session
    assert window.start_button.isEnabled()
    assert window._route_evidence_available
    assert "route/endpoint evidence unavailable" not in window.runtime_blockers.text()
    assert session.bridge is None
    window.close()


def test_production_runtime_uses_configured_minimap_loss_delay(monkeypatch) -> None:
    _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)
    controller = FakeController()
    controller.settings = AlphaAppSettings(
        minimap_loss_delay_seconds=75,
        ship_hp_warning_percent=73,
        ship_rations_warning_percent=27,
    )
    captured_inputs = []

    class Bridge:
        running = False

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.running = False

    def build_bridge(inputs):
        captured_inputs.append(inputs)
        return Bridge()

    monkeypatch.setattr(modern_window, "build_runtime_bridge", build_bridge)
    window = ModernMainWindow(
        controller,
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
    )
    wire_production_runtime(window, controller)
    window.refresh_window_candidates()

    window.start_button.click()

    assert captured_inputs[0].minimap_loss_delay_seconds == 75
    assert captured_inputs[0].ship_hp_warning_percent == 73
    assert captured_inputs[0].ship_rations_warning_percent == 27
    window.stop_button.click()
    window.close()


def test_runtime_status_signal_updates_live_dashboard_cards_and_gauges() -> None:
    app = _app()
    window = ModernMainWindow(FakeController())
    sample = MonitoringSample(7, True, True, hp_percent=81, rations_percent=43)
    status = RuntimeStatus(
        True,
        7,
        NavigationState.MOVING,
        diagnostic="live frame",
        result=MonitoringResult(7, (), NavigationState.MOVING, minimap_visible=True),
        sample=sample,
    )

    window._runtime_signals.runtime_status.emit(status)
    app.processEvents()

    assert window.metric_labels["capture"].text() == "ACTIVE"
    assert window.metric_labels["calibration"].text() == "VALID"
    assert window.metric_labels["minimap"].text() == "VISIBLE"
    assert window.metric_labels["navigation"].text() == "MOVING"
    assert window.hp_gauge.percent == 81
    assert window.rations_gauge.percent == 43
    assert window.model.dashboard.diagnostic == "live frame"
    window.close()


def test_runtime_shadow_metrics_callback_preserves_live_dashboard_and_events() -> None:
    import threading
    import time

    def wait_until(predicate) -> bool:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    class Source:
        def start(self, on_frame, _on_closed) -> None:
            self.on_frame = on_frame

        def stop(self) -> None:
            pass

    class Builder:
        def build(self, sequence, _frame) -> MonitoringSample:
            return MonitoringSample(
                sequence,
                True,
                True,
                hp_percent=81,
                rations_percent=43,
                resource_bars_visible=True,
            )

    class Loop:
        def process(self, sample) -> MonitoringResult:
            return MonitoringResult(
                sample.sequence,
                (),
                NavigationState.MOVING,
                minimap_visible=True,
            )

    class Router:
        def route(self, _event):
            return ()

    class Candidate:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def process(self, _sample):
            self.entered.set()
            assert self.release.wait(1)
            return {"motion": "moving", "reason": "coherent_texture", "quality": "high"}

        def reset(self) -> None:
            pass

    app = _app()
    window = ModernMainWindow(FakeController())
    source = Source()
    candidate = Candidate()
    bridge = RuntimeBridge(
        source,
        Builder(),
        Loop(),
        Router(),
        window._runtime_signals.runtime_status.emit,
        shadow_processor=candidate,
    )

    bridge.start()
    source.on_frame(object())
    assert candidate.entered.wait(1)
    app.processEvents()
    dashboard_before = {
        key: window.metric_labels[key].text()
        for key in ("capture", "calibration", "minimap", "navigation")
    }
    resources_before = window.model.dashboard.resources
    gauges_before = (window.hp_gauge.percent, window.rations_gauge.percent)
    snapshot_before = window.model.dashboard
    events_before = [
        window.recent_events.item(index).text() for index in range(window.recent_events.count())
    ]

    candidate.release.set()
    assert wait_until(lambda: bridge.shadow_accounting.completed == 1)
    app.processEvents()

    assert {key: window.metric_labels[key].text() for key in dashboard_before} == dashboard_before
    assert window.model.dashboard.resources == resources_before == "visible"
    assert (window.hp_gauge.percent, window.rations_gauge.percent) == gauges_before == (81, 43)
    assert window.model.dashboard == snapshot_before
    assert [window.recent_events.item(index).text() for index in range(window.recent_events.count())] == events_before
    bridge.stop()
    window.close()


def test_runtime_status_surfaces_events_with_monitoring_and_adapter_failures() -> None:
    app = _app()
    window = ModernMainWindow(FakeController())
    result = MonitoringResult(
        8,
        (NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL),),
        NavigationState.UNKNOWN,
        diagnostics=("minimap anchor confidence low", "2D minimap not observable"),
        minimap_visible=False,
    )
    status = RuntimeStatus(
        True,
        8,
        NavigationState.UNKNOWN,
        diagnostic="frame processed",
        result=result,
        deliveries=(DeliveryResult("voice", False, "device unavailable"),),
        sample=MonitoringSample(8, True, True),
    )

    window._runtime_signals.runtime_status.emit(status)
    app.processEvents()

    events = [window.recent_events.item(index).text() for index in range(window.recent_events.count())]
    assert any(entry.endswith(" CAPTURE_LOST") for entry in events)
    assert "voice: device unavailable" in window.runtime_diagnostics
    assert "2D minimap not observable" in window.runtime_diagnostics
    assert window.runtime_diagnostics.maxlen == 200
    window.close()


def test_recent_events_show_local_timestamp_and_space_for_five_rows() -> None:
    app = _app()
    window = ModernMainWindow(
        FakeController(),
        event_time_provider=lambda: "14:05:09",
    )
    status = RuntimeStatus(
        True,
        9,
        NavigationState.UNKNOWN,
        result=MonitoringResult(
            9,
            tuple(
                NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL)
                for _ in range(5)
            ),
            NavigationState.UNKNOWN,
        ),
    )

    window._runtime_signals.runtime_status.emit(status)
    app.processEvents()

    assert window.recent_events.item(0).text() == "[14:05:09] CAPTURE_LOST"
    visible_rows_height = sum(
        window.recent_events.sizeHintForRow(index) for index in range(5)
    )
    assert window.recent_events.viewport().height() >= visible_rows_height
    window.close()


def test_export_events_button_writes_one_utf8_event_per_line(tmp_path) -> None:
    app = _app()
    destination = tmp_path / "events.txt"
    window = ModernMainWindow(
        FakeController(),
        event_time_provider=lambda: "14:05:09",
        event_export_path_provider=lambda: destination,
    )
    status = RuntimeStatus(
        True,
        10,
        NavigationState.UNKNOWN,
        result=MonitoringResult(
            10,
            (
                NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL),
                NotificationEvent(EventType.CAPTURE_RESTORED, Severity.INFO),
            ),
            NavigationState.UNKNOWN,
        ),
    )
    window._runtime_signals.runtime_status.emit(status)
    app.processEvents()

    window.findChild(QPushButton, "exportEventsButton").click()

    assert destination.read_text(encoding="utf-8") == (
        "[14:05:09] CAPTURE_RESTORED\n"
        "[14:05:09] CAPTURE_LOST\n"
    )
    window.close()


def test_async_capture_close_is_serialized_after_user_stop(monkeypatch) -> None:
    import threading

    app = _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)
    callbacks = []

    class Bridge:
        def __init__(self) -> None:
            self.running = False
            self.stop_calls = 0

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.stop_calls += 1
            self.running = False

    bridge = Bridge()

    def build_bridge(inputs):
        callbacks.append(inputs.status_callback)
        return bridge

    monkeypatch.setattr(modern_window, "build_runtime_bridge", build_bridge)
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
    )
    session = wire_production_runtime(window, window.controller)
    window.refresh_window_candidates()
    window.start_button.click()

    worker = threading.Thread(
        target=lambda: callbacks[0](
            RuntimeStatus(False, 1, NavigationState.UNKNOWN, diagnostic="capture closed")
        )
    )
    worker.start()
    worker.join()
    window.stop_button.click()
    app.processEvents()

    assert bridge.stop_calls == 1
    assert session.state is RuntimeSessionState.STOPPED
    assert session.bridge is None
    window.close()


def test_closing_production_window_stops_active_runtime_once(monkeypatch) -> None:
    _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)

    class Bridge:
        def __init__(self) -> None:
            self.running = False
            self.stop_calls = 0

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.stop_calls += 1
            self.running = False

    bridge = Bridge()
    monkeypatch.setattr(modern_window, "build_runtime_bridge", lambda inputs: bridge)
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
    )
    session = wire_production_runtime(window, window.controller)
    window.refresh_window_candidates()
    window.start_button.click()

    window.close()

    assert bridge.stop_calls == 1
    assert session.state is RuntimeSessionState.STOPPED
    assert session.bridge is None


def test_runtime_stop_failure_keeps_window_open_and_surfaces_diagnostic() -> None:
    _app()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)

    class FailingRuntimeSession(FakeRuntimeSession):
        def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError("capture cleanup failed")

    session = FailingRuntimeSession()
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout={"schema_version": 1},
        ),
        runtime_session=session,
        route_evidence_available=True,
    )
    window.refresh_window_candidates()
    window.start_button.click()
    window.show()

    closed = window.close()

    assert closed is False
    assert window.isVisible()
    assert session.stop_calls == 1
    assert "runtime shutdown failed: capture cleanup failed" in (
        window.connection_badge.text()
    )
    session.stop = lambda: None
    window.close()


def test_recorder_page_records_selected_window_to_local_directory_with_labels_and_preview(tmp_path) -> None:
    import json
    import numpy as np

    class Source:
        def start(self, on_frame, on_closed) -> None:
            self.on_frame = on_frame
            self.on_closed = on_closed

        def stop(self) -> None:
            pass

    app = _app()
    source = Source()
    candidate = WindowCandidate(9, 10, "BLACK DESERT", "BlackDesert64.exe", 30, 20, False)
    layout = {
        "capture_size": {"width": 30, "height": 20},
        "regions": {"minimap": {"absolute": {"x": 0, "y": 0, "width": 20, "height": 10}}},
    }
    opened = []
    window = ModernMainWindow(
        FakeController(),
        window_discovery=lambda: (candidate,),
        calibration_inspector=lambda: CalibrationReadiness(CalibrationStatus.VALID, "confirmed", layout),
        developer_recorder=True,
        recorder_source_factory=lambda hwnd: source,
        recordings_directory=tmp_path / "local-recordings",
        recordings_folder_opener=lambda directory: opened.append(directory) or True,
    )
    window.refresh_window_candidates()

    window.findChild(QPushButton, "recorderStartButton").click()
    deadline = time.monotonic() + 1
    while not window.recorder_stop_button.isEnabled() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    source.on_frame(np.full((20, 30, 3), 17, dtype=np.uint8))
    app.processEvents()
    window.findChild(QPushButton, "recorderArrivedButton").click()
    window.findChild(QPushButton, "recorderMovingButton").click()
    window.findChild(QPushButton, "recorderStopButton").click()
    deadline = time.monotonic() + 1
    while not window.recorder_start_button.isEnabled() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    manifests = list((tmp_path / "local-recordings").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert [item["label"] for item in manifest["labels"]] == ["ARRIVED", "MOVING"]
    assert window.recorder_preview.pixmap() is not None
    assert window.recorder_moving_button.property("active") is True
    assert window.recorder_arrived_button.property("active") is False
    window.findChild(QPushButton, "recorderOpenFolderButton").click()
    assert opened == [tmp_path / "local-recordings"]
    window.close()


def test_gui_initial_preview_rejects_zero_dimension_array() -> None:
    import numpy as np

    _app()
    flow = FakeCalibrationFlow()
    flow.detect = lambda anchor: setattr(
        flow, "preview", np.zeros((0, 160, 3), dtype=np.uint8)
    )
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=flow,
        calibration_frame_provider=lambda: np.zeros((80, 160, 3), dtype=np.uint8),
    )

    window.begin_calibration_button.click()

    assert not window.confirm_calibration_button.isEnabled()
    pixmap = window.calibration_preview.pixmap()
    assert pixmap is None or pixmap.isNull()
    window.close()


@pytest.mark.parametrize("artifact", ["image", "pixmap"])
def test_gui_initial_preview_rejects_null_qt_raster(monkeypatch, artifact) -> None:
    import numpy as np

    _app()
    flow = FakeCalibrationFlow()
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=flow,
        calibration_frame_provider=lambda: np.zeros((80, 160, 3), dtype=np.uint8),
    )
    original_image = modern_window.QImage
    original_pixmap = modern_window.QPixmap
    if artifact == "image":
        class NullImageFactory:
            Format_RGB888 = original_image.Format_RGB888

            def __new__(cls, *args):
                return original_image()

        monkeypatch.setattr(modern_window, "QImage", NullImageFactory)
    else:
        class NullPixmapFactory:
            @staticmethod
            def fromImage(image):
                return original_pixmap()

        monkeypatch.setattr(modern_window, "QPixmap", NullPixmapFactory)

    window.begin_calibration_button.click()

    assert not window.confirm_calibration_button.isEnabled()
    window.close()


@pytest.mark.parametrize("failed_attempt", ["new_capture", "manual_replacement"])
def test_gui_failed_calibration_attempt_clears_stale_preview_and_controls(failed_attempt) -> None:
    import numpy as np

    _app()
    flow = FakeCalibrationFlow()
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=flow,
        calibration_frame_provider=lambda: np.zeros((80, 160, 3), dtype=np.uint8),
        manual_region_provider=lambda current: {"minimap": object()},
    )
    window.begin_calibration_button.click()
    assert window.confirm_calibration_button.isEnabled()

    if failed_attempt == "new_capture":
        window._calibration_frame_provider = lambda: (_ for _ in ()).throw(
            RuntimeError("capture lost")
        )
        window.begin_calibration_button.click()
    else:
        flow.apply_manual_regions = lambda regions: (_ for _ in ()).throw(
            RuntimeError("replacement failed")
        )
        window.manual_calibration_button.click()

    assert not window.confirm_calibration_button.isEnabled()
    assert not window.manual_calibration_button.isEnabled()
    assert not window.retry_calibration_button.isEnabled()
    pixmap = window.calibration_preview.pixmap()
    assert pixmap is None or pixmap.isNull()
    assert "preview ready" not in window.calibration_diagnostic.text().lower()
    window.close()


@pytest.mark.parametrize("artifact", ["array", "image", "pixmap"])
def test_gui_manual_preview_rejects_invalid_raster(monkeypatch, artifact) -> None:
    import numpy as np

    _app()
    flow = FakeCalibrationFlow()
    window = ModernMainWindow(
        FakeController(),
        calibration_flow=flow,
        calibration_frame_provider=lambda: np.zeros((80, 160, 3), dtype=np.uint8),
        manual_region_provider=lambda current: {"minimap": object()},
    )
    window.begin_calibration_button.click()
    original_apply = flow.apply_manual_regions
    original_image = modern_window.QImage
    original_pixmap = modern_window.QPixmap
    if artifact == "array":
        flow.apply_manual_regions = lambda regions: setattr(
            flow, "preview", np.zeros((80, 0, 3), dtype=np.uint8)
        )
    elif artifact == "image":
        class NullImageFactory:
            Format_RGB888 = original_image.Format_RGB888

            def __new__(cls, *args):
                return original_image()

        monkeypatch.setattr(modern_window, "QImage", NullImageFactory)
    else:
        class NullPixmapFactory:
            @staticmethod
            def fromImage(image):
                return original_pixmap()

        monkeypatch.setattr(modern_window, "QPixmap", NullPixmapFactory)
    if artifact != "array":
        flow.apply_manual_regions = original_apply

    window.manual_calibration_button.click()

    assert not window.confirm_calibration_button.isEnabled()
    assert not window.manual_calibration_button.isEnabled()
    assert not window.retry_calibration_button.isEnabled()
    pixmap = window.calibration_preview.pixmap()
    assert pixmap is None or pixmap.isNull()
    window.close()
