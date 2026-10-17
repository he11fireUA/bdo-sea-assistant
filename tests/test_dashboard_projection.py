from bdo_assistant.dashboard_projection import project_runtime_status
from bdo_assistant.monitoring import MonitoringResult, MonitoringSample
from bdo_assistant.navigation_state import NavigationState
from bdo_assistant.runtime import RuntimeStatus


def test_running_runtime_sample_projects_live_dashboard_values() -> None:
    sample = MonitoringSample(
        sequence=9,
        capture_ok=True,
        calibration_ok=True,
        hp_percent=82.5,
        rations_percent=44.0,
    )
    status = RuntimeStatus(
        running=True,
        sequence=9,
        navigation_state=NavigationState.MOVING,
        result=MonitoringResult(9, (), NavigationState.MOVING, minimap_visible=True),
        sample=sample,
    )

    snapshot = project_runtime_status(status, window_title="BLACK DESERT")

    assert snapshot.monitoring == "running"
    assert snapshot.capture == "active"
    assert snapshot.calibration == "valid"
    assert snapshot.minimap == "visible"
    assert snapshot.navigation == "moving"
    assert snapshot.hp_percent == 82.5
    assert snapshot.rations_percent == 44.0
    assert snapshot.window_title == "BLACK DESERT"


def test_closed_runtime_projects_lost_capture_and_preserves_unknown_sensors() -> None:
    status = RuntimeStatus(
        running=False,
        sequence=10,
        navigation_state=NavigationState.UNKNOWN,
        diagnostic="capture closed",
    )

    snapshot = project_runtime_status(status)

    assert snapshot.monitoring == "error"
    assert snapshot.capture == "lost"
    assert snapshot.calibration == "unknown"
    assert snapshot.hp_percent is None
    assert snapshot.rations_percent is None


def test_hidden_minimap_hides_untrustworthy_resource_values() -> None:
    sample = MonitoringSample(
        sequence=11,
        capture_ok=True,
        calibration_ok=True,
        hp_percent=0,
        rations_percent=0,
    )
    status = RuntimeStatus(
        running=True,
        sequence=11,
        navigation_state=NavigationState.UNKNOWN,
        result=MonitoringResult(11, (), NavigationState.UNKNOWN, minimap_visible=False),
        sample=sample,
    )

    snapshot = project_runtime_status(status)

    assert snapshot.minimap == "lost"
    assert snapshot.hp_percent is None
    assert snapshot.rations_percent is None


def test_visible_minimap_with_absent_ship_hud_projects_resource_absence() -> None:
    sample = MonitoringSample(
        sequence=12,
        capture_ok=True,
        calibration_ok=True,
        resource_bars_visible=False,
    )
    status = RuntimeStatus(
        running=True,
        sequence=12,
        navigation_state=NavigationState.NO_SHIP,
        result=MonitoringResult(12, (), NavigationState.NO_SHIP, minimap_visible=True),
        sample=sample,
    )

    snapshot = project_runtime_status(status)

    assert snapshot.minimap == "visible"
    assert snapshot.resources == "absent"
    assert snapshot.navigation == "no_ship"
    assert snapshot.hp_percent is None
    assert snapshot.rations_percent is None


def test_departed_state_projects_new_3d_detector_status() -> None:
    status = RuntimeStatus(
        running=True,
        sequence=13,
        navigation_state=NavigationState.DEPARTED,
        result=MonitoringResult(13, (), NavigationState.DEPARTED, minimap_visible=True),
    )

    snapshot = project_runtime_status(status)

    assert snapshot.navigation == "departed"
