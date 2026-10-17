from __future__ import annotations

from .modern_ui import DashboardSnapshot
from .navigation_state import NavigationState
from .runtime import RuntimeStatus


_NAVIGATION_VALUES = {
    NavigationState.UNKNOWN: "unknown",
    NavigationState.NO_SHIP: "no_ship",
    NavigationState.MOVING: "moving",
    NavigationState.STATIONARY_CANDIDATE: "stationary",
    NavigationState.ARRIVED: "arrived",
    NavigationState.DEPARTED: "departed",
    NavigationState.STUCK: "stuck",
    NavigationState.RECOVERING: "recovering",
}


def project_runtime_status(
    status: RuntimeStatus,
    *,
    window_title: str | None = None,
) -> DashboardSnapshot:
    if type(status) is not RuntimeStatus:
        raise ValueError("status must be RuntimeStatus")
    sample = status.sample
    if status.running:
        monitoring = "running"
    elif status.diagnostic and status.diagnostic not in ("runtime stopped",):
        monitoring = "error"
    else:
        monitoring = "stopped"
    if sample is not None:
        capture = "active" if sample.capture_ok else "lost"
        calibration = "valid" if sample.calibration_ok else "lost"
        hp_percent = sample.hp_percent
        rations_percent = sample.rations_percent
    else:
        capture = "lost" if "capture closed" in status.diagnostic else "unknown"
        calibration = "unknown"
        hp_percent = None
        rations_percent = None
    minimap_visible = status.result.minimap_visible if status.result is not None else None
    resources = "unknown"
    if sample is not None:
        if sample.resource_bars_visible is True:
            resources = "visible"
        elif sample.resource_bars_visible is False:
            resources = "absent"
    if minimap_visible is False:
        hp_percent = None
        rations_percent = None
        resources = "unknown"
    snapshot = DashboardSnapshot(
        monitoring=monitoring,
        capture=capture,
        calibration=calibration,
        minimap=(
            "visible" if minimap_visible is True else "lost" if minimap_visible is False else "unknown"
        ),
        navigation=_NAVIGATION_VALUES[status.navigation_state],
        hp_percent=hp_percent,
        rations_percent=rations_percent,
        resources=resources,
        window_title=window_title,
        diagnostic=status.diagnostic,
    )
    snapshot.validate()
    return snapshot
