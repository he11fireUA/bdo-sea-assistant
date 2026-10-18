from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .calibration_store import CalibrationReadiness, CalibrationStatus
from .frame_timing import AnalysisCadence
from .minimap_observability import MinimapVisibilityTracker
from .monitoring import AlphaMonitoringLoop
from .navigation_observability import MinimapNavigationMonitor
from .navigation_state import NavigationStateMachine
from .notifications import NotificationAdapter, NotificationRouter
from .resource_bars import ThresholdWarningTracker
from .runtime import (
    CalibratedFrameSampleBuilder,
    SimpleNavigationAnalyzer,
    RuntimeBridge,
    RuntimeStatus,
    WgcWindowFrameSource,
)
from .window_discovery import WindowCandidate


@dataclass(frozen=True)
class RuntimeFactoryInputs:
    candidate: WindowCandidate
    calibration: CalibrationReadiness
    anchor: np.ndarray
    adapters: tuple[NotificationAdapter, ...]
    status_callback: Callable[[RuntimeStatus], None]
    minimap_loss_delay_seconds: int = 10
    ship_hp_warning_percent: int = 60
    ship_rations_warning_percent: int = 10
    repeat_alerts_enabled: bool = False
    repeat_alert_interval_seconds: int = 60


    def __post_init__(self) -> None:
        if type(self.candidate) is not WindowCandidate:
            raise ValueError("candidate must be WindowCandidate")
        if type(self.calibration) is not CalibrationReadiness:
            raise ValueError("calibration must be CalibrationReadiness")
        if (
            not isinstance(self.anchor, np.ndarray)
            or self.anchor.ndim != 3
            or self.anchor.shape[2] != 3
            or self.anchor.size == 0
            or self.anchor.dtype != np.uint8
            or self.anchor.flags.writeable
        ):
            raise ValueError("anchor must be a verified read-only anchor")

        if type(self.adapters) is not tuple:
            raise ValueError("adapters must be a tuple")
        if not callable(self.status_callback):
            raise ValueError("status_callback must be callable")

        if (
            type(self.minimap_loss_delay_seconds) is not int
            or not 10 <= self.minimap_loss_delay_seconds <= 300
        ):
            raise ValueError("minimap_loss_delay_seconds must be from 10 to 300")
        for value in (self.ship_hp_warning_percent, self.ship_rations_warning_percent):
            if type(value) is not int or not 1 <= value <= 99:
                raise ValueError("resource warning percentages must be from 1 to 99")
        if type(self.repeat_alerts_enabled) is not bool:
            raise ValueError("repeat_alerts_enabled must be boolean")
        if type(self.repeat_alert_interval_seconds) is not int or not 30 <= self.repeat_alert_interval_seconds <= 300:
            raise ValueError("repeat_alert_interval_seconds must be from 30 to 300")


def build_runtime_bridge(
    inputs: RuntimeFactoryInputs,
    *,
    source_factory: Callable[[int], object] = WgcWindowFrameSource,
) -> RuntimeBridge:
    if type(inputs) is not RuntimeFactoryInputs:
        raise ValueError("inputs must be RuntimeFactoryInputs")
    if inputs.calibration.status is not CalibrationStatus.VALID or inputs.calibration.layout is None:
        raise ValueError("runtime requires valid confirmed calibration")
    if not callable(source_factory):
        raise ValueError("source_factory must be callable")

    analyzer = SimpleNavigationAnalyzer()
    builder = CalibratedFrameSampleBuilder(inputs.calibration.layout, analyzer)
    try:
        expected_scale = inputs.calibration.layout["regions"]["minimap"]["ui_scale"]
    except (KeyError, TypeError) as exc:
        raise ValueError("calibration is missing minimap UI scale") from exc
    minimap = MinimapNavigationMonitor(
        anchor_template=inputs.anchor,
        expected_scale=expected_scale,
        visibility=MinimapVisibilityTracker(
            lost_after_seconds=inputs.minimap_loss_delay_seconds,
        ),
        navigation=NavigationStateMachine(
            simple=True,
            confirmation_seconds=3.0,
            confirmation_observations=21,
        ),
    )
    loop = AlphaMonitoringLoop(
        minimap=minimap,
        hp_warning=ThresholdWarningTracker(
            warning_below=inputs.ship_hp_warning_percent,
            reset_at=min(100, inputs.ship_hp_warning_percent + 5),
            confirmation_frames=3,
        ),
        rations_warning=ThresholdWarningTracker(
            warning_below=inputs.ship_rations_warning_percent,
            reset_at=min(100, inputs.ship_rations_warning_percent + 5),
            confirmation_frames=3,
        ),
        repeat_alerts_enabled=inputs.repeat_alerts_enabled,
        repeat_alert_interval_seconds=inputs.repeat_alert_interval_seconds,
    )
    router = NotificationRouter(inputs.adapters)
    source = source_factory(inputs.candidate.hwnd)
    return RuntimeBridge(
        source,
        builder,
        loop,
        router,
        inputs.status_callback,
        # Compare native WGC frames at a stable elapsed-time interval. Adjacent
        # display-rate frames contain too little displacement for sea travel.
        analysis_cadence=AnalysisCadence(
            interval_seconds=0.2,
            discontinuity_seconds=2.0,
        ),

    )
