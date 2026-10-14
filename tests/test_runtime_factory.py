import cv2
import numpy as np
import pytest

from bdo_assistant.calibration_store import CalibrationReadiness, CalibrationStatus
from bdo_assistant.runtime import RuntimeBridge, SimpleNavigationAnalyzer
from bdo_assistant.runtime_factory import (
    RuntimeFactoryInputs,
    build_runtime_bridge,
)
from bdo_assistant.runtime_readiness import load_verified_minimap_anchor
from bdo_assistant.window_discovery import WindowCandidate


class FakeSource:
    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self.started = False
        self.stopped = False

    def start(self, on_frame, on_closed) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeAdapter:
    name = "fake"

    def send(self, event) -> None:
        pass


def _layout() -> dict:
    return {
        "schema_version": 1,
        "capture_size": {"width": 1000, "height": 500},
        "regions": {
            "minimap": {
                "absolute": {"x": 700, "y": 20, "width": 250, "height": 200},
                "normalized": {"x": 0.7, "y": 0.04, "width": 0.25, "height": 0.4},
                "confidence": 0.91,
                "ui_scale": 1.0,
            },
            "ship_hp_bar": {
                "absolute": {"x": 50, "y": 400, "width": 190, "height": 8},
                "normalized": {"x": 0.05, "y": 0.8, "width": 0.19, "height": 0.016},
                "confidence": 0.88,
                "ui_scale": 1.0,
            },
            "ship_rations_bar": {
                "absolute": {"x": 50, "y": 420, "width": 190, "height": 8},
                "normalized": {"x": 0.05, "y": 0.84, "width": 0.19, "height": 0.016},
                "confidence": 0.87,
                "ui_scale": 1.0,
            },
        },
        "thresholds": {
            "ship_hp_warning_percent": 60,
            "ship_rations_warning_percent": 10,
        },
    }


def _inputs() -> RuntimeFactoryInputs:
    anchor = load_verified_minimap_anchor("assets/minimap")
    return RuntimeFactoryInputs(
        candidate=WindowCandidate(
            123,
            456,
            "BLACK DESERT",
            "BlackDesert64.exe",
            2560,
            1440,
            False,
        ),
        calibration=CalibrationReadiness(
            CalibrationStatus.VALID,
            "confirmed calibration loaded",
            layout=_layout(),
        ),
        anchor=anchor,
        adapters=(FakeAdapter(),),
        status_callback=lambda status: None,
        minimap_loss_delay_seconds=75,
        ship_hp_warning_percent=73,
        ship_rations_warning_percent=27,
    )


def test_runtime_factory_builds_complete_bridge_without_starting_capture() -> None:
    created = []

    def source_factory(hwnd: int):
        source = FakeSource(hwnd)
        created.append(source)
        return source

    bridge = build_runtime_bridge(_inputs(), source_factory=source_factory)

    assert type(bridge) is RuntimeBridge
    assert created[0].hwnd == 123
    assert not created[0].started
    assert not bridge.running
    assert bridge.loop.minimap.gate.visibility.lost_after_seconds == 75
    assert bridge.loop.hp_warning.warning_below == 73
    assert bridge.loop.rations_warning.warning_below == 27
    assert bridge._shadow is None


def test_runtime_factory_uses_simple_detector_with_five_second_monotonic_confirmation() -> None:
    bridge = build_runtime_bridge(_inputs(), source_factory=FakeSource)

    assert type(bridge.builder._navigation_analyzer) is SimpleNavigationAnalyzer
    assert bridge._analysis_cadence is not None
    assert bridge._analysis_cadence.interval_seconds == 0.2
    assert bridge._analysis_cadence.discontinuity_seconds == 2.0
    navigation = bridge.loop.minimap.gate.navigation
    assert navigation._simple is True
    assert navigation._confirmation_seconds == 5.0


def test_runtime_factory_requires_valid_calibration() -> None:
    valid = _inputs()
    with pytest.raises(ValueError, match="valid confirmed calibration"):
        build_runtime_bridge(
            RuntimeFactoryInputs(
                candidate=valid.candidate,
                calibration=CalibrationReadiness(
                    CalibrationStatus.MISSING,
                    "calibration file not found",
                ),
                anchor=valid.anchor,
                adapters=valid.adapters,
                status_callback=valid.status_callback,
            ),
            source_factory=FakeSource,
        )

def test_runtime_factory_rejects_mutable_or_malformed_anchor() -> None:
    valid = _inputs()
    mutable = np.array(valid.anchor, copy=True)
    assert mutable.flags.writeable

    with pytest.raises(ValueError, match="verified read-only anchor"):
        RuntimeFactoryInputs(
            candidate=valid.candidate,
            calibration=valid.calibration,
            anchor=mutable,
            adapters=valid.adapters,
            status_callback=valid.status_callback,
        )
