import numpy as np
import pytest
import cv2

from bdo_assistant.calibration_flow import (
    CalibrationFlowController,
    CalibrationFlowState,
    build_default_calibration_flow,
)
from bdo_assistant.calibration_store import CalibrationStatus, JsonCalibrationStore
from bdo_assistant.ui_locator import CalibrationResult, ElementDetection, Rect, ShipBarsDetection


def _frame() -> np.ndarray:
    return np.full((100, 200, 3), 40, dtype=np.uint8)


def _anchor() -> np.ndarray:
    return np.full((16, 16, 3), 80, dtype=np.uint8)


def test_calibration_flow_requires_preview_before_explicit_save(tmp_path) -> None:
    saved = []
    result = object()
    preview = np.full((100, 200, 3), 90, dtype=np.uint8)
    layout = {"schema_version": 1}
    controller = CalibrationFlowController(
        detector=lambda frame, anchor: result,
        preview_renderer=lambda frame, detection: preview,
        layout_builder=lambda width, height, detection: layout,
        saver=lambda path, value: saved.append((path, value)),
    )

    controller.capture(_frame())
    assert controller.state is CalibrationFlowState.CAPTURED
    assert saved == []

    controller.detect(_anchor())
    assert controller.state is CalibrationFlowState.PREVIEW_READY
    assert np.array_equal(controller.preview, preview)
    assert saved == []

    destination = tmp_path / "ui_layout.json"
    controller.confirm_and_save(destination)

    assert controller.state is CalibrationFlowState.SAVED
    assert saved == [(destination, layout)]


def test_calibration_save_uses_private_snapshot_not_mutable_layout_views(tmp_path) -> None:
    saved = []
    built_layout = {"regions": {"minimap": {"absolute": {"x": 10}}}}
    controller = CalibrationFlowController(
        detector=lambda frame, anchor: object(),
        preview_renderer=lambda frame, detection: _frame(),
        layout_builder=lambda width, height, detection: built_layout,
        saver=lambda path, value: saved.append(value),
    )
    controller.capture(_frame())
    controller.detect(_anchor())

    built_layout["regions"]["minimap"]["absolute"]["x"] = 20
    exposed_layout = controller.layout
    exposed_layout["regions"]["minimap"]["absolute"]["x"] = 30
    controller.confirm_and_save(tmp_path / "layout.json")

    assert controller.layout["regions"]["minimap"]["absolute"]["x"] == 10
    assert saved[0]["regions"]["minimap"]["absolute"]["x"] == 10


def test_calibration_flow_never_saves_without_explicit_confirm(tmp_path) -> None:
    saved = []
    controller = CalibrationFlowController(
        detector=lambda frame, anchor: object(),
        preview_renderer=lambda frame, detection: _frame(),
        layout_builder=lambda width, height, detection: {"schema_version": 1},
        saver=lambda path, value: saved.append((path, value)),
    )

    with pytest.raises(RuntimeError, match="preview"):
        controller.confirm_and_save(tmp_path / "layout.json")
    controller.capture(_frame())
    with pytest.raises(RuntimeError, match="preview"):
        controller.confirm_and_save(tmp_path / "layout.json")

    assert saved == []


def test_calibration_retry_discards_unconfirmed_detection_and_preview() -> None:
    controller = CalibrationFlowController(
        detector=lambda frame, anchor: object(),
        preview_renderer=lambda frame, detection: _frame(),
        layout_builder=lambda width, height, detection: {"schema_version": 1},
        saver=lambda path, value: None,
    )
    controller.capture(_frame())
    controller.detect(_anchor())

    controller.retry()

    assert controller.state is CalibrationFlowState.IDLE
    assert controller.preview is None
    assert controller.layout is None


def test_calibration_capture_and_anchor_validate_exact_bgr_uint8() -> None:
    controller = CalibrationFlowController(
        detector=lambda frame, anchor: object(),
        preview_renderer=lambda frame, detection: _frame(),
        layout_builder=lambda width, height, detection: {"schema_version": 1},
        saver=lambda path, value: None,
    )

    for invalid in (
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ):
        with pytest.raises(ValueError, match="BGR uint8"):
            controller.capture(invalid)

    controller.capture(_frame())
    with pytest.raises(ValueError, match="BGR uint8"):
        controller.detect(np.zeros((8, 8), dtype=np.uint8))


def test_default_calibration_flow_runs_real_fixture_and_saves_valid_layout(tmp_path) -> None:
    frame = cv2.imread("tests/fixtures/full_frame_arehaza.jpg", cv2.IMREAD_COLOR)
    anchor = cv2.imread("assets/minimap/minimap_2d_anchor.png", cv2.IMREAD_COLOR)
    controller = build_default_calibration_flow()

    controller.capture(frame)
    controller.detect(anchor)
    destination = tmp_path / "ui_layout.json"
    controller.confirm_and_save(destination)

    assert controller.state is CalibrationFlowState.SAVED
    assert controller.preview.shape == frame.shape
    assert JsonCalibrationStore(destination).inspect().status is CalibrationStatus.VALID


def test_manual_region_correction_rebuilds_preview_without_saving() -> None:
    frame = cv2.imread("tests/fixtures/full_frame_arehaza.jpg", cv2.IMREAD_COLOR)
    anchor = cv2.imread("assets/minimap/minimap_2d_anchor.png", cv2.IMREAD_COLOR)
    controller = build_default_calibration_flow()
    controller.capture(frame)
    controller.detect(anchor)
    regions = {}
    for name in ("minimap", "ship_hp_bar", "ship_rations_bar"):
        absolute = controller.layout["regions"][name]["absolute"]
        regions[name] = Rect(
            absolute["x"] + 1,
            absolute["y"],
            absolute["width"],
            absolute["height"],
        )

    controller.apply_manual_regions(regions)

    assert controller.state is CalibrationFlowState.PREVIEW_READY
    for name, rect in regions.items():
        assert controller.layout["regions"][name]["absolute"]["x"] == rect.x


@pytest.mark.parametrize("invalid", [True, 1.0, float("nan"), float("inf")])
def test_manual_regions_reject_non_exact_integer_coordinates_before_render(invalid) -> None:
    render_calls = []
    detection = CalibrationResult(
        minimap=ElementDetection(Rect(10, 10, 20, 20), 0.9, 1.0),
        bars=ShipBarsDetection(Rect(40, 10, 30, 5), Rect(40, 20, 30, 5), 0.8),
    )
    controller = CalibrationFlowController(
        detector=lambda frame, anchor: detection,
        preview_renderer=lambda frame, result: render_calls.append(result) or frame.copy(),
        layout_builder=lambda width, height, result: {"schema_version": 1},
        saver=lambda path, value: None,
    )
    controller.capture(_frame())
    controller.detect(_anchor())
    regions = {
        "minimap": Rect(invalid, 10, 20, 20),
        "ship_hp_bar": Rect(40, 10, 30, 5),
        "ship_rations_bar": Rect(40, 20, 30, 5),
    }

    with pytest.raises(ValueError, match="exact integer"):
        controller.apply_manual_regions(regions)

    assert len(render_calls) == 1


@pytest.mark.parametrize(
    "invalid_rect",
    [Rect(-1, 10, 20, 20), Rect(10, 10, 0, 20), Rect(190, 10, 20, 20)],
)
def test_manual_regions_reject_rectangles_outside_captured_frame(invalid_rect) -> None:
    detection = CalibrationResult(
        minimap=ElementDetection(Rect(10, 10, 20, 20), 0.9, 1.0),
        bars=ShipBarsDetection(Rect(40, 10, 30, 5), Rect(40, 20, 30, 5), 0.8),
    )
    controller = CalibrationFlowController(
        detector=lambda frame, anchor: detection,
        preview_renderer=lambda frame, result: frame.copy(),
        layout_builder=lambda width, height, result: {"schema_version": 1},
        saver=lambda path, value: None,
    )
    controller.capture(_frame())
    controller.detect(_anchor())

    with pytest.raises(ValueError, match="leaves frame"):
        controller.apply_manual_regions(
            {
                "minimap": invalid_rect,
                "ship_hp_bar": Rect(40, 10, 30, 5),
                "ship_rations_bar": Rect(40, 20, 30, 5),
            }
        )


def test_manual_region_correction_rejects_preview_with_wrong_dimensions() -> None:
    frame = cv2.imread("tests/fixtures/full_frame_arehaza.jpg", cv2.IMREAD_COLOR)
    anchor = cv2.imread("assets/minimap/minimap_2d_anchor.png", cv2.IMREAD_COLOR)
    controller = build_default_calibration_flow()
    controller.capture(frame)
    controller.detect(anchor)
    original_detection = controller.detection
    original_preview = controller.preview
    regions = {
        name: Rect(**controller.layout["regions"][name]["absolute"])
        for name in ("minimap", "ship_hp_bar", "ship_rations_bar")
    }
    controller._preview_renderer = lambda captured, detection: np.zeros(
        (10, 10, 3), dtype=np.uint8
    )

    with pytest.raises(ValueError, match="dimensions"):
        controller.apply_manual_regions(regions)

    assert controller.detection is original_detection
    assert controller.preview is original_preview
