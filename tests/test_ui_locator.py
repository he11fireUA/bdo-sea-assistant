from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.ui_locator import (
    CalibrationResult,
    ElementDetection,
    Rect,
    ShipBarsDetection,
    UILocatorError,
    build_layout,
    calibrate_frame,
    locate_minimap,
    locate_ship_bars,
    render_calibration_preview,
    save_layout,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_rect_round_trips_through_normalized_coordinates():
    rect = Rect(x=2458, y=1249, width=191, height=15)

    normalized = rect.normalized(frame_width=5106, frame_height=1440)
    restored = Rect.from_normalized(normalized, frame_width=5106, frame_height=1440)

    assert 0.0 <= normalized["x"] <= 1.0
    assert 0.0 <= normalized["y"] <= 1.0
    assert restored == rect


def test_locates_ship_hp_and_provisions_tracks_in_real_hud_crop():
    image = cv2.imread(str(FIXTURES / "ship_hud.png"))

    result = locate_ship_bars(image)

    assert result.confidence >= 0.75
    assert abs(result.hp.x - 178) <= 5
    assert abs(result.hp.y - 79) <= 6
    assert abs(result.hp.width - 191) <= 8
    assert abs(result.provisions.x - 178) <= 5
    assert abs(result.provisions.y - 96) <= 7
    assert abs(result.provisions.width - 191) <= 8


def test_ship_track_width_is_stable_across_real_iliya_frames():
    paths = sorted((FIXTURES / "ship_bars_iliya").glob("*.png"))
    assert len(paths) == 12

    detections = [locate_ship_bars(cv2.imread(str(path))) for path in paths]
    widths = [item.hp.width for item in detections]

    assert max(widths) - min(widths) <= 4
    assert all(abs(width - 191) <= 4 for width in widths)


def test_expected_ui_scale_stabilizes_historical_126_and_223_width_regressions() -> None:
    fixture_dir = FIXTURES / "ship_bars_regression"
    detections = [
        locate_ship_bars(cv2.imread(str(fixture_dir / name)), expected_scale=1.0)
        for name in ("width_126.png", "width_223.png")
    ]

    assert all(abs(item.hp.width - 191) <= 2 for item in detections)
    assert all(abs(item.hp.x - 58) <= 2 for item in detections)
    assert all(item.hp.width == item.provisions.width for item in detections)


def test_expected_ui_scale_rejects_equipment_menu_resource_pair() -> None:
    image = cv2.imread(str(FIXTURES / "ship_bars_regression" / "menu_false_pair.png"))

    with pytest.raises(UILocatorError):
        locate_ship_bars(image, expected_scale=1.0)


def test_ship_bar_locator_accepts_floating_point_maximum_calibration_scale() -> None:
    image = np.zeros((100, 230, 3), dtype=np.uint8)

    with pytest.raises(UILocatorError):
        locate_ship_bars(image, expected_scale=1.8000000000000012)


@pytest.mark.parametrize(
    "value",
    [True, "1.0", float("nan"), float("inf"), 0.59, 1.81],
)
def test_ship_bar_locator_rejects_invalid_expected_scale(value: object) -> None:
    image = np.zeros((100, 230, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="expected_scale"):
        locate_ship_bars(image, expected_scale=value)


def test_ship_bar_tracks_are_rejected_when_expected_width_exceeds_right_edge() -> None:
    image = np.zeros((120, 230, 3), dtype=np.uint8)
    cv2.rectangle(image, (169, 40), (220, 45), (0, 0, 230), -1)
    cv2.rectangle(image, (169, 57), (220, 62), (0, 210, 220), -1)

    with pytest.raises(UILocatorError, match="outside the frame"):
        locate_ship_bars(image, expected_scale=1.0)


@pytest.mark.parametrize("scale", [0.75, 1.30])
def test_locates_ship_bars_after_ui_scale_and_position_change(scale):
    hud = cv2.imread(str(FIXTURES / "ship_hud.png"))
    scaled = cv2.resize(hud, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((900, 1500, 3), dtype=np.uint8)
    offset_x, offset_y = 620, 310
    height, width = scaled.shape[:2]
    canvas[offset_y:offset_y + height, offset_x:offset_x + width] = scaled
    cv2.rectangle(canvas, (80, 100), (520, 112), (0, 0, 220), -1)
    cv2.rectangle(canvas, (900, 170), (1180, 180), (0, 210, 220), -1)

    result = locate_ship_bars(canvas)

    assert abs(result.hp.x - (offset_x + round(178 * scale))) <= 10
    assert abs(result.hp.y - (offset_y + round(79 * scale))) <= 10
    assert abs(result.hp.width - round(191 * scale)) <= 14
    assert abs(result.provisions.y - (offset_y + round(96 * scale))) <= 11


def test_locates_moved_and_scaled_minimap_from_stable_2d_anchor():
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    scale = 1.25
    scaled = cv2.resize(minimap, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((1000, 1800, 3), dtype=np.uint8)
    offset_x, offset_y = 910, 170
    height, width = scaled.shape[:2]
    canvas[offset_y:offset_y + height, offset_x:offset_x + width] = scaled
    cv2.circle(canvas, (250, 220), 35, (180, 180, 180), 3)

    result = locate_minimap(canvas, anchor)

    expected_x = offset_x + round(7 * scale)
    expected_y = offset_y + round(17 * scale)
    expected_width = round(300 * scale)
    expected_height = round(260 * scale)
    assert result.confidence >= 0.65
    assert abs(result.rect.x - expected_x) <= 12
    assert abs(result.rect.y - expected_y) <= 12
    assert abs(result.rect.width - expected_width) <= 18
    assert abs(result.rect.height - expected_height) <= 18


def test_layout_config_contains_absolute_and_normalized_regions(tmp_path):
    minimap = ElementDetection(Rect(4787, 62, 300, 260), confidence=0.91, scale=1.0)
    bars = ShipBarsDetection(
        hp=Rect(2458, 1249, 191, 15),
        provisions=Rect(2458, 1266, 191, 15),
        confidence=0.94,
    )

    layout = build_layout(5106, 1440, minimap, bars)
    path = tmp_path / "ui_layout.json"
    save_layout(path, layout)

    import json
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["capture_size"] == {"width": 5106, "height": 1440}
    assert saved["regions"]["minimap"]["absolute"]["x"] == 4787
    assert saved["regions"]["ship_hp_bar"]["normalized"]["width"] == pytest.approx(191 / 5106)
    assert saved["regions"]["ship_rations_bar"]["confidence"] == pytest.approx(0.94)
    assert saved["thresholds"] == {
        "ship_hp_warning_percent": 60,
        "ship_rations_warning_percent": 10,
    }


def test_calibrates_all_three_regions_on_real_full_frame():
    frame = cv2.imread(str(FIXTURES / "full_frame_arehaza.jpg"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    result = calibrate_frame(frame, anchor)

    assert result.minimap.confidence >= 0.65
    assert result.minimap.rect.x >= 4700
    assert 2350 <= result.bars.hp.x <= 2550
    assert result.bars.hp.width >= 175
    assert result.bars.provisions.y > result.bars.hp.y


def test_preview_draws_detected_regions_without_mutating_source():
    image = np.zeros((300, 600, 3), dtype=np.uint8)
    original = image.copy()
    result = CalibrationResult(
        minimap=ElementDetection(Rect(350, 20, 200, 180), 0.9, 1.0),
        bars=ShipBarsDetection(Rect(180, 240, 120, 10), Rect(180, 255, 120, 10), 0.92),
    )

    preview = render_calibration_preview(image, result)

    assert np.array_equal(image, original)
    assert preview.shape == image.shape
    assert np.count_nonzero(preview) > 0
