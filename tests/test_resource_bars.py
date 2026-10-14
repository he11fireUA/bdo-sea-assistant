from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.resource_bars import (
    ResourceKind,
    ThresholdWarningTracker,
    estimate_fill_percent,
    measure_resource_bar,
)
from bdo_assistant.ui_locator import Rect, locate_ship_bars


FIXTURES = Path(__file__).parent / "fixtures" / "ship_bars_iliya"


def test_estimate_fill_percent_on_synthetic_red_track() -> None:
    image = np.zeros((20, 120, 3), dtype=np.uint8)
    image[4:16, 5:105] = (25, 25, 25)
    image[6:14, 7:67] = (0, 0, 230)

    percent = estimate_fill_percent(image, Rect(5, 4, 100, 12), ResourceKind.HP)

    assert percent == pytest.approx(60.0, abs=2.0)


@pytest.mark.parametrize(
    ("kind", "color"),
    [
        (ResourceKind.HP, (0, 0, 230)),
        (ResourceKind.RATIONS, (0, 210, 230)),
    ],
)
def test_fully_colored_dark_framed_ship_bar_is_observable_without_pale_rail(
    kind: ResourceKind, color: tuple[int, int, int]
) -> None:
    track = Rect(5, 4, 100, 12)
    image = np.zeros((20, 120, 3), dtype=np.uint8)
    image[4:16, 5:105] = color

    reading = measure_resource_bar(image, track, kind)

    assert reading.observable is True
    assert reading.percent == 100.0
    assert reading.support >= 0.97


def test_resource_reading_distinguishes_empty_visible_track_from_non_ship_scene() -> None:
    track = Rect(5, 4, 100, 12)
    on_ship = np.zeros((20, 120, 3), dtype=np.uint8)
    on_ship[3, 5:105] = (120, 120, 120)  # ship-HUD pale top rail
    on_ship[4:16, 5:105] = (25, 25, 25)
    non_ship_scene = np.full((20, 120, 3), (130, 80, 25), dtype=np.uint8)

    empty_track = measure_resource_bar(on_ship, track, ResourceKind.HP)
    absent_track = measure_resource_bar(non_ship_scene, track, ResourceKind.HP)

    assert empty_track.observable is True
    assert empty_track.percent == 0.0
    assert absent_track.observable is False
    assert absent_track.percent is None


@pytest.mark.parametrize("gray", (25, 80))
def test_resource_reading_rejects_uniform_neutral_scene_as_missing_track(gray: int) -> None:
    scene = np.full((20, 120, 3), gray, dtype=np.uint8)

    reading = measure_resource_bar(scene, Rect(5, 4, 100, 12), ResourceKind.HP)

    assert reading.observable is False
    assert reading.percent is None


@pytest.mark.parametrize(
    "track",
    (
        Rect(-1, 4, 100, 12),
        Rect(5, -1, 100, 12),
        Rect(25, 4, 100, 12),
        Rect(5, 12, 100, 12),
    ),
)
def test_resource_reading_rejects_partially_out_of_frame_calibrated_track(track: Rect) -> None:
    image = np.zeros((20, 120, 3), dtype=np.uint8)
    image[4:16, 5:105] = (25, 25, 25)

    reading = measure_resource_bar(image, track, ResourceKind.HP)

    assert reading.observable is False
    assert reading.percent is None


@pytest.mark.parametrize(
    ("kind", "color"),
    [
        (ResourceKind.HP, (0, 0, 230)),
        (ResourceKind.RATIONS, (0, 210, 230)),
    ],
)
def test_full_resource_track_normalizes_border_inset_to_100_percent(kind, color) -> None:
    image = np.zeros((20, 120, 3), dtype=np.uint8)
    image[4:16, 5:105] = (25, 25, 25)
    image[6:14, 7:105] = color

    percent = estimate_fill_percent(image, Rect(5, 4, 100, 12), kind)

    assert percent == 100.0


def test_real_iliya_resource_estimates_are_stable_across_twelve_frames() -> None:
    hp_values = []
    ration_values = []
    for path in sorted(FIXTURES.glob("*.png")):
        image = cv2.imread(str(path))
        bars = locate_ship_bars(image)
        hp_values.append(estimate_fill_percent(image, bars.hp, ResourceKind.HP))
        ration_values.append(estimate_fill_percent(image, bars.provisions, ResourceKind.RATIONS))

    assert len(hp_values) == 12
    assert np.ptp(hp_values) <= 3.0
    assert np.ptp(ration_values) <= 3.0
    assert 75 <= np.median(hp_values) <= 85
    assert 55 <= np.median(ration_values) <= 65


def test_real_iliya_ship_hud_tracks_are_observable_across_twelve_frames() -> None:
    for path in sorted(FIXTURES.glob("*.png")):
        image = cv2.imread(str(path))
        bars = locate_ship_bars(image)

        assert measure_resource_bar(image, bars.hp, ResourceKind.HP).observable
        assert measure_resource_bar(image, bars.provisions, ResourceKind.RATIONS).observable


@pytest.mark.parametrize(
    ("relative_path", "hp_expected", "rations_expected"),
    [
        ("ship_hud.png", 81.15, 53.40),
        ("full_frame_arehaza.jpg", 81.15, 50.79),
        ("ship_bars_regression/width_126.png", 86.91, 65.45),
        ("ship_bars_regression/width_223.png", 86.39, 53.40),
    ],
)
def test_real_resource_fill_pixel_goldens(
    relative_path: str,
    hp_expected: float,
    rations_expected: float,
) -> None:
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / relative_path))
    bars = locate_ship_bars(image, expected_scale=1.0)

    assert estimate_fill_percent(image, bars.hp, ResourceKind.HP) == pytest.approx(
        hp_expected, abs=1.5
    )
    assert estimate_fill_percent(
        image, bars.provisions, ResourceKind.RATIONS
    ) == pytest.approx(rations_expected, abs=1.5)


def test_warning_tracker_confirms_once_resets_and_rearms_after_refill() -> None:
    tracker = ThresholdWarningTracker(warning_below=60, reset_at=65, confirmation_frames=3)

    assert [tracker.update(value) for value in (59, 58, 57)] == [False, False, True]
    assert [tracker.update(value) for value in (56, 55, 54)] == [False, False, False]
    assert tracker.update(80) is False
    assert [tracker.update(value) for value in (59, 58, 57)] == [False, False, True]


def test_warning_zone_includes_the_selected_percent() -> None:
    tracker = ThresholdWarningTracker(warning_below=73, reset_at=78, confirmation_frames=3)

    assert [tracker.update(73) for _ in range(3)] == [False, False, True]


def test_rations_threshold_does_not_rearm_before_hysteresis_reset() -> None:
    tracker = ThresholdWarningTracker(warning_below=10, reset_at=15, confirmation_frames=3)
    for value in (9, 8, 7):
        fired = tracker.update(value)
    assert fired is True

    assert tracker.update(14) is False
    assert [tracker.update(value) for value in (9, 8, 7)] == [False, False, False]
    assert tracker.update(15) is False
    assert [tracker.update(value) for value in (9, 8, 7)] == [False, False, True]


@pytest.mark.parametrize(
    ("warning_below", "reset_at"),
    [
        (float("nan"), 65),
        (60, float("inf")),
        (-0.1, 65),
        (60, 100.1),
        (60, 60),
        (65, 60),
    ],
)
def test_warning_tracker_rejects_invalid_resource_thresholds(
    warning_below: float,
    reset_at: float,
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ThresholdWarningTracker(warning_below=warning_below, reset_at=reset_at)


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan"), True])
def test_warning_tracker_rejects_invalid_confirmation_count(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ThresholdWarningTracker(
            warning_below=60,
            reset_at=65,
            confirmation_frames=value,
        )
