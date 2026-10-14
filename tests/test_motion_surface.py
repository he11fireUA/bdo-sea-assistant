from __future__ import annotations

import cv2
import numpy as np
import pytest

from bdo_assistant.motion_surface import (
    SurfaceReason,
    prepare_motion_surface,
    track_endpoints_are_valid,
)


def _textured_bgr(seed: int = 19, shape: tuple[int, int] = (120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(45, 205, shape, dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _route_overlay(image: np.ndarray, x: int) -> np.ndarray:
    result = image.copy()
    cv2.line(result, (x, 12), (x, result.shape[0] - 13), (0, 0, 255), 5)
    return result


def test_static_center_border_and_route_overlays_leave_no_accepted_features() -> None:
    previous = np.zeros((120, 160, 3), dtype=np.uint8)
    current = previous.copy()
    previous[:8, :] = (180, 180, 180)
    current[:8, :] = (180, 180, 180)
    cv2.circle(previous, (80, 60), 16, (255, 255, 255), -1)
    cv2.circle(current, (80, 60), 16, (255, 255, 255), -1)
    previous = _route_overlay(previous, 40)
    current = _route_overlay(current, 40)

    surface = prepare_motion_surface(previous, current, min_feature_count=1, min_tile_support=1)

    assert surface.reason is SurfaceReason.INSUFFICIENT_TEXTURE
    assert surface.accepted_feature_count == 0
    assert not surface.previous_feature_mask.any()
    assert not surface.current_feature_mask.any()


def test_union_of_moving_route_masks_prevents_mask_edge_from_becoming_geometry() -> None:
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    previous = _route_overlay(base, 52)
    current = _route_overlay(base, 108)

    surface = prepare_motion_surface(previous, current, min_feature_count=1, min_tile_support=1)

    assert surface.reason is SurfaceReason.INSUFFICIENT_TEXTURE
    assert surface.accepted_feature_count == 0
    assert not surface.previous_feature_mask.any()
    assert not surface.current_feature_mask.any()


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_flat_and_near_flat_crops_report_insufficient_texture(value: int) -> None:
    previous = np.full((120, 160, 3), value, dtype=np.uint8)
    current = previous.copy()

    surface = prepare_motion_surface(previous, current)

    assert surface.reason is SurfaceReason.INSUFFICIENT_TEXTURE
    assert not surface.observable
    assert surface.accepted_feature_count == 0


def test_track_endpoints_are_checked_in_their_respective_feature_masks() -> None:
    previous = _textured_bgr()
    current = _route_overlay(previous, 120)
    surface = prepare_motion_surface(previous, current, min_feature_count=1, min_tile_support=1)

    source = next(zip(*np.nonzero(surface.previous_feature_mask)))
    assert track_endpoints_are_valid(surface, (int(source[1]), int(source[0])), (120, 60)) is False


def test_colored_sea_distractors_do_not_eliminate_textured_surface() -> None:
    previous = _textured_bgr()
    current = previous.copy()
    # Saturated blue/cyan squares are deliberately not route-overlay colors.
    previous[20:45, 20:45] = (255, 190, 0)
    current[20:45, 20:45] = (255, 190, 0)

    surface = prepare_motion_surface(previous, current, min_feature_count=24, min_tile_support=3)

    assert surface.reason is SurfaceReason.OK
    assert surface.observable
    assert surface.accepted_feature_count >= 24
    assert surface.supported_tile_count >= 3
    assert surface.previous_valid_mask[25:40, 25:40].all()


@pytest.mark.parametrize(
    "previous,current",
    [
        (np.zeros((0, 8, 3), dtype=np.uint8), np.zeros((0, 8, 3), dtype=np.uint8)),
        (np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)),
        (np.zeros((8, 8, 3), dtype=np.float32), np.zeros((8, 8, 3), dtype=np.float32)),
        (np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((9, 8, 3), dtype=np.uint8)),
    ],
)
def test_invalid_crops_fail_closed(previous: np.ndarray, current: np.ndarray) -> None:
    with pytest.raises(ValueError, match="BGR|dimensions|at least"):
        prepare_motion_surface(previous, current)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"border_fraction": True},
        {"center_radius_fraction": float("nan")},
        {"tile_rows": 2.0},
        {"tile_columns": False},
        {"min_feature_count": 0},
        {"min_tile_support": 0},
        {"gradient_threshold": float("inf")},
    ],
)
def test_configuration_requires_finite_fractions_and_exact_integer_counts(kwargs: dict[str, object]) -> None:
    image = _textured_bgr()

    with pytest.raises(ValueError):
        prepare_motion_surface(image, image, **kwargs)
