from __future__ import annotations

import cv2
import numpy as np
import pytest

from bdo_assistant.motion_surface import prepare_motion_surface
from bdo_assistant.texture_motion import (
    TextureMotionReason,
    estimate_texture_motion,
    estimate_texture_translation,
)


SHAPE = (144, 192)


def _texture(seed: int = 21) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(20, 236, SHAPE, dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _translated(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _estimate(previous: np.ndarray, current: np.ndarray, **kwargs: object):
    return estimate_texture_translation(prepare_motion_surface(previous, current), **kwargs)


def test_known_subpixel_translation_uses_previous_to_current_sign_convention() -> None:
    previous = _texture()
    evidence = _estimate(previous, _translated(previous, 2.4, -1.7), dt=0.25)

    assert evidence.reason is TextureMotionReason.OK
    assert evidence.valid
    assert evidence.dx == pytest.approx(2.4, abs=0.35)
    assert evidence.dy == pytest.approx(-1.7, abs=0.35)
    assert evidence.dt == pytest.approx(0.25)
    assert evidence.inlier_count <= evidence.track_count
    assert evidence.tile_coverage >= 3


def test_fixed_route_overlay_does_not_prevent_textured_translation() -> None:
    previous = _texture()
    current = _translated(previous, -3.0, 2.0)
    cv2.line(previous, (42, 15), (42, 129), (0, 0, 255), 5)
    cv2.line(current, (42, 15), (42, 129), (0, 0, 255), 5)

    evidence = _estimate(previous, current)

    assert evidence.reason is TextureMotionReason.OK
    assert evidence.dx == pytest.approx(-3.0, abs=0.4)
    assert evidence.dy == pytest.approx(2.0, abs=0.4)


def test_route_overlay_animation_alone_is_frozen_unknown_evidence() -> None:
    previous = np.zeros((*SHAPE, 3), dtype=np.uint8)
    current = previous.copy()
    cv2.line(previous, (50, 15), (50, 129), (0, 0, 255), 5)
    cv2.line(current, (140, 15), (140, 129), (0, 0, 255), 5)

    evidence = _estimate(previous, current)

    assert not evidence.valid
    assert evidence.reason is TextureMotionReason.INSUFFICIENT_TEXTURE
    assert evidence.dx == evidence.dy == evidence.residual == 0.0


def test_one_moving_icon_is_not_spatially_distributed_evidence() -> None:
    previous = np.zeros((*SHAPE, 3), dtype=np.uint8)
    noise = _texture(32)[:32, :32]
    previous[28:60, 25:57] = noise
    current = _translated(previous, 8, 0)
    surface = prepare_motion_surface(
        previous, current, min_feature_count=12, min_tile_support=1
    )

    evidence = estimate_texture_translation(surface, min_track_count=6, min_inlier_count=5, min_tile_coverage=3)

    assert not evidence.valid
    assert evidence.reason is TextureMotionReason.INSUFFICIENT_DISTRIBUTION


def test_outlier_majority_cannot_override_distributed_translation_consensus() -> None:
    previous = _texture()
    current = _translated(previous, 2.0, 1.0)
    # A large independently moving textured patch creates many conflicting tracks.
    patch = _texture(71)[20:124, 18:126]
    previous[20:124, 18:126] = patch
    current[20:124, 18:126] = _translated(patch, -9.0, 5.0)

    evidence = _estimate(previous, current)

    assert evidence.reason is TextureMotionReason.OK
    assert evidence.valid
    assert evidence.dx == pytest.approx(2.0, abs=0.4)
    assert evidence.dy == pytest.approx(1.0, abs=0.4)


def test_flat_image_is_frozen_insufficient_texture() -> None:
    image = np.full((*SHAPE, 3), 80, dtype=np.uint8)

    evidence = _estimate(image, image)

    assert not evidence.valid
    assert evidence.reason is TextureMotionReason.INSUFFICIENT_TEXTURE
    assert evidence.track_count == evidence.inlier_count == evidence.tile_coverage == 0


def test_pure_rotation_about_center_is_rejected_not_reported_as_translation() -> None:
    previous = _texture()
    matrix = cv2.getRotationMatrix2D((SHAPE[1] / 2, SHAPE[0] / 2), 7.0, 1.0)
    current = cv2.warpAffine(previous, matrix, (SHAPE[1], SHAPE[0]), borderMode=cv2.BORDER_REFLECT)

    evidence = _estimate(previous, current)

    assert not evidence.valid
    assert evidence.reason is TextureMotionReason.INCONSISTENT_MOTION


@pytest.mark.parametrize(
    "current",
    [
        pytest.param(lambda previous: cv2.resize(previous, (previous.shape[1] + 2, previous.shape[0] + 2)), id="resize"),
        pytest.param(lambda previous: np.zeros((8, 8, 3), dtype=np.uint8), id="unsupported_dimensions"),
    ],
)
def test_raw_crop_contract_freezes_resize_and_unsupported_dimensions(current) -> None:
    previous = _texture()

    evidence = estimate_texture_motion(previous, current(previous), dt=0.25)

    assert not evidence.valid
    assert evidence.reason is TextureMotionReason.INSUFFICIENT_TEXTURE
    assert evidence.dx == evidence.dy == evidence.residual == 0.0
    assert evidence.dt == pytest.approx(0.25)
    assert evidence.track_count == evidence.inlier_count == evidence.tile_coverage == 0
