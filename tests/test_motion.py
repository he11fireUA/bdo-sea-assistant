from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.motion import (
    ActivityEvidence,
    ActivityMotionTracker,
    MotionClass,
    MotionVector,
    classify_motion_window,
    estimate_activity,
    estimate_translation,
)


def _texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (100, 110), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_estimate_translation_recovers_known_subpixel_shift() -> None:
    previous = _texture()
    transform = np.float32([[1, 0, 1.75], [0, 1, -0.80]])
    current = cv2.warpAffine(previous, transform, (110, 100), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    estimate = estimate_translation(previous, current)

    assert estimate.dx == pytest.approx(1.75, abs=0.25)
    assert estimate.dy == pytest.approx(-0.80, abs=0.25)
    assert estimate.confidence > 0.75


def test_estimate_translation_rejects_non_finite_ecc_confidence(monkeypatch) -> None:
    monkeypatch.setattr(
        cv2,
        "findTransformECC",
        lambda *args, **kwargs: (float("nan"), np.eye(2, 3, dtype=np.float32)),
    )

    estimate = estimate_translation(_texture(), _texture())

    assert estimate.confidence == 0.0


def test_estimate_activity_ignores_border_and_central_marker() -> None:
    previous = np.zeros((100, 110), dtype=np.uint8)
    current = previous.copy()
    current[:5, :] = 255
    cv2.circle(current, (55, 50), 10, 255, -1)

    ignored = estimate_activity(previous, current)
    assert ignored.mae == pytest.approx(0.0)
    assert ignored.changed_fraction == pytest.approx(0.0)

    current[15:35, 15:35] = 20
    visible = estimate_activity(previous, current)
    assert visible.mae > 0.5
    assert 0.03 < visible.changed_fraction < 0.06


def test_motion_window_requires_coherent_net_displacement() -> None:
    coherent = [MotionVector(1.0, 0.1, 0.9) for _ in range(6)]
    bounded_jitter = [
        MotionVector(1.0, 0.0, 0.9),
        MotionVector(-1.0, 0.0, 0.9),
        MotionVector(0.0, 1.0, 0.9),
        MotionVector(0.0, -1.0, 0.9),
    ] * 2

    assert classify_motion_window(coherent) is MotionClass.MOVING
    assert classify_motion_window(bounded_jitter) is MotionClass.JITTER


def test_motion_window_distinguishes_stationary_and_unknown() -> None:
    stationary = [MotionVector(0.01, -0.01, 0.99) for _ in range(8)]
    low_confidence = [MotionVector(3.0, 0.0, 0.1) for _ in range(8)]

    assert classify_motion_window(stationary) is MotionClass.STATIONARY
    assert classify_motion_window(low_confidence) is MotionClass.UNKNOWN


def test_persistent_broad_activity_backdates_veto_onset() -> None:
    tracker = ActivityMotionTracker(window=4, required_broad_frames=3)

    decisions = [
        tracker.update(1076, ActivityEvidence(mae=2.58, changed_fraction=0.029)),
        tracker.update(1077, ActivityEvidence(mae=4.11, changed_fraction=0.061)),
        tracker.update(1078, ActivityEvidence(mae=7.66, changed_fraction=0.190)),
        tracker.update(1079, ActivityEvidence(mae=6.63, changed_fraction=0.136)),
    ]

    assert not any(item.broad_activity_persistent for item in decisions[:3])
    assert decisions[-1].broad_activity_persistent
    assert decisions[-1].onset_sequence == 1076
    assert decisions[-1].confirmed_sequence == 1079


def test_soft_activity_resets_stationarity_without_proving_movement() -> None:
    tracker = ActivityMotionTracker(window=4, required_soft_frames=3)
    decisions = [
        tracker.update(sequence, ActivityEvidence(mae=mae, changed_fraction=0.01))
        for sequence, mae in zip(range(666, 670), (0.37, 0.74, 0.60, 0.52))
    ]

    assert decisions[-1].soft_activity_persistent
    assert not decisions[-1].broad_activity_persistent


def test_isolated_broad_spike_never_becomes_persistent_veto() -> None:
    tracker = ActivityMotionTracker(window=4, required_broad_frames=3)
    samples = (
        ActivityEvidence(0.18, 0.01),
        ActivityEvidence(3.51, 0.062),
        ActivityEvidence(0.20, 0.01),
        ActivityEvidence(0.19, 0.01),
    )

    decisions = [tracker.update(index, sample) for index, sample in enumerate(samples)]

    assert not any(item.broad_activity_persistent for item in decisions)


def test_activity_tracker_rejects_soft_threshold_above_broad_threshold() -> None:
    with pytest.raises(ValueError, match="soft_mae"):
        ActivityMotionTracker(soft_mae=4.0, broad_mae=3.0)


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), 0, -1, 1, 2])
def test_activity_tracker_requires_integer_window_of_at_least_three(value: object) -> None:
    with pytest.raises(ValueError, match="window.*integer.*3"):
        ActivityMotionTracker(window=value)


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), 0, -1, 1, 2, 5])
@pytest.mark.parametrize("field", ["required_soft_frames", "required_broad_frames"])
def test_activity_tracker_requires_multi_frame_integer_confirmation(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"{field}.*integer.*3.*window"):
        ActivityMotionTracker(window=4, **{field: value})


def test_activity_tracker_rejects_non_finite_thresholds() -> None:
    with pytest.raises(ValueError, match="finite"):
        ActivityMotionTracker(soft_mae=float("nan"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"soft_mae": -0.1},
        {"broad_mae": -0.1},
        {"broad_changed_fraction": -0.1},
        {"broad_changed_fraction": 1.1},
    ],
)
def test_activity_tracker_rejects_out_of_range_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="range"):
        ActivityMotionTracker(**kwargs)


def test_activity_tracker_rejects_non_finite_evidence() -> None:
    tracker = ActivityMotionTracker()

    with pytest.raises(ValueError, match="evidence.*finite"):
        tracker.update(1, ActivityEvidence(float("inf"), 0.1))


@pytest.mark.parametrize(
    "evidence",
    [
        ActivityEvidence(-0.1, 0.1),
        ActivityEvidence(1.0, -0.1),
        ActivityEvidence(1.0, 1.1),
    ],
)
def test_activity_tracker_rejects_out_of_range_evidence(evidence: ActivityEvidence) -> None:
    with pytest.raises(ValueError, match="evidence.*range"):
        ActivityMotionTracker().update(1, evidence)


def test_motion_window_rejects_single_valid_vector_among_low_confidence_samples() -> None:
    vectors = [
        MotionVector(2.1, 0.0, 0.95),
        MotionVector(0.0, 0.0, 0.10),
        MotionVector(0.0, 0.0, 0.10),
        MotionVector(0.0, 0.0, 0.10),
        MotionVector(0.0, 0.0, 0.10),
    ]

    assert classify_motion_window(vectors) is MotionClass.UNKNOWN


def test_motion_window_rejects_one_moving_outlier_among_valid_stationary_samples() -> None:
    vectors = [
        MotionVector(2.1, 0.0, 0.95),
        MotionVector(0.0, 0.0, 0.95),
        MotionVector(0.0, 0.0, 0.95),
    ]

    assert classify_motion_window(vectors) is not MotionClass.MOVING


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_valid_samples": 1},
        {"minimum_valid_samples": 0},
        {"minimum_valid_samples": 3.5},
        {"minimum_valid_samples": float("nan")},
        {"minimum_valid_fraction": 0.2},
        {"minimum_valid_fraction": 0.0},
    ],
)
def test_motion_window_rejects_unsafe_evidence_guards(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="evidence guard"):
        classify_motion_window([], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_confidence": float("nan")},
        {"stationary_mean_step": float("inf")},
        {"moving_net_displacement": float("nan")},
        {"jitter_maximum_coherence": float("-inf")},
    ],
)
def test_motion_window_rejects_non_finite_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        classify_motion_window([], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_confidence": -0.1},
        {"minimum_confidence": 1.1},
        {"stationary_mean_step": -0.1},
        {"stationary_net_displacement": -0.1},
        {"moving_net_displacement": -0.1},
        {"moving_mean_step": -0.1},
        {"moving_coherence": -0.1},
        {"moving_coherence": 1.1},
        {"jitter_minimum_path": -0.1},
        {"jitter_maximum_coherence": -0.1},
        {"jitter_maximum_coherence": 1.1},
    ],
)
def test_motion_window_rejects_out_of_range_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="range"):
        classify_motion_window([], **kwargs)


def test_motion_window_ignores_non_finite_vectors_as_invalid_samples() -> None:
    vectors = [
        MotionVector(1.0, 0.0, 0.9),
        MotionVector(1.0, 0.0, 0.9),
        MotionVector(1.0, 0.0, 0.9),
        MotionVector(float("nan"), 0.0, 0.9),
        MotionVector(100.0, 0.0, float("inf")),
    ]

    assert classify_motion_window(vectors) is MotionClass.MOVING
