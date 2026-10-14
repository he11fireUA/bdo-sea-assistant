from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.minimap_observability import (
    MinimapVisibilityEvidence,
    MinimapVisibilityState,
    MinimapVisibilityTracker,
    assess_minimap_visibility,
)
from bdo_assistant.notifications import EventType


FIXTURES = Path(__file__).parent / "fixtures"


def test_anchor_detector_accepts_real_visible_minimap_and_rejects_occlusion() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    visible = assess_minimap_visibility(minimap, anchor, expected_scale=1.0)
    occluded = assess_minimap_visibility(
        np.zeros_like(minimap),
        anchor,
        expected_scale=1.0,
    )

    assert visible.is_visible
    assert visible.confidence >= 0.60
    assert not occluded.is_visible
    assert occluded.confidence < 0.60


@pytest.mark.parametrize(
    "value",
    [
        True,
        "1.0",
        float("nan"),
        float("inf"),
        0.59,
        0.5999999995,
        1.8000000005,
        1.81,
    ],
)
def test_anchor_detector_rejects_invalid_expected_scale(value: object) -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    with pytest.raises(ValueError, match="expected_scale"):
        assess_minimap_visibility(minimap, anchor, expected_scale=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 0.49, 1.1])
def test_anchor_detector_rejects_invalid_confidence_threshold(value: float) -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    with pytest.raises(ValueError, match="minimum_confidence"):
        assess_minimap_visibility(
            minimap,
            anchor,
            expected_scale=1.0,
            minimum_confidence=value,
        )


@pytest.mark.parametrize("value", [0.5999999999999999, 1.8000000000000012])
def test_anchor_detector_accepts_machine_epsilon_at_scale_boundaries(value: float) -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    evidence = assess_minimap_visibility(minimap, anchor, expected_scale=value)

    assert 0 <= evidence.confidence <= 1


def test_anchor_detector_rejects_non_color_frames() -> None:
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    with pytest.raises(ValueError, match="BGR color"):
        assess_minimap_visibility(
            np.zeros((100, 100), dtype=np.uint8),
            anchor,
            expected_scale=1.0,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        "not-an-array",
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.full((20, 20, 3), np.nan, dtype=np.float32),
        np.zeros((20, 20, 3), dtype=np.int64),
    ],
)
@pytest.mark.parametrize("field", ["image", "anchor_template"])
def test_anchor_detector_rejects_unsupported_opencv_inputs(
    field: str,
    bad_value: object,
) -> None:
    values = {
        "image": cv2.imread(str(FIXTURES / "minimap_port.png")),
        "anchor_template": cv2.imread(str(FIXTURES / "minimap_2d_anchor.png")),
    }
    values[field] = bad_value

    with pytest.raises(ValueError, match="non-empty BGR.*uint8"):
        assess_minimap_visibility(**values, expected_scale=1.0)


def test_anchor_detector_rejects_roi_smaller_than_scaled_anchor() -> None:
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))

    with pytest.raises(ValueError, match="fit inside"):
        assess_minimap_visibility(
            np.zeros((20, 20, 3), dtype=np.uint8),
            anchor,
            expected_scale=1.0,
        )


def test_anchor_detector_rejects_degenerate_constant_template() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    constant = np.zeros((31, 31, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="texture"):
        assess_minimap_visibility(minimap, constant, expected_scale=1.0)


def test_anchor_detector_rejects_template_degenerate_after_scaling() -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    tiny_textured = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

    with pytest.raises(ValueError, match="scaled anchor"):
        assess_minimap_visibility(minimap, tiny_textured, expected_scale=0.60)


def test_anchor_detector_rejects_non_finite_match_confidence(monkeypatch) -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    monkeypatch.setattr(
        cv2,
        "minMaxLoc",
        lambda response: (0.0, float("nan"), (0, 0), (0, 0)),
    )

    with pytest.raises(ValueError, match="finite confidence"):
        assess_minimap_visibility(minimap, anchor, expected_scale=1.0)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_anchor_detector_rejects_out_of_range_match_confidence(
    monkeypatch,
    confidence: float,
) -> None:
    minimap = cv2.imread(str(FIXTURES / "minimap_port.png"))
    anchor = cv2.imread(str(FIXTURES / "minimap_2d_anchor.png"))
    monkeypatch.setattr(
        cv2,
        "minMaxLoc",
        lambda response: (0.0, confidence, (0, 0), (0, 0)),
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        assess_minimap_visibility(minimap, anchor, expected_scale=1.0)


@pytest.mark.parametrize(
    ("is_visible", "confidence"),
    [(1, 0.9), (True, float("nan")), (False, float("inf")), (True, -0.1), (True, 1.1)],
)
def test_visibility_evidence_rejects_invalid_values(
    is_visible: object,
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="visibility evidence"):
        MinimapVisibilityEvidence(is_visible=is_visible, confidence=confidence)


@pytest.mark.parametrize(
    ("is_visible", "confidence", "minimum_confidence"),
    [(True, 0.0, 0.50), (True, 0.59, 0.60), (False, 0.60, 0.60)],
)
def test_visibility_evidence_rejects_inconsistent_decision(
    is_visible: bool,
    confidence: float,
    minimum_confidence: float,
) -> None:
    with pytest.raises(ValueError, match="decision must match"):
        MinimapVisibilityEvidence(
            is_visible=is_visible,
            confidence=confidence,
            minimum_confidence=minimum_confidence,
        )


@pytest.mark.parametrize("value", [1, 0, None, "visible"])
def test_visibility_tracker_rejects_non_boolean_observations(value: object) -> None:
    tracker = MinimapVisibilityTracker()

    with pytest.raises(ValueError, match="boolean"):
        tracker.update(value)


def test_transient_minimap_miss_does_not_alert() -> None:
    tracker = MinimapVisibilityTracker(lost_frames=3, restored_frames=2)

    assert tracker.update(True) is None
    assert tracker.update(True) is None
    assert tracker.update(False) is None
    assert tracker.update(False) is None
    assert tracker.update(True) is None
    assert tracker.state is MinimapVisibilityState.VISIBLE


def test_sustained_loss_and_recovery_emit_one_transition_each() -> None:
    tracker = MinimapVisibilityTracker(lost_frames=3, restored_frames=2)

    assert [tracker.update(False) for _ in range(3)] == [
        None,
        None,
        EventType.MINIMAP_VISIBILITY_LOST,
    ]
    assert tracker.state is MinimapVisibilityState.LOST
    assert [tracker.update(False) for _ in range(5)] == [None] * 5

    assert tracker.update(True) is None
    assert tracker.update(True) is EventType.MINIMAP_VISIBILITY_RESTORED
    assert tracker.state is MinimapVisibilityState.VISIBLE


def test_failed_recovery_resets_confirmation_counter() -> None:
    tracker = MinimapVisibilityTracker(lost_frames=2, restored_frames=3)
    assert tracker.update(False) is None
    assert tracker.update(False) is EventType.MINIMAP_VISIBILITY_LOST

    assert tracker.update(True) is None
    assert tracker.update(True) is None
    assert tracker.update(False) is None
    assert tracker.update(True) is None
    assert tracker.update(True) is None
    assert tracker.update(True) is EventType.MINIMAP_VISIBILITY_RESTORED


def test_visibility_loss_uses_elapsed_seconds_when_configured() -> None:
    now = [100.0]
    tracker = MinimapVisibilityTracker(
        lost_after_seconds=10,
        clock=lambda: now[0],
        restored_frames=2,
    )
    assert tracker.update(True) is None
    assert tracker.update(True) is None

    assert tracker.update(False) is None
    now[0] = 109.99
    assert tracker.update(False) is None
    now[0] = 110.0

    assert tracker.update(False) is EventType.MINIMAP_VISIBILITY_LOST


def test_configure_loss_delay_preserves_visibility_but_restarts_pending_loss() -> None:
    now = [0.0]
    tracker = MinimapVisibilityTracker(lost_after_seconds=10, restored_frames=2, clock=lambda: now[0])
    tracker.update(True)
    tracker.update(True)
    tracker.update(False)
    now[0] = 9.0

    tracker.configure_loss_delay(20)

    assert tracker.state is MinimapVisibilityState.VISIBLE
    assert tracker.lost_after_seconds == 20
    assert tracker.update(False) is None
    now[0] = 20.0
    assert tracker.update(False) is None
    now[0] = 29.0
    assert tracker.update(False) is EventType.MINIMAP_VISIBILITY_LOST


@pytest.mark.parametrize("delay", [True, 0, -1, "20", float("nan"), float("inf")])
def test_invalid_live_loss_delay_preserves_pending_loss(delay) -> None:
    now = [0.0]
    tracker = MinimapVisibilityTracker(lost_after_seconds=10, clock=lambda: now[0])
    tracker.update(False)
    with pytest.raises(ValueError, match="lost_after_seconds"):
        tracker.configure_loss_delay(delay)
    assert tracker.lost_after_seconds == 10
    now[0] = 10.0
    assert tracker.update(False) is EventType.MINIMAP_VISIBILITY_LOST


def test_unchanged_loss_delay_keeps_pending_loss() -> None:
    now = [0.0]
    tracker = MinimapVisibilityTracker(lost_after_seconds=10, clock=lambda: now[0])
    tracker.update(False)
    now[0] = 9.0
    tracker.configure_loss_delay(10)
    now[0] = 10.0
    assert tracker.update(False) is EventType.MINIMAP_VISIBILITY_LOST


@pytest.mark.parametrize("already_lost", [False, True])
def test_live_loss_delay_preserves_state_and_partial_recovery(already_lost) -> None:
    tracker = MinimapVisibilityTracker(lost_frames=2, restored_frames=2)
    if already_lost:
        tracker.update(False)
        tracker.update(False)
    state = tracker.state
    tracker.update(True)
    tracker.configure_loss_delay(20)
    assert tracker.state is state
    result = tracker.update(True)
    assert tracker.state is MinimapVisibilityState.VISIBLE
    assert result is (EventType.MINIMAP_VISIBILITY_RESTORED if already_lost else None)


def test_startup_disables_navigation_until_visibility_is_confirmed() -> None:
    tracker = MinimapVisibilityTracker(lost_frames=3, restored_frames=2)

    assert tracker.state is MinimapVisibilityState.UNKNOWN
    assert not tracker.navigation_enabled
    assert tracker.update(True) is None
    assert not tracker.navigation_enabled
    assert tracker.update(True) is None
    assert tracker.state is MinimapVisibilityState.VISIBLE
    assert tracker.navigation_enabled


@pytest.mark.parametrize("value", [0, 1, -1, 1.5, float("nan"), True])
@pytest.mark.parametrize("field", ["lost_frames", "restored_frames"])
def test_visibility_tracker_rejects_invalid_frame_thresholds(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        MinimapVisibilityTracker(**{field: value})
