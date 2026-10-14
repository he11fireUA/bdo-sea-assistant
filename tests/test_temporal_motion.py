from __future__ import annotations

from bdo_assistant.motion import MotionClass
from bdo_assistant.temporal_motion import CandidateMotionBridge, MotionDecision, TemporalMotionTracker
from bdo_assistant.texture_motion import (
    TextureMotionEvidence,
    TextureMotionReason,
)


def _valid(dx: float, dy: float, dt: float = 0.2, residual: float = 0.02) -> TextureMotionEvidence:
    return TextureMotionEvidence(
        dx=dx,
        dy=dy,
        dt=dt,
        residual=residual,
        track_count=20,
        inlier_count=18,
        tile_coverage=5,
        valid=True,
        reason=TextureMotionReason.OK,
    )


def _invalid(dt: float) -> TextureMotionEvidence:
    return TextureMotionEvidence(
        dx=0.0, dy=0.0, dt=dt, residual=0.0, track_count=0, inlier_count=0,
        tile_coverage=0, valid=False, reason=TextureMotionReason.INSUFFICIENT_TEXTURE,
    )


def test_sustained_straight_texture_motion_is_confirmed_moving() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)

    decisions = [tracker.update(index * 0.2, _valid(0.24, 0.0)) for index in range(5)]

    assert decisions[-1].motion_class is MotionClass.MOVING
    assert decisions[-1].confirmed_at == 0.8
    assert decisions[-1].onset_at == 0.0


def test_sustained_curved_texture_motion_is_confirmed_moving() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)

    decisions = [
        tracker.update(index * 0.2, _valid(dx, dy))
        for index, (dx, dy) in enumerate(((0.24, 0.0), (0.20, 0.12), (0.12, 0.20), (0.0, 0.24), (-0.12, 0.20)))
    ]

    assert decisions[-1].motion_class is MotionClass.MOVING


def test_large_closed_loop_is_neither_stationary_nor_bounded_jitter() -> None:
    tracker = TemporalMotionTracker(window_duration=2.0, minimum_duration=0.6, bounded_diameter=0.5)

    decisions = [
        tracker.update(index * 0.2, _valid(dx, dy))
        for index, (dx, dy) in enumerate(((0.60, 0.0), (0.0, 0.60), (-0.60, 0.0), (0.0, -0.60), (0.60, 0.0)))
    ]

    assert decisions[-1].motion_class not in {MotionClass.STATIONARY, MotionClass.JITTER}


def test_small_returning_oscillation_is_bounded_jitter() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)

    decisions = [tracker.update(index * 0.2, _valid(dx, 0.0)) for index, dx in enumerate((0.10, -0.10, 0.10, -0.10, 0.10))]

    assert decisions[-1].motion_class is MotionClass.JITTER


def test_small_nonreturning_path_is_not_proven_bounded_jitter() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)

    decisions = [tracker.update(index * 0.2, _valid(0.08, 0.0)) for index in range(5)]

    assert decisions[-1].motion_class is MotionClass.UNKNOWN


def test_one_spike_among_verified_zeros_cannot_confirm_motion() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)

    decisions = [tracker.update(index * 0.2, _valid(dx, 0.0)) for index, dx in enumerate((0.50, 0.0, 0.0, 0.0, 0.0))]

    assert decisions[-1].motion_class is not MotionClass.MOVING


def test_unverified_texture_never_proves_stationarity() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)

    decisions = [tracker.update(index * 0.2, _invalid(0.2)) for index in range(5)]

    assert decisions[-1].motion_class is MotionClass.UNKNOWN


def test_long_invalid_gap_does_not_confirm_stationarity() -> None:
    tracker = TemporalMotionTracker(window_duration=5.0, minimum_duration=0.6)

    decisions = [
        tracker.update(0.0, _valid(0.01, 0.0)),
        tracker.update(0.2, _valid(0.01, 0.0)),
        tracker.update(3.0, _invalid(2.8)),
        tracker.update(3.2, _valid(0.01, 0.0)),
        tracker.update(3.4, _valid(0.01, 0.0)),
    ]

    assert decisions[-1].motion_class is MotionClass.UNKNOWN


def test_activity_is_only_a_veto_and_never_confirms_recovery() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)
    for index in range(5):
        tracker.update(index * 0.2, _valid(0.24, 0.0))

    decision = tracker.update(1.0, None, activity_veto=True)

    assert decision.motion_class is MotionClass.UNKNOWN
    assert decision.reason == "activity_veto"


def test_equivalent_physical_timeline_at_two_cadences_has_same_motion_class() -> None:
    fast = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)
    slow = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)

    fast_result = [fast.update(index * 0.1, _valid(0.12, 0.0, dt=0.1)) for index in range(10)][-1]
    slow_result = [slow.update(index * 0.2, _valid(0.24, 0.0, dt=0.2)) for index in range(5)][-1]

    assert fast_result.motion_class is slow_result.motion_class is MotionClass.MOVING


def test_invalid_texture_is_unknown_in_the_temporal_tracker() -> None:
    tracker = TemporalMotionTracker(window_duration=1.0, minimum_duration=0.6)
    for index in range(4):
        tracker.update(index * 0.2, _valid(0.24, 0.0))

    decision = tracker.update(0.8, _invalid(0.2))

    assert decision.motion_class is MotionClass.UNKNOWN
    assert decision.reason == "invalid_texture_evidence"


def _decision(motion_class: MotionClass, reason: str = "test") -> MotionDecision:
    return MotionDecision(motion_class, reason)


def test_candidate_bridge_only_projects_one_invalid_unknown_between_verified_motion() -> None:
    bridge = CandidateMotionBridge(maximum_gap_seconds=0.25)

    results = [
        bridge.update(0.0, _decision(MotionClass.MOVING, "sustained_verified_texture")),
        bridge.update(0.2, _decision(MotionClass.MOVING, "sustained_verified_texture")),
        bridge.update(0.4, _decision(MotionClass.UNKNOWN, "invalid_texture_evidence")),
        bridge.update(0.6, _decision(MotionClass.MOVING, "sustained_verified_texture")),
    ]

    assert results[-1].motion_class is MotionClass.MOVING
    assert results[-1].reason == "bridged_micro_unknown_between_verified_motion"


def test_candidate_bridge_keeps_prolonged_or_missing_unknown_evidence_unknown() -> None:
    bridge = CandidateMotionBridge(maximum_gap_seconds=0.25)

    results = [
        bridge.update(0.0, _decision(MotionClass.MOVING, "sustained_verified_texture")),
        bridge.update(0.2, _decision(MotionClass.MOVING, "sustained_verified_texture")),
        bridge.update(0.4, _decision(MotionClass.UNKNOWN, "missing_texture_evidence")),
        bridge.update(0.6, _decision(MotionClass.UNKNOWN, "missing_texture_evidence")),
        bridge.update(0.8, _decision(MotionClass.MOVING, "sustained_verified_texture")),
    ]

    assert results[-2].motion_class is MotionClass.UNKNOWN
    assert results[-2].reason == "missing_texture_evidence"


def test_candidate_bridge_rejects_an_invalid_gap_over_the_documented_bound() -> None:
    bridge = CandidateMotionBridge(maximum_gap_seconds=0.25)

    bridge.update(0.0, _decision(MotionClass.MOVING, "sustained_verified_texture"))
    bridge.update(0.2, _decision(MotionClass.MOVING, "sustained_verified_texture"))
    bridge.update(0.4, _decision(MotionClass.UNKNOWN, "invalid_texture_evidence"))
    result = bridge.update(0.8, _decision(MotionClass.MOVING, "sustained_verified_texture"))

    assert result.motion_class is MotionClass.UNKNOWN
    assert result.reason == "invalid_texture_evidence"
