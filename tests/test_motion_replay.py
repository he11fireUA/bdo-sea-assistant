from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

from bdo_assistant.motion import (
    ActivityMotionTracker,
    MotionClass,
    classify_motion_window,
    estimate_activity,
    estimate_translation,
)
from bdo_assistant.navigation_state import NavigationState, NavigationStateMachine
from bdo_assistant.runtime import ConservativeNavigationAnalyzer


FIXTURES = Path(__file__).parent / "fixtures" / "replay"
LOW_RESOLUTION_THRESHOLDS = {
    "moving_net_displacement": 0.5,
    "moving_mean_step": 0.1,
    "moving_coherence": 0.3,
}


def _segment_vectors(name: str, start_sequence: int, end_sequence: int):
    fixture = FIXTURES / name
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    paths = {
        frame["source_sequence"]: fixture / frame["file"]
        for frame in manifest["frames"]
    }
    images = [
        cv2.imread(str(paths[sequence]), cv2.IMREAD_GRAYSCALE)
        for sequence in range(start_sequence, end_sequence + 1)
    ]
    return [estimate_translation(previous, current) for previous, current in zip(images, images[1:])]


def _classify_low_resolution(name: str, start_sequence: int, end_sequence: int) -> MotionClass:
    return classify_motion_window(
        _segment_vectors(name, start_sequence, end_sequence),
        **LOW_RESOLUTION_THRESHOLDS,
    )


def _activity_decisions(name: str, start_sequence: int, end_sequence: int):
    fixture = FIXTURES / name
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    paths = {
        frame["source_sequence"]: fixture / frame["file"]
        for frame in manifest["frames"]
    }
    tracker = ActivityMotionTracker()
    decisions = []
    for sequence in range(start_sequence, end_sequence + 1):
        previous = cv2.imread(str(paths[sequence - 1]), cv2.IMREAD_GRAYSCALE)
        current = cv2.imread(str(paths[sequence]), cv2.IMREAD_GRAYSCALE)
        decisions.append(tracker.update(sequence, estimate_activity(previous, current)))
    return decisions


def test_two_stage_docking_stays_stationary_after_arrival_latch() -> None:
    name = "iliya_two_stage_docking_latched_arrival"
    assert _classify_low_resolution(name, 285, 294) is MotionClass.MOVING
    assert _classify_low_resolution(name, 295, 307) is MotionClass.STATIONARY
    assert _classify_low_resolution(name, 308, 320) is MotionClass.STATIONARY


def test_combat_jitter_does_not_look_like_route_progress() -> None:
    name = "tinberra_stationary_under_npc_fire"
    result = _classify_low_resolution(name, 297, 321)
    assert result is not MotionClass.MOVING


def test_collision_attempts_do_not_clear_stuck_before_sustained_recovery() -> None:
    name = "player_ship_obstruction_stuck_recovery"
    assert _classify_low_resolution(name, 915, 955) is not MotionClass.MOVING
    assert _classify_low_resolution(name, 956, 1075) is not MotionClass.MOVING
    assert _classify_low_resolution(name, 1076, 1090) is MotionClass.MOVING


def test_false_pause_is_visually_stationary_and_cannot_prove_arrival() -> None:
    assert (
        _classify_low_resolution("dallae_false_pause_then_arrival", 661, 665)
        is MotionClass.STATIONARY
    )
    assert (
        _classify_low_resolution("iliya_false_pause_then_arrival", 147, 151)
        is MotionClass.STATIONARY
    )


def test_activity_channel_resets_false_pause_without_proving_movement() -> None:
    decisions = _activity_decisions("dallae_false_pause_then_arrival", 666, 669)
    assert decisions[-1].soft_activity_persistent
    assert not any(item.broad_activity_persistent for item in decisions)


def test_activity_channel_rejects_combat_spikes() -> None:
    decisions = _activity_decisions("tinberra_stationary_under_npc_fire", 298, 321)
    assert not any(item.broad_activity_persistent for item in decisions)


def test_activity_channel_persists_a_veto_and_backdates_onset() -> None:
    decisions = _activity_decisions("player_ship_obstruction_stuck_recovery", 1076, 1079)
    assert decisions[-1].broad_activity_persistent
    assert decisions[-1].onset_sequence == 1076
    assert decisions[-1].confirmed_sequence == 1079


def test_user_confirmed_sea_and_margoria_motion_replays_never_become_stuck() -> None:
    for name in ("ocean_straight_motion", "margoria_straight_motion"):
        fixture = FIXTURES / name
        manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["scope"] == "2d-minimap-inner-roi-only"
        assert manifest["privacy"] == "no chat, player names, credentials, or full frames"
        frames = []
        for item in manifest["frames"]:
            path = fixture / item["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert image is not None
            frames.append((item["source_sequence"], image))

        analyzer = ConservativeNavigationAnalyzer(route_evidence=lambda sequence, roi: (False, True))
        machine = NavigationStateMachine(stuck_frames=3)
        states = [machine.update(analyzer(sequence, image)) for sequence, image in frames]

        assert NavigationState.STUCK not in states
