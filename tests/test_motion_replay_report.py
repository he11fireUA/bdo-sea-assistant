from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.motion import MotionClass
from bdo_assistant.motion_replay_report import (
    evaluate_manifest,
    main,
    write_report,
)
from bdo_assistant.navigation_state import NavigationObservation


def _manifest(tmp_path: Path, labels: list[str | None]) -> Path:
    frames = []
    for sequence, label in enumerate(labels, 1):
        path = tmp_path / f"frame_{sequence:04d}.png"
        assert cv2.imwrite(str(path), np.full((8, 8, 3), sequence, dtype=np.uint8))
        frame = {
            "file": path.name,
            "source_sequence": sequence,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if label is not None:
            frame["expected_state"] = label
        frames.append(frame)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "frames": frames}), encoding="utf-8")
    return manifest


def test_unknown_frames_are_counted_and_do_not_claim_moving_recall(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["MOVING_CONFIRMED", "MOVING_CONFIRMED"])

    class AlwaysUnknown:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.UNKNOWN, None, None)

    report = evaluate_manifest(manifest, analyzer_factory=AlwaysUnknown)

    assert report["state_counts"] == {"unknown": 2}
    assert report["motion_counts"] == {"unknown": 2}
    assert report["labels"]["moving"]["denominator"] == 2
    assert report["labels"]["moving"]["recall"] == 0.0
    assert report["unknown_fraction"] == 1.0
    assert [frame["expected_state"] for frame in report["frames"]] == [
        "MOVING_CONFIRMED",
        "MOVING_CONFIRMED",
    ]


def test_always_unknown_fails_availability_with_parsed_totals(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["MOVING_CONFIRMED"] * 3)

    class AlwaysUnknown:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.UNKNOWN, None, None)

    report = evaluate_manifest(manifest, analyzer_factory=AlwaysUnknown)

    assert report["totals"] == {
        "parsed_frames": 3,
        "analyzed_frames": 3,
        "labeled_frames": 3,
        "unlabeled_frames": 0,
    }
    assert report["acceptance_matrix"]["availability"]["status"] == "FAIL"
    assert report["acceptance_matrix"]["availability"]["reason"] == "always_unknown_on_labeled_moving"


def test_labeled_true_arrival_event_is_not_counted_as_false_high_consequence(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["ARRIVED"] * 3)

    class Arrival:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.STATIONARY, True, False)

    report = evaluate_manifest(manifest, analyzer_factory=Arrival)

    assert report["events"]["terminal"] == [
        {"state": "arrived", "source_sequence": 3, "target": "arrival"}
    ]
    assert report["events"]["false_high_consequence"] == {"arrived": 0, "stuck": 0}


def test_false_arrived_during_confirmed_movement_is_counted_as_false_high_consequence(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["MOVING_CONFIRMED"] * 3)

    class FalseArrival:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.STATIONARY, True, False)

    report = evaluate_manifest(manifest, analyzer_factory=FalseArrival)

    assert report["false_arrived"] == 1
    assert report["events"]["false_high_consequence"] == {"arrived": 1, "stuck": 0}


def test_false_stuck_during_confirmed_movement_fails_safety(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["MOVING_CONFIRMED"] * 3)

    class FalseStuck:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.STATIONARY, False, True)

    report = evaluate_manifest(manifest, analyzer_factory=FalseStuck)

    assert report["false_stuck"] == 1
    assert report["events"]["false_high_consequence"] == {"arrived": 0, "stuck": 1}
    assert report["acceptance_matrix"]["safety"]["status"] == "FAIL"


def test_missing_native_ground_truth_is_blocked_not_pass_and_uncertainty_has_no_onset(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [None, "ARRIVAL_TRANSITION", None])

    class Unknown:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.UNKNOWN, None, None)

    report = evaluate_manifest(manifest, analyzer_factory=Unknown)

    assert report["totals"]["unlabeled_frames"] == 3
    assert report["ground_truth"]["missing_fields"] == ["expected_state", "scenario_id"]
    assert report["labels"]["arrival"]["latency"] == "BLOCKED_FIELD"
    assert report["acceptance_matrix"]["availability"]["status"] == "BLOCKED_FIELD"
    assert report["acceptance_matrix"]["hard_scenarios"]["status"] == "BLOCKED_FIELD"


def test_fixture_integrity_fails_before_detector_invocation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["MOVING_CONFIRMED"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frames"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    invoked = False

    class Analyzer:
        def __call__(self, sequence, image):
            nonlocal invoked
            invoked = True
            return NavigationObservation(MotionClass.UNKNOWN, None, None)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_manifest(manifest, analyzer_factory=Analyzer)

    assert not invoked


def test_ambiguous_labels_have_no_fabricated_latency_or_false_arrival(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["ARRIVAL_TRANSITION"] * 3)

    class Stationary:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.STATIONARY, True, False)

    report = evaluate_manifest(manifest, analyzer_factory=Stationary)

    assert report["labels"]["arrival"]["denominator"] == 0
    assert report["labels"]["arrival"]["latency"] == "BLOCKED_FIELD"
    assert report["false_arrived"] == 0


def test_report_output_refuses_to_overwrite(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    write_report(report_path, {"schema_version": 1})

    with pytest.raises(FileExistsError):
        write_report(report_path, {"schema_version": 1})


def test_cli_requires_new_explicit_output_path(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ["MOVING_CONFIRMED"])
    output = tmp_path / "report.json"

    assert main(["--manifest", str(manifest), "--analyzer", "legacy", "--output", str(output)]) == 0
    assert output.is_file()
    with pytest.raises(SystemExit, match="2"):
        main(["--manifest", str(manifest), "--analyzer", "legacy", "--output", str(output)])
