from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant.replay_fixtures import (
    FixtureSpec,
    StateRange,
    extract_fixture,
    validate_fixture,
)


def _write_source_journey(path: Path) -> None:
    previews = path / "previews"
    previews.mkdir(parents=True)
    rows = []
    for sequence, color in [(10, (10, 20, 30)), (11, (40, 50, 60))]:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:60, 100:180] = color
        preview = previews / f"preview_{sequence:06d}.jpg"
        cv2.imwrite(str(preview), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        rows.append(
            {
                "sequence": sequence,
                "local_time": f"2026-08-02T00:00:{sequence}+03:00",
                "preview_file": str(preview),
            }
        )
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "local_time", "preview_file"])
        writer.writeheader()
        writer.writerows(rows)
    (path / "source_label.json").write_text('{"label":"synthetic"}\n', encoding="utf-8")


def test_extract_fixture_crops_only_roi_and_writes_traceable_manifest(tmp_path: Path) -> None:
    source = tmp_path / "journey_test"
    _write_source_journey(source)
    output = tmp_path / "fixture"
    spec = FixtureSpec(
        name="synthetic_transition",
        source_label="source_label.json",
        start_sequence=10,
        end_sequence=11,
        crop_normalized=(0.5, 0.2, 0.4, 0.4),
        states=(StateRange(10, 10, "MOVING"), StateRange(11, 11, "ARRIVED")),
    )

    manifest_path = extract_fixture(source, output, spec)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["name"] == "synthetic_transition"
    assert manifest["source"]["journey"] == "journey_test"
    assert manifest["source"]["label"] == "source_label.json"
    assert [frame["source_sequence"] for frame in manifest["frames"]] == [10, 11]
    assert [frame["expected_state"] for frame in manifest["frames"]] == ["MOVING", "ARRIVED"]

    first_path = output / manifest["frames"][0]["file"]
    first = cv2.imread(str(first_path))
    assert first.shape[:2] == (40, 80)
    assert np.mean(first, axis=(0, 1)) == pytest.approx((10, 20, 30), abs=3)
    assert manifest["frames"][0]["sha256"] == hashlib.sha256(first_path.read_bytes()).hexdigest()


def test_extract_fixture_rejects_unlabelled_sequences(tmp_path: Path) -> None:
    source = tmp_path / "journey_test"
    _write_source_journey(source)
    spec = FixtureSpec(
        name="missing_state",
        source_label="source_label.json",
        start_sequence=10,
        end_sequence=11,
        crop_normalized=(0.5, 0.2, 0.4, 0.4),
        states=(StateRange(10, 10, "MOVING"),),
    )

    with pytest.raises(ValueError, match="exactly one expected state"):
        extract_fixture(source, tmp_path / "fixture", spec)


def test_validate_fixture_detects_tampered_frame(tmp_path: Path) -> None:
    source = tmp_path / "journey_test"
    _write_source_journey(source)
    output = tmp_path / "fixture"
    spec = FixtureSpec(
        name="tamper_check",
        source_label="source_label.json",
        start_sequence=10,
        end_sequence=11,
        crop_normalized=(0.5, 0.2, 0.4, 0.4),
        states=(StateRange(10, 10, "MOVING"), StateRange(11, 11, "ARRIVED")),
    )
    manifest_path = extract_fixture(source, output, spec)
    assert validate_fixture(manifest_path) == 2

    (output / "frames" / "frame_0001.jpg").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_fixture(manifest_path)


def test_curated_replay_fixtures_are_compact_minimap_only_and_valid() -> None:
    root = Path(__file__).parent / "fixtures" / "replay"
    expected = {
        "dallae_false_pause_then_arrival",
        "iliya_false_pause_then_arrival",
        "tinberra_stationary_under_npc_fire",
        "player_ship_obstruction_stuck_recovery",
        "iliya_two_stage_docking_latched_arrival",
        "ocean_straight_motion",
        "margoria_straight_motion",
    }
    fixture_dirs = {path.name: path for path in root.iterdir() if path.is_dir()}
    assert fixture_dirs.keys() == expected

    total_frames = 0
    total_bytes = 0
    for path in fixture_dirs.values():
        manifest_path = path / "manifest.json"
        total_frames += validate_fixture(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert ":\\" not in manifest_path.read_text(encoding="utf-8")
        for frame in manifest["frames"]:
            image_path = path / frame["file"]
            image = cv2.imread(str(image_path))
            expected_shape = (80, 93) if path.name in {
                "ocean_straight_motion",
                "margoria_straight_motion",
            } else (100, 110)
            assert image.shape[:2] == expected_shape
            total_bytes += image_path.stat().st_size

    assert total_frames == 326
    assert total_bytes < 2 * 1024 * 1024
