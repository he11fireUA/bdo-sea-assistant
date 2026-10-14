from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bdo_assistant import native_motion_replay
from bdo_assistant.native_motion_replay import load_native_motion_replay


def _native_manifest(tmp_path: Path) -> Path:
    frames = []
    for source_sequence, elapsed_seconds in ((1, 0.0), (7, 0.25), (13, 1.0)):
        image_path = tmp_path / "frames" / f"frame_{source_sequence:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), np.full((4, 5, 3), source_sequence, dtype=np.uint8))
        frames.append({"file": image_path.relative_to(tmp_path).as_posix(), "source_sequence": source_sequence, "elapsed_seconds": elapsed_seconds, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 2, "status": "completed", "reason": "completed", "scope": "raw-local-minimap-roi", "sanitization": "unreviewed",
        "source_provenance": {"kind": "native-frame-source-callback", "sequence_semantics": "strictly-increasing-source-callback-order"},
        "capture_size": {"width": 5, "height": 4}, "minimap_rect": {"x": 0, "y": 0, "width": 5, "height": 4}, "interval_seconds": 0.2,
        "committed_frame_count": len(frames), "skipped_frame_count": 0, "dropped_frame_count": 0, "uncommitted_files": [], "frames": frames,
    }), encoding="utf-8")
    return manifest_path


def test_native_replay_preserves_provenance_and_uses_manifest_timeline(tmp_path: Path) -> None:
    replay = load_native_motion_replay(_native_manifest(tmp_path))

    assert [(item.envelope.source_sequence, item.envelope.processing_ordinal, item.envelope.received_monotonic) for item in replay] == [(1, 0, 0.0), (7, 1, 0.25), (13, 2, 1.0)]
    assert all(item.provenance["kind"] == "native-frame-source-callback" for item in replay)
    assert [item.roi.shape for item in replay] == [(4, 5, 3)] * 3


def test_native_replay_rejects_hash_mismatch_before_returning_any_frame(tmp_path: Path) -> None:
    manifest_path = _native_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_native_motion_replay(manifest_path)


@pytest.mark.parametrize("file", ("frames/missing.png", "../outside.png"))
def test_native_replay_rejects_missing_and_escape_paths(tmp_path: Path, file: str) -> None:
    manifest_path = _native_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["file"] = file
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative and contained"):
        load_native_motion_replay(manifest_path)


def test_native_replay_rejects_dimension_and_configured_frame_count_limits(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _native_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["minimap_rect"]["width"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions do not match"):
        load_native_motion_replay(manifest_path)

    manifest_path = _native_manifest(tmp_path / "count")
    monkeypatch.setattr(native_motion_replay, "MAX_FRAME_COUNT", 2)
    with pytest.raises(ValueError, match="frame count"):
        load_native_motion_replay(manifest_path)


@pytest.mark.parametrize("frame_name", ("../outside.png", "C:outside.png", "C:/outside.png"))
def test_native_replay_rejects_escape_and_windows_drive_relative_paths(tmp_path: Path, frame_name: str) -> None:
    manifest_path = _native_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["file"] = frame_name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative and contained"):
        load_native_motion_replay(manifest_path)


def test_native_replay_rejects_symlink_which_resolves_outside_fixture_root(tmp_path: Path) -> None:
    manifest_path = _native_manifest(tmp_path)
    outside = tmp_path.parent / "outside.png"
    assert cv2.imwrite(str(outside), np.zeros((4, 5, 3), dtype=np.uint8))
    link = tmp_path / "frames" / "linked.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Windows symlink creation is not permitted for this test account")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["file"] = "frames/linked.png"
    manifest["frames"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative and contained"):
        load_native_motion_replay(manifest_path)


@pytest.mark.parametrize("mutate, message", (
    (lambda manifest: manifest.update(schema_version=99), "unsupported native replay schema"),
    (lambda manifest: manifest["frames"][1].update(elapsed_seconds=0.0), "elapsed_seconds must strictly increase"),
    (lambda manifest: manifest.update(committed_frame_count=1), "committed_frame_count"),
))
def test_native_replay_rejects_schema_and_timeline_integrity_failures(tmp_path: Path, mutate, message: str) -> None:
    manifest_path = _native_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_native_motion_replay(manifest_path)


def test_legacy_compact_fixture_uses_timestamp_adapter_without_renumbering_source(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((3, 4, 3), dtype=np.uint8))
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "frames": [
        {"file": "frame.png", "source_sequence": 10, "local_time": "2026-01-01T00:00:01+00:00", "sha256": digest},
        {"file": "frame.png", "source_sequence": 20, "local_time": "2026-01-01T00:00:03+00:00", "sha256": digest},
    ]}), encoding="utf-8")

    replay = load_native_motion_replay(manifest_path)

    assert [(item.envelope.source_sequence, item.envelope.processing_ordinal, item.envelope.received_monotonic) for item in replay] == [(10, 0, 0.0), (20, 1, 2.0)]
    assert replay[0].provenance["kind"] == "legacy-compact-fixture"


def test_curated_legacy_replay_fixtures_remain_readable() -> None:
    root = Path(__file__).parent / "fixtures" / "replay"
    replays = [load_native_motion_replay(path / "manifest.json") for path in root.iterdir() if path.is_dir()]

    assert len(replays) == 7
    assert sum(len(replay) for replay in replays) == 326
    assert all(frame.provenance["kind"] == "legacy-compact-fixture" for replay in replays for frame in replay)