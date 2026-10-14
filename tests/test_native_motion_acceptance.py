from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from bdo_assistant.motion import MotionClass
from bdo_assistant.motion_replay_report import evaluate_ablations, evaluate_manifest
from bdo_assistant.navigation_state import NavigationObservation


def _native_manifest(tmp_path: Path) -> Path:
    frames = []
    for sequence, elapsed in ((4, 0.0), (8, 0.5), (15, 1.0)):
        image = tmp_path / f"frame_{sequence}.png"
        assert cv2.imwrite(str(image), np.full((4, 5, 3), sequence, dtype=np.uint8))
        frames.append({
            "file": image.name,
            "source_sequence": sequence,
            "elapsed_seconds": elapsed,
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 2,
        "status": "completed",
        "reason": "deadline reached",
        "scope": "raw-local-minimap-roi",
        "sanitization": "unreviewed",
        "source_provenance": {
            "kind": "native-frame-source-callback",
            "sequence_semantics": "strictly-increasing-source-callback-order",
        },
        "capture_size": {"width": 5, "height": 4},
        "minimap_rect": {"x": 0, "y": 0, "width": 5, "height": 4},
        "interval_seconds": 0.5,
        "committed_frame_count": 3,
        "skipped_frame_count": 9,
        "dropped_frame_count": 2,
        "uncommitted_files": [],
        "frames": frames,
    }), encoding="utf-8")
    return manifest


def test_unlabelled_native_manifest_is_read_only_and_explicitly_blocked(tmp_path: Path) -> None:
    manifest = _native_manifest(tmp_path)
    original = manifest.read_bytes()

    class Moving:
        def __call__(self, sequence, image):
            return NavigationObservation(MotionClass.MOVING, None, None)

    report = evaluate_manifest(manifest, analyzer_factory=Moving)

    assert manifest.read_bytes() == original
    assert report["totals"]["parsed_frames"] == 3
    assert report["capture_totals"] == {"committed_frames": 3, "skipped_frames": 9, "dropped_frames": 2}
    assert report["session_metrics"]["moving_coverage"] == {"denominator": 0, "matched": 0, "coverage": None}
    assert report["session_metrics"]["geometry_valid_support"] == "BLOCKED_FIELD"
    assert report["acceptance_matrix"]["availability"]["status"] == "BLOCKED_FIELD"
    assert report["acceptance_matrix"]["safety"]["status"] == "BLOCKED_FIELD"


def test_ablation_replays_identical_ordered_input_for_each_variant(tmp_path: Path) -> None:
    manifest = _native_manifest(tmp_path)
    calls: list[tuple[str, int]] = []

    def factory(name: str):
        class Analyzer:
            def __call__(self, sequence, image):
                calls.append((name, sequence))
                return NavigationObservation(MotionClass.UNKNOWN, None, None)
        return Analyzer

    report = evaluate_ablations(manifest, analyzer_factories={
        "legacy_ecc": factory("legacy_ecc"),
        "masked_adjacent_geometry": factory("masked_adjacent_geometry"),
    })

    assert report["totals"] == {"variants": 2, "parsed_frames_per_variant": 3}
    assert calls == [
        ("legacy_ecc", 4), ("legacy_ecc", 8), ("legacy_ecc", 15),
        ("masked_adjacent_geometry", 4), ("masked_adjacent_geometry", 8), ("masked_adjacent_geometry", 15),
    ]
    assert all(item["acceptance_matrix"]["availability"]["status"] == "BLOCKED_FIELD" for item in report["variants"].values())
