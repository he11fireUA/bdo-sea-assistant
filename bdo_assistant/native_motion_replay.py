"""Read-only, fail-closed loader for native minimap replay manifests.

The loader keeps capture provenance (`source_sequence`) separate from the
zero-based ordinal used to feed offline processing.  Native replay time comes
from the recorder's `elapsed_seconds`; compact schema-v1 fixtures are adapted
with their ISO timestamps when present, or documented synthetic ordinal time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator, Mapping

import cv2
import numpy as np

from .frame_timing import FrameEnvelope, synthetic_received_monotonic

MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_FRAME_COUNT = 10_000
MAX_FRAME_DIMENSION = 4_096
MAX_TOTAL_FRAME_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class NativeMotionReplayFrame:
    """One decoded minimap ROI plus metadata safe for offline processing."""

    envelope: FrameEnvelope
    roi: np.ndarray
    provenance: Mapping[str, object]
    expected_state: str | None = None


def _exact_non_negative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_non_negative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return float(value)


def _required_object(manifest: dict, key: str) -> dict:
    value = manifest.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _contained_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("fixture frame path must be a non-empty relative path")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts or ".." in windows.parts:
        raise ValueError("fixture frame path must be relative and contained")
    candidate = root / Path(*posix.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise ValueError("fixture frame path must be relative and contained") from None
    if not resolved.is_file():
        raise ValueError(f"fixture frame is missing: {value}")
    return resolved


def _verify_hash(path: Path, expected: object) -> int:
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return len(payload)


def _decode_roi(path: Path, *, expected_size: tuple[int, int] | None) -> np.ndarray:
    roi = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if roi is None or roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError(f"could not decode fixture frame: {path.name}")
    height, width = roi.shape[:2]
    if width <= 0 or height <= 0 or width > MAX_FRAME_DIMENSION or height > MAX_FRAME_DIMENSION:
        raise ValueError("fixture frame dimensions exceed limits")
    if expected_size is not None and (width, height) != expected_size:
        raise ValueError("fixture frame dimensions do not match minimap_rect")
    return roi


def _legacy_times(frames: list[dict]) -> list[float]:
    raw = [frame.get("local_time") for frame in frames]
    if all(isinstance(item, str) for item in raw):
        try:
            values = [datetime.fromisoformat(item.replace("Z", "+00:00")).timestamp() for item in raw]
        except ValueError:
            values = []
        if values and all(isfinite(item) for item in values) and all(current > previous for previous, current in zip(values, values[1:])):
            origin = values[0]
            return [item - origin for item in values]
    return [synthetic_received_monotonic(ordinal) for ordinal in range(len(frames))]


def _validate_native_manifest(manifest: dict) -> tuple[list[dict], dict, tuple[int, int]]:
    required = {
        "schema_version", "status", "reason", "scope", "sanitization", "source_provenance", "capture_size",
        "minimap_rect", "interval_seconds", "committed_frame_count", "skipped_frame_count", "dropped_frame_count",
        "uncommitted_files", "frames",
    }
    if not required.issubset(manifest):
        raise ValueError("native replay manifest is missing required fields")
    if manifest["schema_version"] != 2:
        raise ValueError("unsupported native replay schema version")
    if manifest["status"] not in {"completed", "aborted", "failed"} or not isinstance(manifest["reason"], str) or not manifest["reason"].strip():
        raise ValueError("native replay manifest has invalid terminal status")
    if manifest["scope"] != "raw-local-minimap-roi" or manifest["sanitization"] != "unreviewed":
        raise ValueError("native replay manifest has unsupported scope")
    provenance = _required_object(manifest, "source_provenance")
    if provenance.get("kind") != "native-frame-source-callback" or not isinstance(provenance.get("sequence_semantics"), str):
        raise ValueError("native replay manifest has invalid source_provenance")
    rect = _required_object(manifest, "minimap_rect")
    for key in ("x", "y", "width", "height"):
        _exact_non_negative_int(rect.get(key), f"minimap_rect.{key}")
    if not rect["width"] or not rect["height"]:
        raise ValueError("minimap_rect dimensions must be positive")
    capture = _required_object(manifest, "capture_size")
    for key in ("width", "height"):
        _exact_non_negative_int(capture.get(key), f"capture_size.{key}")
    if rect["x"] + rect["width"] > capture["width"] or rect["y"] + rect["height"] > capture["height"]:
        raise ValueError("minimap_rect leaves capture_size")
    _finite_non_negative(manifest["interval_seconds"], "interval_seconds")
    for key in ("committed_frame_count", "skipped_frame_count", "dropped_frame_count"):
        _exact_non_negative_int(manifest[key], key)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != manifest["committed_frame_count"]:
        raise ValueError("committed_frame_count does not match frames")
    if not isinstance(manifest["uncommitted_files"], list):
        raise ValueError("uncommitted_files must be a list")
    return frames, provenance, (rect["width"], rect["height"])


def load_native_motion_replay(manifest_path: str | Path) -> tuple[NativeMotionReplayFrame, ...]:
    """Load and fully validate a schema-v2 recording or schema-v1 compact fixture."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds byte limit")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    schema = manifest.get("schema_version")
    if schema == 2:
        frames, provenance, expected_size = _validate_native_manifest(manifest)
        timestamps = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("frame record must be an object")
            timestamps.append(_finite_non_negative(frame.get("elapsed_seconds"), "elapsed_seconds"))
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("elapsed_seconds must strictly increase")
    elif schema == 1:
        frames = manifest.get("frames")
        if not isinstance(frames, list):
            raise ValueError("legacy fixture frames must be a list")
        provenance, expected_size, timestamps = {"kind": "legacy-compact-fixture", "timeline": "manifest-local-time-or-synthetic-ordinal"}, None, None
        timestamps = _legacy_times(frames) if all(isinstance(frame, dict) for frame in frames) else []
    else:
        raise ValueError("unsupported native replay schema version")
    if not frames or len(frames) > MAX_FRAME_COUNT:
        raise ValueError("fixture frame count is outside limits")
    root = path.parent.resolve()
    total_bytes = 0
    replay: list[NativeMotionReplayFrame] = []
    prior_sequence: int | None = None
    for ordinal, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        if not isinstance(frame, dict):
            raise ValueError("frame record must be an object")
        source_sequence = _exact_non_negative_int(frame.get("source_sequence"), "source_sequence")
        if prior_sequence is not None and source_sequence <= prior_sequence:
            raise ValueError("source_sequence must strictly increase")
        prior_sequence = source_sequence
        image_path = _contained_file(root, frame.get("file"))
        total_bytes += _verify_hash(image_path, frame.get("sha256"))
        if total_bytes > MAX_TOTAL_FRAME_BYTES:
            raise ValueError("fixture frames exceed total byte limit")
        roi = _decode_roi(image_path, expected_size=expected_size)
        replay.append(NativeMotionReplayFrame(
            envelope=FrameEnvelope(source_sequence=source_sequence, processing_ordinal=ordinal, received_monotonic=timestamp, calibration_epoch=0),
            roi=roi,
            provenance=dict(provenance),
            expected_state=frame.get("expected_state") if isinstance(frame.get("expected_state"), str) else None,
        ))
    return tuple(replay)


class NativeMotionReplayLoader:
    """Iterable compatibility wrapper for callers preferring an explicit loader."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)

    def load(self) -> tuple[NativeMotionReplayFrame, ...]:
        return load_native_motion_replay(self.manifest_path)

    def __iter__(self) -> Iterator[NativeMotionReplayFrame]:
        return iter(self.load())
