from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2


@dataclass(frozen=True)
class StateRange:
    start_sequence: int
    end_sequence: int
    state: str

    def contains(self, sequence: int) -> bool:
        return self.start_sequence <= sequence <= self.end_sequence


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    source_label: str
    start_sequence: int
    end_sequence: int
    crop_normalized: tuple[float, float, float, float]
    states: tuple[StateRange, ...]


def _expected_state(sequence: int, ranges: Iterable[StateRange]) -> str:
    matches = [item.state for item in ranges if item.contains(sequence)]
    if len(matches) != 1:
        raise ValueError(f"sequence {sequence} must have exactly one expected state")
    return matches[0]


def _crop(image, normalized: tuple[float, float, float, float]):
    x, y, width, height = normalized
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 or y + height > 1:
        raise ValueError("crop_normalized must fit inside the frame")
    frame_height, frame_width = image.shape[:2]
    left = round(x * frame_width)
    top = round(y * frame_height)
    right = round((x + width) * frame_width)
    bottom = round((y + height) * frame_height)
    return image[top:bottom, left:right]


def extract_fixture(source_journey: str | Path, output_dir: str | Path, spec: FixtureSpec) -> Path:
    source = Path(source_journey)
    output = Path(output_dir)
    label_path = source / spec.source_label
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    with (source / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = {int(row["sequence"]): row for row in csv.DictReader(handle)}

    selected = []
    for sequence in range(spec.start_sequence, spec.end_sequence + 1):
        if sequence not in rows:
            raise ValueError(f"source sequence {sequence} is missing from metrics.csv")
        selected.append((sequence, rows[sequence], _expected_state(sequence, spec.states)))

    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames = []
    for ordinal, (sequence, row, state) in enumerate(selected):
        preview_path = Path(row["preview_file"])
        if not preview_path.is_absolute():
            preview_path = source / preview_path
        image = cv2.imread(str(preview_path))
        if image is None:
            raise ValueError(f"could not read preview for sequence {sequence}: {preview_path}")
        cropped = _crop(image, spec.crop_normalized)
        if cropped.size == 0:
            raise ValueError(f"empty crop for sequence {sequence}")
        relative = Path("frames") / f"frame_{ordinal:04d}.jpg"
        destination = output / relative
        if not cv2.imwrite(str(destination), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"could not write {destination}")
        manifest_frames.append(
            {
                "file": relative.as_posix(),
                "source_sequence": sequence,
                "local_time": row["local_time"],
                "expected_state": state,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "name": spec.name,
        "source": {"journey": source.name, "label": spec.source_label},
        "crop_normalized": {
            "x": spec.crop_normalized[0],
            "y": spec.crop_normalized[1],
            "width": spec.crop_normalized[2],
            "height": spec.crop_normalized[3],
        },
        "frames": manifest_frames,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def validate_fixture(manifest_path: str | Path) -> int:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported fixture schema version")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("fixture must contain at least one frame")
    for frame in frames:
        relative = Path(frame["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("fixture frame path must be relative and contained")
        frame_path = path.parent / relative
        if not frame_path.is_file():
            raise ValueError(f"fixture frame is missing: {relative.as_posix()}")
        actual = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        if actual != frame["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative.as_posix()}")
    return len(frames)
