from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
from time import monotonic
from typing import Callable, Protocol
import threading
from uuid import uuid4

import cv2
import numpy as np

from .ui_locator import Rect

MANIFEST_TIME_PRECISION_SECONDS = 0.000001


class FrameSource(Protocol):
    def start(self, on_frame: Callable[[np.ndarray], None], on_closed: Callable[[], None]) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class _RecordedFrame:
    source_sequence: int
    elapsed_seconds: float
    file: str
    sha256: str


class NativeMinimapReplayRecorder:
    """Explicit recorder that persists only a confirmed native minimap ROI."""

    def __init__(
        self,
        destination: str | Path,
        *,
        minimap: Rect,
        interval_seconds: float = 0.2,
        max_frames: int | None = 300,
        expected_capture_size: tuple[int, int] | None = None,
    ) -> None:
        if any(type(value) is not int for value in (minimap.x, minimap.y, minimap.width, minimap.height)):
            raise ValueError("minimap geometry must use exact integers")
        if minimap.width <= 0 or minimap.height <= 0:
            raise ValueError("minimap must have positive dimensions")
        if not isfinite(interval_seconds) or interval_seconds < MANIFEST_TIME_PRECISION_SECONDS:
            raise ValueError("interval_seconds must be finite and at least 0.000001")
        if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
            raise ValueError("max_frames must be a positive integer or None")
        if expected_capture_size is not None and (
            type(expected_capture_size) is not tuple
            or len(expected_capture_size) != 2
            or any(type(value) is not int or value <= 0 for value in expected_capture_size)
        ):
            raise ValueError("expected_capture_size must contain positive integers")
        self._destination = Path(destination)
        self._minimap = minimap
        self._interval_seconds = float(interval_seconds)
        self._max_frames = max_frames
        self._expected_capture_size = expected_capture_size
        self._started = False
        self._finished = False
        self._last_elapsed: float | None = None
        self._last_seen_elapsed: float | None = None
        self._last_sequence: int | None = None
        self._capture_size: tuple[int, int] | None = None
        self._frames: list[_RecordedFrame] = []
        self._labels: list[dict[str, object]] = []
        self._skipped_frame_count = 0
        self._dropped_frame_count = 0

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def at_capacity(self) -> bool:
        return self._max_frames is not None and len(self._frames) >= self._max_frames

    def start(self) -> None:
        if self._started:
            raise RuntimeError("recorder is already started")
        self._destination.mkdir(parents=True, exist_ok=False)
        (self._destination / "frames").mkdir()
        self._started = True

    def record(self, sequence: int, frame: np.ndarray, *, elapsed_seconds: float) -> bool:
        if not self._started or self._finished:
            raise RuntimeError("recorder is not started")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if not isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self._last_seen_elapsed is not None and elapsed_seconds < self._last_seen_elapsed:
            raise ValueError("elapsed_seconds must not decrease")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ValueError("sequence must strictly increase")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a non-empty uint8 BGR image")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("frame must be a non-empty uint8 BGR image")
        if self._expected_capture_size is not None and self._expected_capture_size != (width, height):
            raise ValueError("capture size does not match calibrated capture size")
        if (
            self._minimap.x < 0
            or self._minimap.y < 0
            or self._minimap.x + self._minimap.width > width
            or self._minimap.y + self._minimap.height > height
        ):
            raise ValueError("minimap leaves frame")
        if self._capture_size is None:
            self._capture_size = (width, height)
        elif self._capture_size != (width, height):
            raise ValueError("capture size changed during replay recording")
        self._last_sequence = sequence
        self._last_seen_elapsed = elapsed_seconds
        if self._max_frames is not None and len(self._frames) >= self._max_frames:
            self._dropped_frame_count += 1
            return False
        if self._last_elapsed is not None and elapsed_seconds - self._last_elapsed < self._interval_seconds:
            self._skipped_frame_count += 1
            return False

        crop = frame[
            self._minimap.y : self._minimap.y + self._minimap.height,
            self._minimap.x : self._minimap.x + self._minimap.width,
        ].copy()
        relative = Path("frames") / f"frame_{len(self._frames):04d}.png"
        output = self._destination / relative
        if not cv2.imwrite(str(output), crop):
            raise OSError(f"could not write replay frame: {output}")
        self._frames.append(
            _RecordedFrame(
                source_sequence=sequence,
                elapsed_seconds=round(float(elapsed_seconds), 6),
                file=relative.as_posix(),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            )
        )
        self._last_elapsed = elapsed_seconds
        return True

    def label(self, label: str, *, elapsed_seconds: float) -> None:
        if not self._started or self._finished:
            raise RuntimeError("recorder is not active")
        if label not in {"ARRIVED", "STUCK", "MOVING"}:
            raise ValueError("label must be ARRIVED, STUCK, or MOVING")
        if not isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("label elapsed_seconds must be finite and non-negative")
        rounded_elapsed = round(float(elapsed_seconds), 6)
        if self._labels:
            rounded_elapsed = max(
                rounded_elapsed,
                float(self._labels[-1]["elapsed_seconds"]) + MANIFEST_TIME_PRECISION_SECONDS,
            )
        self._labels.append({
            "label": label,
            "elapsed_seconds": rounded_elapsed,
        })

    def _terminal_manifest(self, *, status: str, reason: str) -> dict[str, object]:
        if status not in {"completed", "aborted", "failed"}:
            raise ValueError("terminal status is invalid")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("terminal reason must be a non-empty string")
        capture_size = None
        if self._capture_size is not None:
            width, height = self._capture_size
            capture_size = {"width": width, "height": height}
        committed_files = {frame.file for frame in self._frames}
        uncommitted_files = []
        frames_directory = self._destination / "frames"
        if frames_directory.is_dir():
            for path in sorted(frames_directory.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(self._destination).as_posix()
                    if relative not in committed_files:
                        uncommitted_files.append(
                            {"file": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        )
        return {
            "schema_version": 2,
            "status": status,
            "reason": reason,
            "scope": "raw-local-minimap-roi",
            "sanitization": "unreviewed",
            "source_provenance": {
                "kind": "native-frame-source-callback",
                "sequence_semantics": "strictly-increasing-source-callback-order",
            },
            "capture_size": capture_size,
            "minimap_rect": {
                "x": self._minimap.x,
                "y": self._minimap.y,
                "width": self._minimap.width,
                "height": self._minimap.height,
            },
            "interval_seconds": self._interval_seconds,
            "committed_frame_count": len(self._frames),
            "skipped_frame_count": self._skipped_frame_count,
            "dropped_frame_count": self._dropped_frame_count,
            "frames": [frame.__dict__ for frame in self._frames],
            "labels": self._labels,
            "uncommitted_files": uncommitted_files,
        }

    def _write_terminal_manifest(self, *, status: str, reason: str) -> Path:
        manifest = self._terminal_manifest(status=status, reason=reason)
        destination = self._destination / "manifest.json"
        if destination.exists():
            raise FileExistsError(f"terminal manifest already exists: {destination}")
        temporary = self._destination / f".manifest-{uuid4().hex}.json.tmp"
        try:
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.link(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._finished = True
        return destination

    def finish(self, *, reason: str = "completed") -> Path:
        if not self._started or self._finished:
            raise RuntimeError("recorder is not active")
        if self._capture_size is None or not self._frames:
            raise RuntimeError("recorder has no frames")
        return self._write_terminal_manifest(status="completed", reason=reason)

    def abort(self, *, reason: str = "aborted") -> Path:
        if not self._started or self._finished:
            raise RuntimeError("recorder is not active")
        return self._write_terminal_manifest(status="aborted", reason=reason)

    def fail(self, *, reason: str) -> Path:
        if not self._started or self._finished:
            raise RuntimeError("recorder is not active")
        return self._write_terminal_manifest(status="failed", reason=reason)


class NativeMinimapReplaySession:
    """Serializes source lifetime separately from terminal evidence."""

    def __init__(self, source: FrameSource, recorder: NativeMinimapReplayRecorder, *, clock: Callable[[], float] = monotonic, deadline_seconds: float | None = None, timer_factory: Callable[[float, Callable[[], None]], object] = threading.Timer, start_wait_seconds: float = 5.0, on_frame_recorded: Callable[[np.ndarray], None] | None = None) -> None:
        if not callable(getattr(source, "start", None)) or not callable(getattr(source, "stop", None)):
            raise ValueError("source must provide start and stop")
        if type(recorder) is not NativeMinimapReplayRecorder:
            raise ValueError("recorder must be NativeMinimapReplayRecorder")
        if not callable(clock) or not callable(timer_factory):
            raise ValueError("clock and timer_factory must be callable")
        if on_frame_recorded is not None and not callable(on_frame_recorded):
            raise ValueError("on_frame_recorded must be callable or None")
        if deadline_seconds is not None and (not isfinite(deadline_seconds) or deadline_seconds <= 0):
            raise ValueError("deadline_seconds must be finite and positive")
        if not isfinite(start_wait_seconds) or start_wait_seconds <= 0:
            raise ValueError("start_wait_seconds must be finite and positive")
        self._source, self._recorder, self._clock = source, recorder, clock
        self._on_frame_recorded = on_frame_recorded
        self._deadline_seconds, self._timer_factory = deadline_seconds, timer_factory
        self._start_wait_seconds = start_wait_seconds
        self._started_at: float | None = None
        self._sequence = 0
        self._state = "NEW"
        self._manifest_path: Path | None = None
        self._cleanup_required = False
        self._deadline_timer: object | None = None
        self._lock = threading.RLock()
        self._start_finished = threading.Event()

    @property
    def lifecycle_state(self) -> str:
        with self._lock: return self._state

    @property
    def cleanup_required(self) -> bool:
        with self._lock: return self._cleanup_required

    def _finalize_locked(self, *, status: str | None = None, reason: str) -> Path:
        if self._manifest_path is None:
            self._manifest_path = self._recorder.fail(reason=reason) if status == "failed" else (self._recorder.finish(reason=reason) if self._recorder.frame_count else self._recorder.abort(reason=reason))
        return self._manifest_path

    def _terminalize(self, *, reason: str, status: str | None = None, cleanup: bool, callback: bool) -> Path:
        with self._lock:
            path = self._finalize_locked(status=status, reason=reason)
            self._state = "TERMINAL"
            if self._deadline_timer is not None:
                self._deadline_timer.cancel()  # type: ignore[attr-defined]
                self._deadline_timer = None
            self._cleanup_required |= cleanup
        if cleanup:
            if callback:
                threading.Thread(target=self.retry_cleanup, daemon=True).start()
            else:
                try: self.retry_cleanup()
                except Exception: pass
        return path

    def _on_deadline(self) -> None:
        self._terminalize(reason="deadline reached", cleanup=True, callback=True)

    def start(self) -> None:
        with self._lock:
            if self._state != "NEW": raise RuntimeError("replay session is already started")
            self._state = "STARTING"
            self._recorder.start()
            self._started_at = float(self._clock())
        try:
            self._source.start(self._on_frame, self._on_closed)
        except Exception as exc:
            self._terminalize(status="failed", reason=f"source start failed: {exc}", cleanup=True, callback=False)
            self._start_finished.set()
            raise
        with self._lock:
            stopping = self._state == "STOPPING"
            if self._state == "STARTING":
                self._state = "RUNNING"
                if self._deadline_seconds is not None:
                    self._deadline_timer = self._timer_factory(self._deadline_seconds, self._on_deadline)
                    self._deadline_timer.start()  # type: ignore[attr-defined]
        self._start_finished.set()
        if stopping: self._terminalize(reason="stopped by caller", cleanup=True, callback=False)

    def _on_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._state not in {"STARTING", "RUNNING"} or self._started_at is None: return
            self._sequence += 1
            try:
                recorded = self._recorder.record(self._sequence, frame, elapsed_seconds=float(self._clock()) - self._started_at)
            except Exception as exc:
                failure, capped, recorded = exc, False, False
            else:
                failure, capped = None, self._recorder.at_capacity
        if failure is not None:
            self._terminalize(status="failed", reason=f"recording failed: {failure}", cleanup=True, callback=True)
            raise failure
        if recorded and self._on_frame_recorded is not None:
            self._on_frame_recorded(frame)
        if capped: self._terminalize(reason="frame cap reached", cleanup=True, callback=True)

    def _on_closed(self) -> None:
        with self._lock:
            active = self._state in {"STARTING", "RUNNING", "STOPPING"}
        if active: self._terminalize(reason="source closed", cleanup=False, callback=True)

    def label(self, label: str) -> None:
        with self._lock:
            if self._state not in {"STARTING", "RUNNING"} or self._started_at is None:
                raise RuntimeError("replay session is not running")
            self._recorder.label(label, elapsed_seconds=float(self._clock()) - self._started_at)

    def stop(self) -> Path:
        with self._lock:
            if self._manifest_path is not None: return self._manifest_path
            if self._state == "STARTING": self._state, wait = "STOPPING", True
            elif self._state == "RUNNING": self._state, wait = "STOPPING", False
            else: raise RuntimeError("replay session is not running")
        if wait:
            if not self._start_finished.wait(self._start_wait_seconds):
                self._terminalize(reason="source start did not finish before stop deadline", status="failed", cleanup=True, callback=False)
                raise TimeoutError("source start did not finish before stop deadline; cleanup remains retryable")
            with self._lock:
                if self._manifest_path is not None: return self._manifest_path
        return self._terminalize(reason="stopped by caller", cleanup=True, callback=False)

    def retry_cleanup(self) -> None:
        with self._lock:
            if not self._cleanup_required: return
        try: self._source.stop()
        except Exception:
            with self._lock: self._cleanup_required = True
            raise
        with self._lock: self._cleanup_required = False
