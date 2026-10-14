import hashlib
import json
import threading

import cv2
import numpy as np
import pytest

from bdo_assistant.minimap_replay_recorder import NativeMinimapReplayRecorder
from bdo_assistant.minimap_replay_recorder import NativeMinimapReplaySession
from bdo_assistant.ui_locator import Rect


def test_native_recorder_requires_explicit_start_and_persists_only_minimap_roi(tmp_path) -> None:
    destination = tmp_path / "replay"
    recorder = NativeMinimapReplayRecorder(
        destination,
        minimap=Rect(100, 20, 80, 60),
        interval_seconds=0.5,
        max_frames=2,
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:80, 100:180] = (10, 20, 30)

    with pytest.raises(RuntimeError, match="not started"):
        recorder.record(1, frame, elapsed_seconds=0.0)
    assert not destination.exists()

    recorder.start()
    assert recorder.record(1, frame, elapsed_seconds=0.0)
    assert not recorder.record(2, frame, elapsed_seconds=0.1)
    assert recorder.record(3, frame, elapsed_seconds=0.5)
    manifest_path = recorder.finish()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "completed"
    assert manifest["reason"] == "completed"
    assert manifest["scope"] == "raw-local-minimap-roi"
    assert manifest["sanitization"] == "unreviewed"
    assert manifest["capture_size"] == {"width": 200, "height": 100}
    assert manifest["minimap_rect"] == {"x": 100, "y": 20, "width": 80, "height": 60}
    assert [item["source_sequence"] for item in manifest["frames"]] == [1, 3]
    assert manifest["committed_frame_count"] == 2
    assert manifest["skipped_frame_count"] == 1
    assert manifest["dropped_frame_count"] == 0
    assert not list(destination.glob("*.jpg"))
    for item in manifest["frames"]:
        image_path = destination / item["file"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert image.shape == (60, 80, 3)
        assert tuple(image[0, 0]) == (10, 20, 30)
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == item["sha256"]


def test_native_recorder_without_cap_keeps_recording_past_default_capacity(tmp_path) -> None:
    recorder = NativeMinimapReplayRecorder(
        tmp_path / "uncapped",
        minimap=Rect(0, 0, 20, 10),
        interval_seconds=0.000001,
        max_frames=None,
    )
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    recorder.start()
    for sequence in range(1, 302):
        assert recorder.record(sequence, frame, elapsed_seconds=sequence * 0.001)
    assert recorder.frame_count == 301
    assert not recorder.at_capacity


@pytest.mark.parametrize("rect", (Rect(0.5, 0, 20, 10), Rect(0, float("nan"), 20, 10)))
def test_native_recorder_rejects_non_integer_minimap_geometry(tmp_path, rect: Rect) -> None:
    with pytest.raises(ValueError, match="minimap"):
        NativeMinimapReplayRecorder(tmp_path / "replay", minimap=rect)


def test_native_recorder_rejects_intervals_below_manifest_time_precision(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 0.000001"):
        NativeMinimapReplayRecorder(
            tmp_path / "replay", minimap=Rect(0, 0, 20, 10), interval_seconds=0.0000005
        )


def test_native_replay_session_captures_only_after_explicit_start_and_finishes_cleanly(tmp_path) -> None:
    class Source:
        stopped = False

        def start(self, on_frame, on_closed) -> None:
            self.on_frame = on_frame
            self.on_closed = on_closed

        def stop(self) -> None:
            self.stopped = True

    source = Source()
    recorder = NativeMinimapReplayRecorder(
        tmp_path / "replay", minimap=Rect(0, 0, 20, 10), interval_seconds=0.1
    )
    session = NativeMinimapReplaySession(source, recorder, clock=iter((0.0, 0.2)).__next__)

    session.start()
    source.on_frame(np.full((20, 30, 3), (1, 2, 3), dtype=np.uint8))
    manifest_path = session.stop()

    assert source.stopped
    assert manifest_path.is_file()


def test_native_replay_session_accepts_frame_delivered_synchronously_during_start(tmp_path) -> None:
    class EagerSource:
        def start(self, on_frame, on_closed) -> None:
            on_frame(np.full((20, 30, 3), (1, 2, 3), dtype=np.uint8))

        def stop(self) -> None:
            pass

    recorder = NativeMinimapReplayRecorder(
        tmp_path / "replay", minimap=Rect(0, 0, 20, 10), interval_seconds=0.1
    )
    session = NativeMinimapReplaySession(EagerSource(), recorder, clock=iter((0.0, 0.2)).__next__)

    session.start()

    assert session.stop().is_file()


def test_native_replay_session_finalizes_when_source_closes_after_recording(tmp_path) -> None:
    class Source:
        def start(self, on_frame, on_closed) -> None:
            self.on_frame = on_frame
            self.on_closed = on_closed

        def stop(self) -> None:
            pass

    source = Source()
    recorder = NativeMinimapReplayRecorder(
        tmp_path / "replay", minimap=Rect(0, 0, 20, 10), interval_seconds=0.1
    )
    session = NativeMinimapReplaySession(source, recorder, clock=iter((0.0, 0.2)).__next__)
    session.start()
    source.on_frame(np.full((20, 30, 3), (1, 2, 3), dtype=np.uint8))

    source.on_closed()

    assert session.stop().is_file()


def test_native_replay_session_finalizes_partial_start_failure(tmp_path) -> None:
    class FailingSource:
        stopped = False

        def start(self, on_frame, on_closed) -> None:
            on_frame(np.full((20, 30, 3), (1, 2, 3), dtype=np.uint8))
            raise RuntimeError("source start failed")

        def stop(self) -> None:
            self.stopped = True

    source = FailingSource()
    recorder = NativeMinimapReplayRecorder(
        tmp_path / "replay", minimap=Rect(0, 0, 20, 10), interval_seconds=0.1
    )
    session = NativeMinimapReplaySession(source, recorder, clock=iter((0.0, 0.2)).__next__)

    with pytest.raises(RuntimeError, match="source start failed"):
        session.start()

    assert source.stopped
    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "source start failed" in manifest["reason"]
    assert manifest["committed_frame_count"] == 1
    assert len(manifest["frames"]) == 1
    frame = tmp_path / "replay" / manifest["frames"][0]["file"]
    assert hashlib.sha256(frame.read_bytes()).hexdigest() == manifest["frames"][0]["sha256"]


def test_native_recorder_abort_preserves_partial_frame_inventory_and_hashes(tmp_path) -> None:
    recorder = NativeMinimapReplayRecorder(
        tmp_path / "replay", minimap=Rect(0, 0, 20, 10), interval_seconds=0.1
    )
    recorder.start()
    assert recorder.record(1, np.full((20, 30, 3), 7, dtype=np.uint8), elapsed_seconds=0.0)

    manifest = json.loads(recorder.abort(reason="operator cancelled").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 2
    assert manifest["status"] == "aborted"
    assert manifest["reason"] == "operator cancelled"
    assert manifest["committed_frame_count"] == 1
    assert len(manifest["frames"]) == 1
    image_path = tmp_path / "replay" / manifest["frames"][0]["file"]
    assert image_path.is_file()
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == manifest["frames"][0]["sha256"]


def test_native_recorder_empty_abort_writes_a_valid_terminal_manifest(tmp_path) -> None:
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10))
    recorder.start()

    manifest = json.loads(recorder.abort(reason="no frames received").read_text(encoding="utf-8"))

    assert manifest["status"] == "aborted"
    assert manifest["reason"] == "no frames received"
    assert manifest["frames"] == []
    assert manifest["committed_frame_count"] == 0
    assert manifest["capture_size"] is None


def test_native_recorder_persists_timestamped_operator_labels_in_terminal_manifest(tmp_path) -> None:
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10))
    recorder.start()

    recorder.label("ARRIVED", elapsed_seconds=1.25)
    recorder.label("STUCK", elapsed_seconds=2.5)
    manifest = json.loads(recorder.abort(reason="operator stopped").read_text(encoding="utf-8"))

    assert manifest["labels"] == [
        {"label": "ARRIVED", "elapsed_seconds": 1.25},
        {"label": "STUCK", "elapsed_seconds": 2.5},
    ]


def test_native_recorder_makes_rapid_operator_label_timestamps_strictly_increasing(tmp_path) -> None:
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10))
    recorder.start()

    recorder.label("ARRIVED", elapsed_seconds=0.0000004)
    recorder.label("MOVING", elapsed_seconds=0.0000005)
    manifest = json.loads(recorder.abort(reason="operator stopped").read_text(encoding="utf-8"))

    timestamps = [item["elapsed_seconds"] for item in manifest["labels"]]
    assert timestamps == [0.0, 0.000001]


def test_native_recorder_rejects_elapsed_rollback_without_consuming_sequence(tmp_path) -> None:
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10))
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    recorder.start()
    assert recorder.record(1, frame, elapsed_seconds=1.0)

    with pytest.raises(ValueError, match="elapsed_seconds must not decrease"):
        recorder.record(2, frame, elapsed_seconds=0.5)

    assert recorder.record(2, frame, elapsed_seconds=1.3)


def test_native_replay_session_disk_error_has_failed_terminal_manifest(tmp_path, monkeypatch) -> None:
    class Source:
        def start(self, on_frame, on_closed) -> None:
            self.on_frame = on_frame
            self.on_closed = on_closed

        def stop(self) -> None:
            pass

    source = Source()
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10))
    session = NativeMinimapReplaySession(source, recorder, clock=iter((0.0, 0.2)).__next__)
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    session.start()

    with pytest.raises(OSError, match="could not write replay frame"):
        source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))

    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "could not write replay frame" in manifest["reason"]
    assert manifest["committed_frame_count"] == 0


def test_native_replay_session_frame_cap_stops_source_and_finalizes(tmp_path) -> None:
    class Source:
        stopped = 0

        def start(self, on_frame, on_closed) -> None:
            self.on_frame = on_frame

        def stop(self) -> None:
            self.stopped += 1

    source = Source()
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10), max_frames=1)
    session = NativeMinimapReplaySession(source, recorder, clock=iter((0.0, 0.2)).__next__)

    session.start()
    source.on_frame(np.zeros((20, 30, 3), dtype=np.uint8))

    assert source.stopped == 1
    assert json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))["reason"] == "frame cap reached"


def test_native_replay_session_deadline_stops_without_new_frames(tmp_path) -> None:
    class Timer:
        def __init__(self, callback) -> None:
            self.callback = callback
            self.started = False

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.started = False

        def fire(self) -> None:
            self.callback()

    class Source:
        stopped = 0

        def start(self, on_frame, on_closed) -> None:
            pass

        def stop(self) -> None:
            self.stopped += 1

    timers = []
    source = Source()
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10))
    session = NativeMinimapReplaySession(
        source, recorder, deadline_seconds=5.0, timer_factory=lambda _delay, callback: timers.append(Timer(callback)) or timers[-1]
    )

    session.start()
    timers[0].fire()

    assert source.stopped == 1
    assert json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))["reason"] == "deadline reached"


def test_native_replay_session_retries_failed_start_cleanup(tmp_path) -> None:
    class Source:
        attempts = 0

        def start(self, on_frame, on_closed) -> None:
            raise RuntimeError("start exploded")

        def stop(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("stop exploded")

    source = Source()
    session = NativeMinimapReplaySession(source, NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10)))

    with pytest.raises(RuntimeError, match="start exploded"):
        session.start()
    assert session.cleanup_required
    session.retry_cleanup()

    assert source.attempts == 2
    assert not session.cleanup_required


def test_native_replay_session_stop_racing_start_leaves_no_live_source(tmp_path) -> None:
    class Source:
        def __init__(self) -> None:
            self.start_entered = threading.Event()
            self.allow_start_return = threading.Event()
            self.stopped = threading.Event()

        def start(self, on_frame, on_closed) -> None:
            self.start_entered.set()
            assert self.allow_start_return.wait(1)

        def stop(self) -> None:
            self.stopped.set()

    source = Source()
    session = NativeMinimapReplaySession(source, NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10)))
    starter = threading.Thread(target=session.start)
    starter.start()
    assert source.start_entered.wait(1)
    stopper = threading.Thread(target=session.stop)
    stopper.start()
    source.allow_start_return.set()
    starter.join(1)
    stopper.join(1)

    assert not starter.is_alive()
    assert not stopper.is_alive()
    assert source.stopped.is_set()
    assert session.lifecycle_state == "TERMINAL"


def test_native_replay_session_duplicate_close_and_stop_finalize_once(tmp_path, monkeypatch) -> None:
    class Source:
        def start(self, on_frame, on_closed) -> None:
            self.on_frame = on_frame
            self.on_closed = on_closed

        def stop(self) -> None:
            pass

    source = Source()
    recorder = NativeMinimapReplayRecorder(tmp_path / "replay", minimap=Rect(0, 0, 20, 10))
    finalize_calls = 0
    original = recorder.abort

    def counted_abort(*, reason: str):
        nonlocal finalize_calls
        finalize_calls += 1
        return original(reason=reason)

    monkeypatch.setattr(recorder, "abort", counted_abort)
    session = NativeMinimapReplaySession(source, recorder)
    session.start()
    source.on_closed()
    source.on_closed()
    session.stop()

    assert finalize_calls == 1
