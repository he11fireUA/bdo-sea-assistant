from dataclasses import dataclass, field
from pathlib import Path
import threading
import wave

import numpy as np
import pytest

from bdo_assistant.notifications import EventType, NotificationEvent, Severity
from bdo_assistant.voice import (
    OfflineVoiceAdapter,
    QueuedVoicePlaybackScheduler,
    SoundDevicePlaybackBackend,
    VoiceSettings,
)


@dataclass
class RecordingAudioBackend:
    calls: list[tuple[Path, float]] = field(default_factory=list)

    def play(self, path: Path, volume: float) -> None:
        self.calls.append((path, volume))


def _voice_pack(root: Path, locale: str = "uk") -> Path:
    folder = root / locale
    folder.mkdir(parents=True)
    (root / "notification_bell.wav").write_bytes(b"RIFF-bell")
    for name in ("arrived", "stuck", "ship_hp_low", "rations_low", "test"):
        (folder / f"{name}.wav").write_bytes(b"RIFF-test")
    return root


def test_voice_adapter_plays_static_localized_clip_and_test_action(tmp_path: Path) -> None:
    backend = RecordingAudioBackend()
    adapter = OfflineVoiceAdapter(
        _voice_pack(tmp_path),
        backend,
        VoiceSettings(locale="uk", volume=0.7),
    )

    adapter.send(NotificationEvent(EventType.ARRIVED, Severity.INFO))
    adapter.test_voice()

    assert backend.calls == [
        (tmp_path / "notification_bell.wav", 0.7),
        (tmp_path / "uk" / "arrived.wav", 0.7),
        (tmp_path / "uk" / "test.wav", 0.7),
    ]


def test_voice_notification_plays_bell_then_phrase_without_an_artificial_gap(tmp_path: Path) -> None:
    backend = RecordingAudioBackend()
    root = _voice_pack(tmp_path)
    bell = root / "notification_bell.wav"
    bell.write_bytes(b"RIFF-bell")
    adapter = OfflineVoiceAdapter(
        root,
        backend,
        VoiceSettings(locale="uk", volume=0.7),
    )

    adapter.send(NotificationEvent(EventType.ARRIVED, Severity.INFO))

    assert backend.calls == [
        (bell, 0.7),
        (tmp_path / "uk" / "arrived.wav", 0.7),
    ]


def test_queued_voice_playback_does_not_block_notification_sender(tmp_path: Path) -> None:
    entered_playback = threading.Event()
    release_playback = threading.Event()

    class BlockingBackend(RecordingAudioBackend):
        def play(self, path: Path, volume: float) -> None:
            super().play(path, volume)
            if path.name == "notification_bell.wav":
                entered_playback.set()
                assert release_playback.wait(1)

    backend = BlockingBackend()
    scheduler = QueuedVoicePlaybackScheduler(backend)
    adapter = OfflineVoiceAdapter(
        _voice_pack(tmp_path),
        backend,
        VoiceSettings(locale="uk"),
        scheduler=scheduler,
    )

    adapter.send(NotificationEvent(EventType.ARRIVED, Severity.INFO))

    assert entered_playback.wait(1)
    assert scheduler.join(timeout=0) is False
    assert [path.name for path, _ in backend.calls] == ["notification_bell.wav"]
    release_playback.set()
    assert scheduler.join(timeout=1)
    assert [path.name for path, _ in backend.calls] == [
        "notification_bell.wav",
        "arrived.wav",
    ]
    scheduler.close(timeout=1)


@pytest.mark.parametrize("notification_first", [False, True])
def test_voice_test_uses_same_worker_without_blocking_caller(
    tmp_path: Path, notification_first: bool
) -> None:
    caller = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    owners = []

    class BlockingBackend(RecordingAudioBackend):
        def play(self, path: Path, volume: float) -> None:
            assert threading.get_ident() != caller, "playback ran on the UI caller"
            owners.append(threading.get_ident())
            super().play(path, volume)
            if len(self.calls) == 1:
                entered.set()
                assert release.wait(2)

    backend = BlockingBackend()
    scheduler = QueuedVoicePlaybackScheduler(backend)
    adapter = OfflineVoiceAdapter(
        _voice_pack(tmp_path), backend, VoiceSettings(volume=0.7), scheduler=scheduler
    )
    try:
        if notification_first:
            adapter.send(NotificationEvent(EventType.ARRIVED, Severity.INFO))
            assert entered.wait(1)
        adapter.test_voice()
        assert entered.wait(1)
        assert scheduler.join(timeout=0) is False
        assert len(backend.calls) == 1
        release.set()
        assert scheduler.join(timeout=1)
        expected = (["notification_bell.wav", "arrived.wav"] if notification_first else [])
        assert [path.name for path, _ in backend.calls] == expected + ["test.wav"]
        assert {volume for _, volume in backend.calls} == {0.7}
        assert len(set(owners)) == 1
        assert scheduler.last_error == ""
    finally:
        release.set()
        scheduler.close(timeout=1)


@pytest.mark.parametrize("test_clip_first", [False, True])
def test_queued_voice_scheduler_records_failure_and_processes_later_notification(
    tmp_path: Path, test_clip_first: bool
) -> None:
    class FailOnceBackend:
        def __init__(self) -> None:
            self.calls = []
            self.failed = False

        def play(self, path: Path, volume: float) -> None:
            self.calls.append(path.name)
            if not self.failed:
                self.failed = True
                raise RuntimeError("speaker unavailable")

    backend = FailOnceBackend()
    scheduler = QueuedVoicePlaybackScheduler(backend)
    adapter = OfflineVoiceAdapter(
        _voice_pack(tmp_path), backend, VoiceSettings(locale="uk"), scheduler=scheduler
    )

    if test_clip_first:
        adapter.test_voice()
    else:
        adapter.send(NotificationEvent(EventType.ARRIVED, Severity.INFO))
    assert scheduler.join(timeout=1)
    assert "speaker unavailable" in scheduler.last_error

    adapter.send(NotificationEvent(EventType.STUCK, Severity.WARNING))
    assert scheduler.join(timeout=1)
    assert backend.calls[-2:] == ["notification_bell.wav", "stuck.wav"]
    scheduler.close(timeout=1)


@pytest.mark.parametrize("pending_test_clip", [False, True])
def test_queued_voice_scheduler_close_does_not_block_when_queue_is_full(
    tmp_path: Path, pending_test_clip: bool
) -> None:
    entered_playback = threading.Event()
    release_playback = threading.Event()
    closed = threading.Event()

    class BlockingBackend(RecordingAudioBackend):
        def play(self, path: Path, volume: float) -> None:
            super().play(path, volume)
            if path.name == "notification_bell.wav":
                entered_playback.set()
                assert release_playback.wait(1)

    backend = BlockingBackend()
    scheduler = QueuedVoicePlaybackScheduler(backend, max_pending=1)
    adapter = OfflineVoiceAdapter(
        _voice_pack(tmp_path), backend, VoiceSettings(locale="uk"), scheduler=scheduler
    )
    try:
        adapter.send(NotificationEvent(EventType.ARRIVED, Severity.INFO))
        assert entered_playback.wait(1)
        if pending_test_clip:
            adapter.test_voice()
        else:
            adapter.send(NotificationEvent(EventType.STUCK, Severity.WARNING))
        with pytest.raises(RuntimeError, match="queue is full"):
            adapter.test_voice()

        closer = threading.Thread(target=lambda: (scheduler.close(timeout=0), closed.set()))
        closer.start()
        assert closed.wait(1)
        closer.join(1)
        with pytest.raises(RuntimeError, match="scheduler is closed"):
            adapter.test_voice()
        release_playback.set()
        assert scheduler.join(timeout=1)
        assert [path.name for path, _ in backend.calls] == [
            "notification_bell.wav", "arrived.wav"
        ]
        assert scheduler.last_error == ""
    finally:
        release_playback.set()
        scheduler.close(timeout=1)


def test_voice_adapter_suppresses_restore_events_by_default(tmp_path: Path) -> None:
    backend = RecordingAudioBackend()
    adapter = OfflineVoiceAdapter(
        _voice_pack(tmp_path),
        backend,
        VoiceSettings(locale="uk"),
    )

    adapter.send(NotificationEvent(EventType.CAPTURE_RESTORED, Severity.INFO))
    adapter.send(NotificationEvent(EventType.MINIMAP_VISIBILITY_RESTORED, Severity.INFO))

    assert backend.calls == []


def test_disabled_voice_adapter_is_silent(tmp_path: Path) -> None:
    backend = RecordingAudioBackend()
    adapter = OfflineVoiceAdapter(
        _voice_pack(tmp_path),
        backend,
        VoiceSettings(enabled=False),
    )

    adapter.send(NotificationEvent(EventType.STUCK, Severity.WARNING))
    adapter.test_voice()

    assert backend.calls == []


@pytest.mark.parametrize(
    "settings",
    [
        VoiceSettings(locale="de"),
        VoiceSettings(volume=-0.1),
        VoiceSettings(volume=1.1),
        VoiceSettings(volume=float("nan")),
    ],
)
def test_voice_adapter_rejects_invalid_settings(tmp_path: Path, settings: VoiceSettings) -> None:
    with pytest.raises(ValueError, match="voice settings"):
        OfflineVoiceAdapter(_voice_pack(tmp_path), RecordingAudioBackend(), settings)


def test_voice_adapter_rejects_scheduler_without_single_clip_support(tmp_path: Path) -> None:
    class NotificationOnlyScheduler:
        def submit(self, bell: Path, phrase: Path, volume: float) -> None:
            pass

    with pytest.raises(ValueError, match="submit_clip"):
        OfflineVoiceAdapter(
            _voice_pack(tmp_path), RecordingAudioBackend(), scheduler=NotificationOnlyScheduler()
        )


def test_sounddevice_backend_scales_pcm16_and_plays_blocking(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    samples = np.array([10000, -10000], dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(samples.tobytes())
    calls = []

    def play(data, sample_rate, *, blocking):
        calls.append((data.copy(), sample_rate, blocking))

    SoundDevicePlaybackBackend(play_function=play).play(path, 0.5)

    assert calls[0][0].tolist() == pytest.approx([5000, -5000], abs=1)
    assert calls[0][1:] == (8000, True)
