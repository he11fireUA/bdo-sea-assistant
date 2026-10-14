from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from queue import Empty, Full, Queue
import threading
from typing import Protocol
import wave

import numpy as np

from .notifications import EventType, NotificationEvent


class AudioPlaybackBackend(Protocol):
    def play(self, path: Path, volume: float) -> None: ...


class VoicePlaybackScheduler(Protocol):
    def submit(self, bell: Path, phrase: Path, volume: float) -> None: ...

    def submit_clip(self, path: Path, volume: float) -> None: ...


class QueuedVoicePlaybackScheduler:
    """Single playback owner for notifications and test clips; callers only enqueue."""

    def __init__(
        self,
        backend: AudioPlaybackBackend,
        *,
        max_pending: int = 16,
    ) -> None:
        if not callable(getattr(backend, "play", None)):
            raise ValueError("audio backend must provide callable play")
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self._backend = backend
        self._jobs: Queue[tuple[tuple[Path, ...], float] | None] = Queue(maxsize=max_pending)
        self._idle = threading.Event()
        self._idle.set()
        self._closed = False
        self._last_error = ""
        self._pending = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="BDOVoice", daemon=True)
        self._worker.start()

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def _is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _record_error(self, error: Exception) -> None:
        with self._lock:
            self._last_error = f"{type(error).__name__}: {error}"

    def submit(self, bell: Path, phrase: Path, volume: float) -> None:
        self._submit((bell, phrase), volume)

    def submit_clip(self, path: Path, volume: float) -> None:
        self._submit((path,), volume)

    def _submit(self, clips: tuple[Path, ...], volume: float) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("voice playback scheduler is closed")
            try:
                self._jobs.put_nowait((clips, volume))
            except Full as exc:
                raise RuntimeError("voice playback queue is full") from exc
            self._pending += 1
            self._idle.clear()

    def _finish_pending_job(self) -> None:
        with self._lock:
            self._pending -= 1
            if self._pending < 0:
                raise RuntimeError("voice playback pending count underflow")
            if self._pending == 0:
                self._idle.set()

    def _discard_pending(self) -> None:
        while True:
            try:
                job = self._jobs.get_nowait()
            except Empty:
                return
            else:
                self._jobs.task_done()
                if job is not None:
                    self._finish_pending_job()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return
            try:
                if not self._is_closed():
                    clips, volume = job
                    for path in clips:
                        self._backend.play(path, volume)
            except Exception as exc:
                self._record_error(exc)
            finally:
                self._jobs.task_done()
                self._finish_pending_job()
            if self._is_closed():
                self._discard_pending()
                return

    def join(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def close(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._jobs.put_nowait(None)
            except Full:
                pass
        self._worker.join(timeout)


class SoundDevicePlaybackBackend:
    def __init__(self, play_function=None) -> None:
        if play_function is None:
            try:
                import sounddevice
            except ImportError as exc:
                raise RuntimeError("sounddevice is required for voice playback") from exc
            play_function = sounddevice.play
        if not callable(play_function):
            raise ValueError("play_function must be callable")
        self._play = play_function

    def play(self, path: Path, volume: float) -> None:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        if channels not in (1, 2) or sample_width != 2:
            raise ValueError("voice WAV must be mono/stereo PCM16")
        samples = np.frombuffer(frames, dtype="<i2")
        if channels == 2:
            samples = samples.reshape(-1, 2)
        scaled = np.clip(np.rint(samples.astype(np.float32) * volume), -32768, 32767).astype(
            np.int16
        )
        self._play(scaled, sample_rate, blocking=True)


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool = True
    locale: str = "uk"
    volume: float = 1.0
    announce_restored: bool = False


class OfflineVoiceAdapter:
    name = "voice"

    def __init__(
        self,
        voice_pack_root: Path,
        backend: AudioPlaybackBackend,
        settings: VoiceSettings = VoiceSettings(),
        *,
        scheduler: VoicePlaybackScheduler | None = None,
    ) -> None:
        root = Path(voice_pack_root)
        valid_volume = (
            not isinstance(settings.volume, bool)
            and isinstance(settings.volume, Real)
            and isfinite(float(settings.volume))
            and 0 <= float(settings.volume) <= 1
        )
        if (
            type(settings) is not VoiceSettings
            or type(settings.enabled) is not bool
            or settings.locale not in ("uk", "en")
            or not valid_volume
            or type(settings.announce_restored) is not bool
        ):
            raise ValueError("voice settings are invalid")
        if not root.is_dir():
            raise ValueError("voice pack root must be an existing directory")
        if not callable(getattr(backend, "play", None)):
            raise ValueError("audio backend must provide callable play")
        if scheduler is not None and not all(
            callable(getattr(scheduler, method, None)) for method in ("submit", "submit_clip")
        ):
            raise ValueError("scheduler must provide callable submit and submit_clip")
        self._root = root
        self._backend = backend
        self.settings = settings
        self._scheduler = scheduler

    def _play(self, clip_name: str) -> None:
        path = self._root / self.settings.locale / f"{clip_name}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"voice clip not found: {path}")
        if self._scheduler is not None:
            self._scheduler.submit_clip(path, float(self.settings.volume))
            return
        self._backend.play(path, float(self.settings.volume))

    def _play_notification(self, clip_name: str) -> None:
        bell = self._root / "notification_bell.wav"
        if not bell.is_file():
            raise FileNotFoundError(f"notification bell not found: {bell}")
        phrase = self._root / self.settings.locale / f"{clip_name}.wav"
        if not phrase.is_file():
            raise FileNotFoundError(f"voice clip not found: {phrase}")
        volume = float(self.settings.volume)
        if self._scheduler is not None:
            self._scheduler.submit(bell, phrase, volume)
            return
        self._backend.play(bell, volume)
        self._backend.play(phrase, volume)

    def send(self, event: NotificationEvent) -> None:
        if not self.settings.enabled:
            return
        if (
            not self.settings.announce_restored
            and event.event_type
            in (EventType.CAPTURE_RESTORED, EventType.MINIMAP_VISIBILITY_RESTORED)
        ):
            return
        self._play_notification(event.event_type.value)

    def test_voice(self) -> None:
        if self.settings.enabled:
            self._play("test")
