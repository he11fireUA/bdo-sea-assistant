"""Capture callback freshness, silence detection, and explicit uncertainty.

A crop hash is only a diagnostic: identical minimap pixels can be a legitimate
still scene. Freshness is established by a delivered callback or validated native
source timestamp. Missing/duplicate native time cues are uncertain and must not be used as stationary evidence. A live callback
stream cannot prove that a renderer is not frozen; that limitation remains
explicit uncertainty rather than an automatic STUCK verdict.
"""
from __future__ import annotations

from enum import Enum
from math import isfinite
from numbers import Real
from threading import RLock, Timer
from time import monotonic
from typing import Callable, Protocol


class CaptureFreshness(str, Enum):
    FRESH = "fresh"
    UNCERTAIN = "uncertain"
    SILENT = "silent"


class _Timer(Protocol):
    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], _Timer]


def _default_timer(delay: float, callback: Callable[[], None]) -> Timer:
    timer = Timer(delay, callback)
    timer.daemon = True
    return timer


def _time(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


class CaptureFreshnessWatchdog:
    """Owner-scoped silence watchdog with stale-generation-safe timer callbacks."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        clock: Callable[[], float] = monotonic,
        timer_factory: TimerFactory = _default_timer,
    ) -> None:
        self.timeout_seconds = _time(timeout_seconds, "timeout_seconds")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not callable(clock) or not callable(timer_factory):
            raise ValueError("clock and timer_factory must be callable")
        self._clock = clock
        self._timer_factory = timer_factory
        self._lock = RLock()
        self._generation = 0
        self._active = False
        self._last_callback: float | None = None
        self._last_native_timestamp: float | None = None
        self._timer: _Timer | None = None
        self._on_silence: Callable[[CaptureFreshness], None] | None = None
        self._silence_reported = False
        self.freshness = CaptureFreshness.UNCERTAIN
        self.identical_crop_count = 0
        self._last_crop_hash: object | None = None

    def start(self, on_silence: Callable[[CaptureFreshness], None]) -> int:
        if not callable(on_silence):
            raise ValueError("on_silence must be callable")
        with self._lock:
            self._cancel_timer_locked()
            self._generation += 1
            self._active = True
            self._last_callback = _time(self._clock(), "clock result")
            self._last_native_timestamp = None
            self._on_silence = on_silence
            self._silence_reported = False
            self.freshness = CaptureFreshness.UNCERTAIN
            self.identical_crop_count = 0
            self._last_crop_hash = None
            self._schedule_locked(self._generation)
            return self._generation

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._active = False
            self._on_silence = None
            self._cancel_timer_locked()

    def frame_arrived(
        self,
        generation: int,
        *,
        source_timestamp: float | None = None,
        crop_hash: object | None = None,
    ) -> CaptureFreshness:
        if type(generation) is not int or generation < 1:
            raise ValueError("generation must be a positive integer")
        native = None if source_timestamp is None else _time(source_timestamp, "source_timestamp")
        with self._lock:
            if generation != self._generation or not self._active:
                return self.freshness
            now = _time(self._clock(), "clock result")
            native_advanced = native is not None and (
                self._last_native_timestamp is None or native > self._last_native_timestamp
            )
            if crop_hash is not None and crop_hash == self._last_crop_hash:
                self.identical_crop_count += 1
            elif crop_hash is not None:
                self._last_crop_hash = crop_hash
            self._last_callback = now
            if native is not None:
                self._last_native_timestamp = native
            self._silence_reported = False
            # A delivered callback is an independent freshness cue when native
            # metadata is unavailable.  Equal image data never changes this.
            self.freshness = (
                CaptureFreshness.FRESH
                if native_advanced or native is None
                else CaptureFreshness.UNCERTAIN
            )
            self._cancel_timer_locked()
            self._schedule_locked(generation)
            return self.freshness

    def _schedule_locked(self, generation: int) -> None:
        timer = self._timer_factory(self.timeout_seconds, lambda: self._timer_fired(generation))
        self._timer = timer
        start = getattr(timer, "start", None)
        if callable(start):
            start()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _timer_fired(self, generation: int) -> None:
        callback: Callable[[CaptureFreshness], None] | None = None
        with self._lock:
            if generation != self._generation or not self._active or self._last_callback is None:
                return
            now = _time(self._clock(), "clock result")
            if now - self._last_callback < self.timeout_seconds:
                self._schedule_locked(generation)
                return
            if self._silence_reported:
                return
            self._silence_reported = True
            self.freshness = CaptureFreshness.SILENT
            callback = self._on_silence
        if callback is not None:
            callback(CaptureFreshness.SILENT)
