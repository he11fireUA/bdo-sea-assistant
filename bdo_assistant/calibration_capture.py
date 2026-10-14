from __future__ import annotations

from math import isfinite
from numbers import Real
import threading
from typing import Callable

import numpy as np

from .runtime import WgcWindowFrameSource
from .window_discovery import WindowCandidate


class OneShotCalibrationCapture:
    def __init__(
        self,
        *,
        source_factory: Callable[[int], object] = WgcWindowFrameSource,
        timeout_seconds: float = 3.0,
    ) -> None:
        if not callable(source_factory):
            raise ValueError("source_factory must be callable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        self._source_factory = source_factory
        self.timeout_seconds = float(timeout_seconds)

    def capture(self, candidate: WindowCandidate) -> np.ndarray:
        if type(candidate) is not WindowCandidate:
            raise ValueError("candidate must be WindowCandidate")
        source = self._source_factory(candidate.hwnd)
        if not callable(getattr(source, "start", None)) or not callable(
            getattr(source, "stop", None)
        ):
            raise ValueError("source must provide start and stop")
        received = threading.Event()
        closed = threading.Event()
        frame_holder: list[np.ndarray] = []

        def on_frame(frame: np.ndarray) -> None:
            if frame_holder:
                return
            if (
                not isinstance(frame, np.ndarray)
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.size == 0
                or frame.dtype != np.uint8
            ):
                return
            copy = frame.copy()
            copy.setflags(write=False)
            frame_holder.append(copy)
            received.set()

        try:
            source.start(on_frame, closed.set)
            if not received.wait(self.timeout_seconds):
                reason = "capture closed" if closed.is_set() else "timeout"
                raise TimeoutError(f"calibration frame {reason}")
        finally:
            source.stop()
        return frame_holder[0]
