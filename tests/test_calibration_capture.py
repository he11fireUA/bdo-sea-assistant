import numpy as np
import pytest

from bdo_assistant.calibration_capture import OneShotCalibrationCapture
from bdo_assistant.window_discovery import WindowCandidate


CANDIDATE = WindowCandidate(123, 456, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)


class ImmediateSource:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.stop_calls = 0

    def start(self, on_frame, on_closed):
        on_frame(np.full((20, 30, 3), 7, dtype=np.uint8))

    def stop(self):
        self.stop_calls += 1


class SilentSource(ImmediateSource):
    def start(self, on_frame, on_closed):
        pass


class FailingStartSource(ImmediateSource):
    def start(self, on_frame, on_closed):
        raise RuntimeError("capture startup failed")


def test_one_shot_capture_returns_read_only_frame_and_stops_source() -> None:
    sources = []

    def factory(hwnd):
        source = ImmediateSource(hwnd)
        sources.append(source)
        return source

    frame = OneShotCalibrationCapture(source_factory=factory, timeout_seconds=0.1).capture(CANDIDATE)

    assert sources[0].hwnd == 123
    assert sources[0].stop_calls == 1
    assert frame.shape == (20, 30, 3)
    assert frame.dtype == np.uint8
    assert not frame.flags.writeable


def test_one_shot_capture_times_out_and_still_stops_source() -> None:
    sources = []

    def factory(hwnd):
        source = SilentSource(hwnd)
        sources.append(source)
        return source

    with pytest.raises(TimeoutError, match="calibration frame"):
        OneShotCalibrationCapture(source_factory=factory, timeout_seconds=0.01).capture(CANDIDATE)

    assert sources[0].stop_calls == 1


def test_one_shot_capture_stops_source_when_start_raises() -> None:
    source = FailingStartSource(CANDIDATE.hwnd)
    capture = OneShotCalibrationCapture(
        source_factory=lambda hwnd: source,
        timeout_seconds=0.1,
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        capture.capture(CANDIDATE)

    assert source.stop_calls == 1
