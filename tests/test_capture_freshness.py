from dataclasses import dataclass

from bdo_assistant.capture_freshness import CaptureFreshness, CaptureFreshnessWatchdog


@dataclass
class FakeTimer:
    callback: object
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


class FakeTimerFactory:
    def __init__(self) -> None:
        self.timers: list[FakeTimer] = []

    def __call__(self, _delay: float, callback: object) -> FakeTimer:
        timer = FakeTimer(callback)
        self.timers.append(timer)
        return timer


def test_watchdog_reports_silence_without_waiting_for_another_frame() -> None:
    now = [0.0]
    factory = FakeTimerFactory()
    reported = []
    watchdog = CaptureFreshnessWatchdog(
        timeout_seconds=2.0,
        clock=lambda: now[0],
        timer_factory=factory,
    )

    generation = watchdog.start(reported.append)
    now[0] = 2.0
    factory.timers[-1].fire()

    assert reported == [CaptureFreshness.SILENT]
    assert watchdog.freshness is CaptureFreshness.SILENT


def test_old_generation_timer_is_inert_after_restart() -> None:
    now = [0.0]
    factory = FakeTimerFactory()
    first, second = [], []
    watchdog = CaptureFreshnessWatchdog(
        timeout_seconds=2.0,
        clock=lambda: now[0],
        timer_factory=factory,
    )

    first_generation = watchdog.start(first.append)
    old_timer = factory.timers[-1]
    second_generation = watchdog.start(second.append)
    now[0] = 2.0
    old_timer.callback()
    factory.timers[-1].fire()

    assert first_generation != second_generation
    assert first == []
    assert second == [CaptureFreshness.SILENT]


def test_duplicate_source_timestamp_is_uncertain_not_fresh_evidence() -> None:
    now = [0.0]
    watchdog = CaptureFreshnessWatchdog(timeout_seconds=2.0, clock=lambda: now[0])
    generation = watchdog.start(lambda freshness: None)

    assert watchdog.frame_arrived(generation, source_timestamp=10.0) is CaptureFreshness.FRESH
    now[0] = 0.5
    assert watchdog.frame_arrived(generation, source_timestamp=10.0) is CaptureFreshness.UNCERTAIN


def test_identical_pixel_diagnostic_does_not_degrade_independently_fresh_frames() -> None:
    now = [0.0]
    watchdog = CaptureFreshnessWatchdog(timeout_seconds=2.0, clock=lambda: now[0])
    generation = watchdog.start(lambda freshness: None)

    assert watchdog.frame_arrived(generation, source_timestamp=10.0, crop_hash="same") is CaptureFreshness.FRESH
    now[0] = 0.5
    assert watchdog.frame_arrived(generation, source_timestamp=11.0, crop_hash="same") is CaptureFreshness.FRESH
    assert watchdog.identical_crop_count == 1
