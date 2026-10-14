import threading

import numpy as np

from bdo_assistant.monitoring import MonitoringSample
from bdo_assistant.shadow_pipeline import ShadowPipelineRunner


def _sample(sequence: int) -> MonitoringSample:
    return MonitoringSample(
        sequence=sequence,
        capture_ok=True,
        calibration_ok=True,
        minimap_roi=np.zeros((20, 30, 3), dtype=np.uint8),
        source_sequence=sequence,
        processing_ordinal=sequence,
        received_monotonic=float(sequence),
    )


def test_shadow_reset_waits_for_inflight_processing_before_resetting_processor() -> None:
    processing_entered = threading.Event()
    release_processing = threading.Event()
    reset_called = threading.Event()
    events = []

    class StatefulProcessor:
        def __init__(self) -> None:
            self.previous = None

        def process(self, sample):
            processing_entered.set()
            assert release_processing.wait(1)
            self.previous = sample.sequence
            events.append(("process", sample.sequence))
            return {"reason": "processed"}

        def reset(self):
            events.append(("reset", self.previous))
            self.previous = None
            reset_called.set()

    processor = StatefulProcessor()
    runner = ShadowPipelineRunner(processor)
    assert runner.submit(_sample(1))
    assert processing_entered.wait(1)

    resetter = threading.Thread(target=runner.reset)
    resetter.start()
    assert not reset_called.wait(0.05)
    release_processing.set()
    resetter.join(1)

    assert not resetter.is_alive()
    assert events == [("process", 1), ("reset", 1)]
    assert processor.previous is None