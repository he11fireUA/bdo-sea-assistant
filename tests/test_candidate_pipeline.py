from __future__ import annotations

import numpy as np

from bdo_assistant.candidate_pipeline import CandidateMotionAuthorityLoop, CandidateTextureMotionPipeline
from bdo_assistant.monitoring import MonitoringSample
from bdo_assistant.motion import MotionClass
from bdo_assistant.navigation_state import NavigationObservation, NavigationState
from bdo_assistant.texture_motion import TextureMotionEvidence, TextureMotionReason


def _sample(timestamp: float) -> MonitoringSample:
    return MonitoringSample(
        sequence=round(timestamp * 10),
        capture_ok=True,
        calibration_ok=True,
        minimap_roi=np.zeros((20, 30, 3), dtype=np.uint8),
        source_sequence=round(timestamp * 10),
        processing_ordinal=round(timestamp * 10),
        received_monotonic=timestamp,
    )


def _valid() -> TextureMotionEvidence:
    return TextureMotionEvidence(0.24, 0.0, 0.2, 0.02, 20, 18, 5, True, TextureMotionReason.OK)


def _invalid() -> TextureMotionEvidence:
    return TextureMotionEvidence(0.0, 0.0, 0.2, 0.0, 0, 0, 0, False, TextureMotionReason.INSUFFICIENT_TEXTURE)


def test_candidate_pipeline_bridges_only_a_one_sample_invalid_gap_between_verified_motion(monkeypatch) -> None:
    evidence = iter((_valid(), _valid(), _valid(), _valid(), _invalid(), _valid()))
    monkeypatch.setattr("bdo_assistant.candidate_pipeline.estimate_texture_motion", lambda *args, **kwargs: next(evidence))
    pipeline = CandidateTextureMotionPipeline()

    results = [pipeline.process(_sample(timestamp)) for timestamp in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)]

    assert results[-1]["motion"] == "moving"
    assert results[-1]["reason"] == "bridged_micro_unknown_between_verified_motion"
    assert results[-1]["valid"] is True


class _Pipeline:
    def __init__(self, motion: str) -> None:
        self.motion = motion

    def process(self, sample: MonitoringSample) -> dict[str, str]:
        return {"motion": self.motion}

    def reset(self) -> None:
        pass


class _Loop:
    minimap = object()

    def __init__(self) -> None:
        self.sample: MonitoringSample | None = None

    def process(self, sample: MonitoringSample):
        self.sample = sample
        return type("Result", (), {"navigation_state": NavigationState.MOVING})()

    def capture_closed(self, sequence: int):
        return type("Result", (), {"navigation_state": NavigationState.UNKNOWN})()


def test_candidate_authority_loop_promotes_only_verified_moving_and_fail_closes_terminals() -> None:
    legacy = _Loop()
    sample = MonitoringSample(
        sequence=1, capture_ok=True, calibration_ok=True,
        minimap_roi=np.zeros((20, 30, 3), dtype=np.uint8),
        navigation_observation=NavigationObservation(MotionClass.STATIONARY, True, False),
    )

    CandidateMotionAuthorityLoop(legacy, motion_pipeline=_Pipeline("stationary")).process(sample)

    assert legacy.sample is not None
    assert legacy.sample.navigation_observation == NavigationObservation(MotionClass.UNKNOWN, True, False)
