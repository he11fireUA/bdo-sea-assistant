import math

import pytest

from bdo_assistant.frame_timing import AnalysisCadence, FrameEnvelope


def _envelope(sequence: int, received_monotonic: float) -> FrameEnvelope:
    return FrameEnvelope(
        source_sequence=sequence,
        processing_ordinal=sequence,
        received_monotonic=received_monotonic,
        calibration_epoch=1,
    )


def test_cadence_selects_the_same_elapsed_time_windows_at_different_callback_rates() -> None:
    slow = AnalysisCadence(interval_seconds=1.0, discontinuity_seconds=2.0)
    fast = AnalysisCadence(interval_seconds=1.0, discontinuity_seconds=2.0)

    slow_selected = [
        item.received_monotonic
        for item in (_envelope(index, time) for index, time in enumerate((0.0, 0.5, 1.0, 1.5, 2.0), 1))
        if slow.accept(item)
    ]
    fast_selected = [
        item.received_monotonic
        for item in (_envelope(index, index / 4) for index in range(9))
        if fast.accept(item)
    ]

    assert slow_selected == [0.0, 1.0, 2.0]
    assert fast_selected == [0.0, 1.0, 2.0]


def test_cadence_does_not_treat_sampled_source_sequence_stride_as_a_gap() -> None:
    cadence = AnalysisCadence(interval_seconds=0.5, discontinuity_seconds=1.0)

    accepted = [
        cadence.accept(_envelope(sequence, time))
        for sequence, time in ((1, 0.0), (7, 0.5), (13, 1.0))
    ]

    assert accepted == [True, True, True]
    assert cadence.reset_required is False


def test_timestamp_discontinuity_requires_reset_before_more_analysis() -> None:
    cadence = AnalysisCadence(interval_seconds=0.5, discontinuity_seconds=1.0)

    assert cadence.accept(_envelope(1, 0.0)) is True
    assert cadence.accept(_envelope(7, 1.5)) is False
    assert cadence.reset_required is True
    assert cadence.accept(_envelope(13, 2.0)) is False


@pytest.mark.parametrize("time", [0.0, 0.25])
def test_duplicate_or_out_of_order_timestamps_fail_closed(time: float) -> None:
    cadence = AnalysisCadence(interval_seconds=0.5, discontinuity_seconds=1.0)
    assert cadence.accept(_envelope(1, 0.5)) is True

    assert cadence.accept(_envelope(7, time)) is False
    assert cadence.reset_required is True


@pytest.mark.parametrize("received_monotonic", [math.nan, math.inf, -0.1])
def test_envelope_rejects_invalid_received_monotonic_time(received_monotonic: float) -> None:
    with pytest.raises(ValueError, match="received_monotonic"):
        _envelope(1, received_monotonic)
