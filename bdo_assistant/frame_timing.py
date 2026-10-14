"""Timestamped frame metadata and elapsed-time analysis cadence.

``source_sequence`` is the capture callback counter, not a proven native GPU
frame identifier.  ``processing_ordinal`` is the ordinal of a frame selected
for analysis.  Legacy callers that only have an ordinal can use
:func:`synthetic_received_monotonic`; RuntimeBridge always supplies a real
monotonic receive time instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_non_negative_time(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


@dataclass(frozen=True)
class FrameEnvelope:
    """Metadata carried with one selected analysis frame."""

    source_sequence: int
    processing_ordinal: int
    received_monotonic: float
    calibration_epoch: int
    native_source_timestamp: float | None = None

    def __post_init__(self) -> None:
        _non_negative_int(self.source_sequence, "source_sequence")
        _non_negative_int(self.processing_ordinal, "processing_ordinal")
        _finite_non_negative_time(self.received_monotonic, "received_monotonic")
        _non_negative_int(self.calibration_epoch, "calibration_epoch")
        if self.native_source_timestamp is not None:
            _finite_non_negative_time(self.native_source_timestamp, "native_source_timestamp")


def synthetic_received_monotonic(processing_ordinal: int) -> float:
    """Compatibility time for legacy sequence-only callers; it is not wall time."""
    return float(_non_negative_int(processing_ordinal, "processing_ordinal"))


class AnalysisCadence:
    """Select real received-time samples without interpolation or duplication."""

    def __init__(self, *, interval_seconds: float, discontinuity_seconds: float) -> None:
        self.interval_seconds = _finite_non_negative_time(interval_seconds, "interval_seconds")
        self.discontinuity_seconds = _finite_non_negative_time(
            discontinuity_seconds, "discontinuity_seconds"
        )
        if self.interval_seconds <= 0 or self.discontinuity_seconds <= 0:
            raise ValueError("cadence intervals must be positive")
        if self.discontinuity_seconds < self.interval_seconds:
            raise ValueError("discontinuity_seconds must be at least interval_seconds")
        self._last_received_monotonic: float | None = None
        self._last_selected_monotonic: float | None = None
        self.reset_required = False

    def accept(self, envelope: FrameEnvelope) -> bool:
        if type(envelope) is not FrameEnvelope:
            raise ValueError("envelope must be FrameEnvelope")
        if self.reset_required:
            return False
        received = envelope.received_monotonic
        if self._last_received_monotonic is not None:
            delta = received - self._last_received_monotonic
            if delta <= 0 or delta > self.discontinuity_seconds:
                self.reset_required = True
                return False
        self._last_received_monotonic = received
        if self._last_selected_monotonic is None or received - self._last_selected_monotonic >= self.interval_seconds:
            self._last_selected_monotonic = received
            return True
        return False

    def reset(self) -> None:
        self._last_received_monotonic = None
        self._last_selected_monotonic = None
        self.reset_required = False
