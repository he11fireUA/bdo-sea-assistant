"""Duration-based classification of verified minimap texture motion."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite

from bdo_assistant.motion import MotionClass
from bdo_assistant.texture_motion import TextureMotionEvidence


@dataclass(frozen=True)
class MotionDecision:
    """A conservative temporal motion result from verified texture evidence."""

    motion_class: MotionClass
    reason: str
    confirmed_at: float | None = None
    onset_at: float | None = None


@dataclass(frozen=True)
class _Sample:
    timestamp: float
    evidence: TextureMotionEvidence


class CandidateMotionBridge:
    """One-sample-lag candidate projection for a verified micro-invalid gap.

    A bridge is emitted only after trusted ``MOVING`` decisions exist on both
    sides of one invalid texture observation, with each adjacent interval no
    longer than ``maximum_gap_seconds``.  Missing and ambiguous evidence is
    never bridged.  The one-sample lag is deliberate: a live causal decision
    cannot know that its right-hand trusted neighbour will arrive.
    """

    def __init__(self, *, maximum_gap_seconds: float = 0.25) -> None:
        if not isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
            raise ValueError("maximum_gap_seconds must be positive and finite")
        self.maximum_gap_seconds = maximum_gap_seconds
        self._prior: tuple[float, MotionDecision] | None = None
        self._pending: tuple[float, MotionDecision] | None = None
        self._last_timestamp: float | None = None

    def update(self, timestamp: float, decision: MotionDecision) -> MotionDecision:
        if not isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must increase")
        if not isinstance(decision, MotionDecision):
            raise ValueError("decision must be MotionDecision")
        self._last_timestamp = timestamp
        if self._pending is None:
            self._pending = (timestamp, decision)
            return MotionDecision(MotionClass.UNKNOWN, "awaiting_adjacent_texture")

        pending_timestamp, pending = self._pending
        projected = pending
        if self._may_bridge(self._prior, (pending_timestamp, pending), (timestamp, decision)):
            projected = MotionDecision(
                MotionClass.MOVING,
                "bridged_micro_unknown_between_verified_motion",
                timestamp,
                pending_timestamp,
            )
        self._prior = self._pending
        self._pending = (timestamp, decision)
        return projected

    def _may_bridge(
        self,
        prior: tuple[float, MotionDecision] | None,
        pending: tuple[float, MotionDecision],
        following: tuple[float, MotionDecision],
    ) -> bool:
        if prior is None:
            return False
        prior_timestamp, prior_decision = prior
        pending_timestamp, pending_decision = pending
        following_timestamp, following_decision = following
        return (
            prior_decision.motion_class is MotionClass.MOVING
            and prior_decision.reason == "sustained_verified_texture"
            and pending_decision.motion_class is MotionClass.UNKNOWN
            and pending_decision.reason == "invalid_texture_evidence"
            and following_decision.motion_class is MotionClass.MOVING
            and following_decision.reason == "sustained_verified_texture"
            and pending_timestamp - prior_timestamp <= self.maximum_gap_seconds
            and following_timestamp - pending_timestamp <= self.maximum_gap_seconds
        )


class TemporalMotionTracker:
    """Classify a bounded time window without treating missing evidence as quiet."""

    def __init__(
        self,
        *,
        window_duration: float = 2.0,
        minimum_duration: float = 0.75,
        minimum_independent_samples: int = 3,
        minimum_coverage: float = 0.60,
        moving_speed: float = 0.75,
        stationary_speed: float = 0.12,
        bounded_diameter: float = 0.5,
        minimum_bounded_path: float = 0.35,
    ) -> None:
        values = (window_duration, minimum_duration, minimum_coverage, moving_speed, stationary_speed, bounded_diameter, minimum_bounded_path)
        if not all(isfinite(value) for value in values):
            raise ValueError("temporal motion thresholds must be finite")
        if window_duration <= 0 or minimum_duration <= 0 or minimum_duration > window_duration:
            raise ValueError("duration thresholds are out of range")
        if type(minimum_independent_samples) is not int or minimum_independent_samples < 3:
            raise ValueError("minimum_independent_samples must be an integer of at least 3")
        if not 0 < minimum_coverage <= 1 or min(moving_speed, stationary_speed, bounded_diameter, minimum_bounded_path) < 0:
            raise ValueError("temporal motion thresholds are out of range")
        self.window_duration = window_duration
        self.minimum_duration = minimum_duration
        self.minimum_independent_samples = minimum_independent_samples
        self.minimum_coverage = minimum_coverage
        self.moving_speed = moving_speed
        self.stationary_speed = stationary_speed
        self.bounded_diameter = bounded_diameter
        self.minimum_bounded_path = minimum_bounded_path
        self._samples: deque[_Sample] = deque()
        self._last_timestamp: float | None = None

    def update(
        self,
        timestamp: float,
        evidence: TextureMotionEvidence | None,
        *,
        activity_veto: bool = False,
    ) -> MotionDecision:
        if not isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must increase")
        if evidence is not None and not isinstance(evidence, TextureMotionEvidence):
            raise ValueError("evidence must be TextureMotionEvidence or None")
        self._last_timestamp = timestamp
        if evidence is not None:
            self._samples.append(_Sample(timestamp, evidence))
        self._discard_before(timestamp - self.window_duration)
        if activity_veto:
            return MotionDecision(MotionClass.UNKNOWN, "activity_veto")
        if evidence is None:
            return MotionDecision(MotionClass.UNKNOWN, "missing_texture_evidence")
        if not evidence.valid or evidence.dt <= 0:
            return MotionDecision(MotionClass.UNKNOWN, "invalid_texture_evidence")
        return self._classify(timestamp)

    def _discard_before(self, cutoff: float) -> None:
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def _classify(self, timestamp: float) -> MotionDecision:
        if not self._samples:
            return MotionDecision(MotionClass.UNKNOWN, "no_texture_evidence")
        accepted = [sample for sample in self._samples if sample.evidence.valid and sample.evidence.dt > 0]
        if len(accepted) < self.minimum_independent_samples:
            return MotionDecision(MotionClass.UNKNOWN, "insufficient_verified_texture")
        accepted_duration = sum(sample.evidence.dt for sample in accepted)
        observed_duration = timestamp - accepted[0].timestamp + accepted[0].evidence.dt
        if accepted_duration < self.minimum_duration or accepted_duration / observed_duration < self.minimum_coverage:
            return MotionDecision(MotionClass.UNKNOWN, "insufficient_verified_coverage")
        path = sum(hypot(sample.evidence.dx, sample.evidence.dy) for sample in accepted)
        positions = [(0.0, 0.0)]
        x = y = 0.0
        for sample in accepted:
            x += sample.evidence.dx
            y += sample.evidence.dy
            positions.append((x, y))
        diameter = max(hypot(ax - bx, ay - by) for ax, ay in positions for bx, by in positions)
        moving_duration = sum(
            sample.evidence.dt
            for sample in accepted
            if hypot(sample.evidence.dx, sample.evidence.dy) / sample.evidence.dt >= self.moving_speed
        )
        if (
            moving_duration >= self.minimum_duration
            and path / accepted_duration >= self.moving_speed
            and self._has_local_coherence(accepted)
        ):
            return MotionDecision(MotionClass.MOVING, "sustained_verified_texture", timestamp, accepted[0].timestamp)
        noise_upper_bound = max(
            self.stationary_speed * sample.evidence.dt + sample.evidence.residual
            for sample in accepted
        )
        if all(hypot(sample.evidence.dx, sample.evidence.dy) <= noise_upper_bound for sample in accepted):
            return MotionDecision(MotionClass.STATIONARY, "verified_texture_within_noise", timestamp, accepted[0].timestamp)
        if (
            diameter <= self.bounded_diameter
            and path >= self.minimum_bounded_path
            and self._has_direction_reversal(accepted)
        ):
            return MotionDecision(MotionClass.JITTER, "bounded_verified_oscillation", timestamp, accepted[0].timestamp)
        return MotionDecision(MotionClass.UNKNOWN, "unbounded_or_incoherent_texture")

    @staticmethod
    def _has_direction_reversal(samples: list[_Sample]) -> bool:
        vectors = [(sample.evidence.dx, sample.evidence.dy) for sample in samples]
        return any(ax * bx + ay * by < 0 for ax, ay in vectors for bx, by in vectors)

    @staticmethod
    def _has_local_coherence(samples: list[_Sample]) -> bool:
        """Require a three-sample local directional run, allowing smooth turns."""
        for first, second, third in zip(samples, samples[1:], samples[2:]):
            vectors = (first.evidence, second.evidence, third.evidence)
            path = sum(hypot(item.dx, item.dy) for item in vectors)
            if path and hypot(sum(item.dx for item in vectors), sum(item.dy for item in vectors)) / path >= 0.5:
                return True
        return False
