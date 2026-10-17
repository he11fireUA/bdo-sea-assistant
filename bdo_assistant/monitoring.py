from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

import numpy as np

from .minimap_observability import MinimapVisibilityTracker
from .frame_timing import FrameEnvelope, synthetic_received_monotonic
from .navigation_observability import MinimapNavigationMonitor
from .navigation_state import NavigationObservation, NavigationState, NavigationStateMachine
from .notifications import EventType, NotificationEvent, Severity
from .resource_bars import ThresholdWarningTracker


@dataclass(frozen=True)
class MonitoringSample:
    sequence: int
    capture_ok: bool
    calibration_ok: bool
    minimap_roi: np.ndarray | None = None
    navigation_observation: NavigationObservation | None = None
    hp_percent: float | None = None
    rations_percent: float | None = None
    resource_bars_visible: bool | None = None
    source_sequence: int | None = None
    processing_ordinal: int | None = None
    received_monotonic: float | None = None
    native_source_timestamp: float | None = None
    calibration_epoch: int = 0

    def __post_init__(self) -> None:
        """Keep sequence-only callers working with clearly synthetic time."""
        source_sequence = self.sequence if self.source_sequence is None else self.source_sequence
        processing_ordinal = self.sequence if self.processing_ordinal is None else self.processing_ordinal
        received_monotonic = (
            synthetic_received_monotonic(processing_ordinal)
            if self.received_monotonic is None
            else self.received_monotonic
        )
        FrameEnvelope(
            source_sequence=source_sequence,
            processing_ordinal=processing_ordinal,
            received_monotonic=received_monotonic,
            native_source_timestamp=self.native_source_timestamp,
            calibration_epoch=self.calibration_epoch,
        )
        object.__setattr__(self, "source_sequence", source_sequence)
        object.__setattr__(self, "processing_ordinal", processing_ordinal)
        object.__setattr__(self, "received_monotonic", float(received_monotonic))


@dataclass(frozen=True)
class MonitoringResult:
    sequence: int
    events: tuple[NotificationEvent, ...]
    navigation_state: NavigationState
    diagnostics: tuple[str, ...] = ()
    minimap_visible: bool | None = None


class _CaptureState(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    LOST = "lost"


class _CaptureHealthTracker:
    def __init__(self, *, lost_frames: int, restored_frames: int) -> None:
        if any(type(value) is not int or value < 2 for value in (lost_frames, restored_frames)):
            raise ValueError("capture confirmation thresholds must be integers of at least 2")
        self.lost_frames = lost_frames
        self.restored_frames = restored_frames
        self.state = _CaptureState.UNKNOWN
        self._lost_count = 0
        self._restored_count = 0

    @property
    def enabled(self) -> bool:
        return self.state is _CaptureState.HEALTHY

    def update(self, capture_ok: bool) -> EventType | None:
        if type(capture_ok) is not bool:
            raise ValueError("capture_ok must be boolean")
        if capture_ok:
            self._lost_count = 0
            self._restored_count += 1
            if self.state is _CaptureState.LOST and self._restored_count >= self.restored_frames:
                self.state = _CaptureState.HEALTHY
                self._restored_count = 0
                return EventType.CAPTURE_RESTORED
            if self.state is _CaptureState.UNKNOWN and self._restored_count >= self.restored_frames:
                self.state = _CaptureState.HEALTHY
                self._restored_count = 0
            return None
        self._restored_count = 0
        self._lost_count += 1
        if self._lost_count >= self.lost_frames and self.state is not _CaptureState.LOST:
            self.state = _CaptureState.LOST
            self._lost_count = 0
            return EventType.CAPTURE_LOST
        return None

    def force_lost(self) -> EventType | None:
        already_lost = self.state is _CaptureState.LOST
        self.state = _CaptureState.LOST
        self._lost_count = 0
        self._restored_count = 0
        return None if already_lost else EventType.CAPTURE_LOST


class _CalibrationHealthTracker:
    def __init__(self, *, lost_frames: int = 2) -> None:
        if type(lost_frames) is not int or lost_frames < 2:
            raise ValueError("calibration loss threshold must be an integer of at least 2")
        self.lost_frames = lost_frames
        self._invalid_count = 0
        self._lost = False

    def reset_pending(self) -> None:
        self._invalid_count = 0

    def update(self, calibration_ok: bool) -> EventType | None:
        if type(calibration_ok) is not bool:
            raise ValueError("calibration_ok must be boolean")
        if calibration_ok:
            self._invalid_count = 0
            self._lost = False
            return None
        self._invalid_count += 1
        if self._invalid_count >= self.lost_frames and not self._lost:
            self._lost = True
            self._invalid_count = 0
            return EventType.UI_CALIBRATION_LOST
        return None


class AlphaMonitoringLoop:
    def __init__(
        self,
        *,
        minimap: MinimapNavigationMonitor,
        capture_lost_frames: int = 2,
        capture_restored_frames: int = 2,
        repeat_alerts_enabled: bool = False,
        repeat_alert_interval_seconds: int = 60,
        hp_warning: ThresholdWarningTracker | None = None,
        rations_warning: ThresholdWarningTracker | None = None,
    ) -> None:
        if type(minimap) is not MinimapNavigationMonitor:
            raise ValueError("minimap must be MinimapNavigationMonitor")
        self.minimap = minimap
        self.capture = _CaptureHealthTracker(
            lost_frames=capture_lost_frames,
            restored_frames=capture_restored_frames,
        )
        self.calibration = _CalibrationHealthTracker()
        if type(repeat_alerts_enabled) is not bool:
            raise ValueError("repeat_alerts_enabled must be boolean")
        if type(repeat_alert_interval_seconds) is not int or not 30 <= repeat_alert_interval_seconds <= 300:
            raise ValueError("repeat_alert_interval_seconds must be from 30 to 300")
        self.repeat_alerts_enabled = repeat_alerts_enabled
        self.repeat_alert_interval_seconds = repeat_alert_interval_seconds
        self._repeat_last_emitted: dict[EventType, float] = {}
        if hp_warning is not None and type(hp_warning) is not ThresholdWarningTracker:
            raise ValueError("resource warning trackers have invalid types")
        if rations_warning is not None and type(rations_warning) is not ThresholdWarningTracker:
            raise ValueError("resource warning trackers have invalid types")
        self.hp_warning = hp_warning if hp_warning is not None else ThresholdWarningTracker(
            warning_below=60,
            reset_at=65,
            confirmation_frames=3,
        )
        self.rations_warning = (
            rations_warning
            if rations_warning is not None
            else ThresholdWarningTracker(
                warning_below=10,
                reset_at=15,
                confirmation_frames=3,
            )
        )
        self._last_sequence: int | None = None

        self._announced_navigation_states: set[NavigationState] = set()

    def configure_resource_warning_zones(
        self,
        ship_hp_warning_percent: int,
        ship_rations_warning_percent: int,
    ) -> None:
        for value in (ship_hp_warning_percent, ship_rations_warning_percent):
            if type(value) is not int or not 1 <= value <= 99:
                raise ValueError("resource warning percentages must be from 1 to 99")
        self.hp_warning = ThresholdWarningTracker(
            warning_below=ship_hp_warning_percent,
            reset_at=min(100, ship_hp_warning_percent + 5),
            confirmation_frames=self.hp_warning.confirmation_frames,
        )
        self.rations_warning = ThresholdWarningTracker(
            warning_below=ship_rations_warning_percent,
            reset_at=min(100, ship_rations_warning_percent + 5),
            confirmation_frames=self.rations_warning.confirmation_frames,
        )

    def configure_repeat_alerts(self, enabled: bool, interval_seconds: int) -> None:
        if type(enabled) is not bool:
            raise ValueError("repeat alerts enabled must be boolean")
        if type(interval_seconds) is not int or not 30 <= interval_seconds <= 300:
            raise ValueError("repeat alert interval must be from 30 to 300")
        self.repeat_alerts_enabled = enabled
        self.repeat_alert_interval_seconds = interval_seconds
        self._repeat_last_emitted.clear()

    def _suspend_navigation(self) -> NavigationState:
        self._repeat_last_emitted.clear()
        self._announced_navigation_states.clear()
        MinimapVisibilityTracker.reset_pending(self.minimap.gate.visibility)
        self.hp_warning.reset()
        self.rations_warning.reset()
        return NavigationStateMachine.suspend(self.minimap.gate.navigation)

    def _pause_for_no_ship(self) -> NavigationState:
        self._repeat_last_emitted.clear()
        self._announced_navigation_states.clear()
        MinimapVisibilityTracker.reset_pending(self.minimap.gate.visibility)
        self.hp_warning.reset()
        self.rations_warning.reset()
        return self.minimap.gate.navigation.no_ship()

    def _repeat_active_alerts(
        self,
        events: list[NotificationEvent],
        state: NavigationState,
        observed_monotonic: float,
    ) -> None:
        active = {
            EventType.ARRIVED: (
                state in (NavigationState.ARRIVED, NavigationState.DEPARTED),
                Severity.INFO,
            ),
            EventType.STUCK: (state is NavigationState.STUCK, Severity.WARNING),
            EventType.SHIP_HP_LOW: (self.hp_warning.alerted, Severity.CRITICAL),
            EventType.RATIONS_LOW: (self.rations_warning.alerted, Severity.WARNING),
        }
        emitted = {event.event_type for event in events}
        for event_type, (is_active, severity) in active.items():
            if not is_active:
                self._repeat_last_emitted.pop(event_type, None)
                continue
            last = self._repeat_last_emitted.get(event_type)
            if event_type not in emitted and self.repeat_alerts_enabled and last is not None and (
                observed_monotonic - last >= self.repeat_alert_interval_seconds
            ):
                events.append(NotificationEvent(event_type, severity))
                emitted.add(event_type)
            # Applying settings clears the schedule, not the active condition.
            # Start a fresh interval at the next valid sample without reannouncing.
            if event_type in emitted or (self.repeat_alerts_enabled and last is None):
                self._repeat_last_emitted[event_type] = observed_monotonic

    def process(self, sample: MonitoringSample) -> MonitoringResult:
        if type(sample) is not MonitoringSample:
            raise ValueError("sample must be MonitoringSample")
        if type(sample.sequence) is not int or sample.sequence < 0:
            raise ValueError("sample sequence must be a non-negative integer")
        if type(sample.capture_ok) is not bool or type(sample.calibration_ok) is not bool:
            raise ValueError("sample health flags must be boolean")
        if sample.resource_bars_visible is not None and type(sample.resource_bars_visible) is not bool:
            raise ValueError("resource bars visibility must be boolean or None")
        for percent in (sample.hp_percent, sample.rations_percent):
            if percent is not None and (
                isinstance(percent, bool)
                or not isinstance(percent, Real)
                or not isfinite(float(percent))
                or not 0 <= float(percent) <= 100
            ):
                raise ValueError("resource percentages must be finite numbers from 0 to 100")
        if self._last_sequence is not None and sample.sequence <= self._last_sequence:
            raise ValueError("sample sequence must strictly increase")
        self._last_sequence = sample.sequence
        capture_event = self.capture.update(sample.capture_ok)
        events: list[NotificationEvent] = []
        if capture_event is EventType.CAPTURE_LOST:
            events.append(NotificationEvent(capture_event, Severity.CRITICAL))
        elif capture_event is EventType.CAPTURE_RESTORED:
            events.append(NotificationEvent(capture_event, Severity.INFO))

        if not sample.capture_ok or (
            not self.capture.enabled and not self.minimap.gate.navigation.simple
        ):
            self.calibration.reset_pending()
            state = self._suspend_navigation()
            return MonitoringResult(sample.sequence, tuple(events), state)

        if sample.resource_bars_visible is False:
            self.calibration.reset_pending()
            state = self._pause_for_no_ship()
            return MonitoringResult(sample.sequence, (), state)

        calibration_input_ok = sample.calibration_ok and sample.minimap_roi is not None
        if not calibration_input_ok:
            calibration_event = self.calibration.update(False)
            if calibration_event is not None:
                events.append(NotificationEvent(calibration_event, Severity.WARNING))
            state = self._suspend_navigation()
            return MonitoringResult(sample.sequence, tuple(events), state)

        try:
            decision = self.minimap.update(
                sample.minimap_roi,
                sample.navigation_observation,
                observed_monotonic=sample.received_monotonic,
            )
        except Exception as exc:
            state = self._suspend_navigation()
            calibration_event = self.calibration.update(False)
            if calibration_event is not None:
                events.append(NotificationEvent(calibration_event, Severity.WARNING))

            diagnostic = f"minimap detector: {type(exc).__name__}: {exc}"
            return MonitoringResult(
                sample.sequence,
                tuple(events),
                state,
                (diagnostic,),
            )
        self.calibration.update(True)
        if not decision.evidence.is_visible:
            # The gate already suspends navigation; retain visibility debounce.
            self._announced_navigation_states.clear()
            self._repeat_last_emitted.clear()
        if decision.event is not None:
            severity = (
                Severity.WARNING
                if decision.event is EventType.MINIMAP_VISIBILITY_LOST
                else Severity.INFO
            )
            events.append(NotificationEvent(decision.event, severity))
        if (
            decision.navigation_state in (NavigationState.ARRIVED, NavigationState.DEPARTED)
            and NavigationState.DEPARTED not in self._announced_navigation_states
        ):
            events.append(NotificationEvent(EventType.ARRIVED, Severity.INFO))
            self._announced_navigation_states.add(NavigationState.DEPARTED)
        elif (
            decision.navigation_state is NavigationState.STUCK
            and NavigationState.STUCK not in self._announced_navigation_states
        ):
            events.append(NotificationEvent(EventType.STUCK, Severity.WARNING))
            self._announced_navigation_states.add(NavigationState.STUCK)
        if decision.navigation_state is NavigationState.MOVING:
            self._announced_navigation_states.clear()
        if not decision.evidence.is_visible or sample.resource_bars_visible is False:
            self.hp_warning.reset()
            self.rations_warning.reset()
        else:
            if sample.hp_percent is None:
                self.hp_warning.reset()
            elif self.hp_warning.update(sample.hp_percent):
                events.append(NotificationEvent(EventType.SHIP_HP_LOW, Severity.CRITICAL))
            if sample.rations_percent is None:
                self.rations_warning.reset()
            elif self.rations_warning.update(sample.rations_percent):
                events.append(NotificationEvent(EventType.RATIONS_LOW, Severity.WARNING))
        self._repeat_active_alerts(
            events,
            decision.navigation_state,
            sample.received_monotonic,
        )

        return MonitoringResult(
            sample.sequence,
            tuple(events),
            decision.navigation_state,
            minimap_visible=decision.evidence.is_visible,
        )

    def capture_closed(self, sequence: int) -> MonitoringResult:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sample sequence must be a non-negative integer")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ValueError("sample sequence must strictly increase")
        self._last_sequence = sequence
        event = self.capture.force_lost()
        self.calibration.reset_pending()
        state = self._suspend_navigation()

        events = (
            (NotificationEvent(event, Severity.CRITICAL),)
            if event is not None
            else ()
        )
        return MonitoringResult(sequence, events, state)
