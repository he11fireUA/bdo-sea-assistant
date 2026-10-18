from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
import threading
from time import monotonic
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import cv2

from .frame_timing import AnalysisCadence, FrameEnvelope
from .capture_freshness import CaptureFreshness, CaptureFreshnessWatchdog
from .monitoring import MonitoringResult, MonitoringSample
from .minimap_observability import _validate_bgr_uint8
from .motion import (
    ActivityMotionTracker,
    MotionClass,
    MotionVector,
    classify_motion_window,
    estimate_activity,
    estimate_translation,
)
from .navigation_state import NavigationObservation, NavigationState
from .notifications import DeliveryResult, EventType, NotificationEvent, Severity
from .route_evidence import RouteGeometryClass, RoutePresence, assess_route_geometry, assess_route_presence
from .resource_bars import ResourceKind, measure_resource_bar
from .shadow_pipeline import ShadowAccounting, ShadowMetrics, ShadowPipelineRunner, ShadowProcessor
from .texture_motion import estimate_texture_motion
from .ui_locator import Rect


class FrameSource(Protocol):
    def start(self, on_frame: Callable[[np.ndarray], None], on_closed: Callable[[], None]) -> None: ...

    def stop(self) -> None: ...


class WgcWindowFrameSource:
    def __init__(self, hwnd: int, *, capture_factory=None) -> None:
        if type(hwnd) is not int or hwnd <= 0:
            raise ValueError("hwnd must be a positive integer")
        if capture_factory is None:
            from windows_capture import WindowsCapture

            capture_factory = WindowsCapture
        if not callable(capture_factory):
            raise ValueError("capture_factory must be callable")
        self.hwnd = hwnd
        self._capture_factory = capture_factory
        self._capture = None
        self._control = None
        self._closed_notified = False
        self._client_close_notified = False
        self._generation = 0
        self._last_error = ""
        self._lifecycle_lock = threading.RLock()

    @property
    def last_error(self) -> str:
        return self._last_error

    def _record_error(self, error: Exception) -> None:
        text = f"{type(error).__name__}: {error}"
        self._last_error = f"{self._last_error}; {text}" if self._last_error else text

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._control is not None and not self._closed_notified

    def start(
        self,
        on_frame: Callable[[np.ndarray], None],
        on_closed: Callable[[], None],
    ) -> None:
        with self._lifecycle_lock:
            try:
                self._start_locked(on_frame, on_closed)
            except Exception:
                cleanup = getattr(self._capture, "stop", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except Exception as cleanup_error:
                        self._record_error(cleanup_error)
                self._generation += 1
                self._closed_notified = True
                self._control = None
                self._capture = None
                raise

    def _start_locked(
        self,
        on_frame: Callable[[np.ndarray], None],
        on_closed: Callable[[], None],
    ) -> None:
        if self._control is not None:
            raise RuntimeError("WGC source is already running")
        if not callable(on_frame) or not callable(on_closed):
            raise ValueError("WGC callbacks must be callable")
        frame_callback = on_frame
        closed_callback = on_closed
        self._generation += 1
        generation = self._generation
        self._closed_notified = False
        self._client_close_notified = False
        self._last_error = ""

        def session_active() -> bool:
            with self._lifecycle_lock:
                return generation == self._generation and not self._closed_notified

        def notify_client_closed() -> None:
            with self._lifecycle_lock:
                if generation != self._generation or self._client_close_notified:
                    return
                self._client_close_notified = True
            try:
                closed_callback()
            except Exception as exc:
                self._record_error(exc)

        def finish_closed() -> None:
            with self._lifecycle_lock:
                if generation != self._generation or self._closed_notified:
                    return
                self._closed_notified = True
                self._control = None
                self._capture = None
            notify_client_closed()

        capture = self._capture_factory(
            cursor_capture=False,
            draw_border=False,
            window_hwnd=self.hwnd,
        )
        self._capture = capture

        @capture.event
        def on_frame_arrived(frame, capture_control) -> None:
            if not session_active():
                return
            try:
                image = frame.convert_to_bgr().frame_buffer
                if (
                    not isinstance(image, np.ndarray)
                    or image.ndim != 3
                    or image.shape[2] != 3
                    or image.size == 0
                ):
                    raise ValueError("WGC returned an invalid BGR frame")
                if not session_active():
                    return
                frame_callback(image.copy())
            except Exception as exc:
                self._record_error(exc)
                try:
                    capture_control.stop()
                except Exception as stop_error:
                    self._record_error(stop_error)
                    notify_client_closed()
                else:
                    finish_closed()

        @capture.event
        def on_closed() -> None:
            finish_closed()

        control = capture.start_free_threaded()
        if self._closed_notified:
            try:
                control.stop()
            except Exception as exc:
                self._record_error(exc)
            self._capture = None
            self._control = None
        else:
            self._control = control

    def stop(self) -> None:
        """Stop native capture without holding the callback lifecycle lock.

        ``control.stop()`` may wait for a WGC callback that itself calls
        ``session_active()``.  Holding ``_lifecycle_lock`` across that native
        wait deadlocks Stop and, transitively, the application shutdown path.
        """
        with self._lifecycle_lock:
            if self._control is None:
                return
            control = self._control
            previous_closed = self._closed_notified
            previous_generation = self._generation
            self._generation += 1
            self._closed_notified = True
        try:
            control.stop()
        except Exception:
            with self._lifecycle_lock:
                if self._control is control:
                    self._generation = previous_generation
                    self._closed_notified = previous_closed
            raise
        with self._lifecycle_lock:
            if self._control is control:
                self._control = None
                self._capture = None


class SampleBuilder(Protocol):
    def build(self, sequence: int, frame: np.ndarray) -> MonitoringSample: ...


class CalibratedFrameSampleBuilder:
    def __init__(
        self,
        layout: Mapping[str, Any],
        navigation_analyzer: Callable[[int, np.ndarray], NavigationObservation],
    ) -> None:
        if not isinstance(layout, Mapping) or layout.get("schema_version") != 1:
            raise ValueError("layout must use schema version 1")
        if not callable(navigation_analyzer):
            raise ValueError("navigation_analyzer must be callable")
        regions = layout.get("regions")
        required = ("minimap", "ship_hp_bar", "ship_rations_bar")
        if not isinstance(regions, Mapping) or any(name not in regions for name in required):
            raise ValueError("layout is missing required regions")
        self._layout = layout
        self._navigation_analyzer = navigation_analyzer

    def _rect(self, name: str, width: int, height: int) -> Rect:
        try:
            values = self._layout["regions"][name]["normalized"]
            coordinates = tuple(values[key] for key in ("x", "y", "width", "height"))
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                for value in coordinates
            ):
                raise ValueError("normalized values must be finite numbers")
            x, y, region_width, region_height = map(float, coordinates)
            if (
                x < 0
                or y < 0
                or region_width <= 0
                or region_height <= 0
                or x + region_width > 1
                or y + region_height > 1
            ):
                raise ValueError("normalized values leave unit bounds")
            rect = Rect.from_normalized(values, width, height)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid normalized region: {name}") from exc
        if (
            rect.x < 0
            or rect.y < 0
            or rect.width <= 0
            or rect.height <= 0
            or rect.x + rect.width > width
            or rect.y + rect.height > height
        ):
            raise ValueError(f"region leaves frame: {name}")
        return rect

    def build(self, sequence: int, frame: np.ndarray) -> MonitoringSample:
        image = _validate_bgr_uint8(frame, "frame")
        height, width = image.shape[:2]
        minimap = self._rect("minimap", width, height)
        hp = self._rect("ship_hp_bar", width, height)
        rations = self._rect("ship_rations_bar", width, height)
        minimap_roi = image[
            minimap.y : minimap.y + minimap.height,
            minimap.x : minimap.x + minimap.width,
        ].copy()
        hp_reading = measure_resource_bar(image, hp, ResourceKind.HP)
        rations_reading = measure_resource_bar(image, rations, ResourceKind.RATIONS)
        paired_candidates = (
            hp_reading.candidate_percent is not None
            and rations_reading.candidate_percent is not None
            and min(hp_reading.candidate_percent, rations_reading.candidate_percent) > 0.0
        )
        hp_percent = hp_reading.percent
        rations_percent = rations_reading.percent
        if paired_candidates:
            # Both independently calibrated colored tracks are present.  This
            # covers decorated live BDO bars whose pale rail is obscured, while
            # paired zero-percent shore/menu backgrounds remain absent.
            hp_percent = hp_reading.candidate_percent
            rations_percent = rations_reading.candidate_percent
        elif hp_reading.observable and rations_percent is None and rations_reading.candidate_percent is not None:
            rations_percent = rations_reading.candidate_percent
        both_visible = hp_percent is not None and rations_percent is not None
        observation = None
        if both_visible:
            observation = self._navigation_analyzer(sequence, minimap_roi)
            if type(observation) is not NavigationObservation:
                raise ValueError("navigation analyzer must return NavigationObservation")
        else:
            self.reset()
        return MonitoringSample(
            sequence=sequence,
            capture_ok=True,
            calibration_ok=True,
            minimap_roi=minimap_roi,
            navigation_observation=observation,
            hp_percent=hp_percent,
            rations_percent=rations_percent,
            resource_bars_visible=both_visible,
        )

    def reset(self) -> None:
        reset = getattr(self._navigation_analyzer, "reset", None)
        if callable(reset):
            reset()


class SimpleNavigationAnalyzer:
    """Detect sustained 3D-map progress and route presence after stationarity."""

    def __init__(
        self,
        *,
        minimum_displacement: float = 0.04,
        motion_window_frames: int = 20,
        minimum_moving_fraction: float = 0.75,
        minimum_sustained_moving_fraction: float = 0.50,
        minimum_progress_ratio: float = 0.30,
    ) -> None:
        if (
            isinstance(minimum_displacement, bool)
            or not isinstance(minimum_displacement, Real)
            or not isfinite(float(minimum_displacement))
            or minimum_displacement < 0
        ):
            raise ValueError("minimum_displacement must be a finite non-negative number")
        if type(motion_window_frames) is not int or motion_window_frames < 2:
            raise ValueError("motion_window_frames must be an integer of at least 2")
        for value, name in (
            (minimum_moving_fraction, "minimum_moving_fraction"),
            (minimum_sustained_moving_fraction, "minimum_sustained_moving_fraction"),
            (minimum_progress_ratio, "minimum_progress_ratio"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or not 0 < float(value) <= 1
            ):
                raise ValueError(f"{name} must be a finite number from 0 exclusive to 1 inclusive")
        if minimum_sustained_moving_fraction > minimum_moving_fraction:
            raise ValueError(
                "minimum_sustained_moving_fraction cannot exceed minimum_moving_fraction"
            )
        self._minimum_displacement = float(minimum_displacement)
        self._motion_window_frames = motion_window_frames
        self._minimum_moving_fraction = float(minimum_moving_fraction)
        self._minimum_sustained_moving_fraction = float(minimum_sustained_moving_fraction)
        self._minimum_progress_ratio = float(minimum_progress_ratio)
        self._motion_window: deque[tuple[float, float]] = deque(maxlen=motion_window_frames)
        self._moving_latched = False
        self._previous: np.ndarray | None = None
        self._last_sequence: int | None = None

    def __call__(self, sequence: int, minimap_roi: np.ndarray) -> NavigationObservation:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        image = _validate_bgr_uint8(minimap_roi, "minimap_roi")
        discontinuity = self._last_sequence is not None and sequence != self._last_sequence + 1
        if discontinuity or (self._previous is not None and self._previous.shape != image.shape):
            self.reset()
        motion = MotionClass.UNKNOWN
        if self._previous is not None:
            evidence = estimate_texture_motion(self._previous, image)
            if evidence.valid:
                self._motion_window.append((evidence.dx, evidence.dy))
                if len(self._motion_window) == self._motion_window_frames:
                    vectors = np.asarray(self._motion_window, dtype=np.float64)
                    magnitudes = np.linalg.norm(vectors, axis=1)
                    moving_fraction = float(
                        np.count_nonzero(magnitudes >= self._minimum_displacement)
                    ) / self._motion_window_frames
                    path_length = float(magnitudes.sum())
                    net_progress = float(np.linalg.norm(vectors.sum(axis=0)))
                    progress_ratio = net_progress / path_length if path_length > 0 else 0.0
                    required_fraction = (
                        self._minimum_sustained_moving_fraction
                        if self._moving_latched
                        else self._minimum_moving_fraction
                    )
                    self._moving_latched = (
                        moving_fraction >= required_fraction
                        and progress_ratio >= self._minimum_progress_ratio
                    )
                    motion = MotionClass.MOVING if self._moving_latched else MotionClass.STATIONARY
            else:
                self._motion_window.clear()
                self._moving_latched = False
        route = (None, None)
        if motion is MotionClass.STATIONARY:
            route = {
                RoutePresence.PRESENT: (False, True),
                RoutePresence.ABSENT: (True, False),
                RoutePresence.UNKNOWN: (None, None),
            }[assess_route_presence(image)]
        self._previous = image.copy()
        self._last_sequence = sequence
        return NavigationObservation(motion, *route)

    def reset(self) -> None:
        self._motion_window.clear()
        self._moving_latched = False
        self._previous = None
        self._last_sequence = None


class ConservativeNavigationAnalyzer:
    def __init__(
        self,
        *,
        route_evidence: Callable[[int, np.ndarray], tuple[bool | None, bool | None]],
        window_size: int = 8,
    ) -> None:
        if not callable(route_evidence):
            raise ValueError("route_evidence must be callable")
        if type(window_size) is not int or window_size < 3:
            raise ValueError("window_size must be an integer of at least 3")
        self._route_evidence = route_evidence
        self._vectors: deque[MotionVector] = deque(maxlen=window_size)
        self._activity = ActivityMotionTracker()
        self._previous: np.ndarray | None = None
        self._previous_image: np.ndarray | None = None
        self._last_sequence: int | None = None

    def _reset_evidence(self) -> None:
        self._vectors.clear()
        self._activity = ActivityMotionTracker()
        self._previous = None
        self._previous_image = None
        reset_route_evidence = getattr(self._route_evidence, "reset", None)
        if callable(reset_route_evidence):
            reset_route_evidence()

    def __call__(self, sequence: int, minimap_roi: np.ndarray) -> NavigationObservation:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        image = _validate_bgr_uint8(minimap_roi, "minimap_roi")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sequence_discontinuity = (
            self._last_sequence is not None and sequence != self._last_sequence + 1
        )
        shape_discontinuity = self._previous is not None and self._previous.shape != gray.shape
        if sequence_discontinuity or shape_discontinuity:
            self._reset_evidence()
        route_complete, endpoint_ahead = self._route_evidence(sequence, image)
        for value in (route_complete, endpoint_ahead):
            if value is not None and type(value) is not bool:
                raise ValueError("route evidence values must be boolean or None")
        motion = MotionClass.UNKNOWN
        if (
            self._previous is not None
            and self._previous_image is not None
            and self._previous.shape == gray.shape
            and self._previous_image.shape == image.shape
        ):
            self._vectors.append(estimate_translation(self._previous, gray))
            motion = classify_motion_window(self._vectors)
            activity = self._activity.update(sequence, estimate_activity(self._previous_image, image))
            if motion is MotionClass.STATIONARY and activity.soft_activity_persistent:
                motion = MotionClass.UNKNOWN
        self._previous = gray.copy()
        self._previous_image = image.copy()
        self._last_sequence = sequence
        return NavigationObservation(motion, route_complete, endpoint_ahead)

    def reset(self) -> None:
        self._reset_evidence()
        self._last_sequence = None


class MonitoringProcessor(Protocol):
    def process(self, sample: MonitoringSample) -> MonitoringResult: ...


class EventRouter(Protocol):
    def route(self, event: NotificationEvent) -> tuple[DeliveryResult, ...]: ...


@dataclass(frozen=True)
class RuntimeStatus:
    running: bool
    sequence: int
    navigation_state: NavigationState
    diagnostic: str = ""
    result: MonitoringResult | None = None
    deliveries: tuple[DeliveryResult, ...] = ()
    sample: MonitoringSample | None = None
    shadow_metrics: ShadowMetrics | None = None
    shadow_accounting: ShadowAccounting | None = None


class RuntimeBridge:
    def __init__(
        self,
        source: FrameSource,
        builder: SampleBuilder,
        loop: MonitoringProcessor,
        router: EventRouter,
        status_callback: Callable[[RuntimeStatus], None],
        *,
        clock: Callable[[], float] = monotonic,
        analysis_cadence: AnalysisCadence | None = None,
        capture_watchdog: CaptureFreshnessWatchdog | None = None,
        shadow_processor: ShadowProcessor | None = None,
        shadow_queue_capacity: int = 1,
    ) -> None:
        for dependency, method in (
            (source, "start"),
            (source, "stop"),
            (builder, "build"),
            (loop, "process"),
            (router, "route"),
        ):
            if not callable(getattr(dependency, method, None)):
                raise ValueError(f"runtime dependency must provide {method}")
        if not callable(status_callback):
            raise ValueError("status_callback must be callable")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if analysis_cadence is not None and type(analysis_cadence) is not AnalysisCadence:
            raise ValueError("analysis_cadence must be AnalysisCadence or None")
        if capture_watchdog is not None and any(
            not callable(getattr(capture_watchdog, method, None))
            for method in ("start", "frame_arrived", "stop")
        ):
            raise ValueError("capture_watchdog must provide start, frame_arrived, and stop")
        if shadow_processor is not None and any(
            not callable(getattr(shadow_processor, method, None)) for method in ("process", "reset")
        ):
            raise ValueError("shadow_processor must provide process and reset")
        self.source = source
        self.builder = builder
        self.loop = loop
        self.router = router
        self.status_callback = status_callback
        self._clock = clock
        self._analysis_cadence = analysis_cadence
        self._capture_watchdog = (
            CaptureFreshnessWatchdog(timeout_seconds=5.0)
            if capture_watchdog is None
            else capture_watchdog
        )
        self._watchdog_generation: int | None = None
        self._shadow = (
            None
            if shadow_processor is None
            else ShadowPipelineRunner(
                shadow_processor,
                queue_capacity=shadow_queue_capacity,
                on_metrics=lambda: self._publish(diagnostic="shadow metrics"),
            )
        )
        self.running = False
        self.sequence = 0
        self._source_sequence = 0
        self.navigation_state = NavigationState.UNKNOWN
        self.last_status: RuntimeStatus | None = None
        self.last_status_error = ""
        self._lifecycle_lock = threading.RLock()
        self._generation = 0
        self._stop_required = False
        self._cleanup_required = False

    @property
    def cleanup_required(self) -> bool:
        with self._lifecycle_lock:
            return self._cleanup_required

    @property
    def shadow_accounting(self) -> ShadowAccounting:
        return ShadowAccounting() if self._shadow is None else self._shadow.accounting

    def configure_resource_warning_zones(
        self,
        ship_hp_warning_percent: int,
        ship_rations_warning_percent: int,
    ) -> None:
        for value in (ship_hp_warning_percent, ship_rations_warning_percent):
            if type(value) is not int or not 1 <= value <= 99:
                raise ValueError("resource warning percentages must be from 1 to 99")
        with self._lifecycle_lock:
            configure = getattr(self.loop, "configure_resource_warning_zones", None)
            if not callable(configure):
                raise RuntimeError("monitoring loop does not support live resource warning zones")
            configure(ship_hp_warning_percent, ship_rations_warning_percent)

    def configure_delivery(self, adapters: tuple, minimap_loss_delay_seconds: int) -> None:
        """Atomically replace delivery adapters and loss timing between frames."""
        from .notifications import NotificationRouter

        if type(adapters) is not tuple:
            raise ValueError("adapters must be a tuple of notification adapters")
        if type(minimap_loss_delay_seconds) is not int or not 10 <= minimap_loss_delay_seconds <= 300:
            raise ValueError("minimap_loss_delay_seconds must be an integer from 10 to 300")
        router = NotificationRouter(adapters)
        with self._lifecycle_lock:
            minimap = getattr(self.loop, "minimap", None)
            visibility = getattr(getattr(minimap, "gate", None), "visibility", None)
            configure = getattr(visibility, "configure_loss_delay", None)
            if not callable(configure):
                raise RuntimeError("monitoring loop does not support live minimap loss delay")
            configure(minimap_loss_delay_seconds)
            self.router = router

    def configure_repeat_alerts(self, enabled: bool, interval_seconds: int) -> None:
        if type(enabled) is not bool or type(interval_seconds) is not int or not 30 <= interval_seconds <= 300:
            raise ValueError("repeat alert settings are invalid")
        with self._lifecycle_lock:
            configure = getattr(self.loop, "configure_repeat_alerts", None)
            if not callable(configure):
                raise RuntimeError("monitoring loop does not support live repeat alerts")
            configure(enabled, interval_seconds)

    def _publish(
        self,
        *,
        diagnostic: str = "",
        result: MonitoringResult | None = None,
        deliveries: tuple[DeliveryResult, ...] = (),
        sample: MonitoringSample | None = None,
    ) -> None:
        status = RuntimeStatus(
            running=self.running,
            sequence=self.sequence,
            navigation_state=self.navigation_state,
            diagnostic=diagnostic,
            result=result,
            deliveries=deliveries,
            sample=sample,
            shadow_metrics=None if self._shadow is None else self._shadow.latest_metrics,
            shadow_accounting=None if self._shadow is None else self._shadow.accounting,
        )
        self.last_status = status
        try:
            self.status_callback(status)
        except Exception as exc:
            self.last_status_error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.running:
                return
            if self._stop_required:
                raise RuntimeError("runtime requires successful stop before restart")
            reset = getattr(self.builder, "reset", None)
            if callable(reset):
                reset()
            if self._shadow is not None:
                self._shadow.reset()
            self._generation += 1
            generation = self._generation
            self.running = True
            self._stop_required = True
            self._cleanup_required = True

            self._watchdog_generation = self._capture_watchdog.start(
                lambda freshness: self._capture_silence_for_generation(generation, freshness)
            )

            def on_frame(frame: np.ndarray) -> None:
                self._process_frame_for_generation(generation, frame)

            def on_closed() -> None:
                self._capture_closed(generation)

            try:
                self.source.start(on_frame, on_closed)
            except Exception as exc:
                self._capture_watchdog.stop()
                if generation == self._generation:
                    self.running = False
                self._publish(diagnostic=f"runtime start: {type(exc).__name__}: {exc}")
                raise
            if generation != self._generation or not self.running:
                return
            source_running = getattr(self.source, "is_running", True)
            if source_running is False:
                self.running = False
                self._capture_watchdog.stop()
                self._publish(diagnostic="capture closed during runtime startup")
                return
            self._stop_required = False
            self._cleanup_required = False
        self._publish(diagnostic="runtime started")

    def stop(self) -> None:
        """Stop capture outside the runtime lock.

        Native WGC stop can wait for an in-flight frame callback. That callback
        enters ``_process_frame_for_generation`` and needs this same lock, so
        keeping it here makes restart → stop intermittently deadlock.
        """
        with self._lifecycle_lock:
            if not self.running and not self._stop_required:
                return
            self.running = False
            self._generation += 1
            self._stop_required = True
            self._cleanup_required = True
        self._capture_watchdog.stop()
        try:
            self.source.stop()
        except Exception as exc:
            self._publish(diagnostic=f"runtime stop: {type(exc).__name__}: {exc}")
            raise
        with self._lifecycle_lock:
            self._stop_required = False
            self._cleanup_required = False
        self._publish(diagnostic="runtime stopped")

    def process_frame(self, frame: np.ndarray) -> None:
        with self._lifecycle_lock:
            generation = self._generation
        self._process_frame_for_generation(generation, frame)

    def _process_frame_for_generation(self, generation: int, frame: np.ndarray) -> None:
        with self._lifecycle_lock:
            if generation != self._generation or not self.running:
                return
            watchdog_generation = self._watchdog_generation
            freshness = self._capture_watchdog.frame_arrived(watchdog_generation)
            if freshness is not CaptureFreshness.FRESH:
                self._capture_uncertain_for_generation(generation)
                return
            self._source_sequence += 1
            try:
                received_monotonic = self._clock()
                if self._analysis_cadence is not None:
                    candidate = FrameEnvelope(
                        source_sequence=self._source_sequence,
                        processing_ordinal=self.sequence + 1,
                        received_monotonic=received_monotonic,
                        calibration_epoch=0,
                    )
                    if not self._analysis_cadence.accept(candidate):
                        if self._analysis_cadence.reset_required:
                            reset_builder = getattr(self.builder, "reset", None)
                            if callable(reset_builder):
                                reset_builder()
                            if self._shadow is not None:
                                self._shadow.reset()
                            self._analysis_cadence.reset()
                        return
                self.sequence += 1
                sample = self.builder.build(self.sequence, frame)
                sample = replace(
                    sample,
                    sequence=self.sequence,
                    source_sequence=self._source_sequence,
                    processing_ordinal=self.sequence,
                    received_monotonic=received_monotonic,
                    native_source_timestamp=None,
                    calibration_epoch=0,
                )
                result = self.loop.process(sample)
                analyzer = getattr(self.builder, "_navigation_analyzer", None)
                if (
                    result.minimap_visible is not True
                    and not isinstance(analyzer, SimpleNavigationAnalyzer)
                ):
                    reset_builder = getattr(self.builder, "reset", None)
                    if callable(reset_builder):
                        reset_builder()
                deliveries = tuple(
                    delivery
                    for event in result.events
                    for delivery in self.router.route(event)
                )
                if self._shadow is not None:
                    self._shadow.submit(sample)
            except Exception as exc:
                self._publish(diagnostic=f"frame processing: {type(exc).__name__}: {exc}")
                return
            if generation != self._generation or not self.running:
                return
            self.navigation_state = result.navigation_state
            self._publish(result=result, deliveries=deliveries, sample=sample)

    def _capture_silence_for_generation(
        self, generation: int, freshness: CaptureFreshness
    ) -> None:
        if freshness is not CaptureFreshness.SILENT:
            return
        self._process_capture_unavailable(generation, "capture silence", force_loss=True)

    def _capture_uncertain_for_generation(self, generation: int) -> None:
        self._process_capture_unavailable(generation, "capture freshness uncertain", force_loss=False)

    def _process_capture_unavailable(
        self, generation: int, diagnostic: str, *, force_loss: bool
    ) -> None:
        with self._lifecycle_lock:
            if generation != self._generation or not self.running:
                return
            self.sequence += 1
            sequence = self.sequence
        try:
            capture_closed = getattr(self.loop, "capture_closed", None)
            result = (
                capture_closed(sequence)
                if force_loss and callable(capture_closed)
                else self.loop.process(
                    MonitoringSample(sequence=sequence, capture_ok=False, calibration_ok=False)
                )
            )
            events = result.events
            if force_loss and not any(event.event_type is EventType.CAPTURE_LOST for event in events):
                events = events + (NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL),)
                result = replace(result, events=events)
            deliveries = tuple(
                delivery for event in events for delivery in self.router.route(event)
            )
            with self._lifecycle_lock:
                if generation != self._generation or not self.running:
                    return
                self.navigation_state = result.navigation_state
        except Exception as exc:
            self._publish(diagnostic=f"{diagnostic}: {type(exc).__name__}: {exc}")
            return
        self._publish(diagnostic=diagnostic, result=result, deliveries=deliveries)

    def _capture_closed(self, generation: int) -> None:
        with self._lifecycle_lock:
            if generation != self._generation or not self.running:
                return
            self.running = False
            self._stop_required = True
            self._cleanup_required = True
        self._capture_watchdog.stop()
        diagnostics = []
        stop_succeeded = False
        try:
            self.source.stop()
        except Exception as exc:
            diagnostics.append(f"source stop: {type(exc).__name__}: {exc}")
        else:
            stop_succeeded = True
            with self._lifecycle_lock:
                if generation == self._generation:
                    self._cleanup_required = False
        with self._lifecycle_lock:
            if generation != self._generation:
                return
            self.sequence += 1
        try:
            capture_closed = getattr(self.loop, "capture_closed", None)
            result = (
                capture_closed(self.sequence)
                if callable(capture_closed)
                else self.loop.process(
                    MonitoringSample(
                        sequence=self.sequence,
                        capture_ok=False,
                        calibration_ok=False,
                    )
                )
            )
            events = result.events
            if not any(event.event_type is EventType.CAPTURE_LOST for event in events):
                events = events + (
                    NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL),
                )
                result = replace(result, events=events)
            with self._lifecycle_lock:
                if generation != self._generation:
                    return
                self.navigation_state = result.navigation_state
        except Exception as exc:
            diagnostics.append(f"loss processing: {type(exc).__name__}: {exc}")
            loss = NotificationEvent(EventType.CAPTURE_LOST, Severity.CRITICAL)
            result = MonitoringResult(
                self.sequence,
                (loss,),
                self.navigation_state,
                tuple(diagnostics),
            )
            events = result.events
        deliveries_list = []
        for event in events:
            try:
                deliveries_list.extend(self.router.route(event))
            except Exception as exc:
                diagnostics.append(f"event routing: {type(exc).__name__}: {exc}")
        diagnostic = "capture closed"
        if diagnostics:
            diagnostic += "; " + "; ".join(diagnostics)
        self._publish(
            diagnostic=diagnostic,
            result=result,
            deliveries=tuple(deliveries_list),
        )
        if stop_succeeded:
            with self._lifecycle_lock:
                if generation == self._generation:
                    self._stop_required = False
