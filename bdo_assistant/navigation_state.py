from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

from .motion import MotionClass


class NavigationState(str, Enum):
    UNKNOWN = "unknown"
    NO_SHIP = "no_ship"
    MOVING = "moving"
    STATIONARY_CANDIDATE = "stationary_candidate"
    ARRIVED = "arrived"
    DEPARTED = "departed"
    STUCK = "stuck"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class NavigationObservation:
    motion: MotionClass
    route_complete: bool | None
    endpoint_ahead: bool | None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        optional_bools = (self.route_complete, self.endpoint_ahead)
        if not isinstance(self.motion, MotionClass) or any(
            value is not None and type(value) is not bool for value in optional_bools
        ):
            raise ValueError("NavigationObservation fields have invalid types")


def _seconds(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number or None")
    return float(value)


def _observed_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0:
        raise ValueError("observed_monotonic must be a finite non-negative number")
    return float(value)


class NavigationStateMachine:
    __slots__ = (
        "_arrival_frames", "_stuck_frames", "_recovery_frames", "_departure_frames",
        "_arrival_seconds", "_stuck_seconds", "_recovery_seconds", "_departure_seconds",
        "_state", "_arrival_count", "_stuck_count", "_recovery_count", "_departure_count",
        "_arrival_since", "_stuck_since", "_recovery_since", "_departure_since",
        "_last_observed_monotonic", "_simple", "_confirmation_seconds", "_pending_state", "_pending_since",
        "_route_seen_ahead", "_recent_near_monotonic",
    )

    def __init__(
        self,
        *,
        arrival_frames: int = 3,
        stuck_frames: int = 15,
        recovery_frames: int = 5,
        departure_frames: int = 5,
        arrival_seconds: float | None = None,
        stuck_seconds: float | None = None,
        recovery_seconds: float | None = None,
        departure_seconds: float | None = None,
        simple: bool = False,
        confirmation_seconds: float = 5.0,
    ) -> None:
        # Frame thresholds remain a compatibility mode; they are never seconds.
        self._arrival_frames = arrival_frames
        self._stuck_frames = stuck_frames
        self._recovery_frames = recovery_frames
        self._departure_frames = departure_frames
        self._arrival_seconds = _seconds(arrival_seconds, "arrival_seconds")
        self._stuck_seconds = _seconds(stuck_seconds, "stuck_seconds")
        self._recovery_seconds = _seconds(recovery_seconds, "recovery_seconds")
        self._departure_seconds = _seconds(departure_seconds, "departure_seconds")
        if type(simple) is not bool:
            raise ValueError("simple must be boolean")
        self._simple = simple
        self._confirmation_seconds = _seconds(confirmation_seconds, "confirmation_seconds")
        NavigationStateMachine._validate_configuration(self)
        self._state = NavigationState.UNKNOWN
        self._arrival_count = self._stuck_count = self._recovery_count = self._departure_count = 0
        self._arrival_since = self._stuck_since = self._recovery_since = self._departure_since = None
        self._last_observed_monotonic: float | None = None
        self._pending_state: NavigationState | None = None
        self._pending_since: float | None = None
        self._route_seen_ahead = False
        self._recent_near_monotonic: float | None = None

    def _validate_configuration(self) -> None:
        thresholds = (self._arrival_frames, self._stuck_frames, self._recovery_frames, self._departure_frames)
        if any(type(value) is not int or value < 2 for value in thresholds):
            raise ValueError("state-machine frame thresholds must be positive integers of at least 2")
        for value, name in (
            (self._arrival_seconds, "arrival_seconds"),
            (self._stuck_seconds, "stuck_seconds"),
            (self._recovery_seconds, "recovery_seconds"),
            (self._departure_seconds, "departure_seconds"),
            (self._confirmation_seconds, "confirmation_seconds"),
        ):
            _seconds(value, name)

    @property
    def arrival_frames(self) -> int:
        return self._arrival_frames

    @property
    def stuck_frames(self) -> int:
        return self._stuck_frames

    @property
    def recovery_frames(self) -> int:
        return self._recovery_frames

    @property
    def departure_frames(self) -> int:
        return self._departure_frames

    @property
    def state(self) -> NavigationState:
        return self._state

    @property
    def simple(self) -> bool:
        return self._simple

    def _reset_approach_counts(self) -> None:
        self._arrival_count = self._stuck_count = 0
        self._arrival_since = self._stuck_since = None

    def _reset_recovery(self) -> None:
        self._recovery_count = 0
        self._recovery_since = None

    def _reset_departure(self) -> None:
        self._departure_count = 0
        self._departure_since = None

    def _reset_simple_pending(self) -> None:
        self._pending_state = None
        self._pending_since = None

    def _reset_simple_route_context(self) -> None:
        self._route_seen_ahead = False
        self._recent_near_monotonic = None

    def _simple_update(self, observation: NavigationObservation, now: float | None) -> NavigationState:
        """Apply the direct 3D-map truth table with five-second stationary terminals."""
        route_present = observation.route_complete is False and observation.endpoint_ahead is True
        route_absent = observation.route_complete is True and observation.endpoint_ahead is False
        if observation.motion is MotionClass.MOVING and route_present:
            self._state = NavigationState.MOVING
            self._reset_simple_pending()
            return self._state

        target = {
            (MotionClass.STATIONARY, True): NavigationState.STUCK,
            (MotionClass.STATIONARY, False): NavigationState.DEPARTED,
        }.get((observation.motion, route_present if route_present or route_absent else None))
        if target is None or now is None:
            self._state = NavigationState.UNKNOWN
            self._reset_simple_pending()
            return self._state
        if target is self._state:
            self._reset_simple_pending()
            return self._state
        if target is not self._pending_state:
            self._pending_state = target
            self._pending_since = now
            self._state = NavigationState.UNKNOWN
            return self._state
        if now - self._pending_since >= self._confirmation_seconds:
            self._state = target
            self._reset_simple_pending()
        else:
            self._state = NavigationState.UNKNOWN
        return self._state

    @staticmethod
    def _elapsed_confirmation(
        count: int,
        since: float | None,
        frames: int,
        seconds: float | None,
        observed_monotonic: float | None,
    ) -> tuple[bool, float | None]:
        if seconds is None:
            return count >= frames, since
        if observed_monotonic is None:
            return False, None
        if since is None:
            return False, observed_monotonic
        return observed_monotonic - since >= seconds, since

    def _validate_time(self, observed_monotonic: float | None) -> float | None:
        if observed_monotonic is None:
            return None
        value = _observed_time(observed_monotonic)
        if self._last_observed_monotonic is not None and value <= self._last_observed_monotonic:
            NavigationStateMachine.suspend(self)
            raise ValueError("observed_monotonic must strictly increase")
        self._last_observed_monotonic = value
        return value

    def suspend(self) -> NavigationState:
        """Discard unconfirmed sensor evidence while preserving confirmed latches."""
        NavigationStateMachine._validate_configuration(self)
        self._reset_approach_counts()
        self._reset_recovery()
        self._reset_departure()
        self._reset_simple_pending()
        self._reset_simple_route_context()
        self._last_observed_monotonic = None
        if self._simple:
            self._state = NavigationState.UNKNOWN
        elif self._state is NavigationState.RECOVERING:
            self._state = NavigationState.STUCK
        elif self._state not in (NavigationState.ARRIVED, NavigationState.STUCK):
            self._state = NavigationState.UNKNOWN
        return self._state

    def no_ship(self) -> NavigationState:
        """Pause navigation and discard all evidence until the paired ship HUD returns."""
        self.suspend()
        self._state = NavigationState.NO_SHIP
        return self._state

    def update(
        self,
        observation: NavigationObservation,
        observed_monotonic: float | None = None,
    ) -> NavigationState:
        NavigationStateMachine._validate_configuration(self)
        if type(observation) is not NavigationObservation:
            NavigationStateMachine.suspend(self)
            raise ValueError("observation must be NavigationObservation")
        try:
            NavigationObservation.validate(observation)
            now = self._validate_time(observed_monotonic)
        except ValueError:
            NavigationStateMachine.suspend(self)
            raise

        if self._simple:
            return self._simple_update(observation, now)

        if self._state in (NavigationState.STUCK, NavigationState.RECOVERING):
            if observation.motion is MotionClass.MOVING:
                self._recovery_count += 1
                confirmed, self._recovery_since = self._elapsed_confirmation(
                    self._recovery_count, self._recovery_since, self._recovery_frames,
                    self._recovery_seconds, now,
                )
                if confirmed:
                    self._state = NavigationState.MOVING
                    self._reset_recovery()
                    self._reset_approach_counts()
                else:
                    self._state = NavigationState.RECOVERING
            else:
                self._reset_recovery()
                self._state = NavigationState.STUCK
            return self._state

        if self._state is NavigationState.ARRIVED:
            if observation.motion is MotionClass.MOVING and observation.route_complete is False:
                self._departure_count += 1
                confirmed, self._departure_since = self._elapsed_confirmation(
                    self._departure_count, self._departure_since, self._departure_frames,
                    self._departure_seconds, now,
                )
                if confirmed:
                    self._state = NavigationState.MOVING
                    self._reset_departure()
            else:
                self._reset_departure()
            return self._state

        # Motion is independently useful telemetry.  Unknown route context may
        # show MOVING, but never supplies the evidence required for a terminal state.
        if observation.motion is MotionClass.MOVING:
            self._reset_approach_counts()
            self._state = NavigationState.MOVING
            return self._state

        if (
            observation.motion is MotionClass.UNKNOWN
            or observation.route_complete is None
            or observation.endpoint_ahead is None
        ):
            return NavigationStateMachine.suspend(self)

        if observation.motion in (MotionClass.STATIONARY, MotionClass.JITTER):
            if observation.route_complete is True and observation.endpoint_ahead is False:
                self._arrival_count += 1
                self._stuck_count = 0
                self._stuck_since = None
                confirmed, self._arrival_since = self._elapsed_confirmation(
                    self._arrival_count, self._arrival_since, self._arrival_frames,
                    self._arrival_seconds, now,
                )
                self._state = NavigationState.ARRIVED if confirmed else NavigationState.STATIONARY_CANDIDATE
                return self._state

            if observation.route_complete is False and observation.endpoint_ahead is True:
                self._stuck_count += 1
                self._arrival_count = 0
                self._arrival_since = None
                confirmed, self._stuck_since = self._elapsed_confirmation(
                    self._stuck_count, self._stuck_since, self._stuck_frames,
                    self._stuck_seconds, now,
                )
                self._state = NavigationState.STUCK if confirmed else NavigationState.STATIONARY_CANDIDATE
                return self._state

            self._reset_approach_counts()
            self._state = NavigationState.STATIONARY_CANDIDATE
            return self._state

        self._reset_approach_counts()
        self._state = NavigationState.UNKNOWN
        return self._state
