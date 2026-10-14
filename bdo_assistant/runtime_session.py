from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .runtime import RuntimeStatus


class RuntimeSessionState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeSessionSnapshot:
    state: RuntimeSessionState
    error: str = ""
    bridge_active: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state, RuntimeSessionState)
            or type(self.error) is not str
            or type(self.bridge_active) is not bool
        ):
            raise ValueError("runtime session snapshot is invalid")


class RuntimeSessionController:
    def __init__(self, bridge_factory: Callable[[], object], state_callback: Callable) -> None:
        if not callable(bridge_factory):
            raise ValueError("bridge_factory must be callable")
        if not callable(state_callback):
            raise ValueError("state_callback must be callable")
        self._bridge_factory = bridge_factory
        self._state_callback = state_callback
        self.state = RuntimeSessionState.STOPPED
        self.bridge: object | None = None
        self.last_error = ""
        self.last_callback_error = ""

    def _set_state(self, state: RuntimeSessionState, error: str = "") -> None:
        self.state = state
        self.last_error = error
        snapshot = RuntimeSessionSnapshot(state, error, self.bridge is not None)
        try:
            self._state_callback(snapshot)
        except Exception as exc:
            self.last_callback_error = f"{type(exc).__name__}: {exc}"

    def set_state_callback(self, callback: Callable[[RuntimeSessionSnapshot], None]) -> None:
        if not callable(callback):
            raise ValueError("state callback must be callable")
        self._state_callback = callback

    def start(self) -> None:
        if self.bridge is not None or self.state in (
            RuntimeSessionState.STARTING,
            RuntimeSessionState.RUNNING,
            RuntimeSessionState.STOPPING,
        ):
            raise RuntimeError("runtime session is already active")
        self._set_state(RuntimeSessionState.STARTING)
        bridge = None
        try:
            bridge = self._bridge_factory()
            if not callable(getattr(bridge, "start", None)) or not callable(
                getattr(bridge, "stop", None)
            ):
                raise ValueError("bridge factory returned an invalid bridge")
            bridge.start()
            if getattr(bridge, "running", True) is not True:
                raise RuntimeError("runtime bridge did not enter running state")
        except Exception as exc:
            cleanup_error = ""
            if bridge is not None and callable(getattr(bridge, "stop", None)):
                try:
                    bridge.stop()
                except Exception as cleanup_exc:
                    cleanup_error = f"; cleanup failed: {cleanup_exc}"
                    self.bridge = bridge
                else:
                    self.bridge = None
            else:
                self.bridge = None
            self._set_state(RuntimeSessionState.ERROR, f"{exc}{cleanup_error}")
            raise
        self.bridge = bridge
        self._set_state(RuntimeSessionState.RUNNING)

    def stop(self) -> None:
        if self.bridge is None:
            if self.state is not RuntimeSessionState.STOPPED:
                self._set_state(RuntimeSessionState.STOPPED)
            return
        bridge = self.bridge
        self._set_state(RuntimeSessionState.STOPPING)
        try:
            bridge.stop()
        except Exception as exc:
            self._set_state(RuntimeSessionState.ERROR, str(exc))
            raise
        self.bridge = None
        self._set_state(RuntimeSessionState.STOPPED)

    def configure_resource_warning_zones(
        self,
        ship_hp_warning_percent: int,
        ship_rations_warning_percent: int,
    ) -> None:
        for value in (ship_hp_warning_percent, ship_rations_warning_percent):
            if type(value) is not int or not 1 <= value <= 99:
                raise ValueError("resource warning percentages must be from 1 to 99")
        if self.bridge is None or self.state is not RuntimeSessionState.RUNNING:
            raise RuntimeError("runtime session is not running")
        configure = getattr(self.bridge, "configure_resource_warning_zones", None)
        if not callable(configure):
            raise RuntimeError("runtime bridge does not support live resource warning zones")
        configure(ship_hp_warning_percent, ship_rations_warning_percent)

    def configure_delivery(self, adapters: tuple, minimap_loss_delay_seconds: int) -> None:
        """Forward a complete delivery configuration to the active runtime."""
        if self.bridge is None or self.state is not RuntimeSessionState.RUNNING:
            raise RuntimeError("runtime session is not running")
        configure = getattr(self.bridge, "configure_delivery", None)
        if not callable(configure):
            raise RuntimeError("runtime bridge does not support live delivery configuration")
        configure(adapters, minimap_loss_delay_seconds)

    def configure_repeat_alerts(self, enabled: bool, interval_seconds: int) -> None:
        if type(enabled) is not bool or type(interval_seconds) is not int or not 30 <= interval_seconds <= 300:
            raise ValueError("repeat alert settings are invalid")
        if self.bridge is None or self.state is not RuntimeSessionState.RUNNING:
            raise RuntimeError("runtime session is not running")
        configure = getattr(self.bridge, "configure_repeat_alerts", None)
        if not callable(configure):
            raise RuntimeError("runtime bridge does not support live repeat alerts")
        configure(enabled, interval_seconds)

    def handle_runtime_status(self, status: RuntimeStatus) -> None:
        if type(status) is not RuntimeStatus:
            raise ValueError("status must be RuntimeStatus")
        if status.running or self.bridge is None:
            return
        if self.state is RuntimeSessionState.STOPPING:
            return
        if self.state not in (RuntimeSessionState.STARTING, RuntimeSessionState.RUNNING):
            return
        cleanup_required = getattr(self.bridge, "cleanup_required", False) is True
        if not cleanup_required:
            self.bridge = None
        diagnostic = status.diagnostic or "runtime bridge stopped unexpectedly"
        self._set_state(RuntimeSessionState.ERROR, diagnostic)
