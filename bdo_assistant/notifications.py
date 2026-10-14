from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


class EventType(str, Enum):
    ARRIVED = "arrived"
    STUCK = "stuck"
    SHIP_HP_LOW = "ship_hp_low"
    RATIONS_LOW = "rations_low"
    CAPTURE_LOST = "capture_lost"
    CAPTURE_RESTORED = "capture_restored"
    MINIMAP_VISIBILITY_LOST = "minimap_visibility_lost"
    MINIMAP_VISIBILITY_RESTORED = "minimap_visibility_restored"
    UI_CALIBRATION_LOST = "ui_calibration_lost"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


def _freeze_payload(value: Any, active_ids: frozenset[int] = frozenset()) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in active_ids:
            raise ValueError("notification parameters must not contain cycles")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("notification parameter keys must be strings")
        nested_ids = active_ids | {object_id}
        return MappingProxyType(
            {key: _freeze_payload(item, nested_ids) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        object_id = id(value)
        if object_id in active_ids:
            raise ValueError("notification parameters must not contain cycles")
        nested_ids = active_ids | {object_id}
        return tuple(_freeze_payload(item, nested_ids) for item in value)
    raise TypeError("notification parameters must contain JSON-like immutable values")


@dataclass(frozen=True)
class NotificationEvent:
    event_type: EventType
    severity: Severity
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Mapping):
            raise TypeError("notification parameters must be a mapping")
        object.__setattr__(self, "parameters", _freeze_payload(self.parameters))


class NotificationAdapter(Protocol):
    name: str

    def send(self, event: NotificationEvent) -> None: ...


@dataclass(frozen=True)
class DeliveryResult:
    adapter: str
    success: bool
    error: str = ""


class NotificationRouter:
    def __init__(self, adapters: Sequence[NotificationAdapter]) -> None:
        validated = []
        for adapter in adapters:
            name = getattr(adapter, "name", None)
            send = getattr(adapter, "send", None)
            if not isinstance(name, str) or not name or not callable(send):
                raise ValueError("notification adapter must have a name and callable send")
            validated.append((name, adapter))
        self._adapters = tuple(validated)

    def route(self, event: NotificationEvent) -> tuple[DeliveryResult, ...]:
        results = []
        for name, adapter in self._adapters:
            try:
                adapter.send(event)
            except Exception as exc:
                results.append(DeliveryResult(name, False, str(exc)))
            else:
                results.append(DeliveryResult(name, True))
        return tuple(results)
