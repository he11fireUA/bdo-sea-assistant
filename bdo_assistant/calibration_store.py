from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class CalibrationStatus(str, Enum):
    MISSING = "missing"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class CalibrationReadiness:
    status: CalibrationStatus
    diagnostic: str
    layout: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CalibrationStatus):
            raise ValueError("status must be CalibrationStatus")
        if type(self.diagnostic) is not str or not self.diagnostic:
            raise ValueError("diagnostic must be a non-empty string")
        if self.status is CalibrationStatus.VALID:
            if not isinstance(self.layout, Mapping):
                raise ValueError("valid readiness requires a layout")
        elif self.layout is not None:
            raise ValueError("non-valid readiness must not include a layout")

    @property
    def ready(self) -> bool:
        return self.status is CalibrationStatus.VALID


class JsonCalibrationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def inspect(self) -> CalibrationReadiness:
        if not self.path.is_file():
            return CalibrationReadiness(
                status=CalibrationStatus.MISSING,
                diagnostic="calibration file not found",
            )
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            _validate_layout(raw)
            frozen = _deep_freeze(raw)
        except (OSError, OverflowError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return CalibrationReadiness(
                status=CalibrationStatus.INVALID,
                diagnostic=f"invalid calibration: {exc}",
            )
        return CalibrationReadiness(
            status=CalibrationStatus.VALID,
            diagnostic="confirmed calibration loaded",
            layout=frozen,
        )


def default_calibration_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / "BDOSeaAssistant" / "ui_layout.json"


def _exact_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not isfinite(converted):
        raise ValueError(f"{label} must be a finite number")
    return converted


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _validate_layout(layout: object) -> None:
    root = _mapping(layout, "layout")
    if set(root) != {"schema_version", "capture_size", "regions", "thresholds"}:
        raise ValueError("layout has unexpected or missing root fields")
    if root.get("schema_version") != 1 or type(root.get("schema_version")) is not int:
        raise ValueError("schema_version must be exact integer 1")
    capture = _mapping(root.get("capture_size"), "capture_size")
    if set(capture) != {"width", "height"}:
        raise ValueError("capture_size must contain exactly width and height")
    frame_width = _exact_positive_int(capture.get("width"), "capture width")
    frame_height = _exact_positive_int(capture.get("height"), "capture height")
    regions = _mapping(root.get("regions"), "regions")
    required_regions = ("minimap", "ship_hp_bar", "ship_rations_bar")
    if set(regions) != set(required_regions):
        raise ValueError("regions must contain exactly minimap and ship resource bars")
    for name in required_regions:
        _validate_region(name, regions[name], frame_width, frame_height)
    thresholds = _mapping(root.get("thresholds"), "thresholds")
    expected_thresholds = ("ship_hp_warning_percent", "ship_rations_warning_percent")
    if set(thresholds) != set(expected_thresholds):
        raise ValueError("thresholds contain unexpected or missing fields")
    for name in expected_thresholds:
        value = _finite_number(thresholds[name], name)
        if not 0 <= value <= 100:
            raise ValueError(f"{name} must be from 0 to 100")


def _validate_region(
    name: str,
    value: object,
    frame_width: int,
    frame_height: int,
) -> None:
    region = _mapping(value, name)
    if set(region) != {"absolute", "normalized", "confidence", "ui_scale"}:
        raise ValueError(f"{name} has unexpected or missing fields")
    absolute = _mapping(region["absolute"], f"{name}.absolute")
    normalized = _mapping(region["normalized"], f"{name}.normalized")
    geometry_fields = ("x", "y", "width", "height")
    if set(absolute) != set(geometry_fields) or set(normalized) != set(geometry_fields):
        raise ValueError(f"{name} geometry fields are incomplete")
    absolute_values: dict[str, int] = {}
    for field in geometry_fields:
        item = absolute[field]
        if type(item) is not int:
            raise ValueError(f"{name}.absolute.{field} must be an integer")
        absolute_values[field] = item
    if (
        absolute_values["x"] < 0
        or absolute_values["y"] < 0
        or absolute_values["width"] <= 0
        or absolute_values["height"] <= 0
        or absolute_values["x"] + absolute_values["width"] > frame_width
        or absolute_values["y"] + absolute_values["height"] > frame_height
    ):
        raise ValueError(f"{name} absolute geometry leaves capture bounds")
    normalized_values = {
        field: _finite_number(normalized[field], f"{name}.normalized.{field}")
        for field in geometry_fields
    }
    if (
        normalized_values["x"] < 0
        or normalized_values["y"] < 0
        or normalized_values["width"] <= 0
        or normalized_values["height"] <= 0
        or normalized_values["x"] + normalized_values["width"] > 1
        or normalized_values["y"] + normalized_values["height"] > 1
    ):
        raise ValueError(f"{name} normalized geometry leaves unit bounds")
    expected = {
        "x": round(normalized_values["x"] * frame_width),
        "y": round(normalized_values["y"] * frame_height),
        "width": round(normalized_values["width"] * frame_width),
        "height": round(normalized_values["height"] * frame_height),
    }
    if any(abs(expected[field] - absolute_values[field]) > 1 for field in geometry_fields):
        raise ValueError(f"{name} geometry mismatch between absolute and normalized values")
    confidence = _finite_number(region["confidence"], f"{name}.confidence")
    if not 0 <= confidence <= 1:
        raise ValueError(f"{name}.confidence must be from 0 to 1")
    scale = _finite_number(region["ui_scale"], f"{name}.ui_scale")
    if not 0.60 <= scale <= 1.80:
        raise ValueError(f"{name}.ui_scale must be from 0.60 to 1.80")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value
