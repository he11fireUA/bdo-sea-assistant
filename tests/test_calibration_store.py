import json

import pytest

from bdo_assistant.calibration_store import (
    CalibrationReadiness,
    CalibrationStatus,
    JsonCalibrationStore,
)


def _layout() -> dict:
    return {
        "schema_version": 1,
        "capture_size": {"width": 1000, "height": 500},
        "regions": {
            "minimap": {
                "absolute": {"x": 700, "y": 20, "width": 250, "height": 200},
                "normalized": {"x": 0.7, "y": 0.04, "width": 0.25, "height": 0.4},
                "confidence": 0.91,
                "ui_scale": 1.0,
            },
            "ship_hp_bar": {
                "absolute": {"x": 50, "y": 400, "width": 190, "height": 8},
                "normalized": {"x": 0.05, "y": 0.8, "width": 0.19, "height": 0.016},
                "confidence": 0.88,
                "ui_scale": 1.0,
            },
            "ship_rations_bar": {
                "absolute": {"x": 50, "y": 420, "width": 190, "height": 8},
                "normalized": {"x": 0.05, "y": 0.84, "width": 0.19, "height": 0.016},
                "confidence": 0.87,
                "ui_scale": 1.0,
            },
        },
        "thresholds": {
            "ship_hp_warning_percent": 60,
            "ship_rations_warning_percent": 10,
        },
    }


def test_missing_calibration_is_explicitly_not_ready(tmp_path) -> None:
    readiness = JsonCalibrationStore(tmp_path / "ui_layout.json").inspect()

    assert readiness == CalibrationReadiness(
        status=CalibrationStatus.MISSING,
        diagnostic="calibration file not found",
    )
    assert not readiness.ready


def test_valid_calibration_loads_as_immutable_runtime_mapping(tmp_path) -> None:
    path = tmp_path / "ui_layout.json"
    path.write_text(json.dumps(_layout()), encoding="utf-8")

    readiness = JsonCalibrationStore(path).inspect()

    assert readiness.status is CalibrationStatus.VALID
    assert readiness.ready
    assert readiness.layout["capture_size"]["width"] == 1000
    with pytest.raises(TypeError):
        readiness.layout["regions"]["minimap"]["normalized"]["x"] = 0.1


def test_invalid_calibration_fails_closed_with_diagnostic(tmp_path) -> None:
    path = tmp_path / "ui_layout.json"
    layout = _layout()
    layout["regions"]["minimap"]["normalized"]["x"] = float("nan")
    path.write_text(json.dumps(layout), encoding="utf-8")

    readiness = JsonCalibrationStore(path).inspect()

    assert readiness.status is CalibrationStatus.INVALID
    assert not readiness.ready
    assert readiness.layout is None
    assert "finite" in readiness.diagnostic


def test_absolute_and_normalized_geometry_must_agree(tmp_path) -> None:
    path = tmp_path / "ui_layout.json"
    layout = _layout()
    layout["regions"]["ship_hp_bar"]["absolute"]["x"] = 500
    path.write_text(json.dumps(layout), encoding="utf-8")

    readiness = JsonCalibrationStore(path).inspect()

    assert readiness.status is CalibrationStatus.INVALID
    assert "geometry mismatch" in readiness.diagnostic


@pytest.mark.parametrize(
    "mutation",
    [
        lambda layout: layout.update(schema_version=True),
        lambda layout: layout["capture_size"].update(width=0),
        lambda layout: layout["regions"].pop("minimap"),
        lambda layout: layout["regions"]["minimap"].update(confidence=1.1),
        lambda layout: layout["regions"]["minimap"].update(ui_scale=2.0),
        lambda layout: layout["thresholds"].update(ship_hp_warning_percent=True),
    ],
)
def test_schema_rejects_malformed_layouts(tmp_path, mutation) -> None:
    path = tmp_path / "ui_layout.json"
    layout = _layout()
    mutation(layout)
    path.write_text(json.dumps(layout), encoding="utf-8")

    readiness = JsonCalibrationStore(path).inspect()

    assert readiness.status is CalibrationStatus.INVALID
    assert readiness.layout is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda layout: layout["thresholds"].update(ship_hp_warning_percent=10**400),
        lambda layout: layout["capture_size"].update(width=10**400),
    ],
)
def test_oversized_json_integers_fail_closed_instead_of_escaping(tmp_path, mutation) -> None:
    path = tmp_path / "ui_layout.json"
    layout = _layout()
    mutation(layout)
    path.write_text(json.dumps(layout), encoding="utf-8")

    readiness = JsonCalibrationStore(path).inspect()

    assert readiness.status is CalibrationStatus.INVALID
    assert readiness.layout is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda layout: layout.update(unexpected=True),
        lambda layout: layout["capture_size"].update(unexpected=1),
    ],
)
def test_root_and_capture_schema_reject_unexpected_fields(tmp_path, mutation) -> None:
    path = tmp_path / "ui_layout.json"
    layout = _layout()
    mutation(layout)
    path.write_text(json.dumps(layout), encoding="utf-8")

    readiness = JsonCalibrationStore(path).inspect()

    assert readiness.status is CalibrationStatus.INVALID
    assert readiness.layout is None
