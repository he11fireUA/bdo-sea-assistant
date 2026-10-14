from pathlib import Path
import json
import shutil

import cv2
import numpy as np

from bdo_assistant.calibration_store import CalibrationReadiness, CalibrationStatus
from bdo_assistant.runtime_readiness import (
    RuntimeDependencyStatus,
    load_verified_minimap_anchor,
)
from bdo_assistant.window_discovery import WindowCandidate


PROJECT_ROOT = Path(__file__).parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "minimap"


def _candidate() -> WindowCandidate:
    return WindowCandidate(1, 2, "BLACK DESERT", "BlackDesert64.exe", 2560, 1440, False)


def _calibration() -> CalibrationReadiness:
    return CalibrationReadiness(
        CalibrationStatus.VALID,
        "confirmed calibration loaded",
        layout={"schema_version": 1},
    )


def test_verified_anchor_loader_returns_read_only_pixels() -> None:
    image = load_verified_minimap_anchor(ASSET_ROOT)

    assert image.shape == (31, 31, 3)
    assert image.dtype == np.uint8
    assert not image.flags.writeable


def test_anchor_manifest_rejects_bool_or_float_substitutions(tmp_path) -> None:
    shutil.copytree(ASSET_ROOT, tmp_path / "minimap")
    manifest_path = tmp_path / "minimap" / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, invalid in (("schema_version", True), ("width", 31.0)):
        modified = dict(original)
        modified[field] = invalid
        manifest_path.write_text(json.dumps(modified), encoding="utf-8")
        try:
            load_verified_minimap_anchor(tmp_path / "minimap")
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed manifest field {field}")


def test_runtime_readiness_lists_every_missing_dependency() -> None:
    status = RuntimeDependencyStatus.assess(
        candidate=None,
        calibration=CalibrationReadiness(
            CalibrationStatus.MISSING,
            "calibration file not found",
        ),
        anchor=None,
        route_evidence_available=False,
        runtime_factory_available=False,
    )

    assert not status.ready
    assert status.blockers == (
        "BDO window not selected",
        "confirmed calibration missing",
        "verified minimap anchor missing",
        "runtime factory unavailable",
    )
    assert status.limitations == ("route/endpoint evidence unavailable",)


def test_runtime_readiness_is_true_only_when_all_dependencies_exist() -> None:
    anchor = load_verified_minimap_anchor(ASSET_ROOT)

    status = RuntimeDependencyStatus.assess(
        candidate=_candidate(),
        calibration=_calibration(),
        anchor=anchor,
        route_evidence_available=True,
        runtime_factory_available=True,
    )

    assert status.ready
    assert status.blockers == ()
    assert status.limitations == ()


def test_runtime_readiness_rejects_malformed_dependency_flags_and_anchor() -> None:
    valid = {
        "candidate": _candidate(),
        "calibration": _calibration(),
        "anchor": load_verified_minimap_anchor(ASSET_ROOT),
        "route_evidence_available": True,
        "runtime_factory_available": True,
    }
    invalid = (
        {**valid, "route_evidence_available": 1},
        {**valid, "runtime_factory_available": "yes"},
        {**valid, "anchor": np.zeros((2, 2), dtype=np.uint8)},
        {**valid, "candidate": object()},
    )
    for values in invalid:
        try:
            RuntimeDependencyStatus.assess(**values)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted malformed runtime dependency")
