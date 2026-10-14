from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration_store import CalibrationReadiness, CalibrationStatus
from .window_discovery import WindowCandidate


_EXPECTED_ANCHOR_HASH = "3a4fd67823a3da637b8fb994968937d0c39b0a3c2762b02af0e05f2c920b3593"
_EXPECTED_MANIFEST = {
    "schema_version": 1,
    "filename": "minimap_2d_anchor.png",
    "sha256": _EXPECTED_ANCHOR_HASH,
    "width": 31,
    "height": 31,
    "channels": 3,
    "dtype": "uint8",
    "source": "tests/fixtures/minimap_2d_anchor.png",
    "scope": "2d-minimap-only",
}


def load_verified_minimap_anchor(asset_root: str | Path) -> np.ndarray:
    root = Path(asset_root)
    manifest_path = root / "manifest.json"
    anchor_path = root / "minimap_2d_anchor.png"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = anchor_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load minimap anchor asset: {exc}") from exc
    if (
        type(manifest) is not dict
        or set(manifest) != set(_EXPECTED_MANIFEST)
        or any(
            type(manifest[field]) is not type(expected)
            or manifest[field] != expected
            for field, expected in _EXPECTED_MANIFEST.items()
        )
    ):
        raise ValueError("minimap anchor manifest does not match the production contract")
    if hashlib.sha256(payload).hexdigest() != _EXPECTED_ANCHOR_HASH:
        raise ValueError("minimap anchor SHA-256 mismatch")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if (
        image is None
        or image.shape != (31, 31, 3)
        or image.dtype != np.uint8
        or float(image.std()) < 1.0
    ):
        raise ValueError("minimap anchor pixels violate the production contract")
    image.setflags(write=False)
    return image


@dataclass(frozen=True)
class RuntimeDependencyStatus:
    ready: bool
    blockers: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.ready) is not bool:
            raise ValueError("ready must be boolean")
        if (
            type(self.blockers) is not tuple
            or any(type(blocker) is not str or not blocker for blocker in self.blockers)
            or type(self.limitations) is not tuple
            or any(type(item) is not str or not item for item in self.limitations)
            or self.ready is bool(self.blockers)
        ):
            raise ValueError("runtime blockers are invalid")

    @classmethod
    def assess(
        cls,
        *,
        candidate: WindowCandidate | None,
        calibration: CalibrationReadiness,
        anchor: np.ndarray | None,
        route_evidence_available: bool,
        runtime_factory_available: bool,
    ) -> RuntimeDependencyStatus:
        if candidate is not None and type(candidate) is not WindowCandidate:
            raise ValueError("candidate must be WindowCandidate or None")
        if type(calibration) is not CalibrationReadiness:
            raise ValueError("calibration must be CalibrationReadiness")
        if anchor is not None and (
            not isinstance(anchor, np.ndarray)
            or anchor.ndim != 3
            or anchor.shape[2] != 3
            or anchor.size == 0
            or anchor.dtype != np.uint8
        ):
            raise ValueError("anchor must be a non-empty BGR uint8 image or None")
        if type(route_evidence_available) is not bool:
            raise ValueError("route_evidence_available must be boolean")
        if type(runtime_factory_available) is not bool:
            raise ValueError("runtime_factory_available must be boolean")
        blockers = []
        if candidate is None:
            blockers.append("BDO window not selected")
        if calibration.status is not CalibrationStatus.VALID:
            blockers.append("confirmed calibration missing")
        if anchor is None:
            blockers.append("verified minimap anchor missing")
        limitations = (
            () if route_evidence_available else ("route/endpoint evidence unavailable",)
        )
        if not runtime_factory_available:
            blockers.append("runtime factory unavailable")
        return cls(
            ready=not blockers,
            blockers=tuple(blockers),
            limitations=limitations,
        )
