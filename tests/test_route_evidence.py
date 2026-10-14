import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from bdo_assistant.route_evidence import (
    RouteEvidenceTracker,
    RouteGeometryClass,
    assess_route_geometry,
)


FIXTURES = Path(__file__).parent / "fixtures" / "route_evidence"


def test_route_fixtures_are_compact_privacy_safe_and_integrity_checked() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["scope"] == "2d-minimap-roi-only"
    assert manifest["privacy"] == "no chat, player names, credentials, or full frames"
    for frame in manifest["frames"]:
        payload = (FIXTURES / frame["file"]).read_bytes()
        image = cv2.imread(str(FIXTURES / frame["file"]), cv2.IMREAD_COLOR)
        assert hashlib.sha256(payload).hexdigest() == frame["sha256"]
        assert image is not None
        assert image.shape == (frame["height"], frame["width"], 3)


def _fixture(name: str):
    image = cv2.imread(str(FIXTURES / name), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_route_geometry_is_location_agnostic_across_real_minimap_crops() -> None:
    assert assess_route_geometry(_fixture("stuck_route_ahead.jpg")) is RouteGeometryClass.AHEAD
    assert assess_route_geometry(_fixture("arrival_near_endpoint.png")) is RouteGeometryClass.NEAR
    assert assess_route_geometry(_fixture("freeform_arrival.png")) is RouteGeometryClass.NEAR
    assert assess_route_geometry(_fixture("special_barter_detour.png")) is RouteGeometryClass.AMBIGUOUS


def test_route_geometry_ignores_large_red_component_that_does_not_touch_center() -> None:
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(minimap, (20, 20), (180, 180), (0, 0, 255), 3)

    assert assess_route_geometry(minimap) is RouteGeometryClass.AMBIGUOUS


def test_route_geometry_ignores_long_red_ui_line_only_near_center() -> None:
    minimap = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.line(minimap, (111, 100), (190, 100), (0, 0, 255), 3)

    assert assess_route_geometry(minimap) is RouteGeometryClass.AMBIGUOUS


def test_route_geometry_rejects_large_terrain_colored_region_crossing_center() -> None:
    """A filled island tint must not become evidence of a route ahead."""
    minimap = np.zeros((260, 300, 3), dtype=np.uint8)
    cv2.ellipse(minimap, (168, 135), (75, 105), 0, 0, 360, (20, 75, 145), -1)

    assert assess_route_geometry(minimap) is RouteGeometryClass.AMBIGUOUS


def test_route_evidence_requires_persistent_ahead_then_persistent_near() -> None:
    tracker = RouteEvidenceTracker(confirmations=3)

    assert tracker.update(RouteGeometryClass.AHEAD) == (None, None)
    assert tracker.update(RouteGeometryClass.AHEAD) == (None, None)
    assert tracker.update(RouteGeometryClass.AHEAD) == (False, True)

    assert tracker.update(RouteGeometryClass.NEAR) == (None, None)
    assert tracker.update(RouteGeometryClass.NEAR) == (None, None)
    assert tracker.update(RouteGeometryClass.NEAR) == (True, False)


def test_route_evidence_elapsed_confirmation_uses_observed_time_not_frame_count() -> None:
    tracker = RouteEvidenceTracker(ahead_seconds=2.0, near_seconds=2.0)

    assert tracker.update(RouteGeometryClass.AHEAD, observed_monotonic=0.0) == (None, None)
    assert tracker.update(RouteGeometryClass.AHEAD, observed_monotonic=0.1) == (None, None)
    assert tracker.update(RouteGeometryClass.AHEAD, observed_monotonic=2.0) == (False, True)
    assert tracker.update(RouteGeometryClass.NEAR, observed_monotonic=2.1) == (None, None)
    assert tracker.update(RouteGeometryClass.NEAR, observed_monotonic=4.2) == (True, False)


def test_observability_reset_discards_prior_route_history() -> None:
    tracker = RouteEvidenceTracker(confirmations=3)
    for _ in range(3):
        tracker.update(RouteGeometryClass.AHEAD)

    tracker.reset()

    assert [tracker.update(RouteGeometryClass.NEAR) for _ in range(3)] == [
        (None, None),
        (None, None),
        (None, None),
    ]


def test_tracker_is_runtime_callable_over_minimap_pixels() -> None:
    tracker = RouteEvidenceTracker(confirmations=3)
    ahead = _fixture("stuck_route_ahead.jpg")

    assert tracker(1, ahead) == (None, None)
    assert tracker(2, ahead) == (None, None)
    assert tracker(3, ahead) == (False, True)
