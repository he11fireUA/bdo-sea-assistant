import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "minimap"


def test_production_minimap_anchor_manifest_and_pixels_are_exact() -> None:
    manifest_path = ASSET_ROOT / "manifest.json"
    anchor_path = ASSET_ROOT / "minimap_2d_anchor.png"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = anchor_path.read_bytes()
    image = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)

    assert manifest == {
        "schema_version": 1,
        "filename": "minimap_2d_anchor.png",
        "sha256": "3a4fd67823a3da637b8fb994968937d0c39b0a3c2762b02af0e05f2c920b3593",
        "width": 31,
        "height": 31,
        "channels": 3,
        "dtype": "uint8",
        "source": "tests/fixtures/minimap_2d_anchor.png",
        "scope": "2d-minimap-only",
    }
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
    assert image is not None
    assert image.shape == (manifest["height"], manifest["width"], manifest["channels"])
    assert image.dtype == np.uint8
    assert float(image.std()) >= 1.0


def test_production_anchor_is_byte_identical_to_reviewed_fixture() -> None:
    asset = (ASSET_ROOT / "minimap_2d_anchor.png").read_bytes()
    fixture = (PROJECT_ROOT / "tests" / "fixtures" / "minimap_2d_anchor.png").read_bytes()

    assert asset == fixture
