import hashlib
import json
from pathlib import Path
import wave

import pytest

PACK_ROOT = Path(__file__).parents[1] / "assets" / "voice"


@pytest.mark.parametrize("locale", ["uk", "en"])
def test_voice_pack_manifest_integrity_and_audio_contract(locale: str) -> None:
    pack = PACK_ROOT / locale
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    clips = manifest["clips"]

    assert manifest["schema_version"] == 1
    assert manifest["locale"] == locale
    assert set(clips) == {
        "arrived",
        "stuck",
        "ship_hp_low",
        "rations_low",
        "capture_lost",
        "capture_restored",
        "minimap_visibility_lost",
        "minimap_visibility_restored",
        "ui_calibration_lost",
        "test",
    }
    assert sum((pack / item["file"]).stat().st_size for item in clips.values()) < 2_000_000

    for item in clips.values():
        path = pack / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        with wave.open(str(path), "rb") as source:
            assert source.getnchannels() == 1
            assert source.getsampwidth() == 2
            assert source.getframerate() == 48000
            duration = source.getnframes() / source.getframerate()
        assert 0.3 <= duration <= 3.0
