from __future__ import annotations

from pathlib import Path

import pytest

from bdo_assistant import app_paths


def test_asset_path_is_independent_of_current_working_directory(monkeypatch, tmp_path: Path) -> None:
    bundle_root = tmp_path / "пакет із пробілом"
    asset = bundle_root / "assets" / "voice" / "uk" / "test.wav"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"wave")
    monkeypatch.chdir(tmp_path)

    assert app_paths.asset_path("voice", "uk", "test.wav", bundle_root=bundle_root) == asset


def test_asset_path_rejects_missing_resource(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Bundled resource is missing"):
        app_paths.asset_path("minimap", "missing.png", bundle_root=tmp_path)


def test_app_data_dir_uses_local_app_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local app data"))

    assert app_paths.app_data_dir() == tmp_path / "local app data" / "BDOSeaAssistant"
