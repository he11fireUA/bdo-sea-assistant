from __future__ import annotations

import os
from pathlib import Path


APP_DATA_DIRECTORY_NAME = "BDOSeaAssistant"


def bundle_root() -> Path:
    """Return the directory containing packaged application resources."""
    return Path(__file__).resolve().parents[1]


def asset_path(*parts: str, bundle_root: Path | None = None) -> Path:
    """Resolve a required read-only asset shipped with the application."""
    root = bundle_root if bundle_root is not None else globals()["bundle_root"]()
    path = root / "assets" / Path(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"Bundled resource is missing: {path}")
    return path


def app_data_dir() -> Path:
    """Return the per-user writable application-data directory."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / APP_DATA_DIRECTORY_NAME
