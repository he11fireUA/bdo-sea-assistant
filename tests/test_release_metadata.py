import hashlib
from pathlib import Path

import pytest

from bdo_assistant.build_identity import DEFAULT_BUILD_IDENTITY
from bdo_assistant.version import __version__
from tools.release_metadata import (
    artifact_names,
    forbidden_release_entries,
    parse_release_tag,
    project_version,
    verify_sha256sums,
    version_mismatches,
    write_sha256sums,
)


def test_release_tag_uses_strict_v_semver() -> None:
    assert parse_release_tag("v0.1.0") == "0.1.0"
    assert parse_release_tag("v12.34.56") == "12.34.56"

    for invalid in ("0.1.0", "v01.1.0", "v1.02.3", "v1.2", "v1.2.3-beta", "refs/tags/v1.2.3"):
        with pytest.raises(ValueError, match="vMAJOR.MINOR.PATCH"):
            parse_release_tag(invalid)


def test_release_version_matches_every_packaging_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    version = project_version(root)

    assert version == "0.2.3"
    assert version_mismatches(root, version) == []


def test_default_window_title_includes_the_runtime_version() -> None:
    assert DEFAULT_BUILD_IDENTITY.window_title == f"BDO Sea Assistant — Alpha {__version__}"


def test_release_artifact_names_are_derived_from_the_version() -> None:
    assert artifact_names("0.1.0") == (
        "BDOSeaAssistant-0.1.0-win-x64-portable.zip",
        "BDOSeaAssistant-0.1.0-win-x64-Setup.exe",
        "SHA256SUMS.txt",
    )


def test_checksum_manifest_is_deterministic_and_detects_tampering(tmp_path: Path) -> None:
    portable = tmp_path / "portable.zip"
    installer = tmp_path / "setup.exe"
    portable.write_bytes(b"portable")
    installer.write_bytes(b"installer")
    manifest = tmp_path / "SHA256SUMS.txt"

    write_sha256sums((installer, portable), manifest)

    assert manifest.read_text(encoding="ascii").splitlines() == [
        f"{hashlib.sha256(b'portable').hexdigest()}  portable.zip",
        f"{hashlib.sha256(b'installer').hexdigest()}  setup.exe",
    ]
    verify_sha256sums((portable, installer), manifest)

    installer.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch for setup.exe"):
        verify_sha256sums((portable, installer), manifest)


def test_checksum_verification_rejects_duplicate_entries(tmp_path: Path) -> None:
    artifact = tmp_path / "portable.zip"
    artifact.write_bytes(b"portable")
    digest = hashlib.sha256(b"portable").hexdigest()
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text(
        f"{digest}  portable.zip\n{digest}  portable.zip\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="Duplicate checksum entry for portable.zip"):
        verify_sha256sums((artifact,), manifest)


def test_checksum_verification_rejects_duplicate_input_names(tmp_path: Path) -> None:
    first = tmp_path / "one" / "artifact.zip"
    second = tmp_path / "two" / "artifact.zip"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manifest = tmp_path / "SHA256SUMS.txt"
    write_sha256sums((first,), manifest)

    with pytest.raises(ValueError, match="unique file names"):
        verify_sha256sums((first, second), manifest)


def test_release_privacy_scan_reports_local_and_secret_material(tmp_path: Path) -> None:
    (tmp_path / "BDOSeaAssistant.exe").write_bytes(b"safe")
    (tmp_path / "_internal" / "assets").mkdir(parents=True)
    (tmp_path / "_internal" / "assets" / "app_icon.ico").write_bytes(b"safe")
    (tmp_path / "captures").mkdir()
    (tmp_path / "captures" / "frame.png").write_bytes(b"private")
    (tmp_path / ".env.local").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "signing.pfx").write_bytes(b"private")

    assert forbidden_release_entries(tmp_path) == [
        ".env.local",
        "captures",
        "captures/frame.png",
        "signing.pfx",
    ]
