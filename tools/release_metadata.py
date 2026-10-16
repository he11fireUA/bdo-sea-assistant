from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Iterable


_RELEASE_TAG = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def parse_release_tag(tag: str) -> str:
    """Return the version from a strict stable release tag."""
    match = _RELEASE_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"Release tag must use strict vMAJOR.MINOR.PATCH syntax: {tag}")
    return ".".join(match.groups())


def project_version(root: Path) -> str:
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(metadata["project"]["version"])


def artifact_names(version: str) -> tuple[str, str, str]:
    return (
        f"BDOSeaAssistant-{version}-win-x64-portable.zip",
        f"BDOSeaAssistant-{version}-win-x64-Setup.exe",
        "SHA256SUMS.txt",
    )


def version_mismatches(root: Path, version: str) -> list[str]:
    """Return release-facing files that do not contain the expected version."""
    major, minor, patch = version.split(".")
    numeric = rf"\({major},\s*{minor},\s*{patch},\s*0\)"
    portable, installer, _checksums = artifact_names(version)
    expected_patterns = {
        "bdo_assistant/version.py": rf'__version__\s*=\s*"{re.escape(version)}"',
        "uv.lock": rf'name = "bdo-sea-assistant"\s+version = "{re.escape(version)}"',
        "packaging/windows-version-info.txt": (
            rf"filevers={numeric}.*prodvers={numeric}.*"
            rf"FileVersion', u'{re.escape(version)}'.*ProductVersion', u'{re.escape(version)}'"
        ),
        "packaging/installer.iss": (
            rf'#define MyAppVersion "{re.escape(version)}".*'
            rf"OutputBaseFilename={re.escape(installer.removesuffix('.exe'))}"
        ),
        "scripts/build_windows.ps1": re.escape(portable),
        "scripts/build_installer.ps1": re.escape(installer),
        "scripts/build_personalized_windows.ps1": rf"BDOSeaAssistant-{re.escape(version)}-win-x64-portable\.zip",
        "docs/distribution.md": rf"{re.escape(portable)}.*{re.escape(installer)}",
    }
    mismatches: list[str] = []
    for relative_path, pattern in expected_patterns.items():
        path = root / relative_path
        if not path.is_file() or re.search(pattern, path.read_text(encoding="utf-8"), re.DOTALL) is None:
            mismatches.append(relative_path)
    return mismatches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(files: Iterable[Path], output: Path) -> None:
    paths = sorted((Path(path) for path in files), key=lambda path: path.name)
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Release artifacts must have unique file names")
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def verify_sha256sums(files: Iterable[Path], manifest: Path) -> None:
    paths = tuple(Path(path) for path in files)
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Release artifacts must have unique file names")
    actual_files = {path.name: path for path in paths}
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None:
            raise ValueError(f"Malformed checksum line: {line}")
        name = match.group(2)
        if name in expected:
            raise ValueError(f"Duplicate checksum entry for {name}")
        expected[name] = match.group(1)
    if set(expected) != set(actual_files):
        raise ValueError("Checksum manifest does not match the release artifact set")
    for name, path in actual_files.items():
        if _sha256(path) != expected[name]:
            raise ValueError(f"Checksum mismatch for {name}")


def forbidden_release_entries(root: Path) -> list[str]:
    forbidden_directories = {".git", ".hermes", "captures", "journeys", "logs", "recordings"}
    forbidden_suffixes = {".key", ".pem", ".pfx", ".p12"}
    findings: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        parts = {part.casefold() for part in relative.parts}
        name = path.name.casefold()
        if (
            parts.intersection(forbidden_directories)
            or name == ".env"
            or name.startswith(".env.")
            or path.suffix.casefold() in forbidden_suffixes
        ):
            findings.append(relative.as_posix())
    return sorted(findings)
