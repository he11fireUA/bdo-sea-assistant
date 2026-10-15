from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_VERSION_PATTERN = re.compile(r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_RELEASE_API_URL = "https://api.github.com/repos/he11fireUA/bdo-sea-assistant/releases/latest"
_USER_AGENT = "BDOSeaAssistant-update-check/0.1"
_MAX_RESPONSE_BYTES = 1_048_576


class UpdateStatus(Enum):
    CURRENT = "current"
    AVAILABLE = "available"
    NO_RELEASE = "no_release"
    ERROR = "error"


@dataclass(frozen=True)
class UpdateCheckResult:
    status: UpdateStatus
    current_version: str
    latest_version: str | None = None
    release_url: str | None = None
    release_name: str | None = None
    release_notes: str | None = None
    error: str | None = None


def _parse_version(value: str) -> tuple[int, int, int]:
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("version must use MAJOR.MINOR.PATCH")
    return tuple(int(part) for part in match.groups())


def check_for_updates(
    current_version: str,
    *,
    opener: Callable[..., object] = urlopen,
    timeout: float = 5.0,
) -> UpdateCheckResult:
    request = Request(
        _RELEASE_API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": _USER_AGENT},
    )
    try:
        with opener(request, timeout=timeout) as response:
            raw_payload = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw_payload) > _MAX_RESPONSE_BYTES:
                raise ValueError("release response is too large")
            payload = json.loads(raw_payload.decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return UpdateCheckResult(UpdateStatus.NO_RELEASE, current_version)
        return UpdateCheckResult(
            UpdateStatus.ERROR,
            current_version,
            error=f"GitHub returned HTTP {exc.code}",
        )
    except URLError as exc:
        return UpdateCheckResult(UpdateStatus.ERROR, current_version, error=str(exc.reason))
    except TimeoutError as exc:
        return UpdateCheckResult(UpdateStatus.ERROR, current_version, error=str(exc))
    except OSError as exc:
        return UpdateCheckResult(UpdateStatus.ERROR, current_version, error=str(exc))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return UpdateCheckResult(
            UpdateStatus.ERROR,
            current_version,
            error="Invalid GitHub release metadata",
        )
    try:
        if not isinstance(payload, Mapping):
            raise ValueError("release response must be an object")
        return check_release_payload(current_version, payload)
    except ValueError:
        return UpdateCheckResult(
            UpdateStatus.ERROR,
            current_version,
            error="Invalid GitHub release metadata",
        )


def check_release_payload(
    current_version: str,
    payload: Mapping[str, object],
) -> UpdateCheckResult:
    current = _parse_version(current_version)
    raw_tag = payload.get("tag_name")
    raw_url = payload.get("html_url")
    if type(raw_tag) is not str or type(raw_url) is not str:
        raise ValueError("release metadata is incomplete")
    parsed_url = urlparse(raw_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc.lower() != "github.com"
        or not parsed_url.path.lower().startswith("/he11fireua/bdo-sea-assistant/releases/")
    ):
        raise ValueError("release URL is outside the project repository")
    latest = _parse_version(raw_tag)
    latest_text = raw_tag.removeprefix("v")
    return UpdateCheckResult(
        status=UpdateStatus.AVAILABLE if latest > current else UpdateStatus.CURRENT,
        current_version=current_version,
        latest_version=latest_text,
        release_url=raw_url,
        release_name=payload.get("name") if type(payload.get("name")) is str else latest_text,
        release_notes=payload.get("body") if type(payload.get("body")) is str else "",
    )
