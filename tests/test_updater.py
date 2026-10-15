from pathlib import Path
import tomllib
from urllib.error import HTTPError, URLError

from bdo_assistant.updater import UpdateStatus, check_for_updates, check_release_payload
from bdo_assistant.version import __version__


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


def test_python_version_matches_project_metadata() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert __version__ == metadata["project"]["version"]


def test_newer_github_release_is_reported_as_available() -> None:
    result = check_release_payload(
        "0.1.0",
        {
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/he11fireUA/bdo-sea-assistant/releases/tag/v0.2.0",
            "name": "BDO Sea Assistant 0.2.0",
            "body": "Safer update checks.",
        },
    )

    assert result.status is UpdateStatus.AVAILABLE
    assert result.current_version == "0.1.0"
    assert result.latest_version == "0.2.0"
    assert result.release_url.endswith("/v0.2.0")
    assert result.release_name == "BDO Sea Assistant 0.2.0"
    assert result.release_notes == "Safer update checks."


def test_same_or_older_release_keeps_the_current_install() -> None:
    same = check_release_payload(
        "0.2.0",
        {"tag_name": "v0.2.0", "html_url": "https://github.com/he11fireUA/bdo-sea-assistant/releases/tag/v0.2.0"},
    )
    older = check_release_payload(
        "0.2.0",
        {"tag_name": "v0.1.9", "html_url": "https://github.com/he11fireUA/bdo-sea-assistant/releases/tag/v0.1.9"},
    )

    assert same.status is UpdateStatus.CURRENT
    assert older.status is UpdateStatus.CURRENT


def test_missing_github_release_is_not_a_network_error() -> None:
    def no_release(_request, *, timeout):
        assert timeout == 5.0
        raise HTTPError("https://api.example.test/releases/latest", 404, "Not Found", {}, None)

    result = check_for_updates("0.1.0", opener=no_release)

    assert result.status is UpdateStatus.NO_RELEASE
    assert result.current_version == "0.1.0"
    assert result.error is None


def test_network_failure_returns_bounded_error_result() -> None:
    def offline(_request, *, timeout):
        raise URLError("offline")

    result = check_for_updates("0.1.0", opener=offline)

    assert result.status is UpdateStatus.ERROR
    assert result.current_version == "0.1.0"
    assert result.error == "offline"


def test_timeout_returns_error_result() -> None:
    def timed_out(_request, *, timeout):
        assert timeout == 5.0
        raise TimeoutError("timed out")

    result = check_for_updates("0.1.0", opener=timed_out)

    assert result.status is UpdateStatus.ERROR
    assert result.error == "timed out"


def test_malformed_release_metadata_returns_error_result() -> None:
    result = check_for_updates(
        "0.1.0",
        opener=lambda _request, **_kwargs: _Response(
            b'{"tag_name":"not-a-version","html_url":"https://example.test/release"}'
        ),
    )

    assert result.status is UpdateStatus.ERROR
    assert result.error == "Invalid GitHub release metadata"


def test_oversized_github_response_is_rejected() -> None:
    valid = b'{"tag_name":"v0.2.0","html_url":"https://github.com/he11fireUA/bdo-sea-assistant/releases/tag/v0.2.0"}'
    result = check_for_updates(
        "0.1.0",
        opener=lambda _request, **_kwargs: _Response(valid + b" " * 1_048_576),
    )

    assert result.status is UpdateStatus.ERROR
    assert result.error == "Invalid GitHub release metadata"


def test_release_url_must_stay_on_the_project_github_repository() -> None:
    result = check_for_updates(
        "0.1.0",
        opener=lambda _request, **_kwargs: _Response(
            b'{"tag_name":"v0.2.0","html_url":"https://evil.example/update.exe"}'
        ),
    )

    assert result.status is UpdateStatus.ERROR
    assert result.error == "Invalid GitHub release metadata"
