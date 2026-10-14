from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildIdentity:
    """Immutable identity compiled into a frozen distribution."""

    recipient: str
    build_id: str
    window_title: str
    credit_line: str | None = None


DEFAULT_BUILD_IDENTITY = BuildIdentity(
    recipient="internal",
    build_id="bsa-alpha-generic",
    window_title="BDO Sea Assistant — Alpha",
)
