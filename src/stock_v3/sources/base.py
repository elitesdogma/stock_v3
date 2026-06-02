"""The SourceResult contract.

Every data fetcher returns a SourceResult — never a bare value. This is how the
"honestly mark gaps" requirement is enforced structurally: a source that fails,
or that free data simply cannot provide, returns status=UNAVAILABLE with a reason
instead of raising or fabricating. Analysis layers branch on status; the report's
coverage panel reads it directly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class Status(str, Enum):
    OK = "ok"  # fresh, usable data
    STALE = "stale"  # usable but older than ideal (e.g. bi-monthly short interest, 45-day-old 13F)
    UNAVAILABLE = "unavailable"  # could not obtain — missing key, API error, or not free


@dataclass(frozen=True)
class SourceResult(Generic[T]):
    value: T | None
    status: Status
    source: str
    as_of: dt.date | dt.datetime | None = None
    note: str | None = None  # human-readable freshness caveat or failure reason
    field_gaps: tuple[str, ...] = field(default_factory=tuple)  # named fields within value that are N/A

    @property
    def ok(self) -> bool:
        return self.status is Status.OK and self.value is not None

    @property
    def usable(self) -> bool:
        """OK or STALE — the analysis layer can still derive signal from it."""
        return self.status in (Status.OK, Status.STALE) and self.value is not None

    def unwrap_or(self, default: T) -> T:
        return self.value if self.value is not None else default

    @classmethod
    def of(
        cls,
        value: T,
        source: str,
        *,
        as_of: dt.date | dt.datetime | None = None,
        stale: bool = False,
        note: str | None = None,
        field_gaps: tuple[str, ...] = (),
    ) -> "SourceResult[T]":
        return cls(
            value=value,
            status=Status.STALE if stale else Status.OK,
            source=source,
            as_of=as_of,
            note=note,
            field_gaps=field_gaps,
        )

    @classmethod
    def unavailable(cls, source: str, reason: str) -> "SourceResult[T]":
        return cls(value=None, status=Status.UNAVAILABLE, source=source, note=reason)


def freshness_label(result: SourceResult[object]) -> str:
    """Compact 'as-of' string for the coverage panel."""
    if result.status is Status.UNAVAILABLE:
        return "N/A"
    if result.as_of is None:
        return "unknown"
    if isinstance(result.as_of, dt.datetime):
        return result.as_of.strftime("%Y-%m-%d %H:%M")
    return result.as_of.isoformat()
