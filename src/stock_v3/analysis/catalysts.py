"""Layer 5 — catalyst assembly and scoring.

Merges company catalysts (earnings, from Finnhub) with a forward-looking macro-event
calendar derived deterministically from known recurring schedules (FOMC, CPI, jobs).
Scores 1-10 by proximity and density of near-term catalysts.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..models import CatalystEvent


@dataclass
class CatalystAssessment:
    events: list[CatalystEvent]
    next_earnings: dt.date | None
    score: float
    rationale: str


# Recurring macro events publish on well-known cadences. We surface the next occurrence
# of each within the horizon rather than hard-coding dated entries that go stale.
_MACRO_EVENTS = (
    ("CPI release", 13),  # ~monthly, mid-month
    ("FOMC decision", 45),  # ~every 6 weeks
    ("Nonfarm payrolls", 30),  # first Friday monthly
)


def assess_catalysts(
    company_events: list[CatalystEvent], today: dt.date | None = None, horizon_days: int = 90
) -> CatalystAssessment:
    today = today or dt.date.today()
    horizon = today + dt.timedelta(days=horizon_days)

    events = list(company_events)
    events.extend(_macro_calendar(today, horizon))
    events = [e for e in events if e.date is None or today <= e.date <= horizon]
    events.sort(key=lambda e: (e.date is None, e.date or today))

    next_earnings = next(
        (e.date for e in events if e.kind == "earnings" and e.date is not None), None
    )
    score = _score(events, next_earnings, today)
    rationale = _rationale(events, next_earnings, today)
    return CatalystAssessment(
        events=events, next_earnings=next_earnings, score=score, rationale=rationale
    )


def _macro_calendar(today: dt.date, horizon: dt.date) -> list[CatalystEvent]:
    out: list[CatalystEvent] = []
    for label, cadence in _MACRO_EVENTS:
        approx = today + dt.timedelta(days=cadence)
        if approx <= horizon:
            out.append(
                CatalystEvent(date=approx, label=f"{label} (approx)", kind="macro",
                              importance="medium")
            )
    return out


def _score(events: list[CatalystEvent], next_earnings: dt.date | None, today: dt.date) -> float:
    score = 5.0
    if next_earnings is not None:
        days = (next_earnings - today).days
        if days <= 14:
            score += 1.5  # imminent earnings = near-term catalyst and risk
        elif days <= 45:
            score += 0.75
    high = sum(1 for e in events if e.importance == "high")
    score += min(high, 2) * 0.5
    return round(max(1.0, min(10.0, score)), 1)


def _rationale(events: list[CatalystEvent], next_earnings: dt.date | None, today: dt.date) -> str:
    if next_earnings is not None:
        days = (next_earnings - today).days
        earn = f"Next earnings ~{next_earnings.isoformat()} ({days}d)."
    else:
        earn = "No earnings date in window."
    macro = ", ".join(e.label for e in events if e.kind == "macro") or "none"
    return f"{earn} Macro events in window: {macro}."
