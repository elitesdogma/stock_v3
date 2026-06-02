"""Layer 2 — institutional positioning assessment.

Assembles the Positioning model from already-fetched pieces (institutional holders,
Form 4 insider events, short interest) and scores it 1-10. Insider weighting follows the
framework: CEO/founder buys carry the most signal, routine sales the least. Pure: it takes
plain values (the caller has already unwrapped SourceResults) and returns an assessment.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..models import InsiderEvent, InstitutionalHolder, Positioning


@dataclass
class PositioningAssessment:
    positioning: Positioning
    institution_stance: str  # Accumulating / Neutral / Distributing
    insider_signal: str
    squeeze_read: str
    score: float  # 1..10
    rationale: str


def assess_positioning(
    holders: list[InstitutionalHolder] | None,
    holders_vintage: dt.date | None,
    insiders: list[InsiderEvent] | None,
    short_pct_float: float | None,
    days_to_cover: float | None,
    short_settlement: dt.date | None,
    borrow_rate: float | None,
) -> PositioningAssessment:
    holders = holders or []
    insiders = insiders or []

    inst_change = _net_institutional_change(holders)
    insider_net = _insider_net_value(insiders)

    positioning = Positioning(
        institutional_holders=holders,
        institutional_total_change=inst_change,
        holders_vintage=holders_vintage,
        insider_events=insiders,
        insider_net_value_90d=insider_net,
        short_interest_pct_float=short_pct_float,
        days_to_cover=days_to_cover,
        short_interest_settlement=short_settlement,
        borrow_rate=borrow_rate,
    )

    stance = _institution_stance(inst_change)
    insider_signal = _insider_signal(insiders)
    squeeze = _squeeze_read(short_pct_float, days_to_cover)
    score = _score(stance, insider_signal, short_pct_float, days_to_cover)

    rationale = (
        f"Institutions {stance.lower()}; insider flow: {insider_signal}; "
        f"short interest {_pct(short_pct_float)} of float, "
        f"{_n(days_to_cover)} days to cover ({squeeze})."
    )
    return PositioningAssessment(
        positioning=positioning,
        institution_stance=stance,
        insider_signal=insider_signal,
        squeeze_read=squeeze,
        score=score,
        rationale=rationale,
    )


def _net_institutional_change(holders: list[InstitutionalHolder]) -> float | None:
    changes = [h.change for h in holders if h.change is not None]
    if not changes:
        return None
    return round(sum(changes), 4)


def _institution_stance(net_change: float | None) -> str:
    if net_change is None:
        return "Neutral"
    if net_change > 0.02:
        return "Accumulating"
    if net_change < -0.02:
        return "Distributing"
    return "Neutral"


def _insider_net_value(insiders: list[InsiderEvent]) -> float | None:
    """Weighted buy-minus-sell using share counts (Form 4 dollar values are unreliable on
    free data). Weighting elevates senior-role purchases over routine activity."""
    if not insiders:
        return None
    net = 0.0
    for e in insiders:
        weight = _role_weight(e.role)
        signed = e.shares if e.transaction == "buy" else -e.shares
        net += weight * signed
    return round(net, 2)


def _role_weight(role: str) -> float:
    r = role.lower()
    if any(k in r for k in ("ceo", "chief executive", "founder", "chair")):
        return 2.0
    if any(k in r for k in ("cfo", "president", "coo", "officer")):
        return 1.5
    if "director" in r:
        return 1.2
    return 1.0


def _insider_signal(insiders: list[InsiderEvent]) -> str:
    if not insiders:
        return "no recent activity"
    buys = [e for e in insiders if e.transaction == "buy"]
    sells = [e for e in insiders if e.transaction == "sell"]
    senior_buys = [e for e in buys if _role_weight(e.role) >= 1.5]
    if senior_buys:
        return f"senior insider buying ({len(senior_buys)} buys)"
    if len(buys) > len(sells):
        return f"net buying ({len(buys)} buys vs {len(sells)} sells)"
    if len(sells) > len(buys):
        return f"net selling ({len(sells)} sells vs {len(buys)} buys)"
    return "mixed"


def _squeeze_read(short_pct: float | None, dtc: float | None) -> str:
    if short_pct is None and dtc is None:
        return "n/a"
    high_short = short_pct is not None and short_pct > 0.15
    crowded = dtc is not None and dtc > 7
    if high_short and crowded:
        return "elevated squeeze potential"
    if high_short or crowded:
        return "moderate squeeze potential"
    return "low / not crowded"


def _score(
    stance: str, insider_signal: str, short_pct: float | None, dtc: float | None
) -> float:
    score = 5.0
    score += {"Accumulating": 1.5, "Neutral": 0.0, "Distributing": -1.5}[stance]

    if "senior insider buying" in insider_signal:
        score += 2.0
    elif "net buying" in insider_signal:
        score += 1.0
    elif "net selling" in insider_signal:
        score -= 1.0

    # A crowded short is a double-edged sentiment signal: bearish positioning, but squeeze
    # fuel. We treat extreme short interest as a mild net negative on conviction.
    if short_pct is not None and short_pct > 0.20:
        score -= 0.5

    return round(max(1.0, min(10.0, score)), 1)


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "n/a"


def _n(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "n/a"
