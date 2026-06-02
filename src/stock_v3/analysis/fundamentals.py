"""Layer 3 — fundamental and valuation assessment.

Produces two scorecard inputs: a fundamentals score (growth, margins, balance sheet,
FCF) and a valuation score (multiples vs. rough sector-neutral thresholds, since free
data gives no reliable peer set). Pure functions over a Fundamentals model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Fundamentals


@dataclass
class FundamentalAssessment:
    growth_class: str
    margin_quality: str
    balance_sheet: str
    fcf_class: str
    valuation_class: str
    fundamentals_score: float  # 1..10
    valuation_score: float  # 1..10
    fundamentals_rationale: str
    valuation_rationale: str


def assess_fundamentals(f: Fundamentals) -> FundamentalAssessment:
    growth_class = _growth_class(f.revenue_growth_yoy)
    margin_quality = _margin_quality(f.operating_margin, f.net_margin)
    balance_sheet = _balance_sheet(f.net_debt, f.current_ratio, f.cash)
    fcf_class = _fcf_class(f.fcf_margin, f.fcf_ttm)
    valuation_class = _valuation_class(f)

    f_score = _fundamentals_score(f, growth_class, margin_quality, balance_sheet, fcf_class)
    v_score = _valuation_score(f, growth_class)

    f_rationale = (
        f"Revenue growth {_pct(f.revenue_growth_yoy)} ({growth_class}); "
        f"op margin {_pct(f.operating_margin)}, net {_pct(f.net_margin)} ({margin_quality}); "
        f"balance sheet {balance_sheet}; FCF {fcf_class}."
    )
    v_rationale = (
        f"P/E {_num(f.pe)}, fwd P/E {_num(f.forward_pe)}, EV/EBITDA {_num(f.ev_ebitda)}, "
        f"P/S {_num(f.price_to_sales)}, PEG {_num(f.peg)} → {valuation_class}."
    )
    return FundamentalAssessment(
        growth_class=growth_class,
        margin_quality=margin_quality,
        balance_sheet=balance_sheet,
        fcf_class=fcf_class,
        valuation_class=valuation_class,
        fundamentals_score=f_score,
        valuation_score=v_score,
        fundamentals_rationale=f_rationale,
        valuation_rationale=v_rationale,
    )


def _growth_class(growth: float | None) -> str:
    if growth is None:
        return "Unknown"
    if growth >= 0.40:
        return "Hyper Growth"
    if growth >= 0.15:
        return "Growth"
    if growth >= 0.0:
        return "Stable"
    return "Contracting"


def _margin_quality(op: float | None, net: float | None) -> str:
    ref = op if op is not None else net
    if ref is None:
        return "Unknown"
    if ref >= 0.25:
        return "Exceptional"
    if ref >= 0.12:
        return "Healthy"
    if ref >= 0.0:
        return "Thin"
    return "Unprofitable"


def _balance_sheet(net_debt: float | None, current_ratio: float | None, cash: float | None) -> str:
    if net_debt is not None and net_debt < 0:
        return "Net cash"
    flags = []
    if current_ratio is not None and current_ratio < 1.0:
        flags.append("liquidity tight")
    if net_debt is not None and cash is not None and cash > 0 and net_debt > 3 * cash:
        flags.append("leveraged")
    if not flags:
        return "Solid"
    return ", ".join(flags).capitalize()


def _fcf_class(fcf_margin: float | None, fcf_ttm: float | None) -> str:
    if fcf_ttm is None:
        return "Unknown"
    if fcf_ttm < 0:
        return "Cash-burning"
    if fcf_margin is None:
        return "Positive"
    if fcf_margin >= 0.20:
        return "Strong"
    if fcf_margin >= 0.08:
        return "Improving"
    return "Neutral"


def _valuation_class(f: Fundamentals) -> str:
    """Coarse rich/cheap read. Free data lacks a clean peer set, so we lean on PEG and
    absolute multiple bands — explicitly a heuristic, surfaced in the rationale."""
    signals = []
    if f.peg is not None:
        if f.peg < 1.0:
            signals.append(-1)
        elif f.peg > 2.0:
            signals.append(1)
    if f.forward_pe is not None:
        if f.forward_pe < 15:
            signals.append(-1)
        elif f.forward_pe > 35:
            signals.append(1)
    if f.ev_ebitda is not None:
        if f.ev_ebitda < 10:
            signals.append(-1)
        elif f.ev_ebitda > 25:
            signals.append(1)
    if not signals:
        return "Fairly Valued"
    net = sum(signals)
    if net <= -2:
        return "Undervalued"
    if net >= 2:
        return "Overvalued"
    return "Fairly Valued"


def _fundamentals_score(
    f: Fundamentals, growth_class: str, margin_quality: str, balance_sheet: str, fcf_class: str
) -> float:
    score = 5.0
    score += {"Hyper Growth": 2.5, "Growth": 1.5, "Stable": 0.0,
              "Contracting": -2.0, "Unknown": 0.0}[growth_class]
    score += {"Exceptional": 1.5, "Healthy": 0.5, "Thin": -0.5,
              "Unprofitable": -2.0, "Unknown": 0.0}[margin_quality]
    score += {"Strong": 1.0, "Improving": 0.5, "Positive": 0.3,
              "Neutral": 0.0, "Cash-burning": -1.5, "Unknown": 0.0}[fcf_class]
    if balance_sheet == "Net cash":
        score += 0.5
    elif "tight" in balance_sheet.lower() or "leveraged" in balance_sheet.lower():
        score -= 1.0
    return round(max(1.0, min(10.0, score)), 1)


def _valuation_score(f: Fundamentals, growth_class: str) -> float:
    """Higher score = more attractively valued. PEG-aware so high-growth names aren't
    auto-penalized for a high P/E."""
    score = 5.0
    if f.peg is not None:
        if f.peg < 1.0:
            score += 2.0
        elif f.peg < 1.5:
            score += 1.0
        elif f.peg > 2.5:
            score -= 1.5
        elif f.peg > 2.0:
            score -= 0.75
    if f.forward_pe is not None:
        if f.forward_pe < 15:
            score += 1.0
        elif f.forward_pe > 40:
            score -= 1.5
        elif f.forward_pe > 30:
            score -= 0.75
    if f.price_to_fcf is not None and f.price_to_fcf > 0:
        if f.price_to_fcf < 20:
            score += 0.75
        elif f.price_to_fcf > 60:
            score -= 1.0
    return round(max(1.0, min(10.0, score)), 1)


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "n/a"


def _num(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "n/a"
