"""Multi-ticker comparison view — a succinct side-by-side summary across tickers.

Distills each full ResearchReport into the decision-grade row (verdict, conviction, R:R,
key fundamentals/technicals, scenario skew) and lays them out as AXIS comparison cards +
a sortable metric table. Best metric in each row is highlighted so the eye finds the leader.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import ResearchReport
from .render import (
    _fmt_big,
    _fmt_money,
    _fmt_n,
    _fmt_pct,
)

_TEMPLATES = Path(__file__).parent / "templates"
_AXIS_CSS = (Path(__file__).parent / "assets" / "axis.css").read_text(encoding="utf-8")

_VERDICT_CLASS = {
    "Strong Buy": "v-strongbuy", "Buy": "v-buy", "Accumulate": "v-accumulate",
    "Watchlist": "v-watchlist", "Hold": "v-hold", "Avoid": "v-avoid",
}

# Verdict rank for "best" detection (higher = more bullish).
_VERDICT_RANK = {
    "Strong Buy": 6, "Buy": 5, "Accumulate": 4, "Watchlist": 3, "Hold": 2, "Avoid": 1,
}


def render_comparison(reports: list[ResearchReport]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("compare.html.j2")

    cols = [_column(r) for r in reports]
    rows = _metric_rows(reports)

    return template.render(
        axis_css=_AXIS_CSS,
        cols=cols,
        rows=rows,
        n=len(reports),
        verdict_class=_VERDICT_CLASS,
        generated_at=reports[0].generated_at if reports else None,
    )


def _column(r: ResearchReport) -> dict:
    """The per-ticker header card summary."""
    upside = None
    if r.consensus and r.consensus.target_mean:
        upside = (r.consensus.target_mean / r.quote.price - 1) * 100
    return {
        "ticker": r.quote.ticker,
        "company": r.quote.company,
        "price": _fmt_money(r.quote.price),
        "sector": r.quote.sector or "—",
        "verdict": r.verdict.label,
        "verdict_class": _VERDICT_CLASS.get(r.verdict.label, "v-hold"),
        "conviction": r.verdict.conviction_score,
        "confidence": r.verdict.confidence,
        "upside": f"{upside:+.0f}%" if upside is not None else "—",
    }


def _metric_rows(reports: list[ResearchReport]) -> list[dict]:
    """Each row is one metric across all tickers, with the 'best' cell flagged.

    `direction` says whether higher (+1) or lower (-1) is better for highlighting; 0 = no
    winner (neutral/descriptive)."""
    specs: list[tuple[str, callable, int, callable]] = [
        ("Conviction", lambda r: r.verdict.conviction_score, +1, lambda v: f"{v:.0f}/100"),
        ("Confidence", lambda r: r.verdict.confidence, +1, lambda v: f"{v:.0f}/100"),
        ("Verdict", lambda r: _VERDICT_RANK.get(r.verdict.label, 0), +1,
         None),  # rendered specially below
        ("Risk / Reward", lambda r: r.trade_plan.risk_reward, +1,
         lambda v: f"{v:.1f}:1" if v else "—"),
        ("Rev growth (YoY)", lambda r: _f(r.fundamentals, "revenue_growth_yoy"), +1,
         lambda v: _fmt_pct(v)),
        ("Operating margin", lambda r: _f(r.fundamentals, "operating_margin"), +1,
         lambda v: _fmt_pct(v)),
        ("FCF margin", lambda r: _f(r.fundamentals, "fcf_margin"), +1, lambda v: _fmt_pct(v)),
        ("Forward P/E", lambda r: _f(r.fundamentals, "forward_pe"), -1, lambda v: _fmt_n(v)),
        ("PEG", lambda r: _f(r.fundamentals, "peg"), -1, lambda v: _fmt_n(v)),
        ("RSI (14)", lambda r: _f(r.technicals, "rsi_14"), 0, lambda v: _fmt_n(v)),
        ("Short % float", lambda r: _short(r), -1, lambda v: _fmt_pct(v)),
        ("Annualized vol", lambda r: _vol(r), 0, lambda v: _fmt_pct(v)),
        ("Bull / Bear prob", lambda r: _skew(r), +1, None),  # rendered specially
    ]

    rows: list[dict] = []
    for label, getter, direction, fmt in specs:
        raw = [getter(r) for r in reports]
        cells = []
        best_idx = _best_index(raw, direction) if direction != 0 else None
        for i, (r, val) in enumerate(zip(reports, raw)):
            if label == "Verdict":
                text = r.verdict.label
            elif label == "Bull / Bear prob":
                text = _skew_text(r)
            else:
                text = fmt(val) if val is not None else "—"
            cells.append({"text": text, "best": best_idx is not None and i == best_idx})
        rows.append({"label": label, "cells": cells})
    return rows


def _best_index(values: list, direction: int) -> int | None:
    pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pairs) < 2:
        return None
    if direction > 0:
        return max(pairs, key=lambda p: p[1])[0]
    return min(pairs, key=lambda p: p[1])[0]


def _f(obj, attr: str):
    return getattr(obj, attr, None) if obj is not None else None


def _short(r: ResearchReport):
    return r.positioning.short_interest_pct_float if r.positioning else None


def _vol(r: ResearchReport):
    return r.probability_cone.annual_vol if r.probability_cone else None


def _skew(r: ResearchReport) -> float:
    """Bull-minus-bear probability — positive = bullish scenario skew."""
    bull = next((s.probability for s in r.scenarios if s.name == "Bull"), 0.0)
    bear = next((s.probability for s in r.scenarios if s.name == "Bear"), 0.0)
    return bull - bear


def _skew_text(r: ResearchReport) -> str:
    bull = next((s.probability for s in r.scenarios if s.name == "Bull"), 0.0)
    bear = next((s.probability for s in r.scenarios if s.name == "Bear"), 0.0)
    return f"{bull * 100:.0f}% / {bear * 100:.0f}%"
