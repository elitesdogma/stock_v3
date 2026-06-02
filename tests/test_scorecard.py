"""Verdict-band boundaries, weight redistribution, and neutralization honesty."""

from __future__ import annotations

from stock_v3.analysis.scorecard import compute_verdict, neutralized_score
from stock_v3.models import CategoryScore

_ALL_CATEGORIES = [
    "Macro Environment",
    "Fundamentals",
    "Valuation",
    "Technicals",
    "Institutional Positioning",
    "Catalysts",
    "Risk/Reward",
]


def _scores(value: float) -> list[CategoryScore]:
    return [CategoryScore(name=n, score=value, rationale="test") for n in _ALL_CATEGORIES]


def test_all_tens_is_strong_buy():
    v = compute_verdict(_scores(10.0))
    assert v.conviction_score == 100.0
    assert v.label == "Strong Buy"


def test_all_ones_is_avoid():
    v = compute_verdict(_scores(1.0))
    assert v.conviction_score == 10.0
    assert v.label == "Avoid"


def test_all_fives_is_hold():
    v = compute_verdict(_scores(5.0))
    assert v.conviction_score == 50.0
    assert v.label == "Hold"


def test_verdict_band_boundaries():
    # 7.0 across the board → 70.0 → Accumulate (lower edge inclusive)
    assert compute_verdict(_scores(7.0)).label == "Accumulate"
    # 8.0 → 80.0 → Buy
    assert compute_verdict(_scores(8.0)).label == "Buy"
    # 6.0 → 60.0 → Watchlist
    assert compute_verdict(_scores(6.0)).label == "Watchlist"


def test_neutralized_category_does_not_drag_score():
    """A neutralized (unavailable) category must not pull a strong thesis toward 50.
    With six 9s and one neutralized, conviction should reflect the 9s, not be diluted."""
    scores = _scores(9.0)
    scores[0] = neutralized_score("Macro Environment", "no FRED key")
    v = compute_verdict(scores)
    # All live categories are 9 → renormalized weighted avg is exactly 90.
    assert v.conviction_score == 90.0
    assert v.label == "Strong Buy"


def test_confidence_drops_with_missing_data():
    full = compute_verdict(_scores(9.0))
    degraded_scores = _scores(9.0)
    for i in range(4):
        degraded_scores[i] = neutralized_score(_ALL_CATEGORIES[i], "unavailable")
    degraded = compute_verdict(degraded_scores)
    assert degraded.confidence < full.confidence


def test_confidence_band_insufficient_edge_near_hold():
    v = compute_verdict(_scores(5.2))  # barely off neutral
    assert v.confidence_band in {"Insufficient Edge", "Speculative"}
