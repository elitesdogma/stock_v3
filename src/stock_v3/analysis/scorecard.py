"""The Institutional Scorecard and final verdict.

Combines the seven category scores into a weighted 0-100 conviction and maps it to a
verdict band. Two honesty mechanisms are central here:

  1. A category whose inputs were unavailable is NOT silently zeroed. The caller passes a
     CategoryScore with neutralized=True (score 5). Its weight is redistributed across the
     categories that DO have data, so missing data neither helps nor unfairly hurts.
  2. Confidence blends signal strength with data completeness — more neutralized categories
     lowers confidence, implementing the framework's "Insufficient Edge" honestly.
"""

from __future__ import annotations

from ..models import CategoryScore, Verdict

# Category weights (sum to 1.0). Fundamentals/valuation/technicals carry the most weight;
# catalysts the least, matching how an institutional desk weights a medium-term thesis.
_WEIGHTS: dict[str, float] = {
    "Macro Environment": 0.12,
    "Fundamentals": 0.22,
    "Valuation": 0.16,
    "Technicals": 0.18,
    "Institutional Positioning": 0.14,
    "Catalysts": 0.08,
    "Risk/Reward": 0.10,
}


def compute_verdict(scores: list[CategoryScore]) -> Verdict:
    by_name = {s.name: s for s in scores}
    # Redistribute weight away from neutralized categories onto categories with real data.
    live = {n: w for n, w in _WEIGHTS.items() if not _is_neutral(by_name.get(n))}
    live_weight_total = sum(live.values())

    if live_weight_total == 0:
        conviction = 50.0  # nothing to go on
    else:
        weighted = 0.0
        for name, base_weight in _WEIGHTS.items():
            score = by_name.get(name)
            if score is None or _is_neutral(score):
                continue
            renorm_weight = base_weight / live_weight_total
            weighted += renorm_weight * score.score
        conviction = weighted * 10.0  # 1-10 scale → 0-100

    # Round once, then band on the rounded value so the displayed score and the verdict
    # label can never disagree because of floating-point drift at a boundary (e.g. a true
    # 80.0 that computes as 79.999… and would otherwise read "Accumulate" beside an "80").
    conviction = round(conviction, 1)
    label = _verdict_label(conviction)
    confidence, band = _confidence(scores, conviction)
    return Verdict(
        conviction_score=conviction,
        label=label,
        confidence=round(confidence, 1),
        confidence_band=band,
    )


def _is_neutral(score: CategoryScore | None) -> bool:
    return score is None or score.neutralized


def _verdict_label(conviction: float) -> str:
    if conviction >= 90:
        return "Strong Buy"
    if conviction >= 80:
        return "Buy"
    if conviction >= 70:
        return "Accumulate"
    if conviction >= 60:
        return "Watchlist"
    if conviction >= 50:
        return "Hold"
    return "Avoid"


def _confidence(scores: list[CategoryScore], conviction: float) -> tuple[float, str]:
    """Confidence rewards both decisiveness (conviction far from the neutral 50) and data
    completeness (few neutralized categories)."""
    total = len(_WEIGHTS)
    neutralized = sum(1 for s in scores if s.neutralized)
    completeness = (total - neutralized) / total  # 0..1

    decisiveness = abs(conviction - 50.0) / 50.0  # 0 (right at Hold) .. 1 (extreme)
    confidence = 100.0 * (0.55 * completeness + 0.45 * decisiveness)
    confidence = max(0.0, min(100.0, confidence))

    if confidence >= 90:
        band = "Exceptional Conviction"
    elif confidence >= 80:
        band = "High Conviction"
    elif confidence >= 70:
        band = "Moderate Conviction"
    elif confidence >= 60:
        band = "Speculative"
    else:
        band = "Insufficient Edge"
    return confidence, band


def neutralized_score(name: str, reason: str) -> CategoryScore:
    """Factory for a category we couldn't assess — explicit, never a silent zero."""
    return CategoryScore(
        name=name,
        score=5.0,
        rationale=f"Neutralized — {reason}",
        neutralized=True,
    )
